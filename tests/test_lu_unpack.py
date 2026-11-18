import itertools
import unittest

import torch

from mps_fp64 import enable


F64 = torch.float64


class LinearProducts(unittest.TestCase):
    def test_rectangular_batched_lu_and_unpack(self):
        generator = torch.Generator().manual_seed(12954)
        shapes = [(5, 3), (2, 4, 6), (2, 3, 4, 4), (2, 3, 0), (2, 0, 3), (0, 4, 3)]
        for shape in shapes:
            with self.subTest(shape=shape):
                # Transpose noncontiguous data in a higher-rank batch.
                source = torch.randn(*shape[:-2], shape[-1], shape[-2], dtype=F64, generator=generator)
                a = source.mT
                lu, piv = torch.linalg.lu_factor(a)
                # Preserve both an offset and a non-unit stride on the device.
                pivot_buffer = torch.stack((piv, piv), -1)
                with enable() as mode:
                    x = source.to("mps").mT
                    p, lower, upper = torch.linalg.lu(x)
                    self.assertEqual((p.shape, lower.shape, upper.shape),
                                     tuple(result.shape for result in torch.linalg.lu(a)))
                    torch.testing.assert_close((p @ lower @ upper).cpu(), a, rtol=1e-13, atol=1e-14)
                    pivots = pivot_buffer.to("mps")[..., 1]
                    packed = lu.mT.contiguous().to("mps").mT
                    for data, perm in itertools.product((False, True), repeat=2):
                        actual = torch.lu_unpack(packed, pivots, data, perm)
                        ref = torch.lu_unpack(lu, piv.contiguous(), data, perm)
                        for got, want in zip(actual, ref):
                            self.assertTrue(torch.equal(got.cpu().view(torch.int64), want.view(torch.int64)))
                    self.assertFalse(mode.cpu_ops)

    def test_pivot_range_and_singular(self):
        a = torch.tensor([[0., 4., 1.], [1., 2., 3.], [2., 4., 6.]], dtype=F64)
        with enable() as mode:
            x = a.to("mps")
            p, lower, upper = torch.linalg.lu(x)
            torch.testing.assert_close((p @ lower @ upper).cpu(), a, rtol=0, atol=0)
            for piv in ([1, 0, 3], [1, 4, 3], [1, 2]):
                with self.assertRaises(RuntimeError):
                    torch.lu_unpack(x, torch.tensor(piv, dtype=torch.int32, device="mps"))
            with self.assertRaises(RuntimeError):
                torch.lu_unpack(x, torch.tensor([1, 2, 3], device="mps"))
            # An ignored pivots argument is never read by the native API.
            out = torch.lu_unpack(x, torch.empty(0, dtype=torch.int64, device="mps"), False, False)
            self.assertEqual([t.numel() for t in out], [0, 0, 0])
            # Legal backward swaps are meaningful to the public unpack routine.
            seq = torch.tensor([3, 1, 2], dtype=torch.int32)
            actual = torch.lu_unpack(x, seq.to("mps")).P.cpu()
            self.assertTrue(torch.equal(actual, torch.lu_unpack(a, seq).P))
            # An all-zero active column can have a finite unpivoted LU.
            diagonal = torch.diag(torch.tensor([0., 2., -3.], dtype=F64))
            p, lower, upper = torch.linalg.lu(diagonal.to("mps"), pivot=False)
            self.assertEqual(p.numel(), 0)
            self.assertTrue(torch.equal((lower @ upper).cpu(), diagonal))
            # An unpivoted zero diagonal above a nonzero entry cannot represent
            # this matrix with a finite multiplier.
            swap = torch.tensor([[0., 1.], [1., 0.]], dtype=F64, device="mps")
            _, lower, _ = torch.linalg.lu(swap, pivot=False)
            self.assertTrue(torch.isinf(lower.cpu()[1, 0]))
            self.assertFalse(mode.cpu_ops)

    def test_householder_reflectors(self):
        generator = torch.Generator().manual_seed(81388)
        for shape, k in [((9, 4), 4), ((2, 3, 10, 6), 3), ((3, 7, 4), 0),
                         ((0, 7, 4), 2), ((5, 0), 0), ((257, 4), 4)]:
            with self.subTest(shape=shape, k=k):
                a = torch.randn(*shape, dtype=F64, generator=generator)
                h, tau = torch.geqrf(a)
                base = h.mT.contiguous()
                tau_base = torch.stack((tau[..., :k], tau[..., :k]), -1)
                expected = torch.linalg.householder_product(h, tau[..., :k])
                with enable() as mode:
                    raw = base.to("mps").mT
                    t = tau_base.to("mps")[..., 1]
                    result = torch.linalg.householder_product(raw, t)
                    torch.testing.assert_close(result.cpu(), expected, rtol=2e-14, atol=3e-15)
                    self.assertEqual(result.stride(), expected.stride())
                    torch.testing.assert_close(torch.orgqr(raw, t).cpu(), expected,
                                               rtol=2e-14, atol=3e-15)
                    self.assertFalse(mode.cpu_ops)

    def test_zero_tau_and_unused_entries(self):
        packed = torch.tensor([[float("nan"), float("nan")], [1., 0.], [float("inf"), -1.]], dtype=F64)
        tau = torch.zeros(2, dtype=F64)
        expected = torch.linalg.householder_product(packed, tau)
        with enable() as mode:
            result = torch.linalg.householder_product(packed.to("mps"), tau.to("mps")).cpu()
            self.assertFalse(mode.cpu_ops)
        self.assertTrue(torch.equal(result.isnan(), expected.isnan()))
        finite = ~expected.isnan()
        self.assertTrue(torch.equal(result.view(torch.int64)[finite], expected.view(torch.int64)[finite]))

    def test_out_and_validation(self):
        a = torch.tensor([[4., -3.], [1., 2.], [-1., 0.]], dtype=F64)
        lu, piv = torch.linalg.lu_factor(a)
        vectors, tau = torch.geqrf(a)
        with enable() as mode:
            x = a.to("mps")
            expected = torch.linalg.lu(a)
            for fn in (lambda outs: torch.linalg.lu(x, out=outs),
                       lambda outs: torch.lu_unpack(lu.to("mps"), piv.to("mps"), out=outs)):
                outs = (torch.empty(0, device="mps", dtype=F64),
                        torch.empty(2, 3, device="mps", dtype=F64).T,
                        torch.empty(2, 4, device="mps", dtype=F64)[:, ::2])
                got = fn(outs)
                for result, dst, want in zip(got, outs, expected):
                    self.assertIs(result, dst)
                    self.assertTrue(torch.equal(dst.cpu(), want))
            v, t = vectors.to("mps"), tau.to("mps")
            reference = torch.linalg.householder_product(vectors, tau)
            for dtype in (F64, torch.float32, torch.complex64, torch.complex128):
                q = torch.empty(0, dtype=dtype, device="mps")
                self.assertIs(torch.linalg.householder_product(v, t, out=q), q)
                self.assertTrue(torch.equal(q.cpu(), reference.to(dtype)))
                if dtype == F64:
                    self.assertEqual(q.stride(), reference.stride())
            for matrix, coeff in [(v.T, t), (v, t[None]), (v[None], t.expand(2, 2)), (v, t.float())]:
                with self.assertRaises(RuntimeError):
                    torch.linalg.householder_product(matrix, coeff)
            self.assertFalse(mode.cpu_ops)

    def test_first_and_second_derivatives(self):
        a = torch.randn(4, 3, generator=torch.Generator().manual_seed(9181), dtype=F64)
        h, tau = torch.geqrf(a)
        pivots = torch.linalg.lu_factor(a).pivots

        def gradients(name, device):
            def variable(tensor):
                return tensor.to(device).detach().requires_grad_()

            if name == "householder":
                args = (variable(h), variable(tau))
                q = torch.linalg.householder_product(*args)
                loss = (q.square() + 0.2 * q).sum()
            else:
                args = (variable(a),)
                if name == "lu":
                    _, lower, upper = torch.linalg.lu(*args)
                else:
                    _, lower, upper = torch.lu_unpack(args[0], pivots.to(device))
                loss = lower.square().sum() + upper.square().sum()
            first = torch.autograd.grad(loss, args, create_graph=True)
            second = torch.autograd.grad(sum(x.square().sum() for x in first), args)
            return [x.detach().cpu() for x in (*first, *second)]

        for name in ("lu", "unpack", "householder"):
            with self.subTest(name=name):
                reference = gradients(name, "cpu")
                with enable() as mode:
                    actual = gradients(name, "mps")
                    self.assertFalse(mode.cpu_ops)
                for value, expected in zip(actual, reference):
                    torch.testing.assert_close(value, expected, rtol=3e-13, atol=1e-13)
