"""Public-API regressions for the new fused, prefix and arbitrary-length FFT paths."""

from concurrent.futures import ThreadPoolExecutor
import unittest

import torch
import torch.nn.functional as F
from mps_fp64 import enable

D = torch.float64
I = torch.int64


class Acceleration(unittest.TestCase):
    def test_fused_aminmax(self):
        rng = torch.Generator().manual_seed(7591)
        x = torch.randn(5, 1029, dtype=D, generator=rng)[:, 1::2]
        signed_zero = torch.tensor([0, -(1 << 63)], dtype=I).view(D)
        payloads = torch.tensor([0x7FF8000000000055, 0x7FF80000000000AA], dtype=I)
        with enable() as mode:
            for host in (x, x.T, signed_zero, signed_zero.flip((0,)), payloads.view(D),
                         torch.randn((), dtype=D), torch.empty((0, 7), dtype=D)):
                gpu = host.to("mps")
                for dim in (None, -1, 0):
                    for keep in (True, False):
                        if host.numel() == 0 and (dim is None or host.shape[dim] == 0):
                            with self.assertRaises((IndexError, RuntimeError)):
                                torch.aminmax(gpu, dim=dim, keepdim=keep)
                            continue
                        expected = torch.aminmax(host, dim=dim, keepdim=keep)
                        actual = torch.aminmax(gpu, dim=dim, keepdim=keep)
                        for got, ref in zip(actual, expected):
                            self.assertEqual(got.dtype, D)
                            self.assertTrue(got.view(I).cpu().equal(ref.view(I)))
            gpu = x.to("mps")
            out = (torch.empty(0, device="mps", dtype=D),
                   torch.empty((5, 2), device="mps", dtype=D)[:, 1])
            actual = torch.aminmax(gpu, dim=-1, out=out)
            for got, dest, expected in zip(actual, out, torch.aminmax(x, dim=-1)):
                self.assertIs(got, dest)
                self.assertTrue(got.cpu().equal(expected))
            self.assertFalse(mode.cpu_ops)

    def test_prime_fft_axes_norm_grad_and_strides(self):
        torch.manual_seed(719)
        host = torch.randn(97, 14, dtype=torch.complex128).T[1::2]
        for norm in ("backward", "forward", "ortho"):
            ref = torch.fft.fftn(host, dim=(0, 1), norm=norm)
            with enable() as mode:
                gpu = host.to("mps")
                got = torch.fft.fftn(gpu, dim=(0, 1), norm=norm)
                torch.testing.assert_close(got.cpu(), ref, rtol=4e-13, atol=4e-13)
                torch.testing.assert_close(
                    torch.fft.ifftn(got, norm=norm).cpu(), host, rtol=5e-13, atol=5e-13
                )
                self.assertFalse(mode.cpu_ops)
        x = torch.randn(257, dtype=D, requires_grad=True)
        ref = torch.autograd.grad(torch.fft.rfft(x).abs().sum(), x)[0]
        with enable():
            a = x.detach().to("mps").requires_grad_()
            grad = torch.autograd.grad(torch.fft.rfft(a).abs().sum(), a)[0]
            torch.testing.assert_close(grad.cpu(), ref, rtol=4e-13, atol=4e-13)

    def test_fft_extreme_rows_and_cached_threads(self):
        x = torch.zeros((3, 71), dtype=D)
        x[:, 0] = torch.tensor(
            [1.25, 1e307, float.fromhex("0x0.0000000000001p-1022")], dtype=D
        )
        with enable():
            z = torch.fft.rfft(x.to("mps")).cpu()
        for row in (1, 2):
            self.assertTrue(torch.equal(z[row].real, x[row, 0].expand_as(z[row].real)))
            self.assertTrue(torch.all(z[row].imag == 0).item())
        torch.testing.assert_close(z[0], torch.fft.rfft(x[0]), rtol=2e-14, atol=2e-14)
        # Exercise a previously unseen length concurrently, including plans first built in inference mode.
        source = torch.randn(199, dtype=D)
        ref = torch.fft.rfft(source)

        def apply(inference):
            with torch.inference_mode(inference), enable():
                return torch.fft.rfft(source.to("mps")).cpu()

        with ThreadPoolExecutor(2) as threads:
            for result in threads.map(apply, (True, False, False, True)):
                torch.testing.assert_close(result, ref, rtol=2e-13, atol=2e-13)

    def test_prefix_ties_nan_payloads_across_blocks(self):
        gen = torch.Generator().manual_seed(791)
        raw = torch.randint(-2, 3, (4, 18001), generator=gen).double().view(I)
        raw[:, 0::173] = -(1 << 63)
        raw[:, 127] = 0x7FF8000000000013
        raw[:, 16383] = 0x7FF8000000000065
        data = raw.T.view(D)
        with enable():
            gpu = data.to("mps")
            for op in (torch.cummin, torch.cummax):
                got = op(gpu, 0)
                ref = op(data, 0)
                self.assertTrue(
                    torch.equal(got.values.cpu().view(I), ref.values.view(I))
                )
                self.assertTrue(torch.equal(got.indices.cpu(), ref.indices))

    def test_scatter_collisions_grad_and_payloads(self):
        idx = torch.tensor([[0, 1, 1, 1, 0, 1]])
        src = torch.tensor([[1.0, 2.0**54, 1.0, -(2.0**54), -0.0, 2**-52]], dtype=D)
        base = torch.tensor([[-0.0, 1.0, 17.0]], dtype=D)
        with enable() as mode:
            x, i, v = (t.to("mps") for t in (base, idx, src))
            for red in ("sum", "prod", "amax", "amin", "mean"):
                for include in (True, False):
                    out = x.scatter_reduce(1, i, v, red, include_self=include)
                    ref = base.scatter_reduce(1, idx, src, red, include_self=include)
                    self.assertTrue(
                        torch.equal(out.cpu().view(I), ref.view(I)), (red, include)
                    )
            payload = torch.tensor([[0x7FF8000000000213, -(1 << 63)]], dtype=I)
            out = x.scatter(
                1, torch.tensor([[1, 0]], device="mps"), payload.view(D).to("mps")
            )
            self.assertEqual(
                out.cpu().view(I).tolist(),
                [[-(1 << 63), 0x7FF8000000000213, 4625478292286210048]],
            )
            with self.assertRaises(RuntimeError):
                x.scatter_add(1, i, v.float())
            g = src.to("mps").requires_grad_()
            result = x.scatter_add(1, i, g)
            grad = torch.autograd.grad(result.square().sum(), g)[0]
            ref = 2 * base.scatter_add(1, idx, src).gather(1, idx)
            torch.testing.assert_close(grad.cpu(), ref)
            self.assertFalse(mode.cpu_ops)

    def test_searchsorted_special_values_and_shape(self):
        bounds = torch.tensor([[2.0, -0.0, 1 + 2**-48, 1.0, float("nan")]], dtype=D)
        queries = torch.tensor(
            [
                [
                    float("-inf"),
                    0.0,
                    1.0,
                    1 + 2**-49,
                    1 + 2**-48,
                    2.0,
                    float("inf"),
                    float("nan"),
                ]
            ],
            dtype=D,
        )
        sorter = bounds.argsort()
        with enable():
            b, q, s = (t.to("mps") for t in (bounds, queries, sorter))
            for side in ("left", "right"):
                a = torch.searchsorted(b, q, side=side, sorter=s, out_int32=True).cpu()
                ref = torch.searchsorted(
                    bounds, queries, side=side, sorter=sorter, out_int32=True
                )
                torch.testing.assert_close(a, ref)

    def test_grouped_point_convolution_and_gradients(self):
        torch.manual_seed(808)
        x, w, b = (
            torch.randn(4, 32, 13, 7, dtype=D),
            torch.randn(32, 16, 1, 1, dtype=D),
            torch.randn(32, dtype=D),
        )
        for op in (F.conv2d, F.conv_transpose2d):

            def compute(device):
                a, weights, bias = [
                    v.to(device).detach().requires_grad_() for v in (x, w, b)
                ]
                y = op(a, weights, bias, groups=2)
                return [
                    t.cpu()
                    for t in (
                        y,
                        *torch.autograd.grad(y.square().mean(), (a, weights, bias)),
                    )
                ]

            ref = compute("cpu")
            with enable():
                for v, expected in zip(compute("mps"), ref):
                    torch.testing.assert_close(v, expected, rtol=3e-13, atol=3e-13)

    def test_unique_with_payloads_signed_zero_and_rows(self):
        raw = torch.tensor(
            [
                0,
                -(1 << 63),
                0x7FF8000000000001,
                0x7FF8000000000024,
                0x3FF0000000000001,
                0x3FF0000000000001,
            ],
            dtype=I,
        )
        x = raw.view(D)
        with enable():
            data = x.to("mps")
            u, inv, counts = torch.unique_consecutive(
                data, return_inverse=True, return_counts=True
            )
            self.assertTrue(torch.equal(u.cpu().view(I), raw[[0, 2, 3, 4]]))
            self.assertEqual(inv.cpu().tolist(), [0, 0, 1, 2, 3, 3])
            self.assertEqual(counts.cpu().tolist(), [2, 1, 1, 2])
            a = torch.tensor([[2.0, 1.0, 1.0, 3.0], [0.0, 2.0, 2.0, 1.0]], dtype=D)
            got = torch.unique(
                a.to("mps"), dim=1, return_counts=True, return_inverse=True
            )
            ref = torch.unique(a, dim=1, return_counts=True, return_inverse=True)
            for g, r in zip(got, ref):
                torch.testing.assert_close(g.cpu(), r, rtol=0, atol=0)

    def test_inference_fft_public_sizing(self):
        x = torch.randn(4, 9, dtype=D)
        tasks = [
            lambda v: torch.fft.rfft(v, n=15, dim=0, norm="ortho"),
            lambda v: torch.fft.ifftn(v, s=(6, 11), dim=(0, 1), norm="forward"),
            lambda v: torch.fft.ihfft(v, n=13, dim=1),
            lambda v: torch.fft.hfft(v, n=16, dim=1),
        ]
        for fn in tasks:
            expected = fn(x)
            with torch.inference_mode(), enable():
                got = fn(x.to("mps")).cpu()
                torch.testing.assert_close(got, expected, atol=3e-13, rtol=3e-13)

    def test_embedding_bag_gradients_and_group_norm(self):
        host = torch.randn(6, 5, dtype=D)
        indices = torch.tensor([2, 4, 2, 0, 1, 4, 4])
        offsets = torch.tensor([0, 4, 4])
        scales = torch.randn(indices.shape, dtype=D)
        for mode in ("sum", "mean", "max"):

            def compute(device):
                w = host.to(device).detach().requires_grad_()
                s = scales.to(device).detach().requires_grad_()
                y = F.embedding_bag(
                    indices.to(device),
                    w,
                    offsets.to(device),
                    mode=mode,
                    padding_idx=0,
                    per_sample_weights=s if mode == "sum" else None,
                )
                return [
                    a.cpu()
                    for a in (
                        y,
                        *torch.autograd.grad(
                            y.square().sum(), (w, s) if mode == "sum" else (w,)
                        ),
                    )
                ]

            ref = compute("cpu")
            with enable():
                for a, b in zip(compute("mps"), ref):
                    torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-13)
        data = torch.randn(2, 8, 4, 5, dtype=D)

        def run(device):
            x = data.to(device).detach().requires_grad_()
            y = F.group_norm(x, 4)
            return y.cpu(), torch.autograd.grad(y.square().sum(), x)[0].cpu()

        ref = run("cpu")
        with enable():
            for a, b in zip(run("mps"), ref):
                torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-13)

    def test_resize_storage_and_log_prefix(self):
        pattern = torch.tensor(
            [0x3FF0000000000001, -(1 << 63), 0x7FF8000000000009], dtype=I
        )
        with enable():
            x = pattern.to("mps").view(D)
            alias = x[:2]
            self.assertIs(x.resize_(3, 5), x)
            self.assertTrue(torch.equal(x.cpu().reshape(-1)[:3].view(I), pattern))
            self.assertTrue(torch.equal(alias.cpu().view(I), pattern[:2]))
            self.assertIs(x.resize_as_(alias), x)
            self.assertEqual(x.shape, (2,))
            a = torch.randn(3, 3000, dtype=D)
            a[0, :17] = float("-inf")
            a[1, 128] = float("inf")
            a[2, 257] = float("nan")
            b = torch.logcumsumexp(a.to("mps"), 1).cpu()
            torch.testing.assert_close(
                b, a.logcumsumexp(1), rtol=1e-13, atol=1e-13, equal_nan=True
            )

    def test_mixed_precision_covariance(self):
        from mps_fp64.backend import CovarianceWeightSums

        torch.manual_seed(701)
        for n in (1, 3, 4, 5, 17, 63, 64, 73, 127, 128):
            for stride in (1, 3):
                x = torch.randn(n * stride, dtype=torch.float32)
                with CovarianceWeightSums():
                    actual = x.to("mps")[::stride].sum().cpu()
                self.assertEqual(actual.item(), x[::stride].sum().item())
        data = torch.randn(3, 11, dtype=D)
        weights = torch.rand(11, dtype=torch.float32)

        def run(device):
            x = data.to(device).detach().requires_grad_()
            w = weights.to(device).detach().requires_grad_()
            cov = torch.cov(x, aweights=w)
            return [
                v.cpu() for v in (cov, *torch.autograd.grad(cov.square().sum(), (x, w)))
            ]

        ref = run("cpu")
        with enable():
            y, dx, dw = run("mps")
            torch.testing.assert_close(y, ref[0], rtol=4e-14, atol=1e-14)
            torch.testing.assert_close(dx, ref[1], rtol=4e-13, atol=1e-13)
            torch.testing.assert_close(dw, ref[2], rtol=3e-6, atol=2e-7)

    def test_large_scatter_destinations_are_distinct(self):
        # Neighbouring logical positions can have the same binary32 representation.
        limit = 1 << 24
        idx = torch.tensor([limit+3,limit+1,limit+2,limit+1],device="mps")
        with enable():
            x = torch.zeros(limit+5,device="mps",dtype=D)
            values = torch.tensor([1.,2.,4.,8.],device="mps",dtype=D)
            result = x.scatter_add(0,idx,values)
            self.assertEqual(result[limit:].cpu().tolist(),[0.,10.,4.,1.,0.])
