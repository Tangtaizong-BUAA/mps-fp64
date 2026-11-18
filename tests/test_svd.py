"""Independent residuals, orthogonality, Moore-Penrose identities and gradients."""

import math
import unittest

import torch

from mps_fp64 import enable

F64 = torch.float64


def residuals(a, u, s, vh):
    k = s.shape[-1]
    reconstructed = (u[..., :k] * s[..., None, :]) @ vh[..., :k, :]
    scale = max(float(a.abs().max()), torch.finfo(F64).tiny)
    reference = torch.linalg.svdvals(a)
    return {
        "reconstruction": float(((reconstructed - a) / scale).abs().max()),
        "singular_values": float(((s - reference) / scale).abs().max()),
        "left_orthogonality": float(
            (u.mT @ u - torch.eye(u.shape[-1], dtype=F64)).abs().max()
        ),
        "right_orthogonality": float(
            (vh @ vh.mT - torch.eye(vh.shape[-2], dtype=F64)).abs().max()
        ),
    }


def sign_invariant_gradients(x, left, right):
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    loss = (
        (u[:, 0].square() * left).sum()
        + (vh[0].square() * right).sum()
        + s.square().sum()
    )
    ds = torch.autograd.grad(loss, x, create_graph=True)[0]
    ds2 = torch.autograd.grad(ds.square().sum(), x)[0]
    dp = torch.autograd.grad(torch.linalg.pinv(x).square().sum(), x, create_graph=True)[
        0
    ]
    dp2 = torch.autograd.grad(dp.square().sum(), x)[0]
    return ds, ds2, dp, dp2


class JacobiSVD(unittest.TestCase):
    def test_factorizations(self):
        rng = torch.Generator().manual_seed(43091)
        for shape in [
            (2, 5, 3),
            (2, 3, 8),
            (4, 4),
            (17, 9),
            (1, 37),
            (129, 1),
            (33, 33),
        ]:
            a = torch.randn(shape, dtype=F64, generator=rng)
            for kind in ("random", "rank_deficient", "zero"):
                x = a.clone()
                if kind == "rank_deficient":
                    x[..., -1] = x[..., 0]
                if kind == "zero":
                    x.zero_()
                for full in (True, False):
                    with (
                        self.subTest(shape=shape, kind=kind, full=full),
                        enable() as md,
                    ):
                        # Transposition also exercises strided matrix inputs.
                        g = x.transpose(-2, -1).contiguous().to("mps").transpose(-2, -1)
                        u, s, vh = [
                            v.cpu() for v in torch.linalg.svd(g, full_matrices=full)
                        ]
                        for value in residuals(x, u, s, vh).values():
                            self.assertLess(value, 4e-13)
                        self.assertTrue(torch.all(s[..., 1:] <= s[..., :-1]))
                        self.assertFalse(md.cpu_ops)

    def test_repeated_extreme_and_empty(self):
        tiny = math.ldexp(1.0, -1074)
        for diag in (
            [3.0, 3.0, 0.0],
            [math.ldexp(1.0, 900), 1.0, math.ldexp(1.0, -900), tiny],
            [-7.0, -7.0, -0.1],
        ):
            a = torch.diag(torch.tensor(diag, dtype=F64))
            expect = torch.tensor(sorted(map(abs, diag), reverse=True), dtype=F64)
            with enable() as md:
                u, s, vh = [v.cpu() for v in torch.linalg.svd(a.to("mps"))]
                torch.testing.assert_close(s, expect, atol=0, rtol=0)
                torch.testing.assert_close((u * s) @ vh, a, atol=0, rtol=0)
                self.assertFalse(md.cpu_ops)
        for shape in [(0, 0), (0, 3), (4, 0), (0, 3, 4)]:
            a = torch.empty(shape, dtype=F64)
            for full in (False, True):
                with enable() as md:
                    outputs = torch.linalg.svd(a.to("mps"), full_matrices=full)
                    for out, ref in zip(
                        outputs, torch.linalg.svd(a, full_matrices=full)
                    ):
                        self.assertEqual(out.shape, ref.shape)
                        torch.testing.assert_close(out.cpu(), ref)
                    self.assertFalse(md.cpu_ops)
        with enable():
            for val in (float("nan"), float("inf")):
                matrix = torch.tensor([[val, 1.0]], device="mps", dtype=F64)
                with self.assertRaises(torch.linalg.LinAlgError):
                    torch.linalg.svd(matrix)

    def test_moore_penrose_identities(self):
        rng = torch.Generator().manual_seed(882)
        for m, n, r in [(9, 3, 3), (3, 7, 3), (7, 8, 3), (2, 4, 0)]:
            a = torch.randn((2, m, r), dtype=F64, generator=rng) @ torch.randn(
                (2, r, n), dtype=F64, generator=rng
            )
            ref = torch.linalg.pinv(a)
            with enable() as md:
                g = a.to("mps")
                z = torch.linalg.pinv(g).cpu()
                self.assertFalse(md.cpu_ops)
            for result, expected in [
                (z, ref),
                (a @ z @ a, a),
                (z @ a @ z, z),
                ((a @ z).mT, a @ z),
                ((z @ a).mT, z @ a),
            ]:
                torch.testing.assert_close(result, expected, atol=3e-13, rtol=2e-12)
        # Hermitian mode must mirror the lower triangle and ignore the upper half.
        a = torch.tensor(
            [[2.0, 999.0, -183.0], [1.0, -3.0, 741.0], [0.25, 1.0, -1.0]], dtype=F64
        )
        with enable() as md:
            result = torch.linalg.pinv(a.to("mps"), hermitian=True).cpu()
            torch.testing.assert_close(
                result, torch.linalg.pinv(a, hermitian=True), atol=1e-13, rtol=1e-13
            )
            s = torch.diag_embed(torch.tensor([[1.0, 1e-7], [2.0, 4e-5]], dtype=F64))
            tol = torch.tensor([1.0e-9, 1.0e-3], dtype=F64)
            g = s.to("mps")
            for opts in [
                {"rtol": tol},
                {"atol": tol},
                {"rtol": tol[:, None], "atol": 1e-9},
                {"rtol": 0.01},
            ]:
                if torch.is_tensor(opts.get("rtol")) and isinstance(
                    opts.get("atol"), float
                ):
                    opts["atol"] = torch.tensor(opts["atol"], dtype=F64)
                torch.testing.assert_close(
                    torch.linalg.pinv(g, **opts).cpu(),
                    torch.linalg.pinv(s, **opts),
                    rtol=1e-13,
                    atol=1e-13,
                )
            self.assertFalse(md.cpu_ops)

    def test_sign_invariant_gradients(self):
        rng = torch.Generator().manual_seed(82)
        data = torch.randn((5, 3), dtype=F64, generator=rng)
        weights = torch.randn(5, dtype=F64, generator=rng)
        target = torch.randn(3, dtype=F64, generator=rng)

        x = data.clone().requires_grad_()
        reference = sign_invariant_gradients(x, weights, target)
        with enable() as md:
            g = data.to("mps").requires_grad_()
            actual = sign_invariant_gradients(g, weights.to("mps"), target.to("mps"))
            for value, expected in zip(actual, reference):
                torch.testing.assert_close(
                    value.cpu(), expected, atol=2e-12, rtol=2e-11
                )
            self.assertFalse(md.cpu_ops)
