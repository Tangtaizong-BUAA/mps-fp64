import math
import unittest

import torch

from mps_fp64 import enable
from mps_fp64.special import _BESSEL_IDS, _POLYNOMIALS

D = torch.float64


class SpecialFunctions(unittest.TestCase):
    def test_all_polynomials_and_switches(self):
        x = torch.cat(
            (
                torch.linspace(-1.5, 1.5, 101, dtype=D),
                torch.tensor(
                    [
                        -math.inf,
                        math.inf,
                        math.nan,
                        0.0,
                        -0.0,
                        math.nextafter(-1.0, 0.0),
                        math.nextafter(1.0, 0.0),
                        math.nextafter(0.0, 1.0),
                        0.5,
                    ],
                    dtype=D,
                ),
            )
        )[:, None]
        ns = torch.tensor(
            [
                -2.1,
                -1,
                -0.9,
                0,
                0.8,
                1.99,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                32,
                100,
                127,
                512,
                513,
            ],
            dtype=D,
        )[None, :]
        with enable() as mode:
            a, n = x.to("mps"), ns.to("mps")
            for name in _POLYNOMIALS:
                with self.subTest(name=name):
                    fun = getattr(torch.ops.aten, name).default
                    torch.testing.assert_close(
                        fun(a, n).cpu(),
                        fun(x, ns),
                        equal_nan=True,
                        rtol=4e-12,
                        atol=1e-12,
                    )
            self.assertFalse(mode.cpu_ops)

    def test_recurrences_single_rounding(self):
        g = torch.Generator().manual_seed(701)
        x = torch.rand(513, dtype=D, generator=g) * 2 - 1
        with enable():
            for name in _POLYNOMIALS:
                fun = getattr(torch.ops.aten, name).n_scalar
                # Degree four is below every trigonometric switch.
                got = fun(x.to("mps"), 4)
                self.assertTrue(
                    torch.equal(
                        got.cpu().view(torch.int64), fun(x, 4).view(torch.int64)
                    ),
                    name,
                )

    def test_degree_limits_and_nonfinite(self):
        ns = torch.tensor(
            [
                math.nan,
                math.inf,
                -math.inf,
                float(2**63),
                -float(2**63),
                -1.9,
                -0.9,
                0,
                0.9,
            ],
            dtype=D,
        )
        with enable() as mode:
            for name in _POLYNOMIALS:
                f = getattr(torch.ops.aten, name).x_scalar
                # The endpoints have closed forms in ATen, including signed
                # integer wrap before conversion for out-of-range positive n.
                endpoint = 0.0 if "laguerre" in name else 1.0
                torch.testing.assert_close(
                    f(endpoint, ns.to("mps")).cpu(), f(endpoint, ns), equal_nan=True
                )
            for n in (4097, torch.tensor([5000], dtype=D).to("mps")):
                with self.assertRaisesRegex(NotImplementedError, "4096"):
                    torch.special.chebyshev_polynomial_t(
                        torch.tensor([math.nextafter(1.0, 2.0)], dtype=D, device="mps"),
                        n,
                    )
                with self.assertRaisesRegex(NotImplementedError, "4096"):
                    torch.special.chebyshev_polynomial_t(
                        torch.tensor([0.2], dtype=D, device="mps"), n
                    )
            q = torch.tensor([0.0, -1.0, 1.0], dtype=D)
            torch.testing.assert_close(
                torch.special.chebyshev_polynomial_t(q.to("mps"), 2**40).cpu(),
                torch.special.chebyshev_polynomial_t(q, 2**40),
            )
            self.assertFalse(mode.cpu_ops)

    def test_bessel_domains_and_piecewise_boundaries(self):
        cuts = (0.0, 0.5, 2.0, 5.0, 8.0, 710.0)
        edges = [
            -math.inf,
            math.inf,
            math.nan,
            -0.0,
            1e-320,
            -1e-320,
            1e-300,
            1e65,
            1e308,
        ]
        for cut in cuts:
            for x in (
                cut,
                math.nextafter(cut, -math.inf),
                math.nextafter(cut, math.inf),
            ):
                edges.extend((x, -x))
        src = torch.cat(
            (torch.linspace(-12, 12, 997, dtype=D), torch.tensor(edges, dtype=D))
        )
        with enable() as mode:
            for name in _BESSEL_IDS:
                with self.subTest(name=name):
                    f = getattr(torch.ops.aten, name).default
                    expected, actual = f(src), f(src.to("mps")).cpu()
                    torch.testing.assert_close(
                        actual, expected, rtol=4e-12, atol=5e-15, equal_nan=True
                    )
                    zeros = expected == 0
                    self.assertTrue(
                        torch.equal(
                            expected[zeros].view(torch.int64),
                            actual[zeros].view(torch.int64),
                        )
                    )
            self.assertFalse(mode.cpu_ops)

    def test_scalar_tensor_broadcast_strides_and_out(self):
        cpu = torch.linspace(0.1, 0.8, 24, dtype=D).reshape(4, 6).t()
        n = torch.arange(12, dtype=torch.int32).reshape(2, 6).t()[:, :1]
        with enable() as mode:
            x, degree = cpu.to("mps"), n.to("mps")
            f = torch.special.chebyshev_polynomial_t
            for a, b, ca, cb in (
                (x, degree, cpu, n),
                (x, 4.9, cpu, 4.9),
                (0.3, x, 0.3, cpu),
            ):
                y, ref = f(a, b), f(ca, cb)
                self.assertEqual(y.stride(), ref.stride())
                torch.testing.assert_close(y.cpu(), ref)
            out = torch.empty(0, device="mps", dtype=D)
            self.assertIs(f(x, 3, out=out), out)
            torch.testing.assert_close(out.cpu(), f(cpu, 3))
            fp32 = torch.empty(cpu.shape, dtype=torch.float32, device="mps")
            self.assertIs(f(x, 3, out=fp32), fp32)
            torch.testing.assert_close(fp32.cpu(), f(cpu, 3).float())
            full = x.clone()
            self.assertIs(f(full, 3, out=full), full)
            torch.testing.assert_close(full.cpu(), f(cpu, 3))
            buf = torch.arange(9, device="mps", dtype=D)
            with self.assertRaises(RuntimeError):
                f(buf[:-1], 2, out=buf[1:])
            self.assertEqual(f(torch.empty(0, dtype=D, device="mps"), 5).numel(), 0)
            self.assertFalse(mode.cpu_ops)


class GammaFunctions(unittest.TestCase):
    def test_near_poles_and_inverse_erf_tails(self):
        import mpmath as mp

        points = [
            math.nextafter(float(k), side)
            for k in (-100.0, -3.0, -1.0)
            for side in (-math.inf, math.inf)
        ]
        points += [1.4616321449683622, math.nextafter(1.4616321449683622, 2.0)]
        xv = torch.tensor(points, dtype=D)
        yv = torch.tensor(
            [sign * (1.0 - 2.0**-k) for k in (10, 30, 40, 52, 53) for sign in (-1, 1)],
            dtype=D,
        )
        with mp.workdps(110), enable() as mode:
            for f, ref in (
                (torch.digamma, mp.digamma),
                (torch.lgamma, lambda v: mp.log(abs(mp.gamma(v)))),
                (lambda t: torch.polygamma(1, t), lambda v: mp.polygamma(1, v)),
            ):
                expected = torch.tensor(
                    [float(ref(mp.mpf(v))) for v in points], dtype=D
                )
                torch.testing.assert_close(
                    f(xv.to("mps")).cpu(), expected, rtol=4e-15, atol=3e-16
                )
            out = torch.erfinv(yv.to("mps")).cpu()
            ref = torch.tensor(
                [float(mp.erfinv(mp.mpf(v))) for v in yv.tolist()], dtype=D
            )
            torch.testing.assert_close(out, ref, rtol=3e-16, atol=0.0)
            singular = torch.tensor(
                [-1.0, -0.0, 0.0, 1.0, -math.inf, math.inf, math.nan],
                dtype=D,
                device="mps",
            )
            t = torch.polygamma(1, singular).cpu()
            self.assertTrue(torch.isposinf(t[:3]).all())
            self.assertEqual(t[5], 0.0)
            self.assertTrue(math.isnan(t[4].item()) and math.isnan(t[6].item()))
            self.assertFalse(mode.cpu_ops)

    def test_hurwitz_scaling_and_reflection(self):
        import mpmath as mp

        with mp.workdps(225), enable() as mode:
            for n in (2, 3, 4, 10, 50, 160, 170):
                pts = [
                    -2.5,
                    math.nextafter(-2.5, 0),
                    math.nextafter(-2.5, -3.0),
                    5.0,
                    1000.0,
                    3000.0,
                ]
                x = torch.tensor(pts, dtype=D, device="mps")
                r = torch.tensor(
                    [float(mp.polygamma(n, mp.mpf(v))) for v in pts], dtype=D
                )
                torch.testing.assert_close(
                    torch.polygamma(n, x).cpu(), r, rtol=7e-14, atol=5e-324
                )
            s = torch.tensor([[2.0], [3.0], [4.0]], dtype=D)
            x = torch.tensor([-1000.5, -0.125, 0.25, 12.0, 1e100], dtype=D)

            # Translate negative q to the positive half-line before calling
            # mpmath. Its direct negative-q asymptotic can lose the pole sum.
            def reference(n, v):
                q = mp.mpf(v)
                shift = max(0, math.ceil(1 - v))
                return float(
                    mp.fsum((q + k) ** (-int(n)) for k in range(shift))
                    + mp.zeta(n, q + shift)
                )

            ref = torch.tensor(
                [[reference(n, v) for v in x.tolist()] for n in s.flatten().tolist()],
                dtype=D,
            )
            torch.testing.assert_close(
                torch.special.zeta(s.to("mps"), x.to("mps")).cpu(),
                ref,
                rtol=5e-15,
                atol=5e-324,
            )
            invalid = torch.tensor([-2.0, -0.5, 1.0], dtype=D, device="mps")
            self.assertTrue(torch.isnan(torch.special.zeta(0.5, invalid)).all().item())
            self.assertFalse(mode.cpu_ops)

    def test_gamma_layout_grad_and_parameter_domain(self):
        src = torch.linspace(0.85, 3.1, 24, dtype=D).reshape(6, 4).t()
        with enable() as mode:
            x = src.to("mps").detach().requires_grad_()
            torch.lgamma(x).sum().backward()
            torch.testing.assert_close(
                x.grad.cpu(), src.digamma(), rtol=3e-14, atol=2e-15
            )
            out = torch.empty_strided(src.shape, src.stride(), device="mps", dtype=D)
            result = torch.special.erfcx(x.detach(), out=out)
            self.assertIs(result, out)
            self.assertEqual(result.stride(), src.stride())
            torch.testing.assert_close(
                result.cpu(), torch.special.erfcx(src), rtol=2e-14, atol=2e-15
            )
            for n in (-1, 171):
                with self.assertRaises((NotImplementedError, RuntimeError)):
                    torch.polygamma(n, x)
            with self.assertRaises(RuntimeError):
                torch.mvlgamma(x, 0)
            # Large p uses the ordinary GPU decomposition beyond the fused
            # kernel's 512-term bound.
            for p in (1, 3, 5, 513):
                torch.testing.assert_close(
                    torch.mvlgamma(x, p).cpu(),
                    torch.mvlgamma(src, p),
                    rtol=2e-14,
                    atol=2e-15,
                )
            self.assertFalse(mode.cpu_ops)


class IncompleteGamma(unittest.TestCase):
    def test_boundaries_broadcast_out_and_aliases(self):
        a = torch.tensor(
            [0.0, -0.0, 2.0, -1.0, math.inf, -math.inf, math.nan], dtype=D
        )[:, None]
        x = torch.tensor(
            [0.0, -0.0, 0.25, -1.0, math.inf, -math.inf, math.nan], dtype=D
        )
        with enable() as mode:
            aa, xx = a.to("mps"), x.to("mps")
            for fn, alias in [
                (torch.igamma, torch.special.gammainc),
                (torch.igammac, torch.special.gammaincc),
            ]:
                ref = fn(a, x)
                for call in (fn, alias):
                    out = torch.empty((len(x), len(a)), dtype=D, device="mps").t()
                    self.assertIs(call(aa, xx, out=out), out)
                    torch.testing.assert_close(
                        out.cpu(), ref, rtol=5e-15, atol=0.0, equal_nan=True
                    )
                empty = fn(aa[:0], xx)
                self.assertEqual(tuple(empty.shape), (0, len(x)))
            self.assertFalse(mode.cpu_ops)

    def test_regularized_tails_and_argument_gradient(self):
        # Rounded independently from arbitrary-precision density integrals.
        triples = [
            (64.0, 64.0, 0.5166239875038265, 0.4833760124961735),
            (1e-300, 0.99, 1.0, 2.2309982579017723e-301),
            (
                68762778.13259603,
                68722229.95106885,
                5.022328840532497e-7,
                0.9999994977671159,
            ),
            (2.7250777396966336e33, 2.725077739696633e33, 1.1872055788354717e-28, 1.0),
            (1e308, 1e308, 0.5, 0.5),
        ]
        a, x, p, q = map(lambda v: torch.tensor(v, dtype=D), zip(*triples))
        with enable() as mode:
            ag, xg = a.to("mps"), x.to("mps")
            for op, ref in ((torch.igamma, p), (torch.igammac, q)):
                torch.testing.assert_close(op(ag, xg).cpu(), ref, rtol=5e-15, atol=0)
                shape = torch.tensor([0.9, 2.5, 7.0], dtype=D)
                limit = torch.tensor([0.2, 3.0, 17.0], dtype=D, requires_grad=True)
                dev = limit.detach().to("mps").requires_grad_()
                (dg,) = torch.autograd.grad(op(shape.to("mps"), dev).sum(), (dev,))
                (dc,) = torch.autograd.grad(op(shape, limit).sum(), (limit,))
                torch.testing.assert_close(dg.cpu(), dc, rtol=3e-15, atol=0)
            self.assertFalse(mode.cpu_ops)
