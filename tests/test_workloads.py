import unittest
import torch
from mps_fp64 import enable
from torch.nn import functional as F

D = torch.float64


class Workloads(unittest.TestCase):
    def test_convolution_gradients(self):
        torch.manual_seed(903)
        for rank in (1, 2, 3):
            for transpose in (False, True):
                op = getattr(
                    F, "conv" + ("_transpose" if transpose else "") + str(rank) + "d"
                )
                x = torch.randn(2, 4, *([5] * rank), dtype=D)
                w = torch.randn(4, 2, *([3] * rank), dtype=D)
                bias = torch.randn(4, dtype=D)

                def run(device):
                    a, b, c = [
                        v.to(device).detach().requires_grad_() for v in (x, w, bias)
                    ]
                    y = op(a, b, c, stride=2, padding=1, groups=2)
                    y.square().mean().backward()
                    return (y.cpu(), a.grad.cpu(), b.grad.cpu(), c.grad.cpu())

                ref = run("cpu")
                with enable():
                    actual = run("mps")
                for a, b in zip(actual, ref):
                    torch.testing.assert_close(a, b, rtol=1e-13, atol=2e-14)

    def test_fft(self):
        torch.manual_seed(1)
        for shape in ((13,), (8, 16), (3, 5, 4)):
            a = torch.randn(shape, dtype=D)
            with enable():
                x = a.to("mps")
                for norm in ("backward", "forward", "ortho"):
                    z = torch.fft.rfftn(x, norm=norm)
                    torch.testing.assert_close(
                        z.cpu(), torch.fft.rfftn(a, norm=norm), rtol=2e-13, atol=2e-13
                    )
                    actual = torch.fft.irfftn(z, s=shape, norm=norm)
                    torch.testing.assert_close(actual.cpu(), a, rtol=2e-13, atol=2e-13)

    def test_stable_sort(self):
        a = torch.tensor(
            [
                [float("nan"), 0.0, -0.0, 1.0, 1.0, float("nan")],
                [1.0, 3.0, 2.0, 1.0, 2.0, 0.0],
            ],
            dtype=D,
        )
        with enable():
            for desc in (False, True):
                vals, idx = a.to("mps").sort(dim=1, descending=desc, stable=True)
                ref, ref_idx = a.sort(dim=1, descending=desc, stable=True)
                self.assertTrue(
                    torch.equal(vals.cpu().view(torch.int64), ref.view(torch.int64))
                )
                self.assertTrue(torch.equal(idx.cpu(), ref_idx))

    def test_rng(self):
        with enable():
            for op, mean, std in [(torch.rand, 0.5, 12**-0.5), (torch.randn, 0, 1)]:
                state = torch.mps.get_rng_state()
                try:
                    torch.mps.manual_seed(120)
                    a = op(131072, device="mps", dtype=D).cpu()
                    torch.mps.manual_seed(120)
                    b = op(131072, device="mps", dtype=D).cpu()
                finally:
                    torch.mps.set_rng_state(state)
                self.assertTrue(torch.equal(a, b))
                self.assertLess(abs(a.mean().item() - mean), 0.015)
                self.assertLess(abs(a.std().item() - std), 0.015)
                self.assertEqual(torch.unique(a.view(torch.int64) & 255).numel(), 256)

    def test_cholesky(self):
        torch.manual_seed(3)
        t = torch.randn(2, 13, 13, dtype=D)
        a = t.mT @ t + torch.eye(13, dtype=D)
        with enable():
            x = a.to("mps")
            for upper in (False, True):
                factor = torch.linalg.cholesky(x, upper=upper)
                reconstructed = factor.mT @ factor if upper else factor @ factor.mT
                torch.testing.assert_close(
                    reconstructed.cpu(), a, rtol=2e-14, atol=2e-14
                )
            _, info = torch.linalg.cholesky_ex((-a).to("mps"))
            self.assertTrue(torch.all(info.cpu() == 1))
