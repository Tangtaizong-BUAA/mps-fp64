import torch
import unittest
from mps_fp64 import enable


class Integration(unittest.TestCase):
    def test_hessian(self):
        torch.manual_seed(23)
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        with enable() as mode:

            def expression(a, b):
                return (a.to("mps") @ b.to("mps")).sin().tanh().mean().cpu()

            for test in (torch.autograd.gradcheck, torch.autograd.gradgradcheck):
                self.assertTrue(test(expression, (x, w), atol=2e-7, rtol=2e-5))
            self.assertFalse(mode.cpu_ops)

    def test_scalar_gather_preserves_bits(self):
        a = torch.tensor(1 + 2**-42, dtype=torch.float64)
        with enable():
            x = a.to("mps")
            i = torch.zeros(1, device="mps", dtype=torch.int64)
            self.assertTrue(
                torch.equal(
                    x.gather(0, i).cpu().view(torch.int64), a.view(1).view(torch.int64)
                )
            )

    def test_string_padding_and_pool_backward(self):
        torch.manual_seed(37)
        a = torch.randn(2, 2, 5, 7, dtype=torch.float64)
        w = torch.randn(3, 2, 2, 3, dtype=torch.float64)

        def run(device):
            x = a.to(device).detach().requires_grad_()
            v = torch.nn.functional.conv2d(x, w.to(device), padding="valid")
            y = torch.nn.functional.max_pool2d(v, 2, ceil_mode=True)
            y.square().sum().backward()
            return y.cpu(), x.grad.cpu()

        ref = run("cpu")
        with enable():
            for x, y in zip(run("mps"), ref):
                torch.testing.assert_close(x, y, rtol=2e-13, atol=3e-13)

    def test_fft_backward(self):
        a = torch.arange(1, 18, dtype=torch.float64).sin()

        def run(device):
            x = a.to(device).detach().requires_grad_()
            torch.fft.rfft(x).abs().sum().backward()
            return x.grad.cpu()

        ref = run("cpu")
        with enable():
            torch.testing.assert_close(run("mps"), ref, rtol=1e-13, atol=2e-13)
