import ctypes
import unittest

import torch

from mps_fp64 import enable
from mps_fp64.kernels import library


D, I = torch.float64, torch.int64


def original_dot(a, b):
    m, k = a.shape
    n = b.shape[1]
    result = torch.empty((m, n), device=a.device, dtype=I)
    strides = (0, *a.stride(), 0, *b.stride())
    library().mm_staged_strided(
        result,
        a,
        b,
        m,
        n,
        k,
        strides,
        threads=(((n + 15) // 16) * 16, ((m + 7) // 8) * 8, 1),
        group_size=(16, 8, 1),
    )
    return result.cpu()


class MatrixRoutes(unittest.TestCase):
    def test_ordinary_dense_and_narrow_preserve_fma(self):
        generator = torch.Generator().manual_seed(8255)
        fma = ctypes.CDLL(None).fma
        fma.argtypes = [ctypes.c_double] * 3
        fma.restype = ctypes.c_double
        for m, k, n in ((385, 391, 393), (4097, 173, 3)):
            a = torch.randn(m, k, dtype=D, generator=generator)
            b = torch.randn(k, n, dtype=D, generator=generator)
            a[0, :6] = torch.tensor(
                [1e300, -1e300, 0.0, -0.0, 2.0**-1074, 1.0], dtype=D
            )
            b[:6, 0] = 1
            a[-1, -1] = torch.inf
            with enable() as mode:
                x, y = a.to("mps"), b.to("mps")
                if n == 3:
                    x = x.T.contiguous().T
                    y = y.T.contiguous().T
                got = (x @ y).cpu()
                self.assertTrue(got.view(I).equal(original_dot(x, y)))
                for row, col in ((0, 0), (1, 1), (m // 2, n // 2)):
                    acc = 0.0
                    for av, bv in zip(a[row].tolist(), b[:, col].tolist()):
                        acc = fma(av, bv, acc)
                    self.assertTrue(
                        got[row, col].view(I).equal(torch.tensor(acc, dtype=D).view(I))
                    )
                self.assertFalse(mode.cpu_ops)

    def test_decomposed_mv_rounding(self):
        gen = torch.Generator().manual_seed(72661)
        for cols in (33, 64, 97, 128, 255, 256, 257, 1025):
            rows = max(159, (65536 + cols - 1) // cols)
            a = torch.randn(rows, cols, dtype=D, generator=gen)
            a[:3, :] = -0.0
            a[2, 0] = torch.inf
            a[3, 0] = torch.tensor(0x7FF8000000000058, dtype=I).view(D)
            v = torch.rand(cols, dtype=D, generator=gen)
            for transposed in (False, True):
                with enable() as mode:
                    x = a.T.contiguous().to("mps").T if transposed else a.to("mps")
                    vector = v.to("mps")
                    got = torch.mv(x, vector)
                    reference = (x * vector).sum(-1)
                    self.assertTrue(got.view(I).cpu().equal(reference.view(I).cpu()))
                    self.assertFalse(mode.cpu_ops)
        with enable():
            h = torch.randn(1024, 67, generator=gen, dtype=D)
            v = torch.randn(67, generator=gen, dtype=D)
            x, y = h.to("mps").requires_grad_(), v.to("mps").requires_grad_()
            dx, dy = torch.autograd.grad(torch.mv(x, y).sum(), (x, y))
            torch.testing.assert_close(dx.cpu(), v.expand_as(h), rtol=0, atol=0)
            torch.testing.assert_close(dy.cpu(), h.sum(0), rtol=2e-13, atol=2e-13)

    def test_out_shape_strides_alias_and_empty(self):
        g = torch.Generator().manual_seed(26402)
        with enable() as mode:
            a = torch.randn(33, 33, dtype=D, generator=g).to("mps")
            b = torch.randn_like(a)
            reference = (a @ b).cpu()
            resize = torch.empty(3, device="mps", dtype=D)
            with self.assertWarns(UserWarning):
                self.assertIs(torch.mm(a, b, out=resize), resize)
            self.assertEqual(resize.shape, reference.shape)
            self.assertTrue(resize.cpu().view(I).equal(reference.view(I)))
            for z in (torch.empty(33, 66, dtype=D, device="mps")[:, 1::2], a):
                self.assertIs(torch.mm(a, b, out=z), z)
                self.assertTrue(z.cpu().view(I).equal(reference.view(I)))
            with self.assertRaises(RuntimeError):
                torch.mm(
                    a, b, out=torch.empty(33, 33, dtype=torch.float32, device="mps")
                )
            with self.assertRaises(RuntimeError):
                torch.mm(a, b, out=torch.empty(1, device="mps", dtype=D).expand(33, 33))
            a = torch.empty(3, 7, 0, device="mps", dtype=D)
            b = torch.empty(3, 0, 9, device="mps", dtype=D)
            out = torch.ones(3, 7, 9, device="mps", dtype=D)
            self.assertIs(torch.bmm(a, b, out=out), out)
            self.assertEqual(out.cpu().abs().max().item(), 0)
            self.assertFalse(mode.cpu_ops)

    def test_lazy_negative_input_and_output(self):
        with enable() as mode:
            a = torch.tensor([[2.0, 4.0], [7.0, 8.0]], device="mps", dtype=D)
            storage = torch.zeros_like(a)
            negative = torch._neg_view(a)
            out = torch._neg_view(storage)
            expected = -(a.cpu() @ a.cpu())
            self.assertIs(torch.mm(negative, a, out=out), out)
            self.assertTrue(out.cpu().equal(expected))
            self.assertTrue(storage.cpu().equal(-expected))
            self.assertFalse(mode.cpu_ops)
