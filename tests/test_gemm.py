import ctypes
import unittest

import numpy as np
import torch

from mps_fp64 import enable


class MatrixProducts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fma = ctypes.CDLL(None).fma
        cls.fma.argtypes = [ctypes.c_double] * 3
        cls.fma.restype = ctypes.c_double

    def check_ordered_dot(self, a, b, actual):
        """Use one hardware binary64 FMA per k, in the kernel's reduction order."""
        if a.ndim == 2:
            a, b, actual = a.unsqueeze(0), b.unsqueeze(0), actual.unsqueeze(0)
        a, b = a.numpy(), b.numpy()
        expected = np.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=np.float64)
        for batch, row, col in np.ndindex(expected.shape):
            acc = 0.0
            for inner in range(a.shape[2]):
                acc = self.fma(a[batch, row, inner], b[batch, inner, col], acc)
            expected[batch, row, col] = acc
        bits = actual.cpu().numpy().view(np.uint64)
        match = (bits == expected.view(np.uint64)) | (np.isnan(bits.view(np.float64)) & np.isnan(expected))
        np.testing.assert_array_equal(match, True)

    def test_strides_broadcast_and_tile_tails(self):
        rng = torch.Generator().manual_seed(31815)
        a = torch.randn(4, 29, 23, dtype=torch.float64, generator=rng)
        b = torch.randn(1, 35, 27, dtype=torch.float64, generator=rng)

        def views(a, b):
            # Both operands have offsets and holes. Reuse one weight matrix in
            # two batches and exercise both transposed threadgroup loads.
            return a[1::2, 1:27:2, 2:20:2].mT, b[:, 1:35:2, 1:27:2].mT.expand(2, 13, 17)

        cpu_a, cpu_b = views(a, b)
        with enable() as mode:
            gpu_a, gpu_b = views(a.to("mps"), b.to("mps"))
            self.check_ordered_dot(cpu_a, cpu_b, torch.bmm(gpu_a, gpu_b))
            self.check_ordered_dot(cpu_a[0], cpu_b[0], gpu_a[0] @ gpu_b[0])
            self.assertFalse(mode.cpu_ops)

    def test_fma_chain_rounding_and_specials(self):
        # Include deeply cancelling products, exact negative powers of two,
        # adjacent representable values and transitions to nonfinite values.
        edge = np.array([0.0, -0.0, -1.0, 1.0, -2.0, 2.0, 0.5, -0.5,
                         1.0 + 2**-52, 1.0 - 2**-53, 2**-52, -2**-53,
                         2**-500, 2**500, np.nextafter(0.0, 1.0),
                         np.finfo(np.float64).max, np.inf, np.nan], dtype=np.float64)
        a = torch.from_numpy(np.stack([np.roll(edge, i) for i in range(9)]))
        b = torch.from_numpy(np.stack([np.roll(edge, i) for i in range(17)], axis=1).copy())
        # Half of the rows / columns contain only moderate finite dyadics so
        # the same matrix also exercises long sequences in the fast window.
        rng = np.random.default_rng(12417)
        a[:5] = torch.from_numpy(rng.choice(edge[:12], (5, a.shape[1])))
        b[:, :8] = torch.from_numpy(rng.choice(edge[:12], (b.shape[0], 8)))
        a[0] = torch.tensor([1., -1., -2., 2., 0., -0.] * 3, dtype=torch.float64)
        b[:, 0] = torch.tensor([1., 1., 2., 1., 1., -1.] * 3, dtype=torch.float64)
        with enable() as mode:
            self.check_ordered_dot(a, b, a.to("mps") @ b.to("mps"))
            self.assertFalse(mode.cpu_ops)

    def test_empty_inner_and_batches(self):
        with enable() as mode:
            def empty(shape):
                return torch.empty(shape, dtype=torch.float64, device="mps")

            for batch, m, k, n in [(3, 7, 0, 11), (0, 7, 6, 11), (2, 0, 9, 17), (2, 8, 9, 0)]:
                got = torch.bmm(empty((batch, m, k)), empty((batch, k, n)))
                self.assertEqual(got.shape, (batch, m, n))
                self.assertEqual(torch.count_nonzero(got.cpu().view(torch.int64)).item(), 0)
            self.assertFalse(mode.cpu_ops)
