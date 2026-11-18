import math
import struct
import unittest
from unittest.mock import patch

import torch

from mps_fp64.exact_gemm import _certificate, try_exact_mm
from mps_fp64.kernels import library


def upload(tensor):
    return tensor.view(torch.int64).to("mps").view(torch.float64)


def strict_product(a, b):
    """Invoke the original ascending-k, individually rounded binary64 FMA kernel."""
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batched = a.ndim == 3
    batches = a.shape[0] if batched else 1
    out = torch.empty((*a.shape[:-2], m, n), device="mps", dtype=torch.int64)
    library().mm_staged(
        out,
        a.view(torch.int64).contiguous(),
        b.view(torch.int64).contiguous(),
        m,
        n,
        k,
        threads=(((n + 15) // 16) * 16, ((m + 7) // 8) * 8, batches),
        group_size=(16, 8, 1),
    )
    return out.cpu()


class ExactLatticeMatrixProducts(unittest.TestCase):
    def setUp(self):
        # Exercise the production kernels on quick matrices, bypassing only
        # the cost threshold for taking the synchronous eligibility decision.
        threshold = patch("mps_fp64.exact_gemm._MIN_WORK", 1)
        threshold.start()
        self.addCleanup(threshold.stop)
        self.rng = torch.Generator().manual_seed(631785)

    def integers(self, n, shape):
        return torch.randint(
            -n, n + 1, shape, generator=self.rng, dtype=torch.int32
        ).double()

    def checked(self, a, b, *, out=None):
        reference = strict_product(a, b)
        actual = try_exact_mm(a, b, a.ndim == 3, out=out)
        self.assertIsNotNone(actual)
        self.assertTrue(torch.equal(actual.view(torch.int64).cpu(), reference))
        if out is not None:
            self.assertEqual(out.data_ptr(), actual.data_ptr())
        return actual.view(torch.int64).cpu().view(torch.float64)

    def test_odd_and_signed_integer_precisions(self):
        for limit in (7, 255, 2047, 32767, 65535):
            with self.subTest(limit=limit):
                a = upload(self.integers(limit, (65, 131)) * 2.0**-17)
                b = upload(self.integers(limit, (131, 71)) * 2.0**8)
                actual = self.checked(a, b)
                if limit >= 2047:
                    self.assertTrue((actual.float().double() != actual).any().item())

    def test_exact_accumulators_around_24_bits(self):
        for k in (1024, 1025):
            # 1024 * 128 * 128 = 2**24. The odd first product also
            # distinguishes a binary32 accumulator from a reduced one.
            a = torch.full((32, k), 128.0, dtype=torch.float64)
            b = torch.full((k, 33), 128.0, dtype=torch.float64)
            a[:, 0], b[0, :] = 127.0, 127.0
            result = self.checked(upload(a), upload(b))
            self.assertEqual(result[0, 0].item(), (k - 1) * 128**2 + 127**2)
            if k == 1025:
                self.assertNotEqual(result[0, 0].float().item(), result[0, 0].item())

    def test_signed_zero_subnormals_and_overflow(self):
        z = torch.zeros(32, 32, dtype=torch.float64)
        z[::2] = -0.0
        b = self.integers(9, (32, 32))
        self.assertFalse(
            self.checked(upload(z), upload(b)).view(torch.int64).any().item()
        )
        tiny = math.ulp(0.0)
        for limit in (3, 2047):
            a = self.integers(limit, (32, 32)) * tiny
            self.checked(upload(a), upload(b))
        self.checked(upload(self.integers(9, (32, 32)) * 2.0**993), upload(b))
        too_large = upload(torch.full((32, 32), 2.0**1023, dtype=torch.float64))
        self.assertIsNone(try_exact_mm(too_large, upload(b), False))
        self.assertIsNone(
            try_exact_mm(
                upload(torch.full_like(z, tiny)), upload(torch.full_like(z, 0.5)), False
            )
        )

    def test_strided_broadcast_batch_and_destination(self):
        a = upload(self.integers(2047, (5, 83, 69)))[1::2, 1:80:2, 2:67:2].mT
        b = upload(self.integers(127, (1, 77, 84)))[:, 1:74:2, 2:82:2].mT.expand(
            2, 40, 37
        )
        dest = torch.empty((2, 33, 37), dtype=torch.int64, device="mps")
        self.checked(a, b, out=dest)
        # The source snapshot is fully read before a qualifying output is written.
        square = upload(self.integers(15, (48, 48)))
        rhs = upload(self.integers(15, (48, 48)))
        self.checked(square, rhs, out=square.view(torch.int64))

    def test_incomplete_samples_never_authorize_and_no_stale_cache(self):
        a = upload(torch.full((64, 64), 1024.0, dtype=torch.float64))
        b = upload(self.integers(7, (64, 64)))
        out = torch.full((64, 64), -12345, device="mps", dtype=torch.int64)
        self.checked(a, b, out=out)
        # Index 7 is outside the initial deterministic sample. It requires a
        # finer grid and a wider digit representation, then becomes ineligible.
        for value in (1025.0, 1024.5):
            word = struct.unpack("=q", struct.pack("=d", value))[0]
            a.view(torch.int64).reshape(-1)[7] = word
            self.checked(a, b, out=out)
        for value in (1.0 + 2.0**-52, float("inf"), float("-inf"), float("nan")):
            a.view(torch.int64).reshape(-1)[7] = struct.unpack(
                "=q", struct.pack("=d", value)
            )[0]
            out.fill_(-12345)
            self.assertIsNone(try_exact_mm(a, b, False, out=out))
            self.assertTrue((out.cpu() == -12345).all().item())
        # Even a zero matrix cannot hide a nonfinite multiplicand.
        self.assertIsNone(
            try_exact_mm(a, upload(torch.zeros(64, 64, dtype=torch.float64)), False)
        )

    def test_wide_panels_cancel_to_a_single_low_bit(self):
        # Both radix digits are nonzero on each side. A long positive run can
        # lose the final odd bit in FP32 before the opposite run cancels it.
        k = 4097
        a = torch.full((32, k), 32767.0, dtype=torch.float64)
        b = torch.full((k, 32), 32765.0, dtype=torch.float64)
        a[:, -1] = 1.0
        b[2048:-1] = -b[2048:-1]
        b[-1] = 1.0
        result = self.checked(upload(a * 2.0**-511), upload(b * 2.0**490))
        self.assertTrue((result == 2.0**-21).all().item())

    def test_full_binary64_integer_bound(self):
        maxbits = struct.unpack("=Q", struct.pack("=d", 32768.0))[0]
        # A smaller odd coefficient elsewhere can make the common exponent zero.
        metadata = ((maxbits, 0), (maxbits, 0))
        k = 1 << 23
        self.assertIsNotNone(_certificate(metadata, k))
        self.assertIsNone(_certificate(metadata, k + 1))
