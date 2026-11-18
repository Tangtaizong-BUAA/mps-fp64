import ctypes
import unittest
import numpy as np
import torch

from mps_fp64.kernels import library


class FusedOperation(unittest.TestCase):
    def test_fused_rounding(self):
        # Calling libSystem fma avoids separate product rounding and Python's
        # overflow exceptions. The CPU returns the IEEE result in all cases.
        fma = ctypes.CDLL(None).fma
        fma.argtypes = [ctypes.c_double] * 3
        fma.restype = ctypes.c_double
        rng = np.random.default_rng(583)
        n = 100000
        mant = rng.uniform(1, 2, (3, n)) * rng.choice([-1.0, 1.0], (3, n))
        exponents = rng.integers(-100, 101, (2, n))
        delta = rng.integers(-55, 26, n)
        a, b, c = [
            np.ldexp(m, e)
            for m, e in zip(mant, [*exponents, exponents.sum(axis=0) + delta])
        ]
        full = rng.bit_generator.random_raw((3, n)).view(np.float64)
        a, b, c = [np.r_[v, v, f] for v, f in zip((a, b, c), full)]
        c[n : 2 * n] = -(a[:n] * b[:n])
        ref = np.fromiter(
            (fma(i, j, k) for i, j, k in zip(a, b, c)), dtype=np.float64, count=len(a)
        )
        src = [torch.from_numpy(x.view(np.int64)).to("mps") for x in (a, b, c)]
        out = torch.empty_like(src[0])
        library().test_nearby_fma(out, *src, threads=(len(a),))
        result = out.cpu().numpy().view(np.float64)
        np.testing.assert_array_equal(
            (result.view(np.uint64) == ref.view(np.uint64))
            | (np.isnan(result) & np.isnan(ref)),
            True,
        )
