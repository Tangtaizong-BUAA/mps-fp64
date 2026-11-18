import ctypes
import math
import unittest

import numpy as np
import torch

from mps_fp64 import enable
from mps_fp64.kernels import library
from mps_fp64.skinny_gemm import launch_skinny_mm, launch_small_bmm


class ShortReductionProducts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fma = ctypes.CDLL(None).fma
        cls.fma.argtypes = [ctypes.c_double] * 3
        cls.fma.restype = ctypes.c_double

    def check(self, a, b, x, y):
        batched = a.ndim == 3
        m, k = a.shape[-2:]
        n = b.size(-1)
        count = a.size(0) if batched else 1
        raw = torch.empty((*a.shape[:-2], m, n), device="mps", dtype=torch.int64)
        shaders = library()
        # The prior full-domain path compares every word, including NaN encodings.
        staged = torch.empty_like(raw)
        strides = [
            x.stride(0) if batched else 0,
            *x.stride()[-2:],
            y.stride(0) if batched else 0,
            *y.stride()[-2:],
        ]
        if batched and m == k == n and k in (3, 4):
            launch_small_bmm(shaders, raw, x, y, strides)
        else:
            self.assertTrue(launch_skinny_mm(shaders, raw, x, y, batched))
        shaders.mm_staged_strided(
            staged,
            x,
            y,
            m,
            n,
            k,
            strides,
            threads=(((n + 15) // 16) * 16, ((m + 7) // 8) * 8, count),
            group_size=(16, 8, 1),
        )
        got = raw.cpu()
        self.assertTrue(torch.equal(got, staged.cpu()))
        public = torch.bmm(x, y) if batched else torch.mm(x, y)
        self.assertTrue(torch.equal(got, public.view(torch.int64).cpu()))

        # Independent CPU hardware FMA in ascending k; BLAS may group an
        # accumulation differently. NaN payloads are platform-specific.
        a, b = a.reshape(count, m, k).numpy(), b.reshape(count, k, n).numpy()
        got = got.reshape(count, m, n).numpy().view(np.float64)
        rows = sorted(set([0, 1, 2, m - 1, *range(0, m, max(m // 17, 1))]))
        cols = sorted(set([0, 1, 2, n - 1, *range(0, n, max(n // 19, 1))]))
        batches = sorted({0, count - 1, *range(0, count, max(1, count // 5))})
        for z in batches:
            for i in rows:
                for j in cols:
                    ref = 0.0
                    for t in range(k):
                        ref = self.fma(a[z, i, t], b[z, t, j], ref)
                    if math.isnan(ref):
                        self.assertTrue(math.isnan(got[z, i, j]))
                    else:
                        self.assertEqual(
                            np.float64(ref).view(np.uint64),
                            got[z, i, j].view(np.uint64),
                        )

    def test_normal_and_rounding_inputs(self):
        generator = torch.Generator().manual_seed(3852)
        for k in (1, 2, 3, 4, 8, 16):
            m, n = (513, 517) if k == 1 else (1025, 1027)
            a = torch.randn(m, k, dtype=torch.float64, generator=generator)
            b = torch.randn(k, n, dtype=torch.float64, generator=generator)
            with enable() as mode:
                self.check(a, b, a.to("mps"), b.to("mps"))
                self.assertFalse(mode.cpu_ops)

    def test_strides_broadcast_and_full_range(self):
        finite = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2**-1022,
            -(2**-1022),
            float(2**-1074),
            -float(2**-1074),
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            math.nextafter(1.0, 0.0),
            math.nextafter(-1.0, -2.0),
            1 + 2**-52,
            -(2.0**400),
            2.0**-410,
            2.0**54,
            2.0**-60,
        ]
        values = torch.tensor(
            finite + [math.inf, -math.inf, math.nan], dtype=torch.float64
        )
        # Preserve explicit signaling, signed and noncanonical payloads all
        # the way to the GPU. The software FMA defines a canonical result.
        nan_words = np.array(
            [
                0x7FF0000000000001,
                0x7FF8123456789ABC,
                0xFFF0000000000001,
                0xFFF8FEDCBA987654,
            ],
            dtype=np.uint64,
        ).view(np.float64)
        values = torch.cat((values, torch.from_numpy(nan_words)))
        gen = torch.Generator().manual_seed(2209)
        for k in (1, 7):
            a = values[torch.randint(len(values), (2, 2 * k + 2, 1034), generator=gen)]
            b = values[torch.randint(len(values), (1, 1042, 2 * k + 3), generator=gen)]

            def views(x, y):
                left = x[:, 1 : 2 * k + 1 : 2, 1:1033:2].mT
                right = y[:, 2:1042, 1 : 2 * k + 1 : 2].mT.expand(2, k, 1040)
                return left, right

            with enable() as mode:
                self.check(*views(a, b), *views(a.to("mps"), b.to("mps")))
                self.assertFalse(mode.cpu_ops)

    def test_cancellation_halfway_and_exponent_shifts(self):
        generator = torch.Generator().manual_seed(92476)
        powers = torch.randint(-95, 96, (1024,), generator=generator)
        fractions = torch.randn(1024, dtype=torch.float64, generator=generator)
        c = torch.ldexp(fractions, powers)
        # Adjoin adjacent significands, exact cancellation and subnormals.
        c[:6] = torch.tensor(
            [1.0, -1.0, -(2**-53), 2**-53, 1 + 2**-52, -(1 + 2**-52)],
            dtype=torch.float64,
        )
        a = torch.stack((c, -fractions), dim=1)
        a[:6, 1] = torch.tensor([-1.0, 1.0, 1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
        b = torch.stack((torch.ones_like(c), c))
        with enable() as mode:
            self.check(a, b, a.to("mps"), b.to("mps"))
            self.assertFalse(mode.cpu_ops)

    def test_many_small_matrices_with_batch_tail(self):
        rng = torch.Generator().manual_seed(17582)
        batch = 4097
        for n in (3, 4):
            a = torch.randn(batch, n, n, dtype=torch.float64, generator=rng)
            b = torch.randn(batch, n, n, dtype=torch.float64, generator=rng)
            a[0, 0] = -0.0
            a[-1, 1, 1] = -math.ulp(0.0)
            b[batch // 2, 0, 0] = math.inf
            with enable() as mode:
                x, y = a.to("mps"), b.to("mps")
                self.check(a, b, x, y)
                # All batches reuse one transposed right operand, at a nonzero
                # storage offset. Each left operand is also transposed.
                b_shared = b[1:2].mT.expand_as(b)
                y_shared = y[1:2].mT.expand_as(y)
                self.check(a.mT, b_shared, x.mT, y_shared)
                self.assertFalse(mode.cpu_ops)

    def test_general_gemm_handles_other_sizes(self):
        with enable():
            for m, k, n in [
                (2048, 32, 2048),
                (4096, 1, 32),
                (128, 2, 128),
                (0, 2, 128),
            ]:
                a = torch.empty((m, k), device="mps", dtype=torch.float64)
                b = torch.empty((k, n), device="mps", dtype=torch.float64)
                sentinel = torch.empty(0, device="mps", dtype=torch.int64)
                self.assertFalse(launch_skinny_mm(library(), sentinel, a, b, False))
