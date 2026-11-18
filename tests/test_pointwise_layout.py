import math
import struct
import unittest

import torch

from mps_fp64 import enable
from mps_fp64.kernels import pointwise


class PointwiseLayout(unittest.TestCase):
    def test_inline_nan_payload(self):
        word = 0xFFF8000000123456
        scalar = struct.unpack("=d", struct.pack("=Q", word))[0]
        x = torch.zeros(11, dtype=torch.int64, device="mps").view(torch.float64)
        actual = pointwise("maximum", x, scalar).view(torch.int64).cpu().tolist()
        self.assertEqual(actual, [word - 2**64] * 11)

    def test_offsets_broadcast_and_signed_scalars(self):
        x = torch.arange(2 * 5 * 9, dtype=torch.float64).reshape(2, 5, 9)
        row = torch.arange(5, dtype=torch.float64)

        def apply(a, b):
            t = a[:, :, 1::2].transpose(1, 2)
            return (
                t + b,  # strided input and trailing broadcast
                t - a[:, :1, 1:2],  # two storage offsets and zero strides
                torch.lt(t, b),  # boolean result
                torch.neg(t),  # unary strided shader
                t * (-0.0),  # Python scalar in a strided kernel
                2.0 - b,  # noncommutative scalar lhs
                torch.copysign(b, -0.0),  # inline constant sign bit
                b + math.ldexp(1.0, -1074),  # uint64 payload 1
                a[:, :, :0] + b[:, None],  # empty broadcast
                a + b[0].reshape(1, 1, 1, 1),  # a scalar increases output rank
                a + a[:, :1, :1].expand_as(a),
            )

        refs = apply(x, row)
        with enable() as mode:
            outs = apply(x.to("mps"), row.to("mps"))
            for actual, expected in zip(outs, refs, strict=True):
                actual = actual.cpu()
                if expected.dtype == torch.float64:
                    actual, expected = actual.view(
                        torch.int64
                    ), expected.contiguous().view(torch.int64)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertFalse(mode.cpu_ops)

    def test_backward_multiplication_order(self):
        rng = torch.Generator().manual_seed(235)
        grad = torch.randn(385, dtype=torch.float64, generator=rng)
        output = torch.randn(385, dtype=torch.float64, generator=rng).reshape(5, 77)
        grad = grad.reshape(5, 77)
        for op, expected in (
            (torch.ops.aten.tanh_backward.default, grad * (1 - output * output)),
            (torch.ops.aten.sigmoid_backward.default, grad * (1 - output) * output),
        ):
            native = op(grad, output)
            with enable() as mode:
                result = op(grad.to("mps"), output.to("mps")).cpu()
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
                # The CPU scalar vector-tail can contract 1-y*y into one FMA.
                torch.testing.assert_close(result, native, atol=5e-16, rtol=1e-14)
                self.assertFalse(mode.cpu_ops)
