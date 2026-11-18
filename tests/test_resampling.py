from itertools import product
import unittest

import torch
from torch.nn import functional as F

from mps_fp64 import enable


MODES = (
    ("linear", 1, False),
    ("bilinear", 2, False),
    ("bicubic", 2, False),
    ("trilinear", 3, False),
    ("bilinear", 2, True),
    ("bicubic", 2, True),
)


class Resampling(unittest.TestCase):
    def test_oppositely_scaled_axes(self):
        # Expanding the width first would allocate 2^32 floats for this output.
        size = 65536
        x = torch.linspace(-2.0, 1.0, size, dtype=torch.float64).reshape(1, 1, size, 1)
        ref = F.interpolate(x, size=(1, size), mode="bilinear", align_corners=False)
        with enable() as backend:
            y = F.interpolate(
                x.to("mps"), size=(1, size), mode="bilinear", align_corners=False
            )
            torch.testing.assert_close(y.cpu(), ref, rtol=0, atol=0)
            self.assertFalse(backend.cpu_ops)

    def test_public_forward_backward(self):
        generator = torch.Generator().manual_seed(74129)
        for (mode, rank, aa), align, kind in product(
            MODES, (False, True), ("size", "scale", "recompute")
        ):
            # Size-one axes, a nonzero storage offset, both enlargement and shrinkage.
            spatial = (3, 1, 7)[-rank:]
            source = torch.randn(
                (2, 3, *spatial[:-1], 2 * spatial[-1] + 1),
                dtype=torch.float64,
                generator=generator,
            )
            options = (
                {"size": (2, 4, 3)[-rank:]}
                if kind == "size"
                else {
                    "scale_factor": (0.75, 3.2, 0.73)[-rank:],
                    "recompute_scale_factor": kind == "recompute",
                }
            )
            spec = {"mode": mode, "align_corners": align, "antialias": aa, **options}
            host = source[..., 1::2].detach().requires_grad_()
            expected = F.interpolate(host, **spec)
            grad = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
            grad2 = torch.randn(host.shape, dtype=torch.float64, generator=generator)
            gx = torch.autograd.grad(expected, host, grad)[0]
            with self.subTest(
                mode=mode, antialias=aa, align=align, sizing=kind
            ), enable() as backend:
                x = source.to("mps")[..., 1::2].detach().requires_grad_()
                upstream = grad.to("mps").requires_grad_()
                y = F.interpolate(x, **spec)
                dx = torch.autograd.grad(y, x, upstream, create_graph=True)[0]
                dgo = torch.autograd.grad(dx, upstream, grad2.to("mps"))[0]
                for value, ref in (
                    (y, expected),
                    (dx, gx),
                    (dgo, F.interpolate(grad2, **spec)),
                ):
                    self.assertEqual(value.dtype, torch.float64)
                    self.assertEqual(value.device.type, "mps")
                    torch.testing.assert_close(
                        value.detach().cpu(), ref.detach(), rtol=2e-14, atol=1e-14
                    )
                self.assertFalse(backend.cpu_ops)

    def test_coordinate_edges_and_nonfinite(self):
        for (mode, rank, aa), align in product(MODES, (False, True)):
            spatial = (4, 8, 9)[-rank:]
            dest = (1, 6, 7)[-rank:]
            x = torch.arange(
                2 * torch.Size(spatial).numel(), dtype=torch.float64
            ).reshape((1, 2, *spatial))
            x.reshape(-1)[::7] = float("inf")
            x.reshape(-1)[::13] = float("-inf")
            spec = {"size": dest, "mode": mode, "align_corners": align, "antialias": aa}
            with self.subTest(
                mode=mode, antialias=aa, align=align
            ), enable() as backend:
                y = F.interpolate(x.to("mps"), **spec).cpu()
                expected = F.interpolate(x, **spec)
                torch.testing.assert_close(
                    y, expected, rtol=3e-14, atol=2e-13, equal_nan=True
                )
                self.assertFalse(backend.cpu_ops)

    def test_antialias_channels_last_empty_and_aten_vec(self):
        x = (
            torch.linspace(-2.0, 3.0, 257 * 3 * 7, dtype=torch.float64)
            .reshape(1, 3, 257, 7)
            .contiguous(memory_format=torch.channels_last)
        )
        operators = (
            torch.ops.aten._upsample_bilinear2d_aa.vec,
            torch.ops.aten._upsample_bicubic2d_aa.vec,
        )
        for op in operators:
            with enable() as backend:
                y = op(x.to("mps"), [2, 1], False, None)
                ref = op(x, [2, 1], False, None)
                torch.testing.assert_close(y.cpu(), ref, rtol=2e-14, atol=2e-14)
                self.assertEqual(y.stride(), ref.stride())
                empty = op(x[:0].to("mps"), [2, 1], True, None)
                self.assertEqual(empty.shape, (0, 3, 2, 1))
                self.assertFalse(backend.cpu_ops)
        with enable():
            with self.assertRaises((ValueError, RuntimeError)):
                torch.ops.aten.upsample_bilinear2d_backward.default(
                    x.to("mps"), [4, 5], x.shape, False
                )
