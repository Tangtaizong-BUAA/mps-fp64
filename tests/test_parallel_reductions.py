import unittest

import numpy as np
import torch
import torch.nn.functional as F

from mps_fp64 import enable
from mps_fp64.kernels import library

I64 = torch.int64
F64 = torch.float64


def legacy_tree(x, op):
    """Run the retained 256-thread reference tree on raw MPS bit patterns."""
    rows, width = x.shape
    while width > 1:
        blocks = (width + 255) // 256
        result = torch.empty((rows, blocks), dtype=I64, device=x.device)
        library().row_sum(
            result,
            x,
            width,
            blocks,
            op,
            threads=(rows * blocks * 256,),
            group_size=(256,),
        )
        x, width = result, blocks
    return x.reshape(rows).cpu()


class ParallelReductions(unittest.TestCase):
    def test_tree_preserves_rounding_and_payloads(self):
        rng = np.random.default_rng(3923)
        edges = np.array(
            [
                0,
                1 << 63,
                1,
                (1 << 63) + 1,
                (1 << 52) - 1,
                1 << 52,
                0x7FEFFFFFFFFFFFFF,
                0xFFEFFFFFFFFFFFFF,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000001234,
                0xFFF0000000000019,
            ],
            dtype=np.uint64,
        )
        for width in (
            1,
            2,
            3,
            4,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            128,
            255,
            256,
            257,
            1051,
            65537,
        ):
            with self.subTest(width=width):
                raw = rng.bit_generator.random_raw((5, width))
                raw[0] = np.resize(edges, width)
                raw[1] = np.resize(edges[::-1], width)
                raw[2] = 1 << 63
                raw[3] = rng.normal(size=width).view(np.uint64)
                words = torch.from_numpy(raw.view(np.int64).copy()).to("mps")
                with enable() as mode:
                    x = words.view(F64)
                    for op, fun in enumerate(
                        (torch.sum, torch.prod, torch.amax, torch.amin)
                    ):
                        # Copy integer bits to the host without changing NaNs or signed zeros.
                        actual = fun(x, dim=-1).view(I64).cpu()
                        self.assertTrue(
                            torch.equal(actual, legacy_tree(words, op)), (width, op)
                        )
                    self.assertFalse(mode.cpu_ops)

    def test_strided_reduce_and_empty_axes(self):
        base = torch.arange(3 * 5 * 7, dtype=F64).reshape(3, 5, 7)
        with enable() as mode:
            x = base.to("mps").transpose(0, 1)[:, :, ::2]
            ref = base.transpose(0, 1)[:, :, ::2]
            for dims in ((0,), (1,), (2,), (1, 0), (2, 0)):
                for fun in (torch.sum, torch.amax, torch.amin):
                    self.assertTrue(
                        torch.equal(
                            fun(x, dim=dims, keepdim=True).cpu(),
                            fun(ref, dim=dims, keepdim=True),
                        )
                    )
            with self.assertRaises((RuntimeError, ValueError)):
                x.sum((0, 0))
            empty = torch.empty((0, 4), device="mps", dtype=F64)
            self.assertTrue(torch.equal(empty.sum(0).cpu(), torch.zeros(4, dtype=F64)))
            single = base[:1].to("mps")
            result = single.sum(0)
            result.fill_(99)
            self.assertTrue(torch.equal(single.cpu(), base[:1]))
            self.assertFalse(mode.cpu_ops)

    def test_column_tiles_against_row_tree(self):
        rng = np.random.default_rng(876)
        for shape, dims in (((257, 513), (0,)), ((3, 8, 33, 199), (1, 2))):
            raw = rng.bit_generator.random_raw(shape).view(np.int64)
            raw.reshape(-1)[::9] = -9223372036854775808
            raw.reshape(-1)[::343] = 0x7FF8000000000041
            words = torch.from_numpy(raw).to("mps")
            keep = tuple(i for i in range(len(shape)) if i not in dims)
            packed = (
                words.permute(*keep, *dims)
                .contiguous()
                .flatten(len(keep))
                .reshape(-1, int(np.prod([shape[i] for i in dims])))
            )
            with enable():
                x = words.view(F64)
                for op, fun in enumerate(
                    (torch.sum, torch.prod, torch.amax, torch.amin)
                ):
                    if op == 1 and len(dims) != 1:
                        continue
                    actual = fun(x, dim=dims[0] if op == 1 else dims)
                    self.assertTrue(
                        torch.equal(
                            actual.view(I64).cpu().reshape(-1), legacy_tree(packed, op)
                        )
                    )

    def test_sort_preserves_native_strides(self):
        base = torch.arange(7 * 11, dtype=F64).reshape(7, 11)
        with enable():
            for transform in (
                lambda a: a,
                lambda a: a.mT,
                lambda a: a[:, ::2],
                lambda a: a[:1].expand(3, -1),
            ):
                host, x = transform(base), transform(base.to("mps"))
                for dim in (0, 1):
                    v, i = x.sort(dim=dim, stable=True)
                    rv, ri = host.sort(dim=dim, stable=True)
                    self.assertEqual(v.stride(), rv.stride())
                    self.assertEqual(i.stride(), ri.stride())
                    self.assertTrue(torch.equal(v.cpu().view(I64), rv.view(I64)))
                    self.assertTrue(torch.equal(i.cpu(), ri))

    def test_stable_sort_and_first_index(self):
        rng = np.random.default_rng(74)
        special = np.array(
            [
                0,
                1 << 63,
                0x7FF8000000000231,
                0xFFF0000000000009,
                1,
                (1 << 63) + 1,
                0x7FF0000000000000,
                0xFFF0000000000000,
            ],
            dtype=np.uint64,
        )
        for width in (1, 3, 8, 17, 32, 33, 64, 255, 256, 257, 1031, 131075):
            with self.subTest(width=width):
                raw = rng.integers(0, 16, size=(4, width), dtype=np.uint64) + np.uint64(
                    0x3FF0000000000000
                )
                raw[1] = np.resize(special, width)
                raw[2] = np.resize(np.array([1 << 63, 0], dtype=np.uint64), width)
                raw[3] = rng.bit_generator.random_raw(width)
                host = torch.from_numpy(raw.copy().view(np.float64))
                # Transposed layout exercises a non-trailing sorting/reduction axis.
                cpu = host.mT
                with enable() as mode:
                    x = host.to("mps").mT
                    for descending in (False, True):
                        actual, ids = x.sort(dim=0, stable=True, descending=descending)
                        expected, expected_ids = cpu.sort(
                            dim=0, stable=True, descending=descending
                        )
                        self.assertTrue(torch.equal(ids.cpu(), expected_ids))
                        self.assertTrue(
                            torch.equal(actual.cpu().view(I64), expected.view(I64))
                        )
                    for fun in (torch.max, torch.min):
                        actual, ids = fun(x, dim=0, keepdim=True)
                        expected, expected_ids = fun(cpu, dim=0, keepdim=True)
                        self.assertTrue(torch.equal(ids.cpu(), expected_ids))
                        # The selected input's payload is part of the backend contract.
                        selected = cpu.view(I64).gather(0, expected_ids)
                        self.assertTrue(torch.equal(actual.cpu().view(I64), selected))
                    self.assertEqual(x.argmax(dim=None, keepdim=True).shape, (1, 1))
                    self.assertFalse(mode.cpu_ops)

    def test_pooling_and_negative_divisor(self):
        torch.manual_seed(72)
        for dims, shape in ((2, (2, 3, 7, 9)), (3, (2, 2, 4, 7, 5))):
            src = torch.randn(shape, dtype=F64)
            avg = getattr(F, f"avg_pool{dims}d")
            maxpool = getattr(F, f"max_pool{dims}d")
            for divisor in (None, -3):
                with self.subTest(dims=dims, divisor=divisor):
                    cpu = src.detach().requires_grad_()
                    ref = avg(
                        cpu,
                        3,
                        stride=2,
                        padding=1,
                        ceil_mode=True,
                        divisor_override=divisor,
                    )
                    grad = torch.randn_like(ref)
                    ref.backward(grad)
                    with enable() as mode:
                        x = src.to("mps").detach().requires_grad_()
                        out = avg(
                            x,
                            3,
                            stride=2,
                            padding=1,
                            ceil_mode=True,
                            divisor_override=divisor,
                        )
                        out.backward(grad.to("mps"))
                        torch.testing.assert_close(
                            out.cpu(), ref.detach(), rtol=2e-15, atol=1e-15
                        )
                        torch.testing.assert_close(
                            x.grad.cpu(), cpu.grad, rtol=3e-15, atol=1e-15
                        )
                        vals, ids = maxpool(
                            x, 3, stride=2, padding=1, return_indices=True
                        )
                        exp_vals, exp_ids = maxpool(
                            src, 3, stride=2, padding=1, return_indices=True
                        )
                        self.assertTrue(
                            torch.equal(vals.cpu().view(I64), exp_vals.view(I64))
                        )
                        self.assertTrue(torch.equal(ids.cpu(), exp_ids))
                        self.assertFalse(mode.cpu_ops)

    def test_square_pool_bits_gradients_and_layout(self):
        rng = np.random.default_rng(707)
        raw = rng.bit_generator.random_raw((2, 3, 16, 20))
        edges = np.array(
            [
                0,
                1 << 63,
                1,
                2,
                3,
                4,
                (1 << 52) + 1,
                0x7FF0000000000000,
                0x7FF8000000000013,
                0xFFF8000000000022,
            ],
            dtype=np.uint64,
        )
        raw.reshape(-1)[::3] = np.resize(edges, raw.reshape(-1)[::3].size)
        cpu = torch.from_numpy(raw.view(np.float64)).contiguous(
            memory_format=torch.channels_last
        )
        for maximum in (False, True):
            for divisor in (None,) if maximum else (None, 1, -4, 17):
                grad = torch.randn(2, 3, 8, 10, dtype=F64)
                grad.reshape(-1)[::11] = -0.0
                grad.reshape(-1)[::17] = torch.finfo(F64).tiny
                a = cpu.detach().requires_grad_()

                def run(t):
                    if maximum:
                        return F.max_pool2d(t, 2, return_indices=True)
                    return F.avg_pool2d(t, 2, divisor_override=divisor), None

                expected, expected_ids = run(a)
                expected.backward(grad)
                with enable() as mode:
                    x = cpu.to("mps").detach().requires_grad_()
                    out, ids = run(x)
                    out.backward(grad.to("mps"))
                    self.assertEqual(out.stride(), expected.stride())
                    self.assertEqual(x.grad.stride(), a.grad.stride())
                    if maximum:
                        self.assertEqual(ids.stride(), expected_ids.stride())
                        self.assertTrue(torch.equal(ids.cpu(), expected_ids))
                        self.assertTrue(
                            torch.equal(
                                out.cpu().view(I64), expected.detach().view(I64)
                            )
                        )
                    else:
                        result, ref = out.detach().cpu(), expected.detach()
                        self.assertTrue(
                            torch.equal(torch.isnan(result), torch.isnan(ref))
                        )
                        valid = ~torch.isnan(ref)
                        self.assertTrue(
                            torch.equal(result[valid].view(I64), ref[valid].view(I64))
                        )
                    self.assertTrue(
                        torch.equal(x.grad.cpu().view(I64), a.grad.view(I64))
                    )
                    self.assertFalse(mode.cpu_ops)

    def test_unpool_collisions_bounds_and_payloads(self):
        words = np.array(
            [
                0x3FF0000000000001,
                0x7FF8000000000080,
                0x8000000000000000,
                0x3FF0000000000003,
            ],
            dtype=np.uint64,
        )
        for rank, s in ((2, (2, 2)), (3, (1, 2, 2))):
            src = torch.from_numpy(words.view(np.float64)).reshape(1, 1, *s)
            # The backend deterministically keeps the last source for each destination.
            idx = torch.tensor([2, 1, 6, 2]).reshape_as(src)
            out_shape = [2 * n for n in s]
            expected = torch.zeros(1, 1, *out_shape, dtype=I64)
            expected.reshape(-1)[[1, 2, 6]] = src.view(I64).reshape(-1)[[1, 3, 2]]
            with enable() as mode:

                def run(indices):
                    return getattr(F, f"max_unpool{rank}d")(
                        src.to("mps"), indices.to("mps"), 2
                    )

                for _ in range(3):
                    actual = run(idx)
                    self.assertTrue(torch.equal(actual.cpu().view(I64), expected))
                for bad in (-1, int(np.prod(out_shape)), 1 << 50):
                    invalid = idx.clone()
                    invalid.reshape(-1)[2] = bad
                    with self.assertRaisesRegex(RuntimeError, "out of range"):
                        run(invalid)
                self.assertFalse(mode.cpu_ops)
