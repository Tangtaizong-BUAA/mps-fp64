import math
import warnings

import torch

from .kernels import bits, F64, I64, library


def axis_check(x, dim):
    if not -max(1, x.ndim) <= dim < max(1, x.ndim):
        raise IndexError("axis out of range")
    return dim % max(1, x.ndim)


def aminmax(x, *, dim=None, keepdim=False, outputs=None):
    if dim is None:
        if not x.numel():
            raise RuntimeError("aminmax requires a dimension for an empty input")
        raw = bits(x).reshape(-1)
        shape = (1,) * x.ndim if keepdim else ()
    else:
        dim = axis_check(x, dim)
        raw = bits(x).movedim(dim, -1) if x.ndim else bits(x).reshape(1)
        if raw.shape[-1] == 0:
            raise IndexError("aminmax reduction dimension is empty")
        shape = tuple(
            1 if i == dim else n for i, n in enumerate(x.shape) if keepdim or i != dim
        )
    width = raw.shape[-1]
    rows = raw.numel() // width
    if width > 2**32 - 1:
        raise NotImplementedError("aminmax width exceeds uint32")
    low = high = raw.contiguous()
    first = True
    while True:
        items = min(8, (width + 31) // 32)
        chunks = (width + 32 * items - 1) // (32 * items)
        threads = rows * chunks * 32
        if threads > 2**32 - 1:
            raise NotImplementedError("aminmax grid exceeds uint32")
        low_out = torch.empty((rows, chunks), device=x.device, dtype=I64)
        high_out = torch.empty_like(low_out)
        if threads:
            kernel = library().aminmax_blocks
            if kernel.thread_execution_width != 32:
                raise NotImplementedError("aminmax requires 32-lane SIMD groups")
            kernel(
                low_out,
                high_out,
                low,
                high,
                (width, chunks, items, first, dim is None),
                threads=threads,
                group_size=32,
            )
        if chunks == 1:
            break
        low, high, width, first = low_out, high_out, chunks, False
    result = tuple(v.reshape(shape).view(F64) for v in (low_out, high_out))
    if outputs is None:
        return result
    for out in outputs:
        if out.dtype != F64 or out.device != x.device:
            raise RuntimeError("aminmax output dtype and device must match the input")
        if out.shape == shape and torch._debug_has_internal_overlap(out) == 1:
            raise RuntimeError("aminmax output has overlapping elements")
    for out, value in zip(outputs, result):
        if out.shape != shape:
            if out.numel():
                warnings.warn(
                    "Resizing a nonempty aminmax output tensor.", stacklevel=3
                )
            out.resize_(shape)
        bits(out).copy_(bits(value))
    return outputs


def sort(x, dim=-1, descending=False, stable=None):
    dim = axis_check(x, dim)
    if not x.ndim:
        return bits(x).clone().view(F64), torch.zeros((), device=x.device, dtype=I64)
    src = bits(x).movedim(dim, -1).contiguous()
    width = src.shape[-1]
    if width >= 2**30 or src.numel() >= 2**32:
        raise NotImplementedError("sort grid exceeds uint32")
    if width <= 1 or not x.numel():
        values, ids = src.clone(), torch.zeros_like(src)
    else:
        lib = library()
        values, ids = torch.empty_like(src), torch.empty_like(src)
        tile = min(256, 1 << (width - 1).bit_length())
        tile = max(tile, 32)
        tiles = (width + tile - 1) // tile
        threads = src.numel() // width * tiles * tile
        shader = lib.sort_simd_tile if tile == 32 else lib.sort_group_tile
        if shader.thread_execution_width == 32 and threads < 2**32:
            shader(
                values,
                ids,
                src,
                width,
                tiles,
                int(descending),
                threads=(threads,),
                group_size=(128 if tile == 32 else tile,),
            )
            run = tile
        else:
            # At run length one each input's column is its original index.
            lib.merge_runs(
                values, ids, src, src, width, 1, int(descending), threads=(src.numel(),)
            )
            run = 2
        if run < width:
            spare, spare_ids = torch.empty_like(src), torch.empty_like(ids)
            while run < width:
                lib.merge_runs(
                    spare,
                    spare_ids,
                    values,
                    ids,
                    width,
                    run,
                    int(descending),
                    threads=(src.numel(),),
                )
                values, spare = spare, values
                ids, spare_ids = spare_ids, ids
                run *= 2
    values, ids = values.movedim(-1, dim), ids.movedim(-1, dim)
    # Native sort preserves dense input layout and densifies sliced or
    # overlapping layouts the same way empty_like(preserve_format) does.
    if values.stride() != x.stride():
        stride = torch.empty_like(x, dtype=I64, device="meta").stride()
        if values.stride() != stride:
            v = torch.empty_strided(x.shape, stride, dtype=I64, device=x.device)
            i = torch.empty_like(v)
            values, ids = v.copy_(values), i.copy_(ids)
    return values.view(F64), ids


def extreme(x, dim, keepdim, maximum):
    if dim is None:
        if keepdim:
            output_shape = (1,) * x.ndim
        else:
            output_shape = ()
        x = bits(x).reshape(-1).view(F64)
        dim = 0
    else:
        dim = axis_check(x, dim)
        output_shape = tuple(
            1 if axis == dim else size
            for axis, size in enumerate(x.shape)
            if keepdim or axis != dim
        )
    if not x.ndim:
        return bits(x).clone().view(F64), torch.zeros((), device=x.device, dtype=I64)
    src = bits(x).movedim(dim, -1).contiguous()
    width = src.shape[-1]
    if not width:
        raise IndexError("empty reduction dimension")
    rows = math.prod(src.shape[:-1])
    if width >= 2**32 or rows >= 2**32:
        raise NotImplementedError("extremum grid exceeds uint32")
    values = torch.empty(output_shape, device=x.device, dtype=I64)
    indices = torch.empty_like(values)
    if not rows:
        return values.view(F64), indices
    lib = library()
    max_chunks = (width + 255) // 256
    if (
        width <= 8
        or lib.extreme_blocks.thread_execution_width != 32
        or rows * max_chunks > (2**32 - 1) // 32
    ):
        lib.row_extremum(values, indices, src, width, int(maximum), threads=(rows,))
    else:
        prior = src
        first = True
        while True:
            # Large enough to amortize each group's shuffle, small enough to
            # expose independent groups even for one very long input row.
            span = min(width, 256)
            chunks = (width + span - 1) // span
            out = (
                values
                if chunks == 1
                else torch.empty((rows, chunks), device=x.device, dtype=I64)
            )
            ids = indices if chunks == 1 else torch.empty_like(out)
            lib.extreme_blocks(
                out,
                ids,
                src,
                prior,
                width,
                chunks,
                span,
                int(maximum),
                int(first),
                threads=(rows * chunks * 32,),
                group_size=(128,),
            )
            if chunks == 1:
                break
            src, prior, width, first = out, ids, chunks, False
    return values.view(F64), indices
