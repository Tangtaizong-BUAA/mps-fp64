"""Scatter by logical offsets. Equal destinations accumulate in source order."""

import math
import torch
from .kernels import F64, I64, bits, library, pointwise, scalar_bits


def accumulate(x, offsets, values, reduction, include_self=True):
    reducers = {"sum": 0, "prod": 1, "amax": 2, "amin": 3, "mean": 4, "set": 5}
    if reduction not in reducers:
        raise RuntimeError(f"invalid scatter reduction {reduction!r}")
    if values.dtype != F64 or values.device != x.device:
        raise RuntimeError("scatter source must match destination dtype and device")
    ordered_values = bits(values).expand(offsets.shape).contiguous().reshape(-1)
    out = bits(x).clone(memory_format=torch.contiguous_format)
    if ordered_values.numel():
        if ordered_values.numel() >= 2**32:
            raise NotImplementedError("scatter source exceeds Metal index range")
        if max(x.numel(), offsets.numel()) > 2**24:
            from .order import sort
            # Positive integer offsets have the same raw ordering as positive subnormals.
            # The native int64 sort on this MPS build rounds returned payloads above 2**24.
            keys_fp, order = sort(offsets.reshape(-1).view(F64), stable=True)
            keys = bits(keys_fp)
        else:
            keys, order = offsets.reshape(-1).sort(stable=True)
        library().scatter_segments(
            out,
            keys,
            order,
            ordered_values,
            keys.numel(),
            reducers[reduction],
            int(include_self),
            threads=(keys.numel(),),
        )
    return out.view(F64)


def offsets_like(x):
    if x.numel() <= 2**24:
        return torch.arange(x.numel(), device=x.device, dtype=I64).reshape(x.shape)
    if x.numel() >= 2**32:
        raise NotImplementedError("scatter target exceeds Metal index grid")
    raw = torch.empty(x.shape, device=x.device, dtype=I64)
    library().offset_identity(raw, threads=(x.numel(),))
    return raw


def selected_offsets(base, dim, index, *, gather=False):
    if base.numel() <= 2**24:
        return base.gather(dim, index) if gather else base.index_select(dim, index)
    from .backend import payload_words
    axis = dim % base.ndim if -base.ndim <= dim < base.ndim else None
    if axis is None:
        raise IndexError("scatter dimension out of range")
    words = payload_words(base.view(F64))
    if gather:
        expanded = index.unsqueeze(-1).expand(*index.shape,4)
        result = words.gather(axis, expanded)
    else:
        result = words.index_select(axis, index)
    return result.contiguous().view(I64).squeeze(-1)


def indexed(func, name, overload, args, kw):
    x, *rest = args
    if x.dtype != F64:
        raise NotImplementedError("scatter destination dtype")
    if name == "index_fill":
        dim, index, value = rest
        if index.dtype != I64 or index.ndim > 1:
            raise RuntimeError("index_fill needs a vector or scalar int64 index")
        if torch.is_tensor(value):
            if value.ndim:
                raise RuntimeError("index_fill requires a scalar value tensor")
            from .backend import tensor64

            value = tensor64(value)
        else:
            value = torch.full((), scalar_bits(value), dtype=I64, device=x.device).view(
                F64
            )
        base = offsets_like(x)
        if not x.ndim:
            if dim not in (-1, 0):
                raise IndexError("index_fill dimension out of range")
            base, dim = base.reshape(1), 0
        size = base.size(dim)
        if index.numel() and ((index < -size) | (index >= size)).any().item():
            raise IndexError("index_fill index out of range")
        chosen = torch.where(index < 0, index + size, index)
        return accumulate(x, selected_offsets(base, dim, chosen.reshape(-1)), value, "set")
    if name == "index_put":
        indices, values, *reduce = rest
        if values.device != x.device or values.dtype != x.dtype:
            raise RuntimeError(
                "index_put source must match destination dtype and device"
            )
        if reduce and reduce[0]:
            return accumulate(
                x, torch.ops.aten.index.Tensor(offsets_like(x), indices), values, "sum"
            )
        return func(bits(x), indices, bits(values)).view(F64)
    if name == "index_add":
        dim, index, src = rest
        offsets = selected_offsets(offsets_like(x), dim, index)
        if src.numel() != offsets.numel() or (
            x.ndim and src.ndim and src.shape != offsets.shape
        ):
            raise RuntimeError("index_add source shape mismatch")
        if src.dtype != x.dtype or src.device != x.device:
            raise RuntimeError(
                "index_add source must match destination dtype and device"
            )
        if kw.get("alpha", 1) != 1:
            src = pointwise("mul", src, kw["alpha"])
        return accumulate(x, offsets, bits(src).reshape_as(offsets).view(F64), "sum")
    if name in ("scatter", "scatter_add", "scatter_reduce"):
        dim, idx, src, *options = rest
        offsets = selected_offsets(offsets_like(x), dim, idx, gather=True)
        if torch.is_tensor(src):
            if src.ndim != idx.ndim or any(i > s for i, s in zip(idx.shape, src.shape)):
                raise RuntimeError("scatter source is smaller than the index")
            src = (
                bits(src)[tuple(slice(s) for s in idx.shape)].view(F64)
                if src.dtype == F64
                else src
            )
        else:
            src = torch.full((), scalar_bits(src), dtype=I64, device=x.device).view(F64)
        reduction = options[0] if options else kw.get("reduce", "sum")
        if name == "scatter":
            if "reduce" in overload:
                if reduction not in ("add", "multiply"):
                    raise RuntimeError("scatter reduce must be add or multiply")
                reduction = "sum" if reduction == "add" else "prod"
            else:
                reduction = "set"
        return accumulate(x, offsets, src, reduction, kw.get("include_self", True))
    src = rest[-1] if name in ("masked_scatter", "index_copy") else rest[0]
    if not torch.is_tensor(src) or src.dtype != F64 or (src.device != x.device):
        raise RuntimeError("scatter source must match destination dtype and device")
    # Forward all indexing metadata to the native int64 overload, preserving the payload bits.
    raw = [bits(v) if torch.is_tensor(v) and v.dtype == F64 else v for v in rest]
    return func(bits(x), *raw, **kw).view(F64)


def search_sorted(
    boundaries, query, *, side=None, right=False, out_int32=False, sorter=None
):
    from .backend import tensor64

    if (
        boundaries.ndim < 1
        or side not in (None, "left", "right")
        or (right and side == "left")
    ):
        raise RuntimeError("invalid searchsorted boundaries or side")
    if torch.is_tensor(query):
        if query.device != boundaries.device:
            raise RuntimeError("searchsorted operands must be on the same device")
        query = tensor64(query)
    else:
        query = torch.full(
            (), scalar_bits(query), device=boundaries.device, dtype=I64
        ).view(F64)
    if boundaries.ndim != 1 and (
        not query.ndim or boundaries.shape[:-1] != query.shape[:-1]
    ):
        raise RuntimeError("searchsorted batch dimensions must match")
    width = boundaries.shape[-1]
    if max(width, query.numel()) >= 2**32 or (out_int32 and width >= 2**31):
        raise NotImplementedError("searchsorted exceeds index range")
    raw = bits(tensor64(boundaries))
    if sorter is not None:
        if (
            sorter.dtype != I64
            or sorter.device != boundaries.device
            or sorter.shape != boundaries.shape
        ):
            raise RuntimeError("invalid searchsorted sorter shape, dtype or device")
        if sorter.numel() and ((sorter < 0) | (sorter >= width)).any().item():
            raise RuntimeError("sorter index is outside its row")
    if not width:
        return torch.zeros(
            query.shape, device=query.device, dtype=torch.int32 if out_int32 else I64
        )
    out = torch.empty(query.shape, device=query.device, dtype=I64)
    if out.numel():
        per_row = out.numel() if boundaries.ndim == 1 else query.shape[-1]
        library().search_sorted(
            out,
            raw.contiguous(),
            bits(query).contiguous(),
            raw if sorter is None else sorter.contiguous(),
            width,
            per_row,
            int(right or side == "right"),
            int(sorter is not None),
            threads=(out.numel(),),
        )
    return out.int() if out_int32 else out


def cumulative_extreme(x, dim, maximum):
    from .order import axis_check

    dim = axis_check(x, dim)
    if not x.ndim:
        return bits(x).clone().view(F64), torch.zeros((), device=x.device, dtype=I64)
    raw = bits(x).movedim(dim, -1).contiguous()
    if raw.numel() >= 2**32:
        raise NotImplementedError("cumulative extrema exceed Metal index range")
    if not raw.numel():
        return bits(x).clone().view(F64), torch.empty(
            x.shape, device=x.device, dtype=I64
        )

    def scan(data, source_ids=None):
        width = data.shape[-1]
        blocks = (width + 127) // 128
        rows = data.numel() // width
        out, ids = torch.empty_like(data), torch.empty_like(data)
        totals, tids = out, ids
        if blocks > 1:
            totals = torch.empty((rows, blocks), device=x.device, dtype=I64)
            tids = torch.empty_like(totals)
        lib = library()
        lib.cumulative_extreme_tile(
            out,
            ids,
            totals,
            tids,
            data,
            data if source_ids is None else source_ids,
            width,
            blocks,
            int(maximum),
            int(source_ids is not None),
            threads=(rows * blocks * 128,),
            group_size=(128,),
        )
        if blocks > 1:
            prior, prior_ids = scan(totals, tids)
            lib.cumulative_extreme_carry(
                out,
                ids,
                prior,
                prior_ids,
                width,
                blocks,
                int(maximum),
                threads=(data.numel(),),
            )
        return out, ids

    values, ids = scan(raw)
    return values.movedim(-1, dim).contiguous().view(F64), ids.movedim(
        -1, dim
    ).contiguous()


def select_rank(x, dim=None, keepdim=False, *, k=None, ignore_nan=False):
    from .order import sort, axis_check

    if dim is None:
        x = bits(x).reshape(-1).view(F64)
        dim = 0
    dim = axis_check(x, dim)
    width = x.shape[dim] if x.ndim else 1
    if k is not None and not 1 <= k <= width:
        raise RuntimeError("kthvalue k is outside the selected dimension")
    if not width:
        raise IndexError("empty order statistic dimension")
    if not x.ndim:
        return bits(x).clone().view(F64), torch.zeros((), device=x.device, dtype=I64)
    raw = bits(x).movedim(dim, -1).contiguous()
    vals, order = sort(raw.view(F64), -1)
    out = torch.empty(raw.shape[:-1], device=x.device, dtype=I64)
    ids = torch.empty_like(out)
    if out.numel():
        library().select_rank(
            out,
            ids,
            bits(vals),
            order,
            width,
            -1 if k is None else k - 1,
            int(ignore_nan),
            threads=(out.numel(),),
        )
    if keepdim:
        out, ids = out.unsqueeze(dim), ids.unsqueeze(dim)
    return out.view(F64), ids


def unique(x, ordered, return_inverse=False, return_counts=False, dim=None):
    from .order import sort, axis_check

    if dim is None:
        data = bits(x).reshape(-1, 1).contiguous()
        feature_shape = ()
    else:
        if not x.ndim:
            raise IndexError("unique cannot select a dimension of a scalar")
        dim = axis_check(x, dim)
        front = bits(x).movedim(dim, 0)
        feature_shape = front.shape[1:]
        if any(not size for size in feature_shape):
            raise RuntimeError(
                "unique cannot compare slices with an empty feature dimension"
            )
        data = front.reshape(front.shape[0], math.prod(feature_shape)).contiguous()
    rows, width = data.shape
    if rows >= 2**30 or data.numel() >= 2**32:
        raise NotImplementedError("unique grid exceeds supported row or element range")
    empty = torch.empty(0, dtype=I64, device=x.device)
    if not rows:
        return (
            data.reshape(0, *feature_shape).movedim(0, dim or 0).view(F64),
            empty.reshape(x.shape) if dim is None and return_inverse else empty,
            empty,
        )
    lib = library()
    permute_values = ordered and dim is not None
    ids = data
    if ordered and not permute_values:
        values, ids = sort(data.reshape(-1).view(F64))
        data = bits(values).reshape(rows, 1)
    elif permute_values:
        ids = torch.arange(rows, device=x.device, dtype=I64)
        spare = torch.empty_like(ids)
        run = 1
        while run < rows:
            lib.unique_merge_rows(spare, ids, data, rows, width, run, threads=(rows,))
            ids, spare = spare, ids
            run *= 2
    mask = torch.empty(rows, dtype=torch.bool, device=x.device)
    lib.unique_starts(mask, data, ids, width, int(permute_values), threads=(rows,))
    starts = mask.nonzero().reshape(-1)
    count = starts.numel()
    values = torch.empty((count, *feature_shape), device=x.device, dtype=I64)
    lib.unique_values(
        values, data, starts, ids, width, int(permute_values), threads=(values.numel(),)
    )
    inverse, counts = empty, empty
    if return_inverse:
        run_ids = mask.to(I64).cumsum(0) - 1
        if ordered:
            inverse = torch.empty_like(run_ids)
            lib.unique_inverse(inverse, ids, run_ids, threads=(rows,))
        else:
            inverse = run_ids
        if dim is None:
            inverse = inverse.reshape(x.shape)
    if return_counts:
        counts = torch.empty_like(starts)
        lib.unique_counts(counts, starts, count, rows, threads=(count,))
    return values.movedim(0, dim or 0).view(F64), inverse, counts


def logcumsumexp(x, dim):
    from .order import axis_check

    dim = axis_check(x, dim)
    if not x.ndim or not x.numel():
        return bits(x).clone().view(F64)
    data = bits(x).movedim(dim, -1).contiguous()
    if data.numel() >= 2**32:
        raise NotImplementedError("logcumsumexp exceeds Metal index range")

    def scan(rows):
        width = rows.shape[-1]
        blocks = (width + 15) // 16
        nrows = rows.numel() // width
        out = torch.empty_like(rows)
        totals = (
            torch.empty((nrows, blocks), dtype=I64, device=x.device)
            if blocks > 1
            else out
        )
        lib = library(True)
        lib.log_prefix(out, totals, rows, width, blocks, threads=(nrows * blocks,))
        if blocks > 1:
            prefix = scan(totals)
            lib.log_prefix_carry(out, prefix, width, blocks, threads=(rows.numel(),))
        return out

    return scan(data).movedim(-1, dim).contiguous().view(F64)
