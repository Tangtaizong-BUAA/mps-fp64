from functools import lru_cache
import math

import torch
from torch._prims_common import suggest_memory_format

from .kernels import F64, I64, bits, library


def _signature(value):
    if torch.is_tensor(value):
        return (value.dtype, tuple(value.shape), value.stride(), value.device)
    return tuple(value) if isinstance(value, list) else value


def _meta(value):
    if isinstance(value, tuple) and value and isinstance(value[0], torch.dtype):
        dtype, shape, stride, _ = value
        return torch.empty_strided(shape, stride, dtype=dtype, device="meta")
    return value


@lru_cache(maxsize=256)
def _pool_plan(func, arguments, keywords):
    # Only immutable descriptors enter the cache. Meta validates every new
    # geometry, including shape, strides, scalar options and gradient inputs.
    name = func._schema.name
    rank = 3 if "3d" in name else 2
    adaptive = "adaptive" in name
    maximum = "max_pool" in name
    backward = name.endswith("_backward")
    args = tuple(map(_meta, arguments))
    kw = {key: _meta(value) for key, value in keywords}
    params = {field.name: field.default_value for field in func._schema.arguments}
    params.update((field.name, arg) for field, arg in zip(func._schema.arguments, args))
    params.update(kw)
    meta = func(*args, **kw)
    x = args[1] if backward else args[0]
    shape = args[0].shape if backward else (meta[0].shape if maximum else meta.shape)
    if backward:
        devices = {
            arg[3]
            for arg in (*arguments, *(v for _, v in keywords))
            if isinstance(arg, tuple) and arg and isinstance(arg[0], torch.dtype)
        }
        if (
            len(devices) != 1
            or x.shape[:-rank] != shape[:-rank]
            or x.dtype != args[0].dtype
        ):
            raise RuntimeError("pooling gradient shape, dtype or device mismatch")
        if maximum and (
            params["indices"].shape != shape or params["indices"].dtype != I64
        ):
            raise RuntimeError("pooling indices must match output shape on MPS")

    def triple(values, fill):
        if isinstance(values, int):
            values = [values] * rank
        if len(values) == 1:
            values = list(values) * rank
        return [fill] * (3 - rank) + list(values)

    window = params.get("kernel_size", [1] * rank)
    fields = [
        *triple(x.shape[-rank:], 1),
        *triple(shape[-rank:], 1),
        *triple(window, 1),
        *triple(params.get("stride") or window, 1),
        *triple(params.get("padding", [0] * rank), 0),
        *triple(params.get("dilation", [1] * rank), 1),
        adaptive,
        maximum,
        params.get("count_include_pad", False),
        params.get("divisor_override") or 0,
    ]
    if (
        max(fields[:-1]) >= 2**30
        or min(fields[:-1]) < 0
        or not -(2**31) <= fields[-1] < 2**31
        or max(x.numel(), shape.numel()) >= 2**31
    ):
        raise NotImplementedError("pool geometry outside int32 kernel domain")
    # Native regular MPS pooling preserves channels-last in both dimensions.
    # Three-dimensional adaptive kernels follow their PyTorch meta layout.
    formatted = meta[0] if maximum and not backward else meta
    memory_format = suggest_memory_format(formatted if rank == 3 and adaptive else x)
    square = (
        rank == 2
        and not adaptive
        and fields[6:12] == [1, 2, 2, 1, 2, 2]
        and fields[12:18] == [0, 0, 0, 1, 1, 1]
        and x.shape[-2] == 2 * shape[-2]
        and x.shape[-1] == 2 * shape[-1]
    )
    return backward, maximum, shape, tuple(map(int, fields)), memory_format, square


def pool(func, args, kw):
    backward, maximum, shape, geometry, memory_format, square = _pool_plan(
        func,
        tuple(map(_signature, args)),
        tuple((k, _signature(v)) for k, v in kw.items()),
    )
    x = args[1] if backward else args[0]
    if backward:
        dx = torch.empty(x.shape, device=x.device, dtype=I64)
        if dx.numel():
            ids = kw.get("indices", args[-1]).contiguous() if maximum else dx
            lib = library()
            kernel = lib.pool_square_backward if square else lib.pool_backward
            count = args[0].numel() if square else dx.numel()
            kernel(
                dx,
                bits(args[0]).contiguous(),
                ids,
                geometry,
                threads=(count,),
                arg_casts={3: "int32"},
            )
        return dx.contiguous(memory_format=memory_format).view(F64)
    out = torch.empty(shape, device=x.device, dtype=I64)
    ids = torch.empty_like(out) if maximum else out
    if out.numel():
        # Bind the geometry as inline Metal constants on the existing stream.
        lib = library()
        kernel = lib.pool_square_forward if square else lib.pool_forward
        kernel(
            out,
            ids,
            bits(x).contiguous(),
            geometry,
            threads=(out.numel(),),
            arg_casts={3: "int32"},
        )
    out = out.contiguous(memory_format=memory_format).view(F64)
    return (out, ids.contiguous(memory_format=memory_format)) if maximum else out


def unpool(func, args, kw):
    x, idx, spatial = args[:3]
    rank = len(spatial)
    if (
        rank not in (2, 3)
        or x.shape != idx.shape
        or idx.dtype != I64
        or x.device != idx.device
    ):
        raise RuntimeError("invalid max-unpool inputs")
    meta = func(
        torch.empty(x.shape, device="meta", dtype=F64),
        torch.empty(idx.shape, device="meta", dtype=I64),
        *args[2:],
        **kw,
    )
    if max(x.numel(), meta.numel()) >= 2**31:
        raise NotImplementedError("unpool grid exceeds int32")
    width = math.prod(spatial)
    if not idx.numel():
        return torch.zeros(meta.shape, device=x.device, dtype=I64).view(F64)
    if not width:
        raise RuntimeError("unpool index out of range")
    raw = torch.empty(meta.shape, device=x.device, dtype=I64)
    # A zero owner means no source in the plane addressed this output. With
    # duplicate indices, atomic max selects the last input in row-major order.
    owner = torch.zeros(meta.shape, device=x.device, dtype=torch.int32)
    error = torch.zeros((), device=x.device, dtype=torch.int32)
    plane = math.prod(x.shape[-rank:])
    lib = library()
    lib.unpool_owners(
        owner, error, idx.contiguous(), plane, width, threads=(idx.numel(),)
    )
    if error.item():
        raise RuntimeError("unpool index out of range")
    lib.unpool_gather(
        raw, bits(x).contiguous(), owner, plane, width, threads=(raw.numel(),)
    )
    return raw.view(F64)
