"""Separable binary64 interpolation on GPU, including antialiased image filters."""

from functools import lru_cache
import math

import torch
from torch._prims_common import suggest_memory_format

from .kernels import F64, I64, bits, library, scalar_bits

_FORWARD = {
    "upsample_linear1d": 1,
    "upsample_bilinear2d": 2,
    "upsample_bicubic2d": 2,
    "upsample_trilinear3d": 3,
    "_upsample_bilinear2d_aa": 2,
    "_upsample_bicubic2d_aa": 2,
}
RESAMPLING = set(_FORWARD) | {name + "_backward" for name in _FORWARD}


@lru_cache(256)
def _layout(func, shape, stride, arguments):
    x = torch.empty_strided(shape, stride, dtype=F64, device="meta")
    meta = func(x, *arguments)
    return tuple(meta.shape), suggest_memory_format(meta)


def _weights(size, output, ratio, align, cubic, aa, duplicate, device):
    support = (2 if cubic else 1) * max(1.0, ratio) if aa else 0.0
    if not math.isfinite(support) or not math.isfinite(output * ratio):
        raise NotImplementedError(
            "interpolation scale exceeds binary64 coordinate range"
        )
    taps = min(size, 2 * math.ceil(support) + 1) if aa else (4 if cubic else 2)
    if output * taps >= 2**32:
        raise NotImplementedError("interpolation weight table exceeds uint32 grid")
    ranges = torch.empty((output, 2), dtype=torch.int32, device=device)
    weights = torch.empty((output, taps), dtype=I64, device=device)
    plan = (
        size,
        taps,
        cubic,
        aa,
        align,
        duplicate,
        scalar_bits(ratio),
        scalar_bits(support),
        scalar_bits(1.0 / max(1.0, ratio)),
    )
    library().resample_weights(ranges, weights, plan, threads=output)
    return ranges, weights


def resample(func, args, kw):
    fields = func._schema.arguments
    params = {field.name: field.default_value for field in fields}
    params.update((field.name, val) for field, val in zip(fields, args))
    params.update(kw)
    x = params[fields[0].name]
    name = func._schema.name.split("::")[-1]
    forward = name.removesuffix("_backward")
    rank = _FORWARD[forward]
    backward = name != forward
    options = tuple(
        tuple(v) if isinstance(v, list) else v
        for v in [params[f.name] for f in fields[1:]]
    )
    result_shape, fmt = _layout(func, tuple(x.shape), x.stride(), options)
    input_shape = result_shape if backward else tuple(x.shape)
    output_shape = tuple(x.shape) if backward else result_shape
    if backward and (
        output_shape[:-rank] != input_shape[:-rank]
        or output_shape[-rank:] != tuple(params["output_size"])
    ):
        raise RuntimeError("interpolation gradient shape does not match output size")
    if (
        max(input_shape + output_shape) >= 2**31
        or max(math.prod(input_shape), math.prod(output_shape)) >= 2**32
    ):
        raise NotImplementedError(
            "interpolation grid exceeds uint32 or a spatial extent exceeds int32"
        )
    if x.numel() == 0:
        return torch.empty(
            result_shape, dtype=I64, device=x.device, memory_format=fmt
        ).view(F64)
    scales = params.get("scale_factors")
    if scales is None:
        keys = (
            ("scales",) if rank == 1 else ("scales_d", "scales_h", "scales_w")[-rank:]
        )
        scales = [params.get(k) for k in keys]
    align = params["align_corners"]
    aa = forward.endswith("_aa")
    cubic = "bicubic" in forward
    data = bits(x).contiguous()
    # Native bicubic backward copies the whole gradient for an identity shape.
    if backward and cubic and not aa and input_shape == output_shape:
        return data.clone(memory_format=fmt).view(F64)
    kernel = (
        library().resample_axis_backward
        if backward
        else library().resample_axis_forward
    )
    axes = list(range(rank)) if backward else list(reversed(range(rank)))
    # A tall narrow image resized to a short wide image can otherwise need
    # billions of temporary elements with only thousands of inputs and outputs.
    # In that case contract all shrinking axes first. The axis maps commute.
    elements = data.numel()
    limit = min(2**32 - 1, 4 * max(elements, math.prod(result_shape)))
    for axis in axes:
        dim = axis + 2
        elements = elements // x.shape[dim] * result_shape[dim]
        if elements > limit:
            axes = sorted(axes, key=lambda d: result_shape[d + 2] > x.shape[d + 2])
            break
    allocated = False
    for axis in axes:
        dim = axis + 2
        size = input_shape[dim]
        output = output_shape[dim]
        # PyTorch's separable AA forward skips every unchanged axis. The native
        # backward recomputes its support even if an explicit scale preserves size.
        if aa and not backward and size == output:
            continue
        factor = scales[axis]
        if align:
            ratio = (size - 1) / (output - 1) if output > 1 else 0.0
        else:
            ratio = 1 / factor if factor is not None and factor > 0 else size / output
        duplicate = not cubic and not aa and size == output
        ranges, weights = _weights(
            size, output, ratio, align, cubic, aa, duplicate, x.device
        )
        shape = list(data.shape)
        shape[dim] = size if backward else output
        out = torch.empty(shape, device=x.device, dtype=I64)
        geom = (size, output, math.prod(shape[dim + 1 :]), weights.shape[1], duplicate)
        kernel(out, data, ranges, weights, geom, threads=out.numel())
        data = out
        allocated = True
    result = (
        data.contiguous(memory_format=fmt)
        if allocated
        else data.clone(memory_format=fmt)
    )
    return result.view(F64)
