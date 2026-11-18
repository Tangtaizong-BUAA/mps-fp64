from functools import lru_cache
import math
import torch
from .kernels import bits, F64, I64, library, mm, sum_dims


def geometry(x, w, shape, stride, padding, dilation, transposed, groups, bias):
    rank = x.ndim - 2
    if rank not in (1, 2, 3):
        raise NotImplementedError("convolution spatial rank")

    def axes(values, fill):
        values = list(values)
        if len(values) == 1:
            values *= rank
        if len(values) != rank:
            raise RuntimeError("convolution parameter length differs from spatial rank")
        return [fill] * (3 - rank) + values

    fields = [
        x.shape[0],
        x.shape[1],
        shape[1],
        groups,
        *axes(x.shape[2:], 1),
        *axes(shape[2:], 1),
        *axes(w.shape[2:], 1),
        *axes(stride, 1),
        *axes(padding, 0),
        *axes(dilation, 1),
        int(transposed),
        int(bias),
    ]
    if max(fields) >= 2**30 or max(x.numel(), w.numel()) >= 2**31:
        raise NotImplementedError("convolution dimensions exceed supported grid")
    return fields


def forward(x, w, b, stride, padding, dilation, transposed, groups, shape):
    count = math.prod(shape)
    if max(count, x.numel(), w.numel()) >= 2**31:
        raise NotImplementedError("convolution exceeds supported grid")
    spatial = math.prod(x.shape[2:])
    ci, co = x.shape[1] // groups, shape[1] // groups
    if (
        count
        and all(k == 1 for k in w.shape[2:])
        and all(s == 1 for s in stride)
        and not any(padding)
        and tuple(shape[2:]) == tuple(x.shape[2:])
        and min(ci, co, spatial) >= 8
        and count * ci >= 32768
    ):
        # Each group is a channel matrix times a matrix of spatial columns.
        bsz = x.shape[0]
        weights = (
            bits(w).reshape(groups, ci, co).mT
            if transposed
            else bits(w).reshape(groups, co, ci)
        )
        weights = weights.expand(bsz, groups, co, ci).reshape(bsz * groups, co, ci)
        images = bits(x).reshape(bsz * groups, ci, spatial)
        product = mm(weights.view(F64), images.view(F64), True)
        out = bits(product).reshape(shape)
        if b is not None:
            library().conv_bias(
                out, bits(b).contiguous(), spatial, shape[1], threads=(count,)
            )
        return out.view(F64)
    out = torch.empty(shape, dtype=I64, device=x.device)
    if out.numel():
        params = geometry(
            x, w, shape, stride, padding, dilation, transposed, groups, b is not None
        )
        bias = bits(b).contiguous() if b is not None else out
        library().conv_forward(
            out,
            bits(x).contiguous(),
            bits(w).contiguous(),
            bias,
            params,
            arg_casts={4: "int32"},
            threads=(out.numel(),),
        )
    return out.view(F64)


def output_shape(x, w, b, stride, padding, dilation, transposed, out_padding, groups):
    return _output_shape(
        tuple(x.shape),
        tuple(w.shape),
        None if b is None else tuple(b.shape),
        x.dtype,
        tuple(stride),
        tuple(padding),
        tuple(dilation),
        transposed,
        tuple(out_padding),
        groups,
    )


@lru_cache(maxsize=128)
def _output_shape(
    x, w, b, dtype, stride, padding, dilation, transposed, out_padding, groups
):
    def meta(shape):
        return None if shape is None else torch.empty(shape, device="meta", dtype=dtype)

    return torch.ops.aten.convolution.default(
        meta(x),
        meta(w),
        meta(b),
        stride,
        padding,
        dilation,
        transposed,
        out_padding,
        groups,
    ).shape


def convolution(x, w, b, stride, padding, dilation, transposed, out_padding, groups):
    tensors = [t for t in (x, w, b) if t is not None]
    if any(t.dtype != F64 or t.device != x.device for t in tensors):
        raise RuntimeError(
            "convolution input, weight and bias must match in dtype and device"
        )

    shape = output_shape(
        x, w, b, stride, padding, dilation, transposed, out_padding, groups
    )
    return forward(x, w, b, stride, padding, dilation, transposed, groups, shape)


def backward(
    dy,
    x,
    w,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    out_padding,
    groups,
    mask,
):
    if any(t.dtype != F64 or t.device != x.device for t in (x, w, dy)):
        raise RuntimeError("convolution gradient dtype or device mismatch")
    expected = output_shape(
        x, w, None, stride, padding, dilation, transposed, out_padding, groups
    )
    if dy.shape != expected:
        raise RuntimeError("convolution gradient shape does not match forward output")
    dx = (
        forward(dy, w, None, stride, padding, dilation, not transposed, groups, x.shape)
        if mask[0]
        else None
    )
    dw = None
    if mask[1]:
        raw = torch.empty(w.shape, dtype=I64, device=w.device)
        if raw.numel():
            params = geometry(
                x, w, dy.shape, stride, padding, dilation, transposed, groups, False
            )
            library().conv_grad_weight(
                raw,
                bits(x).contiguous(),
                bits(dy).contiguous(),
                params,
                arg_casts={3: "int32"},
                threads=(raw.numel(),),
            )
        dw = raw.view(F64)
    db = sum_dims(dy, [0, *range(2, dy.ndim)], False) if mask[2] else None
    return dx, dw, db
