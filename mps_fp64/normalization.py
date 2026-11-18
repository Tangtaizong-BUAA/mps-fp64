import math
import torch
from .kernels import F64, I64, bits, library


def batch_norm(x, w, b, running_mean, running_var, training, momentum, eps):
    if x.ndim < 2:
        raise RuntimeError("batch normalization needs a channel dimension")
    channels = x.shape[1]
    for param in (w, b, running_mean, running_var):
        if param is not None and (
            param.shape != (channels,)
            or param.device != x.device
            or param.dtype != x.dtype
        ):
            raise RuntimeError(
                "batch normalization parameter shape, device or dtype mismatch"
            )
    dims = (0, *range(2, x.ndim))
    size = math.prod(x.shape[i] for i in dims)
    if training:
        var, mean = torch.var_mean(x, dims, correction=0)
        invstd = torch.rsqrt(var + eps)
        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + mean * momentum)
        if running_var is not None:
            unbiased = var * size / (size - 1)
            running_var.copy_(running_var * (1 - momentum) + unbiased * momentum)
        saved_mean, saved_inverse = mean, invstd
    else:
        if running_mean is None or running_var is None:
            raise RuntimeError("inference requires running mean and variance")
        mean = running_mean
        invstd = torch.rsqrt(running_var + eps)
        saved_mean, saved_inverse = (x.new_empty(0), x.new_empty(0))
    raw = torch.empty(x.shape, dtype=I64, device=x.device)
    if raw.numel() >= 2**32:
        raise NotImplementedError("batch normalization grid exceeds uint32")
    if raw.numel():
        m, inv = bits(mean).contiguous(), bits(invstd).contiguous()
        library().batch_norm_affine(
            raw,
            bits(x).contiguous(),
            m,
            inv,
            m if w is None else bits(w).contiguous(),
            m if b is None else bits(b).contiguous(),
            math.prod(x.shape[2:]),
            channels,
            int(w is not None) | (int(b is not None) << 1),
            threads=(raw.numel(),),
        )
    result = raw.view(F64)
    return result, saved_mean, saved_inverse


def batch_norm_backward(
    dy, x, w, running_mean, running_var, mean, invstd, training, eps, mask
):
    dims = (0, *range(2, x.ndim))
    size = math.prod(x.shape[i] for i in dims)
    shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
    if not training:
        mean = running_mean
        invstd = torch.rsqrt(running_var + eps)
    normalized = (
        (x - mean.view(shape)) * invstd.view(shape)
        if mask[1] or (training and mask[0])
        else None
    )
    db = dy.sum(dims) if mask[2] or (training and mask[0]) else None
    dw = (dy * normalized).sum(dims) if normalized is not None else None
    dx = None
    if mask[0]:
        dx = dy
        if training:
            dx = dx - db.view(shape) / size - normalized * (dw.view(shape) / size)
        dx = dx * invstd.view(shape)
        if w is not None:
            dx = dx * w.view(shape)
    return dx, dw if mask[1] else None, db if mask[2] else None


def group_norm(x, weight, bias, n, c, spatial, groups, eps):
    if (
        x.ndim < 2
        or x.shape[:2] != (n, c)
        or math.prod(x.shape[2:]) != spatial
        or groups < 1
        or c % groups
    ):
        raise RuntimeError("invalid group normalization geometry")
    if any(
        p is not None
        and (p.dtype != x.dtype or p.device != x.device or p.shape != (c,))
        for p in (weight, bias)
    ):
        raise RuntimeError(
            "group normalization parameters must match input channels, dtype and device"
        )
    width = c // groups * spatial
    row = x.reshape(n, groups, width)
    var, mean = torch.var_mean(row, -1, correction=0)
    rstd = torch.rsqrt(var + eps)
    y = ((row - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)).reshape(x.shape)
    shape = (1, c, *([1] * (x.ndim - 2)))
    if weight is not None:
        y = y * weight.view(shape)
    if bias is not None:
        y = y + bias.view(shape)
    return y, mean, rstd


def group_grad(dy, x, mean, rstd, weight, n, c, spatial, groups, output_mask):
    width = c // groups * spatial
    z = (x.reshape(n, groups, width) - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
    dims = (0, *range(2, x.ndim))
    dx = None
    if output_mask[0]:
        channel_shape = (1, c, *([1] * (x.ndim - 2)))
        scaled = dy if weight is None else dy * weight.reshape(channel_shape)
        v = scaled.reshape(n, groups, width)
        dx = (
            (v - v.mean(-1, keepdim=True) - z * (v * z).mean(-1, keepdim=True))
            * rstd.unsqueeze(-1)
        ).reshape(x.shape)
    dw = (dy * z.reshape(x.shape)).sum(dims) if output_mask[1] else None
    db = dy.sum(dims) if output_mask[2] else None
    return dx, dw, db
