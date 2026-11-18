"""Use the MPS generator for entropy; assemble all 53 uniform mantissa bits."""

import math
import torch
from .kernels import I64, F64, library, pointwise


def uniform01(shape, device, generator):
    chunks = [
        torch.randint(0, 1 << k, shape, device=device, dtype=I64, generator=generator)
        for k in (27, 26)
    ]
    out = torch.empty(shape, dtype=I64, device=device)
    if out.numel() >= 2**32:
        raise NotImplementedError("uniform draw exceeds Metal index range")
    if out.numel():
        library().uniform_from_words(out, *chunks, threads=(out.numel(),))
    return out.view(F64)


def draw(kind, shape, device, generator=None):
    if kind == "uniform":
        return uniform01(shape, device, generator)
    u = pointwise("sub", 1.0, uniform01(shape, device, generator))
    angle = pointwise("mul", 2 * math.pi, uniform01(shape, device, generator))
    radius = pointwise("sqrt", pointwise("mul", -2.0, pointwise("log", u)))
    return pointwise("mul", radius, pointwise("cos", angle))
