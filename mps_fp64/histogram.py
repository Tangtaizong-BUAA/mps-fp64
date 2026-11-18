"""Binary64 histograms with source-ordered weights and integer atomic counts."""

from itertools import accumulate as prefix_sums
import math
import warnings

import torch

from .indexing import accumulate
from .kernels import F64, I64, bits, library, scalar_bits, sum_dims
from .order import sort


def _write(out, value):
    if out is None:
        return value
    if out.dtype != value.dtype or out.device != value.device:
        raise RuntimeError("output dtype and device must match")
    if out.shape != value.shape:
        if out.numel():
            warnings.warn("Resizing a nonempty histogram output tensor.", UserWarning, stacklevel=3)
        out.resize_(value.shape)
    if torch._debug_has_internal_overlap(out) == 1:
        raise RuntimeError("overlapping output elements")
    if value.dtype == F64:
        bits(out).copy_(bits(value))
    else:
        out.copy_(value)
    return out


def _bounds(points, ranges, legacy=False):
    samples, dims = points.shape
    if ranges is not None and len(ranges) != 2 * dims:
        raise RuntimeError("histogram range requires two endpoints per dimension")
    if ranges is None and samples:
        lower = sum_dims(points, (0,), False, 3)
        upper = sum_dims(points, (0,), False, 2)
        bounds = torch.stack((bits(lower), bits(upper)), dim=-1)
        if (((bounds >> 52) & 2047) == 2047).any().item():
            raise RuntimeError("autodetected histogram range is not finite")
    else:
        ranges = [0., 1.] * dims if ranges is None else ranges
        if not all(math.isfinite(x) for x in ranges):
            raise RuntimeError("histogram range is not finite")
        if any(lo > hi for lo, hi in zip(ranges[::2], ranges[1::2])):
            raise RuntimeError("histogram range minimum exceeds maximum")
        bounds = torch.tensor([scalar_bits(x) for x in ranges], dtype=I64,
                              device=points.device).reshape(dims, 2)
    corrected = torch.empty_like(bounds)
    if dims:
        library().histogram_expand(corrected, bounds, int(legacy), threads=(dims,))
    return corrected


def _edges(points, bins, ranges):
    dims = points.shape[-1]
    if isinstance(bins, int):
        bins = [bins] * dims
    if len(bins) != dims:
        raise RuntimeError("histogram needs one bin specification per coordinate")
    if bins and torch.is_tensor(bins[0]):
        if any(e.ndim != 1 or e.numel() < 2 or e.dtype != F64 or e.device != points.device for e in bins):
            raise RuntimeError("bin edges must be matching FP64 vectors with at least two values")
        return list(bins)
    if any(n < 1 for n in bins):
        raise RuntimeError("bin counts must be positive")
    if any(n >= 2**32 - 1 for n in bins):
        raise NotImplementedError("bin edge grid exceeds uint32")
    limits = _bounds(points, ranges)
    edges = []
    for d, n in enumerate(bins):
        e = torch.empty(n + 1, dtype=I64, device=points.device)
        library().linspace64(e, limits[d, 0], limits[d, 1], n + 1, threads=(n + 1,))
        edges.append(e.view(F64))
    return edges


def _values(points, edges, weight, density, legacy_bins=0):
    samples, dims = points.shape
    shape = (legacy_bins,) if legacy_bins else tuple(e.numel() - 1 for e in edges)
    bins = math.prod(shape)
    if max(bins, samples) >= 2**32 or sum(e.numel() for e in edges) >= 2**32:
        raise NotImplementedError("histogram index range exceeds uint32")
    if dims == 0:
        return torch.full((), scalar_bits(float("nan")) if density else 0,
                          device=points.device, dtype=I64).view(F64)
    offsets = [0, *prefix_sums(e.numel() for e in edges)]
    sequence = bits(edges[0]).contiguous() if dims == 1 else torch.cat([bits(e) for e in edges])
    raw = bits(points).contiguous()
    if weight is None:
        hist = torch.empty(shape, dtype=I64, device=points.device)
        if samples:
            counts = torch.zeros(bins, dtype=torch.int32, device=points.device)
            library().histogram_counts(counts, raw, sequence, offsets, dims, bins, legacy_bins,
                                       threads=(samples,))
            library().histogram_count_double(hist, counts, threads=(bins,))
        else:
            hist.zero_()
    else:
        # The extra bin receives out-of-range and NaN coordinates. A discarded
        # NaN weight can never contaminate an in-range bin.
        seed = torch.zeros(bins + 1, dtype=I64, device=points.device).view(F64)
        if samples:
            keys = torch.empty(samples, dtype=I64, device=points.device)
            library().histogram_keys(keys, raw, sequence, offsets, dims, bins, threads=(samples,))
            seed = accumulate(seed, keys, weight, "sum")
        hist = bits(seed)[:bins].reshape(shape)
    if density:
        total = sum_dims(hist.view(F64), None, False)
        library().histogram_density(hist, total, sequence, offsets, hist.stride(), dims, threads=(bins,))
    return hist.view(F64)


def _isin(elements, tests, *, assume_unique=False, invert=False, out=None):
    from .backend import tensor64
    for t in (elements, tests):
        if isinstance(t, (bool, complex)) or (torch.is_tensor(t) and (t.dtype == torch.bool or t.is_complex())):
            raise RuntimeError("isin does not support boolean or complex operands")
    device = elements.device if torch.is_tensor(elements) else tests.device
    def operand(value):
        if torch.is_tensor(value):
            return tensor64(value)
        return torch.full((), scalar_bits(value), dtype=I64, device=device).view(F64)
    x = operand(elements)
    candidates = bits(operand(tests)).reshape(-1).contiguous().view(F64)
    if max(x.numel(), candidates.numel()) >= 2**32:
        raise NotImplementedError("isin grid exceeds uint32")
    if candidates.numel() > 16:
        candidates = sort(candidates)[0]
    result = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    if result.numel():
        if candidates.numel():
            library().histogram_isin(result, bits(x).contiguous(), candidates, candidates.numel(),
                                     int(candidates.numel() > 16), int(invert), threads=(x.numel(),))
        else:
            result.fill_(invert)
    return _write(out, result)


def histogram_op(name, x, bins=100, *args, **kw):
    if name == "isin":
        return _isin(x, bins, **kw)
    if x.dtype != F64:
        raise RuntimeError("histogram input must be float64")
    if name == "histc":
        lo, hi = (list(args) + [0., 0.][len(args):])[:2]
        lo, hi = kw.get("min", lo), kw.get("max", hi)
        if bins < 1:
            raise RuntimeError("bin count must be positive")
        points = bits(x).reshape(-1, 1).view(F64)
        ranges = None if lo == hi and x.numel() else (lo, hi)
        extents = _bounds(points, ranges, legacy=True).reshape(-1).view(F64)
        return _write(kw.get("out"), _values(points, [extents], None, False, legacy_bins=bins))

    ranges = kw.get("range")
    weight = kw.get("weight")
    density = kw.get("density", False)
    # The public aten.histogramdd schema also accepts positional range,
    # weight and density. Its decomposition usually dispatches the three
    # private operators below with keyword-only options.
    if args:
        ranges, weight, density = (list(args) + [ranges, weight, density][len(args):])
    if name == "histogram":
        points = bits(x).reshape(-1, 1).view(F64)
        spec = [bins]
    else:
        if x.ndim < 2:
            raise RuntimeError("histogramdd input needs at least two dimensions")
        points = bits(x).reshape(math.prod(x.shape[:-1]), x.shape[-1]).view(F64)
        spec = bins
    edges = _edges(points, spec, ranges)
    if name == "_histogramdd_bin_edges":
        if "out" in kw:
            if len(kw["out"]) != len(edges):
                raise RuntimeError("incorrect number of output bin-edge tensors")
            for dest, e in zip(kw["out"], edges):
                _write(dest, e)
            return None
        return edges
    if weight is not None:
        shape_ok = (weight.numel() == x.numel()) if name == "histogram" else (
            tuple(weight.shape or (1,)) == tuple(x.shape[:-1]))
        if not shape_ok or weight.dtype != F64 or weight.device != x.device:
            raise RuntimeError("histogram weights must match sample shape, dtype and device")
        weight = bits(weight).reshape(-1).view(F64)
    hist = _values(points, edges, weight, density)
    if name == "histogram":
        # The one-dimensional public result owns its edge storage.
        e = bits(edges[0]).clone().view(F64) if torch.is_tensor(bins) else edges[0]
        return _write(kw.get("hist"), hist), _write(kw.get("bin_edges"), e)
    if name == "histogramdd":
        return hist, edges
    return _write(kw.get("out"), hist)
