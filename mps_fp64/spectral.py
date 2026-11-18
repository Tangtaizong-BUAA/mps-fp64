"""GPU FFT over real/imaginary binary64 pairs. Radix-2 and Bluestein FFTs."""

from functools import lru_cache
import math
import torch
from .kernels import F64, I64, bits, library, pointwise, scalar_bits


def pair(z):
    if z.is_conj():
        raw = bits(torch.view_as_real(z.conj())).clone()
        raw[..., 1].bitwise_xor_(-(1 << 63))
        return raw
    return bits(torch.view_as_real(z))


def complex_tensor(raw):
    return torch.view_as_complex(raw.view(F64))


def _roots(width, forward, device):
    phase = torch.empty(width, dtype=I64, device=device)
    lib = library()
    lib.fft_angles(phase, width, scalar_bits(math.pi / (2 * width)), threads=(width,))
    c = bits(pointwise("cos", phase.view(F64)))
    s = bits(pointwise("sin", phase.view(F64)))
    roots = torch.empty((width, 2), dtype=I64, device=device)
    lib.fft_roots(roots, c, s, width, int(forward), threads=(width,))
    return roots


_cached_roots = lru_cache(maxsize=16)(_roots)


def roots_for(width, forward, device):
    # 16 * 32768 * 16 bytes maximum in this cache. No user tensor is retained.
    factory = _cached_roots if width <= 32768 else _roots
    return factory(width, bool(forward), device)


def radix2(raw, roots):
    width = raw.shape[-2]
    count = raw.numel() // 2
    out = torch.empty_like(raw)
    stages = width.bit_length() - 1
    if width <= 1024:
        lanes = min(width // 2, 128)
        library().fft_local(
            out,
            raw,
            roots,
            width,
            stages,
            threads=(count // width * lanes,),
            group_size=(lanes,),
        )
    else:
        library().bit_reverse(out, raw, width, stages, threads=(count,))
        for stage in range(1, stages + 1):
            library().butterflies(out, roots, width, 1 << stage, threads=(count // 2,))
    return out


def _chirp_plan(width, length, forward, device):
    roots = roots_for(2 * width, forward, device)
    chirp = torch.empty((width, 2), device=device, dtype=I64)
    mirror = torch.empty((length, 2), device=device, dtype=I64)
    library().fft_chirp_plan(chirp, mirror, roots, width, length, threads=(length,))
    transformed = radix2(mirror, roots_for(length, True, device))
    # The convolution's inverse FFT is unnormalized. 1/length is an exact binary power.
    spectrum = bits(pointwise("mul", transformed.view(F64), 1.0 / length))
    return chirp, spectrum, roots


_cached_plan = lru_cache(maxsize=8)(_chirp_plan)


def axis_fft(raw, axis, forward):
    raw = raw.movedim(axis, -2).contiguous()
    width = raw.shape[-2]
    count = raw.numel() // 2
    if width <= 1 or not count:
        return raw.clone().movedim(-2, axis)
    if count > 2**32 - 1:
        raise NotImplementedError("FFT grid exceeds uint32")
    device = raw.device
    if width & (width - 1) == 0:
        out = radix2(raw, roots_for(width, forward, device))
    elif width <= 64:
        out = torch.empty_like(raw)
        library().simple_dft(
            out, raw, roots_for(width, forward, device), width, threads=(count,)
        )
    else:
        length = 1 << (2 * width - 2).bit_length()
        if length > 2**31:
            raise NotImplementedError(
                "Bluestein convolution exceeds the Metal index range"
            )
        plan = _cached_plan if length <= 32768 else _chirp_plan
        chirp, spectrum, roots = plan(width, length, bool(forward), device)
        positive = roots_for(length, False, device)
        negative = roots_for(length, True, device)
        rows = raw.reshape(-1, width, 2)
        out = torch.empty_like(raw)
        target = out.reshape_as(rows)
        # Bound one padded working buffer to 16 MiB (or one full transform).
        per_batch = max(1, (1 << 20) // length)
        for start in range(0, len(rows), per_batch):
            source = rows[start : start + per_batch]
            size = len(source)
            padded = torch.empty((size, length, 2), dtype=I64, device=device)
            extreme = torch.zeros(size, dtype=torch.int32, device=device)
            library().fft_modulate(
                padded, source, chirp, extreme, width, length, threads=(size * length,)
            )
            transformed = radix2(padded, negative)
            library().fft_multiply(
                transformed, spectrum, length, threads=(size * length,)
            )
            convolved = radix2(transformed, positive)
            library().fft_demodulate(
                target[start : start + size],
                convolved,
                source,
                chirp,
                roots,
                extreme,
                width,
                length,
                threads=(size * width,),
            )
    return out.movedim(-2, axis)


def transform(name, x, dims, normalization, extra):
    if name == "_fft_r2c":
        raw = torch.stack((bits(x), torch.zeros_like(bits(x))), -1)
    else:
        raw = pair(x)
    rank = raw.ndim - 1
    if (
        not dims
        or any(not -rank <= d < rank for d in dims)
        or len(set(d % rank for d in dims)) != len(dims)
    ):
        raise RuntimeError("invalid FFT dimensions")
    if normalization not in (0, 1, 2):
        raise RuntimeError("invalid FFT normalization")
    axes = tuple(d % rank for d in dims)
    if any(raw.shape[d] == 0 for d in axes):
        raise RuntimeError("FFT requires positive transform lengths")
    total = 1
    for axis in axes:
        if name == "_fft_c2r" and axis == axes[-1]:
            size = extra
            if size // 2 + 1 != raw.shape[axis]:
                raise RuntimeError("invalid Hermitian input length")
            tail = raw.narrow(axis, 1, (size - 1) // 2).flip((axis,)).clone()
            tail[..., 1].bitwise_xor_(-(1 << 63))
            raw = torch.cat((raw, tail), dim=axis)
        total *= raw.shape[axis]
        raw = axis_fft(raw, axis, name == "_fft_r2c" or (name == "_fft_c2c" and extra))
    if normalization:
        divisor = math.sqrt(total) if normalization == 1 else total
        raw = bits(pointwise("div", raw.view(F64), divisor))
    if name == "_fft_c2r":
        return raw[..., 0].contiguous().view(F64)
    if name == "_fft_r2c" and extra:
        axis = axes[-1]
        raw = raw.narrow(axis, 0, raw.shape[axis] // 2 + 1)
    return complex_tensor(raw)


def public_fft(name, x, size=None, dim=None, norm=None):
    """The public ATen composite can reach the mode directly under inference_mode."""
    op = name.removeprefix("fft_")
    multi = op.endswith(("2", "n"))
    kind = op[:-1] if multi else op
    if kind not in {"fft", "ifft", "rfft", "irfft", "hfft", "ihfft"}:
        raise NotImplementedError(name)
    if norm not in (None, "backward", "ortho", "forward"):
        raise RuntimeError("invalid FFT norm")
    if not multi and size is not None and size <= 0:
        raise RuntimeError("FFT requires a positive transform length")
    if x.ndim == 0:
        raise IndexError("FFT dimension out of range")
    shape = None if size is None else (tuple(size) if multi else (size,))
    if dim is None:
        if not multi:
            axes = (-1,)
        elif op.endswith("2"):
            axes = (-2, -1)
        else:
            axes = (
                tuple(range(x.ndim)) if shape is None else tuple(range(-len(shape), 0))
            )
    else:
        axes = tuple(dim) if multi else (dim,)
    if any(a < -x.ndim or a >= x.ndim for a in axes):
        raise IndexError("FFT dimension out of range")
    axes = tuple(d % x.ndim for d in axes)
    if len(set(axes)) != len(axes) or (shape is not None and len(shape) != len(axes)):
        raise RuntimeError("invalid FFT dimensions or transform lengths")
    to_real = kind in ("irfft", "hfft")
    from_real = kind in ("rfft", "ihfft")
    if from_real and x.is_complex():
        raise RuntimeError("this transform requires a real input")
    if not axes and (to_real or from_real):
        raise RuntimeError("real transform needs a dimension")
    conjugated = kind in ("hfft", "ihfft")
    forward = kind in ("fft", "rfft", "hfft")
    scaling = 1 if norm == "ortho" else (2 if (norm == "forward") == forward else 0)
    raw = pair(x) if x.is_complex() else bits(x)
    if shape is None:
        shape = (-1,) * len(axes)
    length = None
    for pos, (axis, required) in enumerate(zip(axes, shape)):
        current = raw.shape[axis]
        hermitian_axis = to_real and pos == len(axes) - 1
        if required == -1:
            required = 2 * (current - 1) if hermitian_axis else current
        if required < 1:
            raise RuntimeError("FFT requires positive transform lengths")
        length = required
        need = required // 2 + 1 if hermitian_axis else required
        if need < current:
            raw = raw.narrow(axis, 0, need)
        elif need > current:
            padded_shape = list(raw.shape)
            padded_shape[axis] = need
            padded = torch.zeros(padded_shape, device=x.device, dtype=I64)
            padded.narrow(axis, 0, current).copy_(raw)
            raw = padded
    if not x.is_complex() and (to_real or kind in ("fft", "ifft")):
        raw = torch.stack((raw, torch.zeros_like(raw)), -1)
    if not axes:
        return complex_tensor(raw.clone())
    if to_real:
        if conjugated:
            raw = raw.clone()
            raw[..., 1].bitwise_xor_(-(1 << 63))
        return transform("_fft_c2r", complex_tensor(raw), axes, scaling, length)
    if from_real:
        z = transform("_fft_r2c", raw.view(F64), axes, scaling, True)
        return z.conj() if conjugated else z
    return transform("_fft_c2c", complex_tensor(raw), axes, scaling, forward)
