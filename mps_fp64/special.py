"""ATen special functions evaluated on resident binary64 encodings."""

from functools import lru_cache
from pathlib import Path
import warnings

import torch
from torch._prims_common import is_non_overlapping_and_dense_or_false

from .kernels import F64, I64, scalar_bits

_POLYNOMIALS = (
    *(f"special_chebyshev_polynomial_{k}" for k in "tuvw"),
    *(f"special_shifted_chebyshev_polynomial_{k}" for k in "tuvw"),
    "special_hermite_polynomial_h",
    "special_hermite_polynomial_he",
    "special_laguerre_polynomial_l",
    "special_legendre_polynomial_p",
)
_BESSEL = (
    "special_bessel_j0",
    "special_bessel_j1",
    "special_bessel_y0",
    "special_bessel_y1",
    "special_modified_bessel_i0",
    "special_modified_bessel_i1",
    "special_modified_bessel_k0",
    "special_modified_bessel_k1",
    "special_scaled_modified_bessel_k0",
    "special_scaled_modified_bessel_k1",
    "special_spherical_bessel_j0",
    "special_i0e",
    "special_i1",
    "special_i1e",
)
_BESSEL_IDS = {name: i for i, name in enumerate(_BESSEL)} | {"i0": 4, "special_i0": 4}
_GAMMA_IDS = {
    "lgamma": 0,
    "special_gammaln": 0,
    "digamma": 1,
    "special_digamma": 1,
    "special_psi": 1,
    "erfinv": 2,
    "special_erfinv": 2,
    "special_erfcx": 3,
    "polygamma": 4,
    "special_polygamma": 4,
    "mvlgamma": 5,
    "special_multigammaln": 5,
}
_INCOMPLETE_GAMMA = {
    "igamma": 0,
    "special_gammainc": 0,
    "igammac": 1,
    "special_gammaincc": 1,
}
SPECIAL_NAMES = frozenset(_POLYNOMIALS).union(
    _BESSEL_IDS, _GAMMA_IDS, _INCOMPLETE_GAMMA, {"special_zeta"}
)


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).parent
    files = (
        "vendor/softfloat64.metal",
        "soft_math.metal",
        "vendor/openlibm.metal",
        "pointwise.metal",
        "special.metal",
        "special_gamma.metal",
        "special_igamma.metal",
    )
    return torch.mps.compile_shader(
        "\n".join((root / file).read_text() for file in files)
    )


def _spec(arg):
    return (
        (arg.dtype, tuple(arg.shape), arg.stride())
        if torch.is_tensor(arg)
        else type(arg)
    )


@lru_cache(maxsize=512)
def _layout(func, specs):
    # ATen determines promotion, broadcasting and the output memory format.
    # Only shape/dtype/stride tuples enter the cache, never user storage.
    probes = []
    for spec in specs:
        if isinstance(spec, tuple):
            dtype, shape, strides = spec
            probes.append(
                torch.empty_strided(shape, strides, dtype=dtype, device="meta")
            )
        else:
            probes.append(spec(0) if isinstance(spec, type) else spec)
    out = func(*probes)
    return out.shape, out.stride(), out.dtype


@lru_cache(maxsize=512)
def _offsets(shape, output_stride, specs):
    steps = []
    for sizes, strides in specs:
        steps.append(
            [0] * (len(shape) - len(sizes))
            + [s if n != 1 else 0 for n, s in zip(sizes, strides)]
        )
    if len(steps) == 1:
        steps.append([0] * len(shape))
    axes = []
    for dim in sorted(range(len(shape)), key=lambda d: output_stride[d]):
        size, sx, sn = shape[dim], steps[0][dim], steps[1][dim]
        if size == 1:
            continue
        if axes and sx == axes[-1][0] * axes[-1][1] and sn == axes[-1][0] * axes[-1][2]:
            axes[-1][0] *= size
        else:
            axes.append([size, sx, sn])
    return (len(axes), *(v for ax in axes for v in ax))


def _check_out(out, inputs, dtype):
    if out.device.type != "mps" or not torch.can_cast(dtype, out.dtype):
        raise RuntimeError(f"expected MPS out accepting {dtype}")
    if torch._debug_has_internal_overlap(out) == 1:
        raise RuntimeError("multiple output elements refer to the same memory location")
    if not out.numel() or not is_non_overlapping_and_dense_or_false(out):
        return
    for x in inputs:
        if (
            not torch.is_tensor(x)
            or not x.numel()
            or x.device != out.device
            or not is_non_overlapping_and_dense_or_false(x)
        ):
            continue
        if not torch._C._overlaps(out, x):
            continue
        start = out.storage_offset() * out.element_size()
        other = x.storage_offset() * x.element_size()
        end, other_end = (
            start + out.numel() * out.element_size(),
            other + x.numel() * x.element_size(),
        )
        if max(start, other) < min(end, other_end) and (start, end, out.stride()) != (
            other,
            other_end,
            x.stride(),
        ):
            raise RuntimeError("input and output partially overlap")


def special(func, *args, out=None):
    from .backend import gpu_cast, tensor64

    name = func._schema.name.removeprefix("aten::")
    if out is not None:
        base = func._schema.overload_name.removesuffix("_out")
        func = getattr(
            getattr(torch.ops.aten, name), "default" if base == "out" else base
        )
    specs = tuple(map(_spec, args))
    inputs = args
    gamma_kind = _GAMMA_IDS.get(name)
    order = 0
    if gamma_kind == 4:
        order, x = args
        if order < 0:
            raise RuntimeError("polygamma derivative order must be nonnegative")
        if order > 170:
            raise NotImplementedError(
                "polygamma supports derivative orders 0 through 170"
            )
        inputs, specs = (x,), (order, _spec(x))
    if gamma_kind == 5:
        x, order = args
        if order < 1:
            raise RuntimeError("multivariate gamma dimension must be positive")
        if order > 512:
            raise NotImplementedError("mvlgamma supports dimensions 1 through 512")
        inputs, specs = (x,), (_spec(x), order)
    shape, stride, dtype = _layout(func, specs)
    if out is not None:
        _check_out(out, args, dtype)
    if dtype != F64:
        # E.g. a zero-dimensional FP64 operand and a dimensional FP32 operand
        # promote to FP32. Evaluate at that dtype using the installed MPS kernel.
        native = [
            gpu_cast(x, dtype)
            if torch.is_tensor(x) and x.dtype == F64 and x.device.type == "mps"
            else x
            for x in args
        ]
        result = func(*native)
    else:
        values = tuple(
            tensor64(x) if torch.is_tensor(x) else scalar_bits(x) for x in inputs
        )
        # Casts can densify a noncontiguous operand. Index the actual promoted
        # storage in the physical order selected by the original TensorIterator.
        operands = tuple(
            (tuple(v.shape), v.stride()) if torch.is_tensor(v) else ((), ())
            for v in values
        )
        layout = _offsets(shape, stride, operands)
        raw = torch.empty_strided(shape, stride, dtype=I64, device="mps")
        nitems = raw.numel()
        if nitems >= 2**32:
            raise NotImplementedError("special-function grid exceeds uint32")
        if nitems and name in _BESSEL_IDS:
            _library().bessel_kernel(
                raw, values[0], layout, _BESSEL_IDS[name], threads=nitems
            )
        elif nitems and gamma_kind is not None:
            _library().gamma_kernel(
                raw, values[0], layout, gamma_kind, order, threads=nitems
            )
        elif nitems and name in _INCOMPLETE_GAMMA:
            status = torch.zeros((), dtype=torch.int32, device="mps")
            _library().incomplete_gamma_kernel(
                raw, *values, layout, _INCOMPLETE_GAMMA[name], status, threads=nitems
            )
            if status.item():
                raise NotImplementedError(
                    "incomplete gamma did not converge within 512 iterations"
                )
        elif nitems and name == "special_zeta":
            _library().zeta_kernel(raw, *values, layout, threads=nitems)
        elif nitems:
            kind = _POLYNOMIALS.index(name)
            n = args[1]
            needs_flag = kind not in (8, 9) and (torch.is_tensor(n) or n > 4096)
            flag = (
                torch.zeros((), dtype=torch.int32, device="mps") if needs_flag else raw
            )
            _library().polynomial_kernel(
                raw, *values, layout, kind, flag, threads=nitems
            )
            if needs_flag:
                status = flag.item()
                if status == 2:
                    raise NotImplementedError(
                        "Chebyshev phase above degree 4096 has unverified platform libm agreement"
                    )
                if status == 1:
                    raise NotImplementedError(
                        "a polynomial recurrence exceeded 4096 steps before reaching an absorbing NaN"
                    )
        result = raw.view(F64)
    if out is None:
        return result
    if out.shape != shape:
        if out.numel():
            warnings.warn(
                "resizing a non-empty special-function output",
                UserWarning,
                stacklevel=2,
            )
        out.resize_(shape)
    converted = gpu_cast(result, out.dtype) if result.dtype != out.dtype else result
    if out.dtype == F64:
        out.view(I64).copy_(converted.view(I64))
    else:
        out.copy_(converted)
    return out
