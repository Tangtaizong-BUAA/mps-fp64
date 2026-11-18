from functools import lru_cache
from pathlib import Path
import math
import struct
import warnings

import torch

F64 = torch.float64
I64 = torch.int64
BINARY = {op: f"__softfloat64_f{op}(x,y,0)" for op in ("add", "sub", "mul", "div")}
BINARY.update(maximum="fp_max(x,y)", minimum="fp_min(x,y)")
BINARY.update(
    nextafter="fp_nextafter(x,y)",
    copysign="(x & ~(1ul<<63)) | (y & (1ul<<63))",
    fmax="fp_nan(x) ? y : fp_nan(y) ? x : fp_max(x,y)",
    fmin="fp_nan(x) ? y : fp_nan(y) ? x : fp_min(x,y)",
    tanh_backward="__softfloat64_fmul(x,__softfloat64_fsub(0x3ff0000000000000ul,__softfloat64_fmul(y,y,0),0),0)",
    sigmoid_backward="__softfloat64_fmul(__softfloat64_fmul(x,__softfloat64_fsub(0x3ff0000000000000ul,y,0),0),y,0)",
)
COMPARE = {op: f"__softfloat64_f{op}(x,y)" for op in ("eq", "lt", "le", "gt", "ge")}
COMPARE.update(
    logical_and="(x<<1)!=0 && (y<<1)!=0",
    logical_or="(x<<1)!=0 || (y<<1)!=0",
    logical_xor="((x<<1)!=0) != ((y<<1)!=0)",
)
COMPARE["ne"] = "!__softfloat64_feq(x,y)"
UNARY = {
    "round": "fp_round(x)",
    "logical_not": "(x<<1)==0",
    "sign": "(fp_nan(x) || (x<<1)==0) ? 0ul : ((x&(1ul<<63)) | 0x3ff0000000000000ul)",
    "rsqrt": "__softfloat64_fdiv(0x3ff0000000000000ul,__softfloat64_fsqrt(x,0),0)",
    "relu": "fp_max(x,0ul)",
    "neg": "x ^ (1ul << 63)",
    "abs": "x & ~(1ul << 63)",
    "sqrt": "__softfloat64_fsqrt(x,0)",
    "reciprocal": "__softfloat64_fdiv(0x3ff0000000000000ul,x,0)",
    "signbit": "bool(x >> 63)",
    "isnan": "(x << 1) > 0xffe0000000000000ul",
    "isinf": "(x << 1) == 0xffe0000000000000ul",
    "isfinite": "(x << 1) < 0xffe0000000000000ul",
}
PREDICATES = set(COMPARE) | {"signbit", "isnan", "isinf", "isfinite", "logical_not"}


MATH_UNARY = {}
for family, names in (
    ("e", ("exp", "log", "asin", "acos", "sinh", "cosh", "acosh", "atanh")),
    (
        "s",
        (
            "expm1",
            "log1p",
            "atan",
            "tanh",
            "sin",
            "cos",
            "tan",
            "floor",
            "ceil",
            "trunc",
            "erf",
            "asinh",
        ),
    ),
):
    for name in names:
        symbol = "__ieee754_" + name if family == "e" else name
        MATH_UNARY[name] = f"ol_{family}_{name}::{symbol}(Soft(x,true)).v"
for name in ("log2", "log10"):
    MATH_UNARY[name] = f"ol_e_{name}::__ieee754_{name}(Soft(x,true)).v"
MATH_UNARY["exp2"] = "ol_s_exp2::exp2(Soft(x,true)).v"
MATH_UNARY["erfc"] = "ol_s_erf::erfc(Soft(x,true)).v"
MATH_UNARY.update(
    sin="fast_math::trig<false>(x)",
    cos="fast_math::trig<true>(x)",
    tanh="fast_math::odd<fast_math::Tanh>(x)",
    atan="fast_math::odd<fast_math::Atan>(x)",
    erf="fast_math::odd<fast_math::Erf>(x)",
    asinh="fast_math::odd<fast_math::Asinh>(x)",
    exp="fast_math::exp_bits<false>(x)",
    expm1="fast_math::exp_bits<true>(x)",
    log="fast_math::log_bits<fast_math::Ln>(x)",
    log2="fast_math::log_bits<fast_math::Log2>(x)",
    log10="fast_math::log_bits<fast_math::Log10>(x)",
)
MATH_BINARY = {
    name: f"ol_e_{name}::__ieee754_{name}(Soft(x,true),Soft(y,true)).v"
    for name in ("atan2", "pow", "hypot", "fmod")
}


MATH_BINARY.update(
    logaddexp="logadd_pair(x,y,false)", logaddexp2="logadd_pair(x,y,true)"
)


@lru_cache(2)
def library(math_functions=False):
    path = Path(__file__).parent
    files = ["vendor/softfloat64.metal", "pointwise.metal"]
    if math_functions:
        files.extend(
            (
                "soft_math.metal",
                "vendor/openlibm.metal",
                "fast_math_tables.metal",
                "fast_math.metal",
                "log_scan.metal",
            )
        )
        operations = MATH_UNARY | MATH_BINARY
    else:
        files.extend(
            (
                "gemv_arithmetic.metal",
                "kernels.metal",
                "fma.metal",
                "skinny_gemm.metal",
                "scatter.metal",
                "histogram.metal",
                "neural.metal",
                "fft.metal",
                "casts.metal",
                "ordering.metal",
                "pooling.metal",
                "conv.metal",
                "linear.metal",
                "resampling.metal",
                "svd.metal",
                "gemm.metal",
                "gemm_dense.metal",
            )
        )
        operations = BINARY | COMPARE | UNARY
    source = "\n".join((path / file).read_text() for file in files)
    for name, expr in operations.items():
        output = "bool" if name in PREDICATES else "ulong"
        if name in UNARY or name in MATH_UNARY:
            source += f"""
kernel void op_{name}(device {output}* out, const device ulong* a, uint i [[thread_position_in_grid]]) {{
    ulong x = a[i]; out[i] = {expr};
}}
kernel void op_{name}_strided(device {output}* out, const device ulong* a, constant long* layout,
                              uint i [[thread_position_in_grid]]) {{
    ulong x = a[pointwise_offsets(i,layout).x]; out[i] = {expr};
}}
"""
        else:
            for suffix, ia, ib in [
                ("", "i", "i"),
                ("_lhs", "0", "i"),
                ("_rhs", "i", "0"),
            ]:
                source += f"""
kernel void op_{name}{suffix}(device {output}* out, const device ulong* a, const device ulong* b,
                               uint i [[thread_position_in_grid]]) {{
    ulong x=a[{ia}], y=b[{ib}]; out[i]={expr};
}}
"""
            source += f"""
kernel void op_{name}_strided(device {output}* out, const device ulong* a, const device ulong* b,
                              constant long* layout, uint i [[thread_position_in_grid]]) {{
    ulong2 pos=pointwise_offsets(i,layout);
    ulong x=a[pos.x], y=b[pos.y]; out[i]={expr};
}}
"""
    return torch.mps.compile_shader(source)


def scalar_bits(value):
    return struct.unpack("=q", struct.pack("=d", value))[0]


def bits(tensor):
    return tensor.view(I64)


@lru_cache(512)
def _pointwise_layout(a_shape, a_stride, b_shape, b_stride):
    shape = torch.broadcast_shapes(a_shape, b_shape)

    def expanded_strides(sizes, strides):
        return (0,) * (len(shape) - len(sizes)) + tuple(
            step if size != 1 else 0 for size, step in zip(sizes, strides)
        )

    a = expanded_strides(a_shape, a_stride)
    b = expanded_strides(b_shape, b_stride)
    axes = []
    for size, sa, sb in reversed(tuple(zip(shape, a, b))):
        if size == 1:
            continue
        if axes and sa == axes[-1][0] * axes[-1][1] and sb == axes[-1][0] * axes[-1][2]:
            axes[-1][0] *= size
        else:
            axes.append([size, sa, sb])
    return shape, (len(axes), *(v for axis in axes for v in axis))


@lru_cache(None)
def _pointwise_kernel(name, suffix):
    lib = library(True) if name in MATH_UNARY or name in MATH_BINARY else library()
    return getattr(lib, "op_" + name + suffix)


def pointwise(name, *inputs):
    a = inputs[0]
    b = inputs[1] if len(inputs) == 2 else None
    a_is_tensor, b_is_tensor = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    suffix, extra = "", ()
    if (
        a_is_tensor
        and a.is_contiguous()
        and (b is None or b_is_tensor and b.is_contiguous() and a.shape == b.shape)
    ):
        shape = a.shape
    elif (
        a_is_tensor
        and a.is_contiguous()
        and (not b_is_tensor or (b.ndim <= a.ndim and b.numel() == 1))
    ):
        shape, suffix = a.shape, "_rhs"
    elif (
        b_is_tensor
        and b.is_contiguous()
        and (not a_is_tensor or (a.ndim <= b.ndim and a.numel() == 1))
    ):
        shape, suffix = b.shape, "_lhs"
    else:
        shape, layout = _pointwise_layout(
            a.shape if a_is_tensor else (),
            a.stride() if a_is_tensor else (),
            b.shape if b_is_tensor else (),
            b.stride() if b_is_tensor else (),
        )
        suffix, extra = "_strided", (layout,)
    # compile_shader binds Python ints as signed 64-bit inline constants. Passing
    # the binary64 representation preserves signed zero and NaN payloads.
    prepared = (
        inputs
        if a_is_tensor and (b_is_tensor or b is None)
        else tuple(t if isinstance(t, torch.Tensor) else scalar_bits(t) for t in inputs)
    )
    output = torch.empty(
        shape, device="mps", dtype=torch.bool if name in PREDICATES else I64
    )
    if output.numel():
        if output.numel() > 2**32 - 1:
            raise NotImplementedError("pointwise grid exceeds uint32")
        _pointwise_kernel(name, suffix)(
            output, *prepared, *extra, threads=output.numel()
        )
    return output if name in PREDICATES else output.view(F64)


def sum_dims(x, dims, keepdim, op=0):
    rank = x.ndim
    dims = tuple(dims or range(rank))
    if any(d < -max(rank, 1) or d >= max(rank, 1) for d in dims):
        raise IndexError("reduction dimension out of range")
    dims = tuple(d % max(rank, 1) for d in dims)
    if len(set(dims)) != len(dims):
        raise RuntimeError("duplicate reduction dimension")
    if not rank:
        return bits(x).clone().view(F64)
    keep = [i for i in range(rank) if i not in dims]
    kept_shape = tuple(x.shape[i] for i in keep)
    width = math.prod(x.shape[i] for i in dims)
    rows = math.prod(kept_shape)
    if width > 2**32 - 1 or rows * ((width + 255) // 256) > 2**32 - 1:
        raise NotImplementedError("reduction grid exceeds uint32")
    if not width or not rows:
        if not width and op >= 2:
            raise IndexError("extremum of an empty reduction")
        data = torch.full(
            kept_shape, scalar_bits(1) if op == 1 else 0, dtype=I64, device=x.device
        )
    else:
        lib = library()
        inner = math.prod(x.shape[dims[-1] + 1 :])
        # A contiguous band of interior dimensions has the physical layout
        # [outer, width, inner]. Tile transposed reads directly on its first pass.
        column_path = (
            x.is_contiguous()
            and dims == tuple(range(dims[0], dims[-1] + 1))
            and width >= 128
            and 16 <= inner < 2**32
            and x.numel() >= 131072
        )
        if column_path:
            blocks = (width + 255) // 256
            groups = rows // inner * blocks * ((inner + 15) // 16)
            shader = getattr(
                lib, ("column_sum", "column_prod", "column_max", "column_min")[op]
            )
            column_path = shader.thread_execution_width == 32 and groups < 2**32
        if column_path:
            data = torch.empty((rows, blocks), dtype=I64, device=x.device)
            shader(
                data,
                x,
                width,
                blocks,
                inner,
                threads=(groups * 256,),
                group_size=(256,),
            )
            width = blocks
        else:
            data = bits(x).permute(*keep, *dims).contiguous().reshape(rows, width)
            # A length-one reduction still owns independent result storage.
            if width == 1:
                data = data.clone()
        while width > 1:
            blocks = (width + 255) // 256
            result = torch.empty((rows, blocks), dtype=I64, device=x.device)
            kernel = getattr(lib, ("simd_sum", "simd_prod", "simd_max", "simd_min")[op])
            lanes = 1 << (width - 1).bit_length() if width <= 32 else 32
            if (
                kernel.thread_execution_width == 32
                and rows * blocks <= (2**32 - 1) // lanes
            ):
                kernel(
                    result,
                    data,
                    width,
                    blocks,
                    threads=(lanes * rows * blocks,),
                    group_size=(128,),
                )
            else:
                lib.row_sum(
                    result,
                    data,
                    width,
                    blocks,
                    op,
                    threads=(256 * rows * blocks,),
                    group_size=(256,),
                )
            data, width = result, blocks
    shape = (
        tuple(1 if i in dims else x.shape[i] for i in range(rank))
        if keepdim
        else kept_shape
    )
    return data.reshape(shape).view(F64)


def mv_tree(a, v):
    rows, width = a.shape
    blocks = (width + 255) // 256
    lib = library()
    column_major = a.stride(0) == 1
    kernel = lib.gemv_tree_columns if column_major else lib.gemv_tree_rows
    threads = blocks * (((rows + 15) // 16) * 256 if column_major else rows * 32)
    if kernel.thread_execution_width != 32 or max(rows, width, threads) >= 2**32:
        raise NotImplementedError(
            "matrix-vector reduction exceeds the supported SIMD grid"
        )
    partial = torch.empty((rows, blocks), device=a.device, dtype=I64)
    kernel(
        partial,
        a,
        v,
        (width, blocks, rows, *a.stride(), v.stride(0)),
        threads=threads,
        group_size=256 if column_major else 128,
    )
    if blocks == 1:
        return partial.view(F64).reshape(rows)
    return sum_dims(partial.view(F64), (1,), False)


def mm(a, b, batched, *, out=None):
    rank = 3 if batched else 2
    if (
        a.ndim != rank
        or b.ndim != rank
        or a.shape[-1] != b.shape[-2]
        or a.shape[:-2] != b.shape[:-2]
    ):
        raise RuntimeError("matrix multiplication shape mismatch")
    m, k = a.shape[-2:]
    n = b.shape[-1]
    shape = (*a.shape[:-2], m, n)
    direct = False
    if out is not None:
        if out.device != a.device or out.dtype != F64 or out.layout != torch.strided:
            raise RuntimeError(
                "matrix output must match the input dtype, device and layout"
            )
        if out.shape == shape and torch._debug_has_internal_overlap(out) == 1:
            raise RuntimeError("matrix output has overlapping elements")
        direct = (
            out.shape == shape
            and out.is_contiguous()
            and not torch._C._overlaps(a, out)
            and not torch._C._overlaps(b, out)
        )
    data = bits(out) if direct else torch.empty(shape, device=a.device, dtype=I64)
    if not k:
        data.zero_()
    elif data.numel():
        _launch_mm(data, a, b, batched)
    if out is None:
        return data.view(F64)
    if not direct:
        if out.shape != shape:
            if out.numel():
                warnings.warn("Resizing a nonempty matrix output tensor.", stacklevel=3)
            out.resize_(shape)
        bits(out).copy_(data)
    return out


def _launch_mm(data, a, b, batched):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if batched else 1
    if max(m, n, k, batch) >= 2**32 - 16:
        raise NotImplementedError("matrix dimensions exceed uint32")
    # The Metal buffer includes the view's storage offset. Zero batch strides
    # support expanded operands without materializing a second weight matrix.
    strides = (
        a.stride(0) if batched else 0,
        *a.stride()[-2:],
        b.stride(0) if batched else 0,
        *b.stride()[-2:],
    )
    lib = library()
    if (
        batched
        and batch >= 4096
        and m == k == n
        and k in (3, 4)
        and batch * k * k < 2**32
    ):
        from .skinny_gemm import launch_small_bmm

        launch_small_bmm(lib, data, a, b, strides)
        return
    if k <= 16 and m >= 128 and n >= 128:
        from .skinny_gemm import launch_skinny_mm

        if launch_skinny_mm(lib, data, a, b, batched):
            return
    if n <= 8 and k >= 32 and m * batch >= 4096 and m * n < 2**32:
        lib.mm_narrow_columns(
            data, a, b, (m, k, n, *strides), threads=(m * n, batch), group_size=(128, 1)
        )
        return
    if min(m, n, k) >= 32 and batch * m * n * k >= 512**3:
        from .exact_gemm import try_exact_mm

        if try_exact_mm(a, b, batched, out=data) is not None:
            return
    if a.is_contiguous() and b.is_contiguous():
        dense_2d = not batched and min(m, n, k) >= 384 and k < 2**32 - 32
        kernel, extra = (lib.mm_dense_22 if dense_2d else lib.mm_staged), ()
    else:
        kernel, extra = lib.mm_staged_strided, (strides,)
    kernel(
        data,
        a,
        b,
        m,
        n,
        k,
        *extra,
        threads=(((n + 15) // 16) * 16, ((m + 7) // 8) * 8, batch),
        group_size=(16, 8, 1),
    )
