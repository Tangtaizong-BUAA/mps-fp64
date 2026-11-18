"""Scoped FP64 dispatch for eager PyTorch. Keep MPS FP64 tensors inside the scope."""

from collections import Counter
from contextlib import contextmanager
from functools import lru_cache

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

from .special import SPECIAL_NAMES, special
from .indexing import (
    cumulative_extreme,
    indexed,
    search_sorted,
    select_rank,
    unique,
    logcumsumexp,
)
from .convolution import convolution, backward as convolution_backward
from .spectral import transform, pair, complex_tensor, public_fft
from .normalization import batch_norm, batch_norm_backward, group_norm, group_grad
from .histogram import histogram_op
from .embedding import bag as embedding_bag, dense_grad as embedding_grad, bag_grad
from .svd import svd, pinv
from .resampling import RESAMPLING, resample
from .pooling import pool, unpool
from .order import aminmax, sort, extreme
from .linear import (
    cholesky,
    triangular,
    lu_factor,
    lu_solve,
    solve_ex,
    lu as decompose_lu,
    lu_unpack,
    householder_product,
)
from .kernels import (
    BINARY,
    COMPARE,
    MATH_BINARY,
    MATH_UNARY,
    PREDICATES,
    UNARY,
    F64,
    I64,
    bits,
    library,
    mm,
    mv_tree,
    pointwise,
    scalar_bits,
    sum_dims,
)


_UNARY_NAMES = UNARY.keys() | MATH_UNARY.keys()
_BINARY_NAMES = BINARY.keys() | COMPARE.keys()
_INPLACE_UNARY = {name + "_" for name in _UNARY_NAMES - PREDICATES}
_HOT_UNARY = {
    getattr(torch.ops.aten, name).default: name
    for name in _UNARY_NAMES
    if hasattr(getattr(torch.ops.aten, name), "default")
}
_HOT_BINARY = {}
for _name in _BINARY_NAMES | MATH_BINARY.keys():
    _packet = getattr(torch.ops.aten, _name)
    for _overload in ("Tensor", "Tensor_Tensor", "default"):
        if hasattr(_packet, _overload):
            _op = getattr(_packet, _overload)
            _HOT_BINARY[_op] = _name


_HOT_VIEWS = {
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.expand.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.diagonal.default,
    torch.ops.aten.unfold.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.alias.default,
}
_HOT_SCALARS = {}
for _name in ("add", "sub", "mul", "div", "lt", "le", "gt", "ge", "eq", "ne"):
    for _overload in ("Scalar", "Tensor"):
        _packet = getattr(torch.ops.aten, _name)
        if hasattr(_packet, _overload):
            _HOT_SCALARS[getattr(_packet, _overload)] = _name


@lru_cache(maxsize=None)
def _schema_names(func):
    schema = func._schema
    return schema.name.removeprefix("aten::"), schema.overload_name


def is_mps(x):
    return (
        x.type == "mps"
        if isinstance(x, torch.device)
        else x is not None and torch.device(x).type == "mps"
    )


def cpu_value(x):
    if not isinstance(x, torch.Tensor) or x.device.type != "mps":
        return x
    if x.dtype == torch.complex128:
        return complex_tensor(pair(x).cpu())
    if x.dtype == F64 and x.is_neg():
        return bits(torch._neg_view(x)).bitwise_xor(-(1 << 63)).cpu().view(F64)
    return bits(x).cpu().view(F64) if x.dtype == F64 else x.cpu()


def upload(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype == torch.complex128:
        return complex_tensor(pair(x).to("mps"))
    if x.dtype == F64 and x.is_neg():
        x = x.resolve_neg()
    return x.view(I64).to("mps").view(F64) if x.dtype == F64 else x.to("mps")


def gpu_cast(x, dtype):
    if x.dtype == dtype:
        return x
    exact_casts = {
        torch.float32: "to_f32",
        torch.float16: "to_f16",
        torch.bfloat16: "to_bf16",
        torch.bool: "to_bool",
        I64: "to_i64",
        torch.int32: "to_i32",
    }
    if x.dtype == F64 and dtype in exact_casts:
        name = exact_casts[dtype]
    elif x.dtype == F64 and dtype in (torch.int16, torch.int8, torch.uint8):
        return gpu_cast(x, torch.int32).to(dtype)
    elif dtype == F64 and x.dtype == torch.float32:
        name = "from_f32"
    elif dtype == F64 and x.dtype in (
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ):
        name = "from_i64"
        x = x.to(I64)
    elif dtype == F64 and x.dtype in (torch.float16, torch.bfloat16):
        name = "from_f16" if x.dtype == torch.float16 else "from_bf16"
    else:
        raise NotImplementedError(f"GPU cast from {x.dtype} to {dtype}")
    out = torch.empty_like(
        x, dtype=I64 if dtype == F64 else dtype, memory_format=torch.contiguous_format
    )
    if x.numel():
        if x.numel() > 2**32 - 1:
            raise NotImplementedError("cast exceeds uint32 grid")
        arg = bits(x) if x.dtype == F64 else x
        getattr(library(), name)(out, arg.contiguous(), threads=(x.numel(),))
    return out.view(F64) if dtype == F64 else out


def tensor64(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.device.type == "cpu" and x.ndim == 0:
        return upload(x.to(F64))
    if not is_mps(x.device):
        raise RuntimeError("expected MPS tensor or CPU scalar")
    return gpu_cast(x, F64)


def binary(name, a, b, alpha=1):
    if torch.result_type(a, b) != F64:
        raise NotImplementedError("mixed-type scalar promotion")
    a, b = tensor64(a), tensor64(b)
    if alpha != 1:
        b = pointwise("mul", b, alpha)
    return pointwise(name, a, b)


VIEWS = {
    "view",
    "reshape",
    "_unsafe_view",
    "as_strided",
    "transpose",
    "t",
    "mT",
    "mH",
    "adjoint",
    "permute",
    "unsqueeze",
    "squeeze",
    "select",
    "slice",
    "expand",
    "detach",
    "alias",
    "diagonal",
    "unfold",
    "split",
    "split_with_sizes",
    "unbind",
    "narrow",
}
FACTORIES = {
    "empty",
    "empty_strided",
    "zeros",
    "ones",
    "full",
    "empty_like",
    "ones_like",
    "zeros_like",
    "full_like",
    "new_empty",
    "new_empty_strided",
    "new_zeros",
    "new_ones",
    "new_full",
}
OPAQUE = {
    "clone",
    "repeat",
    "flip",
    "roll",
    "cat",
    "stack",
    "index_select",
    "gather",
    "masked_select",
    "index",
    "take",
    "contiguous",
    "tile",
    "triu",
    "tril",
}


def payload_words(x):
    raw = bits(x).contiguous()
    # PyTorch ignores the stride of singleton dimensions in is_contiguous().
    # Reinterpret a dense span with an explicit unit stride before splitting each word.
    return raw.as_strided((raw.numel(),), (1,)).view(torch.int16).reshape(*x.shape, 4)


def as_storage(x):
    return bits(x) if isinstance(x, torch.Tensor) and x.dtype == F64 else x


def as_double(x):
    return x.view(F64) if isinstance(x, torch.Tensor) else x


class CovarianceWeightSums(TorchDispatchMode):
    """Keep low-precision weight sums reproducible inside an FP64 covariance."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        x = args[0] if args else None
        if (
            func is torch.ops.aten.sum.default
            and not kwargs
            and x.dtype == torch.float32
            and x.device.type == "mps"
            and x.ndim == 1
            and 0 < x.numel() <= 128
        ):
            out = torch.empty((), device=x.device, dtype=torch.float32)
            lanes = 4 if x.stride(0) == 1 and x.numel() >= 4 else 1
            library().cov_short_weight_sum(
                out, x.contiguous(), x.numel(), lanes, threads=(1,)
            )
            return out
        return func(*args, **(kwargs or {}))


class ConstructorMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kw = kwargs or {}
        if func is torch.linalg.svd and kw.get("out") is None:
            a = args[0] if args else kw.get("A")
            if (
                isinstance(a, torch.Tensor)
                and a.dtype == F64
                and a.device.type == "mps"
                and len(args) <= 2
                and not (args and "A" in kw)
                and not (len(args) == 2 and "full_matrices" in kw)
                and kw.keys() <= {"A", "full_matrices", "driver", "out"}
            ):
                # The public MPS wrapper hides the ATen SVD autograd node.
                full = args[1] if len(args) == 2 else kw.get("full_matrices", True)
                values = torch.ops.aten._linalg_svd.default(
                    a, full, True, driver=kw.get("driver")
                )
                return torch.return_types.linalg_svd(values)
        if func in (torch.cov, torch.Tensor.cov, torch.ops.aten.cov.default):
            x = args[0] if args else kw.get("input")
            weights = kw.get("aweights")
            if (
                isinstance(x, torch.Tensor)
                and x.dtype == F64
                and x.device.type == "mps"
                and isinstance(weights, torch.Tensor)
                and weights.dtype == torch.float32
                and weights.numel() <= 128
            ):
                with CovarianceWeightSums():
                    return func(*args, **kw)
        if func in (torch.tensor, torch.as_tensor, torch.asarray) and is_mps(
            kw.get("device")
        ):
            dtype = kw.get("dtype")
            value = args[0] if args else kw["data"]
            if dtype is None:
                if isinstance(value, torch.Tensor):
                    dtype = value.dtype
                else:
                    dtype = torch.as_tensor(value).dtype
            if dtype in (F64, torch.complex128):
                if isinstance(value, torch.Tensor):
                    if func is torch.tensor:
                        out = value.detach().to(device="mps", dtype=dtype, copy=True)
                        return out.requires_grad_(kw.get("requires_grad", False))
                    if func is torch.as_tensor:
                        return value.to(device="mps", dtype=dtype)
                    if (
                        func is torch.asarray
                        and is_mps(value.device)
                        and value.dtype == dtype
                    ):
                        # Preserve aliasing when the requested dtype and device already match.
                        out = value.clone() if kw.get("copy") else value
                        want_grad = kw.get("requires_grad", False)
                        if want_grad == out.requires_grad:
                            return out
                        return out.detach().requires_grad_(want_grad)
                if func is torch.asarray and kw.get("copy") is False:
                    raise ValueError("cannot alias CPU input on MPS")
                host = func(*args, **(kw | {"device": "cpu"}))
                result = host.to("mps")
                if func is torch.tensor and result.requires_grad:
                    return result.detach().requires_grad_()
                return result
        return func(*args, **kw)


class Backend(TorchDispatchMode):
    def __init__(self, fallback):
        super().__init__()
        self.fallback = fallback
        self.gpu_ops = Counter()
        self.cpu_ops = Counter()
        self.decomposing = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kw = kwargs or {}
        if not kw and args and isinstance(args[0], torch.Tensor):
            first = args[0]
            if first.dtype == F64 and first.device.type == "mps":
                if func in _HOT_VIEWS:
                    result = func(bits(first), *args[1:]).view(F64)
                    self.gpu_ops[str(func)] += 1
                    return result
                if (
                    len(args) == 2
                    and type(args[1]) in (float, int, bool)
                    and (op := _HOT_SCALARS.get(func))
                ):
                    if op == "sub" and isinstance(args[1], bool):
                        raise RuntimeError(
                            "subtraction with a bool operand is not supported"
                        )
                    result = pointwise(op, first, args[1])
                    self.gpu_ops[str(func)] += 1
                    return result
            if len(args) == 1 and (op := _HOT_UNARY.get(func)) is not None:
                if first.device.type == "mps" and first.dtype == F64:
                    result = pointwise(op, first)
                    self.gpu_ops[str(func)] += 1
                    return result
                if first.dtype != torch.complex128 or first.device.type != "mps":
                    return func(*args)
            elif len(args) == 2 and (op := _HOT_BINARY.get(func)) is not None:
                second = args[1]
                if (
                    isinstance(second, torch.Tensor)
                    and first.dtype == second.dtype == F64
                    and first.device.type == "mps"
                    and second.device == first.device
                ):
                    result = pointwise(op, first, second)
                    self.gpu_ops[str(func)] += 1
                    return result
        flat = (*args, *kw.values())
        if any(isinstance(x, (tuple, list, dict)) for x in flat):
            flat = tree_flatten((args, kw))[0]
        tensors = [x for x in flat if isinstance(x, torch.Tensor)]
        has_double = any(
            is_mps(x.device) and x.dtype in (F64, torch.complex128) for x in tensors
        )
        device = kw.get("device") or (tensors[0].device if tensors else None)
        dtype = kw.get("dtype") or (
            tensors[0].dtype if tensors else torch.get_default_dtype()
        )
        if (
            func._schema.name in ("aten::to", "prims::convert_element_type")
            and len(args) > 1
        ):
            for field, value in zip(func._schema.arguments[1:], args[1:]):
                if field.name in ("dtype", "a_dtype") or isinstance(value, torch.dtype):
                    dtype = value
                elif field.name == "device":
                    device = value
                elif field.name == "other" and torch.is_tensor(value):
                    device, dtype = value.device, value.dtype
        if not has_double and not (is_mps(device) and dtype in (F64, torch.complex128)):
            return func(*args, **kw)
        try:
            result = self.run(
                func, args, kw, any(t.dtype == torch.complex128 for t in tensors)
            )
        except NotImplementedError:
            from torch._decomp import decomposition_table

            decompose = decomposition_table.get(func)
            if decompose is None and torch._C._dispatch_has_kernel_for_dispatch_key(
                func.name(), "CompositeImplicitAutograd"
            ):
                decompose = func.decompose
            if (
                decompose
                and func not in self.decomposing
                and not func._schema.is_mutable
                and not any(r.alias_info for r in func._schema.returns)
                and torch.Tag.nondeterministic_seeded not in func.tags
            ):
                self.decomposing.add(func)
                try:
                    with self:
                        return decompose(*args, **kw)
                except NotImplementedError:
                    pass
                finally:
                    self.decomposing.remove(func)
            if not self.fallback:
                raise
            # CPU replay is allowed only for functional operators with fresh outputs.
            schema = func._schema
            if (
                schema.is_mutable
                or any(r.alias_info for r in schema.returns)
                or torch.Tag.nondeterministic_seeded in func.tags
            ):
                raise
            if (
                any(x.device.type != "mps" and x.ndim != 0 for x in tensors)
                and "device" not in kw
            ):
                raise
            host_kw = tree_map(cpu_value, kw)
            if "device" in kw:
                host_kw["device"] = torch.device("cpu")
            result = func(*tree_map(cpu_value, args), **host_kw)
            self.cpu_ops[str(func)] += 1
            return tree_map(upload, result) if is_mps(device) else result
        self.gpu_ops[str(func)] += 1
        return result

    def run(self, func, args, kw, has_complex=None):
        name, overload = _schema_names(func)
        if name == "to" and overload in ("device", "dtype", "other", "dtype_layout"):
            options = {
                field.name: v for field, v in zip(func._schema.arguments[1:], args[1:])
            } | kw
            x = args[0]
            other = options.pop("other", None)
            if other is not None:
                options.update(device=other.device, dtype=other.dtype)
            copy = options.pop("copy", False)
            if (
                not copy
                and options.get("device", x.device) == x.device
                and options.get("dtype", x.dtype) == x.dtype
                and options.get("memory_format") in (None, torch.preserve_format)
            ):
                return x
            return self.run(torch.ops.aten._to_copy.default, (x,), options)
        if name.startswith("fft_") and overload == "":
            options = {("size" if k in ("s", "n") else k): v for k, v in kw.items()}
            return public_fft(name, *args, **options)
        if name in ("_fft_c2c", "_fft_r2c", "_fft_c2r") and overload == "":
            return transform(name, *args)
        if name in ("complex", "polar") and overload == "":
            a, b = args
            if a.dtype != F64 or b.dtype != F64 or a.device != b.device:
                raise RuntimeError("complex components must match dtype and device")
            if name == "polar":
                a, b = (
                    pointwise("mul", a, pointwise("cos", b)),
                    pointwise("mul", a, pointwise("sin", b)),
                )
            ar, bi = torch.broadcast_tensors(bits(a), bits(b))
            return complex_tensor(torch.stack((ar, bi), -1))
        if name in ("view_as_real", "view_as_complex"):
            return func(*args, **kw)
        if (
            name in ("linalg_householder_product", "orgqr")
            and overload in ("", "out")
            and args[0].dtype == F64
            and args[1].dtype == F64
        ):
            return householder_product(*args, **kw)
        if has_complex is None:
            has_complex = any(
                torch.is_tensor(x) and x.dtype == torch.complex128
                for x in tree_flatten(args)[0]
            )
        if has_complex or kw.get("dtype") == torch.complex128:
            return self.complex_op(func, name, args, kw)
        if name in ("glu", "glu_backward") and overload == "":
            x = args[0] if name == "glu" else args[1]
            dim = (
                args[1]
                if name == "glu" and len(args) > 1
                else args[2]
                if name == "glu_backward"
                else -1
            )
            if not x.ndim or x.shape[dim] % 2:
                raise RuntimeError(
                    "GLU requires a non-scalar input with even split dimension"
                )
            width = x.shape[dim] // 2
            a = bits(x).narrow(dim, 0, width).view(F64)
            b = bits(x).narrow(dim, width, width).view(F64)
            with self:
                gate = b.sigmoid()
                if name == "glu":
                    return a * gate
                grad = args[0]
                return torch.cat(
                    (grad * gate, torch.ops.aten.sigmoid_backward(grad * a, gate)),
                    dim=dim,
                )
        if name == "_trilinear" and overload == "":
            a, b, c, ea, eb, ec, reduce, *order = args
            n = a.ndim + len(ea)
            if n > 52:
                raise NotImplementedError(
                    "tensor contraction exceeds einsum label range"
                )
            if not 0 <= (order[0] if order else 1) < n:
                raise IndexError("invalid contraction unroll dimension")

            def labels(exclude):
                if any(d < -n or d >= n for d in exclude) or len(
                    set(d % n for d in exclude)
                ) != len(exclude):
                    raise RuntimeError("invalid contraction dimensions")
                omitted = {d % n for d in exclude}
                return [i for i in range(n) if i not in omitted]

            with self:
                return torch.einsum(
                    a, labels(ea), b, labels(eb), c, labels(ec), labels(reduce)
                )
        if name == "native_group_norm" and overload == "":
            with self:
                return group_norm(*args, **kw)
        if name == "native_group_norm_backward" and overload == "":
            with self:
                return group_grad(*args, **kw)
        if (
            name in ("pixel_shuffle", "pixel_unshuffle", "_chunk_cat")
            and overload == ""
        ):
            if name == "_chunk_cat" and any(t.dtype != F64 for t in args[0]):
                raise RuntimeError("chunk_cat requires matching input types")
            return func(*tree_map(as_storage, args), **kw).view(F64)
        if name == "embedding_renorm_" and overload == "":
            w, indices, max_norm, norm_type = args
            if w.ndim != 2:
                raise RuntimeError("embedding weights must be two-dimensional")
            chosen = indices.reshape(-1).unique().long()
            if not chosen.numel():
                return w
            with self:
                rows = w.index_select(0, chosen)
                length = torch.linalg.vector_norm(
                    rows, ord=norm_type, dim=1, keepdim=True
                )
                updated = torch.where(
                    length > max_norm, rows * (max_norm / (length + 1e-7)), rows
                )
                result = w.index_copy(0, chosen, updated)
            bits(w).copy_(bits(result))
            return w
        if name in ("_embedding_bag", "_embedding_bag_forward_only") and overload == "":
            return embedding_bag(*args, **kw)
        if name == "_embedding_bag_dense_backward" and overload == "":
            with self:
                return bag_grad(*args, **kw)
        if name == "_embedding_bag_backward" and overload == "":
            (
                g,
                idx,
                offsets,
                owners,
                bag_size,
                maxima,
                n,
                scale,
                mode,
                sparse,
                *extra,
            ) = args
            if sparse:
                raise NotImplementedError("sparse embedding bag gradient")
            with self:
                return bag_grad(
                    g, idx, owners, bag_size, maxima, n, scale, mode, *extra
                )
        if name == "_embedding_bag_per_sample_weights_backward" and overload == "":
            g, w, idx, off, owners, mode, *extra = args
            if mode != 0:
                raise RuntimeError(
                    "per-sample gradients require embedding bag sum mode"
                )
            with self:
                v = (
                    g.index_select(0, owners.long()) * w.index_select(0, idx.long())
                ).sum(1)
                return (
                    v
                    if not extra or extra[0] == -1
                    else v.masked_fill(idx == extra[0], 0.0)
                )
        if name == "embedding_dense_backward" and overload == "":
            with self:
                return embedding_grad(*args, **kw)
        if name == "lerp_" and overload in ("Scalar", "Tensor"):
            with self:
                v = torch.lerp(*args, **kw)
            bits(args[0]).copy_(bits(v))
            return args[0]
        if name == "clamp_" and overload in ("", "Tensor"):
            with self:
                v = torch.clamp(*args, **kw)
            bits(args[0]).copy_(bits(v))
            return args[0]
        if name in (
            "histogram",
            "histogramdd",
            "histc",
            "isin",
            "_histogramdd_bin_edges",
            "_histogramdd_from_bin_cts",
            "_histogramdd_from_bin_tensors",
        ):
            return histogram_op(name, *args, **kw)
        if name in RESAMPLING and overload in ("", "vec"):
            return resample(func, args, kw)
        if name in SPECIAL_NAMES:
            return special(func, *args, **kw)
        if func.namespace == "prims":
            return self.primitive(name, args, kw)
        if name == "convolution" and overload == "":
            return convolution(*args)
        if name == "convolution_backward" and overload == "":
            return convolution_backward(*args)
        if name == "_linalg_svd" and overload == "":
            return svd(*args, **kw)
        if name == "linalg_pinv" and overload in (
            "",
            "atol_rtol_tensor",
            "atol_rtol_float",
            "rcond_tensor",
        ):
            with self:
                return pinv(*args, **kw)
        if name == "linalg_lu" and overload in ("", "out"):
            return decompose_lu(*args, **kw)
        if name == "lu_unpack" and overload in ("", "out"):
            return lu_unpack(*args, **kw)
        if name == "linalg_cholesky_ex" and overload == "":
            return cholesky(*args, **kw)
        if name == "linalg_solve_triangular" and overload == "":
            return triangular(*args, **kw)
        if name == "cholesky_solve" and overload == "":
            b, a = args[:2]
            upper = args[2] if len(args) > 2 else False
            factor = bits(a).mT.view(F64) if upper else a
            y = triangular(factor, b, False)
            return triangular(bits(factor).mT.view(F64), y, True)
        if name == "linalg_inv_ex" and overload == "":
            a = args[0]
            if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
                raise RuntimeError("inverse requires square matrices")
            identity = torch.eye(a.shape[-1], device=a.device, dtype=I64) * scalar_bits(
                1
            )
            inverse, _, _, info = solve_ex(a, identity.expand(a.shape).view(F64), **kw)
            return inverse, info
        if name == "linalg_lu_factor_ex" and overload == "":
            return lu_factor(*args, **kw)
        if name == "linalg_lu_solve" and overload == "":
            return lu_solve(*args, **kw)
        if name == "_linalg_solve_ex" and overload == "":
            return solve_ex(*args, **kw)
        if (
            name in ("rand", "randn", "rand_like", "randn_like", "uniform_", "normal_")
            and "out" not in kw
        ):
            from .random import draw

            if kw.get("dtype", F64) != F64 and name.endswith("_like"):
                return func(bits(args[0]), **kw)
            kind = "normal" if name in ("randn", "randn_like", "normal_") else "uniform"
            target = args[0] if name.endswith("_") else None
            like = name.endswith("_like")
            if target is not None and torch._debug_has_internal_overlap(target) == 1:
                raise RuntimeError("overlapping tensor for random fill")
            shape = args[0].shape if target is not None or like else args[0]
            low = args[1] if len(args) > 1 else 0.0
            high = args[2] if len(args) > 2 else 1.0
            if (kind == "normal" and high < 0) or (
                kind == "uniform" and not low <= high
            ):
                raise RuntimeError("invalid distribution parameters")
            val = draw(kind, shape, "mps", kw.get("generator"))
            scale = high if kind == "normal" else high - low
            if scale != 1:
                val = pointwise("mul", val, scale)
            if low != 0:
                val = pointwise("add", val, low)
            if target is not None:
                bits(target).copy_(bits(val))
                return target
            if like:
                storage = torch.empty_like(
                    bits(args[0]),
                    dtype=I64,
                    memory_format=kw.get("memory_format", torch.preserve_format),
                )
                storage.copy_(bits(val))
                return storage.view(F64)
            return val
        if name == "aminmax" and overload in ("", "out"):
            outputs = (kw["min"], kw["max"]) if overload == "out" else None
            return aminmax(
                args[0], dim=kw.get("dim"), keepdim=kw.get("keepdim", False),
                outputs=outputs,
            )
        if name == "sort" and overload in ("", "stable"):
            return sort(*args, **kw)
        if name in ("argmin", "argmax") and overload == "":
            x = args[0]
            dim = args[1] if len(args) > 1 else None
            keepdim = args[2] if len(args) > 2 else False
            _, idx = extreme(x, dim, keepdim, name == "argmax")
            return idx.reshape([1] * x.ndim) if dim is None and keepdim else idx
        if name in ("min", "max") and overload == "dim":
            return extreme(
                args[0], args[1], args[2] if len(args) > 2 else False, name == "max"
            )
        if name == "topk" and overload == "":
            x, k, *options = args
            dim, largest, ordered = options + [-1, True, True][len(options) :]
            if not 0 <= k <= (x.shape[dim] if x.ndim else 1):
                raise RuntimeError("selected index k out of range")
            vals, idx = sort(x, dim, largest)
            if not x.ndim:
                return vals, idx
            return bits(vals).narrow(dim, 0, k).view(F64), idx.narrow(dim, 0, k)
        if name in ("native_batch_norm", "_native_batch_norm_legit") and overload in (
            "",
            "no_stats",
        ):
            inputs = (
                (*args[:3], None, None, *args[3:]) if overload == "no_stats" else args
            )
            with self:
                return batch_norm(*inputs)
        if name == "native_batch_norm_backward" and overload == "":
            with self:
                return batch_norm_backward(*args)
        if (
            name
            in (
                "index_put",
                "index_add",
                "index_copy",
                "index_fill",
                "masked_scatter",
                "scatter",
                "scatter_add",
                "scatter_reduce",
                "as_strided_scatter",
                "diagonal_scatter",
                "select_scatter",
                "slice_scatter",
            )
            and "out" not in kw
        ):
            return indexed(func, name, overload, args, kw)
        if name in ("_unique2", "unique_dim", "unique_consecutive") and overload == "":
            x, *rest = args
            if name == "unique_consecutive":
                return unique(x, False, *rest, **kw)
            if name == "unique_dim":
                dim, *rest = rest
            else:
                dim = None
            # This PyTorch version sorts the CPU result for both values of sorted.
            return unique(x, True, *rest[1:], dim=dim, **kw)
        if name in ("cummin", "cummax") and overload == "":
            return cumulative_extreme(*args, maximum=name == "cummax")
        if name in ("median", "nanmedian") and overload in ("", "dim"):
            if overload == "" and not args[0].numel():
                return torch.full(
                    (), scalar_bits(float("nan")), device=args[0].device, dtype=I64
                ).view(F64)
            result = select_rank(*args, **kw, ignore_nan=name == "nanmedian")
            return result[0] if overload == "" else result
        if name == "kthvalue" and overload == "":
            return select_rank(
                args[0],
                args[2] if len(args) > 2 else -1,
                args[3] if len(args) > 3 else False,
                k=args[1],
            )
        if name == "floor_divide" or (
            name == "div" and overload in ("Tensor_mode", "Scalar_mode")
        ):
            if "out" in kw or overload.endswith("out"):
                raise NotImplementedError(str(func))
            a, b = args[:2]
            with self:
                q = torch.true_divide(a, b)
                mode = "floor" if name == "floor_divide" else kw.get("rounding_mode")
                if mode is None:
                    return q
                if mode == "trunc":
                    return q.trunc()
                if mode != "floor":
                    raise RuntimeError("unknown division rounding mode")
                mod = torch.fmod(a, b)
                quotient = (a - mod) / b
                adjust = (mod != 0) & ((mod < 0) != (b < 0))
                quotient = torch.where(adjust, quotient - 1, quotient)
                base = quotient.floor()
                rounded = torch.where(quotient - base > 0.5, base + 1, base)
                rounded = torch.where(
                    quotient == 0, torch.copysign(torch.zeros_like(q), q), rounded
                )
                if torch.is_tensor(b):
                    return torch.where(b == 0, q, rounded)
                return q if b == 0 else rounded
        if name == "searchsorted" and "out" not in kw:
            return search_sorted(*args, **kw)
        if name == "bucketize" and "out" not in kw:
            if args[1].ndim != 1:
                raise RuntimeError("bucketize requires one-dimensional boundaries")
            return search_sorted(args[1], args[0], **kw)
        if name == "angle" and overload == "":
            x = args[0]
            with self:
                return torch.where(
                    x.isnan(),
                    x,
                    torch.where(
                        x < 0,
                        torch.full_like(x, 3.141592653589793),
                        torch.zeros_like(x),
                    ),
                )
        if name == "logsumexp" and overload == "":
            x, dims, *args_keep = args
            keep = args_keep[0] if args_keep else False
            with self:
                if not x.numel():
                    return x.exp().sum(dims, keepdim=keep).log()
                mx = x.amax(dims, keepdim=True)
                center = torch.where(mx.isinf(), 0.0, mx)
                total = (x - center).exp().sum(dims, keepdim=True).log() + center
                if keep:
                    return total
                return total.squeeze(tuple(dims) or tuple(range(x.ndim)))
        if name == "_batch_norm_with_update" and overload == "":
            with self:
                y, mean, inv = batch_norm(*args[:5], True, *args[5:])
            return y, mean, inv, torch.empty(0, dtype=torch.uint8, device=y.device)
        if (
            name == "_euclidean_dist" or (name == "_cdist_forward" and args[2] == 2)
        ) and overload == "":
            a, b = args[:2]
            if (
                min(a.ndim, b.ndim) < 2
                or a.shape[-1] != b.shape[-1]
                or a.device != b.device
                or a.dtype != F64
                or b.dtype != F64
            ):
                raise RuntimeError("invalid Euclidean distance operands")
            batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            na, width = a.shape[-2:]
            nb = b.shape[-2]
            out = torch.empty((*batch, na, nb), device=a.device, dtype=I64)
            if out.numel() >= 2**32 or width >= 2**32:
                raise NotImplementedError(
                    "Euclidean distance exceeds Metal index range"
                )
            if out.numel() and width:
                library().euclidean_dist(
                    out,
                    bits(a).expand(*batch, na, width).contiguous(),
                    bits(b).expand(*batch, nb, width).contiguous(),
                    na,
                    nb,
                    width,
                    threads=(out.numel(),),
                )
            elif not width:
                out.zero_()
            return out.view(F64)
        if name == "_cdist_forward" and overload == "":
            a, b, p = args[:3]
            if p < 0 or min(a.ndim, b.ndim) < 2 or a.shape[-1] != b.shape[-1]:
                raise RuntimeError("invalid cdist operands")
            with self:
                delta = a.unsqueeze(-2) - b.unsqueeze(-3)
                return torch.linalg.vector_norm(delta, p, dim=delta.ndim - 1)
        if name in ("logaddexp", "logaddexp2") and overload == "":
            return binary(name, *args)
        if name == "logcumsumexp" and overload == "":
            return logcumsumexp(*args, **kw)
        if name == "frac" and overload == "":
            x = args[0]
            return pointwise("sub", x, pointwise("trunc", x))
        if name == "remainder" and overload in ("Tensor", "Scalar"):
            a, b = args
            with self:
                r = torch.fmod(a, b)
                return torch.where((r != 0) & ((r < 0) != (b < 0)), r + b, r)
        if name in ("clamp_min_", "clamp_max_") and overload in ("", "Tensor"):
            dest, value = args
            op = "maximum" if name == "clamp_min_" else "minimum"
            bits(dest).copy_(bits(binary(op, dest, value)))
            return dest
        if name == "equal":
            a, b = args
            if a.shape != b.shape:
                return False
            return bool(binary("eq", a, b).all().item())
        if name == "allclose":
            with self:
                return bool(torch.isclose(*args, **kw).all().item())
        if name == "cholesky" and overload == "":
            return cholesky(args[0], args[1] if len(args) > 1 else False, True)[0]
        if name == "triangular_solve" and overload == "":
            b, a, *flags = args
            upper, transpose, unit = flags + [True, False, False][len(flags) :]
            coeff = bits(a).mT.view(F64) if transpose else a
            return triangular(
                coeff, b, not upper if transpose else upper, unitriangular=unit
            ), bits(a).clone().view(F64)
        if name in ("addbmm", "addr") and overload == "":
            c, a, b = args
            if not all(x.dtype == F64 and is_mps(x.device) for x in (c, a, b)):
                raise RuntimeError("operands must share dtype and device")
            if name == "addbmm":
                prod = sum_dims(mm(a, b, True), (0,), False)
            else:
                if a.ndim != 1 or b.ndim != 1:
                    raise RuntimeError("addr expects vectors")
                prod = pointwise("mul", bits(a)[:, None].view(F64), b)
            alpha, beta = kw.get("alpha", 1), kw.get("beta", 1)
            c = bits(c).expand(prod.shape).view(F64)
            prod = pointwise("mul", prod, alpha) if alpha != 1 else prod
            return (
                prod if beta == 0 else pointwise("add", prod, pointwise("mul", c, beta))
            )
        if name in ("_linalg_det", "_linalg_slogdet") and overload == "":
            a = args[0]
            if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
                raise RuntimeError("determinant requires square matrices")
            lu, piv, info = lu_factor(a)
            diag = bits(lu).diagonal(dim1=-2, dim2=-1).view(F64)
            with self:
                swap = (
                    piv
                    != torch.arange(
                        1, a.shape[-1] + 1, device=a.device, dtype=piv.dtype
                    )
                ).sum(-1)
                parity = (1 - 2 * (swap % 2)).double()
                if name == "_linalg_det":
                    return parity * diag.prod(-1), lu, piv
                return parity * diag.sign().prod(-1), diag.abs().log().sum(-1), lu, piv
        if name == "max_pool2d_backward" and overload == "":
            fields = {f.name: f.default_value for f in func._schema.arguments}
            fields.update({f.name: v for f, v in zip(func._schema.arguments, args)})
            fields.update(kw)
            fargs = tuple(
                fields[key]
                for key in (
                    "self",
                    "kernel_size",
                    "stride",
                    "padding",
                    "dilation",
                    "ceil_mode",
                )
            )
            _, indices = pool(torch.ops.aten.max_pool2d_with_indices.default, fargs, {})
            return pool(
                torch.ops.aten.max_pool2d_with_indices_backward.default,
                (args[0], *fargs, indices),
                {},
            )
        pool_ops = {
            "avg_pool2d",
            "avg_pool3d",
            "max_pool2d_with_indices",
            "max_pool3d_with_indices",
            "_adaptive_avg_pool2d",
            "_adaptive_avg_pool3d",
            "adaptive_max_pool2d",
            "adaptive_max_pool3d",
        }
        if name.removesuffix("_backward") in pool_ops and overload == "":
            return pool(func, args, kw)
        if name in ("max_unpool2d", "max_unpool3d") and overload == "":
            return unpool(func, args, kw)
        if name == "diag_embed" and overload == "":
            raw = bits(args[0])
            shape = func(
                torch.empty(raw.shape, device="meta", dtype=F64), *args[1:], **kw
            ).shape
            params = list(args[1:]) + [0, -2, -1][len(args) - 1 :]
            out = torch.zeros(shape, device=raw.device, dtype=I64)
            out.diagonal(*params).copy_(raw)
            return out.view(F64)
        if name == "nonzero" and overload == "":
            return torch.nonzero(gpu_cast(args[0], torch.bool))
        if name == "cholesky_inverse" and overload == "":
            a = args[0]
            eye = torch.eye(a.shape[-1], device=a.device, dtype=I64) * scalar_bits(1)
            return self.run(
                torch.ops.aten.cholesky_solve.default,
                (eye.expand(a.shape).view(F64), *args),
                {},
            )
        if name == "empty_permuted" and overload == "":
            return func(*args, **(kw | {"dtype": I64})).view(F64)
        if name == "eye" and overload in ("", "m"):
            raw = func(*args, **(kw | {"dtype": I64}))
            return (raw * scalar_bits(1)).view(F64)
        if name in ("linspace", "logspace") and "out" not in kw:
            start, end, steps, *extra = args
            if steps < 0:
                raise RuntimeError("number of steps must be non-negative")
            if steps >= 2**32:
                raise NotImplementedError("linspace exceeds the Metal grid")

            def endpoint(value):
                if not torch.is_tensor(value):
                    return scalar_bits(value)
                if value.ndim:
                    raise RuntimeError("linspace endpoints must be scalars")
                return bits(tensor64(value))

            raw = torch.empty(steps, device="mps", dtype=I64)
            if steps:
                library().linspace64(
                    raw, endpoint(start), endpoint(end), steps, threads=(steps,)
                )
            result = raw.view(F64)
            return (
                pointwise("pow", extra[0] if extra else kw.get("base", 10.0), result)
                if name == "logspace"
                else result
            )
        if name == "scalar_tensor" and overload == "":
            return torch.full((), scalar_bits(args[0]), device="mps", dtype=I64).view(
                F64
            )
        if name == "arange" and overload in ("", "start", "start_step"):
            start, end, step = (
                (0, args[0], 1)
                if len(args) == 1
                else (*args, 1)
                if len(args) == 2
                else args
            )
            shape = func(*args, **(kw | {"device": "meta"})).shape
            out = torch.empty(shape, device="mps", dtype=I64)
            if out.numel():
                if out.numel() > 2**32 - 1:
                    raise NotImplementedError("range grid exceeds uint32")
                library().range64(
                    out, scalar_bits(start), scalar_bits(step), threads=(out.numel(),)
                )
            return out.view(F64)
        if name == "index_select" and overload == "" and args[0].ndim:
            x, dim, idx = args
            if not -x.ndim <= dim < x.ndim:
                raise IndexError("index dimension out of range")
            words = payload_words(x)
            return (
                torch.index_select(words, dim % x.ndim, idx)
                .contiguous()
                .view(F64)
                .squeeze(-1)
            )
        if name == "gather" and overload == "":
            x, dim, idx = args[:3]
            if idx.numel() == 0:
                return torch.empty(idx.shape, device=x.device, dtype=I64).view(F64)
            if idx.dtype not in (I64, torch.int32):
                raise RuntimeError("gather index requires int64")
            idx = idx.to(I64)
            if not x.ndim or not idx.ndim:
                if max(x.ndim, idx.ndim) > 1 or dim not in (-1, 0):
                    raise RuntimeError("invalid scalar gather dimensions")
                gathered = self.run(
                    func, (bits(x).reshape(-1).view(F64), 0, idx.reshape(-1)), kw
                )
                return bits(gathered).reshape(idx.shape).view(F64)
            if dim < -x.ndim or dim >= x.ndim:
                raise IndexError("gather dimension out of range")
            words = payload_words(x)
            idx = idx.unsqueeze(-1).expand(*idx.shape, 4)
            return (
                torch.gather(words, dim % x.ndim, idx)
                .contiguous()
                .view(F64)
                .squeeze(-1)
            )
        if name == "constant_pad_nd" and overload == "":
            x, pads = args[:2]
            if len(pads) % 2 or len(pads) > 2 * x.ndim:
                raise RuntimeError("invalid padding dimensions")
            widths = [(0, 0)] * x.ndim
            for i in range(len(pads) // 2):
                widths[-1 - i] = pads[2 * i : 2 * i + 2]
            shape = tuple(n + left + right for n, (left, right) in zip(x.shape, widths))
            if any(n < 0 for n in shape):
                raise RuntimeError("negative padded dimension")
            target = torch.full(
                shape,
                scalar_bits(args[2] if len(args) > 2 else 0),
                dtype=I64,
                device=x.device,
            )
            source = tuple(
                slice(max(-left, 0), n - max(-right, 0))
                for n, (left, right) in zip(x.shape, widths)
            )
            dest = tuple(
                slice(max(left, 0), n - max(right, 0))
                for n, (left, right) in zip(shape, widths)
            )
            target[dest].copy_(bits(x)[source])
            return target.view(F64)
        if (
            name in ("amin", "amax", "prod", "min", "max")
            and overload in ("", "dim_int")
            and "out" not in kw
        ):
            x = args[0]
            if kw.get("dtype", F64) != F64:
                raise NotImplementedError(str(func))
            dims = args[1] if len(args) > 1 else None
            if isinstance(dims, int):
                dims = (dims,)
            return sum_dims(
                x,
                dims,
                args[2] if len(args) > 2 else False,
                {"amax": 2, "max": 2, "amin": 3, "min": 3, "prod": 1}[name],
            )
        if name == "_to_copy":
            x = args[0]
            dtype = kw.get("dtype", x.dtype)
            device = kw.get("device", x.device)
            options = {
                k: v for k, v in kw.items() if k in ("memory_format", "non_blocking")
            }
            if not is_mps(device):
                return cpu_value(x).to(device=device, dtype=dtype, copy=True, **options)
            if not is_mps(x.device):
                return upload(x.to(dtype=dtype, **options))
            if x.dtype == F64 and dtype in (torch.complex64, torch.complex32):
                return gpu_cast(
                    x, torch.float32 if dtype == torch.complex64 else torch.float16
                ).to(dtype)
            result = gpu_cast(x, dtype)
            if result is x or "memory_format" in options:
                raw = as_storage(result).clone(
                    memory_format=options.get("memory_format", torch.preserve_format)
                )
                return raw.view(dtype)
            return result
        if name == "copy_" and overload == "":
            dst, src = args[:2]
            if dst.dtype == F64 and is_mps(dst.device):
                if src.device.type == "cpu":
                    src = upload(src.to(F64))
                src = gpu_cast(src, F64)
                source_bits = bits(torch._neg_view(src) if src.is_neg() else src)
                if dst.is_neg() != src.is_neg():
                    source_bits = source_bits.bitwise_xor(-(1 << 63))
                bits(torch._neg_view(dst) if dst.is_neg() else dst).copy_(source_bits)
            elif is_mps(dst.device):
                dst.copy_(gpu_cast(src, dst.dtype))
            else:
                dst.copy_(cpu_value(src))
            return dst
        if name in ("squeeze_", "unsqueeze_", "transpose_", "t_"):
            # These operators only alter shape/stride on the original TensorImpl.
            return func(*args, **kw)
        if name == "view" and overload == "dtype":
            return func(*args, **kw)
        if name == "_local_scalar_dense":
            return cpu_value(args[0]).item()
        if name in FACTORIES and "out" not in kw:
            if not is_mps(
                kw.get("device")
                or (args[0].device if args and torch.is_tensor(args[0]) else "mps")
            ):
                raise NotImplementedError(str(func))
            target = kw.get("dtype", F64)
            if target != F64:
                return func(*tree_map(as_storage, args), **kw)
            values = list(tree_map(as_storage, args))
            if "full" in name:
                index = 2 if name == "new_full" else 1
                if isinstance(values[index], complex):
                    raise NotImplementedError(str(func))
                values[index] = scalar_bits(values[index])
            if "ones" in name:
                # Preserve all factory layout/options while filling the FP64 bit pattern.
                raw = func(*values, **(kw | {"dtype": I64}))
                raw.fill_(scalar_bits(1))
            else:
                raw = func(*values, **(kw | {"dtype": I64}))
            return raw.view(F64)
        if name in ("resize_", "resize_as_") and overload == "":
            # The native MPS allocator grows byte storage and updates this TensorImpl.
            return func(*args, **kw)
        if name in ("tril_", "triu_") and overload == "":
            func(bits(args[0]), *args[1:], **kw)
            return args[0]
        if name in _INPLACE_UNARY and overload == "":
            dest = args[0]
            if torch._debug_has_internal_overlap(dest) == 1:
                raise RuntimeError(
                    "in-place rounding requires non-overlapping elements"
                )
            bits(dest).copy_(bits(pointwise(name[:-1], dest)))
            return dest
        if name in ("masked_fill", "masked_fill_") and overload in ("Scalar", "Tensor"):
            x, mask, value = args
            if mask.dtype != torch.bool or mask.device != x.device:
                raise RuntimeError(
                    "masked_fill requires a bool mask on the input device"
                )
            if torch.is_tensor(value):
                if value.ndim:
                    raise RuntimeError("masked_fill value must be a scalar")
                raw = bits(tensor64(value))
            else:
                raw = torch.full((), scalar_bits(value), device=x.device, dtype=I64)
            result = torch.where(mask, raw, bits(x))
            if name == "masked_fill_":
                bits(x).copy_(result)
                return x
            return result.view(F64)
        if name in ("fill_", "zero_") and overload in ("Scalar", ""):
            value = args[1] if len(args) > 1 else 0
            bits(args[0]).fill_(scalar_bits(value))
            return args[0]
        if name == "_neg_view" and overload == "" and args[0].dtype == F64:
            return func(*args, **kw)
        if name == "clone" and args[0].dtype == F64 and args[0].is_neg():
            # The Negative dispatch key resolves a lazy negative operand by
            # calling clone before mm. Preserve the layout and negate the bits.
            raw = bits(torch._neg_view(args[0]))
            return func(raw, **kw).bitwise_xor_(-(1 << 63)).view(F64)
        if name in VIEWS and "out" not in kw:
            return tree_map(as_double, func(*tree_map(as_storage, args), **kw))
        if name in OPAQUE and "out" not in kw:
            if name in {"cat", "stack"} and any(t.dtype != F64 for t in args[0]):
                raise NotImplementedError("mixed sequence dtype")
            return func(*tree_map(as_storage, args), **kw).view(F64)
        if name.rstrip("_") in _BINARY_NAMES and overload in (
            "",
            "Tensor",
            "Scalar",
            "out",
            "Scalar_out",
        ):
            op = name.rstrip("_")
            result = binary(op, *args[:2], alpha=kw.get("alpha", 1))
            dest = args[0] if name.endswith("_") else kw.get("out")
            if dest is None:
                return result
            if dest.shape != result.shape or dest.dtype != result.dtype:
                raise NotImplementedError("dtype or shape changing out")
            as_storage(dest).copy_(as_storage(result))
            return dest
        if name in _UNARY_NAMES and overload == "":
            return pointwise(name, args[0])
        if name == "pow" and overload == "Tensor_Scalar":
            x, exp = args
            if type(exp) is int and 0 <= exp <= 64:
                if exp == 0:
                    return torch.full_like(bits(x), scalar_bits(1)).view(F64)
                if exp == 1:
                    return bits(x).clone().view(F64)
                result = None
                while exp:
                    if exp & 1:
                        result = x if result is None else pointwise("mul", result, x)
                    exp >>= 1
                    if exp:
                        x = pointwise("mul", x, x)
                return result
        if name in MATH_BINARY and overload in (
            "",
            "Tensor_Tensor",
            "Tensor",
            "Tensor_Scalar",
            "Scalar",
        ):
            return binary(name, *args)
        if name in ("var", "var_mean", "std", "std_mean") and overload == "correction":
            x = args[0]
            dims = args[1] if len(args) > 1 else None
            keepdim = kw.get("keepdim", False)
            from math import prod

            n = prod(x.shape[d] for d in dims) if dims and x.ndim else x.numel()
            center = pointwise("div", sum_dims(x, dims, True), n)
            delta = pointwise("sub", x, center)
            ss = sum_dims(pointwise("mul", delta, delta), dims, keepdim)
            correction = kw.get("correction")
            value = pointwise(
                "div", ss, max(0, n - (1 if correction is None else correction))
            )
            if name.startswith("std"):
                value = pointwise("sqrt", value)
            return (
                (value, center.reshape(value.shape)) if name.endswith("mean") else value
            )
        if (
            name in ("sum", "mean")
            and overload in ("", "dim_IntList", "dim")
            and kw.get("dtype", F64) == F64
        ):
            x = args[0]
            dims = args[1] if len(args) > 1 else None
            result = sum_dims(
                gpu_cast(x, F64), dims, args[2] if len(args) > 2 else False
            )
            if name == "mean":
                import math

                count = (
                    math.prod(x.shape[d] for d in dims)
                    if dims and x.ndim
                    else x.numel()
                )
                result = pointwise("div", result, count)
            return result
        if (
            name == "mv" and overload == "" and len(args) == 2
            and all(x.dtype == F64 and is_mps(x.device) for x in args)
            and args[0].ndim == 2 and args[1].ndim == 1
            and args[0].shape[1] == args[1].numel() and args[1].numel() > 32
            and args[0].numel() >= 65536
        ):
            return mv_tree(*args)
        if (
            name in ("mm", "bmm")
            and overload in ("", "out")
            and all(x.dtype == F64 and is_mps(x.device) for x in args)
        ):
            return mm(*args, name == "bmm", **kw)
        if name == "where" and overload == "self":
            cond, a, b = args
            if cond.dtype != torch.bool or not is_mps(cond.device):
                raise NotImplementedError(str(func))
            a, b = [
                v
                if torch.is_tensor(v)
                else torch.full((), scalar_bits(v), device="mps", dtype=I64).view(F64)
                for v in (a, b)
            ]
            if torch.result_type(a, b) != F64:
                raise NotImplementedError(str(func))
            a, b = tensor64(a), tensor64(b)
            cond, a, b = torch.broadcast_tensors(cond, bits(a), bits(b))
            out = torch.empty(a.shape, dtype=I64, device=a.device)
            if out.numel():
                if out.numel() > 2**32 - 1:
                    raise NotImplementedError("where grid exceeds uint32")
                library().choose(
                    out,
                    cond.contiguous(),
                    a.contiguous(),
                    b.contiguous(),
                    threads=(out.numel(),),
                )
            return out.view(F64)
        raise NotImplementedError(f"MPS FP64 GPU implementation missing for {func}")

    def complex_op(self, func, name, args, kw):
        x = args[0] if args else None
        if name == "_to_copy":
            dtype = kw.get("dtype", x.dtype)
            device = kw.get("device", x.device)
            if not is_mps(device):
                return cpu_value(x).to(device=device, dtype=dtype, copy=True)
            if x.device.type == "cpu":
                return upload(x.to(dtype))
            if x.dtype == F64 and dtype == torch.complex128:
                return complex_tensor(
                    torch.stack((bits(x), torch.zeros_like(bits(x))), -1)
                )
            if x.dtype == torch.complex128:
                raw = pair(x)
                if dtype == torch.complex128:
                    return complex_tensor(raw.clone())
                if dtype in (torch.complex64, torch.complex32):
                    parts = gpu_cast(
                        raw.view(F64),
                        torch.float32 if dtype == torch.complex64 else torch.float16,
                    )
                    return torch.view_as_complex(parts)
                real = raw[..., 0].contiguous().view(F64)
                if dtype == torch.bool:
                    return ((raw[..., 0] << 1) != 0) | ((raw[..., 1] << 1) != 0)
                return (
                    gpu_cast(real, dtype)
                    if dtype != F64
                    else bits(real).clone().view(F64)
                )
            if dtype == torch.complex128 and x.is_complex():
                return complex_tensor(
                    bits(gpu_cast(torch.view_as_real(x.resolve_conj()), F64))
                )
        if name in FACTORIES and (
            kw.get("dtype") not in (None, torch.complex128)
            or not is_mps(
                kw.get("device") or (x.device if torch.is_tensor(x) else "mps")
            )
        ):
            raise NotImplementedError(
                "complex prototype factory with changed dtype/device"
            )
        if name in ("empty", "zeros", "ones") and "out" not in kw:
            shape = args[0]
            raw = torch.empty((*shape, 2), device="mps", dtype=I64)
            if name != "empty":
                raw.zero_()
                if name == "ones":
                    raw[..., 0].fill_(scalar_bits(1))
            return complex_tensor(raw)
        if name in ("empty_like", "zeros_like", "ones_like", "new_empty", "new_zeros"):
            shape = args[1] if name.startswith("new") else x.shape
            return self.complex_op(
                func, name.replace("_like", "").removeprefix("new_"), (shape,), kw
            )
        if (
            name == "copy_"
            and not x.is_conj()
            and x.dtype == torch.complex128
            and args[1].dtype == torch.complex128
        ):
            pair(x).copy_(pair(args[1]))
            return x
        if name == "sgn":
            raw = pair(x)
            with self:
                r, i = raw[..., 0].view(F64), raw[..., 1].view(F64)
                mag = torch.hypot(r, i)
                safe = torch.where(mag == 0, 1.0, mag)
                return torch.complex(r / safe, i / safe)
        if name == "angle" and x.dtype == torch.complex128:
            raw = pair(x)
            return pointwise("atan2", raw[..., 1].view(F64), raw[..., 0].view(F64))
        if name in ("transpose_", "squeeze_", "unsqueeze_"):
            return func(*args, **kw)
        if name == "_conj":
            return func(*args, **kw)
        if name in ("_conj_physical", "conj_physical"):
            raw = pair(x).clone()
            raw[..., 1].bitwise_xor_(-(1 << 63))
            return complex_tensor(raw)
        if name == "constant_pad_nd":
            raw = pair(x)
            val = complex(args[2] if len(args) > 2 else 0)
            real, imag = [
                self.run(func, (raw[..., i].view(F64), args[1], v), {})
                for i, v in enumerate((val.real, val.imag))
            ]
            return complex_tensor(torch.stack((bits(real), bits(imag)), -1))
        if name in ("add", "sub", "mul") and func._schema.overload_name in (
            "Tensor",
            "Scalar",
        ):
            with self:

                def parts(t):
                    if isinstance(t, torch.Tensor):
                        if t.is_complex():
                            raw = pair(t)
                            return raw[..., 0].view(F64), raw[..., 1].view(F64)
                        return t, 0.0
                    val = complex(t)
                    return val.real, val.imag

                ar, ai = parts(args[0])
                br, bi = parts(args[1])
                if name == "mul":
                    real, imag = ar * br - ai * bi, ar * bi + ai * br
                else:
                    alpha = kw.get("alpha", 1)
                    if isinstance(alpha, complex):
                        raise NotImplementedError("complex add alpha")
                    scale = alpha if name == "add" else -alpha
                    real, imag = ar + scale * br, ai + scale * bi
                return torch.complex(real, imag)
        if name in VIEWS:
            return func(*args, **kw)
        if name == "abs":
            raw = pair(x)
            return pointwise("hypot", raw[..., 0].view(F64), raw[..., 1].view(F64))
        if name == "clone":
            return complex_tensor(pair(x).clone())
        raise NotImplementedError("MPS complex128 operation missing for " + str(func))

    def primitive(self, name, args, kw):
        op = name.removeprefix("prims::")
        if op in UNARY or op in MATH_UNARY:
            return pointwise(op, args[0])
        if op in BINARY or op in COMPARE or op in MATH_BINARY:
            return binary(op, *args)
        if op in ("sum", "amax", "amin", "prod"):
            return sum_dims(
                args[0], args[1], False, {"sum": 0, "prod": 1, "amax": 2, "amin": 3}[op]
            )
        if op == "var":
            value, dims, *rest = args
            return self.run(
                torch.ops.aten.var.correction,
                (value, dims),
                {"correction": rest[0] if rest else kw.get("correction", 1)},
            )
        if op == "reshape":
            return bits(args[0]).reshape(args[1]).view(F64)
        if op == "collapse_view" or op == "collapse":
            return bits(args[0]).flatten(args[1], args[2]).view(F64)
        if op == "transpose":
            return bits(args[0]).permute(args[1]).view(F64)
        if op == "convert_element_type":
            return gpu_cast(args[0], args[1])
        if op == "broadcast_in_dim":
            x, shape, axes = args
            view = [1] * len(shape)
            for i, axis in enumerate(axes):
                view[axis] = x.shape[i]
            return bits(x).reshape(view).expand(shape).view(F64)
        if op == "where":
            return self.run(torch.ops.aten.where.self, args, kw)
        raise NotImplementedError(
            "MPS FP64 GPU implementation missing for prims::" + op
        )


@contextmanager
def enable(*, cpu_fallback=False):
    """Activate in this thread, including backward and copying results back to CPU."""
    mode = Backend(cpu_fallback)
    with mode, ConstructorMode():
        yield mode
