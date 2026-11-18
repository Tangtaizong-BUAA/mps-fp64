import math
import torch
from .kernels import bits, library, F64


def lu_factor(a, pivot=True, check_errors=False):
    if a.ndim < 2 or a.dtype != F64:
        raise RuntimeError("LU requires a matrix of double values")
    rows, cols = a.shape[-2:]
    batches = math.prod(a.shape[:-2])
    if max(rows, cols, batches) >= 2**31:
        raise NotImplementedError("LU dimensions exceed int32")
    packed = bits(a).clone(memory_format=torch.contiguous_format)
    pivots = torch.empty(
        (*a.shape[:-2], min(rows, cols) if pivot else 0),
        device=a.device,
        dtype=torch.int32,
    )
    info = torch.zeros(a.shape[:-2], device=a.device, dtype=torch.int32)
    if packed.numel():
        # Metal requires a valid buffer even when pivot output has zero length.
        pivot_buffer = (
            pivots if pivot else torch.empty(1, device=a.device, dtype=torch.int32)
        )
        library().lu_factor(
            packed,
            pivot_buffer,
            info,
            rows,
            cols,
            int(pivot),
            threads=(128 * batches,),
            group_size=(128,),
        )
        if check_errors and info.any().item():
            raise torch.linalg.LinAlgError("LU factorization encountered a zero pivot")
    return packed.view(F64), pivots, info


def lu_solve(lu, pivots, b, left=True, adjoint=False, trusted=False):
    if (
        min(lu.ndim, b.ndim) < 2
        or lu.shape[-1] != lu.shape[-2]
        or lu.dtype != F64
        or b.dtype != F64
    ):
        raise RuntimeError("LU solve requires compatible FP64 matrices")
    if not (lu.device == pivots.device == b.device):
        raise RuntimeError("LU solve requires operands on one device")
    if pivots.dtype != torch.int32 or pivots.shape != lu.shape[:-1]:
        raise RuntimeError("invalid LU pivot shape or dtype")
    if not left:
        b = bits(b).transpose(-2, -1).view(F64)
        adjoint = not adjoint
    n, nrhs = b.shape[-2:]
    if n != lu.shape[-1]:
        raise RuntimeError("right-hand side dimension mismatch")
    batch = torch.broadcast_shapes(lu.shape[:-2], b.shape[:-2])
    out = bits(b).expand(*batch, n, nrhs).clone(memory_format=torch.contiguous_format)
    if out.numel():
        if max(n, nrhs, math.prod(batch) * nrhs) >= 2**31:
            raise NotImplementedError("LU solve grid exceeds int32")
        if not trusted and ((pivots < 1) | (pivots > n)).any().item():
            raise RuntimeError("pivot index is outside matrix")
        packed = bits(lu).expand(*batch, n, n).contiguous()
        piv = pivots.expand(*batch, n).contiguous()
        library().lu_solve(
            out, packed, piv, n, nrhs, int(adjoint), threads=(math.prod(batch) * nrhs,)
        )
    return (out if left else out.transpose(-2, -1)).view(F64)


def solve_ex(a, b, left=True, check_errors=False):
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise RuntimeError("solve requires square coefficient matrices")
    vector = b.ndim == 1 or (b.ndim == a.ndim - 1 and b.shape == a.shape[:-1])
    if vector and not left:
        raise RuntimeError("right solve requires a matrix right-hand side")
    lu, piv, info = lu_factor(a, check_errors=check_errors)
    matrix = bits(b).unsqueeze(-1).view(F64) if vector else b
    result = lu_solve(lu, piv, matrix, left=left, trusted=True)
    if vector:
        result = bits(result).squeeze(-1).view(F64)
    return result, lu, piv, info


def cholesky(a, upper=False, check_errors=False):
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or a.dtype != F64:
        raise RuntimeError("Cholesky requires square FP64 matrices")
    n = a.shape[-1]
    b = math.prod(a.shape[:-2])
    if max(n, b) >= 2**31:
        raise NotImplementedError("Cholesky grid exceeds int32")
    data = bits(a)
    raw = (data.mT if upper else data).clone(memory_format=torch.contiguous_format)
    info = torch.zeros(a.shape[:-2], device=a.device, dtype=torch.int32)
    if raw.numel():
        library().cholesky_factor(raw, info, n, threads=(b * 128,), group_size=(128,))
        if check_errors and info.any().item():
            raise torch.linalg.LinAlgError("matrix is not positive definite")
    return (raw.mT if upper else raw).view(F64), info


def triangular(a, b, upper, left=True, unitriangular=False):
    if (
        min(a.ndim, b.ndim) < 2
        or a.shape[-2] != a.shape[-1]
        or a.dtype != b.dtype
        or a.dtype != F64
        or a.device != b.device
    ):
        raise RuntimeError("invalid input to triangular solve")
    a, b = bits(a), bits(b)
    if not left:
        a, b = a.mT, b.mT
        upper = not upper
    n, cols = b.shape[-2:]
    if n != a.shape[-1]:
        raise RuntimeError("triangular right-hand side dimension mismatch")
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    result = b.expand(*batch, n, cols).clone(memory_format=torch.contiguous_format)
    if result.numel():
        count = math.prod(batch) * cols
        if max(n, count) >= 2**31:
            raise NotImplementedError("triangular grid exceeds int32")
        library().triangular(
            result,
            a.expand(*batch, n, n).contiguous(),
            n,
            cols,
            int(upper),
            int(unitriangular),
            threads=(count,),
        )
    return (result if left else result.mT).view(F64)


def _write_linalg_outputs(values, outputs, *, allow_cast=False):
    if all(out is None for out in outputs):
        return values
    if any(out is None for out in outputs):
        raise RuntimeError("all output tensors must be provided")
    for value, out in zip(values, outputs):
        if out.device != value.device:
            raise RuntimeError("output tensor must be on the input device")
        if out.dtype != value.dtype and not (allow_cast and torch.can_cast(value.dtype, out.dtype)):
            raise RuntimeError("invalid output dtype for the factorization")
        if out.shape == value.shape and torch._debug_has_internal_overlap(out) == 1:
            raise RuntimeError("output tensor has overlapping elements")
    for value, out in zip(values, outputs):
        if out.shape != value.shape:
            if out.numel():
                import warnings
                warnings.warn(
                    f"Resizing a nonempty output from {tuple(out.shape)} to {tuple(value.shape)}.",
                    UserWarning, stacklevel=3,
                )
            if allow_cast and not out.numel() and out.dtype == value.dtype:
                # A fresh same-dtype Householder destination follows LAPACK's
                # column-major layout. Existing correctly sized views keep their strides.
                out.resize_(value.mT.shape)
                out.transpose_(-2, -1)
            else:
                out.resize_(value.shape)
        if out.dtype == F64:
            bits(out).copy_(bits(value))
        elif out.is_complex():
            from .backend import gpu_cast
            target = out.conj() if out.is_conj() else out
            real = torch.view_as_real(target)
            if real.dtype == F64:
                bits(real[..., 0]).copy_(bits(value))
                bits(real[..., 1]).fill_(-(1 << 63) if out.is_conj() else 0)
            else:
                real[..., 0].copy_(gpu_cast(value, real.dtype))
                real[..., 1].fill_(-0.0 if out.is_conj() else 0.0)
        else:
            from .backend import gpu_cast
            out.copy_(gpu_cast(value, out.dtype))
    return outputs


def _unpack_lu(packed, pivots, unpack_data, unpack_pivots, *, trusted=False):
    if packed.ndim < 2 or packed.dtype != F64:
        raise RuntimeError("lu_unpack requires an FP64 tensor with at least two dimensions")
    batch = tuple(packed.shape[:-2])
    batches = math.prod(batch)
    rows, cols = packed.shape[-2:]
    width = min(rows, cols)
    if max(rows, cols, batches) >= 2**31:
        raise NotImplementedError("LU unpack grid exceeds int32")
    if unpack_pivots:
        if pivots.device != packed.device or pivots.dtype != torch.int32:
            raise RuntimeError("LU pivots must be int32 on the input device")
        if tuple(pivots.shape) != (*batch, width):
            raise RuntimeError("LU pivot shape must match the batch and min(rows, columns)")
        if pivots.numel() and not trusted and ((pivots < 1) | (pivots > rows)).any().item():
            raise RuntimeError("LU pivot index is outside [1, rows]")

    def allocate(shape):
        return torch.empty(shape, dtype=torch.int64, device=packed.device)

    p = allocate((*batch, rows, rows) if unpack_pivots else (0,))
    lower = allocate((*batch, rows, width) if unpack_data else (0,))
    upper = allocate((*batch, width, cols) if unpack_data else (0,))
    if p.numel():
        p.zero_()
        if width:
            library().lu_permutation(p, pivots.contiguous(), rows, width, threads=(rows, batches))
        else:
            p.diagonal(dim1=-2, dim2=-1).fill_(0x3FF0000000000000)
    if unpack_data and packed.numel():
        library().lu_triangles(lower, upper, bits(packed).contiguous(), rows, cols,
                               threads=(cols, rows, batches))
    return p.view(F64), lower.view(F64), upper.view(F64)


def lu_unpack(packed, pivots, unpack_data=True, unpack_pivots=True, *, P=None, L=None, U=None):
    values = _unpack_lu(packed, pivots, unpack_data, unpack_pivots)
    return _write_linalg_outputs(values, (P, L, U))


def lu(a, *, pivot=True, P=None, L=None, U=None):
    packed, pivots, _ = lu_factor(a, pivot=pivot)
    values = _unpack_lu(packed, pivots, True, pivot, trusted=True)
    return _write_linalg_outputs(values, (P, L, U))


def householder_product(a, tau, *, out=None):
    if a.ndim < 2 or a.dtype != F64 or tau.dtype != F64 or a.device != tau.device:
        raise RuntimeError("Householder input and tau must be FP64 on the same device")
    batch = tuple(a.shape[:-2])
    rows, cols = a.shape[-2:]
    if tau.ndim != a.ndim - 1 or tuple(tau.shape[:-1]) != batch:
        raise RuntimeError("Householder input and tau must have equal batch dimensions")
    reflectors = tau.shape[-1]
    if rows < cols or cols < reflectors:
        raise RuntimeError("Householder product requires rows >= columns >= number of reflectors")
    batches = math.prod(batch)
    if max(rows, cols, batches) >= 2**31 or cols * 32 >= 2**32:
        raise NotImplementedError("Householder grid exceeds uint32")
    # LAPACK exposes column-major matrices. Each output column is evaluated
    # independently so the Metal threadgroup also writes consecutive rows.
    raw = torch.empty((*batch, cols, rows), dtype=torch.int64, device=a.device)
    if raw.numel() and reflectors:
        library().householder_columns(raw, bits(a).mT.contiguous(), bits(tau).contiguous(),
                                      rows, cols, reflectors, threads=(cols * 32, batches),
                                      group_size=(32, 1))
    elif raw.numel():
        raw.zero_()
        raw.diagonal(dim1=-2, dim2=-1).fill_(0x3FF0000000000000)
    return _write_linalg_outputs((raw.mT.view(F64),), (out,), allow_cast=True)[0]
