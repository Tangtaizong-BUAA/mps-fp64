"""Batched real binary64 SVD with scaled one-sided Jacobi rotations on Metal."""

import math

import torch

from .kernels import F64, I64, bits, library, scalar_bits


def _dispatch_groups(kernel, count, *args):
    width = kernel.thread_execution_width
    kernel(*args, threads=count * width, group_size=width)


def _empty_matrix(batch, rows, columns, device, identity=False):
    if identity and rows == columns and math.prod(batch):
        eye = torch.eye(rows, device=device, dtype=I64) * scalar_bits(1.0)
        raw = eye.expand(*batch, columns, rows).clone(
            memory_format=torch.contiguous_format
        )
    else:
        raw = torch.empty((*batch, columns, rows), device=device, dtype=I64)
    return raw.mT.view(F64)


def svd(a, full_matrices=False, compute_uv=True, *, driver=None):
    if a.dtype != F64 or a.ndim < 2:
        raise RuntimeError("SVD requires a real float64 matrix or batch of matrices")
    if driver is not None:
        raise RuntimeError(
            "linalg.svd: specifying driver is only supported for CUDA inputs with cuSOLVER backend"
        )
    m, n = a.shape[-2:]
    batch = tuple(a.shape[:-2])
    batches = math.prod(batch)
    k = min(m, n)
    rows = max(m, n)
    device = a.device

    def empty():
        return torch.empty(0, dtype=I64, device=device).view(F64)

    if not k or not batches:
        s = torch.empty((*batch, k), dtype=I64, device=device).view(F64)
        if not compute_uv:
            return empty(), s, empty()
        return (
            _empty_matrix(batch, m, m if full_matrices else k, device, full_matrices),
            s,
            _empty_matrix(batch, n if full_matrices else k, n, device, full_matrices),
        )
    columns = rows if full_matrices and compute_uv else k
    # The reflector kernel processes the active thin basis in one SIMD group.
    # Keep its longest serial dimension bounded on the tested Apple GPU family.
    if (
        k > 128
        or rows > 8192
        or max(batches * k * rows, batches * columns * rows) >= 2**32
    ):
        raise NotImplementedError(
            "MPS FP64 Jacobi SVD supports min(m,n)<=128, max(m,n)<=8192 and uint32 grids"
        )
    src = (bits(a).mT if m >= n else bits(a)).contiguous().reshape(batches, k, rows)
    work = torch.empty_like(src)
    exponents = torch.empty((batches, k), device=device, dtype=torch.int32)
    info = torch.zeros((), device=device, dtype=torch.int32)
    lib = library()
    _dispatch_groups(lib.svd_init, batches * k, src, work, exponents, info, (rows,))
    status = info.item()
    if status & 1:
        raise torch.linalg.LinAlgError("SVD input contains nonfinite values")
    if status & 2:
        raise torch.linalg.LinAlgError(
            "within-column exponent range exceeds the normalized binary64 Jacobi workspace"
        )
    if compute_uv:
        v = (
            (torch.eye(k, dtype=I64, device=device) * scalar_bits(1.0))
            .expand(batches, k, k)
            .clone(memory_format=torch.contiguous_format)
        )
    else:
        v = torch.empty(1, dtype=I64, device=device)
    if k > 1:
        even = k + (k & 1)
        pairs = even // 2
        rounds = even - 1
        tolerance = scalar_bits(max(2.0, math.sqrt(rows) * 0.5) * torch.finfo(F64).eps)
        for _ in range(40):
            info.zero_()
            for turn in range(rounds):
                _dispatch_groups(
                    lib.svd_round,
                    batches * pairs,
                    work,
                    v,
                    exponents,
                    info,
                    (rows, k, turn, tolerance, compute_uv),
                )
            if info.item() == 0:
                break
        else:
            raise torch.linalg.LinAlgError(
                "one-sided Jacobi SVD did not converge within 40 sweeps"
            )
    s = torch.empty((batches, k), device=device, dtype=I64)
    norm = torch.empty_like(s)
    _dispatch_groups(lib.svd_norms, batches * k, work, exponents, s, norm, rows)
    sorted_s, order = torch.empty_like(s), torch.empty_like(s)
    lib.svd_order(s, sorted_s, order, k, threads=s.numel())
    s = sorted_s.reshape(*batch, k).view(F64)
    if not compute_uv:
        return empty(), s, empty()
    z = torch.empty((batches, columns, rows), device=device, dtype=I64)
    geometry = (rows, k, columns)
    lib.svd_vectors(work, norm, sorted_s, order, z, geometry, threads=z.numel())
    if full_matrices and columns > k or (sorted_s[:, -1] == 0).any().item():
        refl = z[:, :k].clone(memory_format=torch.contiguous_format)
        tau = torch.empty_like(sorted_s)
        _dispatch_groups(lib.svd_reflectors, batches, refl, tau, sorted_s, geometry)
        _dispatch_groups(
            lib.svd_complete, batches * columns, z, refl, tau, sorted_s, geometry
        )
    sorted_v = torch.empty_like(v)
    lib.svd_gather_right(v, order, sorted_v, k, threads=v.numel())
    v = sorted_v.reshape(*batch, k, k)
    z = z.reshape(*batch, columns, rows)
    # ATen returns both U and Vh stored in column-major matrix order.
    u, vh = (z.mT, v) if m >= n else (v.mT, z)
    return u.view(F64), s, vh.mT.contiguous().mT.view(F64)


def pinv(a, rcond=None, hermitian=False, *, atol=None, rtol=None):
    if a.ndim < 2 or a.dtype != F64:
        raise RuntimeError("linalg.pinv requires a matrix of floating-point values")
    m, n = a.shape[-2:]
    if hermitian and m != n:
        raise RuntimeError("hermitian pinv requires square matrices")
    if not a.numel():
        return bits(a).mT.clone().view(F64)
    if rcond is not None:
        rtol = rcond
    lib = library()
    if hermitian:
        raw = bits(a).contiguous()
        symmetric = torch.empty_like(raw)
        lib.svd_lower_symmetric(raw, symmetric, n, threads=raw.numel())
        a = symmetric.view(F64)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)

    def tolerance(value):
        if isinstance(value, torch.Tensor):
            if value.is_complex():
                raise RuntimeError("pinv tolerances must be real")
            if value.dtype == F64:
                return bits(value).to(device=a.device)
            from .backend import gpu_cast

            return bits(gpu_cast(value.to(device=a.device), F64))
        return torch.tensor(
            scalar_bits(0.0 if value is None else value), device=a.device, dtype=I64
        )

    absolute, relative = tolerance(atol), tolerance(rtol)
    batch = torch.broadcast_shapes(s.shape[:-1], absolute.shape, relative.shape)
    raw = bits(s).expand(*batch, s.shape[-1]).contiguous()
    absolute = absolute.expand(batch).contiguous()
    relative = relative.expand(batch).contiguous()
    inv = torch.empty_like(raw)
    if inv.numel():
        if inv.numel() >= 2**32:
            raise NotImplementedError(
                "broadcast pseudoinverse threshold exceeds uint32 grid"
            )
        lib.svd_reciprocals(
            raw,
            absolute,
            relative,
            inv,
            (s.shape[-1], rtol is None, scalar_bits(max(m, n) * torch.finfo(F64).eps)),
            threads=raw.numel(),
        )
    v = bits(vh).mT.view(F64)
    ut = bits(u).mT.view(F64)
    return (v * inv.unsqueeze(-2).view(F64)) @ ut
