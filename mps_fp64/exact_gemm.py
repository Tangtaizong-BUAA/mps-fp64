"""Proven-exact native GPU multiplication for dense matrices on a binary64 lattice.

The GPU finds each snapshot's largest magnitude and least set bit. Write
A=I*2**p and B=J*2**q with integer matrices I and J. Every exact product and
every partial sum, in any addition order, has integer magnitude at most
k*max(abs(I))*max(abs(J)). A bound <=2**53, p+q>=-1074 and a finite maximum
scaled result mean every ascending binary64 FMA is exact. Starting from +0,
cancellations and zero products give +0.

For <=8-bit magnitudes and a <=2**24 bound, native binary32 accumulators
already supply all required bits. For larger bounds or <=16-bit magnitudes,
K panels limit every native dot to 2**24. A balanced radix-256 split and
three products (low, high and their sums) handle wider inputs. Each panel
is reconstructed with signed int64, then the exact integer is encoded back
to binary64. The Metal shader fixes input and accumulator precision directly.

The GPU sample can only reject. The complete input is revalidated from the
same words used to form each immutable packed snapshot on every invocation.
"""

from functools import lru_cache
from pathlib import Path

import torch

_MIN_WORK = 512**3


@lru_cache(1)
def _library():
    if not torch.backends.mps.get_name().startswith("Apple"):
        return None
    source = Path(__file__).with_suffix(".metal").read_text()
    try:
        library = torch.mps.compile_shader(source)
    except (SyntaxError, RuntimeError):
        # An older Metal compiler/device may lack mixed SIMD-group matrices.
        # The caller can still evaluate this input using the integer FMA kernel.
        return None
    return library if library.exact_mma.thread_execution_width == 32 else None


def _layout(t, batched):
    dense = t.is_contiguous() or t.mT.is_contiguous()
    return [
        t.numel(),
        *t.shape[-2:],
        t.stride(0) if batched else 0,
        *t.stride()[-2:],
        int(dense),
    ]


def _certificate(metadata, k):
    maxima = []
    for magnitude, exponent in metadata:
        if magnitude >= 0x7FF0000000000000:
            return None
        field = magnitude >> 52
        base = max(1, field) - 1075
        sig = (magnitude & ((1 << 52) - 1)) | ((1 << 52) if field else 0)
        shift = base - exponent
        maxima.append(sig << shift if shift >= 0 else sig >> -shift)
    if not all(maxima):
        return 0, 0, 0, 0
    p, q = metadata[0][1], metadata[1][1]
    bound = k * maxima[0] * maxima[1]
    if (
        max(maxima) > 65535
        or bound > 1 << 53
        or p + q < -1074
        or p + q + bound.bit_length() - 1 > 1023
    ):
        return None
    return p, q, *maxima


def try_exact_mm(a, b, batched, *, out=None):
    """Multiply a certified MPS binary64 pair, or return None to the caller.

    An optional contiguous int64 output receives the final bit patterns only
    after both source snapshots pass the complete certificate.
    """
    rank = 3 if batched else 2
    if (
        a.ndim != rank
        or b.ndim != rank
        or a.dtype != torch.float64
        or b.dtype != a.dtype
        or a.device.type != "mps"
        or b.device != a.device
        or a.shape[:-2] != b.shape[:-2]
        or a.shape[-1] != b.shape[-2]
        or a.is_neg()
        or b.is_neg()
    ):
        return None
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if batched else 1
    # A GPU-to-host eligibility decision should not crowd out short GEMMs.
    if min(m, n, k) < 32 or batch * m * n * k < _MIN_WORK or not batch:
        return None
    mp, np, kp = (32 * ((v + 31) // 32) for v in (m, n, k))
    if max(batch * mp * kp, batch * kp * np, batch * m * n) >= 2**32:
        return None
    lib = _library()
    if lib is None:
        return None
    shape = (*a.shape[:-2], m, n)
    if out is not None and (
        out.dtype != torch.int64
        or out.device != a.device
        or tuple(out.shape) != shape
        or not out.is_contiguous()
    ):
        return None
    la, lb = _layout(a, batched), _layout(b, batched)
    layouts = la + lb
    flags = torch.empty(5, dtype=torch.int32, device=a.device)
    lib.exact_sample(flags, a, b, layouts, threads=(32,), group_size=(32,))
    possible, pa, pb, width_a, width_b = flags.cpu().tolist()
    if not possible:
        return None
    groups = min(2048, (max(a.numel(), b.numel()) + 255) // 256)
    ranges = torch.empty((2, groups, 2), dtype=torch.int64, device=a.device)
    metadata = torch.empty((2, 2), dtype=torch.int64, device=a.device)
    planes = 3 if max(width_a, width_b) > 8 else 1
    # Missing a rare low bit or wide value in the sample can change both
    # radix and scale. At most one retry uses the complete observed range.
    for _ in range(2):
        left = torch.empty(
            (planes, batch, mp, kp), dtype=torch.float16, device=a.device
        )
        right = torch.empty(
            (planes, batch, kp, np), dtype=torch.float16, device=a.device
        )
        for operand, (src, packed, layout, anchor) in enumerate(
            ((a, left, la, pa), (b, right, lb, pb))
        ):
            lib.exact_certify_pack(
                packed,
                ranges,
                src,
                layout,
                packed.shape[-1],
                packed.shape[-2],
                batch,
                anchor,
                groups,
                operand,
                planes,
                threads=(32 * groups,),
                group_size=(32,),
            )
        lib.exact_finish(metadata, ranges, groups, threads=(32, 2), group_size=(32, 1))
        cert = _certificate(metadata.cpu().tolist(), k)
        if cert is None:
            return None
        p, q, max_a, max_b = cert
        if not max_a or not max_b:
            if out is None:
                out = torch.zeros(shape, dtype=torch.int64, device=a.device)
            else:
                out.zero_()
            return out.view(torch.float64)
        actual_planes = 3 if max(max_a, max_b) > 255 else 1
        bound = k * max_a * max_b
        plain = actual_planes == 1 and bound <= 1 << 24
        if actual_planes == planes:
            if (pa, pb) == (p, q) or (plain and p <= pa <= p + 7 and q <= pb <= q + 7):
                break
        pa, pb, planes = p, q, actual_planes
    else:
        return None

    if out is None:
        out = torch.empty(shape, dtype=torch.int64, device=a.device)
    grid = (np // 32 * 128, mp // 32, batch)
    if plain:
        lib.exact_mma(
            out,
            left,
            right,
            m,
            n,
            mp,
            np,
            kp,
            pa + pb,
            threads=grid,
            group_size=(128, 1, 1),
        )
    else:
        if planes == 3:
            # Bounds for |low+high| also dominate both individual digit planes.
            a_limit = min(128, max_a) + (max_a + 128) // 256
            b_limit = min(128, max_b) + (max_b + 128) // 256
            kernel = lib.exact_panel_three
        else:
            a_limit, b_limit = max_a, max_b
            kernel = lib.exact_panel_one
        panel = min(kp, ((1 << 24) // (a_limit * b_limit) // 8) * 8)
        kernel(
            out,
            left,
            right,
            m,
            n,
            mp,
            np,
            kp,
            p + q,
            panel,
            batch,
            threads=grid,
            group_size=(128, 1, 1),
        )
    return out.view(torch.float64)
