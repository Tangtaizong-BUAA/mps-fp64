"""Large-output, short-reduction FP64 matrix products on resident MPS storage."""

import torch


def launch_skinny_mm(lib, out, a, b, batched):
    """Fill the preallocated contiguous output, or leave it to the general GEMM."""
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if batched else 1
    if not (1 <= k <= 16 and min(m, n) >= 128 and batch * m * n >= 2**18):
        return False
    # Two small dispatches amortize decoding when the reduction has several
    # terms. The single-term kernel needs only its output allocation.
    if k > 1 and batch * m * n < 2**20:
        return False
    a_batch = a.stride(0) if batched else 0
    b_batch = b.stride(0) if batched else 0
    layout = [m, n, k, a_batch, *a.stride()[-2:], b_batch, *b.stride()[-2:]]
    if k == 1:
        lib.skinny_outer(
            out, a, b, layout, threads=(n, m, batch), group_size=(128, 1, 1)
        )
        return True

    # Expanded batch inputs need just one decoded copy. Preserve all other
    # strides, including zero row/column strides and views with storage offsets.
    a_step = m * k if a_batch else 0
    b_step = k * n if b_batch else 0
    a_count = (batch if a_batch else 1) * m * k
    count = a_count + (batch if b_batch else 1) * k * n
    if count >= 2**32:
        return False
    layout.extend((a_count, a_step, b_step))
    terms = torch.empty((count, 2), dtype=torch.int64, device=out.device)
    lib.skinny_pack_operands(terms, a, b, layout, threads=count)
    lib.skinny_dot4(
        out, terms, layout, threads=(n, (m + 3) // 4, batch), group_size=(128, 1, 1)
    )
    return True


def launch_small_bmm(lib, out, a, b, strides):
    """Pack 3x3 or 4x4 output matrices across consecutive GPU lanes."""
    rank = a.shape[-1]
    suffix = "" if a.is_contiguous() and b.is_contiguous() else "_strided"
    kernel = getattr(lib, f"skinny_bmm{rank}{suffix}")
    kernel(out, a, b, strides, threads=out.numel(), group_size=32)
