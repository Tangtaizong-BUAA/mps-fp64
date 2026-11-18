import torch
from .kernels import bits, F64, I64, library


def bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    if weight.ndim != 2 or weight.dtype != F64 or mode not in (0, 1, 2):
        raise RuntimeError("invalid embedding bag table or reduction")
    if any(
        t.ndim != 1 or t.dtype not in (I64, torch.int32) or t.device != weight.device
        for t in (indices, offsets)
    ):
        raise RuntimeError(
            "indices and offsets must be integer vectors on the weight device"
        )
    size, width = weight.shape
    n = indices.numel()
    bags = offsets.numel() - int(include_last_offset)
    if bags < 0 or (not offsets.numel() and n):
        raise RuntimeError("embedding bag requires starting offsets")
    if max(size, width, n, bags, bags * width) >= 2**31:
        raise NotImplementedError("embedding bag exceeds signed Metal indices")
    if per_sample_weights is not None and (
        mode != 0
        or per_sample_weights.shape != indices.shape
        or per_sample_weights.dtype != F64
        or per_sample_weights.device != weight.device
    ):
        raise RuntimeError(
            "per-sample weights require sum mode and matching dtype and index shape"
        )
    idx, off = indices.to(I64).contiguous(), offsets.to(I64).contiguous()
    if idx.numel() and ((idx < 0) | (idx >= size)).any().item():
        raise IndexError("embedding bag index out of range")
    if (
        offsets.numel()
        and ((off[:1] != 0).any() | (off[1:] < off[:-1]).any() | (off > n).any()).item()
    ):
        raise RuntimeError(
            "embedding bag offsets must start at zero and be nondecreasing within input"
        )
    if padding_idx < -1 or padding_idx >= size:
        raise RuntimeError("padding index exceeds table size")
    integer_dtype = torch.promote_types(indices.dtype, offsets.dtype)
    output = torch.empty((bags, width), dtype=I64, device=weight.device)
    owners = torch.zeros(n, device=weight.device, dtype=I64)
    lengths = torch.zeros(bags, device=weight.device, dtype=I64)
    maxima = torch.zeros_like(output) if mode == 2 else lengths
    if bags and n:
        library().embedding_ranges(
            owners,
            lengths,
            idx,
            off,
            bags,
            n,
            int(include_last_offset),
            mode,
            padding_idx,
            threads=(bags,),
        )
        if output.numel():
            library().embedding_values(
                output,
                maxima,
                bits(weight).contiguous(),
                idx,
                off,
                bits(per_sample_weights).contiguous()
                if per_sample_weights is not None
                else idx,
                bags,
                width,
                n,
                int(include_last_offset),
                mode,
                padding_idx,
                int(per_sample_weights is not None),
                threads=(output.numel(),),
            )
    elif output.numel():
        output.zero_()
    return (
        output.view(F64),
        owners.to(integer_dtype),
        lengths.to(integer_dtype),
        maxima.to(integer_dtype),
    )


def dense_grad(grad, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False):
    idx = indices.reshape(-1).long()
    values = grad.reshape(idx.numel(), grad.shape[-1])
    if padding_idx != -1:
        keep = idx != padding_idx
        idx, values = idx[keep], values[keep]
    if scale_grad_by_freq:
        freq = torch.zeros(num_weights, device=idx.device, dtype=I64)
        freq.index_add_(0, idx, torch.ones_like(idx))
        values = values / freq.index_select(0, idx).double().unsqueeze(1)
    out = torch.zeros(num_weights, grad.shape[-1], device=grad.device, dtype=F64)
    return out.index_add(0, idx, values)


def bag_grad(
    grad,
    indices,
    owners,
    bag_size,
    maxima,
    num_weights,
    scale_grad_by_freq,
    mode,
    per_sample_weights=None,
    padding_idx=-1,
):
    idx, owners = indices.reshape(-1).long(), owners.reshape(-1).long()
    if mode == 2:
        valid = bag_size != 0
        active = maxima[valid].long()
        out = torch.zeros(num_weights, grad.shape[-1], device=grad.device, dtype=F64)
        return out.scatter_add(0, active, grad[valid])
    values = grad.index_select(0, owners)
    if mode == 1:
        values = values / bag_size.index_select(0, owners).clamp_min(
            1
        ).double().unsqueeze(1)
    if per_sample_weights is not None:
        values = values * per_sample_weights.reshape(-1, 1)
    return dense_grad(values, idx, num_weights, padding_idx, scale_grad_by_freq)
