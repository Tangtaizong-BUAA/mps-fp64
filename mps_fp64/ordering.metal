// Ordering the integer encoding avoids float conversions. Equal zeros share a
// key and all NaNs share the largest key; the raw value always travels intact.
inline ulong fp_order_key(ulong value) {
    if (fp_nan(value)) return ~0ul;
    if ((value << 1) == 0) value = 0;
    return (value >> 63) ? ~value : (value ^ (1ul << 63));
}
inline ulong extreme_score(ulong value, bool maximum) {
    const ulong key = fp_order_key(value);
    return maximum || fp_nan(value) ? key : ~key;
}
inline ulong order_shuffle_down(ulong value, ushort offset) {
    uint2 words = as_type<uint2>(value);
    return as_type<ulong>(simd_shuffle_down(words, offset));
}

// Keep the earlier value for equal keys and propagate the earliest NaN.
// ATen's all-elements reduction orders -0 before +0; its axis kernel ties them.
inline ulong aminmax_pick(ulong left, ulong right, bool maximum, bool order_zero) {
    if (fp_nan(left)) return left;
    if (fp_nan(right)) return right;
    ulong a = fp_order_key(left), b = fp_order_key(right);
    if (order_zero) {
        if (left == (1ul << 63)) --a;
        if (right == (1ul << 63)) --b;
    }
    return (maximum ? b > a : b < a) ? right : left;
}

kernel void aminmax_blocks(device ulong *mins, device ulong *maxs,
                           const device ulong *low, const device ulong *high,
                           constant long *params, uint gid [[thread_position_in_grid]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint width = uint(params[0]), chunks = uint(params[1]);
    const uint items = uint(params[2]), group = gid >> 5;
    const uint start = (group % chunks) * (32 * items);
    const ulong base = ulong(group / chunks) * width;
    const uint offset = min(start + lane * items, width - 1);
    const bool first = bool(params[3]), order_zero = bool(params[4]);
    ulong lo = low[base + offset], hi = first ? lo : high[base + offset];
    const uint count = min(items, width - offset);
    for (uint j = 1; j < count; ++j) {
        const ulong x = low[base + offset + j];
        lo = aminmax_pick(lo, x, false, order_zero);
        hi = aminmax_pick(hi, first ? x : high[base + offset + j], true, order_zero);
    }
    // Increasing step keeps each partial result in original left-to-right order.
    for (ushort step = 1; step < 32; step <<= 1) {
        lo = aminmax_pick(lo, order_shuffle_down(lo, step), false, order_zero);
        hi = aminmax_pick(hi, order_shuffle_down(hi, step), true, order_zero);
    }
    if (lane == 0) { mins[group] = lo; maxs[group] = hi; }
}

// A small row (or a tile of a longer row) is sorted by a stable bitonic
// network. Indices break ties, including equal zeros and distinct NaN payloads.
// Packing short rows into separate SIMD groups avoids threadgroup barriers.
kernel void sort_simd_tile(device ulong *out, device long *ids, const device ulong *src,
                           constant uint &width, constant uint &tiles, constant uint &descending,
                           uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint tile = gid >> 5, row = tile / tiles, col = (tile % tiles) * 32 + lane;
    const ulong base = ulong(row) * width;
    const ulong raw = col < width ? src[base + col] : 0ul;
    ulong key = col < width ? fp_order_key(raw) : ~0ul;
    if (descending && col < width) key = ~key;
    uint id = col < width ? col : ~0u;
    for (ushort size = 2; size <= 32; size <<= 1) {
        for (ushort dist = size >> 1; dist; dist >>= 1) {
            const ulong other = as_type<ulong>(simd_shuffle_xor(as_type<uint2>(key), dist));
            const uint other_id = simd_shuffle_xor(id, dist);
            const bool lower = (lane & dist) == 0;
            const bool want_minimum = lower == ((lane & size) == 0);
            const bool other_first = other < key || (other == key && other_id < id);
            if (want_minimum == other_first) { key = other; id = other_id; }
        }
    }
    if (col < width) { out[base + col] = src[base + id]; ids[base + col] = long(id); }
}

// Longer tiles exchange keys through threadgroup memory only when a partner
// crosses SIMD groups. All shorter strides stay in registers. The secondary
// index key makes every stage stable, even across a partial tile.
kernel void sort_group_tile(device ulong *out, device long *ids, const device ulong *src,
                            constant uint &width, constant uint &tiles, constant uint &descending,
                            uint group [[threadgroup_position_in_grid]], ushort lid [[thread_index_in_threadgroup]],
                            uint tile_size [[threads_per_threadgroup]]) {
    threadgroup ulong keys[256];
    threadgroup uint original[256];
    const uint row = group / tiles, col = (group % tiles) * tile_size + lid;
    const ulong base = ulong(row) * width;
    const ulong raw = col < width ? src[base + col] : 0ul;
    ulong key = col < width ? fp_order_key(raw) : ~0ul;
    if (descending && col < width) key = ~key;
    uint id = col < width ? col : ~0u;
    for (ushort size = 2; size <= tile_size; size <<= 1) {
        for (ushort dist = size >> 1; dist; dist >>= 1) {
            ulong other;
            uint other_id;
            if (dist < 32) {
                other = as_type<ulong>(simd_shuffle_xor(as_type<uint2>(key), dist));
                other_id = simd_shuffle_xor(id, dist);
            } else {
                keys[lid] = key; original[lid] = id;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                other = keys[lid ^ dist]; other_id = original[lid ^ dist];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const bool want_minimum = ((lid & dist) == 0) == ((lid & size) == 0);
            const bool other_first = other < key || (other == key && other_id < id);
            if (want_minimum == other_first) { key = other; id = other_id; }
        }
    }
    if (col < width) { out[base + col] = src[base + id]; ids[base + col] = long(id); }
}

// Rank an item in a merge of two sorted runs. A lower bound for the left
// run and an upper bound for the right run preserve the original tie order.
kernel void merge_runs(device ulong *dest, device long *dest_ids, const device ulong *src,
                       const device long *ids, constant uint &width, constant uint &run,
                       constant uint &descending, uint i [[thread_position_in_grid]]) {
    uint col = i % width, base = i - col, start = (col / (2 * run)) * (2 * run);
    uint middle = min(start + run, width);
    bool left = col < middle;
    uint first = left ? middle : start, lo = first, hi = left ? min(middle + run, width) : middle;
    ulong value = src[i], key = fp_order_key(value);
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        ulong other = fp_order_key(src[base + mid]);
        bool before = descending ? (left ? other > key : other >= key) : (left ? other < key : other <= key);
        if (before) lo = mid + 1; else hi = mid;
    }
    uint out = base + start + (col - (left ? start : middle)) + (lo - first);
    dest[out] = value; dest_ids[out] = run == 1 ? long(col) : ids[i];
}

kernel void row_extremum(device ulong *vals, device long *indices, const device ulong *x,
                         constant uint &width, constant uint &maximum, uint row [[thread_position_in_grid]]) {
    const ulong base = ulong(row)*width;
    ulong best = x[base], score = extreme_score(best, maximum);
    uint index = 0;
    for (uint k = 1; k < width; ++k) {
        ulong value = x[base+k], rank = extreme_score(value, maximum);
        if (rank > score) { best = value; score = rank; index = k; }
    }
    vals[row] = best; indices[row] = long(index);
}

// Each SIMD group reduces a contiguous chunk. Lower original indices win
// score ties at every level, giving first-occurrence semantics across chunks.
kernel void extreme_blocks(device ulong *vals, device long *indices, const device ulong *x,
                           const device long *prior_ids, constant uint &width, constant uint &chunks,
                           constant uint &span, constant uint &maximum, constant uint &first,
                           uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint chunk = gid >> 5, row = chunk / chunks, start = (chunk % chunks) * span;
    const ulong base = ulong(row) * width;
    uint id = ~0u;
    ulong score = 0, best = 0;
    for (uint offset = lane; offset < min(width - start, span); offset += 32) {
        const uint k = start + offset;
        const ulong value = x[base + k], rank = extreme_score(value, maximum);
        const uint candidate = first ? k : uint(prior_ids[base + k]);
        if (rank > score || (rank == score && candidate < id)) {
            id = candidate; best = value; score = rank;
        }
    }
    for (ushort step = 16; step; step >>= 1) {
        const ulong other = order_shuffle_down(score, step), value = order_shuffle_down(best, step);
        const uint candidate = simd_shuffle_down(id, step);
        if (other > score || (other == score && candidate < id)) {
            id = candidate; best = value; score = other;
        }
    }
    if (lane == 0) { vals[chunk] = best; indices[chunk] = long(id); }
}
