// Constant-valued data uses ±0.5 for histogramdd. The legacy histc range
// expands by at least one representable value on either side when possible.
kernel void histogram_expand(device ulong *out, const device ulong *bounds,
                             constant uint &legacy, uint i [[thread_position_in_grid]]) {
    ulong lo = bounds[2*i], hi = bounds[2*i+1];
    if (__softfloat64_feq(lo, hi)) {
        ulong step = legacy ? 0x3ff0000000000000ul : 0x3fe0000000000000ul;
        ulong a = __softfloat64_fsub(lo, step, 0), b = __softfloat64_fadd(hi, step, 0);
        if (legacy) {
            a = fp_min(a, fp_nextafter(lo, 0xffeffffffffffffful));
            b = fp_max(b, fp_nextafter(hi, 0x7feffffffffffffful));
        }
        lo = a;
        hi = b;
    }
    out[2*i] = lo;
    out[2*i+1] = hi;
}

inline uint histc_index(ulong x, ulong lo, ulong hi, uint bins) {
    if (__softfloat64_feq(x, hi)) return bins - 1;
    ulong delta = __softfloat64_fsub(x, lo, 0), span = __softfloat64_fsub(hi, lo, 0);
    ulong count = __softfloat64_cvt_i64_to_f64(bins, 0);
    ulong numerator = __softfloat64_fmul(delta, count, 0);
    ulong ratio;
    if ((span << 1) == 0xffe0000000000000ul) {
        // Both endpoints may be finite even when their difference overflows.
        constexpr ulong half_bits = 0x3fe0000000000000ul;
        ulong half_lo = __softfloat64_fmul(lo, half_bits, 0);
        delta = __softfloat64_fsub(__softfloat64_fmul(x, half_bits, 0), half_lo, 0);
        span = __softfloat64_fsub(__softfloat64_fmul(hi, half_bits, 0), half_lo, 0);
        ratio = __softfloat64_fmul(__softfloat64_fdiv(delta, span, 0), count, 0);
    } else if ((numerator << 1) == 0xffe0000000000000ul) {
        ratio = __softfloat64_fmul(__softfloat64_fdiv(delta, span, 0), count, 0);
    } else {
        // histc uses this precise order and truncation. A value one ULP
        // away from an edge can therefore differ from torch.histogram.
        ratio = __softfloat64_fdiv(numerator, span, 0);
    }
    return min(bins - 1, uint(__softfloat64_cvt_f64_to_i64(ratio, 3)));
}

inline uint histogram_key(const device ulong *point, const device ulong *edges,
                          constant long *offsets, uint dimensions, uint invalid, uint legacy_bins) {
    uint key = 0;
    for (uint d = 0; d < dimensions; ++d) {
        const device ulong *e = edges + offsets[d];
        uint num_edges = uint(offsets[d+1] - offsets[d]);
        ulong x = point[d], first = e[0], last = e[num_edges-1];
        if (!__softfloat64_fge(x, first) || !__softfloat64_fle(x, last)) return invalid;
        if (legacy_bins) return histc_index(x, first, last, legacy_bins);
        uint bins = num_edges - 1;
        uint lo = 1, hi = num_edges;
        while (lo < hi) {
            uint mid = lo + (hi-lo)/2;
            if (__softfloat64_flt(x, e[mid])) hi = mid;
            else lo = mid + 1;
        }
        // All bins are left-inclusive. Only the final bin includes its right edge.
        key = key * bins + min(bins - 1, lo - 1);
    }
    return key;
}

kernel void histogram_keys(device long *out, const device ulong *points, const device ulong *edges,
                           constant long *offsets, constant uint &dimensions, constant uint &invalid,
                           uint i [[thread_position_in_grid]]) {
    out[i] = long(histogram_key(points + ulong(i)*dimensions, edges, offsets, dimensions, invalid, 0));
}

kernel void histogram_counts(device atomic_uint *out, const device ulong *points, const device ulong *edges,
                             constant long *offsets, constant uint &dimensions, constant uint &invalid,
                             constant uint &legacy_bins, uint i [[thread_position_in_grid]]) {
    uint key = histogram_key(points + ulong(i)*dimensions, edges, offsets, dimensions, invalid, legacy_bins);
    if (key != invalid) atomic_fetch_add_explicit(out + key, 1u, memory_order_relaxed);
}

kernel void histogram_count_double(device ulong *out, const device uint *counts,
                                   uint i [[thread_position_in_grid]]) {
    out[i] = __softfloat64_cvt_i64_to_f64(long(counts[i]), 0);
}

kernel void histogram_density(device ulong *hist, const device ulong *total, const device ulong *edges,
                              constant long *offsets, constant long *strides, constant uint &dimensions,
                              uint i [[thread_position_in_grid]]) {
    ulong value = __softfloat64_fdiv(hist[i], total[0], 0);
    for (uint d = 0; d < dimensions; ++d) {
        uint count = uint(offsets[d+1] - offsets[d] - 1);
        uint j = uint(offsets[d]) + (i / uint(strides[d])) % count;
        value = __softfloat64_fdiv(value, __softfloat64_fsub(edges[j+1], edges[j], 0), 0);
    }
    hist[i] = value;
}

kernel void histogram_isin(device bool *out, const device ulong *x, const device ulong *candidates,
                           constant uint &n, constant uint &sorted, constant uint &invert,
                           uint i [[thread_position_in_grid]]) {
    ulong value = x[i];
    bool found = false;
    if (sorted) {
        uint lo = 0, hi = n;
        while (lo < hi) {
            uint mid = lo + (hi-lo)/2;
            if (__softfloat64_flt(candidates[mid], value)) lo = mid + 1;
            else hi = mid;
        }
        found = lo < n && __softfloat64_feq(candidates[lo], value);
    } else {
        for (uint j = 0; j < n && !found; ++j)
            found = __softfloat64_feq(candidates[j], value);
    }
    out[i] = found != bool(invert);
}
