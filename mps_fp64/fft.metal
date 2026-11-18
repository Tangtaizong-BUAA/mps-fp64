inline ulong2 complex_product(ulong2 a, ulong2 b) {
    constexpr ulong sign = 1ul << 63;
    ulong real = nearby_fma(a.x, b.x, __softfloat64_fmul(a.y, b.y, 0) ^ sign);
    ulong imag = nearby_fma(a.x, b.y, __softfloat64_fmul(a.y, b.x, 0));
    return ulong2(real, imag);
}

inline ulong2 complex_sum(ulong2 a, ulong2 b) {
    return ulong2(__softfloat64_fadd(a.x, b.x, 0), __softfloat64_fadd(a.y, b.y, 0));
}

inline ulong2 complex_difference(ulong2 a, ulong2 b) {
    return ulong2(__softfloat64_fsub(a.x, b.x, 0), __softfloat64_fsub(a.y, b.y, 0));
}

// Reduce the phase in integer arithmetic before evaluating sine and cosine.
// delta * pi/(2*n) always lies in [-pi/4,pi/4], including for prime lengths.
kernel void fft_angles(device ulong *phase, constant uint &n, constant ulong &scale,
                       uint k [[thread_position_in_grid]]) {
    long four_k = 4l * k;
    long quadrant = (four_k + n/2) / n;
    long delta = four_k - quadrant * n;
    phase[k] = __softfloat64_fmul(__softfloat64_cvt_i64_to_f64(delta, 0), scale, 0);
}

kernel void fft_roots(device ulong2 *out, const device ulong *cosines,
                      const device ulong *sines, constant uint &n, constant uint &forward,
                      uint k [[thread_position_in_grid]]) {
    constexpr ulong sign = 1ul << 63;
    uint q = uint((4ul*k + n/2) / n) & 3u;
    ulong c = cosines[k], s = sines[k];
    ulong2 root = (q & 1u) ? ulong2(s ^ sign, c) : ulong2(c, s);
    if (q & 2u) root ^= sign;
    if (forward) root.y ^= sign;
    if ((root.x << 1) == 0) root.x = 0;
    if ((root.y << 1) == 0) root.y = 0;
    out[k] = root;
}

kernel void bit_reverse(device ulong2 *out, const device ulong2 *a, constant uint &width,
                        constant uint &stages, uint i [[thread_position_in_grid]]) {
    uint k = i % width, j = reverse_bits(k) >> (32-stages);
    out[i] = a[i-k+j];
}

kernel void butterflies(device ulong2 *data, const device ulong2 *roots, constant uint &width,
                        constant uint &span, uint i [[thread_position_in_grid]]) {
    uint t = i % (span/2), j = (i/(span/2))*span + t;
    ulong2 a = data[j], b = complex_product(data[j+span/2], roots[t*(width/span)]);
    data[j] = complex_sum(a, b);
    data[j+span/2] = complex_difference(a, b);
}

// One group owns one row. A barrier separates consecutive radix-2 stages.
kernel void fft_local(device ulong2 *out, const device ulong2 *src, const device ulong2 *roots,
                      constant uint &width, constant uint &stages, uint lane [[thread_index_in_threadgroup]],
                      uint row [[threadgroup_position_in_grid]], uint lanes [[threads_per_threadgroup]]) {
    threadgroup ulong2 values[1024];
    ulong base = ulong(row)*width;
    for (uint j=lane; j<width; j+=lanes)
        values[j] = src[base+(reverse_bits(j) >> (32-stages))];
    for (uint span=2; span<=width; span*=2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k=lane; k<width/2; k+=lanes) {
            uint t = k & (span/2-1), j = (k/(span/2))*span + t;
            ulong2 a = values[j], b = complex_product(values[j+span/2], roots[t*(width/span)]);
            values[j] = complex_sum(a,b);
            values[j+span/2] = complex_difference(a,b);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j=lane; j<width; j+=lanes) out[base+j] = values[j];
}

inline ulong2 fft_direct_value(const device ulong2 *row, const device ulong2 *roots,
                               uint n, uint k, uint root_stride) {
    ulong2 acc = 0;
    for (uint j=0; j<n; ++j)
        acc = complex_sum(acc, complex_product(row[j], roots[root_stride*(ulong(k)*j % n)]));
    return acc;
}

kernel void simple_dft(device ulong2 *out, const device ulong2 *a, const device ulong2 *roots,
                       constant uint &width, uint i [[thread_position_in_grid]]) {
    uint k = i % width;
    out[i] = fft_direct_value(a+i-k, roots, width, k, 1);
}

kernel void fft_chirp_plan(device ulong2 *chirp, device ulong2 *mirror, const device ulong2 *roots,
                           constant uint &n, constant uint &m, uint k [[thread_position_in_grid]]) {
    uint j = min(k, m-k);
    if (j >= n) { mirror[k] = 0; return; }
    ulong2 phase = roots[(ulong(j)*j) % (2ul*n)];
    if (k < n) chirp[k] = phase;
    phase.y ^= 1ul << 63;
    mirror[k] = phase;
}

kernel void fft_modulate(device ulong2 *out, const device ulong2 *src, const device ulong2 *chirp,
                         device atomic_uint *extreme, constant uint &n, constant uint &m,
                         uint i [[thread_position_in_grid]]) {
    uint row = i/m, k = i%m;
    if (k >= n) { out[i] = 0; return; }
    ulong2 value = src[ulong(row)*n+k];
    ulong2 magnitude = value & 0x7ffffffffffffffful;
    // A convolution can overflow or underflow internally even when the DFT is representable.
    if (any(magnitude >= (1983ul<<52)) || any((magnitude != 0) & (magnitude < (63ul<<52))))
        atomic_store_explicit(extreme+row, 1u, memory_order_relaxed);
    out[i] = complex_product(value, chirp[k]);
}

kernel void fft_multiply(device ulong2 *data, const device ulong2 *filter,
                         constant uint &m, uint i [[thread_position_in_grid]]) {
    data[i] = complex_product(data[i], filter[i%m]);
}

kernel void fft_demodulate(device ulong2 *out, const device ulong2 *convolved, const device ulong2 *src,
                           const device ulong2 *chirp, const device ulong2 *roots, const device uint *extreme,
                           constant uint &n, constant uint &m, uint i [[thread_position_in_grid]]) {
    uint row = i/n, k = i%n;
    out[i] = extreme[row] ? fft_direct_value(src+ulong(row)*n, roots, n, k, 2)
                          : complex_product(convolved[ulong(row)*m+k], chirp[k]);
}
