// Short reductions with many independent output elements. A lane preserves the
// input k order and the single nearest-even rounding at every fused operation.
namespace skinny_gemm {
struct Term {
    long significand;  // signed 53-bit significand shifted left ten bits
    int exponent;
};

inline Term unpack(ulong bits) {
    long sign = as_type<long>(bits) >> 63;
    long magnitude = as_type<long>(((bits << 10) & 0x3ffffffffffffc00ul) |
                                   0x4000000000000000ul);
    return {(magnitude ^ sign) - sign, int((bits >> 52) & 2047) - 1023};
}

inline ulong pack(Term x) {
    long sign = x.significand >> 63;
    ulong magnitude = as_type<ulong>((x.significand ^ sign) - sign) >> 10;
    return (as_type<ulong>(sign) & (1ul << 63)) |
           ((ulong(x.exponent + 1022) << 52) + magnitude);
}

inline Term normalized(long significand, int exponent) {
    if (significand == (1l << 53) || significand == -(1l << 53)) {
        significand >>= 1;
        ++exponent;
    }
    return {as_type<long>(as_type<ulong>(significand) << 10), exponent};
}

__attribute__((noinline)) inline ulong full_first(ulong a, ulong b) {
    // In particular, fma(-0, nonzero, +0) has a different zero sign from mul.
    return __softfloat64_fma(a, b, 0, 0);
}

__attribute__((noinline)) inline Term full_step(Term a, Term b, Term c) {
    return unpack(wide_nearby_fma(pack(a), pack(b), pack(c)));
}

inline ulong outer_product(ulong a, ulong b) {
    const uint ea = uint((a >> 52) & 2047), eb = uint((b >> 52) & 2047);
    const int exponent = int(ea + eb) - 1023;
    if (max(ea - 1, eb - 1) >= 2046 || uint(exponent - 1) >= 2044)
        return full_first(a, b);

    const ulong x = (a << 11) | (1ul << 63), y = (b << 11) | (1ul << 63);
    const ulong high = mulhi(x, y);
    const uint extra = uint(high >> 63), cut = 10 + extra;
    const uint halfway = 1u << (cut - 1);
    const uint remainder = uint(high) & (2 * halfway - 1);
    ulong significand = high >> cut;
    significand += remainder > halfway ||
                  (remainder == halfway && ((significand & 1) || x * y != 0));
    return ((a ^ b) & (1ul << 63)) |
           ((ulong(exponent - 1 + int(extra)) << 52) + significand);
}

inline Term first(Term a, Term b) {
    if (uint((a.exponent + 400) | (b.exponent + 400)) >= 800)
        return unpack(full_first(pack(a), pack(b)));

    const long high = mulhi(a.significand, b.significand);
    const int top = int(high >> 32);
    const uint cut = 11u - clz(as_type<uint>(top ^ (top >> 31)));
    const uint halfway = 1u << (cut - 1);
    const uint remainder = uint(high) & (2 * halfway - 1);
    long significand = high >> cut;
    significand += remainder > halfway ||
        (remainder == halfway && ((significand & 1) ||
         as_type<ulong>(a.significand) * as_type<ulong>(b.significand) != 0));
    return normalized(significand, a.exponent + b.exponent - 8 + int(cut));
}

inline Term step(Term a, Term b, Term c) {
    const int product_exponent = a.exponent + b.exponent;
    const int delta = c.exponent - product_exponent;
    const uint window = uint((a.exponent + 400) | (b.exponent + 400) | (c.exponent + 400));
    if (window >= 800) return full_step(a, b, c);

    // Signed high multiply yields floor(product) in units 2^(ea+eb-60).
    // Shift each summand to units 2^(max(ea+eb,ec)-60). A right shift of
    // 63 already produces its sign for both of these bounded significands.
    const long high = mulhi(a.significand, b.significand);
    const uint product_shift = uint(clamp(delta, 0, 63));
    const uint add_shift = uint(clamp(2 - min(0, delta), 2, 63));
    const long sum = (high >> product_shift) + (c.significand >> add_shift);
    const int top = int(sum >> 32);
    const uint leading = as_type<uint>(top ^ (top >> 31));
    if (leading < (1u << 21)) return full_step(a, b, c);

    const uint cut = 11u - clz(leading), halfway = 1u << (cut - 1);
    const uint remainder = uint(sum) & (2 * halfway - 1);
    // Both floored summands can have a fractional tail only if delta < -8.
    // Each tail is in [0,1), so their carry can alter the rounding decision
    // below halfway only at halfway-1.
    if (delta < -8 && remainder == halfway - 1) return full_step(a, b, c);

    long significand = sum >> cut;
    if (remainder > halfway) {
        ++significand;
    } else if (remainder == halfway) {
        const ulong a_bits = as_type<ulong>(a.significand);
        const ulong b_bits = as_type<ulong>(b.significand);
        const bool product_tail = product_shift == 63 ? high != 0 :
            (as_type<ulong>(high) & ((1ul << product_shift) - 1)) != 0;
        const bool add_tail = add_shift == 63 ? c.significand != 0 :
            (as_type<ulong>(c.significand) & ((1ul << add_shift) - 1)) != 0;
        significand += (significand & 1) || a_bits * b_bits != 0 || product_tail || add_tail;
    }
    return normalized(significand, max(product_exponent, c.exponent) - 8 + int(cut));
}
} // namespace skinny_gemm

kernel void skinny_outer(device ulong *out, const device ulong *a, const device ulong *b,
                         constant ulong *s, uint3 p [[thread_position_in_grid]]) {
    const ulong m = s[0], n = s[1];
    const ulong av = a[p.z * s[3] + p.y * s[4]];
    const ulong bv = b[p.z * s[6] + p.x * s[8]];
    out[(p.z * m + p.y) * n + p.x] = skinny_gemm::outer_product(av, bv);
}

kernel void skinny_pack_operands(device skinny_gemm::Term *out, const device ulong *a,
                                 const device ulong *b, constant ulong *s,
                                 uint i [[thread_position_in_grid]]) {
    const uint m = uint(s[0]), n = uint(s[1]), k = uint(s[2]), a_count = uint(s[9]);
    if (i < a_count) {
        const uint row = (i / k) % m, inner = i % k, batch = (i / k) / m;
        out[i] = skinny_gemm::unpack(a[batch * s[3] + row * s[4] + inner * s[5]]);
    } else {
        const uint j = i - a_count, col = j % n, inner = (j / n) % k, batch = (j / n) / k;
        out[i] = skinny_gemm::unpack(b[batch * s[6] + inner * s[7] + col * s[8]]);
    }
}

kernel void skinny_dot4(device ulong *out, const device skinny_gemm::Term *terms,
                        constant ulong *s, uint3 p [[thread_position_in_grid]]) {
    using namespace skinny_gemm;
    constexpr uint rows = 4;
    const uint m = uint(s[0]), n = uint(s[1]), k = uint(s[2]), row = p.y * rows;
    const device Term *a = terms + p.z * s[10] + ulong(row) * k;
    const device Term *b = terms + s[9] + p.z * s[11] + p.x;
    const ulong offset = (ulong(p.z) * m + row) * n + p.x;
    Term sum[rows];
    #pragma unroll
    for (uint r = 0; r < rows; ++r)
        sum[r] = first(a[ulong(min(r, m - 1 - row)) * k], b[0]);
    for (uint t = 1; t < k; ++t) {
        const Term bv = b[t * ulong(n)];
        #pragma unroll
        for (uint r = 0; r < rows; ++r)
            sum[r] = step(a[ulong(min(r, m - 1 - row)) * k + t], bv, sum[r]);
    }
    #pragma unroll
    for (uint r = 0; r < rows; ++r)
        if (row + r < m) out[offset + ulong(r) * n] = pack(sum[r]);
}

// Assign consecutive output elements, including adjacent small matrices, to
// consecutive lanes. A compile-time rank specializes each row/column address
// and unrolls all three or four fused steps.
template<uint N, bool strided>
inline void tiny_bmm(device ulong *out, const device ulong *a, const device ulong *b,
                     constant ulong *s, uint index) {
    const uint row = (index / N) % N, col = index % N, batch = index / (N * N);
    const ulong a0 = strided ? batch * s[0] + row * s[1] : ulong(batch) * N * N + row * N;
    const ulong b0 = strided ? batch * s[3] + col * s[5] : ulong(batch) * N * N + col;
    const ulong sa = strided ? s[2] : 1, sb = strided ? s[4] : N;
    using namespace skinny_gemm;
    Term acc = first(unpack(a[a0]), unpack(b[b0]));
    #pragma unroll
    for (uint t = 1; t < N; ++t)
        acc = step(unpack(a[a0 + t * sa]), unpack(b[b0 + t * sb]), acc);
    out[index] = pack(acc);
}

#define SMALL_BMM(NAME, N, STRIDED) \
kernel void NAME(device ulong *out, const device ulong *a, const device ulong *b, \
                 constant ulong *strides, uint i [[thread_position_in_grid]]) { \
    tiny_bmm<N, STRIDED>(out, a, b, strides, i); \
}
SMALL_BMM(skinny_bmm3, 3, false)
SMALL_BMM(skinny_bmm4, 4, false)
SMALL_BMM(skinny_bmm3_strided, 3, true)
SMALL_BMM(skinny_bmm4_strided, 4, true)
#undef SMALL_BMM
