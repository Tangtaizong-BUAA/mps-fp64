// The fast interval uses an exact dyadic argument and a fixed-point enclosure.
// If its two endpoints can round differently, use the full-range OpenLibm kernel.
// Inputs and exceptional results never pass through the hardware float unit.
namespace fast_math {
constant ulong sign_mask = 1ul << 63;
constant ulong one = 0x3ff0000000000000ul;

// x is positive, in [2^-17, 8].  r = (x - k/64) * 2^69 exactly, |r| <= 2^62.
inline long residual(ulong x, thread uint& k) {
    uint e = uint(x >> 52);
    ulong m = (x & ((1ul << 52) - 1)) | (1ul << 52);
    uint cut = 1069u - e;
    k = uint((m + (1ul << (cut - 1))) >> cut);
    return as_type<long>((m - (ulong(k) << cut)) << (e - 1006));
}

// Signed multiply with floor rounding.  Both helpers are used only where the
// scaled result fits in 64 bits.  Unsigned low-half arithmetic allows either sign.
inline long mul_q69(long a, long b) { return mulhi(a,b) >> 5; }
inline long mul_q62(long a, long b) {
    return as_type<long>((as_type<ulong>(mulhi(a,b)) << 2) |
                         ((as_type<ulong>(a) * as_type<ulong>(b)) >> 62));
}

// Q62 table head plus Q69 correction. Bound is an absolute radius in Q69 units.
// A small result or an ambiguous midpoint goes to the full-range implementation.
inline bool certified_pack(long head, long correction, uint bound, thread ulong& bits, int scale=0) {
    long hi = head + (correction >> 7);
    uint lo = uint(correction) & 127u;
    bool neg = hi < 0;
    ulong mag = neg ? ulong(-hi) - uint(lo != 0) : ulong(hi);
    lo = neg ? (-lo & 127u) : lo;
    if (mag < (1ul << 52)) return false;
    uint cut = 11 - uint(clz(mag));
    uint rest = uint(mag & ((1ul << cut) - 1)) * 128u + lo;
    uint midpoint = 64u << cut;
    if (uint(abs(int(rest)-int(midpoint))) <= bound) return false;
    ulong sig = (mag >> cut) + uint(rest > midpoint);
    bits = (ulong(neg) << 63) | ((ulong(1012 + int(cut) + scale) << 52) + sig);
    return true;
}

template<bool cosine>
__attribute__((noinline)) ulong trig_full(ulong x) {
    return cosine ? ol_s_cos::cos(Soft(x,true)).v : ol_s_sin::sin(Soft(x,true)).v;
}

template<bool cosine>
inline ulong trig(ulong x) {
    ulong ax = x & ~sign_mask;
    if (ax < 0x3e30000000000000ul) return cosine ? one : x;  // |x| < 2^-28.
    if (ax < 0x3ee0000000000000ul || ax > 0x4020000000000000ul) return trig_full<cosine>(x);
    uint k;
    long r = residual(ax,k);
    // Taylor coefficients are Q62. The largest omitted term is |r|^8/8!.
    // Horner's intermediate errors contract by |r| <= 1/128 at each step.
    constant TrigNode& node = trig_grid[k];
    constant long* even = cosine ? node.c : node.s;
    constant long* odd = cosine ? node.s : node.c;
    long odd_sign = cosine ? -1 : 1;
    long p = -odd_sign * odd[6];
    p = -even[5] + mul_q69(p,r);
    p = odd_sign * odd[4] + mul_q69(p,r);
    p = even[3] + mul_q69(p,r);
    p = -odd_sign * odd[2] + mul_q69(p,r);
    p = -even[1] + mul_q69(p,r);
    p = odd_sign * odd[0] + mul_q69(p,r);
    long base = even[0];
    long low = cosine ? node.c_low : node.s_low;
    long correction = low + mul_q62(p,r);
    ulong result;
    if (!certified_pack(base,correction,6,result)) return trig_full<cosine>(x);
    return result ^ (cosine ? 0ul : x & sign_mask);
}

// Coefficients a_i describe f(node+r)/2^scale.  The Q62 Horner sum for
// i>=1 has error <= 1.5/(1-1/128) Q62; multiplying by r gives <1.52 Q69.
// Include <1 Q69 for mul_q62 and <=0.5 for the table head's Q69 residual.
inline long polynomial_delta(constant TaylorNode& node, long r) {
    long p = node.a[9];
    #pragma clang loop unroll(full)
    for (int i=8; i>=1; --i) p = node.a[i] + mul_q69(p,r);
    return mul_q62(p,r);
}

enum OddKind { Tanh, Atan, Erf, Asinh };
template<OddKind kind> __attribute__((noinline)) ulong odd_full(ulong x) {
    if (kind == Tanh) return ol_s_tanh::tanh(Soft(x,true)).v;
    if (kind == Atan) return ol_s_atan::atan(Soft(x,true)).v;
    if (kind == Erf) return ol_s_erf::erf(Soft(x,true)).v;
    return ol_s_asinh::asinh(Soft(x,true)).v;
}

template<OddKind kind> inline ulong odd(ulong x) {
    ulong ax = x & ~sign_mask;
    if (kind != Erf && ax < 0x3e30000000000000ul) return x;
    if (ax < 0x3ee0000000000000ul || ax > 0x4020000000000000ul) return odd_full<kind>(x);
    uint k;
    long r = residual(ax,k);
    constant TaylorNode& node = kind == Tanh ? tanh_grid[k] : kind == Atan ? atan_grid[k] :
                                kind == Erf ? erf_grid[k] : asinh_grid[k];
    // The degree-9 Taylor remainders are bounded in FAST_MATH.md.
    long correction = node.low + polynomial_delta(node,r);
    ulong result;
    if (!certified_pack(node.a[0],correction,8,result,node.scale)) return odd_full<kind>(x);
    return result ^ (x & sign_mask);
}

template<bool minus_one> __attribute__((noinline)) ulong exp_full(ulong x) {
    return minus_one ? ol_s_expm1::expm1(Soft(x,true)).v : ol_e_exp::__ieee754_exp(Soft(x,true)).v;
}
template<bool minus_one> inline ulong exp_bits(ulong x) {
    ulong ax = x & ~sign_mask;
    if (ax < 0x3c80000000000000ul) return minus_one ? x : one; // |x| < 2^-55.
    if (ax < 0x3ee0000000000000ul || ax > 0x4020000000000000ul) return exp_full<minus_one>(x);
    uint k;
    long r = residual(ax,k);
    bool negative = x >> 63;
    constant ExpNode& n = exp_grid[negative ? 512-k : 512+k];
    long head = minus_one ? n.minus_one : n.poly.a[0];
    long low = minus_one ? n.minus_one_low : n.poly.low;
    long correction = low + polynomial_delta(n.poly,negative ? -r : r);
    ulong result;
    if (!certified_pack(head,correction,8,result,n.poly.scale)) return exp_full<minus_one>(x);
    return result;
}

enum LogKind { Ln, Log2, Log10 };
template<LogKind kind> __attribute__((noinline)) ulong log_full(ulong x) {
    if (kind == Ln) return ol_e_log::__ieee754_log(Soft(x,true)).v;
    if (kind == Log2) return ol_e_log2::__ieee754_log2(Soft(x,true)).v;
    return ol_e_log10::__ieee754_log10(Soft(x,true)).v;
}
template<LogKind kind> inline ulong log_bits(ulong x) {
    // log(x) = log(m) + e*log(2), with m in [1,2). Separate the exponent
    // first, so one 65-entry table covers every positive normal binade.
    uint biased = uint(x >> 52);
    if (biased-1u >= 2046u) return log_full<kind>(x);
    int e = int(biased)-1023;
    ulong m = one | (x & ((1ul<<52)-1));
    uint k;
    long r = residual(m,k);
    constant TaylorNode& node = kind == Ln ? log_grid[k-64] :
                                kind == Log2 ? log2_grid[k-64] : log10_grid[k-64];
    long head = node.a[0];
    long correction = node.low + polynomial_delta(node,r);
    uint scale = 0;
    if (e != 0) {
        uint ae = uint(abs(e));
        scale = 32-clz(ae);
        long mask = (1l << scale)-1;
        constant LogConstant& c = log_constants[kind];
        // c = (head + tail/2^18)*2^-62 to within 0.5 Q80. Splitting before
        // multiplication by e keeps integer products in their representable ranges.
        long rem = (c.head & mask) * ae;
        long eh = (c.head >> scale) * ae + (rem >> scale);
        long el = (((rem & mask) << 18) + long(c.tail) * ae) >> (scale+11);
        correction = ((((head & mask) << 7) + correction) >> scale) + (e<0 ? -el : el);
        head = (head >> scale) + (e<0 ? -eh : eh);
    }
    ulong result;
    if (!certified_pack(head,correction,10,result,int(scale))) return log_full<kind>(x);
    return result;
}
}
