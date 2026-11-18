// RN-even operations for the matrix-vector reduction. Its historical API
// rounds each product before summing it in a fixed 256-leaf tree.
using namespace metal;
namespace gemv_fp64 {
constant ulong unit = 1ul << 52;
constant ulong sign_bit = 1ul << 63;

__attribute__((noinline)) ulong general_add(ulong a, ulong b) { return __softfloat64_fadd(a, b, 0); }
__attribute__((noinline)) ulong general_mul(ulong a, ulong b) { return __softfloat64_fmul(a, b, 0); }

inline ulong mul(ulong a, ulong b) {
    int ea = int((a >> 52) & 2047) - 1023, eb = int((b >> 52) & 2047) - 1023;
    if (uint((ea + 256) | (eb + 256)) >= 512) return general_mul(a, b);
    auto product = softfloat64_internal::u128_mul_u64((a & (unit-1)) | unit, (b & (unit-1)) | unit);
    uint norm = uint(product.hi >> 41), shift = 52 + norm;
    ulong head = (product.hi << (64-shift)) | (product.lo >> shift);
    ulong half_ulp = 1ul << (shift-1), tail = product.lo & (2*half_ulp-1);
    head += tail > half_ulp || (tail == half_ulp && (head & 1));
    return ((a ^ b) & sign_bit) | ((ulong(ea+eb+1022+int(norm)) << 52) + head);
}

inline ulong add(ulong a, ulong b) {
    int ea = int((a >> 52) & 2047) - 1023, eb = int((b >> 52) & 2047) - 1023;
    if (uint((ea + 256) | (eb + 256)) >= 512) {
        if (((a | b) << 1) == 0) return a & b;
        if ((a << 1) == 0 && (b << 1) <= 0xffe0000000000000ul) return b;
        if ((b << 1) == 0 && (a << 1) <= 0xffe0000000000000ul) return a;
        return general_add(a, b);
    }
    int gap = abs(ea-eb);
    if (gap > 9) {
        // Alignment keeps guard, round and sticky bits. At this distance
        // cancellation can lower the leading exponent by at most one.
        ulong big = ea > eb ? a : b, small = ea > eb ? b : a;
        if (gap > 54) return big;
        ulong sig = ((big & (unit-1)) | unit) << 3;
        ulong low = ((small & (unit-1)) | unit) << 3;
        ulong aligned = (low >> gap) | ulong((low << (64-gap)) != 0);
        ulong sum = ((big ^ small) & sign_bit) ? sig - aligned : sig + aligned;
        int adjust = int(sum >> 56) - int(sum < (unit << 3));
        if (adjust == 1) sum = (sum >> 1) | (sum & 1);
        else if (adjust == -1) sum <<= 1;
        ulong head = sum >> 3;
        uint tail = uint(sum) & 7;
        head += tail > 4 || (tail == 4 && (head & 1));
        return (big & sign_bit) | ((ulong(max(ea, eb)+adjust+1022) << 52) + head);
    }
    // Both signed aligned significands and their exact sum fit in 64 bits.
    int base = min(ea, eb);
    ulong ma = ((a & (unit-1)) | unit) << uint(ea-base);
    ulong mb = ((b & (unit-1)) | unit) << uint(eb-base);
    ulong sa = as_type<ulong>(as_type<long>(a) >> 63);
    ulong sb = as_type<ulong>(as_type<long>(b) >> 63);
    ulong sum = ((ma ^ sa) - sa) + ((mb ^ sb) - sb);
    if (!sum) return 0ul;
    ulong sign = as_type<ulong>(as_type<long>(sum) >> 63);
    ulong mag = (sum ^ sign) - sign;
    uint top = 63 - uint(clz(mag));
    ulong head;
    if (top <= 52) head = mag << (52-top);
    else {
        uint shift = top-52, half_ulp = 1u << (shift-1);
        uint tail = uint(mag) & (2*half_ulp-1);
        head = mag >> shift;
        head += tail > half_ulp || (tail == half_ulp && (head & 1));
    }
    return (sign & sign_bit) | ((ulong(base+int(top)-52+1022) << 52) + head);
}
} // namespace gemv_fp64
