// Exact fused multiply-add for nearby normal exponents. Both operands of the
// addition fit in a single, unrounded 128-bit fixed point representation.
// The full-domain implementation handles all other inputs and deep cancellation.
inline ulong wide_nearby_fma(ulong a,ulong b,ulong c) {
    using namespace softfloat64_internal;
    int ea=int((a>>52)&2047)-1023, eb=int((b>>52)&2047)-1023, ec=int((c>>52)&2047)-1023;
    int delta=ec-ea-eb;
    if(abs(ea)>256 || abs(eb)>256 || abs(ec)>256 || delta < -50 || delta > 20)
        return __softfloat64_fma(a,b,c,0);
    constexpr ulong unit=1ul<<52, fraction=unit-1, signbit=1ul<<63;
    U128 product=u128_mul_u64(unit|(a&fraction),unit|(b&fraction));
    ulong mantissa=unit|(c&fraction);
    uint align=uint(52+delta);
    U128 term = align<64 ? U128{mantissa>>(64-align),mantissa<<align} : U128{mantissa<<(align-64),0};
    ulong sign=(a^b)&signbit;
    U128 exact;
    if(sign==(c&signbit)) exact=u128_add(product,term);
    else if(u128_cmp(product,term)>=0) exact=u128_sub(product,term);
    else {exact=u128_sub(term,product); sign=c&signbit;}
    if(!exact.hi) return __softfloat64_fma(a,b,c,0);
    uint high=uint(exact.hi>>32);
    uint top=high?127u-clz(high):95u-clz(uint(exact.hi));
    uint shift=top-52;
    ulong sig,halfway,discarded;
    if(shift<=64) {
        sig=shift==64 ? exact.hi : (exact.hi<<(64-shift))|(exact.lo>>shift);
        halfway=1ul<<(shift-1);
        discarded=exact.lo&((halfway<<1)-1);
    } else {
        uint hshift=shift-64;
        sig=exact.hi>>hshift;
        // One extra low bit represents all bits in the discarded low word.
        halfway=1ul<<hshift;
        discarded=((exact.hi&(halfway-1))<<1)|ulong(exact.lo!=0);
    }
    sig+=discarded>halfway || (discarded==halfway && (sig&1));
    int exponent=ea+eb-104+int(top);
    // Addition naturally carries a rounded 54th significand bit into the exponent.
    return sign|((ulong(exponent+1022)<<52)+sig);
}

// A decoded binary64 term is mantissa * 2^(exponent - 52). The implicit
// leading bit is also installed for zeros, subnormals and nonfinite values,
// so unpacking followed by packing always preserves the exact input bits.
// Only the normal, bounded-exponent window below uses it as a numeric value.
struct DotTerm {
    long mantissa;
    int exponent;
};

inline DotTerm unpack_dot_term(ulong bits) {
    long sign = as_type<long>(bits) >> 63;
    long magnitude = long((bits & 0xffffffffffffful) | (1ul << 52));
    return {(magnitude ^ sign) - sign, int((bits >> 52) & 2047) - 1023};
}

inline ulong pack_dot_term(DotTerm x) {
    long sign = x.mantissa >> 63;
    ulong magnitude = ulong((x.mantissa ^ sign) - sign);
    // An exact-power-of-two result can carry a 54th bit after rounding.
    // Adding the magnitude folds that carry into the packed exponent.
    return ((ulong(x.exponent + 1022) << 52) + magnitude)
         | (as_type<ulong>(sign) & (1ul << 63));
}

__attribute__((noinline))
DotTerm wide_dot_fma(DotTerm a, DotTerm b, DotTerm c) {
    return unpack_dot_term(wide_nearby_fma(
        pack_dot_term(a), pack_dot_term(b), pack_dot_term(c)));
}

inline DotTerm dot_fma(DotTerm a, DotTerm b, DotTerm c) {
    int product_exponent = a.exponent + b.exponent;
    int delta = c.exponent - product_exponent;
    // These bounds keep the result away from overflow and subnormal
    // rounding. The fallback also covers infinities, NaNs and signed zeros.
    uint exponent_range = uint((a.exponent + 256) | (b.exponent + 256)
                               | (c.exponent + 256));
    if (exponent_range >= 512 || uint(delta + 8) > 62)
        return wide_dot_fma(a, b, c);

    // Split each signed integer into a signed high limb and an unsigned
    // low limb. In units of 2^(product_exponent - 60), product is the
    // mathematical floor of the exact product, with its low 44 bits omitted.
    int ah = int(a.mantissa >> 32), bh = int(b.mantissa >> 32);
    uint al = uint(a.mantissa), bl = uint(b.mantissa);
    ulong low_product = ulong(al) * bl;
    long cross = long(ah) * long(bl) + long(bh) * long(al) + long(low_product >> 32);
    long high_product = long(ah) * bh;
    long product = as_type<long>(as_type<ulong>(high_product) << 20) + (cross >> 12);

    // Rescale to units of 2^(max(product_exponent, c.exponent) - 60).
    // The addend is integral in these units. Floor commutes with integer
    // addition even when the product is negative, so no carry is lost.
    uint right = uint(max(0, delta)), left = uint(8 + min(0, delta));
    long floor_sum = (product >> right)
                   + as_type<long>(as_type<ulong>(c.mantissa) << left);
    int high = int(floor_sum >> 32);
    uint leading = as_type<uint>(high ^ (high >> 31));
    if (leading < (1u << 21))
        return wide_dot_fma(a, b, c);  // cancellation needs more product bits

    // At least one integer bit will be discarded. The entire remaining
    // fractional tail therefore contributes exactly one sticky bit. Apply
    // nearest-even to the arithmetic floor on either side of zero.
    uint cut = 11u - clz(leading);
    uint halfway = 1u << (cut - 1);
    uint remainder = uint(floor_sum) & (2 * halfway - 1);
    bool sticky = uint(low_product) || (uint(cross) & 0xfffu)
               || (as_type<ulong>(product) & ((1ul << right) - 1));
    long mantissa = floor_sum >> cut;
    mantissa += remainder > halfway || (remainder == halfway && ((mantissa & 1) || sticky));
    return {mantissa, max(product_exponent, c.exponent) - 8 + int(cut)};
}

inline ulong nearby_fma(ulong a, ulong b, ulong c) {
    return pack_dot_term(dot_fma(unpack_dot_term(a), unpack_dot_term(b), unpack_dot_term(c)));
}
kernel void test_nearby_fma(device ulong *out,const device ulong *a,const device ulong *b,
                            const device ulong *c,uint i [[thread_position_in_grid]]) {
    out[i]=nearby_fma(a[i],b[i],c[i]);
}
