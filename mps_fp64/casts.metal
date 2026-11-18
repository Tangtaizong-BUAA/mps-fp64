template<uint E,uint F> inline ushort narrow_float(ulong x) {
    constexpr int bias=(1<<(E-1))-1;
    constexpr uint top=(1u<<E)-1;
    uint sign=uint(x>>63)<<(E+F);
    uint source_exp=uint((x>>52)&2047);
    ulong frac=x&0xffffffffffffful;
    if(source_exp==2047) return ushort(sign|(top<<F)|(frac?(1u<<(F-1)):0));
    if(!source_exp) return ushort(sign);
    int exponent=int(source_exp)-1023;
    if(exponent>bias) return ushort(sign|(top<<F));
    int shift=52-int(F)+max(0,1-bias-exponent);
    if(shift>53) return ushort(sign);
    ulong sig=(1ul<<52)|frac;
    ulong result=sig>>shift;
    ulong half_ulp=1ul<<(shift-1), tail=sig&((half_ulp<<1)-1);
    result+=tail>half_ulp || (tail==half_ulp && (result&1));
    uint base=uint(max(0,exponent+bias-1))<<F;
    return ushort(sign|(base+uint(result)));
}
template<uint E,uint F> inline ulong widen_float(ushort x) {
    constexpr int bias=(1<<(E-1))-1;
    constexpr uint top=(1u<<E)-1;
    ulong sign=ulong(x>>(E+F))<<63;
    uint exp=(x>>F)&top, frac=x&((1u<<F)-1);
    if(exp==top) return sign|0x7ff0000000000000ul|(frac?(ulong(frac)<<(52-F))|(1ul<<51):0ul);
    if(!exp && !frac) return sign;
    int exponent=int(exp)-bias;
    if(!exp) {
        int shift=int(F)-(31-int(clz(frac)));
        frac=(frac<<shift)&((1u<<F)-1);
        exponent=1-bias-shift;
    }
    return sign|(ulong(exponent+1023)<<52)|(ulong(frac)<<(52-F));
}
#define CAST(NAME, DST, SRC, EXPR) \
kernel void NAME(device DST *out,const device SRC *src,uint i [[thread_position_in_grid]]) {out[i]=EXPR;}
CAST(to_bool,bool,ulong,(src[i]<<1)!=0)
CAST(to_i64,long,ulong,__softfloat64_cvt_f64_to_i64(src[i],3))
CAST(to_i32,int,ulong,int(clamp(__softfloat64_cvt_f64_to_i64(src[i],3),-2147483648l,2147483647l)))
CAST(to_f16,ushort,ulong,(narrow_float<5,10>(src[i])))
CAST(to_bf16,ushort,ulong,(narrow_float<8,7>(src[i])))
CAST(from_f16,ulong,ushort,(widen_float<5,10>(src[i])))
CAST(from_bf16,ulong,ushort,(widen_float<8,7>(src[i])))

// Exactly the same 27+26 random bits as the two public MPS generator draws.
kernel void uniform_from_words(device ulong *out, const device ulong *hi, const device ulong *lo,
                               uint i [[thread_position_in_grid]]) {
    ulong word = (hi[i]<<26) | lo[i];
    out[i] = word ? __softfloat64_cvt_i64_to_f64(long(word),0) - (53ul<<52) : 0;
}

kernel void linspace64(device ulong *out, constant ulong &start, constant ulong &end, constant uint &n,
                       uint i [[thread_position_in_grid]]) {
    if (n==1) { out[i]=start; return; }
    ulong step=__softfloat64_fdiv(__softfloat64_fsub(end,start,0),__softfloat64_cvt_i64_to_f64(n-1,0),0);
    bool lower=i<n/2;
    ulong count=__softfloat64_cvt_i64_to_f64(lower?i:n-1-i,0);
    out[i]=nearby_fma(lower?count:count^(1ul<<63),step,lower?start:end);
}
