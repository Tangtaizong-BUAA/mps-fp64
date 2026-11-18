// Include after fma.metal. 2D, contiguous A/B/out; launch a 16x8 threadgroup.
// Each accumulator is updated once per k, always in ascending order.
struct Dense22Term {
    long scaled; // signed 53-bit mantissa shifted ten positions towards the sign bit
    int exponent;
    int trailing;
};
inline Dense22Term dense22_decode(ulong raw) {
    DotTerm t=unpack_dot_term(raw);
    return {as_type<long>(as_type<ulong>(t.mantissa)<<10), t.exponent,
            int(ctz(as_type<ulong>(t.mantissa)))};
}
inline DotTerm dense22_wide(Dense22Term a,Dense22Term b,DotTerm c) {
    return wide_dot_fma({a.scaled>>10,a.exponent},{b.scaled>>10,b.exponent},c);
}
inline DotTerm dense22_fma(Dense22Term a,Dense22Term b,DotTerm c) {
    int ep=a.exponent+b.exponent, delta=c.exponent-ep;
    uint bounds=uint((a.exponent+256)|(b.exponent+256)|(c.exponent+256));
    if(bounds>=512 || uint(delta+8)>62) return dense22_wide(a,b,c);
    // Write both signed significands exactly as H*2^22+L with 0<=L<2^22.
    // HH, the two cross terms, and the discarded-bit carry all fit in 64 bits.
    int ah=int(a.scaled>>32), bh=int(b.scaled>>32);
    uint al=uint(a.scaled)>>10, bl=uint(b.scaled)>>10;
    long cross=long(ah)*int(bl)+long(bh)*int(al);
    ulong tail=((as_type<ulong>(cross)&0x3ffffful)<<22)+ulong(al)*bl;
    long product=long(ah)*bh+(cross>>22)+long(tail>>44);
    uint right=uint(max(0,delta)), left=uint(8+min(0,delta));
    long floor_sum=(product>>right)+as_type<long>(as_type<ulong>(c.mantissa)<<left);
    int high=int(floor_sum>>32);
    uint leading=uint(high^(high>>31));
    if(leading<0x200000u) return dense22_wide(a,b,c);
    uint cut=11u-clz(leading), halfbit=1u<<(cut-1), mask=2*halfbit-1;
    long rounded=(floor_sum+halfbit)>>cut;
    // The full fractional product vanishes below bit 44+right exactly when the
    // factors' combined number of trailing zeros reaches that bit position.
    rounded-=((uint(floor_sum)&mask)==halfbit && a.trailing+b.trailing>=44+int(right) && (rounded&1));
    return {rounded,max(ep,c.exponent)-8+int(cut)};
}
kernel void mm_dense_22(device ulong* out,const device ulong* a,const device ulong* b,
                        constant uint& m,constant uint& n,constant uint& k,
                        uint3 group [[threadgroup_position_in_grid]],uint3 local [[thread_position_in_threadgroup]]) {
    constexpr uint H=8,W=16,K=32;
    threadgroup Dense22Term left[H*K], right[K*W];
    uint row=H*group.y+local.y,col=W*group.x+local.x,lane=W*local.y+local.x;
    DotTerm sum=unpack_dot_term(0);
    for(uint start=0;start<k;start+=K) {
        for(uint i=lane;i<H*K;i+=W*H) {
            uint r=H*group.y+i/K,c=start+i%K;
            left[i]=dense22_decode(r<m && c<k ? a[ulong(r)*k+c] : 0);
        }
        for(uint i=lane;i<K*W;i+=W*H) {
            uint r=start+i/W,c=W*group.x+i%W;
            right[i]=dense22_decode(r<k && c<n ? b[ulong(r)*n+c] : 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint t=0;t<min(K,k-start);++t)
            sum=dense22_fma(left[local.y*K+t],right[t*W+local.x],sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<m && col<n) out[ulong(row)*n+col]=pack_dot_term(sum);
}
