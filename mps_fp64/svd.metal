// One-sided Jacobi SVD. A column is represented as b * 2^e, where max |b|
// lies in [1,2). This avoids squaring the original matrix or forming A^T A.
namespace jacobi {
constant ulong sign_bit = 1ul << 63;
constant ulong one = 0x3ff0000000000000ul;
constant ulong inf_bits = 0x7ff0000000000000ul;
inline ulong add(ulong x,ulong y){ return __softfloat64_fadd(x,y,0); }
inline ulong sub(ulong x,ulong y){ return __softfloat64_fsub(x,y,0); }
inline ulong mul(ulong x,ulong y){ return __softfloat64_fmul(x,y,0); }
inline ulong div64(ulong x,ulong y){ return __softfloat64_fdiv(x,y,0); }
inline ulong fma64(ulong x,ulong y,ulong z){ return __softfloat64_fma(x,y,z,0); }
inline ulong sqrt64(ulong x){ return __softfloat64_fsqrt(x,0); }
inline ulong mag(ulong x){ return x & ~sign_bit; }
inline int word_exponent(ulong x) { return softfloat64_internal::unpack(x).exp; }
inline ulong word_scale(ulong x,int delta){
    if (!mag(x)) return x;
    auto u=softfloat64_internal::unpack(x);
    int e=u.exp+delta;
    if (e>1023) return (x&sign_bit)|inf_bits;
    if (e>=-1022) return (x&sign_bit)|(ulong(e+1023)<<52)|(u.mantissa & ((1ul<<52)-1));
    int k=-1022-e;
    if (k>53) return x&sign_bit;
    ulong retained=u.mantissa>>k, halfbit=1ul<<(k-1), rest=u.mantissa&((halfbit<<1)-1);
    return (x&sign_bit) | (retained+uint(rest>halfbit || (rest==halfbit && (retained&1ul))));
}
inline ulong sum_or_max(ulong value,uint width,bool maximum=false){
    for (ushort step=ushort(width/2);step;step/=2) {
        ulong other=as_type<ulong>(metal::simd_shuffle_down(as_type<uint2>(value),step));
        value=maximum ? max(value,other) : add(value,other);
    }
    return as_type<ulong>(metal::simd_broadcast_first(as_type<uint2>(value)));
}
}

kernel void svd_init(const device ulong* src,device ulong* b,device int* scales,
                     device atomic_int* error,constant long* g,
                     uint col [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                     uint width [[threads_per_simdgroup]]) {
    using namespace jacobi;
    uint m=uint(g[0]); ulong base=ulong(col)*m, peak=0;
    for (uint r=lane;r<m;r+=width) peak=max(peak,mag(src[base+r]));
    peak=sum_or_max(peak,width,true);
    if (peak>=inf_bits) { if (!lane) metal::atomic_fetch_or_explicit(error,1,metal::memory_order_relaxed); return; }
    int e=peak ? word_exponent(peak) : 0;
    if (!lane) scales[col]=e;
    bool lost=false;
    for (uint r=lane;r<m;r+=width) {
        ulong original=src[base+r], normalized=word_scale(original,-e);
        b[base+r]=normalized;
        lost |= mag(original) && mag(normalized)<(1ul<<52);
    }
    if (metal::simd_any(lost) && !lane) metal::atomic_fetch_or_explicit(error,2,metal::memory_order_relaxed);
}

kernel void svd_round(device ulong* b, device ulong* v, device int* scales,device atomic_int* changed,
                      constant long* g,uint group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    using namespace jacobi;
    uint m=uint(g[0]),n=uint(g[1]),turn=uint(g[2]),ring=(n+(n&1))-1;
    uint pairs=(ring+1)/2, batch=group/pairs, pair=group%pairs;
    uint p=pair ? (turn+pair)%ring : ring, q=pair ? (turn+ring-pair)%ring : turn;
    if (p>=n || q>=n) return;
    uint ic=batch*n+p, jc=batch*n+q;
    device ulong *bp=b+ulong(ic)*m,*bq=b+ulong(jc)*m;
    int ep=scales[ic], eq=scales[jc];
    ulong aa=0,bb=0,ab=0;
    for (uint r=lane;r<m;r+=width) {
        ulong x=bp[r],y=bq[r];
        aa=fma64(x,x,aa); bb=fma64(y,y,bb); ab=fma64(x,y,ab);
    }
    aa=sum_or_max(aa,width); bb=sum_or_max(bb,width); ab=sum_or_max(ab,width);
    if (!aa || !bb || mag(ab)<=mul(ulong(g[3]),sqrt64(mul(aa,bb)))) return;
    if (!lane) metal::atomic_store_explicit(changed,1,metal::memory_order_relaxed);
    ulong c,t,tp,tq;
    if (abs(ep-eq)>64) {
        // To working precision c=1. Compute the projection on the smaller
        // column's own word_scale even when the corresponding sine underflows.
        c=one;
        if (ep>eq) {
            tq=div64(ab,aa)^sign_bit; t=word_scale(tq,eq-ep); tp=word_scale(t,eq-ep);
        } else {
            tp=div64(ab,bb); t=word_scale(tp,ep-eq); tq=word_scale(t,ep-eq);
        }
    } else {
        int e=max(ep,eq);
        ulong delta=sub(word_scale(bb,2*(eq-e)),word_scale(aa,2*(ep-e)));
        ulong twice=word_scale(ab,1+ep+eq-2*e);
        ulong hyp=sqrt64(fma64(delta,delta,mul(twice,twice)));
        t=div64(twice,add(mag(delta),hyp));
        if (delta & sign_bit) t^=sign_bit;
        c=div64(one,sqrt64(fma64(t,t,one)));
        tp=word_scale(t,eq-ep); tq=word_scale(t,ep-eq);
    }
    ulong peakp=0,peakq=0;
    for (uint r=lane;r<m;r+=width) {
        ulong x=bp[r],y=bq[r];
        ulong xp=mul(c,fma64(tp^sign_bit,y,x)), xq=mul(c,fma64(tq,x,y));
        bp[r]=xp; bq[r]=xq; peakp=max(peakp,mag(xp)); peakq=max(peakq,mag(xq));
    }
    peakp=sum_or_max(peakp,width,true); peakq=sum_or_max(peakq,width,true);
    int dp=peakp ? word_exponent(peakp) : 0, dq=peakq ? word_exponent(peakq) : 0;
    for (uint r=lane;r<m;r+=width) {
        if (dp) bp[r]=word_scale(bp[r],-dp);
        if (dq) bq[r]=word_scale(bq[r],-dq);
    }
    if (!lane) {scales[ic]=ep+dp;scales[jc]=eq+dq;}
    if (g[4]) {
        device ulong *vp=v+ulong(ic)*n,*vq=v+ulong(jc)*n;
        for (uint r=lane;r<n;r+=width) {
            ulong x=vp[r],y=vq[r];
            vp[r]=mul(c,fma64(t^sign_bit,y,x)); vq[r]=mul(c,fma64(t,x,y));
        }
    }
}

kernel void svd_norms(const device ulong* b,const device int* scales,device ulong* singular,
                      device ulong* norm, constant long& rows,uint col [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    using namespace jacobi;
    ulong v=0;
    for (uint r=lane;r<uint(rows);r+=width){ ulong x=b[ulong(col)*rows+r]; v=fma64(x,x,v); }
    v=sqrt64(sum_or_max(v,width));
    if (!lane) { norm[col]=v;singular[col]=word_scale(v,scales[col]); }
}

kernel void svd_vectors(const device ulong* b,const device ulong* norm,const device ulong* singular,
                        const device long* order, device ulong* u,constant long* g,
                        uint i [[thread_position_in_grid]]) {
    uint m=uint(g[0]), n=uint(g[1]), k=uint(g[2]), row=i%m, pos=i/m, col=pos%k, batch=pos/k;
    if (col>=n || !singular[batch*n+col]) { u[i]=0; return; }
    ulong source=ulong(batch)*n+order[batch*n+col];
    u[i]=jacobi::div64(b[source*m+row],norm[source]);
}

// A Householder QR of the nonzero left singular vectors produces a compact
// orthogonal map. Its remaining columns complete the null space in O(m^2 r).
kernel void svd_reflectors(device ulong* refl,device ulong* tau,const device ulong* singular,
                           constant long* g,uint batch [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    using namespace jacobi;
    uint m=uint(g[0]),n=uint(g[1]),k=uint(g[2]);
    uint active=0;while(active<n && singular[ulong(batch)*n+active]) ++active;
    if(active==k) return;
    device ulong* base=refl+ulong(batch)*n*m;
    for(uint j=0;j<active;++j) {
        device ulong* v=base+ulong(j)*m;
        ulong square=0;
        for(uint i=j+lane;i<m;i+=width) square=fma64(v[i],v[i],square);
        ulong norm=sqrt64(sum_or_max(square,width)),a=v[j];
        ulong beta=norm|((a^sign_bit)&sign_bit), denom=sub(a,beta);
        ulong factor=div64(sub(beta,a),beta);
        if(!lane) tau[ulong(batch)*n+j]=factor;
        for(uint i=j+lane;i<m;i+=width) v[i]=i==j ? one : div64(v[i],denom);
        metal::simdgroup_barrier(metal::mem_flags::mem_device);
        for(uint l=j+1;l<active;++l){
            device ulong* col=base+ulong(l)*m; ulong dot=0;
            for(uint i=j+lane;i<m;i+=width) dot=fma64(v[i],col[i],dot);
            dot=mul(factor,sum_or_max(dot,width));
            for(uint i=j+lane;i<m;i+=width) col[i]=fma64(dot^sign_bit,v[i],col[i]);
        }
        metal::simdgroup_barrier(metal::mem_flags::mem_device);
    }
}

kernel void svd_complete(device ulong* u,const device ulong* refl,const device ulong* tau,
                         const device ulong* singular,constant long* g,uint group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    using namespace jacobi;
    uint m=uint(g[0]),n=uint(g[1]),k=uint(g[2]),batch=group/k,col=group%k;
    if(col<n && singular[ulong(batch)*n+col]) return;
    uint active=0;while(active<n && singular[ulong(batch)*n+active]) ++active;
    device ulong* v=u+ulong(group)*m;
    for(uint i=lane;i<m;i+=width) v[i]=(i==col) ? one : 0;
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
    for(int j=int(active)-1;j>=0;--j){
        const device ulong* h=refl+(ulong(batch)*n+j)*m;ulong dot=0;
        for(uint i=j+lane;i<m;i+=width) dot=fma64(h[i],v[i],dot);
        dot=mul(tau[ulong(batch)*n+j],sum_or_max(dot,width));
        for(uint i=j+lane;i<m;i+=width) v[i]=fma64(dot^sign_bit,h[i],v[i]);
        metal::simdgroup_barrier(metal::mem_flags::mem_device);
    }
}

// Sort IEEE words by their complete nonnegative magnitude. Stable tie breaking
// also preserves a reproducible basis order for equal singular values.
kernel void svd_order(const device ulong* sigma,device ulong* sorted,device long* order,
                      constant long& columns,uint i [[thread_position_in_grid]]) {
    uint n=uint(columns),j=i%n,base=i-j,pos=0; ulong s=sigma[i];
    for(uint t=0;t<n;++t) { ulong other=sigma[base+t];pos+=uint(other>s || (other==s && t<j)); }
    sorted[base+pos]=s;order[base+pos]=j;
}

kernel void svd_gather_right(const device ulong* v,const device long* order,device ulong* result,
                             constant long& columns,uint i [[thread_position_in_grid]]) {
    uint n=uint(columns),r=i%n,pos=i/n,batch=pos/n;
    result[i]=v[(ulong(batch)*n+order[pos])*n+r];
}

kernel void svd_lower_symmetric(const device ulong* a,device ulong* sym,constant long& size,
                                uint i [[thread_position_in_grid]]) {
    uint n=uint(size),col=i%n,row=(i/n)%n,base=i-col-row*n;
    sym[i]=row>=col ? a[i] : a[base+col*n+row];
}
kernel void svd_reciprocals(const device ulong* sigma,const device ulong* atol,const device ulong* rtol,
                            device ulong* inv,constant long* spec,uint i [[thread_position_in_grid]]) {
    using namespace jacobi;
    uint n=uint(spec[0]),b=i/n; ulong a=atol[b],r=rtol[b];
    if(spec[1]) r=__softfloat64_fgt(a,0) ? 0 : ulong(spec[2]);
    ulong rel=mul(r,sigma[b*n]);
    // NaN in either tolerance propagates to the threshold, so no value passes.
    bool active=__softfloat64_fgt(sigma[i],a) && __softfloat64_fgt(sigma[i],rel);
    inv[i]=active ? div64(one,sigma[i]) : 0;
}
