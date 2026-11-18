// Binary64 interpolation on separable axes. Each output owns its full sum, in
// both directions, so no floating-point atomics or ordering races are needed.
using namespace metal;
namespace resampling {
constant ulong one = 0x3ff0000000000000ul;
constant ulong half_bit = 0x3fe0000000000000ul;
constant ulong two = 0x4000000000000000ul;
inline ulong add(ulong a, ulong b) { return __softfloat64_fadd(a,b,0); }
inline ulong sub(ulong a, ulong b) { return __softfloat64_fsub(a,b,0); }
inline ulong mul(ulong a, ulong b) { return __softfloat64_fmul(a,b,0); }
inline ulong fma64(ulong a, ulong b, ulong c) { return __softfloat64_fma(a,b,c,0); }
inline ulong num(int n) { return __softfloat64_cvt_i64_to_f64(n,0); }

inline ulong cubic1(ulong x, bool aa) {
    // ((a+2)*x - (a+3))*x^2 + 1, in the Keys cubic kernel.
    ulong ap2 = aa ? 0x3ff8000000000000ul : 0x3ff4000000000000ul;
    ulong ap3 = aa ? 0x4004000000000000ul : 0x4002000000000000ul;
    return fma64(mul(fma64(ap2,x,ap3 ^ (1ul<<63)),x),x,one);
}
inline ulong cubic2(ulong x, bool aa) {
    ulong a = aa ? 0xbfe0000000000000ul : 0xbfe8000000000000ul;
    ulong minus5a = aa ? 0x4004000000000000ul : 0x400e000000000000ul;
    ulong plus8a = aa ? 0xc010000000000000ul : 0xc018000000000000ul;
    ulong minus4a = aa ? two : 0x4008000000000000ul;
    return fma64(fma64(fma64(a,x,minus5a),x,plus8a),x,minus4a);
}
inline int bounded_index(ulong value, int size) {
    if (value >> 63) return 0;
    if (value >= num(size)) return size;
    return int(__softfloat64_cvt_f64_to_i64(value,3));
}
inline int sample_index(int2 range, int t, int size, bool duplicate) {
    return int(clamp(long(range.x) + (duplicate ? 0 : t), 0l, long(size)-1));
}
}

kernel void resample_weights(device int2* ranges, device ulong* weights,
                             constant long* plan, uint j [[thread_position_in_grid]]) {
    using namespace resampling;
    int size = int(plan[0]), taps = int(plan[1]);
    bool cubic = plan[2], aa = plan[3], align = plan[4], duplicate = plan[5];
    ulong ratio = ulong(plan[6]), support = ulong(plan[7]), invscale = ulong(plan[8]);
    device ulong* w = weights + ulong(j)*taps;
    ulong pos = num(int(j));
    if (aa) {
        ulong center = mul(add(pos,half_bit),ratio);
        int start = bounded_index(add(sub(center,support),half_bit),size);
        int end = bounded_index(add(add(center,support),half_bit),size);
        int count = clamp(end-start,0,taps);
        ranges[j] = int2(start,count);
        ulong total = 0;
        for (int t=0; t<count; ++t) {
            ulong dist = mul(add(sub(num(start+t),center),half_bit),invscale) & 0x7ffffffffffffffful;
            ulong weight = cubic ? (dist<one ? cubic1(dist,true) : dist<two ? cubic2(dist,true) : 0ul)
                                 : (dist<one ? sub(one,dist) : 0ul);
            w[t] = weight;
            total = add(total,weight);
        }
        if (total << 1) {
            for (int t=0; t<count; ++t) w[t] = __softfloat64_fdiv(w[t],total,0);
        }
        return;
    }
    if (duplicate) {
        ranges[j] = int2(int(j),2); w[0]=one; w[1]=0; return;
    }
    ulong src = align ? mul(ratio,pos) : fma64(ratio,add(pos,half_bit),half_bit ^ (1ul<<63));
    if (!cubic && src>>63) src=0;
    // ATen guard_index_and_lambda uses floorf for its integer guard even for
    // FP64 samples. Keep the fractional distance and every weight in binary64.
    float guard = floor(as_type<float>(__softfloat64_cvt_f64_to_f32(src,0)));
    int base = guard >= 2147483648.0f ? size-1 : min(int(guard),size-1);
    ulong frac = sub(src,num(base));
    frac = frac>>63 ? 0ul : min(frac,one);
    ulong other = sub(one,frac);
    ranges[j] = int2(base-int(cubic),taps);
    if (cubic) {
        w[0]=cubic2(add(frac,one),false); w[1]=cubic1(frac,false);
        w[2]=cubic1(other,false); w[3]=cubic2(add(other,one),false);
    } else { w[0]=other; w[1]=frac; }
}

kernel void resample_axis_forward(device ulong* out, const device ulong* x,
                                  const device int2* ranges, const device ulong* weights,
                                  constant long* geom, uint i [[thread_position_in_grid]]) {
    using namespace resampling;
    uint size=uint(geom[0]), output=uint(geom[1]), inner=uint(geom[2]), taps=uint(geom[3]);
    bool duplicate=geom[4];
    uint inner_i=i % inner, rows=i/inner, dest=rows%output;
    ulong offset=ulong(rows/output)*size*inner+inner_i;
    int2 range=ranges[dest];
    ulong value=0;
    for (int t=0; t<range.y; ++t) {
        int index=sample_index(range,t,int(size),duplicate);
        ulong term=mul(x[offset+ulong(index)*inner],weights[ulong(dest)*taps+t]);
        value=t==0 ? term : add(value,term);
    }
    out[i]=value;
}

kernel void resample_axis_backward(device ulong* out, const device ulong* grad,
                                   const device int2* ranges, const device ulong* weights,
                                   constant long* geom, uint i [[thread_position_in_grid]]) {
    using namespace resampling;
    uint size=uint(geom[0]), output=uint(geom[1]), inner=uint(geom[2]), taps=uint(geom[3]);
    bool duplicate=geom[4];
    uint inner_i=i%inner, rows=i/inner;
    int dest=int(rows%size);
    ulong offset=ulong(rows/size)*output*inner+inner_i;
    // The windows are monotone. Find the first ending at this source index or
    // later, then visit only the overlapping output locations in ascending order.
    uint lo=0, hi=output;
    while (lo<hi) {
        uint mid=lo+(hi-lo)/2; int2 range=ranges[mid];
        int last=sample_index(range,range.y-1,int(size),duplicate);
        if (last<dest) lo=mid+1; else hi=mid;
    }
    ulong value=0;
    for (uint j=lo; j<output; ++j) {
        int2 range=ranges[j];
        if (sample_index(range,0,int(size),duplicate)>dest) break;
        // Interior samples have one tap per window. At either image edge all
        // out-of-bounds cubic taps clamp to that edge and must also contribute.
        long pos=long(dest)-range.x;
        int begin=(duplicate || dest==0) ? 0 : int(clamp(pos,0l,long(range.y)));
        int end=(duplicate || dest==int(size)-1) ? range.y : int(clamp(pos+1,0l,long(range.y)));
        for (int t=begin; t<end; ++t)
            value=add(value,mul(grad[offset+ulong(j)*inner],weights[ulong(j)*taps+t]));
    }
    out[i]=value;
}
