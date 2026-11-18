struct Pool {
    uint size[3],output[3],window[3],stride[3],pad[3],dilation[3];
    uint adaptive,maximum,include_pad;
    int divisor;
};
inline ulong pool_add_zero(ulong v) {
    if ((v << 1) == 0) return 0;
    return fp_nan(v) ? __softfloat64_fadd(v, 0ul, 0) : v;
}
inline ulong pool_divide(ulong value, long divisor) {
    // A power-of-two divisor changes only the exponent when both operands
    // and the result are normal. Preserve gradual underflow on the slow path.
    const ulong mag = ulong(divisor < 0 ? -divisor : divisor);
    if (mag && !(mag & (mag - 1))) {
        const uint shift = ctz(mag), exponent = uint((value >> 52) & 0x7fful);
        if (exponent > shift && exponent < 0x7ff) {
            return (value - (ulong(shift) << 52)) ^ (divisor < 0 ? (1ul << 63) : 0);
        }
    }
    return __softfloat64_fdiv(value, __softfloat64_cvt_i64_to_f64(divisor, 0), 0);
}
struct PoolAxis {int start,end,step;};
inline PoolAxis pool_range(uint out,uint axis,constant Pool &p) {
    if(p.adaptive) {
        return {int(ulong(out)*p.size[axis]/p.output[axis]),
                int((ulong(out+1)*p.size[axis]+p.output[axis]-1)/p.output[axis]),1};
    }
    int start=int(out*p.stride[axis])-int(p.pad[axis]);
    return {start,min(start+int((p.window[axis]-1)*p.dilation[axis]+1),int(p.size[axis]+p.pad[axis])),
            int(p.dilation[axis])};
}
kernel void pool_forward(device ulong *out,device long *ids,const device ulong *src,constant Pool &p,
                         uint i [[thread_position_in_grid]]) {
    uint nx=p.output[2],ny=p.output[1],nz=p.output[0],batch=i/nx/ny/nz;
    PoolAxis x=pool_range(i%nx,2,p),y=pool_range((i/nx)%ny,1,p),z=pool_range((i/nx/ny)%nz,0,p);
    ulong acc=p.maximum ? 0xfff0000000000000ul : 0ul;
    long best=-1;
    ulong score=0;
    uint count=0;
    for(int iz=z.start;iz<z.end;iz+=z.step) for(int iy=y.start;iy<y.end;iy+=y.step)
    for(int ix=x.start;ix<x.end;ix+=x.step) {
        bool valid=uint(iz)<p.size[0] && uint(iy)<p.size[1] && uint(ix)<p.size[2];
        if(valid || p.include_pad) ++count;
        if(!valid) continue;
        ulong id=(ulong(iz)*p.size[1]+iy)*p.size[2]+ix;
        ulong v=src[ulong(batch)*p.size[0]*p.size[1]*p.size[2]+id];
        if(p.maximum) {
            const ulong key = fp_order_key(v);
            if (key > score || fp_nan(v)) { best = long(id); acc = v; score = key; }
        } else acc=__softfloat64_fadd(acc,v,0);
    }
    if(p.maximum) {out[i]=acc;ids[i]=best;}
    else {
        const long divisor = p.divisor ? long(p.divisor) : long(count);
        out[i] = pool_divide(acc, divisor);
    }
}
inline int2 covering_outputs(uint pos,uint axis,constant Pool& p) {
    if(p.adaptive)
        return int2(ulong(pos)*p.output[axis]/p.size[axis],(ulong(pos+1)*p.output[axis]+p.size[axis]-1)/p.size[axis]);
    int end=int(pos+p.pad[axis]), first=end-int((p.window[axis]-1)*p.dilation[axis]);
    return int2(max(0,(first+int(p.stride[axis])-1)/int(p.stride[axis])),
                min(int(p.output[axis]),end/int(p.stride[axis])+1));
}
inline uint sample_count(PoolAxis a,uint size,bool pad) {
    if(pad) return uint((a.end-a.start+a.step-1)/a.step);
    // Average pooling has unit dilation. Adaptive windows always lie in range.
    return uint(max(0,min(int(size),a.end)-max(0,a.start)));
}
kernel void pool_backward(device ulong *dx,const device ulong *dy,const device long *ids,
                          constant Pool& p,uint id [[thread_position_in_grid]]) {
    uint px=id%p.size[2], py=id/p.size[2]%p.size[1], pz=id/p.size[2]/p.size[1]%p.size[0];
    uint volume=p.size[0]*p.size[1]*p.size[2], row=id/volume;
    int2 xs=covering_outputs(px,2,p),ys=covering_outputs(py,1,p),zs=covering_outputs(pz,0,p);
    ulong value=0;
    for(int z=zs.x;z<zs.y;++z) for(int y=ys.x;y<ys.y;++y) for(int x=xs.x;x<xs.y;++x) {
        ulong idx=((ulong(row)*p.output[0]+z)*p.output[1]+y)*p.output[2]+x;
        if(p.maximum) {
            if(ids[idx]==long(id%volume)) value=__softfloat64_fadd(value,dy[idx],0);
        } else {
            long count=p.divisor;
            if(!count) count=sample_count(pool_range(z,0,p),p.size[0],p.include_pad)*
                             sample_count(pool_range(y,1,p),p.size[1],p.include_pad)*
                             sample_count(pool_range(x,2,p),p.size[2],p.include_pad);
            value=__softfloat64_fadd(value,pool_divide(dy[idx], count),0);
        }
    }
    dx[id]=value;
}
// Contiguous 2x2, stride-2 tiles cover each input exactly once. The four
// reads stay in row-major order so average rounding and max ties match the
// general kernel. Both directions use one work item per tile.
kernel void pool_square_forward(device ulong *out, device long *ids, const device ulong *src,
                                constant Pool &p, uint i [[thread_position_in_grid]]) {
    const uint w = p.size[2], ow = p.output[2], area = w * p.size[1];
    const uint row = i / (ow * p.output[1]);
    const uint column = (i % ow) * 2 + (i / ow % p.output[1]) * (2 * w);
    const ulong base = ulong(row) * area + column;
    const ulong a = src[base], b = src[base + 1], c = src[base + w], d = src[base + w + 1];
    if (p.maximum) {
        ulong best = a, score = fp_order_key(a);
        uint offset = 0;
        const ulong bs = fp_order_key(b), cs = fp_order_key(c), ds = fp_order_key(d);
        if (bs > score || fp_nan(b)) { score = bs; best = b; offset = 1; }
        if (cs > score || fp_nan(c)) { score = cs; best = c; offset = w; }
        if (ds > score || fp_nan(d)) { best = d; offset = w + 1; }
        out[i] = best; ids[i] = long(column + offset);
    } else {
        ulong sum = __softfloat64_fadd(pool_add_zero(a), b, 0);
        sum = __softfloat64_fadd(sum, c, 0);
        sum = __softfloat64_fadd(sum, d, 0);
        out[i] = pool_divide(sum, p.divisor ? p.divisor : 4);
    }
}
kernel void pool_square_backward(device ulong *dx, const device ulong *dy, const device long *ids,
                                 constant Pool &p, uint i [[thread_position_in_grid]]) {
    const uint w = p.size[2], ow = p.output[2], area = w * p.size[1];
    const uint row = i / (ow * p.output[1]);
    const uint column = (i % ow) * 2 + (i / ow % p.output[1]) * (2 * w);
    const ulong base = ulong(row) * area + column;
    const ulong v = pool_add_zero(p.maximum ? dy[i] : pool_divide(dy[i], p.divisor ? p.divisor : 4));
    const long index = p.maximum ? ids[i] : 0;
    dx[base] = !p.maximum || index == long(column) ? v : 0;
    dx[base + 1] = !p.maximum || index == long(column + 1) ? v : 0;
    dx[base + w] = !p.maximum || index == long(column + w) ? v : 0;
    dx[base + w + 1] = !p.maximum || index == long(column + w + 1) ? v : 0;
}

// Encoding the source column plus one leaves zero available for empty outputs.
kernel void unpool_owners(device atomic_uint *owner, device atomic_uint *error,
                          const device long *indices, constant uint &in_size,
                          constant uint &out_size, uint i [[thread_position_in_grid]]) {
    const long col = indices[i];
    if (col < 0 || ulong(col) >= out_size) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    const uint source = i % in_size, row = i / in_size;
    atomic_fetch_max_explicit(owner + ulong(row) * out_size + col, source + 1, memory_order_relaxed);
}
kernel void unpool_gather(device ulong *out, const device ulong *src, const device uint *owner,
                          constant uint &in_size, constant uint &out_size, uint i [[thread_position_in_grid]]) {
    const uint source = owner[i];
    out[i] = source ? src[ulong(i / out_size) * in_size + source - 1] : 0;
}
