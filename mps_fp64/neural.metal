kernel void batch_norm_affine(device ulong *out, const device ulong *x,
                              const device ulong *mean, const device ulong *invstd,
                              const device ulong *weight, const device ulong *bias,
                              constant uint &spatial, constant uint &channels, constant uint &flags,
                              uint i [[thread_position_in_grid]]) {
    uint c = (i/spatial) % channels;
    ulong norm = __softfloat64_fmul(__softfloat64_fsub(x[i],mean[c],0),invstd[c],0);
    if (flags & 1u) norm = __softfloat64_fmul(norm,weight[c],0);
    if (flags & 2u) norm = __softfloat64_fadd(norm,bias[c],0);
    out[i] = norm;
}

kernel void euclidean_dist(device ulong *out, const device ulong *a, const device ulong *b,
                           constant uint &rows_a, constant uint &rows_b, constant uint &width,
                           uint i [[thread_position_in_grid]]) {
    uint y = i % rows_b, x = (i/rows_b) % rows_a, batch = i/rows_b/rows_a;
    a += (ulong(batch)*rows_a+x)*width;
    b += (ulong(batch)*rows_b+y)*width;
    ulong sum = 0;
    for (uint k=0; k<width; ++k) {
        ulong delta = __softfloat64_fsub(a[k],b[k],0);
        sum = nearby_fma(delta,delta,sum);
    }
    out[i] = __softfloat64_fsqrt(sum,0);
}

kernel void embedding_ranges(device long *owners, device long *lengths, const device long *idx,
                             const device long *offsets, constant uint &bags, constant uint &n,
                             constant uint &last, constant uint &mode, constant long &pad,
                             uint bag [[thread_position_in_grid]]) {
    uint start=uint(offsets[bag]), end=(bag+1<bags || last)?uint(offsets[bag+1]):n;
    uint count=0;
    for (uint i=start;i<end;++i) { owners[i]=bag; count+=idx[i]!=pad; }
    lengths[bag]=mode==0?0:count;
}

kernel void embedding_values(device ulong *out, device long *maxima, const device ulong *w,
                             const device long *idx, const device long *offsets, const device ulong *scales,
                             constant uint &bags, constant uint &width, constant uint &n,
                             constant uint &last, constant uint &mode, constant long &pad, constant uint &weighted,
                             uint p [[thread_position_in_grid]]) {
    uint b=p/width, c=p%width, start=uint(offsets[b]), end=(b+1<bags || last)?uint(offsets[b+1]):n;
    ulong acc=0;
    uint count=0;
    long winner=0;
    for (uint i=start;i<end;++i) {
        long row=idx[i];
        if (row==pad) continue;
        ulong value=w[ulong(row)*width+c];
        if (mode==2) {
            if (!count || __softfloat64_fgt(value,acc)) { acc=value; winner=row; }
        } else if (weighted) acc=nearby_fma(scales[i],value,acc);
        else acc=__softfloat64_fadd(acc,value,0);
        ++count;
    }
    if (mode==1 && count) acc=__softfloat64_fdiv(acc,__softfloat64_cvt_i64_to_f64(count,0),0);
    if (mode==2) maxima[p]=winner;
    out[p]=acc;
}

// Reproduce the arm64 CPU sum grouping used by this PyTorch build for short
// FP32 covariance weights. The double-data path keeps its FP64 accumulator.
inline uint cov_add_f32(uint a, uint b) {
    return __softfloat64_cvt_f64_to_f32(__softfloat64_fadd(
        __softfloat64_cvt_f32_to_f64(a),__softfloat64_cvt_f32_to_f64(b),0),0);
}
kernel void cov_short_weight_sum(device uint *out, const device uint *input, constant uint &n,
                                 constant uint &lanes, uint tid [[thread_position_in_grid]]) {
    uint sums[4][4] = {}, next_level[4][4] = {};
    uint vectors=n/lanes, groups=vectors/4;
    for (uint i=0;i<groups;++i) {
        for (uint j=0;j<4;++j)
            for (uint lane=0;lane<lanes;++lane)
                sums[j][lane]=cov_add_f32(sums[j][lane],input[(4*i+j)*lanes+lane]);
        if (!((i+1)%16)) {
            for (uint j=0;j<4;++j) for (uint lane=0;lane<lanes;++lane) {
                next_level[j][lane]=cov_add_f32(next_level[j][lane],sums[j][lane]);
                sums[j][lane]=0;
            }
        }
    }
    for (uint j=0;j<4;++j) for (uint lane=0;lane<lanes;++lane)
        sums[j][lane]=cov_add_f32(sums[j][lane],next_level[j][lane]);
    for (uint i=4*groups;i<vectors;++i)
        for (uint lane=0;lane<lanes;++lane)
            sums[0][lane]=cov_add_f32(sums[0][lane],input[i*lanes+lane]);
    uint value=0;
    for (uint i=vectors*lanes;i<n;++i) value=cov_add_f32(value,input[i]);
    for (uint lane=0;lane<lanes;++lane) {
        for (uint j=1;j<4;++j) sums[0][lane]=cov_add_f32(sums[0][lane],sums[j][lane]);
        value=cov_add_f32(value,sums[0][lane]);
    }
    out[0]=value;
}
