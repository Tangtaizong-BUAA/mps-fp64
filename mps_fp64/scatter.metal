// Each distinct destination has exactly one writer. A stable sort places input entries
// in their original order inside each segment, including duplicate indices.
kernel void scatter_segments(device ulong *out, const device long *keys, const device long *order,
                             const device ulong *values, constant uint &count, constant uint &reduce,
                             constant uint &include_self, uint i [[thread_position_in_grid]]) {
    long key = keys[i];
    if (i && keys[i-1] == key) return;
    uint start = i;
    if (reduce == 5) {
        while (i+1<count && keys[i+1]==key) ++i;
        out[key] = values[order[i]];
        return;
    }
    ulong acc = include_self ? out[key] : values[order[i++]];
    for (; i<count && keys[i]==key; ++i) {
        ulong value = values[order[i]];
        if (reduce == 1) acc = __softfloat64_fmul(acc,value,0);
        else if (reduce == 2) acc = fp_max(acc,value);
        else if (reduce == 3) acc = fp_min(acc,value);
        else acc = __softfloat64_fadd(acc,value,0);
    }
    if (reduce == 4)
        acc = __softfloat64_fdiv(acc,__softfloat64_cvt_i64_to_f64(long(i-start)+include_self,0),0);
    out[key] = acc;
}

kernel void search_sorted(device long *out, const device ulong *boundaries, const device ulong *query,
                          const device long *sorter, constant uint &width, constant uint &per_row,
                          constant uint &right, constant uint &use_sorter, uint i [[thread_position_in_grid]]) {
    ulong base = ulong(i/per_row)*width;
    boundaries += base;
    uint lo = 0, hi = width;
    while (lo < hi) {
        uint mid = lo+(hi-lo)/2;
        ulong boundary = boundaries[use_sorter ? uint(sorter[base+mid]) : mid];
        bool upper = right ? __softfloat64_fgt(boundary,query[i]) : __softfloat64_fge(boundary,query[i]);
        if (upper) hi = mid;
        else lo = mid+1;
    }
    out[i] = lo;
}

inline bool scan_choose_right(ulong left, ulong right, bool maximum) {
    return fp_nan(right) || (!fp_nan(left) &&
           (maximum ? __softfloat64_fge(right,left) : __softfloat64_fle(right,left)));
}

// Inclusive prefix in each SIMD group, followed by a scan of the four group summaries.
kernel void cumulative_extreme_tile(device ulong *out, device long *indices,
                                    device ulong *totals, device long *total_indices,
                                    const device ulong *values, const device long *input_indices,
                                    constant uint &width, constant uint &blocks,
                                    constant uint &maximum, constant uint &use_indices,
                                    uint group [[threadgroup_position_in_grid]],
                                    uint lid [[thread_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong warp_values[4];
    threadgroup uint warp_ids[4];
    threadgroup ulong prefix_values[4];
    threadgroup uint prefix_ids[4];
    uint row = group / blocks, column = group % blocks * 128 + lid;
    uint actual = min(column,width-1);
    ulong position = ulong(row)*width+actual;
    ulong value = values[position];
    uint index = use_indices ? uint(input_indices[position]) : actual;
    for (ushort d=1; d<32; d*=2) {
        ulong left = as_type<ulong>(simd_shuffle_up(as_type<uint2>(value),d));
        uint prior = simd_shuffle_up(index,d);
        if (lane >= d && !scan_choose_right(left,value,maximum)) { value=left; index=prior; }
    }
    if (lane==31) { warp_values[lid/32]=value; warp_ids[lid/32]=index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid<4) {
        ulong v = warp_values[lid];
        uint id = warp_ids[lid];
        for (uint w=0; w<lid; ++w) {
            // Visit preceding groups backwards to preserve the last equal extremum.
            uint j = lid-1-w;
            if (!scan_choose_right(warp_values[j],v,maximum)) { v=warp_values[j]; id=warp_ids[j]; }
        }
        prefix_values[lid]=v; prefix_ids[lid]=id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid>=32 && !scan_choose_right(prefix_values[lid/32-1],value,maximum)) {
        value=prefix_values[lid/32-1]; index=prefix_ids[lid/32-1];
    }
    if (column<width) { out[position]=value; indices[position]=index; }
    if (lid==127 && blocks>1) { totals[group]=value; total_indices[group]=index; }
}

kernel void cumulative_extreme_carry(device ulong *out, device long *ids, const device ulong *prefix,
                                     const device long *pids, constant uint &width, constant uint &blocks,
                                     constant uint &maximum, uint i [[thread_position_in_grid]]) {
    uint block = (i%width)/128;
    if (!block) return;
    ulong p = ulong(i/width)*blocks+block-1;
    if (!scan_choose_right(prefix[p],out[i],maximum)) { out[i]=prefix[p]; ids[i]=pids[p]; }
}

kernel void select_rank(device ulong *out, device long *ids, const device ulong *sorted,
                        const device long *permutation, constant uint &width, constant int &rank,
                        constant uint &ignore_nan, uint row [[thread_position_in_grid]]) {
    ulong base = ulong(row)*width;
    uint k;
    if (rank>=0) k=uint(rank);
    else {
        uint lo=0, hi=width;
        while (lo<hi) {
            uint mid=lo+(hi-lo)/2;
            if (fp_nan(sorted[base+mid])) hi=mid; else lo=mid+1;
        }
        k = !ignore_nan && lo<width ? lo : (max(1u,lo)-1)/2;
    }
    out[row]=sorted[base+k]; ids[row]=permutation[base+k];
}

inline bool row_before(const device ulong *a, const device ulong *b, uint width) {
    for (uint c=0; c<width; ++c) {
        if (__softfloat64_flt(a[c],b[c])) return true;
        if (__softfloat64_fgt(a[c],b[c])) return false;
        if (fp_nan(a[c])!=fp_nan(b[c])) return fp_nan(b[c]);
    }
    return false;
}

// Parallel stable merge of row IDs; each thread computes its own rank.
kernel void unique_merge_rows(device long *out, const device long *ids, const device ulong *data,
                              constant uint &rows, constant uint &width, constant uint &run,
                              uint i [[thread_position_in_grid]]) {
    uint base = i/(2*run)*(2*run), middle = min(rows,base+run), end=min(rows,middle+run);
    bool right = i>=middle;
    uint from = right ? base : middle, lo=from, hi=right ? middle : end;
    const device ulong *current = data+ulong(ids[i])*width;
    while (lo<hi) {
        uint m=lo+(hi-lo)/2;
        const device ulong *other = data+ulong(ids[m])*width;
        bool before = right ? !row_before(current,other,width) : row_before(other,current,width);
        if (before) lo=m+1; else hi=m;
    }
    out[base+(i-(right?middle:base))+(lo-from)] = ids[i];
}

kernel void unique_starts(device bool *mask, const device ulong *data, const device long *ids,
                          constant uint &width, constant uint &use_ids, uint i [[thread_position_in_grid]]) {
    if (!i) { mask[0]=true; return; }
    ulong a=ulong(use_ids?ids[i-1]:long(i-1))*width, b=ulong(use_ids?ids[i]:long(i))*width;
    for (uint c=0; c<width; ++c) {
        if (!__softfloat64_feq(data[a+c],data[b+c])) { mask[i]=true; return; }
    }
    mask[i]=false;
}

kernel void unique_values(device ulong *out, const device ulong *data, const device long *starts,
                          const device long *ids, constant uint &width, constant uint &use_ids,
                          uint i [[thread_position_in_grid]]) {
    long row=starts[i/width];
    out[i]=data[ulong(use_ids?ids[row]:row)*width+i%width];
}

kernel void unique_inverse(device long *out, const device long *ids, const device long *run_ids,
                           uint i [[thread_position_in_grid]]) {
    out[ids[i]]=run_ids[i];
}

kernel void unique_counts(device long *out, const device long *starts, constant uint &runs,
                          constant uint &rows, uint i [[thread_position_in_grid]]) {
    out[i]=(i+1==runs?long(rows):starts[i+1])-starts[i];
}

kernel void offset_identity(device long *out, uint i [[thread_position_in_grid]]) { out[i]=long(i); }
