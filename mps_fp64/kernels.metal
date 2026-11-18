using namespace metal;

inline bool fp_nan(ulong x) { return (x << 1) > 0xffe0000000000000ul; }
inline ulong fp_max(ulong a, ulong b) {
    return fp_nan(a) || (!fp_nan(b) && __softfloat64_fge(a,b)) ? a : b;
}
inline ulong fp_min(ulong a, ulong b) {
    return fp_nan(a) || (!fp_nan(b) && __softfloat64_fle(a,b)) ? a : b;
}
inline ulong combine(ulong a, ulong b, uint op) {
    if (op == 1) return __softfloat64_fmul(a,b,0);
    if (op == 2) return fp_max(a,b);
    if (op == 3) return fp_min(a,b);
    return __softfloat64_fadd(a,b,0);
}
// Each block reduces 256 consecutive columns of one logical row.
kernel void row_sum(device ulong *out, const device ulong *x, constant uint &width,
                    constant uint &blocks, constant uint &op, uint lane [[thread_index_in_threadgroup]],
                    uint block [[threadgroup_position_in_grid]]) {
    threadgroup ulong partial[256];
    uint row = block / blocks;
    uint col = (block % blocks) * 256 + lane;
    ulong identity = op == 1 ? 0x3ff0000000000000ul : op == 2 ? 0xfff0000000000000ul : op == 3 ? 0x7ff0000000000000ul : 0ul;
    partial[lane] = col < width ? x[ulong(row)*width + col] : identity;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (lane < step) partial[lane] = combine(partial[lane], partial[lane+step], op);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) out[block] = partial[0];
}

// Multiple short rows share one SIMD group. In the 256-element tree, all
// pairings above the next power of two combine each leaf with an identity.
// Repeating this pairing has the same effect as doing it once.
template<uint Op> inline void reduce_small_rows(device ulong *out, const device ulong *x,
                                            uint width, uint gid, ushort lane) {
    constexpr ulong identity = Op == 1 ? 0x3ff0000000000000ul : Op == 2 ? 0xfff0000000000000ul :
                               Op == 3 ? 0x7ff0000000000000ul : 0ul;
    const ushort span = 1u << (32 - clz(width - 1u)), within = lane & (span - 1);
    const uint row = gid / span;
    ulong v = within < width ? x[ulong(row) * width + within] : identity;
    if (Op == 0 && (v << 1) == 0) v = 0;
    if (Op < 2 && fp_nan(v)) v = combine(v, identity, Op);
    for (ushort step = span >> 1; step; step >>= 1) {
        const ulong other = as_type<ulong>(simd_shuffle_down(as_type<uint2>(v), step));
        if (within < step) v = combine(v, other, Op);
    }
    if (within == 0) out[row] = v;
}

template<uint Op, bool MatVec> inline ulong reduce_pair(ulong a, ulong b) {
    return MatVec ? gemv_fp64::add(a, b) : combine(a, b, Op);
}

// One SIMD group reproduces the 256-lane reference tree. Its first three
// pairings are vertical in registers; its final five use lane shuffles.
// The same leaves, padding identities and pairings preserve rounding and ties.
template<uint Op, bool MatVec = false> inline void reduce_simd_chunk(
        device ulong *out, const device ulong *x, uint width, uint blocks, uint block, ushort lane,
        const device ulong *vector = nullptr, ulong row_stride = 0, ulong col_stride = 1,
        ulong vector_stride = 1) {
    constexpr ulong identity = Op == 1 ? 0x3ff0000000000000ul : Op == 2 ? 0xfff0000000000000ul :
                               Op == 3 ? 0x7ff0000000000000ul : 0ul;
    const ulong base = ulong(block / blocks) * (MatVec ? row_stride : width);
    const uint col = (block % blocks) * 256 + lane;
    ulong p[8];
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 8; ++k) {
        const ulong ix = ulong(col) + k * 32;
        ulong v = identity;
        if (ix < width) {
            v = x[base + ix * (MatVec ? col_stride : 1ul)];
            if (MatVec) v = gemv_fp64::mul(v, vector[ix * vector_stride]);
        }
        p[k] = v;
    }
    ulong v;
    if (MatVec && width <= 128) {
        // Every live leaf meets padded +0 before meeting any other live leaf.
        // fmul has already quieted every NaN. Normalize a possible -0.
        for (uint i=0; i<4; ++i) if ((p[i] << 1) == 0) p[i] = 0;
        if (width <= 64) v = gemv_fp64::add(p[0], p[1]);
        else v = gemv_fp64::add(gemv_fp64::add(p[0],p[2]), gemv_fp64::add(p[1],p[3]));
    } else {
        const ulong a = reduce_pair<Op,MatVec>(reduce_pair<Op,MatVec>(p[0],p[4]), reduce_pair<Op,MatVec>(p[2],p[6]));
        const ulong b = reduce_pair<Op,MatVec>(reduce_pair<Op,MatVec>(p[1],p[5]), reduce_pair<Op,MatVec>(p[3],p[7]));
        v = reduce_pair<Op,MatVec>(a, b);
    }
    for (ushort step = 16; step; step >>= 1) {
        const ulong other = as_type<ulong>(simd_shuffle_down(as_type<uint2>(v), step));
        if (lane < step) v = reduce_pair<Op,MatVec>(v, other);
    }
    if (lane == 0) out[block] = v;
}
#define FP64_REDUCE_SIMD(name, op) \
kernel void name(device ulong *out, const device ulong *x, constant uint &width, constant uint &blocks, \
                 uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { \
    if (width <= 32) reduce_small_rows<op>(out, x, width, gid, lane); \
    else reduce_simd_chunk<op>(out, x, width, blocks, gid >> 5, lane); \
}
FP64_REDUCE_SIMD(simd_sum, 0)
FP64_REDUCE_SIMD(simd_prod, 1)
FP64_REDUCE_SIMD(simd_max, 2)
FP64_REDUCE_SIMD(simd_min, 3)
#undef FP64_REDUCE_SIMD

// Reduce an interior axis of a contiguous tensor directly. Adjacent lanes
// read adjacent output columns while registers and threadgroup memory perform
// the same 256-leaf tree as the contiguous-row path.
template<uint Op, bool MatVec = false> inline void reduce_column_tile(
        device ulong *out, const device ulong *x, uint width, uint blocks, uint inner,
        uint group, ushort lid, threadgroup ulong *partial, const device ulong *vector = nullptr,
        ulong col_stride = 0, ulong vector_stride = 1) {
    constexpr ulong identity = Op == 1 ? 0x3ff0000000000000ul : Op == 2 ? 0xfff0000000000000ul :
                               Op == 3 ? 0x7ff0000000000000ul : 0ul;
    const uint tiles = (inner - 1) / 16 + 1, row = group / tiles;
    const uint column = (group % tiles) * 16 + (lid & 15), block = row % blocks;
    const uint outer = row / blocks;
    const ushort vertical = lid >> 4;
    const ulong base = ulong(outer) * width * inner + column;
    ulong v[16];
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 16; ++k) {
        const ulong at = ulong(block) * 256 + vertical + k * 16;
        ulong leaf = identity;
        if (at < width && column < inner) {
            leaf = x[base + at * (MatVec ? col_stride : ulong(inner))];
            if (MatVec) leaf = gemv_fp64::mul(leaf, vector[at * vector_stride]);
        }
        v[k] = leaf;
    }
    #pragma clang loop unroll(full)
    for (ushort step = 8; step; step >>= 1) {
        #pragma clang loop unroll(full)
        for (ushort k = 0; k < step; ++k) v[k] = reduce_pair<Op,MatVec>(v[k], v[k + step]);
    }
    ulong value = v[0];
    for (ushort step = 8; step > 1; step >>= 1) {
        partial[lid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vertical < step) value = reduce_pair<Op,MatVec>(value, partial[lid + 16 * step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const ulong other = as_type<ulong>(simd_shuffle_down(as_type<uint2>(value), 16));
    if (vertical == 0 && column < inner) {
        out[(ulong(outer) * inner + column) * blocks + block] = reduce_pair<Op,MatVec>(value, other);
    }
}
#define FP64_REDUCE_COLUMNS(name, op) \
kernel void name(device ulong *out, const device ulong *x, constant uint &width, constant uint &blocks, \
                 constant uint &inner, uint group [[threadgroup_position_in_grid]], ushort lid [[thread_index_in_threadgroup]]) { \
    threadgroup ulong partial[256]; \
    reduce_column_tile<op>(out, x, width, blocks, inner, group, lid, partial); \
}
FP64_REDUCE_COLUMNS(column_sum, 0)
FP64_REDUCE_COLUMNS(column_prod, 1)
FP64_REDUCE_COLUMNS(column_max, 2)
FP64_REDUCE_COLUMNS(column_min, 3)
#undef FP64_REDUCE_COLUMNS

// Public mv in the reference backend evaluates a rounded multiply followed by
// the 256-leaf sum tree. Reading its leaves directly eliminates the full m*k
// binary64 intermediate while retaining every multiplication and add rounding.
kernel void gemv_tree_rows(device ulong *out, const device ulong *a, const device ulong *v,
                           constant ulong *params, uint gid [[thread_position_in_grid]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    reduce_simd_chunk<0, true>(out, a, uint(params[0]), uint(params[1]), gid >> 5,
                              lane, v, params[3], params[4], params[5]);
}
kernel void gemv_tree_columns(device ulong *out, const device ulong *a, const device ulong *v,
                              constant ulong *params, uint group [[threadgroup_position_in_grid]],
                              ushort lid [[thread_index_in_threadgroup]]) {
    threadgroup ulong partial[256];
    reduce_column_tile<0, true>(out, a, uint(params[0]), uint(params[1]), uint(params[2]),
                               group, lid, partial, v, params[4], params[5]);
}

// 8x8 output tile, reusing both inputs in threadgroup memory.
kernel void batch_mm(device ulong *out, const device ulong *a, const device ulong *b,
                     constant uint &m, constant uint &n, constant uint &k,
                     uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup ulong left[8][8], right[8][8];
    uint row = gid.y, col = gid.x;
    ulong batch = gid.z;
    ulong acc = 0;
    for (uint start = 0; start < k; start += 8) {
        uint ka = start + lid.x, kb = start + lid.y;
        left[lid.y][lid.x] = row < m && ka < k ? a[(batch*m+row)*k+ka] : 0;
        right[lid.y][lid.x] = col < n && kb < k ? b[(batch*k+kb)*n+col] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < min(8u,k-start); ++i)
            acc = __softfloat64_fma(left[lid.y][i], right[i][lid.x], acc, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < n) out[(batch*m+row)*n+col] = acc;
}

kernel void bandwidth(device ulong *out, const device ulong *a, const device ulong *b,
                      uint i [[thread_position_in_grid]]) {
    out[i] = a[i] ^ b[i];
}

kernel void from_f32(device ulong *out, const device uint *a, uint i [[thread_position_in_grid]]) {
    out[i] = __softfloat64_cvt_f32_to_f64(a[i]);
}
kernel void to_f32(device uint *out, const device ulong *a, uint i [[thread_position_in_grid]]) {
    out[i] = __softfloat64_cvt_f64_to_f32(a[i],0);
}
kernel void from_i64(device ulong *out, const device long *a, uint i [[thread_position_in_grid]]) {
    out[i] = __softfloat64_cvt_i64_to_f64(a[i],0);
}
kernel void choose(device ulong *out, const device bool *cond, const device ulong *a,
                   const device ulong *b, uint i [[thread_position_in_grid]]) {
    out[i] = cond[i] ? a[i] : b[i];
}
kernel void range64(device ulong *out, constant long& start, constant long& step,
                    uint i [[thread_position_in_grid]]) {
    out[i] = __softfloat64_fma(__softfloat64_cvt_u64_to_f64(i,0),as_type<ulong>(step),as_type<ulong>(start),0);
}
inline ulong fp_round(ulong x) {
    if((x&0x7ffffffffffffffful)>=0x4330000000000000ul) return x;
    long i=__softfloat64_cvt_f64_to_i64(x,0);
    return i==0 ? x & (1ul<<63) : __softfloat64_cvt_i64_to_f64(i,0);
}
inline ulong fp_nextafter(ulong x,ulong y) {
    if(fp_nan(x) || fp_nan(y)) return 0x7ff8000000000000ul;
    if(__softfloat64_feq(x,y)) return y;
    if((x<<1)==0) return (y & (1ul<<63))|1ul;
    return (__softfloat64_flt(x,y) != bool(x>>63)) ? x+1 : x-1;
}
