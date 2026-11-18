// Register-blocked GEMM. All tiles retain the original ascending-k FMA order.
template<uint MR, uint NR, uint K, uint W, uint H, bool nearby>
inline void tile_mm(device ulong *out, const device ulong *a, const device ulong *b,
                    uint m,uint n,uint k,uint3 group,uint3 local,
                    threadgroup ulong *left,threadgroup ulong *right) {
    constexpr uint M=H*MR, N=W*NR;
    uint row=group.y*M, col=group.x*N, lane=local.y*W+local.x;
    ulong batch=group.z;
    ulong accum[MR][NR]={};
    for (uint start=0;start<k;start+=K) {
        for (uint i=lane;i<M*K;i+=W*H) {
            uint r=row+i/K, c=start+i%K;
            left[i] = r<m && c<k ? a[(batch*m+r)*k+c] : 0;
        }
        for (uint i=lane;i<K*N;i+=W*H) {
            uint r=start+i/N,c=col+i%N;
            right[i] = r<k && c<n ? b[(batch*k+r)*n+c] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t=0;t<min(K,k-start);++t) {
            #pragma unroll
            for (uint i=0;i<MR;++i) {
                ulong av=left[(i*H+local.y)*K+t];
                #pragma unroll
                for (uint j=0;j<NR;++j)
                    accum[i][j]=nearby ? wide_nearby_fma(av,right[t*N+j*W+local.x],accum[i][j]) : __softfloat64_fma(av,right[t*N+j*W+local.x],accum[i][j],0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i=0;i<MR;++i) for (uint j=0;j<NR;++j) {
        uint r=row+i*H+local.y,c=col+j*W+local.x;
        if (r<m && c<n) out[(batch*m+r)*n+c]=accum[i][j];
    }
}
#define MAKE_TILE(NAME,MR,NR,K,W,H,NEAR) \
kernel void NAME(device ulong *out,const device ulong *a,const device ulong *b, \
                 constant uint &m,constant uint &n,constant uint &k, \
                 uint3 group [[threadgroup_position_in_grid]],uint3 local [[thread_position_in_threadgroup]]) { \
    threadgroup ulong left[H*MR*K], right[K*W*NR]; \
    tile_mm<MR,NR,K,W,H,NEAR>(out,a,b,m,n,k,group,local,left,right); \
}
MAKE_TILE(mm_2216,2,2,16,8,8,false)
MAKE_TILE(mm_4216,4,2,16,8,8,false)
MAKE_TILE(mm_2116,2,1,16,16,4,false)
MAKE_TILE(mm_118,1,1,8,16,8,false)
MAKE_TILE(mm_1116,1,1,16,16,8,false)
MAKE_TILE(mm_near,1,1,8,16,8,true)

// The staged terms are reused across 8 rows / 16 columns. A thread owns its
// output for the entire dot product, preserving ascending-k binary64 FMA.
template<bool strided>
inline void staged_mm(device ulong *out, const device ulong *a, const device ulong *b,
                      uint m, uint n, uint k, constant ulong *strides,
                      uint3 group, uint3 local, threadgroup DotTerm *left,
                      threadgroup DotTerm *right) {
    constexpr uint W = 16, H = 8, K = 8;
    uint first_row = H * group.y, first_col = W * group.x;
    uint lane = W * local.y + local.x;
    ulong batch = group.z;
    bool transpose_a = strided && strides[1] < strides[2];
    bool transpose_b = strided && strides[4] < strides[5];
    DotTerm sum = unpack_dot_term(0);

    for (uint start = 0; start < k; start += K) {
        // Traverse the smaller stride first when gathering transposed views.
        if (lane < H * K) {
            uint y = transpose_a ? lane % H : lane / K;
            uint t = transpose_a ? lane / H : lane % K;
            uint row = first_row + y, inner = start + t;
            ulong address = strided ? batch * strides[0] + row * strides[1] + inner * strides[2]
                                    : (batch * m + row) * k + inner;
            left[y * K + t] = unpack_dot_term(row < m && inner < k ? a[address] : 0);
        }
        uint t = transpose_b ? lane % K : lane / W;
        uint x = transpose_b ? lane / K : lane % W;
        uint inner = start + t, col = first_col + x;
        ulong address = strided ? batch * strides[3] + inner * strides[4] + col * strides[5]
                                : (batch * k + inner) * n + col;
        right[t * W + x] = unpack_dot_term(inner < k && col < n ? b[address] : 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint offset = 0; offset < min(K, k - start); ++offset)
            sum = dot_fma(left[local.y * K + offset], right[offset * W + local.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row = first_row + local.y, col = first_col + local.x;
    if (row < m && col < n)
        out[(batch * m + row) * n + col] = pack_dot_term(sum);
}

kernel void mm_staged(device ulong *out, const device ulong *a, const device ulong *b,
                      constant uint &m, constant uint &n, constant uint &k,
                      uint3 group [[threadgroup_position_in_grid]],
                      uint3 local [[thread_position_in_threadgroup]]) {
    threadgroup DotTerm left[8 * 8], right[8 * 16];
    staged_mm<false>(out, a, b, m, n, k, nullptr, group, local, left, right);
}

kernel void mm_staged_strided(device ulong *out, const device ulong *a, const device ulong *b,
                              constant uint &m, constant uint &n, constant uint &k,
                              constant ulong *strides,
                              uint3 group [[threadgroup_position_in_grid]],
                              uint3 local [[thread_position_in_threadgroup]]) {
    threadgroup DotTerm left[8 * 8], right[8 * 16];
    staged_mm<true>(out, a, b, m, n, k, strides, group, local, left, right);
}

// One thread per real output avoids the 16-column tile's inactive lanes when
// multiplying a tall matrix by one or a few column vectors.
kernel void mm_narrow_columns(device ulong *out, const device ulong *a, const device ulong *b,
                              constant ulong *p, uint2 gid [[thread_position_in_grid]]) {
    const uint k = uint(p[1]), n = uint(p[2]);
    const ulong at = gid.x, batch = gid.y;
    a += batch * p[3] + (at / n) * p[4];
    b += batch * p[6] + (at % n) * p[8];
    DotTerm acc = unpack_dot_term(0);
    for (uint j = 0; j < k; ++j)
        acc = dot_fma(unpack_dot_term(a[j * p[5]]), unpack_dot_term(b[j * p[7]]), acc);
    out[batch * p[0] * n + at] = pack_dot_term(acc);
}
