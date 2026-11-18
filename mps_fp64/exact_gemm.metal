#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct ExactRange {
    ulong largest;
    long least_bit;
};

// x = significand * 2^base, including unnormalised subnormals.
inline int exact_base(ulong magnitude) {
    return int(max(1u, uint(magnitude >> 52))) - 1075;
}

inline ulong exact_significand(ulong magnitude) {
    return (magnitude & 0x000ffffffffffffful) | (magnitude >> 52 ? (1ul << 52) : 0ul);
}

inline ulong exact_address(uint i, constant long *layout) {
    if (layout[6]) return i;  // the underlying matrix storage is dense, in either order
    uint cols = uint(layout[2]), rows = uint(layout[1]);
    uint rc = i / cols;
    return ulong(rc / rows) * layout[3] + ulong(rc % rows) * layout[4] + ulong(i % cols) * layout[5];
}

inline bool exact_short(ulong x) {
    ulong magnitude = x & 0x7ffffffffffffffful;
    if (!magnitude) return true;
    if (magnitude >= 0x7ff0000000000000ul) return false;
    ulong sig = exact_significand(magnitude);
    return 64 - int(clz(sig)) - int(ctz(sig)) <= 16;
}

// An inexpensive necessary condition. Failing a sample rejects the whole
// specialization; passing it never authorizes the matrix multiply.
kernel void exact_sample(device int *flags, const device ulong *a, const device ulong *b,
                         constant long *layouts, uint lane [[thread_index_in_threadgroup]]) {
    bool ok = true;
    for (uint operand = 0; operand < 2; ++operand) {
        constant long *layout = layouts + operand * 7;
        const device ulong *x = operand ? b : a;
        uint i = uint(ulong(lane) * (ulong(layout[0])-1) / 31ul);
        ulong value = x[exact_address(i,layout)];
        ulong mag = value & 0x7ffffffffffffffful;
        ok = ok && exact_short(value);
        int exponent = mag ? exact_base(mag) + int(ctz(exact_significand(mag))) : 4096;
        int anchor = simd_min(exponent);
        int top = mag ? exact_base(mag) + 63 - int(clz(exact_significand(mag))) : -4096;
        int width = max(0, simd_max(top) - anchor + 1);
        ok = ok && width <= 16;
        if (!lane) { flags[1+operand] = anchor; flags[3+operand] = width; }
    }
    bool whole = simd_all(ok);
    if (!lane) flags[0] = int(whole);
}

// Construct the native payload directly from the original significand. On a
// successful certificate, each payload has <=8 significant bits and magnitude
// between 2^-7 and 255 (or is zero), so this binary16 encoding is exact.
inline half exact_scaled_half(ulong bits, ulong sig, int base, int anchor) {
    int top = 63 - int(clz(sig));
    int exponent = base + top - anchor;
    if (exponent < -14 || exponent > 15) return half(0);  // full validation rejects this scale
    uint fraction = uint(top >= 10 ? sig >> uint(top-10) : sig << uint(10-top)) & 1023u;
    ushort word = ushort((uint(bits >> 63) << 15) | (uint(exponent+15) << 10) | fraction);
    return as_type<half>(word);
}

// The same word observed here supplies both its certificate and its immutable
// packed snapshot. A change to the original after this dispatch cannot make
// the GEMM consume an unvalidated value.
kernel void exact_certify_pack(device half *packed, device ExactRange *out,
                               const device ulong *source, constant long *layout,
                               constant uint &padded_cols, constant uint &padded_rows,
                               constant uint &batches, constant int &anchor,
                               constant uint &groups, constant uint &operand, constant uint &planes,
                               uint grid [[thread_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    ulong largest = 0;
    int least = 4096;
    uint rows = uint(layout[1]), cols = uint(layout[2]);
    bool row_major = (layout[5] == 1 && layout[4] == long(cols)
                     && (batches == 1 || ulong(layout[3]) == ulong(rows)*cols)
                     && padded_cols == cols && padded_rows == rows);
    ulong stop = ulong(padded_rows)*padded_cols*batches;
    for (ulong i = grid; i < stop; i += ulong(groups)*32) {
        ulong bits = 0;
        if (row_major) {
            bits = source[i];
        } else {
            uint rc = uint(i) / padded_cols;
            uint col = uint(i) % padded_cols, row = rc % padded_rows;
            if (row < rows && col < cols) {
                ulong address = ulong(rc/padded_rows)*layout[3] + ulong(row)*layout[4] + ulong(col)*layout[5];
                bits = source[address];
            }
        }
        ulong mag = bits & 0x7ffffffffffffffful;
        largest = max(largest,mag);
        half low = 0, high = 0;
        if (mag) {
            int base = exact_base(mag);
            ulong sig = exact_significand(mag);
            least = min(least,base + int(ctz(sig)));
            if (planes == 1) {
                low = exact_scaled_half(bits,sig,base,anchor);
            } else {
                int shift = base-anchor;
                // An unrepresentable preliminary anchor is never consumed:
                // the host checks the exact common exponent of this snapshot.
                int top = 63 - int(clz(sig)) + shift;
                if (top >= 0 && top < 16) {
                    uint mag_i = uint(shift < 0 ? sig >> uint(-shift) : sig << uint(shift));
                    int signed_i = bits >> 63 ? -int(mag_i) : int(mag_i);
                    int upper = (signed_i + 128) >> 8;
                    high = half(upper);
                    low = half(signed_i - 256*upper);
                }
            }
        }
        packed[i] = low;
        if (planes == 3) {
            packed[stop+i] = high;
            packed[2*stop+i] = low+high;
        }
    }
    uint hi = simd_max(uint(largest >> 32));
    uint lo = simd_max(uint(largest >> 32) == hi ? uint(largest) : 0u);
    least = simd_min(least);
    if (!lane) out[operand*groups+grid/32] = {(ulong(hi)<<32) | lo, least};
}

kernel void exact_finish(device ExactRange *out, const device ExactRange *ranges,
                         constant uint &groups, uint2 pos [[thread_position_in_grid]]) {
    uint lane = pos.x, operand = pos.y;
    ulong largest = 0;
    int least = 4096;
    for (uint i = lane; i < groups; i+=32) {
        ExactRange p = ranges[operand*groups+i];
        largest = max(largest,p.largest);
        least = min(least,int(p.least_bit));
    }
    uint hi = simd_max(uint(largest >> 32));
    uint lo = simd_max(uint(largest >> 32) == hi ? uint(largest) : 0u);
    least = simd_min(least);
    if (!lane) out[operand] = {(ulong(hi)<<32) | lo, least};
}

// The two native inputs can be scaled by sampled anchors above their actual
// common lattice exponents (for example if a rare odd coefficient was missed).
// Convert native FP32 directly to the correspondingly scaled binary64 bits.
inline ulong exact_pack_result(float value, int anchor_sum) {
    uint native = as_type<uint>(value), mag = native & 0x7fffffffu;
    if (!mag) return 0ul;  // ascending FMA, initialized to +0
    int exponent = int(mag >> 23) - 127 + anchor_sum;
    uint fraction = mag & 0x7fffffu;
    ulong bits;
    if (exponent >= -1022) {
        bits = (ulong(exponent+1023) << 52) | (ulong(fraction) << 29);
    } else {
        int shift = exponent+1051;  // significand unit exponent is exponent-23
        ulong mantissa = fraction | (1u<<23);
        bits = shift >= 0 ? mantissa << uint(shift) : mantissa >> uint(-shift);
    }
    return bits | (ulong(native >> 31) << 63);
}

// Four 32-lane SIMD groups per output tile. The elementwise packing at the
// interface is binary16 (every accepted integer is at most 255); matrix
// multiply has an explicitly binary32 accumulator, independent of PyTorch's
// matmul precision flags, MPSGraph settings or a library's autotuner.
kernel void exact_mma(device ulong *out, const device half *a, const device half *b,
                      constant uint &m, constant uint &n, constant uint &mp, constant uint &np,
                      constant uint &kp, constant int &scale,
                      uint3 group [[threadgroup_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    constexpr uint M=32, N=32, K=32;
    threadgroup half left[M*K], right[K*N];
    threadgroup float converted[M*N];
    simdgroup_float8x8 acc[2][2] = {{simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f)},
                                     {simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f)}};
    uint wr=16*(warp/2), wc=16*(warp%2), row0=group.y*M, col0=group.x*N;
    a += (ulong(group.z)*mp+row0)*kp;
    b += ulong(group.z)*kp*np+col0;
    for(uint start=0; start<kp; start+=K) {
        #pragma unroll
        for(uint i=lid; i<M*K; i+=128) {
            left[i]=a[ulong(i/K)*kp+start+i%K];
            right[i]=b[ulong(start+i/N)*np+i%N];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint k=0;k<K;k+=8) {
            simdgroup_half8x8 av[2],bv[2];
            #pragma unroll
            for(uint i=0;i<2;++i) {
                simdgroup_load(av[i],left+(wr+8*i)*K+k,K);
                simdgroup_load(bv[i],right+k*N+wc+8*i,N);
            }
            #pragma unroll
            for(uint i=0;i<2;++i) {
                #pragma unroll
                for(uint j=0;j<2;++j) simdgroup_multiply_accumulate(acc[i][j],av[i],bv[j],acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<2;++i) {
        #pragma unroll
        for(uint j=0;j<2;++j) simdgroup_store(acc[i][j],converted+(wr+8*i)*N+wc+8*j,N);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=lid;i<M*N;i+=128) {
        uint row=row0+i/N, col=col0+i%N;
        if(row<m && col<n)
            out[(ulong(group.z)*m+row)*n+col]=exact_pack_result(converted[i],scale);
    }
}

// At most 53 magnitude bits were admitted by the certificate. Nonzero
// subnormals are integer multiples of 2^-1074, so this pack never rounds.
inline ulong exact_pack_integer(long value, int scale) {
    if (!value) return 0;
    ulong mag = ulong(value < 0 ? -value : value);
    int top = 63 - int(clz(mag)), exponent = scale+top;
    ulong bits;
    if (exponent >= -1022) {
        ulong sig = top <= 52 ? mag << uint(52-top) : mag >> uint(top-52);
        bits = (ulong(exponent+1023)<<52) | (sig & 0x000ffffffffffffful);
    } else {
        bits = mag << uint(scale+1074);
    }
    return bits | (value<0 ? 1ul<<63 : 0ul);
}

// Channel 0 contains the low signed radix-256 digit, channel 1 the high
// digit, and channel 2 their sum. Karatsuba recovers the cross-terms from
// three native products. Each panel's binary32 result is an exact integer.
template<uint channels>
inline void panel_mma(device ulong *out, const device half *a, const device half *b,
                      uint m, uint n, uint mp, uint np, uint kp, int scale, uint panel, uint batches,
                      uint3 group, uint lid, uint warp, threadgroup float *tile) {
    // Each SIMD group produces eight consecutive rows and 32 columns.
    // Adjacent SIMD groups reuse the same four right-hand matrix tiles.
    constexpr uint W=32, cols_per_warp=4;
    uint wr=8*warp, row0=group.y*W, col0=group.x*W;
    ulong astep=ulong(batches)*mp*kp, bstep=ulong(batches)*kp*np;
    a += (ulong(group.z)*mp+row0+wr)*kp;
    b += ulong(group.z)*kp*np+col0;
    long total[8]={};
    for(uint start=0; start<kp; start+=panel) {
        simdgroup_float8x8 acc[channels][cols_per_warp];
        #pragma unroll
        for(uint c=0;c<channels;++c) {
            #pragma unroll
            for(uint j=0;j<cols_per_warp;++j) acc[c][j]=simdgroup_float8x8(0.0f);
        }
        uint end=min(start+panel,kp);
        for(uint k=start;k<end;k+=8) {
            #pragma unroll
            for(uint c=0;c<channels;++c) {
                simdgroup_half8x8 av;
                simdgroup_load(av,a+c*astep+k,kp);
                #pragma unroll
                for(uint j=0;j<cols_per_warp;++j) {
                    simdgroup_half8x8 bv;
                    simdgroup_load(bv,b+c*bstep+ulong(k)*np+8*j,np);
                    simdgroup_multiply_accumulate(acc[c][j],av,bv,acc[c][j]);
                }
            }
        }
        #pragma unroll
        for(uint c=0;c<channels;++c) {
            #pragma unroll
            for(uint j=0;j<cols_per_warp;++j)
                simdgroup_store(acc[c][j],tile+c*W*W+wr*W+8*j,W);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint j=0;j<8;++j) {
            uint i=lid+j*128;
            long low=int(tile[i]);
            if (channels==3) {
                long high=int(tile[W*W+i]), mixed=int(tile[2*W*W+i]);
                total[j]+=low + (mixed-low-high)*256l + high*65536l;
            } else {
                total[j]+=low;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint j=0;j<8;++j) {
        uint i=lid+j*128,row=row0+i/W,col=col0+i%W;
        if(row<m && col<n) out[(ulong(group.z)*m+row)*n+col] = exact_pack_integer(total[j],scale);
    }
}

#define EXACT_PANEL(NAME, CHANNELS) \
kernel void NAME(device ulong *out, const device half *a, const device half *b, \
                 constant uint &m, constant uint &n, constant uint &mp, constant uint &np, \
                 constant uint &kp, constant int &scale, constant uint &panel, constant uint &batches, \
                 uint3 group [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]], \
                 uint warp [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float tile[CHANNELS*32*32]; \
    panel_mma<CHANNELS>(out,a,b,m,n,mp,np,kp,scale,panel,batches,group,lid,warp,tile); \
}
EXACT_PANEL(exact_panel_one,1)
EXACT_PANEL(exact_panel_three,3)
