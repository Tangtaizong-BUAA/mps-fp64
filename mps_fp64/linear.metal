// Each threadgroup factors one independent row-major matrix with partial pivoting.
kernel void lu_factor(device ulong *matrix, device int *pivots, device int *info,
                      constant uint &rows, constant uint &cols, constant uint &pivoting,
                      uint lane [[thread_index_in_threadgroup]], uint batch [[threadgroup_position_in_grid]]) {
    device ulong *a = matrix + ulong(batch)*rows*cols;
    uint count = min(rows,cols);
    threadgroup uint pivot;
    if (!lane) info[batch] = 0;
    for (uint k=0; k<count; ++k) {
        if (!lane) {
            pivot = k;
            if (pivoting) {
                ulong largest = a[ulong(k)*cols+k] & 0x7ffffffffffffffful;
                for (uint r=k+1; r<rows; ++r) {
                    ulong value = a[ulong(r)*cols+k] & 0x7ffffffffffffffful;
                    if (value>largest) {largest=value; pivot=r;}
                }
                pivots[ulong(batch)*count+k] = int(pivot+1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pivot!=k) for (uint j=lane; j<cols; j+=128) {
            ulong tmp=a[ulong(k)*cols+j];
            a[ulong(k)*cols+j]=a[ulong(pivot)*cols+j];
            a[ulong(pivot)*cols+j]=tmp;
        }
        threadgroup_barrier(mem_flags::mem_device);
        ulong diagonal = a[ulong(k)*cols+k];
        bool zero_pivot = (diagonal << 1) == 0;
        if (zero_pivot && !lane && !info[batch]) info[batch] = int(k+1);
        if (!zero_pivot || !pivoting) {
            // An unpivoted zero diagonal with a nonzero entry underneath
            // has no finite multiplier. Propagate the IEEE divide by zero.
            // A wholly zero active column already has a valid zero update.
            for (uint r=k+1+lane; r<rows; r+=128) {
                ulong value = a[ulong(r)*cols+k];
                if (!zero_pivot || (value << 1) != 0)
                    a[ulong(r)*cols+k]=__softfloat64_fdiv(value,diagonal,0);
            }
            threadgroup_barrier(mem_flags::mem_device);
            ulong tail=cols-k-1;
            for (ulong idx=lane; idx<ulong(rows-k-1)*tail; idx+=128) {
                ulong r=k+1+idx/tail, c=k+1+idx%tail;
                ulong multiplier = a[r*cols+k];
                if (!zero_pivot || (multiplier << 1) != 0)
                    a[r*cols+c]=nearby_fma(multiplier^(1ul<<63),a[ulong(k)*cols+c],a[r*cols+c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

// Each thread solves one right-hand-side column. adjoint selects A^T.
kernel void lu_solve(device ulong *out, const device ulong *matrix, const device int *pivots,
                     constant uint &n, constant uint &nrhs, constant uint &adjoint,
                     uint id [[thread_position_in_grid]]) {
    uint batch=id/nrhs, col=id%nrhs;
    device ulong *x=out+ulong(batch)*n*nrhs+col;
    const device ulong *a=matrix+ulong(batch)*n*n;
    const device int *p=pivots+ulong(batch)*n;
    // Invalid pivots are rejected on the host before dispatch.
    if (!adjoint) {
        for (uint i=0;i<n;++i) {
            uint j=uint(p[i]-1);
            ulong tmp=x[ulong(i)*nrhs];x[ulong(i)*nrhs]=x[ulong(j)*nrhs];x[ulong(j)*nrhs]=tmp;
        }
    }
    for (uint i=0;i<n;++i) {
        ulong v=x[ulong(i)*nrhs];
        for (uint j=0;j<i;++j) {
            ulong coeff=adjoint?a[ulong(j)*n+i]:a[ulong(i)*n+j];
            v=nearby_fma(coeff^(1ul<<63),x[ulong(j)*nrhs],v);
        }
        x[ulong(i)*nrhs]=adjoint?__softfloat64_fdiv(v,a[ulong(i)*n+i],0):v;
    }
    for (uint end=n;end>0;--end) {
        uint i=end-1;
        ulong v=x[ulong(i)*nrhs];
        for (uint j=i+1;j<n;++j) {
            ulong coeff=adjoint?a[ulong(j)*n+i]:a[ulong(i)*n+j];
            v=nearby_fma(coeff^(1ul<<63),x[ulong(j)*nrhs],v);
        }
        x[ulong(i)*nrhs]=adjoint?v:__softfloat64_fdiv(v,a[ulong(i)*n+i],0);
    }
    if (adjoint) for (uint end=n;end>0;--end) {
        uint i=end-1,j=uint(p[i]-1);
        ulong tmp=x[ulong(i)*nrhs];x[ulong(i)*nrhs]=x[ulong(j)*nrhs];x[ulong(j)*nrhs]=tmp;
    }
}
kernel void cholesky_factor(device ulong *a,device int *info,constant uint &n,
                            uint lane [[thread_index_in_threadgroup]],uint batch [[threadgroup_position_in_grid]]) {
    a+=ulong(batch)*n*n;
    threadgroup ulong root;
    threadgroup bool invalid;
    for(uint k=0;k<n;++k) {
        if(!lane) {
            ulong pivot=a[ulong(k)*n+k];
            invalid=!__softfloat64_fgt(pivot,0);
            if(invalid) info[batch]=int(k+1);
            else {root=__softfloat64_fsqrt(pivot,0);a[ulong(k)*n+k]=root;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(invalid) break;
        for(uint i=k+1+lane;i<n;i+=128)
            a[ulong(i)*n+k]=__softfloat64_fdiv(a[ulong(i)*n+k],root,0);
        threadgroup_barrier(mem_flags::mem_device);
        uint tail=n-k-1;
        for(ulong idx=lane;idx<ulong(tail)*tail;idx+=128) {
            uint r=k+1+idx/tail,c=k+1+idx%tail;
            if(c<=r) a[ulong(r)*n+c]=nearby_fma(a[ulong(r)*n+k]^(1ul<<63),a[ulong(c)*n+k],a[ulong(r)*n+c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    for(ulong idx=lane;idx<ulong(n)*n;idx+=128) if(idx%n>idx/n) a[idx]=0;
}
kernel void triangular(device ulong *out,const device ulong *a,constant uint &n,
                       constant uint &cols,constant uint &upper,constant uint &unit,
                       uint id [[thread_position_in_grid]]) {
    uint batch=id/cols;
    out+=ulong(batch)*n*cols+id%cols;
    a+=ulong(batch)*n*n;
    for(uint t=0;t<n;++t) {
        uint i=upper?n-1-t:t;
        ulong value=out[ulong(i)*cols];
        for(uint s=0;s<t;++s) {
            uint j=upper?n-1-s:s;
            value=nearby_fma(a[ulong(i)*n+j]^(1ul<<63),out[ulong(j)*cols],value);
        }
        out[ulong(i)*cols]=unit?value:__softfloat64_fdiv(value,a[ulong(i)*n+i],0);
    }
}

kernel void lu_triangles(device ulong *lower, device ulong *upper, const device ulong *packed,
                         constant uint &rows, constant uint &cols,
                         uint3 pos [[thread_position_in_grid]]) {
    uint c = pos.x, r = pos.y, width = min(rows, cols);
    ulong batch = pos.z;
    ulong value = packed[(batch * rows + r) * cols + c];
    if (c < width)
        lower[(batch * rows + r) * width + c] = r == c ? 0x3ff0000000000000ul : r > c ? value : 0;
    if (r < width)
        upper[(batch * width + r) * cols + c] = r <= c ? value : 0;
}

// Trace one original row through the forward LAPACK swaps. Its final position
// is a column of P, since the factorization is A = P L U.
kernel void lu_permutation(device ulong *out, const device int *pivots,
                           constant uint &rows, constant uint &width,
                           uint2 pos [[thread_position_in_grid]]) {
    uint row = pos.x, column = row;
    const device int *p = pivots + ulong(pos.y) * width;
    for (uint j = 0; j < width; ++j) {
        uint partner = uint(p[j] - 1);
        if (column == j) column = partner;
        else if (column == partner) column = j;
    }
    out[(ulong(pos.y) * rows + row) * rows + column] = 0x3ff0000000000000ul;
}

// Form one column of H[0] H[1] ... H[k-1]. Later reflectors leave e_col
// unchanged. Initialize directly from H[col], then apply the preceding
// reflectors in reverse order. The reflector diagonal is implicitly one.
kernel void householder_columns(device ulong *out, const device ulong *values,
                                const device ulong *tau, constant uint &rows,
                                constant uint &cols, constant uint &count,
                                uint2 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]]) {
    constexpr ulong one = 0x3ff0000000000000ul, sign = 1ul << 63;
    ulong batch = group.y;
    uint col = group.x;
    device ulong *q = out + (batch * cols + col) * rows;
    const device ulong *a = values + batch * cols * rows;
    tau += batch * count;
    bool has_reflector = col < count;
    ulong initial = has_reflector ? tau[col] : 0;
    for (uint row = lane; row < rows; row += 32) {
        ulong value = row == col ? __softfloat64_fsub(one, initial, 0) : 0;
        if (has_reflector && row > col)
            value = __softfloat64_fmul(initial ^ sign, a[ulong(col) * rows + row], 0);
        q[row] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup ulong partial[32];
    for (uint end = min(col, count); end > 0; --end) {
        uint j = end - 1;
        ulong factor = tau[j];
        if ((factor << 1) == 0) continue;
        const device ulong *v = a + ulong(j) * rows;
        ulong sum = !lane ? q[j] : 0;
        for (uint row = j + 1 + lane; row < rows; row += 32)
            sum = nearby_fma(v[row], q[row], sum);
        partial[lane] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 16; stride; stride >>= 1) {
            if (lane < stride)
                partial[lane] = __softfloat64_fadd(partial[lane], partial[lane + stride], 0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        ulong alpha = __softfloat64_fmul(factor ^ sign, partial[0], 0);
        if (!lane) q[j] = __softfloat64_fadd(q[j], alpha, 0);
        for (uint row = j + 1 + lane; row < rows; row += 32)
            q[row] = nearby_fma(alpha, v[row], q[row]);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
