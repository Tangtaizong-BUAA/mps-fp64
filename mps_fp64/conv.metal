// Spatial axes are left-padded with ones for 1D and 2D convolutions.
struct Geometry {
    uint batches, ci, co, groups;
    uint id, ih, iw, od, oh, ow, kd, kh, kw;
    int sd, sh, sw, pd, ph, pw, dd, dh, dw;
    uint transposed, bias;
};
inline int conv_coord(int o,int k,int stride,int pad,int dilation,bool transposed) {
    if (!transposed) return o*stride-pad+k*dilation;
    int p=o+pad-k*dilation;
    return p%stride==0 ? p/stride : -1;
}
inline ulong weight_index(uint oc,uint ic,uint k,constant Geometry &g) {
    ulong offset=g.transposed ? ulong(ic)*(g.co/g.groups)+oc%(g.co/g.groups)
                              : ulong(oc)*(g.ci/g.groups)+ic%(g.ci/g.groups);
    return offset*g.kd*g.kh*g.kw+k;
}
kernel void conv_forward(device ulong *out,const device ulong *x,const device ulong *w,
                         const device ulong *bias,constant Geometry &g,uint i [[thread_position_in_grid]]) {
    uint ox=i%g.ow,oy=(i/g.ow)%g.oh,oz=(i/g.ow/g.oh)%g.od;
    uint oc=(i/g.ow/g.oh/g.od)%g.co,batch=i/g.ow/g.oh/g.od/g.co;
    uint first=(oc/(g.co/g.groups))*(g.ci/g.groups);
    ulong acc=0;
    for(uint ic=first;ic<first+g.ci/g.groups;++ic)
    for(uint kz=0;kz<g.kd;++kz) for(uint ky=0;ky<g.kh;++ky) for(uint kx=0;kx<g.kw;++kx) {
        int iz=conv_coord(oz,kz,g.sd,g.pd,g.dd,g.transposed);
        int iy=conv_coord(oy,ky,g.sh,g.ph,g.dh,g.transposed);
        int ix=conv_coord(ox,kx,g.sw,g.pw,g.dw,g.transposed);
        if(uint(iz)>=g.id || uint(iy)>=g.ih || uint(ix)>=g.iw) continue;
        ulong xi=((((ulong(batch)*g.ci+ic)*g.id+iz)*g.ih+iy)*g.iw+ix);
        uint k=(kz*g.kh+ky)*g.kw+kx;
        acc=nearby_fma(x[xi],w[weight_index(oc,ic,k,g)],acc);
    }
    out[i]=g.bias ? __softfloat64_fadd(acc,bias[oc],0) : acc;
}
kernel void conv_grad_weight(device ulong *out,const device ulong *x,const device ulong *dy,
                             constant Geometry &g,uint i [[thread_position_in_grid]]) {
    uint kx=i%g.kw,ky=i/g.kw%g.kh,kz=i/g.kw/g.kh%g.kd, pair=i/g.kw/g.kh/g.kd;
    uint ic,oc;
    if(g.transposed) {
        ic=pair/(g.co/g.groups); oc=ic/(g.ci/g.groups)*(g.co/g.groups)+pair%(g.co/g.groups);
    } else {
        oc=pair/(g.ci/g.groups); ic=oc/(g.co/g.groups)*(g.ci/g.groups)+pair%(g.ci/g.groups);
    }
    ulong acc=0;
    for(uint b=0;b<g.batches;++b) for(uint oz=0;oz<g.od;++oz)
    for(uint oy=0;oy<g.oh;++oy) for(uint ox=0;ox<g.ow;++ox) {
        int iz=conv_coord(oz,kz,g.sd,g.pd,g.dd,g.transposed);
        int iy=conv_coord(oy,ky,g.sh,g.ph,g.dh,g.transposed);
        int ix=conv_coord(ox,kx,g.sw,g.pw,g.dw,g.transposed);
        if(uint(iz)>=g.id || uint(iy)>=g.ih || uint(ix)>=g.iw) continue;
        ulong xi=((((ulong(b)*g.ci+ic)*g.id+iz)*g.ih+iy)*g.iw+ix);
        ulong yi=((((ulong(b)*g.co+oc)*g.od+oz)*g.oh+oy)*g.ow+ox);
        acc=nearby_fma(x[xi],dy[yi],acc);
    }
    out[i]=acc;
}

kernel void conv_bias(device ulong *out, const device ulong *bias, constant uint &spatial,
                      constant uint &channels, uint i [[thread_position_in_grid]]) {
    out[i] = __softfloat64_fadd(out[i], bias[(i/spatial)%channels], 0);
}
