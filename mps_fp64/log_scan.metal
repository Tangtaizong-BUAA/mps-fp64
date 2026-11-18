inline ulong logadd_pair(ulong a, ulong b, bool base2) {
    constexpr ulong inf=0x7ff0000000000000ul, sign=1ul<<63, mag=sign-1;
    if ((a&mag)>inf || (b&mag)>inf) return __softfloat64_fadd(a,b,0);
    ulong largest=__softfloat64_fgt(a,b)?a:b;
    if ((largest&mag)==inf) return largest;
    ulong delta=(__softfloat64_fsub(a,b,0)&mag)|sign;
    ulong e = base2 ? ol_s_exp2::exp2(Soft(delta,true)).v : fast_math::exp_bits<false>(delta);
    ulong tail=ol_s_log1p::log1p(Soft(e,true)).v;
    if (base2) tail=__softfloat64_fdiv(tail,0x3fe62e42fefa39eful,0);
    return __softfloat64_fadd(largest,tail,0);
}

// One lane evaluates a short ordered prefix. Block summaries are scanned recursively.
kernel void log_prefix(device ulong *out, device ulong *totals, const device ulong *x,
                       constant uint &width, constant uint &blocks, uint i [[thread_position_in_grid]]) {
    uint row=i/blocks, start=(i%blocks)*16, end=start+min(16u,width-start);
    ulong base=ulong(row)*width, v=x[base+start];
    out[base+start]=v;
    for (uint j=start+1;j<end;++j) { v=logadd_pair(v,x[base+j],false); out[base+j]=v; }
    if (blocks>1) totals[i]=v;
}

kernel void log_prefix_carry(device ulong *out, const device ulong *prefix,
                             constant uint &width, constant uint &blocks, uint i [[thread_position_in_grid]]) {
    uint b=i%width/16;
    if (b) out[i]=logadd_pair(prefix[ulong(i/width)*blocks+b-1],out[i],false);
}
