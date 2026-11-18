// OpenLibm e_lgamma_r.c at 5fe399749f9276eaa0b8403e507470da05cbbb3f.

/* @(#)e_lgamma_r.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 *
 */


namespace ol_e_lgamma_r {
constant Soft
two52= Soft(0x4330000000000000ul,true),
one_half= Soft(0x3fe0000000000000ul,true),
one = Soft(0x3ff0000000000000ul,true),
pi = Soft(0x400921fb54442d18ul,true),
a0 = Soft(0x3fb3c467e37db0c8ul,true),
a1 = Soft(0x3fd4a34cc4a60fadul,true),
a2 = Soft(0x3fb13e001a5562a7ul,true),
a3 = Soft(0x3f951322ac92547bul,true),
a4 = Soft(0x3f7e404fb68fefe8ul,true),
a5 = Soft(0x3f67add8ccb7926bul,true),
a6 = Soft(0x3f538a94116f3f5dul,true),
a7 = Soft(0x3f40b6c689b99c00ul,true),
a8 = Soft(0x3f2cf2eced10e54dul,true),
a9 = Soft(0x3f1c5088987dfb07ul,true),
a10 = Soft(0x3efa7074428cfa52ul,true),
a11 = Soft(0x3f07858e90a45837ul,true),
tc = Soft(0x3ff762d86356be3ful,true),
tf = -Soft(0x3fbf19b9bcc38a42ul,true),
tt = -Soft(0x3c50c7caa48a971ful,true),
t0 = Soft(0x3fdef72bc8ee38a2ul,true),
t1 = -Soft(0x3fc2e4278dc6c509ul,true),
t2 = Soft(0x3fb08b4294d5419bul,true),
t3 = -Soft(0x3fa0c9a8df35b713ul,true),
t4 = Soft(0x3f9266e7970af9ecul,true),
t5 = -Soft(0x3f851f9fba91ec6aul,true),
t6 = Soft(0x3f78fce0e370e344ul,true),
t7 = -Soft(0x3f6e2effb3e914d7ul,true),
t8 = Soft(0x3f6282d32e15c915ul,true),
t9 = -Soft(0x3f56fe8ebf2d1af1ul,true),
t10 = Soft(0x3f4cdf0cef61a8e9ul,true),
t11 = -Soft(0x3f41a6109c73e0ecul,true),
t12 = Soft(0x3f34af6d6c0ebbf7ul,true),
t13 = -Soft(0x3f347f24ecc38c38ul,true),
t14 = Soft(0x3f35fd3ee8c2d3f4ul,true),
u0 = -Soft(0x3fb3c467e37db0c8ul,true),
u1 = Soft(0x3fe4401e8b005dfful,true),
u2 = Soft(0x3ff7475cd119bd6ful,true),
u3 = Soft(0x3fef497644ea8450ul,true),
u4 = Soft(0x3fcd4eaef6010924ul,true),
u5 = Soft(0x3f8b678bbf2bab09ul,true),
v1 = Soft(0x4003a5d7c2bd619cul,true),
v2 = Soft(0x40010725a42b18f5ul,true),
v3 = Soft(0x3fe89dfbe45050aful,true),
v4 = Soft(0x3fbaae55d6537c88ul,true),
v5 = Soft(0x3f6a5abb57d0cf61ul,true),
s0 = -Soft(0x3fb3c467e37db0c8ul,true),
s1 = Soft(0x3fcb848b36e20878ul,true),
s2 = Soft(0x3fd4d98f4f139f59ul,true),
s3 = Soft(0x3fc2bb9cbee5f2f7ul,true),
s4 = Soft(0x3f9b481c7e939961ul,true),
s5 = Soft(0x3f5e26b67368f239ul,true),
s6 = Soft(0x3f00bfecdd17e945ul,true),
r1 = Soft(0x3ff645a762c4ab74ul,true),
r2 = Soft(0x3fe71a1893d3dcdcul,true),
r3 = Soft(0x3fc601edccfbdf27ul,true),
r4 = Soft(0x3f9317ea742ed475ul,true),
r5 = Soft(0x3f497ddaca41a95bul,true),
r6 = Soft(0x3edebaf7a5b38140ul,true),
w0 = Soft(0x3fdacfe390c97d69ul,true),
w1 = Soft(0x3fb555555555553bul,true),
w2 = -Soft(0x3f66c16c16b02e5cul,true),
w3 = Soft(0x3f4a019f98cf38b6ul,true),
w4 = -Soft(0x3f4380cb8c0fe741ul,true),
w5 = Soft(0x3f4b67ba4cdad5d1ul,true),
w6 = -Soft(0x3f5ab89d0b9e43e4ul,true);
constant Soft zero= Soft(0x0000000000000000ul,true);
 inline Soft sin_pi(Soft x)
{
 Soft y,z;
 int n,ix;
 GET_HIGH_WORD(ix,x);
 ix &= 0x7fffffff;
 if(ix<0x3fd00000) return ol_k_sin::__kernel_sin(pi*x,zero,0);
 y = -x;
 z = ol_s_floor::floor(y);
 if(z!=y) {
     y *= Soft(0x3fe0000000000000ul,true);
     y = Soft(0x4000000000000000ul,true)*(y - ol_s_floor::floor(y));
     n = (int) (y*Soft(0x4010000000000000ul,true));
 } else {
            if(ix>=0x43400000) {
                y = zero; n = 0;
            } else {
                if(ix<0x43300000) z = y+two52;
  GET_LOW_WORD(n,z);
  n &= 1;
                y = n;
                n<<= 2;
            }
        }
 switch (n) {
     case 0: y = ol_k_sin::__kernel_sin(pi*y,zero,0); break;
     case 1:
     case 2: y = ol_k_cos::__kernel_cos(pi*(Soft(0x3fe0000000000000ul,true)-y),zero); break;
     case 3:
     case 4: y = ol_k_sin::__kernel_sin(pi*(one-y),zero,0); break;
     case 5:
     case 6: y = -ol_k_cos::__kernel_cos(pi*(y-Soft(0x3ff8000000000000ul,true)),zero); break;
     default: y = ol_k_sin::__kernel_sin(pi*(y-Soft(0x4000000000000000ul,true)),zero,0); break;
     }
 return -y;
}
inline Soft
__ieee754_lgamma_r(Soft x, thread int *signgamp)
{
 Soft t,y,z,nadj,p,p1,p2,p3,q,r,w;
 int32_t hx;
 int i,lx,ix;
 nadj = 0;
 EXTRACT_WORDS(hx,lx,x);
 *signgamp = 1;
 ix = hx&0x7fffffff;
 if(ix>=0x7ff00000) return x*x;
 if((ix|lx)==0) return one/zero;
 if(ix<0x3b900000) {
     if(hx<0) {
         *signgamp = -1;
         return -ol_e_log::__ieee754_log(-x);
     } else return -ol_e_log::__ieee754_log(x);
 }
 if(hx<0) {
     if(ix>=0x43300000)
  return one/zero;
     t = sin_pi(x);
     if(t==zero) return one/zero;
     nadj = ol_e_log::__ieee754_log(pi/fabs(t*x));
     if(t<zero) *signgamp = -1;
     x = -x;
 }
 if((((ix-0x3ff00000)|lx)==0)||(((ix-0x40000000)|lx)==0)) r = 0;
 else if(ix<0x40000000) {
     if(ix<=0x3feccccc) {
  r = -ol_e_log::__ieee754_log(x);
  if(ix>=0x3FE76944) {y = one-x; i= 0;}
  else if(ix>=0x3FCDA661) {y= x-(tc-one); i=1;}
    else {y = x; i=2;}
     } else {
    r = zero;
         if(ix>=0x3FFBB4C3) {y=Soft(0x4000000000000000ul,true)-x;i=0;}
         else if(ix>=0x3FF3B4C4) {y=x-tc;i=1;}
  else {y=x-one;i=2;}
     }
     switch(i) {
       case 0:
  z = y*y;
  p1 = a0+z*(a2+z*(a4+z*(a6+z*(a8+z*a10))));
  p2 = z*(a1+z*(a3+z*(a5+z*(a7+z*(a9+z*a11)))));
  p = y*p1+p2;
  r += (p-Soft(0x3fe0000000000000ul,true)*y); break;
       case 1:
  z = y*y;
  w = z*y;
  p1 = t0+w*(t3+w*(t6+w*(t9 +w*t12)));
  p2 = t1+w*(t4+w*(t7+w*(t10+w*t13)));
  p3 = t2+w*(t5+w*(t8+w*(t11+w*t14)));
  p = z*p1-(tt-w*(p2+y*p3));
  r += (tf + p); break;
       case 2:
  p1 = y*(u0+y*(u1+y*(u2+y*(u3+y*(u4+y*u5)))));
  p2 = one+y*(v1+y*(v2+y*(v3+y*(v4+y*v5))));
  r += (-Soft(0x3fe0000000000000ul,true)*y + p1/p2);
     }
 }
 else if(ix<0x40200000) {
     i = (int)x;
     y = x-(Soft)i;
     p = y*(s0+y*(s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6))))));
     q = one+y*(r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6)))));
     r = one_half*y+p/q;
     z = one;
     switch(i) {
     case 7: z *= (y+Soft(0x4018000000000000ul,true));
     case 6: z *= (y+Soft(0x4014000000000000ul,true));
     case 5: z *= (y+Soft(0x4010000000000000ul,true));
     case 4: z *= (y+Soft(0x4008000000000000ul,true));
     case 3: z *= (y+Soft(0x4000000000000000ul,true));
      r += ol_e_log::__ieee754_log(z); break;
     }
 } else if (ix < 0x43900000) {
     t = ol_e_log::__ieee754_log(x);
     z = one/x;
     y = z*z;
     w = w0+z*(w1+y*(w2+y*(w3+y*(w4+y*(w5+y*w6)))));
     r = (x-one_half)*(t-one)+w;
 } else
     r = x*(ol_e_log::__ieee754_log(x)-one);
 if(hx<0) r = nadj - r;
 return r;
}

}

// The digamma and Euler-Maclaurin coefficients, the inverse-erf seed, and
// the associated interval branches follow PyTorch native/Math.h at the revision
// and under the copyright notices at the beginning of special.metal.
namespace special64 {
constant Soft gamma_pi = Soft(0x400921fb54442d18ul, true);
constant Soft inv_sqrt_pi = Soft(0x3fe20dd750429b6dul, true);
constant Soft epsilon64 = Soft(0x3ca0000000000000ul, true);
constant ulong gamma_digamma_table[] = {
    0x3fb5555555555555ul, 0xbf95995995995996ul, 0x3f7f07c1f07c1f08ul,
    0xbf71111111111111ul, 0x3f70410410410410ul, 0xbf81111111111111ul,
    0x3fb5555555555555ul,
};
constant ulong inv_erf_a[] = {
    0xbfc1fd52e9286f82ul, 0x3fed449b6c715c28ul, 0xbffa535a1fcae55bul,
    0x3fec5bf883814bbcul,
};
constant ulong inv_erf_b[] = {
    0x3f890bf01aa717a9ul, 0xbfd50fef0607abf1ul, 0x3ff7155790be315cul,
    0xc000f270054b995cul, 0x3ff0000000000000ul,
};
constant ulong inv_erf_c[] = {
    0x3ffa42f34d0301d7ul, 0x400b6fc13e8a6585ul, 0xbff9ff9df367e38bul,
    0xbfff888ffff9054bul,
};
constant ulong inv_erf_d[] = {
    0x3ffa316e01655aceul, 0x400c59e294b52e4bul, 0x3ff0000000000000ul,
};
constant ulong hurwitz_den[] = {
    0x4028000000000000ul, 0xc086800000000000ul, 0x40dd880000000000ul,
    0xc132750000000000ul, 0x4186d73000000000ul, 0xc1dc331393146053ul,
    0x423165e990000000ul, 0xc285770bf3df7150ul, 0x42da7b51e18d632bul,
    0xc33055d6586f1431ul, 0x438427237ec0a9deul, 0xc3d8dcd555480632ul,
};
constant ulong factorials[] = {
    0x3ff0000000000000ul, 0x3ff0000000000000ul, 0x4000000000000000ul,
    0x4018000000000000ul, 0x4038000000000000ul, 0x405e000000000000ul,
    0x4086800000000000ul, 0x40b3b00000000000ul, 0x40e3b00000000000ul,
    0x4116260000000000ul, 0x414baf8000000000ul, 0x418308a800000000ul,
    0x41bc8cfc00000000ul, 0x41f7328cc0000000ul, 0x42344c3b28000000ul,
    0x4273077775800000ul, 0x42b3077775800000ul, 0x42f437eeecd80000ul,
    0x4336beecca730000ul, 0x437b02b930689000ul, 0x43c0e1b3be415a00ul,
    0x4406283be9b5c620ul, 0x444e77526159f06cul, 0x4495e5c335f8a4ceul,
    0x44e06c52687a7b9aul, 0x4529a940c33f6121ul, 0x4574d9849ea37eebul,
    0x45c19787e5d9f316ul, 0x460ec92dd23d6967ul, 0x465be6518687a785ul,
    0x46aa27ec6e1f2d0dul, 0x46f956ad0aae33a4ul, 0x474956ad0aae33a4ul,
    0x479a21627303a541ul, 0x47ebc3789a33df96ul, 0x483e5dcbe8a8bc8cul,
    0x489114c2b2deea0ful, 0x48e3c0011ed1bea1ul, 0x493774015499125ful,
    0x498c95619f1a8e64ul, 0x49e1dd5d037098feul, 0x4a36e39f2c684406ul,
    0x4a8e0ac0ea48d948ul, 0x4ae42f399d68f1fcul, 0x4b3bc0ef38704cbbul,
    0x4b9383a833aef5f3ul, 0x4bec0d41ca4b818eul, 0x4c4499bc508f7324ul,
    0x4c9ee69a78d72cb6ul, 0x4cf7a88e4484be3bul, 0x4d527baf2587b49eul,
    0x4dad751f23d047dcul, 0x4e07ef294d193a63ul, 0x4e63d20e33d8e45aul,
    0x4ec0b93bfbbf00acul, 0x4f1cbe5f18b04928ul, 0x4f792693359a4003ul,
    0x4fd6665b1bbd6102ul, 0x50344cc291239feaul, 0x5092b6c35dccd76cul,
    0x50f18b5727f009f5ul, 0x5150b8cf1210c97eul, 0x51b0330899804332ul,
    0x520fe478ee34844aul, 0x526fe478ee34844aul, 0x52d0320568f6ab2eul,
    0x5330b395943e6087ul, 0x53917c0097314d0dul, 0x53f293c0a0a461deul,
    0x5454074bad313983ul, 0x54b5e7fac56dd6e8ul, 0x55184d5a3305da69ul,
    0x557b5705796695b6ul, 0x55df2f423e7902c4ul, 0x564207524c1df599ul,
    0x56a5209471331bd0ul, 0x570916b0466cb107ul, 0x576e2f4c14bac4fcul,
    0x57d264d25ca1d009ul, 0x5836b473aa57bcccul, 0x589c619094edabfful,
    0x5901f5bd7e3e66d7ul, 0x596702dac9bff3c4ul, 0x59cdd7b3bda4f022ul,
    0x5a33958df4743d96ul, 0x5a9a02a088aa61cbul, 0x5b0179c3dbd279b5ul,
    0x5b67c1863ed21d72ul, 0x5bd0550c4b30743eul, 0x5c36b645188f61a6ul,
    0x5c9ff0512a89a152ul, 0x5d06b4d9b43dd8b0ul, 0x5d7051fc798c73bful,
    0x5dd7b722e0a01831ul, 0x5e416a7d9cf591c4ul, 0x5ea9da1274fc845ful,
    0x5f13638dd7bd6347ul, 0x5f7d62e2fafb0a78ul, 0x5fe67fb5c8283404ul,
    0x605166c698cf183bul, 0x60bb30964ec395dcul, 0x612574569a265440ul,
    0x619118b502d68b23ul, 0x61fb83c3509147ecul, 0x62665b0eb1760a70ul,
    0x62d256b20d92d490ul, 0x633e5f96e67b300eul, 0x63a963e824aafa2cul,
    0x64156c4bdef04315ul, 0x64823e389bd89920ul, 0x64ef5af14bdc472ful,
    0x655b30dd3fc905baul, 0x65c7cac197cfe503ul, 0x663500fee805882dul,
    0x66a2b4e306a4ed48ul, 0x6710ce83f7f82d2ful, 0x677e764f3171d1e4ul,
    0x67ebd824633209dbul, 0x6859ab418b722116ul, 0x68c7dd36efa41ac2ul,
    0x69365f6380a9d916ul, 0x69a5262c0fa08f37ul, 0x6a142861fee50880ul,
    0x6a835ece2af0162bul, 0x6af2c3d7b998957aul, 0x6b625340ab3f01f9ul,
    0x6bd209f3a89205f1ul, 0x6c41e5dfc140e1e5ul, 0x6cb1e5dfc140e1e5ul,
    0x6d2209ab80c363a9ul, 0x6d9251d22ec67138ul, 0x6e02bfbd1bdf17dful,
    0x6e7355bb04be109eul, 0x6ee4171452ed7d44ul, 0x6f55082946d09f23ul,
    0x6fc62e9b88b007d7ul, 0x70379185413b0855ul, 0x70a939c09fd12eebul,
    0x711b3243ac4d8695ul, 0x718d88957d1c3026ul, 0x720026b1c06b6a55ul,
    0x7271ca9fcdf65321ul, 0x72e3bcc9487d4439ul, 0x73560ce8defbf238ul,
    0x73c8ce85fadb707eul, 0x743c19f3c62c956ful, 0x74b006cd07056d39ul,
    0x752267cf76103b70ul, 0x75954807e082c4b9ul, 0x7608c5d92b583900ul,
    0x767d07da7ecb62ccul, 0x76f11fa1e0c9f746ul, 0x776455903aefd5a3ul,
    0x77d84e466672ad5dul, 0x784d3e2cb341f894ul, 0x78c1b4a51088f182ul,
    0x793594292c26e656ul, 0x79aa77ba8027b686ul, 0x7a2055e51b1882a7ul,
    0x7a944ab297a8724bul, 0x7b095d5f3d928edeul, 0x7b7fe771cb7257b3ul,
    0x7bf4307602be5b7ful, 0x7c69b5b6477e6884ul, 0x7ce07868c5ccfaf4ul,
    0x7d553b370efa3b7ful, 0x7dcb88cb676c8529ul, 0x7e41f63cb077cadeul,
    0x7eb7932fa79d3a43ul, 0x7f2f2054eb4d96ecul, 0x7fa4ab7864418639ul,
};
inline Soft loggamma(Soft x) { int sign; return ol_e_lgamma_r::__ieee754_lgamma_r(x, &sign); }
inline Soft trunc64(Soft x) { return ol_s_trunc::trunc(x); }
inline Soft pow64(Soft x, Soft y) { return ol_e_pow::__ieee754_pow(x, y); }
inline Soft complementary_erf(Soft x) { return ol_s_erf::erfc(x); }

// Taylor coefficients about the positive zero of psi, rounded from
// (-1)^(k+1) zeta(k+1, root) at 180 decimal digits. A split root preserves
// relative accuracy at the two representable arguments bracketing the zero.
constant ulong digamma_root_coeff[] = {
    0x3f408bdae1a261d4ul, 0xbf482f6345c65b35ul, 0x3f51acebbd761089ul,
    0xbf59d626f71d1f7aul, 0x3f62e23345f79aaful, 0xbf6b9afc7cee8a14ul,
    0x3f742e1acf81d8dcul, 0xbf7d828079282eb8ul, 0x3f85955caaa962f3ul,
    0xbf8f970508e1b6a2ul, 0x3f9723d6807edcc0ul, 0xbfa0fa7ec36a7d8ful,
    0x3fa8fce02b239ca7ul, 0xbfb27baba261cc2cul, 0x3fbb9a5b6370f3abul,
    0xbfc4fc1317257da8ul, 0x3fd08b4294d50381ul, 0xbfdc563b54aa1a35ul,
    0x3feef72bc8ee38acul,
};
inline Soft digamma64(Soft x) {
    if (x == Soft(0)) return copysign(pinf, -x);
    Soft reflection = 0;
    if (x < Soft(0)) {
        if (trunc64(x) == x) return qnan;
        // Reduce at the nearest pole; subtracting a nearby integer is exact.
        Soft f = x - trunc64(x);
        if (f < -half64) f += Soft(1);
        if (fabs(f) != half64) reflection = -gamma_pi / ol_s_tan::tan(gamma_pi * f);
        x = Soft(1) - x;
    }
    if (x > raw(0x3ff6000000000000ul) && x < raw(0x3ff9000000000000ul)) {
        const Soft delta = (x - raw(0x3ff762d86356be3ful)) - raw(0x3c9b86a722197829ul);
        return reflection + delta * horner(delta, digamma_root_coeff);
    }
    Soft result = 0;
    for (uint k = 0; k < 10 && x < Soft(10); ++k) { result -= Soft(1) / x; x += Soft(1); }
    if (x == Soft(10)) return reflection + (result + raw(0x40020396dc85cc95ul));
    const Soft z = Soft(1) / (x * x);
    const Soft correction = z * horner(z, gamma_digamma_table);
    return reflection + ((result + logarithm(x) - half64 / x) - correction);
}

inline Soft inverse_erf(Soft input) {
    const Soft y = fabs(input);
    if (nan(y) || y > Soft(1)) return qnan;
    if (y == Soft(1)) return copysign(pinf, input);
    if (y.v < 0x3e10000000000000ul) return input * raw(0x3fec5bf891b4ef6bul);
    const Soft q = Soft(1) - y;
    const bool tail = y > raw(0x3fe6666666666666ul);
    Soft x;
    if (tail) {
        const Soft z = sqrt(-logarithm(q * half64));
        x = horner(z, inv_erf_c) / horner(z, inv_erf_d);
    } else {
        const Soft z = y*y;
        x = y * horner(z, inv_erf_a) / horner(z, inv_erf_b);
    }
    // The erfc residual retains all bits of 1-y near the endpoints. erf(x)-y
    // can round to zero well before x reaches the root in that interval.
    for (uint i=0; i<4; ++i) {
        const Soft residual = tail ? complementary_erf(x) - q : y - ol_s_erf::erf(x);
        x += residual / ((inv_sqrt_pi + inv_sqrt_pi) * exponential(-(x*x)));
    }
    return copysign(x, input);
}

inline Soft erfcx64(Soft x) {
    if (nan(x)) return qnan;
    const Soft a = fabs(x);
    if (x < Soft(-27)) return pinf;
    if (a.v == pinf.v) return Soft(0);
    const Soft square = a * a;
    Soft v;
    if (a < Soft(8)) {
        const Soft square_lo = fma64(a, a, -square);
        v = exponential(square) * complementary_erf(a) * exponential(square_lo);
    } else {
        // The alternating expansion has a remainder bounded by the first
        // omitted term. At x=8 all 32 terms are decreasing; larger inputs
        // reach a sub-ULP term earlier. Divide by a last to cover DBL_MAX.
        const Soft ratio = (half64/a)/a;
        Soft term = 1, sum = 1;
        for (uint k=1; k<=32; ++k) {
            term *= -integer(2*k-1) * ratio; sum += term;
            if (fabs(term) < epsilon64*half64) break;
        }
        v = (inv_sqrt_pi * sum) / a;
    }
    if (x < Soft(0)) {
        const Soft e = exponential(square) * exponential(fma64(a,a,-square));
        return fma64(Soft(2), e, -v);
    }
    return v;
}

inline Soft scaled_power(Soft a, Soft exponent, Soft factor) {
    const Soft p = pow64(a, exponent);
    if (factor == Soft(1) || p.v >= 0x0010000000000000ul) return factor*p;
    // Avoid a zero intermediate when a high derivative's factorial rescales
    // a tiny inverse power into the representable range.
    // Four equal powers keep the intermediate root normal for every
    // representable final product with a finite binary64 factorial.
    const Soft quarter = pow64(a, exponent * raw(0x3fd0000000000000ul));
    return (((factor * quarter) * quarter) * quarter) * quarter;
}

inline Soft positive_hurwitz(Soft s, Soft q, Soft factor) {
    Soft sum = 0, low = 0, a = q;
    for (uint i=0; i<16; ++i) {
        const Soft term = scaled_power(a, -s, factor);
        if (term.v == pinf.v) return pinf;
        const Soft y = term - low, t = sum + y;
        if (t.v == pinf.v) return pinf;
        low = (t - sum) - y; sum = t;
        a += Soft(1);
        // Bound the complete remaining tail with a first term and the
        // integral from a to infinity. A single tiny term alone is not enough.
        if (sum > Soft(0) && term < epsilon64*sum && a < s) return sum;
    }
    const Soft base = scaled_power(a, -s, factor);
    const Soft integral = scaled_power(a, Soft(1)-s, factor) / (s-Soft(1));
    if (base.v == 0) return sum + integral;
    Soft tail = integral + half64 * base;
    Soft derivative = base * (s/a);
    for (uint k=0; k<12; ++k) {
        const Soft t = derivative / raw(hurwitz_den[k]);
        tail += t;
        if (fabs(t) <= epsilon64 * fabs(tail)) break;
        derivative *= (s+integer(2*k+1))/a; derivative *= (s+integer(2*k+2))/a;
    }
    return sum + (tail-low);
}

inline Soft hurwitz(Soft s, Soft q, Soft factor) {
    if (s == Soft(1)) return pinf;
    if (s < Soft(1)) return qnan;
    if (q <= Soft(0) && trunc64(q) == q) return pinf;
    if (nan(q) || nan(s) || q.v == pinf.v) return qnan;
    if (s.v == pinf.v) return q < Soft(1) ? pinf : q == Soft(1) ? Soft(1) : qnan;
    if (q > Soft(0)) return positive_hurwitz(s, q, factor);
    if (s != trunc64(s)) return qnan;
    const bool odd = s.v < 0x4340000000000000ul && (__softfloat64_cvt_f64_to_i64(s.v,3) & 1);
    const Soft f = trunc64(q)-q, r = Soft(1)-f;
    const Soft far = positive_hurwitz(s, Soft(1)-q, factor);
    if (odd && f == half64) return far;
    if (odd && fabs(f-half64) * s < raw(0x3f747ae147ae147bul)) {
        // Taylor-expand the odd difference about 1/2, eliminating loss of
        // significance when two nearly equal half-lattice sums cancel.
        const Soft d = f - half64;
        Soft coeff = Soft(2) * d * s, delta = 0;
        for (uint k=1; k<=9; k+=2) {
            // Incorporate the small coefficient into the scale before
            // summing; the unscaled high derivative may already overflow.
            const Soft term = positive_hurwitz(s+integer(k), half64, factor*fabs(coeff));
            delta += copysign(term, coeff);
            coeff *= d*(s+integer(k))/integer(k+1);
            coeff *= d*(s+integer(k+1))/integer(k+2);
        }
        return delta + far;
    }
    const Soft pos = positive_hurwitz(s,r,factor);
    const Soft neg = positive_hurwitz(s,f,factor);
    return odd ? (pos-neg)+far : pos+(neg-far);
}
} // namespace special64

kernel void gamma_kernel(device ulong *out, const device ulong *src, constant long *layout,
                         constant uint &kind, constant uint &order, uint i [[thread_position_in_grid]]) {
    const Soft x = special64::raw(src[pointwise_offsets(i,layout).x]);
    Soft v;
    switch (kind) {
        case 0: v=loggamma(x); break;
        case 1: v=digamma64(x); break;
        case 2: v=inverse_erf(x); break;
        case 3: v=erfcx64(x); break;
        case 4: {
            if (order==0) v=digamma64(x);
            else if (order==1) {
                if (x.v == pinf.v) v = Soft(0);
                else if (x.v == 0xfff0000000000000ul) v = qnan;
                else v=hurwitz(Soft(2),x,Soft(1));
            }
            else {
                const Soft factor=raw(factorials[order]);
                v=hurwitz(integer(order+1),x,factor);
                if (!(order & 1)) v=-v;
            }
            break;
        }
        default: {
            v=integer(order*(order-1))*raw(0x3fd250d048e7a1bdul);
            Soft low=0;
            for (uint k=0; k<order; ++k) {
                const Soft term=loggamma(x-integer(k)*half64);
                if (nan(term) || term.v==pinf.v) { v=term; break; }
                const Soft y=term-low,t=v+y; low=(t-v)-y; v=t;
            }
            break;
        }
    }
    out[i]=v.v;
}
kernel void zeta_kernel(device ulong *out, const device ulong *s, const device ulong *q,
                        constant long *layout, uint i [[thread_position_in_grid]]) {
    const ulong2 p=pointwise_offsets(i,layout);
    out[i]=hurwitz(raw(s[p.x]),raw(q[p.y]),Soft(1)).v;
}
