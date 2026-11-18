/* @(#)s_floor.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_floor {
constant Soft huge = Soft(0x7e37e43c8800759cul,true);
inline Soft
floor(Soft x)
{
 int32_t i0,i1,j0;
 uint i,j;
 EXTRACT_WORDS(i0,i1,x);
 j0 = ((i0>>20)&0x7ff)-0x3ff;
 if(j0<20) {
     if(j0<0) {
  if(huge+x>Soft(0x0000000000000000ul,true)) {
      if(i0>=0) {i0=i1=0;}
      else if(((i0&0x7fffffff)|i1)!=0)
   { i0=0xbff00000;i1=0;}
  }
     } else {
  i = (0x000fffff)>>j0;
  if(((i0&i)|i1)==0) return x;
  if(huge+x>Soft(0x0000000000000000ul,true)) {
      if(i0<0) i0 += (0x00100000)>>j0;
      i0 &= (~i); i1=0;
  }
     }
 } else if (j0>51) {
     if(j0==0x400) return x+x;
     else return x;
 } else {
     i = ((uint)(0xffffffff))>>(j0-20);
     if((i1&i)==0) return x;
     if(huge+x>Soft(0x0000000000000000ul,true)) {
  if(i0<0) {
      if(j0==20) i0+=1;
      else {
   j = i1+(1<<(52-j0));
   if(j<i1) i0 +=1 ;
   i1=j;
      }
  }
  i1 &= (~i);
     }
 }
 INSERT_WORDS(x,i0,i1);
 return x;
}

}

/* @(#)s_scalbn.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * scalbn (double x, int n)
 * scalbn(x,n) returns x* 2**n  computed by  exponent
 * manipulation rather than by actually performing an
 * exponentiation or a multiplication.
 */


namespace ol_s_scalbn {
constant Soft
two54 = Soft(0x4350000000000000ul,true),
twom54 = Soft(0x3c90000000000000ul,true),
huge = Soft(0x7e37e43c8800759cul,true),
tiny = Soft(0x01a56e1fc2f8f359ul,true);
inline Soft
scalbn (Soft x, int n)
{
 int32_t k,hx,lx;
 EXTRACT_WORDS(hx,lx,x);
        k = (hx&0x7ff00000)>>20;
        if (k==0) {
            if ((lx|(hx&0x7fffffff))==0) return x;
     x *= two54;
     GET_HIGH_WORD(hx,x);
     k = ((hx&0x7ff00000)>>20) - 54;
            if (n< -50000) return tiny*x;
     }
        if (k==0x7ff) return x+x;
        k = k+n;
        if (k > 0x7fe) return huge*copysign(huge,x);
        if (k > 0)
     {SET_HIGH_WORD(x,(hx&0x800fffff)|(k<<20)); return x;}
        if (k <= -54) {
            if (n > 50000)
  return huge*copysign(huge,x);
     else return tiny*copysign(tiny,x);
        }
 k += 54;
 SET_HIGH_WORD(x,(hx&0x800fffff)|(k<<20));
        return x*twom54;
}


}


/* @(#)e_exp.c 1.6 04/04/22 */
/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_exp {
constant Soft
one = Soft(0x3ff0000000000000ul,true),
halF[2] = {Soft(0x3fe0000000000000ul,true),-Soft(0x3fe0000000000000ul,true),},
huge = Soft(0x7e37e43c8800759cul,true),
o_threshold= Soft(0x40862e42fefa39eful,true),
u_threshold= -Soft(0x40874910d52d3051ul,true),
ln2HI[2] ={ Soft(0x3fe62e42fee00000ul,true),
      -Soft(0x3fe62e42fee00000ul,true),},
ln2LO[2] ={ Soft(0x3dea39ef35793c76ul,true),
      -Soft(0x3dea39ef35793c76ul,true),},
invln2 = Soft(0x3ff71547652b82feul,true),
P1 = Soft(0x3fc555555555553eul,true),
P2 = -Soft(0x3f66c16c16bebd93ul,true),
P3 = Soft(0x3f11566aaf25de2cul,true),
P4 = -Soft(0x3ebbbd41c5d26bf1ul,true),
P5 = Soft(0x3e66376972bea4d0ul,true);
constant Soft
twom1000= Soft(0x0170000000000000ul,true);
inline Soft
__ieee754_exp(Soft x)
{
 Soft y,hi=Soft(0x0000000000000000ul,true),lo=Soft(0x0000000000000000ul,true),c,t,twopk;
 int32_t k=0,xsb;
 uint hx;
 GET_HIGH_WORD(hx,x);
 xsb = (hx>>31)&1;
 hx &= 0x7fffffff;
 if(hx >= 0x40862E42) {
            if(hx>=0x7ff00000) {
         uint lx;
  GET_LOW_WORD(lx,x);
  if(((hx&0xfffff)|lx)!=0)
       return x+x;
  else return (xsb==0)? x:Soft(0x0000000000000000ul,true);
     }
     if(x > o_threshold) return huge*huge;
     if(x < u_threshold) return twom1000*twom1000;
 }
        if (x == Soft(0x3ff0000000000000ul,true))
            return Soft(0x4005bf0a8b145769ul,true);
 if(hx > 0x3fd62e42) {
     if(hx < 0x3FF0A2B2) {
  hi = x-ln2HI[xsb]; lo=ln2LO[xsb]; k = 1-xsb-xsb;
     } else {
  k = (int)(invln2*x+halF[xsb]);
  t = k;
  hi = x - t*ln2HI[0];
  lo = t*ln2LO[0];
     }
     STRICT_ASSIGN(Soft, x, hi - lo);
 }
 else if(hx < 0x3e300000) {
     if(huge+x>one) return one+x;
 }
 else k = 0;
 t = x*x;
 if(k >= -1021)
     INSERT_WORDS(twopk,0x3ff00000+(k<<20), 0);
 else
     INSERT_WORDS(twopk,0x3ff00000+((k+1000)<<20), 0);
 c = x - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
 if(k==0) return one-((x*c)/(c-Soft(0x4000000000000000ul,true))-x);
 else y = one-((lo-(x*c)/(Soft(0x4000000000000000ul,true)-c))-hi);
 if(k >= -1021) {
     if (k==1024) return y*Soft(0x4000000000000000ul,true)*Soft(0x7fe0000000000000ul,true);
     return y*twopk;
 } else {
     return y*twopk*twom1000;
 }
}

}


/* @(#)e_log.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_log {
constant Soft
ln2_hi = Soft(0x3fe62e42fee00000ul,true),
ln2_lo = Soft(0x3dea39ef35793c76ul,true),
two54 = Soft(0x4350000000000000ul,true),
Lg1 = Soft(0x3fe5555555555593ul,true),
Lg2 = Soft(0x3fd999999997fa04ul,true),
Lg3 = Soft(0x3fd2492494229359ul,true),
Lg4 = Soft(0x3fcc71c51d8e78aful,true),
Lg5 = Soft(0x3fc7466496cb03deul,true),
Lg6 = Soft(0x3fc39a09d078c69ful,true),
Lg7 = Soft(0x3fc2f112df3e5244ul,true);
constant Soft zero = Soft(0x0000000000000000ul,true);
inline Soft
__ieee754_log(Soft x)
{
 Soft hfsq,f,s,z,R,w,t1,t2,dk;
 int32_t k,hx,i,j;
 uint lx;
 EXTRACT_WORDS(hx,lx,x);
 k=0;
 if (hx < 0x00100000) {
     if (((hx&0x7fffffff)|lx)==0)
  return -two54/zero;
     if (hx<0) return (x-x)/zero;
     k -= 54; x *= two54;
     GET_HIGH_WORD(hx,x);
 }
 if (hx >= 0x7ff00000) return x+x;
 k += (hx>>20)-1023;
 hx &= 0x000fffff;
 i = (hx+0x95f64)&0x100000;
 SET_HIGH_WORD(x,hx|(i^0x3ff00000));
 k += (i>>20);
 f = x-Soft(0x3ff0000000000000ul,true);
 if((0x000fffff&(2+hx))<3) {
     if(f==zero) {
  if(k==0) {
      return zero;
  } else {
      dk=(Soft)k;
      return dk*ln2_hi+dk*ln2_lo;
  }
     }
     R = f*f*(Soft(0x3fe0000000000000ul,true)-Soft(0x3fd5555555555555ul,true)*f);
     if(k==0) return f-R; else {dk=(Soft)k;
           return dk*ln2_hi-((R-dk*ln2_lo)-f);}
 }
  s = f/(Soft(0x4000000000000000ul,true)+f);
 dk = (Soft)k;
 z = s*s;
 i = hx-0x6147a;
 w = z*z;
 j = 0x6b851-hx;
 t1= w*(Lg2+w*(Lg4+w*Lg6));
 t2= z*(Lg1+w*(Lg3+w*(Lg5+w*Lg7)));
 i |= j;
 R = t2+t1;
 if(i>0) {
     hfsq=Soft(0x3fe0000000000000ul,true)*f*f;
     if(k==0) return f-(hfsq-s*(hfsq+R)); else
       return dk*ln2_hi-((hfsq-(s*(hfsq+R)+dk*ln2_lo))-f);
 } else {
     if(k==0) return f-s*(f-R); else
       return dk*ln2_hi-((s*(f-R)-dk*ln2_lo)-f);
 }
}

}

/* @(#)s_expm1.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_expm1 {
constant Soft
one = Soft(0x3ff0000000000000ul,true),
huge = Soft(0x7e37e43c8800759cul,true),
tiny = Soft(0x01a56e1fc2f8f359ul,true),
o_threshold = Soft(0x40862e42fefa39eful,true),
ln2_hi = Soft(0x3fe62e42fee00000ul,true),
ln2_lo = Soft(0x3dea39ef35793c76ul,true),
invln2 = Soft(0x3ff71547652b82feul,true),
Q1 = -Soft(0x3fa11111111110f4ul,true),
Q2 = Soft(0x3f5a01a019fe5585ul,true),
Q3 = -Soft(0x3f14ce199eaadbb7ul,true),
Q4 = Soft(0x3ed0cfca86e65239ul,true),
Q5 = -Soft(0x3e8afdb76e09c32dul,true);
inline Soft
expm1(Soft x)
{
 Soft y,hi,lo,c,t,e,hxs,hfx,r1,twopk;
 int32_t k,xsb;
 uint hx;
 GET_HIGH_WORD(hx,x);
 xsb = hx&0x80000000;
 hx &= 0x7fffffff;
 if(hx >= 0x4043687A) {
     if(hx >= 0x40862E42) {
                if(hx>=0x7ff00000) {
      uint low;
      GET_LOW_WORD(low,x);
      if(((hx&0xfffff)|low)!=0)
           return x+x;
      else return (xsb==0)? x:-Soft(0x3ff0000000000000ul,true);
         }
         if(x > o_threshold) return huge*huge;
     }
     if(xsb!=0) {
  if(x+tiny<Soft(0x0000000000000000ul,true))
  return tiny-one;
     }
 }
 if(hx > 0x3fd62e42) {
     if(hx < 0x3FF0A2B2) {
  if(xsb==0)
      {hi = x - ln2_hi; lo = ln2_lo; k = 1;}
  else
      {hi = x + ln2_hi; lo = -ln2_lo; k = -1;}
     } else {
  k = int(invln2*x+((xsb==0)?Soft(0x3fe0000000000000ul,true):-Soft(0x3fe0000000000000ul,true)));
  t = k;
  hi = x - t*ln2_hi;
  lo = t*ln2_lo;
     }
     STRICT_ASSIGN(Soft, x, hi - lo);
     c = (hi-x)-lo;
 }
 else if(hx < 0x3c900000) {
     t = huge+x;
     return x - (t-(huge+x));
 }
 else k = 0;
 hfx = Soft(0x3fe0000000000000ul,true)*x;
 hxs = x*hfx;
 r1 = one+hxs*(Q1+hxs*(Q2+hxs*(Q3+hxs*(Q4+hxs*Q5))));
 t = Soft(0x4008000000000000ul,true)-r1*hfx;
 e = hxs*((r1-t)/(Soft(0x4018000000000000ul,true) - x*t));
 if(k==0) return x - (x*e-hxs);
 else {
     INSERT_WORDS(twopk,0x3ff00000+(k<<20),0);
     e = (x*(e-c)-c);
     e -= hxs;
     if(k== -1) return Soft(0x3fe0000000000000ul,true)*(x-e)-Soft(0x3fe0000000000000ul,true);
     if(k==1) {
         if(x < -Soft(0x3fd0000000000000ul,true)) return -Soft(0x4000000000000000ul,true)*(e-(x+Soft(0x3fe0000000000000ul,true)));
         else return one+Soft(0x4000000000000000ul,true)*(x-e);
     }
     if (k <= -2 || k>56) {
         y = one-(e-x);
  if (k == 1024) y = y*Soft(0x4000000000000000ul,true)*Soft(0x7fe0000000000000ul,true);
  else y = y*twopk;
         return y-one;
     }
     t = one;
     if(k<20) {
         SET_HIGH_WORD(t,0x3ff00000 - (0x200000>>k));
         y = t-(e-x);
  y = y*twopk;
    } else {
  SET_HIGH_WORD(t,((0x3ff-k)<<20));
         y = x-(e+t);
         y += one;
  y = y*twopk;
     }
 }
 return y;
}

}

/* @(#)s_log1p.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_log1p {
constant Soft
ln2_hi = Soft(0x3fe62e42fee00000ul,true),
ln2_lo = Soft(0x3dea39ef35793c76ul,true),
two54 = Soft(0x4350000000000000ul,true),
Lp1 = Soft(0x3fe5555555555593ul,true),
Lp2 = Soft(0x3fd999999997fa04ul,true),
Lp3 = Soft(0x3fd2492494229359ul,true),
Lp4 = Soft(0x3fcc71c51d8e78aful,true),
Lp5 = Soft(0x3fc7466496cb03deul,true),
Lp6 = Soft(0x3fc39a09d078c69ful,true),
Lp7 = Soft(0x3fc2f112df3e5244ul,true);
constant Soft zero = Soft(0x0000000000000000ul,true);
inline Soft
log1p(Soft x)
{
 Soft hfsq,f,c,s,z,R,u;
 int32_t k,hx,hu,ax;
 GET_HIGH_WORD(hx,x);
 ax = hx&0x7fffffff;
 k = 1;
 if (hx < 0x3FDA827A) {
     if(ax>=0x3ff00000) {
  if(x==-Soft(0x3ff0000000000000ul,true)) return -two54/zero;
  else return (x-x)/(x-x);
     }
     if(ax<0x3e200000) {
  if(two54+x>zero
             &&ax<0x3c900000)
      return x;
  else
      return x - x*x*Soft(0x3fe0000000000000ul,true);
     }
     if(hx>0||hx<=((int32_t)0xbfd2bec4)) {
  k=0;f=x;hu=1;}
 }
 if (hx >= 0x7ff00000) return x+x;
 if(k!=0) {
     if(hx<0x43400000) {
  STRICT_ASSIGN(Soft,u,Soft(0x3ff0000000000000ul,true)+x);
  GET_HIGH_WORD(hu,u);
         k = (hu>>20)-1023;
         c = (k>0)? Soft(0x3ff0000000000000ul,true)-(u-x):x-(u-Soft(0x3ff0000000000000ul,true));
  c /= u;
     } else {
  u = x;
  GET_HIGH_WORD(hu,u);
         k = (hu>>20)-1023;
  c = 0;
     }
     hu &= 0x000fffff;
     if(hu<0x6a09e) {
         SET_HIGH_WORD(u,hu|0x3ff00000);
     } else {
         k += 1;
  SET_HIGH_WORD(u,hu|0x3fe00000);
         hu = (0x00100000-hu)>>2;
     }
     f = u-Soft(0x3ff0000000000000ul,true);
 }
 hfsq=Soft(0x3fe0000000000000ul,true)*f*f;
 if(hu==0) {
     if(f==zero) {
  if(k==0) {
      return zero;
  } else {
      c += k*ln2_lo;
      return k*ln2_hi+c;
  }
     }
     R = hfsq*(Soft(0x3ff0000000000000ul,true)-Soft(0x3fe5555555555555ul,true)*f);
     if(k==0) return f-R; else
           return k*ln2_hi-((R-(k*ln2_lo+c))-f);
 }
  s = f/(Soft(0x4000000000000000ul,true)+f);
 z = s*s;
 R = z*(Lp1+z*(Lp2+z*(Lp3+z*(Lp4+z*(Lp5+z*(Lp6+z*Lp7))))));
 if(k==0) return f-(hfsq-s*(hfsq+R)); else
   return k*ln2_hi-((hfsq-(s*(hfsq+R)+(k*ln2_lo+c)))-f);
}

}

/* @(#)s_atan.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_atan {
constant Soft atanhi[] = {
  Soft(0x3fddac670561bb4ful,true),
  Soft(0x3fe921fb54442d18ul,true),
  Soft(0x3fef730bd281f69bul,true),
  Soft(0x3ff921fb54442d18ul,true),
};
constant Soft atanlo[] = {
  Soft(0x3c7a2b7f222f65e2ul,true),
  Soft(0x3c81a62633145c07ul,true),
  Soft(0x3c7007887af0cbbdul,true),
  Soft(0x3c91a62633145c07ul,true),
};
constant Soft aT[] = {
  Soft(0x3fd555555555550dul,true),
 -Soft(0x3fc999999998ebc4ul,true),
  Soft(0x3fc24924920083fful,true),
 -Soft(0x3fbc71c6fe231671ul,true),
  Soft(0x3fb745cdc54c206eul,true),
 -Soft(0x3fb3b0f2af749a6dul,true),
  Soft(0x3fb10d66a0d03d51ul,true),
 -Soft(0x3fadde2d52defd9aul,true),
  Soft(0x3fa97b4b24760debul,true),
 -Soft(0x3fa2b4442c6a6c2ful,true),
  Soft(0x3f90ad3ae322da11ul,true),
};
 constant Soft
one = Soft(0x3ff0000000000000ul,true),
huge = Soft(0x7e37e43c8800759cul,true);
inline Soft
atan(Soft x)
{
 Soft w,s1,s2,z;
 int32_t ix,hx,id;
 GET_HIGH_WORD(hx,x);
 ix = hx&0x7fffffff;
 if(ix>=0x44100000) {
     uint low;
     GET_LOW_WORD(low,x);
     if(ix>0x7ff00000||
  (ix==0x7ff00000&&(low!=0)))
  return x+x;
     if(hx>0) return atanhi[3]+atanlo[3];
     else return -atanhi[3]-atanlo[3];
 } if (ix < 0x3fdc0000) {
     if (ix < 0x3e400000) {
  if(huge+x>one) return x;
     }
     id = -1;
 } else {
 x = fabs(x);
 if (ix < 0x3ff30000) {
     if (ix < 0x3fe60000) {
  id = 0; x = (Soft(0x4000000000000000ul,true)*x-one)/(Soft(0x4000000000000000ul,true)+x);
     } else {
  id = 1; x = (x-one)/(x+one);
     }
 } else {
     if (ix < 0x40038000) {
  id = 2; x = (x-Soft(0x3ff8000000000000ul,true))/(one+Soft(0x3ff8000000000000ul,true)*x);
     } else {
  id = 3; x = -Soft(0x3ff0000000000000ul,true)/x;
     }
 }}
 z = x*x;
 w = z*z;
 s1 = z*(aT[0]+w*(aT[2]+w*(aT[4]+w*(aT[6]+w*(aT[8]+w*aT[10])))));
 s2 = w*(aT[1]+w*(aT[3]+w*(aT[5]+w*(aT[7]+w*aT[9]))));
 if (id<0) return x - x*(s1+s2);
 else {
     z = atanhi[id] - ((x*(s1+s2) - atanlo[id]) - x);
     return (hx<0)? -z:z;
 }
}

}


/* @(#)e_atan2.c 1.3 95/01/18 */
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


namespace ol_e_atan2 {
constant Soft
tiny = Soft(0x01a56e1fc2f8f359ul,true);
constant Soft
zero = Soft(0x0000000000000000ul,true),
pi_o_4 = Soft(0x3fe921fb54442d18ul,true),
pi_o_2 = Soft(0x3ff921fb54442d18ul,true),
pi = Soft(0x400921fb54442d18ul,true);
constant Soft
pi_lo = Soft(0x3ca1a62633145c07ul,true);
inline Soft
__ieee754_atan2(Soft y, Soft x)
{
 Soft z;
 int32_t k,m,hx,hy,ix,iy;
 uint lx,ly;
 EXTRACT_WORDS(hx,lx,x);
 ix = hx&0x7fffffff;
 EXTRACT_WORDS(hy,ly,y);
 iy = hy&0x7fffffff;
 if(((ix|((lx|-lx)>>31))>0x7ff00000)||
    ((iy|((ly|-ly)>>31))>0x7ff00000))
    return x+y;
 if(hx==0x3ff00000&&lx==0) return ol_s_atan::atan(y);
 m = ((hy>>31)&1)|((hx>>30)&2);
 if((iy|ly)==0) {
     switch(m) {
  case 0:
  case 1: return y;
  case 2: return pi+tiny;
  case 3: return -pi-tiny;
     }
 }
 if((ix|lx)==0) return (hy<0)? -pi_o_2-tiny: pi_o_2+tiny;
 if(ix==0x7ff00000) {
     if(iy==0x7ff00000) {
  switch(m) {
      case 0: return pi_o_4+tiny;
      case 1: return -pi_o_4-tiny;
      case 2: return Soft(0x4008000000000000ul,true)*pi_o_4+tiny;
      case 3: return -Soft(0x4008000000000000ul,true)*pi_o_4-tiny;
  }
     } else {
  switch(m) {
      case 0: return zero ;
      case 1: return -zero ;
      case 2: return pi+tiny ;
      case 3: return -pi-tiny ;
  }
     }
 }
 if(iy==0x7ff00000) return (hy<0)? -pi_o_2-tiny: pi_o_2+tiny;
 k = (iy-ix)>>20;
 if(k > 60) {
     z=pi_o_2+Soft(0x3fe0000000000000ul,true)*pi_lo;
     m&=1;
 }
 else if(hx<0&&k<-60) z=Soft(0x0000000000000000ul,true);
 else z=ol_s_atan::atan(fabs(y/x));
 switch (m) {
     case 0: return z ;
     case 1: return -z ;
     case 2: return pi-(z-pi_lo);
     default:
          return (z-pi_lo)-pi;
 }
}

}


/* @(#)e_asin.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_asin {
constant Soft
one = Soft(0x3ff0000000000000ul,true),
huge = Soft(0x7e37e43c8800759cul,true),
pio2_hi = Soft(0x3ff921fb54442d18ul,true),
pio2_lo = Soft(0x3c91a62633145c07ul,true),
pio4_hi = Soft(0x3fe921fb54442d18ul,true),
pS0 = Soft(0x3fc5555555555555ul,true),
pS1 = -Soft(0x3fd4d61203eb6f7dul,true),
pS2 = Soft(0x3fc9c1550e884455ul,true),
pS3 = -Soft(0x3fa48228b5688f3bul,true),
pS4 = Soft(0x3f49efe07501b288ul,true),
pS5 = Soft(0x3f023de10dfdf709ul,true),
qS1 = -Soft(0x40033a271c8a2d4bul,true),
qS2 = Soft(0x40002ae59c598ac8ul,true),
qS3 = -Soft(0x3fe6066c1b8d0159ul,true),
qS4 = Soft(0x3fb3b8c5b12e9282ul,true);
inline Soft
__ieee754_asin(Soft x)
{
 Soft t=Soft(0x0000000000000000ul,true),w,p,q,c,r,s;
 int32_t hx,ix;
 GET_HIGH_WORD(hx,x);
 ix = hx&0x7fffffff;
 if(ix>= 0x3ff00000) {
     uint lx;
     GET_LOW_WORD(lx,x);
     if(((ix-0x3ff00000)|lx)==0)
  return x*pio2_hi+x*pio2_lo;
     return (x-x)/(x-x);
 } else if (ix<0x3fe00000) {
     if(ix<0x3e500000) {
  if(huge+x>one) return x;
     }
     t = x*x;
     p = t*(pS0+t*(pS1+t*(pS2+t*(pS3+t*(pS4+t*pS5)))));
     q = one+t*(qS1+t*(qS2+t*(qS3+t*qS4)));
     w = p/q;
     return x+x*w;
 }
 w = one-fabs(x);
 t = w*Soft(0x3fe0000000000000ul,true);
 p = t*(pS0+t*(pS1+t*(pS2+t*(pS3+t*(pS4+t*pS5)))));
 q = one+t*(qS1+t*(qS2+t*(qS3+t*qS4)));
 s = sqrt(t);
 if(ix>=0x3FEF3333) {
     w = p/q;
     t = pio2_hi-(Soft(0x4000000000000000ul,true)*(s+s*w)-pio2_lo);
 } else {
     w = s;
     SET_LOW_WORD(w,0);
     c = (t-w*w)/(s+w);
     r = p/q;
     p = Soft(0x4000000000000000ul,true)*s*r-(pio2_lo-Soft(0x4000000000000000ul,true)*c);
     q = pio4_hi-Soft(0x4000000000000000ul,true)*w;
     t = pio4_hi-(p-q);
 }
 if(hx>0) return t; else return -t;
}

}


/* @(#)e_acos.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_acos {
constant Soft
one= Soft(0x3ff0000000000000ul,true),
pi = Soft(0x400921fb54442d18ul,true),
pio2_hi = Soft(0x3ff921fb54442d18ul,true);
constant Soft
pio2_lo = Soft(0x3c91a62633145c07ul,true);
constant Soft
pS0 = Soft(0x3fc5555555555555ul,true),
pS1 = -Soft(0x3fd4d61203eb6f7dul,true),
pS2 = Soft(0x3fc9c1550e884455ul,true),
pS3 = -Soft(0x3fa48228b5688f3bul,true),
pS4 = Soft(0x3f49efe07501b288ul,true),
pS5 = Soft(0x3f023de10dfdf709ul,true),
qS1 = -Soft(0x40033a271c8a2d4bul,true),
qS2 = Soft(0x40002ae59c598ac8ul,true),
qS3 = -Soft(0x3fe6066c1b8d0159ul,true),
qS4 = Soft(0x3fb3b8c5b12e9282ul,true);
inline Soft
__ieee754_acos(Soft x)
{
 Soft z,p,q,r,w,s,c,df;
 int32_t hx,ix;
 GET_HIGH_WORD(hx,x);
 ix = hx&0x7fffffff;
 if(ix>=0x3ff00000) {
     uint lx;
     GET_LOW_WORD(lx,x);
     if(((ix-0x3ff00000)|lx)==0) {
  if(hx>0) return Soft(0x0000000000000000ul,true);
  else return pi+Soft(0x4000000000000000ul,true)*pio2_lo;
     }
     return (x-x)/(x-x);
 }
 if(ix<0x3fe00000) {
     if(ix<=0x3c600000) return pio2_hi+pio2_lo;
     z = x*x;
     p = z*(pS0+z*(pS1+z*(pS2+z*(pS3+z*(pS4+z*pS5)))));
     q = one+z*(qS1+z*(qS2+z*(qS3+z*qS4)));
     r = p/q;
     return pio2_hi - (x - (pio2_lo-x*r));
 } else if (hx<0) {
     z = (one+x)*Soft(0x3fe0000000000000ul,true);
     p = z*(pS0+z*(pS1+z*(pS2+z*(pS3+z*(pS4+z*pS5)))));
     q = one+z*(qS1+z*(qS2+z*(qS3+z*qS4)));
     s = sqrt(z);
     r = p/q;
     w = r*s-pio2_lo;
     return pi - Soft(0x4000000000000000ul,true)*(s+w);
 } else {
     z = (one-x)*Soft(0x3fe0000000000000ul,true);
     s = sqrt(z);
     df = s;
     SET_LOW_WORD(df,0);
     c = (z-df*df)/(s+df);
     p = z*(pS0+z*(pS1+z*(pS2+z*(pS3+z*(pS4+z*pS5)))));
     q = one+z*(qS1+z*(qS2+z*(qS3+z*qS4)));
     r = p/q;
     w = r*s+c;
     return Soft(0x4000000000000000ul,true)*(df+w);
 }
}

}

/* @(#)s_tanh.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_tanh {
constant Soft one = Soft(0x3ff0000000000000ul,true), two = Soft(0x4000000000000000ul,true), tiny = Soft(0x01a56e1fc2f8f359ul,true), huge = Soft(0x7e37e43c8800759cul,true);
inline Soft
tanh(Soft x)
{
 Soft t,z;
 int32_t jx,ix;
 GET_HIGH_WORD(jx,x);
 ix = jx&0x7fffffff;
 if(ix>=0x7ff00000) {
     if (jx>=0) return one/x+one;
     else return one/x-one;
 }
 if (ix < 0x40360000) {
     if (ix<0x3e300000) {
  if(huge+x>one) return x;
     }
     if (ix>=0x3ff00000) {
  t = ol_s_expm1::expm1(two*fabs(x));
  z = one - two/(t+two);
     } else {
         t = ol_s_expm1::expm1(-two*fabs(x));
         z= -t/(t+two);
     }
 } else {
     z = one - tiny;
 }
 return (jx>=0)? z: -z;
}

}


/* @(#)k_sin.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_k_sin {
constant Soft
one_half = Soft(0x3fe0000000000000ul,true),
S1 = -Soft(0x3fc5555555555549ul,true),
S2 = Soft(0x3f8111111110f8a6ul,true),
S3 = -Soft(0x3f2a01a019c161d5ul,true),
S4 = Soft(0x3ec71de357b1fe7dul,true),
S5 = -Soft(0x3e5ae5e68a2b9cebul,true),
S6 = Soft(0x3de5d93a5acfd57cul,true);
inline Soft
__kernel_sin(Soft x, Soft y, int iy)
{
 Soft z,r,v,w;
 z = x*x;
 w = z*z;
 r = S2+z*(S3+z*S4) + z*w*(S5+z*S6);
 v = z*x;
 if(iy==0) return x+v*(S1+z*r);
 else return x-((z*(one_half*y-v*r)-y)-v*S1);
}

}


/* @(#)k_cos.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_k_cos {
constant Soft
one = Soft(0x3ff0000000000000ul,true),
C1 = Soft(0x3fa555555555554cul,true),
C2 = -Soft(0x3f56c16c16c15177ul,true),
C3 = Soft(0x3efa01a019cb1590ul,true),
C4 = -Soft(0x3e927e4f809c52adul,true),
C5 = Soft(0x3e21ee9ebdb4b1c4ul,true),
C6 = -Soft(0x3da8fae9be8838d4ul,true);
inline Soft
__kernel_cos(Soft x, Soft y)
{
 Soft hz,z,r,w;
 z = x*x;
 w = z*z;
 r = z*(C1+z*(C2+z*C3)) + w*w*(C4+z*(C5+z*C6));
 hz = Soft(0x3fe0000000000000ul,true)*z;
 w = one-hz;
 return w + (((one-w)-hz) + (z*r-x*y));
}

}

/* @(#)k_tan.c 1.5 04/04/22 SMI */

/*
 * ====================================================
 * Copyright 2004 Sun Microsystems, Inc.  All Rights Reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* INDENT OFF */

namespace ol_k_tan {
constant Soft xxx[] = {
   Soft(0x3fd5555555555563ul,true),
   Soft(0x3fc111111110fe7aul,true),
   Soft(0x3faba1ba1bb341feul,true),
   Soft(0x3f9664f48406d637ul,true),
   Soft(0x3f8226e3e96e8493ul,true),
   Soft(0x3f6d6d22c9560328ul,true),
   Soft(0x3f57dbc8fee08315ul,true),
   Soft(0x3f4344d8f2f26501ul,true),
   Soft(0x3f3026f71a8d1068ul,true),
   Soft(0x3f147e88a03792a6ul,true),
   Soft(0x3f12b80f32f0a7e9ul,true),
  -Soft(0x3ef375cbdb605373ul,true),
   Soft(0x3efb2a7074bf7ad4ul,true),
           Soft(0x3ff0000000000000ul,true),
            Soft(0x3fe921fb54442d18ul,true),
              Soft(0x3c81a62633145c07ul,true)
};
Soft
__kernel_tan(Soft x, Soft y, int iy) {
 Soft z, r, v, w, s;
 int32_t ix, hx;
 GET_HIGH_WORD(hx,x);
 ix = hx & 0x7fffffff;
 if (ix >= 0x3FE59428) {
  if (hx < 0) {
   x = -x;
   y = -y;
  }
  z = xxx[14] - x;
  w = xxx[15] - y;
  x = z + w;
  y = Soft(0x0000000000000000ul,true);
 }
 z = x * x;
 w = z * z;
 r = xxx[1] + w * (xxx[3] + w * (xxx[5] + w * (xxx[7] + w * (xxx[9] +
  w * xxx[11]))));
 v = z * (xxx[2] + w * (xxx[4] + w * (xxx[6] + w * (xxx[8] + w * (xxx[10] +
  w * xxx[12])))));
 s = z * x;
 r = y + z * (s * (r + v) + y);
 r += xxx[0] * s;
 w = x + r;
 if (ix >= 0x3FE59428) {
  v = (Soft) iy;
  return (Soft) (1 - ((hx >> 30) & 2)) *
   (v - Soft(0x4000000000000000ul,true) * (x - (w * w / (w + v) - r)));
 }
 if (iy == 1)
  return w;
 else {
  Soft a, t;
  z = w;
  SET_LOW_WORD(z,0);
  v = r - (z - x);
  t = a = -Soft(0x3ff0000000000000ul,true) / w;
  SET_LOW_WORD(t,0);
  s = Soft(0x3ff0000000000000ul,true) + t * z;
  return t + a * (s + t * v);
 }
}

}


/* @(#)k_rem_pio2.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_k_rem_pio2 {
constant int init_jk[] = {3,4,4,6};
constant int32_t ipio2[] = {
0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
};
constant Soft PIo2[] = {
  Soft(0x3ff921fb40000000ul,true),
  Soft(0x3e74442d00000000ul,true),
  Soft(0x3cf8469880000000ul,true),
  Soft(0x3b78cc5160000000ul,true),
  Soft(0x39f01b8380000000ul,true),
  Soft(0x387a252040000000ul,true),
  Soft(0x36e3822280000000ul,true),
  Soft(0x3569f31d00000000ul,true),
};
constant Soft
zero = Soft(0x0000000000000000ul,true),
one = Soft(0x3ff0000000000000ul,true),
two24 = Soft(0x4170000000000000ul,true),
twon24 = Soft(0x3e70000000000000ul,true);
inline int
__kernel_rem_pio2(thread Soft *x, thread Soft *y, int e0, int nx, int prec)
{
 int32_t jz,jx,jv,jp,jk,carry,n,iq[20],i,j,k,m,q0,ih;
 Soft z,fw,f[20],fq[20],q[20];
 jk = init_jk[prec];
 jp = jk;
 jx = nx-1;
 jv = (e0-3)/24; if(jv<0) jv=0;
 q0 = e0-24*(jv+1);
 j = jv-jx; m = jx+jk;
 for(i=0;i<=m;i++,j++) f[i] = (j<0)? zero : (Soft) ipio2[j];
 for (i=0;i<=jk;i++) {
     for(j=0,fw=Soft(0x0000000000000000ul,true);j<=jx;j++) fw += x[j]*f[jx+i-j];
     q[i] = fw;
 }
 jz = jk;
while (true) {
 for(i=0,j=jz,z=q[jz];j>0;i++,j--) {
     fw = (Soft)((int32_t)(twon24* z));
     iq[i] = (int32_t)(z-two24*fw);
     z = q[j-1]+fw;
 }
 z = ol_s_scalbn::scalbn(z,q0);
 z -= Soft(0x4020000000000000ul,true)*ol_s_floor::floor(z*Soft(0x3fc0000000000000ul,true));
 n = (int32_t) z;
 z -= (Soft)n;
 ih = 0;
 if(q0>0) {
     i = (iq[jz-1]>>(24-q0)); n += i;
     iq[jz-1] -= i<<(24-q0);
     ih = iq[jz-1]>>(23-q0);
 }
 else if(q0==0) ih = iq[jz-1]>>23;
 else if(z>=Soft(0x3fe0000000000000ul,true)) ih=2;
 if(ih>0) {
     n += 1; carry = 0;
     for(i=0;i<jz ;i++) {
  j = iq[i];
  if(carry==0) {
      if(j!=0) {
   carry = 1; iq[i] = 0x1000000- j;
      }
  } else iq[i] = 0xffffff - j;
     }
     if(q0>0) {
         switch(q0) {
         case 1:
         iq[jz-1] &= 0x7fffff; break;
      case 2:
         iq[jz-1] &= 0x3fffff; break;
         }
     }
     if(ih==2) {
  z = one - z;
  if(carry!=0) z -= ol_s_scalbn::scalbn(one,q0);
     }
 }
 if(z==zero) {
     j = 0;
     for (i=jz-1;i>=jk;i--) j |= iq[i];
     if(j==0) {
  for(k=1;iq[jk-k]==0;k++);
  for(i=jz+1;i<=jz+k;i++) {
      f[jx+i] = (Soft) ipio2[jv+i];
      for(j=0,fw=Soft(0x0000000000000000ul,true);j<=jx;j++) fw += x[j]*f[jx+i-j];
      q[i] = fw;
  }
  jz += k;
  continue;
     }
 }
 break; }
 if(z==Soft(0x0000000000000000ul,true)) {
     jz -= 1; q0 -= 24;
     while(iq[jz]==0) { jz--; q0-=24;}
 } else {
     z = ol_s_scalbn::scalbn(z,-q0);
     if(z>=two24) {
  fw = (Soft)((int32_t)(twon24*z));
  iq[jz] = (int32_t)(z-two24*fw);
  jz += 1; q0 += 24;
  iq[jz] = (int32_t) fw;
     } else iq[jz] = (int32_t) z ;
 }
 fw = ol_s_scalbn::scalbn(one,q0);
 for(i=jz;i>=0;i--) {
     q[i] = fw*(Soft)iq[i]; fw*=twon24;
 }
 for(i=jz;i>=0;i--) {
     for(fw=Soft(0x0000000000000000ul,true),k=0;k<=jp&&k<=jz-i;k++) fw += PIo2[k]*q[i+k];
     fq[jz-i] = fw;
 }
 switch(prec) {
     case 0:
  fw = Soft(0x0000000000000000ul,true);
  for (i=jz;i>=0;i--) fw += fq[i];
  y[0] = (ih==0)? fw: -fw;
  break;
     case 1:
     case 2:
  fw = Soft(0x0000000000000000ul,true);
  for (i=jz;i>=0;i--) fw += fq[i];
  STRICT_ASSIGN(Soft,fw,fw);
  y[0] = (ih==0)? fw: -fw;
  fw = fq[0]-fw;
  for (i=1;i<=jz;i++) fw += fq[i];
  y[1] = (ih==0)? fw: -fw;
  break;
     case 3:
  for (i=jz;i>0;i--) {
      fw = fq[i-1]+fq[i];
      fq[i] += fq[i-1]-fw;
      fq[i-1] = fw;
  }
  for (i=jz;i>1;i--) {
      fw = fq[i-1]+fq[i];
      fq[i] += fq[i-1]-fw;
      fq[i-1] = fw;
  }
  for (fw=Soft(0x0000000000000000ul,true),i=jz;i>=2;i--) fw += fq[i];
  if(ih==0) {
      y[0] = fq[0]; y[1] = fq[1]; y[2] = fw;
  } else {
      y[0] = -fq[0]; y[1] = -fq[1]; y[2] = -fw;
  }
 }
 return n&7;
}

}


/* @(#)e_rem_pio2.c 1.4 95/01/18 */
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
 * Optimized by Bruce D. Evans.
 */


namespace ol_e_rem_pio2 {
constant Soft
zero = Soft(0x0000000000000000ul,true),
two24 = Soft(0x4170000000000000ul,true),
invpio2 = Soft(0x3fe45f306dc9c883ul,true),
pio2_1 = Soft(0x3ff921fb54400000ul,true),
pio2_1t = Soft(0x3dd0b4611a626331ul,true),
pio2_2 = Soft(0x3dd0b4611a600000ul,true),
pio2_2t = Soft(0x3ba3198a2e037073ul,true),
pio2_3 = Soft(0x3ba3198a2e000000ul,true),
pio2_3t = Soft(0x397b839a252049c1ul,true);
__inline int
__ieee754_rem_pio2(Soft x, thread Soft *y)
{
 Soft z,w,t,r,fn;
 Soft tx[3],ty[2];
 int32_t e0,i,j,nx,n,ix,hx;
 uint low;
 GET_HIGH_WORD(hx,x);
 ix = hx&0x7fffffff;
 do { if (ix <= 0x400f6a7a) {
     if ((ix & 0xfffff) == 0x921fb)
  break;
     if (ix <= 0x4002d97c) {
  if (hx > 0) {
      z = x - pio2_1;
      y[0] = z - pio2_1t;
      y[1] = (z-y[0])-pio2_1t;
      return 1;
  } else {
      z = x + pio2_1;
      y[0] = z + pio2_1t;
      y[1] = (z-y[0])+pio2_1t;
      return -1;
  }
     } else {
  if (hx > 0) {
      z = x - 2*pio2_1;
      y[0] = z - 2*pio2_1t;
      y[1] = (z-y[0])-2*pio2_1t;
      return 2;
  } else {
      z = x + 2*pio2_1;
      y[0] = z + 2*pio2_1t;
      y[1] = (z-y[0])+2*pio2_1t;
      return -2;
  }
     }
 }
 if (ix <= 0x401c463b) {
     if (ix <= 0x4015fdbc) {
  if (ix == 0x4012d97c)
      break;
  if (hx > 0) {
      z = x - 3*pio2_1;
      y[0] = z - 3*pio2_1t;
      y[1] = (z-y[0])-3*pio2_1t;
      return 3;
  } else {
      z = x + 3*pio2_1;
      y[0] = z + 3*pio2_1t;
      y[1] = (z-y[0])+3*pio2_1t;
      return -3;
  }
     } else {
  if (ix == 0x401921fb)
      break;
  if (hx > 0) {
      z = x - 4*pio2_1;
      y[0] = z - 4*pio2_1t;
      y[1] = (z-y[0])-4*pio2_1t;
      return 4;
  } else {
      z = x + 4*pio2_1;
      y[0] = z + 4*pio2_1t;
      y[1] = (z-y[0])+4*pio2_1t;
      return -4;
  }
     }
 }
 } while(false); if(ix<0x413921fb) {

     STRICT_ASSIGN(Soft,fn,x*invpio2+Soft(0x4338000000000000ul,true));
     fn = fn-Soft(0x4338000000000000ul,true);
     n = (int32_t)fn;
     r = x-fn*pio2_1;
     w = fn*pio2_1t;
     {
         uint high;
         j = ix>>20;
         y[0] = r-w;
  GET_HIGH_WORD(high,y[0]);
         i = j-((high>>20)&0x7ff);
         if(i>16) {
      t = r;
      w = fn*pio2_2;
      r = t-w;
      w = fn*pio2_2t-((t-r)-w);
      y[0] = r-w;
      GET_HIGH_WORD(high,y[0]);
      i = j-((high>>20)&0x7ff);
      if(i>49) {
       t = r;
       w = fn*pio2_3;
       r = t-w;
       w = fn*pio2_3t-((t-r)-w);
       y[0] = r-w;
      }
  }
     }
     y[1] = (r-y[0])-w;
     return n;
 }
 if(ix>=0x7ff00000) {
     y[0]=y[1]=x-x; return 0;
 }
 GET_LOW_WORD(low,x);
 e0 = (ix>>20)-1046;
 INSERT_WORDS(z, ix - ((int32_t)(e0<<20)), low);
 for(i=0;i<2;i++) {
  tx[i] = (Soft)((int32_t)(z));
  z = (z-tx[i])*two24;
 }
 tx[2] = z;
 nx = 3;
 while(tx[nx-1]==zero) nx--;
 n = ol_k_rem_pio2::__kernel_rem_pio2(tx,ty,e0,nx,1);
 if(hx<0) {y[0] = -ty[0]; y[1] = -ty[1]; return -n;}
 y[0] = ty[0]; y[1] = ty[1]; return n;
}

}

/* @(#)s_sin.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_sin {
inline Soft
sin(Soft x)
{
 Soft y[2],z=Soft(0x0000000000000000ul,true);
 int32_t n, ix;
 GET_HIGH_WORD(ix,x);
 ix &= 0x7fffffff;
 if(ix <= 0x3fe921fb) {
     if(ix<0x3e500000)
        {if((int)x==0) return x;}
     return ol_k_sin::__kernel_sin(x,z,0);
 }
 else if (ix>=0x7ff00000) return x-x;
 else {
     n = ol_e_rem_pio2::__ieee754_rem_pio2(x,y);
     switch(n&3) {
  case 0: return ol_k_sin::__kernel_sin(y[0],y[1],1);
  case 1: return ol_k_cos::__kernel_cos(y[0],y[1]);
  case 2: return -ol_k_sin::__kernel_sin(y[0],y[1],1);
  default:
   return -ol_k_cos::__kernel_cos(y[0],y[1]);
     }
 }
}

}

/* @(#)s_cos.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_cos {
inline Soft
cos(Soft x)
{
 Soft y[2],z=Soft(0x0000000000000000ul,true);
 int32_t n, ix;
 GET_HIGH_WORD(ix,x);
 ix &= 0x7fffffff;
 if(ix <= 0x3fe921fb) {
     if(ix<0x3e46a09e)
  if(((int)x)==0) return Soft(0x3ff0000000000000ul,true);
     return ol_k_cos::__kernel_cos(x,z);
 }
 else if (ix>=0x7ff00000) return x-x;
 else {
     n = ol_e_rem_pio2::__ieee754_rem_pio2(x,y);
     switch(n&3) {
  case 0: return ol_k_cos::__kernel_cos(y[0],y[1]);
  case 1: return -ol_k_sin::__kernel_sin(y[0],y[1],1);
  case 2: return -ol_k_cos::__kernel_cos(y[0],y[1]);
  default:
          return ol_k_sin::__kernel_sin(y[0],y[1],1);
     }
 }
}

}

/* @(#)s_tan.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_tan {
inline Soft
tan(Soft x)
{
 Soft y[2],z=Soft(0x0000000000000000ul,true);
 int32_t n, ix;
 GET_HIGH_WORD(ix,x);
 ix &= 0x7fffffff;
 if(ix <= 0x3fe921fb) {
     if(ix<0x3e400000)
  if((int)x==0) return x;
     return ol_k_tan::__kernel_tan(x,z,1);
 }
 else if (ix>=0x7ff00000) return x-x;
 else {
     n = ol_e_rem_pio2::__ieee754_rem_pio2(x,y);
     return ol_k_tan::__kernel_tan(y[0],y[1],1-((n&1)<<1));
 }
}

}

/* @(#)e_pow.c 1.5 04/04/22 SMI */
/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_pow {
constant Soft
bp[] = {Soft(0x3ff0000000000000ul,true), Soft(0x3ff8000000000000ul,true),},
dp_h[] = { Soft(0x0000000000000000ul,true), Soft(0x3fe2b80340000000ul,true),},
dp_l[] = { Soft(0x0000000000000000ul,true), Soft(0x3e4cfdeb43cfd006ul,true),},
zero = Soft(0x0000000000000000ul,true),
one = Soft(0x3ff0000000000000ul,true),
two = Soft(0x4000000000000000ul,true),
two53 = Soft(0x4340000000000000ul,true),
huge = Soft(0x7e37e43c8800759cul,true),
tiny = Soft(0x01a56e1fc2f8f359ul,true),
L1 = Soft(0x3fe3333333333303ul,true),
L2 = Soft(0x3fdb6db6db6fabfful,true),
L3 = Soft(0x3fd55555518f264dul,true),
L4 = Soft(0x3fd17460a91d4101ul,true),
L5 = Soft(0x3fcd864a93c9db65ul,true),
L6 = Soft(0x3fca7e284a454eeful,true),
P1 = Soft(0x3fc555555555553eul,true),
P2 = -Soft(0x3f66c16c16bebd93ul,true),
P3 = Soft(0x3f11566aaf25de2cul,true),
P4 = -Soft(0x3ebbbd41c5d26bf1ul,true),
P5 = Soft(0x3e66376972bea4d0ul,true),
lg2 = Soft(0x3fe62e42fefa39eful,true),
lg2_h = Soft(0x3fe62e4300000000ul,true),
lg2_l = -Soft(0x3e205c610ca86c39ul,true),
ovt = Soft(0x3c971547652b82feul,true),
cp = Soft(0x3feec709dc3a03fdul,true),
cp_h = Soft(0x3feec709e0000000ul,true),
cp_l = -Soft(0x3e3e2fe0145b01f5ul,true),
ivln2 = Soft(0x3ff71547652b82feul,true),
ivln2_h = Soft(0x3ff7154760000000ul,true),
ivln2_l = Soft(0x3e54ae0bf85ddf44ul,true);
inline Soft
__ieee754_pow(Soft x, Soft y)
{
 Soft z,ax,z_h,z_l,p_h,p_l;
 Soft y1,t1,t2,r,s,t,u,v,w;
 int32_t i,j,k,yisint,n;
 int32_t hx,hy,ix,iy;
 uint lx,ly;
 EXTRACT_WORDS(hx,lx,x);
 EXTRACT_WORDS(hy,ly,y);
 ix = hx&0x7fffffff; iy = hy&0x7fffffff;
 if((iy|ly)==0) return one;
 if (hx==0x3ff00000 && lx == 0) return one;
 if(ix > 0x7ff00000 || ((ix==0x7ff00000)&&(lx!=0)) ||
    iy > 0x7ff00000 || ((iy==0x7ff00000)&&(ly!=0)))
  return (x+Soft(0x0000000000000000ul,true))+(y+Soft(0x0000000000000000ul,true));
 yisint = 0;
 if(hx<0) {
     if(iy>=0x43400000) yisint = 2;
     else if(iy>=0x3ff00000) {
  k = (iy>>20)-0x3ff;
  if(k>20) {
      j = ly>>(52-k);
      if((j<<(52-k))==ly) yisint = 2-(j&1);
  } else if(ly==0) {
      j = iy>>(20-k);
      if((j<<(20-k))==iy) yisint = 2-(j&1);
  }
     }
 }
 if(ly==0) {
     if (iy==0x7ff00000) {
         if(((ix-0x3ff00000)|lx)==0)
      return one;
         else if (ix >= 0x3ff00000)
      return (hy>=0)? y: zero;
         else
      return (hy<0)?-y: zero;
     }
     if(iy==0x3ff00000) {
  if(hy<0) return one/x; else return x;
     }
            if(hy==0x40000000) return x*x;
            if(hy==0x40080000) return x*x*x;
            if(hy==0x40100000) {
                u = x*x;
                return u*u;
            }
     if(hy==0x3fe00000) {
  if(hx>=0)
                    return sqrt(x);
     }
 }
 ax = fabs(x);
 if(lx==0) {
     if(ix==0x7ff00000||ix==0||ix==0x3ff00000){
  z = ax;
  if(hy<0) z = one/z;
  if(hx<0) {
      if(((ix-0x3ff00000)|yisint)==0) {
   z = (z-z)/(z-z);
      } else if(yisint==1)
   z = -z;
  }
  return z;
     }
 }
 n = ((uint)hx>>31)-1;
 if((n|yisint)==0) return (x-x)/(x-x);
 s = one;
 if((n|(yisint-1))==0) s = -one;
 if(iy>0x41e00000) {
     if(iy>0x43f00000){
  if(ix<=0x3fefffff) return (hy<0)? huge*huge:tiny*tiny;
  if(ix>=0x3ff00000) return (hy>0)? huge*huge:tiny*tiny;
     }
     if(ix<0x3fefffff) return (hy<0)? s*huge*huge:s*tiny*tiny;
     if(ix>0x3ff00000) return (hy>0)? s*huge*huge:s*tiny*tiny;
     t = ax-one;
     w = (t*t)*(Soft(0x3fe0000000000000ul,true)-t*(Soft(0x3fd5555555555555ul,true)-t*Soft(0x3fd0000000000000ul,true)));
     u = ivln2_h*t;
     v = t*ivln2_l-w*ivln2;
     t1 = u+v;
     SET_LOW_WORD(t1,0);
     t2 = v-(t1-u);
 } else {
     Soft ss,s2,s_h,s_l,t_h,t_l;
     n = 0;
     if(ix<0x00100000)
  {ax *= two53; n -= 53; GET_HIGH_WORD(ix,ax); }
     n += ((ix)>>20)-0x3ff;
     j = ix&0x000fffff;
     ix = j|0x3ff00000;
     if(j<=0x3988E) k=0;
     else if(j<0xBB67A) k=1;
     else {k=0;n+=1;ix -= 0x00100000;}
     SET_HIGH_WORD(ax,ix);
     u = ax-bp[k];
     v = one/(ax+bp[k]);
     ss = u*v;
     s_h = ss;
     SET_LOW_WORD(s_h,0);
     t_h = zero;
     SET_HIGH_WORD(t_h,((ix>>1)|0x20000000)+0x00080000+(k<<18));
     t_l = ax - (t_h-bp[k]);
     s_l = v*((u-s_h*t_h)-s_h*t_l);
     s2 = ss*ss;
     r = s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6)))));
     r += s_l*(s_h+ss);
     s2 = s_h*s_h;
     t_h = Soft(0x4008000000000000ul,true)+s2+r;
     SET_LOW_WORD(t_h,0);
     t_l = r-((t_h-Soft(0x4008000000000000ul,true))-s2);
     u = s_h*t_h;
     v = s_l*t_h+t_l*ss;
     p_h = u+v;
     SET_LOW_WORD(p_h,0);
     p_l = v-(p_h-u);
     z_h = cp_h*p_h;
     z_l = cp_l*p_h+p_l*cp+dp_l[k];
     t = (Soft)n;
     t1 = (((z_h+z_l)+dp_h[k])+t);
     SET_LOW_WORD(t1,0);
     t2 = z_l-(((t1-t)-dp_h[k])-z_h);
 }
 y1 = y;
 SET_LOW_WORD(y1,0);
 p_l = (y-y1)*t1+y*t2;
 p_h = y1*t1;
 z = p_l+p_h;
 EXTRACT_WORDS(j,i,z);
 if (j>=0x40900000) {
     if(((j-0x40900000)|i)!=0)
  return s*huge*huge;
     else {
  if(p_l+ovt>z-p_h) return s*huge*huge;
     }
 } else if((j&0x7fffffff)>=0x4090cc00 ) {
     if(((j-0xc090cc00)|i)!=0)
  return s*tiny*tiny;
     else {
  if(p_l<=z-p_h) return s*tiny*tiny;
     }
 }
 i = j&0x7fffffff;
 k = (i>>20)-0x3ff;
 n = 0;
 if(i>0x3fe00000) {
     n = j+(0x00100000>>(k+1));
     k = ((n&0x7fffffff)>>20)-0x3ff;
     t = zero;
     SET_HIGH_WORD(t,n&~(0x000fffff>>k));
     n = ((n&0x000fffff)|0x00100000)>>(20-k);
     if(j<0) n = -n;
     p_h -= t;
 }
 t = p_l+p_h;
 SET_LOW_WORD(t,0);
 u = t*lg2_h;
 v = (p_l-(t-p_h))*lg2+t*lg2_l;
 z = u+v;
 w = v-(z-u);
 t = z*z;
 t1 = z - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
 r = (z*t1)/(t1-two)-(w+z*w);
 z = one-(r-z);
 GET_HIGH_WORD(j,z);
 j += (n<<20);
 if((j>>20)<=0) z = ol_s_scalbn::scalbn(z,n);
 else SET_HIGH_WORD(z,j);
 return s*z;
}

}

/* @(#)s_ceil.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_ceil {
constant Soft huge = Soft(0x7e37e43c8800759cul,true);
inline Soft
ceil(Soft x)
{
 int32_t i0,i1,j0;
 uint i,j;
 EXTRACT_WORDS(i0,i1,x);
 j0 = ((i0>>20)&0x7ff)-0x3ff;
 if(j0<20) {
     if(j0<0) {
  if(huge+x>Soft(0x0000000000000000ul,true)) {
      if(i0<0) {i0=0x80000000;i1=0;}
      else if((i0|i1)!=0) { i0=0x3ff00000;i1=0;}
  }
     } else {
  i = (0x000fffff)>>j0;
  if(((i0&i)|i1)==0) return x;
  if(huge+x>Soft(0x0000000000000000ul,true)) {
      if(i0>0) i0 += (0x00100000)>>j0;
      i0 &= (~i); i1=0;
  }
     }
 } else if (j0>51) {
     if(j0==0x400) return x+x;
     else return x;
 } else {
     i = ((uint)(0xffffffff))>>(j0-20);
     if((i1&i)==0) return x;
     if(huge+x>Soft(0x0000000000000000ul,true)) {
  if(i0>0) {
      if(j0==20) i0+=1;
      else {
   j = i1 + (1<<(52-j0));
   if(j<i1) i0+=1;
   i1 = j;
      }
  }
  i1 &= (~i);
     }
 }
 INSERT_WORDS(x,i0,i1);
 return x;
}

}

/* @(#)s_floor.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_trunc {
constant Soft huge = Soft(0x7e37e43c8800759cul,true);
inline Soft
trunc(Soft x)
{
 int32_t i0,i1,j0;
 uint i;
 EXTRACT_WORDS(i0,i1,x);
 j0 = ((i0>>20)&0x7ff)-0x3ff;
 if(j0<20) {
     if(j0<0) {
  if(huge+x>Soft(0x0000000000000000ul,true)) {
      i0 &= 0x80000000U;
      i1 = 0;
  }
     } else {
  i = (0x000fffff)>>j0;
  if(((i0&i)|i1)==0) return x;
  if(huge+x>Soft(0x0000000000000000ul,true)) {
      i0 &= (~i); i1=0;
  }
     }
 } else if (j0>51) {
     if(j0==0x400) return x+x;
     else return x;
 } else {
     i = ((uint)(0xffffffff))>>(j0-20);
     if((i1&i)==0) return x;
     if(huge+x>Soft(0x0000000000000000ul,true))
  i1 &= (~i);
 }
 INSERT_WORDS(x,i0,i1);
 return x;
}

}

/* @(#)s_erf.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_erf {
constant Soft
tiny = Soft(0x01a56e1fc2f8f359ul,true),
one_half= Soft(0x3fe0000000000000ul,true),
one = Soft(0x3ff0000000000000ul,true),
two = Soft(0x4000000000000000ul,true),
erx = Soft(0x3feb0ac160000000ul,true),
efx = Soft(0x3fc06eba8214db69ul,true),
efx8= Soft(0x3ff06eba8214db69ul,true),
pp0 = Soft(0x3fc06eba8214db68ul,true),
pp1 = -Soft(0x3fd4cd7d691cb913ul,true),
pp2 = -Soft(0x3f9d2a51dbd7194ful,true),
pp3 = -Soft(0x3f77a291236668e4ul,true),
pp4 = -Soft(0x3ef8ead6120016acul,true),
qq1 = Soft(0x3fd97779cddadc09ul,true),
qq2 = Soft(0x3fb0a54c5536cebaul,true),
qq3 = Soft(0x3f74d022c4d36b0ful,true),
qq4 = Soft(0x3f215dc9221c1a10ul,true),
qq5 = -Soft(0x3ed09c4342a26120ul,true),
pa0 = -Soft(0x3f6359b8bef77538ul,true),
pa1 = Soft(0x3fda8d00ad92b34dul,true),
pa2 = -Soft(0x3fd7d240fbb8c3f1ul,true),
pa3 = Soft(0x3fd45fca805120e4ul,true),
pa4 = -Soft(0x3fbc63983d3e28ecul,true),
pa5 = Soft(0x3fa22a36599795ebul,true),
pa6 = -Soft(0x3f61bf380a96073ful,true),
qa1 = Soft(0x3fbb3e6618eee323ul,true),
qa2 = Soft(0x3fe14af092eb6f33ul,true),
qa3 = Soft(0x3fb2635cd99fe9a7ul,true),
qa4 = Soft(0x3fc02660e763351ful,true),
qa5 = Soft(0x3f8bedc26b51dd1cul,true),
qa6 = Soft(0x3f888b545735151dul,true),
ra0 = -Soft(0x3f843412600d6435ul,true),
ra1 = -Soft(0x3fe63416e4ba7360ul,true),
ra2 = -Soft(0x40251e0441b0e726ul,true),
ra3 = -Soft(0x404f300ae4cba38dul,true),
ra4 = -Soft(0x40644cb184282266ul,true),
ra5 = -Soft(0x4067135cebccabb2ul,true),
ra6 = -Soft(0x4054526557e4d2f2ul,true),
ra7 = -Soft(0x4023a0efc69ac25cul,true),
sa1 = Soft(0x4033a6b9bd707687ul,true),
sa2 = Soft(0x4061350c526ae721ul,true),
sa3 = Soft(0x407b290dd58a1a71ul,true),
sa4 = Soft(0x40842b1921ec2868ul,true),
sa5 = Soft(0x407ad02157700314ul,true),
sa6 = Soft(0x405b28a3ee48ae2cul,true),
sa7 = Soft(0x401a47ef8e484a93ul,true),
sa8 = -Soft(0x3faeeff2ee749a62ul,true),
rb0 = -Soft(0x3f84341239e86f4aul,true),
rb1 = -Soft(0x3fe993ba70c285deul,true),
rb2 = -Soft(0x4031c209555f995aul,true),
rb3 = -Soft(0x4064145d43c5ed98ul,true),
rb4 = -Soft(0x4083ec881375f228ul,true),
rb5 = -Soft(0x409004616a2e5992ul,true),
rb6 = -Soft(0x407e384e9bdc383ful,true),
sb1 = Soft(0x403e568b261d5190ul,true),
sb2 = Soft(0x40745cae221b9f0aul,true),
sb3 = Soft(0x409802eb189d5118ul,true),
sb4 = Soft(0x40a8ffb7688c246aul,true),
sb5 = Soft(0x40a3f219cedf3be6ul,true),
sb6 = Soft(0x407da874e79fe763ul,true),
sb7 = -Soft(0x403670e242712d62ul,true);
inline Soft
erf(Soft x)
{
 int32_t hx,ix,i;
 Soft R,S,P,Q,s,y,z,r;
 GET_HIGH_WORD(hx,x);
 ix = hx&0x7fffffff;
 if(ix>=0x7ff00000) {
     i = ((uint)hx>>31)<<1;
     return (Soft)(1-i)+one/x;
 }
 if(ix < 0x3feb0000) {
     if(ix < 0x3e300000) {
         if (ix < 0x00800000)
      return Soft(0x3fc0000000000000ul,true)*(Soft(0x4020000000000000ul,true)*x+efx8*x);
  return x + efx*x;
     }
     z = x*x;
     r = pp0+z*(pp1+z*(pp2+z*(pp3+z*pp4)));
     s = one+z*(qq1+z*(qq2+z*(qq3+z*(qq4+z*qq5))));
     y = r/s;
     return x + x*y;
 }
 if(ix < 0x3ff40000) {
     s = fabs(x)-one;
     P = pa0+s*(pa1+s*(pa2+s*(pa3+s*(pa4+s*(pa5+s*pa6)))));
     Q = one+s*(qa1+s*(qa2+s*(qa3+s*(qa4+s*(qa5+s*qa6)))));
     if(hx>=0) return erx + P/Q; else return -erx - P/Q;
 }
 if (ix >= 0x40180000) {
     if(hx>=0) return one-tiny; else return tiny-one;
 }
 x = fabs(x);
  s = one/(x*x);
 if(ix< 0x4006DB6E) {
     R=ra0+s*(ra1+s*(ra2+s*(ra3+s*(ra4+s*(
    ra5+s*(ra6+s*ra7))))));
     S=one+s*(sa1+s*(sa2+s*(sa3+s*(sa4+s*(
    sa5+s*(sa6+s*(sa7+s*sa8)))))));
 } else {
     R=rb0+s*(rb1+s*(rb2+s*(rb3+s*(rb4+s*(
    rb5+s*rb6)))));
     S=one+s*(sb1+s*(sb2+s*(sb3+s*(sb4+s*(
    sb5+s*(sb6+s*sb7))))));
 }
 z = x;
 SET_LOW_WORD(z,0);
 r = ol_e_exp::__ieee754_exp(-z*z-Soft(0x3fe2000000000000ul,true))*ol_e_exp::__ieee754_exp((z-x)*(z+x)+R/S);
 if(hx>=0) return one-r/x; else return r/x-one;
}
inline Soft
erfc(Soft x)
{
 int32_t hx,ix;
 Soft R,S,P,Q,s,y,z,r;
 GET_HIGH_WORD(hx,x);
 ix = hx&0x7fffffff;
 if(ix>=0x7ff00000) {
     return (Soft)(((uint)hx>>31)<<1)+one/x;
 }
 if(ix < 0x3feb0000) {
     if(ix < 0x3c700000)
  return one-x;
     z = x*x;
     r = pp0+z*(pp1+z*(pp2+z*(pp3+z*pp4)));
     s = one+z*(qq1+z*(qq2+z*(qq3+z*(qq4+z*qq5))));
     y = r/s;
     if(hx < 0x3fd00000) {
  return one-(x+x*y);
     } else {
  r = x*y;
  r += (x-one_half);
         return one_half - r ;
     }
 }
 if(ix < 0x3ff40000) {
     s = fabs(x)-one;
     P = pa0+s*(pa1+s*(pa2+s*(pa3+s*(pa4+s*(pa5+s*pa6)))));
     Q = one+s*(qa1+s*(qa2+s*(qa3+s*(qa4+s*(qa5+s*qa6)))));
     if(hx>=0) {
         z = one-erx; return z - P/Q;
     } else {
  z = erx+P/Q; return one+z;
     }
 }
 if (ix < 0x403c0000) {
     x = fabs(x);
      s = one/(x*x);
     if(ix< 0x4006DB6D) {
         R=ra0+s*(ra1+s*(ra2+s*(ra3+s*(ra4+s*(
    ra5+s*(ra6+s*ra7))))));
         S=one+s*(sa1+s*(sa2+s*(sa3+s*(sa4+s*(
    sa5+s*(sa6+s*(sa7+s*sa8)))))));
     } else {
  if(hx<0&&ix>=0x40180000) return two-tiny;
         R=rb0+s*(rb1+s*(rb2+s*(rb3+s*(rb4+s*(
    rb5+s*rb6)))));
         S=one+s*(sb1+s*(sb2+s*(sb3+s*(sb4+s*(
    sb5+s*(sb6+s*sb7))))));
     }
     z = x;
     SET_LOW_WORD(z,0);
     r = ol_e_exp::__ieee754_exp(-z*z-Soft(0x3fe2000000000000ul,true))*
   ol_e_exp::__ieee754_exp((z-x)*(z+x)+R/S);
     if(hx>0) return r/x; else return two-r/x;
 } else {
     if(hx>0) return tiny*tiny; else return two-tiny;
 }
}

}

/* @(#)s_asinh.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


namespace ol_s_asinh {
constant Soft
one = Soft(0x3ff0000000000000ul,true),
ln2 = Soft(0x3fe62e42fefa39eful,true),
huge= Soft(0x7e37e43c8800759cul,true);
inline Soft
asinh(Soft x)
{
 Soft t,w;
 int32_t hx,ix;
 GET_HIGH_WORD(hx,x);
 ix = hx&0x7fffffff;
 if(ix>=0x7ff00000) return x+x;
 if(ix< 0x3e300000) {
     if(huge+x>one) return x;
 }
 if(ix>0x41b00000) {
     w = ol_e_log::__ieee754_log(fabs(x))+ln2;
 } else if (ix>0x40000000) {
     t = fabs(x);
     w = ol_e_log::__ieee754_log(Soft(0x4000000000000000ul,true)*t+one/(sqrt(x*x+one)+t));
 } else {
     t = x*x;
     w =ol_s_log1p::log1p(fabs(x)+t/(one+sqrt(one+t)));
 }
 if(hx>0) return w; else return -w;
}

}


/* @(#)e_acosh.c 1.3 95/01/18 */
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


namespace ol_e_acosh {
constant Soft
one = Soft(0x3ff0000000000000ul,true),
ln2 = Soft(0x3fe62e42fefa39eful,true);
inline Soft
__ieee754_acosh(Soft x)
{
 Soft t;
 int32_t hx;
 uint lx;
 EXTRACT_WORDS(hx,lx,x);
 if(hx<0x3ff00000) {
     return (x-x)/(x-x);
 } else if(hx >=0x41b00000) {
     if(hx >=0x7ff00000) {
         return x+x;
     } else
  return ol_e_log::__ieee754_log(x)+ln2;
 } else if(((hx-0x3ff00000)|lx)==0) {
     return Soft(0x0000000000000000ul,true);
 } else if (hx > 0x40000000) {
     t=x*x;
     return ol_e_log::__ieee754_log(Soft(0x4000000000000000ul,true)*x-one/(x+sqrt(t-one)));
 } else {
     t = x-one;
     return ol_s_log1p::log1p(t+sqrt(Soft(0x4000000000000000ul,true)*t+t*t));
 }
}

}


/* @(#)e_atanh.c 1.3 95/01/18 */
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


namespace ol_e_atanh {
constant Soft one = Soft(0x3ff0000000000000ul,true), huge = Soft(0x7e37e43c8800759cul,true);
constant Soft zero = Soft(0x0000000000000000ul,true);
inline Soft
__ieee754_atanh(Soft x)
{
 Soft t;
 int32_t hx,ix;
 uint lx;
 EXTRACT_WORDS(hx,lx,x);
 ix = hx&0x7fffffff;
 if ((ix|((lx|(-lx))>>31))>0x3ff00000)
     return (x-x)/(x-x);
 if(ix==0x3ff00000)
     return x/zero;
 if(ix<0x3e300000&&(huge+x)>zero) return x;
 SET_HIGH_WORD(x,ix);
 if(ix<0x3fe00000) {
     t = x+x;
     t = Soft(0x3fe0000000000000ul,true)*ol_s_log1p::log1p(t+t*x/(one-x));
 } else
     t = Soft(0x3fe0000000000000ul,true)*ol_s_log1p::log1p((x+x)/(one-x));
 if(hx>=0) return t; else return -t;
}

}


/* @(#)e_hypot.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_hypot {
inline Soft
__ieee754_hypot(Soft x, Soft y)
{
 Soft a,b,t1,t2,y1,y2,w;
 int32_t j,k,ha,hb;
 GET_HIGH_WORD(ha,x);
 ha &= 0x7fffffff;
 GET_HIGH_WORD(hb,y);
 hb &= 0x7fffffff;
 if(hb > ha) {a=y;b=x;j=ha; ha=hb;hb=j;} else {a=x;b=y;}
 a = fabs(a);
 b = fabs(b);
 if((ha-hb)>0x3c00000) {return a+b;}
 k=0;
 if(ha > 0x5f300000) {
    if(ha >= 0x7ff00000) {
        uint low;
        w = fabs(x+Soft(0x0000000000000000ul,true))-fabs(y+Soft(0x0000000000000000ul,true));
        GET_LOW_WORD(low,a);
        if(((ha&0xfffff)|low)==0) w = a;
        GET_LOW_WORD(low,b);
        if(((hb^0x7ff00000)|low)==0) w = b;
        return w;
    }
    ha -= 0x25800000; hb -= 0x25800000; k += 600;
    SET_HIGH_WORD(a,ha);
    SET_HIGH_WORD(b,hb);
 }
 if(hb < 0x20b00000) {
     if(hb <= 0x000fffff) {
         uint low;
  GET_LOW_WORD(low,b);
  if((hb|low)==0) return a;
  t1=0;
  SET_HIGH_WORD(t1,0x7fd00000);
  b *= t1;
  a *= t1;
  k -= 1022;
     } else {
         ha += 0x25800000;
  hb += 0x25800000;
  k -= 600;
  SET_HIGH_WORD(a,ha);
  SET_HIGH_WORD(b,hb);
     }
 }
 w = a-b;
 if (w>b) {
     t1 = 0;
     SET_HIGH_WORD(t1,ha);
     t2 = a-t1;
     w = sqrt(t1*t1-(b*(-b)-t2*(a+t1)));
 } else {
     a = a+a;
     y1 = 0;
     SET_HIGH_WORD(y1,hb);
     y2 = b - y1;
     t1 = 0;
     SET_HIGH_WORD(t1,ha+0x00100000);
     t2 = a - t1;
     w = sqrt(t1*y1-(w*(-w)-(t1*y2+t2*b)));
 }
 if(k!=0) {
     uint high;
     t1 = Soft(0x3ff0000000000000ul,true);
     GET_HIGH_WORD(high,t1);
     SET_HIGH_WORD(t1,high+(k<<20));
     return t1*w;
 } else return w;
}

}

/*-
 * Copyright (c) 2011 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */


namespace ol_k_exp {
constant uint k = 1799;
constant Soft kln2 = Soft(0x40937be319ba0da4ul,true);
static Soft
__frexp_exp(Soft x, thread int *expt)
{
 Soft exp_x;
 uint hx;
 exp_x = ol_e_exp::__ieee754_exp(x - kln2);
 GET_HIGH_WORD(hx, exp_x);
 *expt = (hx >> 20) - (0x3ff + 1023) + k;
 SET_HIGH_WORD(exp_x, (hx & 0xfffff) | ((0x3ff + 1023) << 20));
 return (exp_x);
}
inline Soft
__ldexp_exp(Soft x, int expt)
{
 Soft exp_x, scale;
 int ex_expt;
 exp_x = __frexp_exp(x, &ex_expt);
 expt += ex_expt;
 INSERT_WORDS(scale, (0x3ff + expt) << 20, 0);
 return (exp_x * scale);
}

}


/* @(#)e_sinh.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_sinh {
constant Soft one = Soft(0x3ff0000000000000ul,true), shuge = Soft(0x7fac7b1f3cac7433ul,true);
inline Soft
__ieee754_sinh(Soft x)
{
 Soft t,h;
 int32_t ix,jx;
 GET_HIGH_WORD(jx,x);
 ix = jx&0x7fffffff;
 if(ix>=0x7ff00000) return x+x;
 h = Soft(0x3fe0000000000000ul,true);
 if (jx<0) h = -h;
 if (ix < 0x40360000) {
     if (ix<0x3e300000)
  if(shuge+x>one) return x;
     t = ol_s_expm1::expm1(fabs(x));
     if(ix<0x3ff00000) return h*(Soft(0x4000000000000000ul,true)*t-t*t/(t+one));
     return h*(t+t/(t+one));
 }
 if (ix < 0x40862E42) return h*ol_e_exp::__ieee754_exp(fabs(x));
 if (ix<=0x408633CE)
     return h*Soft(0x4000000000000000ul,true)*ol_k_exp::__ldexp_exp(fabs(x), -1);
 return x*shuge;
}

}


/* @(#)e_cosh.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_cosh {
constant Soft one = Soft(0x3ff0000000000000ul,true), one_half=Soft(0x3fe0000000000000ul,true), huge = Soft(0x7e37e43c8800759cul,true);
inline Soft
__ieee754_cosh(Soft x)
{
 Soft t,w;
 int32_t ix;
 GET_HIGH_WORD(ix,x);
 ix &= 0x7fffffff;
 if(ix>=0x7ff00000) return x*x;
 if(ix<0x3fd62e43) {
     t = ol_s_expm1::expm1(fabs(x));
     w = one+t;
     if (ix<0x3c800000) return w;
     return one+(t*t)/(w+w);
 }
 if (ix < 0x40360000) {
  t = ol_e_exp::__ieee754_exp(fabs(x));
  return one_half*t+one_half/t;
 }
 if (ix < 0x40862E42) return one_half*ol_e_exp::__ieee754_exp(fabs(x));
 if (ix<=0x408633CE)
     return ol_k_exp::__ldexp_exp(fabs(x), -1);
 return huge*huge;
}

}


/* @(#)e_log10.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_log2 {
constant Soft
Lg1 = Soft(0x3fe5555555555593ul,true),
Lg2 = Soft(0x3fd999999997fa04ul,true),
Lg3 = Soft(0x3fd2492494229359ul,true),
Lg4 = Soft(0x3fcc71c51d8e78aful,true),
Lg5 = Soft(0x3fc7466496cb03deul,true),
Lg6 = Soft(0x3fc39a09d078c69ful,true),
Lg7 = Soft(0x3fc2f112df3e5244ul,true);
static inline Soft
k_log1p(Soft f)
{
 Soft hfsq,s,z,R,w,t1,t2;
  s = f/(Soft(0x4000000000000000ul,true)+f);
 z = s*s;
 w = z*z;
 t1= w*(Lg2+w*(Lg4+w*Lg6));
 t2= z*(Lg1+w*(Lg3+w*(Lg5+w*Lg7)));
 R = t2+t1;
 hfsq=Soft(0x3fe0000000000000ul,true)*f*f;
 return s*(hfsq+R);
}
constant Soft
two54 = Soft(0x4350000000000000ul,true),
ivln2hi = Soft(0x3ff7154765200000ul,true),
ivln2lo = Soft(0x3de705fc2eefa200ul,true);
constant Soft zero = Soft(0x0000000000000000ul,true);
inline Soft
__ieee754_log2(Soft x)
{
 Soft f,hfsq,hi,lo,r,val_hi,val_lo,w,y;
 int32_t i,k,hx;
 uint lx;
 EXTRACT_WORDS(hx,lx,x);
 k=0;
 if (hx < 0x00100000) {
     if (((hx&0x7fffffff)|lx)==0)
  return -two54/zero;
     if (hx<0) return (x-x)/zero;
     k -= 54; x *= two54;
     GET_HIGH_WORD(hx,x);
 }
 if (hx >= 0x7ff00000) return x+x;
 if (hx == 0x3ff00000 && lx == 0)
     return zero;
 k += (hx>>20)-1023;
 hx &= 0x000fffff;
 i = (hx+0x95f64)&0x100000;
 SET_HIGH_WORD(x,hx|(i^0x3ff00000));
 k += (i>>20);
 y = (Soft)k;
 f = x - Soft(0x3ff0000000000000ul,true);
 hfsq = Soft(0x3fe0000000000000ul,true)*f*f;
 r = k_log1p(f);
 hi = f - hfsq;
 SET_LOW_WORD(hi,0);
 lo = (f - hi) - hfsq + r;
 val_hi = hi*ivln2hi;
 val_lo = (lo+hi)*ivln2lo + lo*ivln2hi;
 w = y + val_hi;
 val_lo += (y - w) + val_hi;
 val_hi = w;
 return val_lo + val_hi;
}

}


/* @(#)e_log10.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_log10 {
constant Soft
Lg1 = Soft(0x3fe5555555555593ul,true),
Lg2 = Soft(0x3fd999999997fa04ul,true),
Lg3 = Soft(0x3fd2492494229359ul,true),
Lg4 = Soft(0x3fcc71c51d8e78aful,true),
Lg5 = Soft(0x3fc7466496cb03deul,true),
Lg6 = Soft(0x3fc39a09d078c69ful,true),
Lg7 = Soft(0x3fc2f112df3e5244ul,true);
static inline Soft
k_log1p(Soft f)
{
 Soft hfsq,s,z,R,w,t1,t2;
  s = f/(Soft(0x4000000000000000ul,true)+f);
 z = s*s;
 w = z*z;
 t1= w*(Lg2+w*(Lg4+w*Lg6));
 t2= z*(Lg1+w*(Lg3+w*(Lg5+w*Lg7)));
 R = t2+t1;
 hfsq=Soft(0x3fe0000000000000ul,true)*f*f;
 return s*(hfsq+R);
}
constant Soft
two54 = Soft(0x4350000000000000ul,true),
ivln10hi = Soft(0x3fdbcb7b15200000ul,true),
ivln10lo = Soft(0x3dbb9438ca9aadd5ul,true),
log10_2hi = Soft(0x3fd34413509f6000ul,true),
log10_2lo = Soft(0x3d59fef311f12b36ul,true);
constant Soft zero = Soft(0x0000000000000000ul,true);
inline Soft
__ieee754_log10(Soft x)
{
 Soft f,hfsq,hi,lo,r,val_hi,val_lo,w,y,y2;
 int32_t i,k,hx;
 uint lx;
 EXTRACT_WORDS(hx,lx,x);
 k=0;
 if (hx < 0x00100000) {
     if (((hx&0x7fffffff)|lx)==0)
  return -two54/zero;
     if (hx<0) return (x-x)/zero;
     k -= 54; x *= two54;
     GET_HIGH_WORD(hx,x);
 }
 if (hx >= 0x7ff00000) return x+x;
 if (hx == 0x3ff00000 && lx == 0)
     return zero;
 k += (hx>>20)-1023;
 hx &= 0x000fffff;
 i = (hx+0x95f64)&0x100000;
 SET_HIGH_WORD(x,hx|(i^0x3ff00000));
 k += (i>>20);
 y = (Soft)k;
 f = x - Soft(0x3ff0000000000000ul,true);
 hfsq = Soft(0x3fe0000000000000ul,true)*f*f;
 r = k_log1p(f);
 hi = f - hfsq;
 SET_LOW_WORD(hi,0);
 lo = (f - hi) - hfsq + r;
 val_hi = hi*ivln10hi;
 y2 = y*log10_2hi;
 val_lo = y*log10_2lo + (lo+hi)*ivln10lo + lo*ivln10hi;
 w = y2 + val_hi;
 val_lo += (y2 - w) + val_hi;
 val_hi = w;
 return val_lo + val_hi;
}

}

/*-
 * Copyright (c) 2005 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */


namespace ol_s_exp2 {
constant Soft
    huge = Soft(0x7e70000000000000ul,true),
    redux = Soft(0x42b8000000000000ul,true),
    P1 = Soft(0x3fe62e42fefa39eful,true),
    P2 = Soft(0x3fcebfbdff82c575ul,true),
    P3 = Soft(0x3fac6b08d704a0a6ul,true),
    P4 = Soft(0x3f83b2ab88f70400ul,true),
    P5 = Soft(0x3f55d88003875c74ul,true);
constant Soft twom1000 = Soft(0x0170000000000000ul,true);
constant Soft tbl[(1 << 8) * 2] = {
 Soft(0x3fe6a09e667f3d5dul,true), Soft(0x3d39880000000000ul,true),
 Soft(0x3fe6b052fa751744ul,true), Soft(0x3cd8000000000000ul,true),
 Soft(0x3fe6c012750bd9feul,true), -Soft(0x3d28780000000000ul,true),
 Soft(0x3fe6cfdcddd476bful,true), Soft(0x3d1ec00000000000ul,true),
 Soft(0x3fe6dfb23c651a29ul,true), -Soft(0x3cd8000000000000ul,true),
 Soft(0x3fe6ef9298593ae3ul,true), -Soft(0x3cbc000000000000ul,true),
 Soft(0x3fe6ff7df9519386ul,true), -Soft(0x3d2fd80000000000ul,true),
 Soft(0x3fe70f7466f42da3ul,true), -Soft(0x3d2c880000000000ul,true),
 Soft(0x3fe71f75e8ec5fc3ul,true), Soft(0x3d13c00000000000ul,true),
 Soft(0x3fe72f8286eacf05ul,true), -Soft(0x3d38300000000000ul,true),
 Soft(0x3fe73f9a48a58152ul,true), -Soft(0x3d00c00000000000ul,true),
 Soft(0x3fe74fbd35d7ccfcul,true), Soft(0x3d2f880000000000ul,true),
 Soft(0x3fe75feb564267f1ul,true), Soft(0x3d03e00000000000ul,true),
 Soft(0x3fe77024b1ab6d48ul,true), -Soft(0x3d27d00000000000ul,true),
 Soft(0x3fe780694fde5d38ul,true), -Soft(0x3cdd000000000000ul,true),
 Soft(0x3fe790b938ac1d00ul,true), Soft(0x3ce3000000000000ul,true),
 Soft(0x3fe7a11473eb0178ul,true), -Soft(0x3ced000000000000ul,true),
 Soft(0x3fe7b17b0976d060ul,true), Soft(0x3d20400000000000ul,true),
 Soft(0x3fe7c1ed0130c133ul,true), Soft(0x3ca0000000000000ul,true),
 Soft(0x3fe7d26a62ff8636ul,true), -Soft(0x3d26900000000000ul,true),
 Soft(0x3fe7e2f336cf4e3bul,true), -Soft(0x3d02e00000000000ul,true),
 Soft(0x3fe7f3878491c3e8ul,true), -Soft(0x3d24580000000000ul,true),
 Soft(0x3fe80427543e1b4eul,true), Soft(0x3d33000000000000ul,true),
 Soft(0x3fe814d2add1071aul,true), Soft(0x3d0f000000000000ul,true),
 Soft(0x3fe82589994ccd7eul,true), -Soft(0x3d21c00000000000ul,true),
 Soft(0x3fe8364c1eb942d0ul,true), Soft(0x3d29d00000000000ul,true),
 Soft(0x3fe8471a4623cab5ul,true), Soft(0x3d47100000000000ul,true),
 Soft(0x3fe857f4179f5bbcul,true), Soft(0x3d22600000000000ul,true),
 Soft(0x3fe868d99b4491aful,true), -Soft(0x3d32c40000000000ul,true),
 Soft(0x3fe879cad931a395ul,true), -Soft(0x3d23000000000000ul,true),
 Soft(0x3fe88ac7d98a65b8ul,true), -Soft(0x3d2a800000000000ul,true),
 Soft(0x3fe89bd0a4785800ul,true), -Soft(0x3ced000000000000ul,true),
 Soft(0x3fe8ace5422aa223ul,true), Soft(0x3d33280000000000ul,true),
 Soft(0x3fe8be05bad619faul,true), Soft(0x3d42b40000000000ul,true),
 Soft(0x3fe8cf3216b54383ul,true), -Soft(0x3d2ed00000000000ul,true),
 Soft(0x3fe8e06a5e08664cul,true), -Soft(0x3d20500000000000ul,true),
 Soft(0x3fe8f1ae99157807ul,true), Soft(0x3d28280000000000ul,true),
 Soft(0x3fe902fed0282c0eul,true), -Soft(0x3d1cb00000000000ul,true),
 Soft(0x3fe9145b0b91ff96ul,true), -Soft(0x3d05e00000000000ul,true),
 Soft(0x3fe925c353aa2ff9ul,true), Soft(0x3cf5400000000000ul,true),
 Soft(0x3fe93737b0cdc64aul,true), Soft(0x3d17200000000000ul,true),
 Soft(0x3fe948b82b5f98aeul,true), -Soft(0x3d09000000000000ul,true),
 Soft(0x3fe95a44cbc852cbul,true), Soft(0x3d25680000000000ul,true),
 Soft(0x3fe96bdd9a766f21ul,true), -Soft(0x3d36d00000000000ul,true),
 Soft(0x3fe97d829fde4e2aul,true), -Soft(0x3d01000000000000ul,true),
 Soft(0x3fe98f33e47a23a3ul,true), Soft(0x3d2d000000000000ul,true),
 Soft(0x3fe9a0f170ca0604ul,true), -Soft(0x3d38a40000000000ul,true),
 Soft(0x3fe9b2bb4d53ff89ul,true), Soft(0x3d355c0000000000ul,true),
 Soft(0x3fe9c49182a3f15bul,true), Soft(0x3d26b80000000000ul,true),
 Soft(0x3fe9d674194bb8c5ul,true), -Soft(0x3cec000000000000ul,true),
 Soft(0x3fe9e86319e3238eul,true), Soft(0x3d17d00000000000ul,true),
 Soft(0x3fe9fa5e8d07f302ul,true), Soft(0x3d16400000000000ul,true),
 Soft(0x3fea0c667b5de54dul,true), -Soft(0x3cf5000000000000ul,true),
 Soft(0x3fea1e7aed8eb8f6ul,true), Soft(0x3d09e00000000000ul,true),
 Soft(0x3fea309bec4a2e27ul,true), Soft(0x3d2ad80000000000ul,true),
 Soft(0x3fea42c980460a5dul,true), -Soft(0x3d1af00000000000ul,true),
 Soft(0x3fea5503b23e259bul,true), Soft(0x3d0b600000000000ul,true),
 Soft(0x3fea674a8af46213ul,true), Soft(0x3d38880000000000ul,true),
 Soft(0x3fea799e1330b3a7ul,true), Soft(0x3d11200000000000ul,true),
 Soft(0x3fea8bfe53c12e8dul,true), Soft(0x3d06c00000000000ul,true),
 Soft(0x3fea9e6b5579fcd2ul,true), -Soft(0x3d29b80000000000ul,true),
 Soft(0x3feab0e521356fb8ul,true), Soft(0x3d2b700000000000ul,true),
 Soft(0x3feac36bbfd3f381ul,true), Soft(0x3cd9000000000000ul,true),
 Soft(0x3fead5ff3a3c2780ul,true), Soft(0x3ce4000000000000ul,true),
 Soft(0x3feae89f995ad2a3ul,true), -Soft(0x3d2c900000000000ul,true),
 Soft(0x3feafb4ce622f367ul,true), Soft(0x3d16500000000000ul,true),
 Soft(0x3feb0e07298db790ul,true), Soft(0x3d2fd40000000000ul,true),
 Soft(0x3feb20ce6c9a89a9ul,true), Soft(0x3d12700000000000ul,true),
 Soft(0x3feb33a2b84f1a4bul,true), Soft(0x3d4d470000000000ul,true),
 Soft(0x3feb468415b747e7ul,true), -Soft(0x3d38380000000000ul,true),
 Soft(0x3feb59728de5593aul,true), Soft(0x3c98000000000000ul,true),
 Soft(0x3feb6c6e29f1c56aul,true), Soft(0x3d0ad00000000000ul,true),
 Soft(0x3feb7f76f2fb5e50ul,true), Soft(0x3cde800000000000ul,true),
 Soft(0x3feb928cf22749b2ul,true), -Soft(0x3d04c00000000000ul,true),
 Soft(0x3feba5b030a10603ul,true), -Soft(0x3d0d700000000000ul,true),
 Soft(0x3febb8e0b79a6f66ul,true), Soft(0x3d0d900000000000ul,true),
 Soft(0x3febcc1e904bc1fful,true), Soft(0x3d02a00000000000ul,true),
 Soft(0x3febdf69c3f3a16ful,true), -Soft(0x3d1f780000000000ul,true),
 Soft(0x3febf2c25bd71db8ul,true), -Soft(0x3d10a00000000000ul,true),
 Soft(0x3fec06286141b2e9ul,true), -Soft(0x3d11400000000000ul,true),
 Soft(0x3fec199bdd8552e0ul,true), Soft(0x3d0be00000000000ul,true),
 Soft(0x3fec2d1cd9fa64eeul,true), -Soft(0x3d09400000000000ul,true),
 Soft(0x3fec40ab5fffd02ful,true), -Soft(0x3d0ed00000000000ul,true),
 Soft(0x3fec544778fafd15ul,true), Soft(0x3d39660000000000ul,true),
 Soft(0x3fec67f12e57d0cbul,true), -Soft(0x3d1a100000000000ul,true),
 Soft(0x3fec7ba88988c1b6ul,true), -Soft(0x3d58458000000000ul,true),
 Soft(0x3fec8f6d9406e733ul,true), -Soft(0x3d1a480000000000ul,true),
 Soft(0x3feca3405751c4dful,true), Soft(0x3ccb000000000000ul,true),
 Soft(0x3fecb720dcef9094ul,true), Soft(0x3d01400000000000ul,true),
 Soft(0x3feccb0f2e6d1689ul,true), Soft(0x3cf0200000000000ul,true),
 Soft(0x3fecdf0b555dc412ul,true), Soft(0x3cf3600000000000ul,true),
 Soft(0x3fecf3155b5bab3bul,true), -Soft(0x3d06900000000000ul,true),
 Soft(0x3fed072d4a0789bcul,true), Soft(0x3d09a00000000000ul,true),
 Soft(0x3fed1b532b08c8faul,true), -Soft(0x3d15e00000000000ul,true),
 Soft(0x3fed2f87080d8a85ul,true), Soft(0x3d1d280000000000ul,true),
 Soft(0x3fed43c8eacaa203ul,true), Soft(0x3d01a00000000000ul,true),
 Soft(0x3fed5818dcfba491ul,true), Soft(0x3cdf000000000000ul,true),
 Soft(0x3fed6c76e862e6a1ul,true), -Soft(0x3d03a00000000000ul,true),
 Soft(0x3fed80e316c9834eul,true), -Soft(0x3d0cd80000000000ul,true),
 Soft(0x3fed955d71ff6090ul,true), Soft(0x3cf4c00000000000ul,true),
 Soft(0x3feda9e603db32aeul,true), Soft(0x3cff900000000000ul,true),
 Soft(0x3fedbe7cd63a8325ul,true), Soft(0x3ce9800000000000ul,true),
 Soft(0x3fedd321f301b445ul,true), -Soft(0x3cf5200000000000ul,true),
 Soft(0x3fede7d5641c05bful,true), -Soft(0x3d1d700000000000ul,true),
 Soft(0x3fedfc97337b9aecul,true), -Soft(0x3d16140000000000ul,true),
 Soft(0x3fee11676b197d5eul,true), Soft(0x3d0b480000000000ul,true),
 Soft(0x3fee264614f5a3e7ul,true), Soft(0x3d40ce0000000000ul,true),
 Soft(0x3fee3b333b16ee5cul,true), Soft(0x3d0c680000000000ul,true),
 Soft(0x3fee502ee78b3fb4ul,true), -Soft(0x3d09300000000000ul,true),
 Soft(0x3fee653924676d68ul,true), -Soft(0x3ce5000000000000ul,true),
 Soft(0x3fee7a51fbc74c44ul,true), -Soft(0x3d07f80000000000ul,true),
 Soft(0x3fee8f7977cdb726ul,true), -Soft(0x3cf3700000000000ul,true),
 Soft(0x3feea4afa2a490e8ul,true), Soft(0x3ce5d00000000000ul,true),
 Soft(0x3feeb9f4867ccae4ul,true), Soft(0x3d161a0000000000ul,true),
 Soft(0x3feecf482d8e680dul,true), Soft(0x3cf5500000000000ul,true),
 Soft(0x3feee4aaa2188514ul,true), Soft(0x3cc6400000000000ul,true),
 Soft(0x3feefa1bee615a13ul,true), -Soft(0x3cee800000000000ul,true),
 Soft(0x3fef0f9c1cb64106ul,true), -Soft(0x3cfa880000000000ul,true),
 Soft(0x3fef252b376bb963ul,true), -Soft(0x3d2c900000000000ul,true),
 Soft(0x3fef3ac948dd7275ul,true), Soft(0x3caa000000000000ul,true),
 Soft(0x3fef50765b6e4524ul,true), -Soft(0x3cf4f00000000000ul,true),
 Soft(0x3fef6632798844fdul,true), Soft(0x3cca800000000000ul,true),
 Soft(0x3fef7bfdad9cbe38ul,true), Soft(0x3cfabc0000000000ul,true),
 Soft(0x3fef91d802243c82ul,true), -Soft(0x3cd4600000000000ul,true),
 Soft(0x3fefa7c1819e908eul,true), -Soft(0x3d0b0c0000000000ul,true),
 Soft(0x3fefbdba3692d511ul,true), -Soft(0x3cc0e00000000000ul,true),
 Soft(0x3fefd3c22b8f7194ul,true), -Soft(0x3d10de8000000000ul,true),
 Soft(0x3fefe9d96b2a23eeul,true), Soft(0x3cee430000000000ul,true),
 Soft(0x3ff0000000000000ul,true), Soft(0x0000000000000000ul,true),
 Soft(0x3ff00b1afa5abcbeul,true), -Soft(0x3cb3400000000000ul,true),
 Soft(0x3ff0163da9fb3303ul,true), -Soft(0x3d12170000000000ul,true),
 Soft(0x3ff02168143b0282ul,true), Soft(0x3cba400000000000ul,true),
 Soft(0x3ff02c9a3e77806cul,true), Soft(0x3cef980000000000ul,true),
 Soft(0x3ff037d42e11bbcaul,true), -Soft(0x3cc7400000000000ul,true),
 Soft(0x3ff04315e86e7f89ul,true), Soft(0x3cd8300000000000ul,true),
 Soft(0x3ff04e5f72f65467ul,true), -Soft(0x3d1a3f0000000000ul,true),
 Soft(0x3ff059b0d315855aul,true), -Soft(0x3d02840000000000ul,true),
 Soft(0x3ff0650a0e3c1f95ul,true), Soft(0x3cf1600000000000ul,true),
 Soft(0x3ff0706b29ddf71aul,true), Soft(0x3d15240000000000ul,true),
 Soft(0x3ff07bd42b72a82dul,true), -Soft(0x3ce9a00000000000ul,true),
 Soft(0x3ff0874518759bd0ul,true), Soft(0x3ce6400000000000ul,true),
 Soft(0x3ff092bdf66607c8ul,true), -Soft(0x3d00780000000000ul,true),
 Soft(0x3ff09e3ecac6f383ul,true), -Soft(0x3c98000000000000ul,true),
 Soft(0x3ff0a9c79b1f3930ul,true), Soft(0x3cffa00000000000ul,true),
 Soft(0x3ff0b5586cf988fcul,true), -Soft(0x3cfac80000000000ul,true),
 Soft(0x3ff0c0f145e46c8aul,true), Soft(0x3cd9c00000000000ul,true),
 Soft(0x3ff0cc922b724816ul,true), Soft(0x3d05200000000000ul,true),
 Soft(0x3ff0d83b23395dd8ul,true), -Soft(0x3cfad00000000000ul,true),
 Soft(0x3ff0e3ec32d3d1f3ul,true), Soft(0x3d1bac0000000000ul,true),
 Soft(0x3ff0efa55fdfa9a6ul,true), -Soft(0x3d04e80000000000ul,true),
 Soft(0x3ff0fb66affed2f0ul,true), -Soft(0x3d0d300000000000ul,true),
 Soft(0x3ff1073028d7234bul,true), Soft(0x3cf1500000000000ul,true),
 Soft(0x3ff11301d0125b5bul,true), Soft(0x3cec000000000000ul,true),
 Soft(0x3ff11edbab5e2af9ul,true), Soft(0x3d16bc0000000000ul,true),
 Soft(0x3ff12abdc06c31d5ul,true), Soft(0x3ce8400000000000ul,true),
 Soft(0x3ff136a814f2047dul,true), -Soft(0x3d0ed00000000000ul,true),
 Soft(0x3ff1429aaea92de9ul,true), Soft(0x3ce8e00000000000ul,true),
 Soft(0x3ff14e95934f3138ul,true), Soft(0x3ceb400000000000ul,true),
 Soft(0x3ff15a98c8a58e71ul,true), Soft(0x3d05300000000000ul,true),
 Soft(0x3ff166a45471c3dful,true), Soft(0x3d03380000000000ul,true),
 Soft(0x3ff172b83c7d5211ul,true), Soft(0x3d28d40000000000ul,true),
 Soft(0x3ff17ed48695bb9ful,true), -Soft(0x3d05d00000000000ul,true),
 Soft(0x3ff18af9388c8d93ul,true), -Soft(0x3d1c880000000000ul,true),
 Soft(0x3ff1972658375d66ul,true), Soft(0x3d11f00000000000ul,true),
 Soft(0x3ff1a35beb6fcba7ul,true), Soft(0x3d10480000000000ul,true),
 Soft(0x3ff1af99f81387e3ul,true), -Soft(0x3d47390000000000ul,true),
 Soft(0x3ff1bbe084045d54ul,true), Soft(0x3d24e40000000000ul,true),
 Soft(0x3ff1c82f95281c43ul,true), -Soft(0x3d0a200000000000ul,true),
 Soft(0x3ff1d4873168b9b2ul,true), Soft(0x3ce3800000000000ul,true),
 Soft(0x3ff1e0e75eb44031ul,true), Soft(0x3ceac00000000000ul,true),
 Soft(0x3ff1ed5022fcd938ul,true), Soft(0x3d01900000000000ul,true),
 Soft(0x3ff1f9c18438cdf7ul,true), -Soft(0x3d1b780000000000ul,true),
 Soft(0x3ff2063b88628d8ful,true), Soft(0x3d2d940000000000ul,true),
 Soft(0x3ff212be3578a81eul,true), Soft(0x3cd8000000000000ul,true),
 Soft(0x3ff21f49917ddd41ul,true), Soft(0x3d2b340000000000ul,true),
 Soft(0x3ff22bdda2791323ul,true), Soft(0x3d19f80000000000ul,true),
 Soft(0x3ff2387a6e7561e7ul,true), -Soft(0x3d19c80000000000ul,true),
 Soft(0x3ff2451ffb821427ul,true), Soft(0x3d02300000000000ul,true),
 Soft(0x3ff251ce4fb2a602ul,true), -Soft(0x3d13480000000000ul,true),
 Soft(0x3ff25e85711eceb0ul,true), Soft(0x3d12700000000000ul,true),
 Soft(0x3ff26b4565e27d16ul,true), Soft(0x3d11d00000000000ul,true),
 Soft(0x3ff2780e341de00ful,true), Soft(0x3d31ee0000000000ul,true),
 Soft(0x3ff284dfe1f5633eul,true), -Soft(0x3d14c00000000000ul,true),
 Soft(0x3ff291ba7591bb30ul,true), -Soft(0x3d13d80000000000ul,true),
 Soft(0x3ff29e9df51fdf09ul,true), Soft(0x3d08b00000000000ul,true),
 Soft(0x3ff2ab8a66d10e9bul,true), -Soft(0x3d227c0000000000ul,true),
 Soft(0x3ff2b87fd0dada3aul,true), Soft(0x3d2a340000000000ul,true),
 Soft(0x3ff2c57e39771af9ul,true), -Soft(0x3d10800000000000ul,true),
 Soft(0x3ff2d285a6e402d9ul,true), -Soft(0x3d0ed00000000000ul,true),
 Soft(0x3ff2df961f641579ul,true), -Soft(0x3cf4200000000000ul,true),
 Soft(0x3ff2ecafa93e2ecful,true), -Soft(0x3d24980000000000ul,true),
 Soft(0x3ff2f9d24abd8822ul,true), -Soft(0x3d16300000000000ul,true),
 Soft(0x3ff306fe0a31b625ul,true), -Soft(0x3d32360000000000ul,true),
 Soft(0x3ff31432edeea50bul,true), -Soft(0x3d70df8000000000ul,true),
 Soft(0x3ff32170fc4cd7b8ul,true), -Soft(0x3d22480000000000ul,true),
 Soft(0x3ff32eb83ba8e9a2ul,true), -Soft(0x3d25980000000000ul,true),
 Soft(0x3ff33c08b2641766ul,true), Soft(0x3d1ed00000000000ul,true),
 Soft(0x3ff3496266e3fa27ul,true), -Soft(0x3cdc000000000000ul,true),
 Soft(0x3ff356c55f929f0ful,true), -Soft(0x3d30d80000000000ul,true),
 Soft(0x3ff36431a2de88b9ul,true), Soft(0x3d22c80000000000ul,true),
 Soft(0x3ff371a7373aaa39ul,true), Soft(0x3d20600000000000ul,true),
 Soft(0x3ff37f26231e74feul,true), -Soft(0x3d16600000000000ul,true),
 Soft(0x3ff38cae6d05d838ul,true), -Soft(0x3d0ae00000000000ul,true),
 Soft(0x3ff39a401b713ec3ul,true), -Soft(0x3d44720000000000ul,true),
 Soft(0x3ff3a7db34e5a020ul,true), Soft(0x3d08200000000000ul,true),
 Soft(0x3ff3b57fbfec6e95ul,true), Soft(0x3d3e800000000000ul,true),
 Soft(0x3ff3c32dc313a8f2ul,true), Soft(0x3cef800000000000ul,true),
 Soft(0x3ff3d0e544ede122ul,true), -Soft(0x3d17a00000000000ul,true),
 Soft(0x3ff3dea64c1234bbul,true), Soft(0x3d26300000000000ul,true),
 Soft(0x3ff3ec70df1c4eccul,true), -Soft(0x3d48a60000000000ul,true),
 Soft(0x3ff3fa4504ac7e8cul,true), -Soft(0x3d3cdc0000000000ul,true),
 Soft(0x3ff40822c367a0bbul,true), Soft(0x3d25b80000000000ul,true),
 Soft(0x3ff4160a21f72e95ul,true), Soft(0x3d1ec00000000000ul,true),
 Soft(0x3ff423fb27094646ul,true), -Soft(0x3d13600000000000ul,true),
 Soft(0x3ff431f5d950a920ul,true), Soft(0x3d23980000000000ul,true),
 Soft(0x3ff43ffa3f84b9ebul,true), Soft(0x3cfa000000000000ul,true),
 Soft(0x3ff44e0860618919ul,true), -Soft(0x3cf6c00000000000ul,true),
 Soft(0x3ff45c2042a7d201ul,true), -Soft(0x3d0bc00000000000ul,true),
 Soft(0x3ff46a41ed1d0016ul,true), -Soft(0x3d12800000000000ul,true),
 Soft(0x3ff4786d668b3326ul,true), Soft(0x3d30e00000000000ul,true),
 Soft(0x3ff486a2b5c13c00ul,true), -Soft(0x3d2d400000000000ul,true),
 Soft(0x3ff494e1e192af04ul,true), Soft(0x3d0c200000000000ul,true),
 Soft(0x3ff4a32af0d7d372ul,true), -Soft(0x3d1e500000000000ul,true),
 Soft(0x3ff4b17dea6db801ul,true), Soft(0x3d07800000000000ul,true),
 Soft(0x3ff4bfdad53629e1ul,true), -Soft(0x3d13800000000000ul,true),
 Soft(0x3ff4ce41b817c132ul,true), Soft(0x3d00800000000000ul,true),
 Soft(0x3ff4dcb299fddddbul,true), Soft(0x3d2c700000000000ul,true),
 Soft(0x3ff4eb2d81d8ab96ul,true), -Soft(0x3d1ce00000000000ul,true),
 Soft(0x3ff4f9b2769d2d02ul,true), Soft(0x3d19200000000000ul,true),
 Soft(0x3ff508417f4531c1ul,true), -Soft(0x3d08c00000000000ul,true),
 Soft(0x3ff516daa2cf662aul,true), -Soft(0x3cfa000000000000ul,true),
 Soft(0x3ff5257de83f51eaul,true), Soft(0x3d4a080000000000ul,true),
 Soft(0x3ff5342b569d4edaul,true), -Soft(0x3d26d80000000000ul,true),
 Soft(0x3ff542e2f4f6ac1aul,true), -Soft(0x3d32440000000000ul,true),
 Soft(0x3ff551a4ca5d94dbul,true), Soft(0x3d483c0000000000ul,true),
 Soft(0x3ff56070dde9116bul,true), Soft(0x3d24b00000000000ul,true),
 Soft(0x3ff56f4736b529deul,true), Soft(0x3d415a0000000000ul,true),
 Soft(0x3ff57e27dbe2c40eul,true), -Soft(0x3d29e00000000000ul,true),
 Soft(0x3ff58d12d497c76ful,true), -Soft(0x3d23080000000000ul,true),
 Soft(0x3ff59c0827ff0b4cul,true), Soft(0x3d4dec0000000000ul,true),
 Soft(0x3ff5ab07dd485427ul,true), -Soft(0x3cc4000000000000ul,true),
 Soft(0x3ff5ba11fba87af4ul,true), Soft(0x3d30080000000000ul,true),
 Soft(0x3ff5c9268a59460bul,true), -Soft(0x3d26c80000000000ul,true),
 Soft(0x3ff5d84590998e3ful,true), Soft(0x3d469a0000000000ul,true),
 Soft(0x3ff5e76f15ad20e1ul,true), -Soft(0x3d1b400000000000ul,true),
 Soft(0x3ff5f6a320dcebcaul,true), Soft(0x3d17700000000000ul,true),
 Soft(0x3ff605e1b976dcb8ul,true), Soft(0x3d26f80000000000ul,true),
 Soft(0x3ff6152ae6cdf715ul,true), Soft(0x3d01000000000000ul,true),
 Soft(0x3ff6247eb03a5531ul,true), -Soft(0x3d15d00000000000ul,true),
 Soft(0x3ff633dd1d1929b5ul,true), -Soft(0x3d12d00000000000ul,true),
 Soft(0x3ff6434634ccc313ul,true), -Soft(0x3cea800000000000ul,true),
 Soft(0x3ff652b9febc8efaul,true), -Soft(0x3d28600000000000ul,true),
 Soft(0x3ff6623882553397ul,true), Soft(0x3d71fe0000000000ul,true),
 Soft(0x3ff671c1c708328eul,true), -Soft(0x3d37200000000000ul,true),
 Soft(0x3ff68155d44ca97eul,true), Soft(0x3ce6800000000000ul,true),
 Soft(0x3ff690f4b19e9471ul,true), -Soft(0x3d29780000000000ul,true),
};
inline Soft
exp2(Soft x)
{
 Soft r, t, twopk, twopkp1000, z;
 uint hx, ix, lx, i0;
 int k;
 GET_HIGH_WORD(hx,x);
 ix = hx & 0x7fffffff;
 if(ix >= 0x40900000) {
  if(ix >= 0x7ff00000) {
   GET_LOW_WORD(lx,x);
   if(((ix & 0xfffff) | lx) != 0 || (hx & 0x80000000) == 0)
    return (x + x);
   else
    return (Soft(0x0000000000000000ul,true));
  }
  if(x >= Soft(0x4090000000000000ul,true))
   return (huge * huge);
  if(x <= -Soft(0x4090cc0000000000ul,true))
   return (twom1000 * twom1000);
 } else if (ix < 0x3c900000) {
  return (Soft(0x3ff0000000000000ul,true) + x);
 }
 STRICT_ASSIGN(Soft, t, x + redux);
 GET_LOW_WORD(i0, t);
 i0 += (1 << 8) / 2;
 k = (i0 >> 8) << 20;
 i0 = (i0 & ((1 << 8) - 1)) << 1;
 t -= redux;
 z = x - t;
 t = tbl[i0];
 z -= tbl[i0 + 1];
 if (k >= -(1021 << 20))
  INSERT_WORDS(twopk, 0x3ff00000 + k, 0);
 else
  INSERT_WORDS(twopkp1000, 0x3ff00000 + k + (1000 << 20), 0);
 r = t + t * z * (P1 + z * (P2 + z * (P3 + z * (P4 + z * P5))));
 if(k >= -(1021 << 20)) {
  if (k == 1024 << 20)
   return (r * Soft(0x4000000000000000ul,true) * Soft(0x7fe0000000000000ul,true));
  return (r * twopk);
 } else {
  return (r * twopkp1000 * twom1000);
 }
}

}


/* @(#)e_fmod.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */


namespace ol_e_fmod {
constant Soft one = Soft(0x3ff0000000000000ul,true), Zero[] = {Soft(0x0000000000000000ul,true), -Soft(0x0000000000000000ul,true),};
inline Soft
__ieee754_fmod(Soft x, Soft y)
{
 int32_t n,hx,hy,hz,ix,iy,sx,i;
 uint lx,ly,lz;
 EXTRACT_WORDS(hx,lx,x);
 EXTRACT_WORDS(hy,ly,y);
 sx = hx&0x80000000;
 hx ^=sx;
 hy &= 0x7fffffff;
 if((hy|ly)==0||(hx>=0x7ff00000)||
   ((hy|((ly|-ly)>>31))>0x7ff00000))
     return (x*y)/(x*y);
 if(hx<=hy) {
     if((hx<hy)||(lx<ly)) return x;
     if(lx==ly)
  return Zero[(uint)sx>>31];
 }
 if(hx<0x00100000) {
     if(hx==0) {
  for (ix = -1043, i=lx; i>0; i<<=1) ix -=1;
     } else {
  for (ix = -1022,i=(hx<<11); i>0; i<<=1) ix -=1;
     }
 } else ix = (hx>>20)-1023;
 if(hy<0x00100000) {
     if(hy==0) {
  for (iy = -1043, i=ly; i>0; i<<=1) iy -=1;
     } else {
  for (iy = -1022,i=(hy<<11); i>0; i<<=1) iy -=1;
     }
 } else iy = (hy>>20)-1023;
 if(ix >= -1022)
     hx = 0x00100000|(0x000fffff&hx);
 else {
     n = -1022-ix;
     if(n<=31) {
         hx = (hx<<n)|(lx>>(32-n));
         lx <<= n;
     } else {
  hx = lx<<(n-32);
  lx = 0;
     }
 }
 if(iy >= -1022)
     hy = 0x00100000|(0x000fffff&hy);
 else {
     n = -1022-iy;
     if(n<=31) {
         hy = (hy<<n)|(ly>>(32-n));
         ly <<= n;
     } else {
  hy = ly<<(n-32);
  ly = 0;
     }
 }
 n = ix - iy;
 while(n--) {
     hz=hx-hy;lz=lx-ly; if(lx<ly) hz -= 1;
     if(hz<0){hx = hx+hx+(lx>>31); lx = lx+lx;}
     else {
      if((hz|lz)==0)
      return Zero[(uint)sx>>31];
      hx = hz+hz+(lz>>31); lx = lz+lz;
     }
 }
 hz=hx-hy;lz=lx-ly; if(lx<ly) hz -= 1;
 if(hz>=0) {hx=hz;lx=lz;}
 if((hx|lx)==0)
     return Zero[(uint)sx>>31];
 while(hx<0x00100000) {
     hx = hx+hx+(lx>>31); lx = lx+lx;
     iy -= 1;
 }
 if(iy>= -1022) {
     hx = ((hx-0x00100000)|((iy+1023)<<20));
     INSERT_WORDS(x,hx|sx,lx);
 } else {
     n = -1022 - iy;
     if(n<=20) {
  lx = (lx>>n)|((uint)hx<<(32-n));
  hx >>= n;
     } else if (n<=31) {
  lx = (hx<<(32-n))|(lx>>n); hx = sx;
     } else {
  lx = hx>>(n-32); hx = sx;
     }
     INSERT_WORDS(x,hx|sx,lx);
     x *= one;
 }
 return x;
}

}
