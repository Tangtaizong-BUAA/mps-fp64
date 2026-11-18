// Formula branches, coefficient sets and operation order adapted from ATen's
// native/Math.h (PyTorch 2.11.0, 70d99e998b4955e0049d13a98d77ae1b14db1f45).
// The Bessel series include the Cephes approximations. Their coefficient
// literals below are rounded to binary64 at generation time.
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
/*
 * Copyright (c) 2018, Steven Moshier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * * Neither the name of the nor the
 * names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Steven Moshier BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

namespace special64 {
constant ulong j0_pp[] = {
    0x3f4a1d30983b6b27ul, 0x3fb534b0b35dd1cful, 0x3ff3d5214e680b98ul,
    0x4015c9fbe97a0956ul, 0x40217e8c69409888ul, 0x40153684a59425a1ul,
    0x3ff0000000000000ul,
};
constant ulong j0_pq[] = {
    0x3f4e4a80ce039737ul, 0x3fb5ebc5ab5454e3ul, 0x3ff40e72c9b3069ful,
    0x4015e247e68162bbul, 0x40218618ea1b21a1ul, 0x40153965ed423a19ul,
    0x3ff0000000000000ul,
};
constant ulong j0_qp[] = {
    0xbf87474238a5384aul, 0xbff4853b3a321174ul, 0xc0338dcff50e2c0cul,
    0xc0574d2f5a6de8c4ul, 0xc06635cc20cae8eaul, 0xc062627aec17392dul,
    0xc049b48c55b218cdul, 0xc0183358d1b9a1ddul,
};
constant ulong j0_qq[] = {
    0x40501457413c25acul, 0x408ac370b1759c7ful, 0x40ae54cdbd748cb5ul,
    0x40bc4877bdefd63eul, 0x40b72aba1d733b11ul, 0x40a01c2fc7319e82ul,
    0x406e402f06280a54ul,
};
constant ulong j0_rp[] = {
    0xc1f1dc53ad1c8325ul, 0x427c7751c772990dul, 0xc2ec5614e0d900f7ul,
    0x43413ef869ff5fb4ul,
};
constant ulong j0_rq[] = {
    0x407f3902a696b78cul, 0x410536cb36a21a67ul, 0x418719342eac0634ul,
    0x4204d5b009444914ul, 0x427ebeb372182e46ul, 0x42f1a6a28c9748e9ul,
    0x435c41417e7b2e9cul, 0x43b7be34c7b662ccul,
};
constant ulong j1_pp[] = {
    0x3f48f92c4c6c651bul, 0x3fb2b948a3fec4b6ul, 0x3ff208fec21596d6ul,
    0x401472c4f8b13a6aul, 0x4020d91c8b5d2f16ul, 0x4014dbaa142f81a2ul,
    0x3ff0000000000000ul,
};
constant ulong j1_pq[] = {
    0x3f42b89b13443d69ul, 0x3fb19fdd5948aa83ul, 0x3ff1aea9b850eed6ul,
    0x40144ba2f7d251a1ul, 0x4020ccb9dda2fd65ul, 0x4014d6dd4762b4d9ul,
    0x3ff0000000000000ul,
};
constant ulong j1_qp[] = {
    0x3faa27fa6b70ba40ul, 0x4013edb5c66d8fd6ul, 0x4052f4b99acf1c67ul,
    0x4076ec7947aa180dul, 0x408636d9b66f6e50ul, 0x4082abeab9e802d0ul,
    0x406a760a4c54bb0bul, 0x403934ff4d159eb5ul,
};
constant ulong j1_qq[] = {
    0x40528f3060895077ul, 0x409081cba20e5f6ful, 0x40b37a691bfdfe81ul,
    0x40c2ad28d280d118ul, 0x40bf3d0aa6973d14ul, 0x40a61462b4bd1781ul,
    0x4075017f6ae75997ul,
};
constant ulong j1_rp[] = {
    0xc1cad23c4cda4fc5ul, 0x425a52ba0d438c6bul, 0xc2d08a92e6ccf175ul,
    0x432a2b42a697c482ul,
};
constant ulong j1_rq[] = {
    0x408366b11b7086e7ul, 0x410f5eda0dd701b2ul, 0x4193e954dc92a1b1ul,
    0x4214a13f7befeac1ul, 0x429146fb8076ffa8ul, 0x43064b0a3eccf45ful,
    0x4373e0bff4653f81ul, 0x43d2779576702939ul,
};
constant ulong y0_yp[] = {
    0x40ce7437e896898ful, 0xc16bf81f32e48896ul, 0x41f43f78f0284cddul,
    0xc26c957be1d6bd2bul, 0x42d3ea723cc3ac2dul, 0xc328a121d1d8cc02ul,
    0x4363a94b660b4003ul, 0xc3506d4b5906367bul,
};
constant ulong y0_yq[] = {
    0x40904522576dfcb6ul, 0x41231b76a907bc0cul, 0x41b007635164d101ul,
    0x42341ddb2b8664bcul, 0x42b275fcc57e828eul, 0x43268910dfeb596dul,
    0x438bd25fbcf9b5d0ul,
};
constant ulong y1_yp[] = {
    0x41d2d2be62f9b6c5ul, 0xc262d72d58836521ul, 0x42da0954b0910feful,
    0xc33ce01a37a1b083ul, 0x438679ad0b7366b1ul, 0xc3a59e415dde2b17ul,
};
constant ulong y1_yq[] = {
    0x40829269a93f7ac2ul, 0x410cc160be58ef7ful, 0x419184efa9c8aceeul,
    0x421178c3906b7b83ul, 0x428c3f5efda99316ul, 0x4301a326d71d1e4eul,
    0x436e83e3c547a488ul, 0x43cb90f190f6747ful,
};
constant ulong i0_a[] = {
    0xbc545cb72134d0eful, 0x3c833362977da589ul, 0xbcb184eb721ebbb4ul,
    0x3cdee6d893f65ebaul, 0xbd0a5022c297fbebul, 0x3d359b464b262627ul,
    0xbd61164c62ee1af0ul, 0x3d89fe2fe19bd324ul, 0xbdb2fc957a946abcul,
    0x3dda98becc743c10ul, 0xbe01d4fe13ae9556ul, 0x3e26d903a454cb34ul,
    0xbe4beaf68c0b30abul, 0x3e703b769d4d6435ul, 0xbe91ec638f227f8dul,
    0x3eb2bf24978cf4acul, 0xbed2866fcba56427ul, 0x3ef13f58be9a2859ul,
    0xbf0e2b2659c41d5aul, 0x3f28b51b74107cabul, 0xbf42e2fd1f15eb52ul,
    0x3f5adc758a12100eul, 0xbf71b65e201aa849ul, 0x3f859961f3dde3ddul,
    0xbf984e9ef121b6f0ul, 0x3fa93e8acea8a32dul, 0xbfb84b70342d06eaul,
    0x3fc5f7ac77ac88c0ul, 0xbfd37febc057cd8dul, 0x3fe5a84e9035a22aul,
};
constant ulong i0_b[] = {
    0xbc60adb754ca8b19ul, 0xbc5646da66119130ul, 0x3c89be1812d98421ul,
    0x3c83f3dd076041cdul, 0xbcb4600babd21fe4ul, 0xbcb8aee7d908de38ul,
    0x3cdfee7da3eafb1ful, 0x3cf12a919094e6d7ul, 0xbd0583fe7e65629aul,
    0xbd275d99cf68bb32ul, 0x3d1156ff0d5fc545ul, 0x3d5b1c8c6b83c073ul,
    0x3d694347fa268cecul, 0xbd7f904303178d66ul, 0xbdad0fd7357e7bf2ul,
    0xbdc1511d08397425ul, 0x3daa24feabe8004ful, 0x3e00f9ccc0f46f75ul,
    0x3e2d2c64a9225b87ul, 0x3e58569280d6d56dul, 0x3e8b8007d9cd616eul,
    0x3ec8412bc101c586ul, 0x3f120fa378999e52ul, 0x3f6b998ca2e59049ul,
    0x3fe9be62aca809cbul,
};
constant ulong i1_a[] = {
    0x3c499f2a0c3c4014ul, 0xbc7857d0c38a0576ul, 0x3ca663e3e593bfacul,
    0xbcd3eaaa7e0d1573ul, 0x3d011d7f0615290cul, 0xbd2c628e1c8f0b3bul,
    0x3d56af784779d955ul, 0xbd8173835fb70366ul, 0x3da9cee2b21d3154ul,
    0xbdd2510397eb07deul, 0x3df8ea34b43fdf6cul, 0xbe20361b28ea67e6ul,
    0x3e44258e02395010ul, 0xbe67dd3e24b8c3e8ul, 0x3e8ae344b347d108ul,
    0xbeacc0798363992aul, 0x3ecd1c4ed511afc5ul, 0xbeebd5f9b8debbcful,
    0x3f0911b542c70d0bul, 0xbf2533cad3d694feul, 0x3f40c95db6c6df7dul,
    0xbf58cc620b3cd4a4ul, 0x3f71065349d3a1b4ul, 0xbf85a29f7913a26aul,
    0x3f9951e3e7bb2349ul, 0xbfab1bbc537c9ebcul, 0x3fba46dad536f53cul,
    0xbfc694d10469192eul, 0x3fd02a63724a7ffaul,
};
constant ulong i1_b[] = {
    0x3c61556db352e8e6ul, 0x3c545b8aea87b950ul, 0xbc8acea3b2532277ul,
    0xbc82806c9c773320ul, 0x3cb55915fceb588aul, 0x3cb7d68e5f04a2d1ul,
    0xbce0efcd8bc4d22aul, 0xbcf12db5138afbc7ul, 0x3d0776e1762d31e8ul,
    0x3d280d3c26b3281eul, 0xbd17a9482e6d22a0ul, 0xbd5cbc458e73e255ul,
    0xbd6953e1076ab493ul, 0x3d81e7d3f6439fa3ul, 0x3daf101f653c457bul,
    0x3dc1e1a1f1587865ul, 0xbdb4dcf9d4504c0cul, 0xbe0334ca5423dd80ul,
    0xbe30790b9ad53528ul, 0xbe5c415394bb46c1ul, 0xbe90dbfd2e9e5443ul,
    0xbed048df49ca0373ul, 0xbf1cfd7f804aa9a6ul, 0xbf83fda053fcdb4cul,
    0x3fe8ea18b55b1514ul,
};
constant ulong k0_a[] = {
    0x3ca3cee1e6a7fd77ul, 0x3d27fb0ac384c2aaul, 0x3da6c25c47512144ul,
    0x3e205c1367e58a13ul, 0x3e9102bce7f95efcul, 0x3ef7f900fe8cfca0ul,
    0x3f54b82e75633d73ul, 0x3fa26bf6554a9085ul, 0x3fd608d881213db9ul,
    0xbfe12166e9d2f61ful,
};
constant ulong k0_b[] = {
    0x3c5871a5cf8c9ee1ul, 0xbc72fecc93812603ul, 0x3c8e092e41a8cdf4ul,
    0xbca82f9474d77641ul, 0x3cc3dbf96b94785bul, 0xbce0a690ecfadd36ul,
    0x3cfc8d9c4a7ddc98ul, 0xbd19145ebb056fc2ul, 0x3d369c4ecd94cfe2ul,
    0xbd54f87721a470d1ul, 0x3d740fa32fccfc25ul, 0xbd93dafc3f005143ul,
    0x3db46808475fbcc7ul, 0xbdd5df95d2e7c935ul, 0x3df8956c819ff608ul,
    0xbe1d29d5f22bf5dbul, 0x3e4269a0033f428eul, 0xbe6905541b54f2aful,
    0x3e92915981e3e46ful, 0xbebebb547f18d3a9ul, 0x3eed413fcc7470a9ul,
    0xbf20d797e7889f42ul, 0x3f59b891fca79342ul, 0xbfa019f72d4ff71eul,
    0x400385bd9f4e6907ul,
};
constant ulong k1_a[] = {
    0xbc6032283d3cda56ul, 0xbce5dd517a0399e0ul, 0xbd677502ddd0e045ul,
    0xbde3663bb84626caul, 0xbe57c41d145c31d0ul, 0xbec469b32c832e3aul,
    0xbf26ade2e5a3bd02ul, 0xbf7c92939d7d4192ul, 0xbfbf637243c1db74ul,
    0xbfd69a1b757b0dd4ul, 0x3ff867a1361008caul,
};
constant ulong k1_b[] = {
    0xbc5a8c5d42c43a08ul, 0x3c74af1a838f5ed3ul, 0xbc90661517771d52ul,
    0x3caa7d5e44ee2c0bul, 0xbcc5d2a3d8758ef2ul, 0x3ce25cedefee81bdul,
    0xbcff9d52364185abul, 0x3d1be3e959151f65ul, 0xbd394465d4cbcfb3ul,
    0x3d578ffa040ab0b1ul, 0xbd76adec61fc88f8ul, 0x3d969ab846c04f10ul,
    0xbdb76b539401ce7bul, 0x3dd956d008a42f6aul, 0xbdfccbc00365cf27ul,
    0x3e214f242a73d24eul, 0xbe4637a49fe1e088ul, 0x3e6ed27c668fc461ul,
    0xbe9780528fada5c6ul, 0x3ec42fe31752d1b6ul, 0xbef44d711dcdb2e5ul,
    0x3f29965888f6908eul, 0xbf676946be66b48aul, 0x3fba9abef9e023fbul,
    0x4005c3d7aa062c8aul,
};

inline Soft raw(ulong bits) { return Soft(bits, true); }
inline Soft integer(long i) { return raw(__softfloat64_cvt_i64_to_f64(i, 0)); }
inline bool nan(Soft x) { return (x.v << 1) > 0xffe0000000000000ul; }
inline Soft fma64(Soft a, Soft b, Soft c) { return raw(__softfloat64_fma(a.v,b.v,c.v,0)); }
inline Soft sine(Soft x) { return ol_s_sin::sin(x); }
inline Soft cosine(Soft x) { return ol_s_cos::cos(x); }
inline Soft logarithm(Soft x) { return ol_e_log::__ieee754_log(x); }
inline Soft exponential(Soft x) { return ol_e_exp::__ieee754_exp(x); }
constant Soft half64 = Soft(0x3fe0000000000000ul, true);
constant Soft qnan = Soft(0x7ff8000000000000ul, true);
constant Soft pinf = Soft(0x7ff0000000000000ul, true);

// ATen's integer endpoint expressions wrap on the Apple arm64 build before
// converting to double. Use unsigned arithmetic to make that behavior defined.
inline Soft endpoint(ulong magnitude, bool negative) {
    return integer(as_type<long>(negative ? 0ul - magnitude : magnitude));
}

inline Soft chebyshev_trig(uint family, bool shifted, Soft x, long n) {
    const Soft angle = ol_e_acos::__ieee754_acos(x);
    const ulong un = ulong(n);
    if (family == 0) return cosine(integer(n) * angle);
    if (family == 1) {
        const Soft d = sine(angle), m = endpoint(un + 1, false);
        return d.v << 1 ? sine(m * angle) / d : m * cosine(m * angle) / x;
    }
    const Soft h = half64 * angle;
    const Soft phase = (integer(n) + half64) * angle;
    if (family == 2) {
        if (sine(h) != Soft(1)) return cosine(phase) / cosine(h);
        return endpoint(2ul * un + 1, un & 1);
    }
    if (cosine(h) != Soft(1)) return sine(phase) / sine(h);
    if (!shifted && x > Soft(0)) return endpoint(2ul * un + 1, false);
    return endpoint(1, un & 1);
}

inline Soft polynomial(uint kind, Soft x, Soft degree, device atomic_uint *error) {
    const bool hermite = kind == 8 || kind == 9;
    if (kind == 8 && (degree.v << 1) >= 0xffe0000000000000ul) return Soft(0);
    const long n = __softfloat64_cvt_f64_to_i64(degree.v, 3);
    if (n < 0) return Soft(0);
    if (n == 0) return Soft(1);
    if (hermite && n > 512) return qnan;
    // The pinned arm64 wheel returns one when the initial NaN bypasses both
    // of these upstream recurrence loops (ATen leaves r uninitialized there).
    if ((kind == 4 || kind == 10) && n >= 2 && nan(x)) return Soft(1);
    const ulong un = ulong(n);
    const uint family = kind & 3;
    const bool shifted = kind >= 4 && kind < 8;
    Soft p = 1, q = x, factor = x;
    if (kind < 8) {
        const bool left = shifted ? x == Soft(0) : x == Soft(-1);
        const bool right = x == Soft(1);
        if (left || right) {
            if (family == 1) return endpoint(un + 1, left && (un & 1));
            if ((family == 2 && left) || (family == 3 && right))
                return endpoint(2ul * un + 1, left && (un & 1));
            return endpoint(1, left && (un & 1));
        }
        const Soft z = shifted ? x + x - Soft(1) : x;
        const long threshold = shifted ? (family == 3 ? 4 : 6) : (family == 0 ? 6 : 8);
        if (n > threshold && fabs(z) < Soft(1)) {
            // Different binary64 libms can round acos one ULP apart. At very
            // large degree that becomes an order-one phase discrepancy. Zero
            // has the same rounded pi/2 in the pinned CPU and GPU libraries.
            if (n > 4096 && z != Soft(0)) {
                atomic_fetch_max_explicit(error, 2u, memory_order_relaxed);
                return qnan;
            }
            return chebyshev_trig(family, shifted, z, n);
        }
        factor = z + z;
        q = family == 0 ? z : family == 1 ? factor : family == 2 ? factor - Soft(1) : factor + Soft(1);
    } else if (kind == 8) {
        factor = q = x + x;
    } else if (kind == 10) {
        if (x == Soft(0)) return Soft(1);
        q = Soft(1) - x;
    } else if (kind == 11 && fabs(x) == Soft(1)) {
        return endpoint(1, x < Soft(0) && (un & 1));
    }
    // Preserve the arm64 CPU's single-rounded multiply-subtract at each
    // recurrence step. Stop immediately once a NaN becomes absorbing.
    const uint stop = uint(min(n, 4096l));
    for (uint k = 1; k < stop && !nan(q); ++k) {
        Soft next;
        if (kind < 8) next = fma64(factor, q, -p);
        else if (hermite) next = fma64(factor, q, -(integer(kind == 8 ? 2u*k : k) * p));
        else {
            Soft scale = kind == 10 ? integer(2u*k) + (Soft(1) - x) : integer(2u*k + 1u) * x;
            next = fma64(scale, q, -(integer(k) * p)) / integer(k + 1);
        }
        p = q; q = next;
    }
    if (n > stop && !nan(q)) atomic_fetch_max_explicit(error, 1u, memory_order_relaxed);
    return q;
}

template<uint N> inline Soft horner(Soft x, constant ulong (&table)[N]) {
    Soft v = raw(table[0]);
    for (uint i = 1; i < N; ++i) v = fma64(v, x, raw(table[i]));
    return v;
}
template<uint N> inline Soft clenshaw(Soft x, constant ulong (&table)[N]) {
    Soft prev = 0, previous = 0, value = raw(table[0]);
    for (uint i = 1; i < N; ++i) {
        previous = prev; prev = value;
        value = fma64(x, prev, -previous) + raw(table[i]);
    }
    return half64 * (value - previous);
}

// Ordinary Bessel functions share the oscillatory large-argument expansion.
// Integer template parameters specialize the four small-argument rational forms.
template<bool first, bool second_kind> inline Soft cylindrical(Soft x) {
    if (nan(x)) return qnan;
    const bool neg = x < Soft(0);
    if (second_kind && (neg || x == Soft(0))) return neg ? qnan : -pinf;
    Soft ax = second_kind || !neg ? x : -x;
    Soft value;
    if (ax <= Soft(5)) {
        const Soft xx = ax * ax;
        if constexpr (second_kind) {
            const Soft ratio = first ? horner(xx, y1_yp) / horner(xx, y1_yq) : horner(xx, y0_yp) / horner(xx, y0_yq);
            const Soft k = raw(0x3fe45f306dc9c883ul);
            if (first) return fma64(ax, ratio, k * fma64(cylindrical<true,false>(ax), logarithm(ax), -(Soft(1)/ax)));
            return fma64(k * logarithm(ax), cylindrical<false,false>(ax), ratio);
        }
        if (first) {
            const Soft r = horner(xx, j1_rp) / horner(xx, j1_rq);
            value = r * ax * (xx - raw(0x402d5d2b4189822cul)) * (xx - raw(0x40489bf66072a432ul));
        } else if (ax < raw(0x3ee4f8b588e368f1ul)) {
            value = Soft(1) - xx / Soft(4);
        } else {
            value = (xx - raw(0x401721fb80462bbbul)) * (xx - raw(0x403e78a4a621dd6ful))
                    * horner(xx, j0_rp) / horner(xx, j0_rq);
        }
    } else {
        const Soft inv = Soft(5) / ax, z = first ? inv * inv : Soft(25) / (ax * ax);
        const Soft pp = first ? horner(z, j1_pp) / horner(z, j1_pq) : horner(z, j0_pp) / horner(z, j0_pq);
        const Soft qp = first ? horner(z, j1_qp) / horner(z, j1_qq) : horner(z, j0_qp) / horner(z, j0_qq);
        const Soft phase = ax - raw(first ? 0x4002d97c7f3321d2ul : 0x3fe921fb54442d18ul);
        const Soft s = sine(phase), c = cosine(phase);
        value = (second_kind ? fma64(pp, s, (inv * qp) * c) : fma64(pp, c, -((inv * qp) * s)))
                * raw(0x3fe9884533d43651ul) / sqrt(ax);
    }
    return first && neg ? -value : value;
}

template<bool first> inline Soft modified_i(Soft x, bool scaled, bool aten_i1=false) {
    const Soft ax = fabs(x);
    Soft value;
    if (ax <= Soft(8)) {
        const Soft z = ax / Soft(2) - Soft(2);
        value = first ? clenshaw(z, i1_a) : clenshaw(z, i0_a);
        if (first && !scaled && aten_i1) value = (exponential(ax) * ax) * value;
        else {
            if (first) value *= ax;
            if (!scaled) value *= exponential(ax);
        }
    } else {
        const Soft z = Soft(32) / ax - Soft(2);
        value = first ? clenshaw(z, i1_b) : clenshaw(z, i0_b);
        if (!scaled) value *= exponential(ax);
        value /= sqrt(ax);
    }
    return first && x < Soft(0) ? -value : value;
}
template<bool first> inline Soft modified_k(Soft x, bool scaled) {
    if (x == Soft(0)) return pinf;
    if (x < Soft(0)) return qnan;
    if (x <= Soft(2)) {
        const Soft z = fma64(x, x, Soft(-2));
        const Soft k = first ? clenshaw(z, k1_a) : clenshaw(z, k0_a);
        const Soft l = logarithm(half64 * x);
        const Soft v = first ? fma64(l, modified_i<true>(x, false), k / x) : fma64(-l, modified_i<false>(x, false), k);
        return scaled ? v * exponential(x) : v;
    }
    const Soft z = Soft(8) / x - Soft(2);
    Soft value = first ? clenshaw(z, k1_b) : clenshaw(z, k0_b);
    return (scaled ? value : exponential(-x) * value) / sqrt(x);
}
inline Soft spherical(Soft x) {
    if ((x.v << 1) == 0xffe0000000000000ul) return Soft(0);
    if (fabs(x) >= half64 || nan(x)) return sine(x) / x;
    const Soft xx = x * x;
    Soft p = raw(0x3de6124613a86d09ul);
    p = fma64(xx, p, raw(0xbe5ae64567f544e4ul));
    p = fma64(xx, p, raw(0x3ec71de3a556c734ul));
    p = fma64(xx, p, raw(0xbf2a01a01a01a01aul));
    p = fma64(xx, p, raw(0x3f81111111111111ul));
    p = fma64(xx, p, raw(0xbfc5555555555555ul));
    return fma64(xx, p, Soft(1));
}

} // namespace special64

using namespace special64;
kernel void polynomial_kernel(device ulong *out, const device ulong *x, const device ulong *n,
                              constant long *layout, constant uint &kind, device atomic_uint *error,
                              uint i [[thread_position_in_grid]]) {
    const ulong2 pos = pointwise_offsets(i, layout);
    out[i] = polynomial(kind, raw(x[pos.x]), raw(n[pos.y]), error).v;
}
kernel void bessel_kernel(device ulong *out, const device ulong *x, constant long *layout,
                          constant uint &kind, uint i [[thread_position_in_grid]]) {
    const Soft a = raw(x[pointwise_offsets(i,layout).x]);
    Soft v;
    switch (kind) {
        case 0: v = cylindrical<false,false>(a); break;
        case 1: v = cylindrical<true,false>(a); break;
        case 2: v = cylindrical<false,true>(a); break;
        case 3: v = cylindrical<true,true>(a); break;
        case 4: v = modified_i<false>(a,false); break;
        case 5: v = modified_i<true>(a,false); break;
        case 6: v = modified_k<false>(a,false); break;
        case 7: v = modified_k<true>(a,false); break;
        case 8: v = modified_k<false>(a,true); break;
        case 9: v = modified_k<true>(a,true); break;
        case 10: v = spherical(a); break;
        case 11: v = modified_i<false>(a,true); break;
        case 12: v = modified_i<true>(a,false,true); break;
        default: v = modified_i<true>(a,true); break;
    }
    out[i] = v.v;
}
