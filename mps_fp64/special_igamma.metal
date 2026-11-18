// Temme coefficients from PyTorch 2.11 native/Math.h at
// 70d99e998b4955e0049d13a98d77ae1b14db1f45, derived from SciPy.
// This file is concatenated after special.metal, which retains the PyTorch
// and Cephes notices. The SciPy redistribution terms are included below.
/*
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

namespace special64 {
constant ulong ig_temme[14][25] = {
    {
        0xbfd5555555555555ul, 0x3fb5555555555555ul, 0xbf8e573ac901e574ul, 0x3f52f684bda12f68ul,
        0x3f371de3a556c734ul, 0xbf276e06fec7273bul, 0x3f048c5892f7cd83ul, 0xbec255370652afc2ul,
        0xbebf1b22f594c6b5ul, 0x3eabd6d21e4b4109ul, 0xbe87b5f9a2d0465cul, 0x3e3ccf5ceb7f0d9ful,
        0x3e46097d55c37c1cul, 0xbe32d2197c7a2faaul, 0x3e0f6e66d24d5c8aul, 0xbdbc0d9b6edf2b0cul,
        0xbdd0070a87340428ul, 0x3dbac9475c463659ul, 0xbd961ca701fd754aul, 0x3d3ef98008f5eec2ul,
        0x3d57ba0759769d7dul, 0xbd43989bebb193c0ul, 0x3d20104fc4369a3cul, 0xbcc283fe7950ad7bul,
        0xbce1ca914d71a27cul,
    },
    {
        0xbf5e573ac901e574ul, 0xbf6c71c71c71c71cul, 0x3f65ac056b015ac0ul, 0xbf50394f6f09e723ul,
        0x3f2af83440e53dbcul, 0xbe9af83440e53dbcul, 0xbef2fa4ae89e5af0ul, 0x3ee00a9cabd6b83eul,
        0xbebb0bdfcc629cbaul, 0x3e33f59230a8357cul, 0x3e8280f2cde3f847ul, 0xbe6ee23d0cba8aeeul,
        0x3e49aa7a30de114cul, 0xbdb349fbca3a377bul, 0xbe11564ecff73d58ul, 0x3dfc9b434bf3c34eul,
        0xbdd78a5056f8ce45ul, 0x3d3113e3a466db9eul, 0x3d9f8041c5540ea2ul, 0xbd89ccf2fab4608bul,
        0x3d6519580a10cd82ul, 0xbcaf3b7a5dcd1851ul, 0xbd2c068b448455eaul, 0x3d16d8a9ef5c1827ul,
        0xbcf29b03783db2a2ul,
    },
    {
        0x3f70ee643b990ee6ul, 0xbf65f7268edab4c8ul, 0x3f4948b0fcd6e9e0ul, 0x3ec0db20a88f4696ul,
        0xbf1c253efaa1a933ul, 0x3f0bbf43daf4fe53ul, 0xbeeac2d05890f2c3ul, 0x3e626154ae39151dul,
        0x3eb7058929663936ul, 0xbea522cb05171911ul, 0x3e832ac81c15d3d7ul, 0xbdec24bd0e740a6cul,
        0xbe4e437343a46f5dul, 0x3e3ac0d455e25360ul, 0xbe177c5829460138ul, 0x3d70962774f638bbul,
        0x3de1b1056c188672ul, 0xbdce9778dbc61371ul, 0x3daa55da34225759ul, 0xbcf2c681309d6007ul,
        0xbd733f39f65c6eeeul, 0x3d60675f56b95f3bul, 0xbd3be16182b001e8ul, 0x3c75d3b42a398b8ful,
        0x3d03f2fe637bc2b8ul,
    },
    {
        0x3f4547d93b34e2b6ul, 0x3f2e13ce465fa859ul, 0xbf3ebfb188b7ca00ul, 0x3f318b9b5bf2d984ul,
        0xbf13d2a3a29b5d9dul, 0xbe90152a1871f27aul, 0x3ee73df462204ef4ul, 0xbed7cd6f27b3f020ul,
        0x3eb7e0201539310eul, 0xbdbea23269c140a7ul, 0xbe86c2dcffbefeeful, 0x3e75bde8ef4c4dc7ul,
        0xbe54853ced169327ul, 0x3d850c3f0dd501ebul, 0x3e21b66a39794ba9ul, 0xbe1040c53b2491f0ul,
        0x3ded9b15465daec1ul, 0xbd0f46057e1c9d1ful, 0xbdb812d3d94d533bul, 0x3da587d7a7c1a668ul,
        0xbd8328e9df2eb8b6ul, 0x3c91e54cdbaa3443ul, 0x3d4def3f46a086e5ul, 0xbd3a4d8ed36b49dcul,
        0x3d17075e8dcfddd0ul,
    },
    {
        0xbf4c3e0b02da7bf9ul, 0x3f49b0ff6874f2c4ul, 0xbf33999a85a4237aul, 0xbeb88f2ae1def9d0ul,
        0x3f116908b48ce058ul, 0xbf04ce3fd902bcadul, 0x3ee7db4c02846e81ul, 0x3df13b3c5b7cb45eul,
        0xbebc71c074985d3ful, 0x3eade37d9f09164cul, 0xbe8ec676cf33153cul, 0x3dc041515bab6adaul,
        0x3e5efe94304ac16bul, 0xbe4e78e449f4e3beul, 0x3e2d9a9f1a8b7696ul, 0xbd5033ba70791e5eul,
        0xbdfb14f212618752ul, 0x3de9911dbca7ce93ul, 0xbdc7f2fac5e22aaeul, 0x3cd7088090f49aabul,
        0x3d949465337812c4ul, 0xbd82e7ac3cc20208ul, 0x3d614577d11fe2b7ul, 0xbc5d3b49b9fd2152ul,
        0xbd2c6716fd28d001ul,
    },
    {
        0xbf36128ac5a4fa71ul, 0xbf1247604839c039ul, 0x3f322be87360ef1ful, 0xbf2a2042c5148e27ul,
        0x3f11d1e9cb24760bul, 0x3e830bdcf208080eul, 0xbeec823fc1b3cc36ul, 0x3ee0d0e229150428ul,
        0xbec338eb19652fd9ul, 0xbdf659cfde0bb2ebul, 0x3e9741504e5c87c2ul, 0xbe88c267becd0c0ful,
        0x3e69e630225a095bul, 0xbd14411c5ac40e35ul, 0xbe3b15bbf334c8c3ul, 0x3e2b2a3adb58623dul,
        0xbe0af0f32d677057ul, 0x3cf762c060bd9bdaul, 0x3dd9b9c5831849dcul, 0xbdc8d0152b8692baul,
        0x3da7bf5ea6674b5ful, 0xbc851bfdafa33430ul, 0xbd754d6b090f18dbul, 0x3d63fcc249cb50d9ul,
        0xbd42a5b16d7de31eul,
    },
    {
        0x3f4168ef1b0931c8ul, 0xbf436773bdb97b48ul, 0x3f31c0950d3ecb9dul, 0x3eaa8411da6cab49ul,
        0xbf15600945495b37ul, 0x3f0d6bdf83130dc1ul, 0xbef3382f4cf48618ul, 0xbe2a74243fa27729ul,
        0x3ecd115d4f5dcc68ul, 0xbec10587854fcb37ul, 0x3ea36c8903447d35ul, 0x3d5074e709bf4b8bul,
        0xbe77b2f7de505322ul, 0x3e69778c6d79bcc1ul, 0xbe4af0ea334cc20dul, 0x3d3858ba968e7d04ul,
        0x3e1cf0f99fa070bcul, 0xbe0d77155071f99bul, 0x3dedaf3327a51b54ul, 0xbccb6df73b581619ul,
        0xbdbd4a717ac2b965ul, 0x3dacbb55e3e29ba5ul, 0xbd8bf888fe9ca81cul, 0x3c55b9bd2acc211ful,
        0x3d59f7d14e8f487bul,
    },
    {
        0x3f3691879c01efb4ul, 0x3f0b1d75d3346711ul, 0xbf35f3385098cebful, 0x3f326eeb5ece1d9ful,
        0xbf1cc642787368ceul, 0xbe8119c70312e0a2ul, 0x3efd179830b113abul, 0xbef3269164e3e304ul,
        0x3ed8467d794bd7f2ul, 0x3e00f82da50cdaeeul, 0xbeb1c6acec59f442ul, 0x3ea4b12ad51452d5ul,
        0xbe87929779607d63ul, 0xbd76d32eed259534ul, 0x3e5cf11fbdf49e99ul, 0xbe4f4e88c5d1cae1ul,
        0x3e30b2830e4dfce1ul, 0xbc865f59322ddf56ul, 0xbe024e8da0f96246ul, 0x3df2daf0a8add2abul,
        0xbdd33ada96417614ul, 0x3c7ddc4a629af677ul, 0x3da379df6a52f424ul, 0xbd935d870109f334ul,
        0x3d731d6a00ba6216ul,
    },
    {
        0xbf45629b3187b744ul, 0x3f4b8239c670e690ul, 0xbf3cb967b4446107ul, 0xbea762676b30cfd6ul,
        0x3f25d1157082916dul, 0xbf20c16fcea7ddb2ul, 0x3f084637d3f583cdul, 0x3e33937992ec9b02ul,
        0xbee6384af9ac219dul, 0x3edc738f198ab550ul, 0xbec1adec9530a7adul, 0xbdb2ed3c124b7492ul,
        0x3e9952f970ac9b03ul, 0xbe8d599e3b2187a2ul, 0x3e70b282393d4893ul, 0x3cc7c54ec550bd4bul,
        0xbe44985ee872fc56ul, 0x3e3663fd6d84752eul, 0xbe180990f0dfb26aul, 0x3cc36412c0552a81ul,
        0x3deac79309fc7363ul, 0xbddbd671f048b194ul, 0x3dbcac1ee5de78aaul, 0xbc5779b4a6572e09ul,
        0xbd8da96613f7775aul,
    },
    {
        0xbf438dff1cc96982ul, 0xbf12e31f9b7913eaul, 0x3f463969bb825829ul, 0xbf44f9f2582dd0a5ul,
        0x3f322fb20c28e8a0ul, 0x3e886c71c8cebf16ul, 0xbf163a803aebc9b7ul, 0x3f100120036172b0ul,
        0xbef618fcc48d37bcul, 0xbe0e7018e8be3330ul, 0x3ed2fe63d892e1a9ul, 0xbec7d8d3a891d8baul,
        0x3ead3850f27b27e8ul, 0x3d903901807110d2ul, 0xbe849865a9b6fd04ul, 0x3e77ca3da4d350ceul,
        0xbe5b0abf9d310d85ul, 0xbd0706d644652279ul, 0x3e30bcbd16605be3ul, 0xbe2244bad2fffd4ful,
        0x3e03b6549adcccb6ul, 0xbc0bdbb7a0bc6bceul, 0xbdd63f0cfd72ae16ul, 0x3dc74cd688c73fedul,
        0xbda831b3a872b284ul,
    },
    {
        0x3f55d4ae684527bful, 0xbf5f5dbcaf756cdeul, 0x3f522b37f1b46951ul, 0x3eb0a9ef61e90004ul,
        0xbf40aba998a532bful, 0x3f3c01c0b52c3345ul, 0xbf2618e482f9d22aul, 0xbe41759e6f571329ul,
        0x3f07bdf837b4e130ul, 0xbf00650f761692a2ul, 0x3ee5ea3af60786b1ul, 0x3dcaa0a6ef89a12aul,
        0xbec205588c7220b7ul, 0x3eb64d9971a80133ul, 0xbe9b0abf52fc4d58ul, 0xbd48b97eb7553f43ul,
        0x3e72d454a640f7f8ul, 0xbe65b19dcac0a663ul, 0x3e48a3e9b486f0daul, 0x3c524830817ba61ful,
        0xbe1e96b1d57d29c3ul, 0x3e10bf3a2f6afa8aul, 0xbdf22546bbf739c6ul, 0x3c5ab9618d3701a6ul,
        0x3dc4b273207b9023ul,
    },
    {
        0x3f59e1dba8ec5904ul, 0x3f254d241144693ful, 0xbf60e7245b5e0240ul, 0x3f6185be08721041ul,
        0xbf508fd64cc4d9d6ul, 0xbe9ac8f35a61360ful, 0x3f37bf3a7a227118ul, 0xbf3271c35d1a742aul,
        0x3f1b648cb8b91d61ul, 0x3e223870b487d429ul, 0xbefb081c1069b36aul, 0x3ef21f0d8e42b54dul,
        0xbed7a962022d07b2ul, 0xbda83e23f727e2feul, 0x3eb2d456933154b1ul, 0xbea70cb7c2ec0c52ul,
        0x3e8bb865efbb7c49ul, 0x3d2a4c4ee6f7598aul, 0xbe631e2f7c2057ddul, 0x3e55fafc6207f6ceul,
        0xbe38f34113f0801aul, 0xbca3353e1d7f8951ul, 0x3e0f0bacd0370f00ul, 0xbe010ac14ad52ebaul,
        0x3de2876b3785de06ul,
    },
    {
        0xbf70ae56a5daa127ul, 0x3f7a3a699f4a401bul, 0xbf708d50006f5e0eul, 0xbec25187cdea1eebul,
        0x3f61cf4d14eb1812ul, 0xbf60237b58c76530ul, 0x3f4b647f0b161ed3ul, 0x3e54e11fb9ab4d6eul,
        0xbf30e5103ef55b59ul, 0x3f28eab17b1a5667ul, 0xbf11bf09035d225dul, 0xbde3d8d849a65517ul,
        0x3ef079cba3747641ul, 0xbee59bec2daecc91ul, 0x3ecbb865dacf43baul, 0x3d6c166cf2213dbeul,
        0xbea581f5664ec1e3ul, 0x3e9a1a0baff44abeul, 0xbe7f3011553e9943ul, 0xbce95f1e554e1fc0ul,
        0x3e555806ce2925ddul, 0xbe487f75dac1bcf1ul, 0x3e2bcb20d29db62eul, 0x3be8a6832e8ed281ul,
        0xbe0153438ee8db49ul,
    },
    {
        0xbf785c7ccbc5fc12ul, 0xbf41b33b019b3e6ful, 0x3f82010998f1553aul, 0xbf84303ce949bb43ul,
        0x3f748900f8e29435ul, 0x3eb57cc9e9a6596ful, 0xbf60e596fb46b154ul, 0x3f5c0816b1314cf1ul,
        0xbf462eb1c560282dul, 0xbe3da3e6523aaa76ul, 0x3f28b6bb2cc02754ul, 0xbf218eb043924ff5ul,
        0x3f084156dd77628dul, 0x3dc602512b27e94cul, 0xbee581f634675d03ul, 0x3edbbbac7672b130ul,
        0xbec18b098b674d56ul, 0xbd4d7cac376be54ful, 0x3e9aae08a5f2b88dul, 0xbe9013a5585a3e12ul,
        0x3e731ba687133284ul, 0x3cd01431e3f933d6ul, 0xbe49fce55cf292a0ul, 0x3e3dc92e5d9a0e3bul,
        0xbe20e39a0cc2ec99ul,
    },
};
// log(Gamma(1+t))/t at zero: -EulerGamma, then (-1)^k zeta(k)/k.
constant ulong ig_lgamma1_coeff[] = {
    0x3f91111111111111ul, 0xbf915b1e5f75270dul, 0x3f91a7b9611a7b96ul, 0xbf91f7047dc11f70ul,
    0x3f92492492492492ul, 0xbf929e4129e4129eul, 0x3f92f684bda12f69ul, 0xbf93521cfb2b78c2ul,
    0x3f93b13b13b13b15ul, 0xbf94141414141417ul, 0x3f947ae147ae1480ul, 0xbf94e5e0a72f0544ul,
    0x3f9555555555556bul, 0xbf95c9882b931083ul, 0x3f9642c8590b21bdul, 0xbf96c16c16c16ccdul,
    0x3f9745d1745d18baul, 0xbf97d05f417d08eeul, 0x3f98618618618c31ul, 0xbf98f9c18f9c2577ul,
    0x3f9999999999b333ul, 0xbf9a41a41a41d89eul, 0x3f9af286bca21af3ul, 0xbf9bacf914c29837ul,
    0x3f9c71c71c738e39ul, 0xbf9d41d41d457c58ul, 0x3f9e1e1e1e25a5a6ul, 0xbf9f07c1f08ba2eaul,
    0x3fa0000000100002ul, 0xbfa08421086318ceul, 0x3fa111111155556dul, 0xbfa1a7b961a7b9aaul,
    0x3fa24924936db7bcul, 0xbfa2f684c00002bcul, 0x3fa3b13b189d925eul, 0xbfa47ae151eb9fb7ul,
    0x3fa555556aaafdcdul, 0xbfa642c88591b66dul, 0x3fa745d1d1778df9ul, 0xbfa86186db77bfbful,
    0x3fa9999b3352d5baul, 0xbfaaf28a1b5688a0ul, 0x3fac71ce3a20b419ul, 0xbfae1e2d311e8abdul,
    0x3fb00010064cdeb2ul, 0xbfb11133476e7fe0ul, 0x3fb2496df8320c5ful, 0xbfb3b1d971fc5985ul,
    0x3fb556ad63243bc4ul, 0xbfb748c33114c6d6ul, 0x3fb9a01e385d5f8ful, 0xbfbc806706d57db4ul,
    0x3fc010b36af86397ul, 0xbfc2703a1dcea3aeul, 0x3fc5b40cb100c306ul, 0xbfca8b9c17aa6149ul,
    0x3fd151322ac7d848ul, 0xbfd9a4d55beab2d7ul, 0x3fea51a6625307d3ul, 0xbfe2788cfc6fb619ul,
};
// Odd Stirling correction terms B[2k]/(2k(2k-1)) in descending order.
constant ulong ig_stirling[] = {
    0x3fc6fe96381e0680ul, 0xbf9e4286cb0f5398ul, 0x3f7a41a41a41a41aul, 0xbf5f6ab0d9993c7dul,
    0x3f4b951e2b18ff23ul, 0xbf43813813813814ul, 0x3f4a01a01a01a01aul, 0xbf66c16c16c16c17ul,
    0x3fb5555555555555ul,
};
constant Soft ig_inv_sqrt_tau = Soft(0x3fd9884533d43651ul,true);
constant Soft ig_ln2 = Soft(0x3fe62e42fefa39eful,true);
constant Soft ig_ln2_lo = Soft(0x3c7abc9e3b39803ful,true);

// Two-component arithmetic keeps the small tail's exponential argument precise.
// A one-ulp error in the argument at 700 can cost hundreds of ulps in the tail.
struct IgPair { Soft hi, lo; };
inline IgPair ig_sum(Soft x, Soft y) {
    const Soft h = x+y, b = h-x;
    return {h, (x-(h-b))+(y-b)};
}
inline IgPair ig_add(IgPair x, IgPair y) {
    const IgPair s = ig_sum(x.hi,y.hi);
    return ig_sum(s.hi, (s.lo+x.lo)+y.lo);
}
inline IgPair ig_neg(IgPair x) { return {-x.hi,-x.lo}; }
inline IgPair ig_mul(IgPair x, IgPair y) {
    const Soft h = x.hi*y.hi;
    const Soft l = fma64(x.hi,y.hi,-h) + (x.hi*y.lo+x.lo*y.hi);
    return ig_sum(h,l);
}
inline IgPair ig_div(IgPair x, IgPair y) {
    const Soft q = x.hi/y.hi;
    const Soft r = fma64(-q,y.hi,x.hi) + (x.lo-q*y.lo);
    return ig_sum(q,r/y.hi);
}
inline IgPair ig_one(Soft x) { return {x,0}; }

// Sum 1/3+u^2/5+u^4/7+... . Both call sites reduce |u| to at most 1/3.
// At that bound the neglected term after 36 powers is below 2^-120.
inline IgPair ig_atanh_series(IgPair u) {
    const IgPair square = ig_mul(u,u);
    IgPair power = ig_one(1), sum = ig_one(0);
    for (uint k=0;k<37;++k) {
        const IgPair term=ig_div(power,ig_one(integer(2*k+3)));
        sum=ig_add(sum,term);
        if (fabs(term.hi)<raw(0x3920000000000000ul)) break;
        power=ig_mul(power,square);
    }
    return sum;
}

inline IgPair ig_log(Soft x) {
    int adjust=0;
    if (x.v < 0x0010000000000000ul) { x *= raw(0x4350000000000000ul); adjust=-54; }
    int k = int((x.v>>52)&2047ul)-1023+adjust;
    Soft m=raw((x.v&0x000ffffffffffffful)|0x3ff0000000000000ul);
    if (m.v > 0x3ff6a09e667f3bcdul) { m *= half64; ++k; }
    const IgPair u=ig_div(ig_one(m-Soft(1)),ig_sum(m,Soft(1)));
    const IgPair s=ig_atanh_series(u);
    const IgPair two_u=ig_add(u,u);
    const IgPair logm=ig_add(two_u,ig_mul(ig_mul(two_u,ig_mul(u,u)),s));
    return ig_add(ig_mul(ig_one(integer(k)),{ig_ln2,ig_ln2_lo}),logm);
}

// d = x-a is exact on this interval. Express the deviance a*(sigma-log1p(sigma))
// using u=sigma/(2+sigma), without subtracting nearly equal logarithms.
inline IgPair ig_near_deviance(Soft a, Soft d) {
    const IgPair sigma=ig_div(ig_one(d),ig_one(a));
    const IgPair denom=ig_add(ig_one(2),sigma);
    const IgPair u=ig_div(sigma,denom);
    const IgPair factor=ig_mul(ig_div(ig_one(2),denom),ig_mul(u,ig_atanh_series(u)));
    return ig_mul(ig_mul(ig_one(d),u),ig_add(ig_one(1),ig_neg(factor)));
}

inline Soft ig_exp_times(IgPair x, Soft factor) {
    if (x.hi < Soft(-1600)) return Soft(0);
    const Soft adjust=exponential(x.lo);
    if (x.hi < Soft(-680)) {
        const Soft h=exponential(x.hi*half64);
        return h*(h*(adjust*factor));
    }
    return exponential(x.hi)*(adjust*factor);
}

inline Soft ig_lgamma_series(Soft r) {
    Soft p=raw(ig_lgamma1_coeff[0]);
    for (uint k=1;k<60;++k) p=fma64(p,r,raw(ig_lgamma1_coeff[k]));
    return p;
}

// Reduce Gamma(1+a) to a Taylor expansion within half a unit of 1. The
// finite recurrence product is accumulated in two components before its log.
inline IgPair ig_lgamma1p(Soft a) {
    if (a<=half64) return ig_mul(ig_one(a),ig_one(ig_lgamma_series(a)));
    const uint n=uint(int(a+half64));
    const Soft r=a-integer(n);
    IgPair product=ig_one(1);
    for (uint k=1;k<=n;++k)
        product=ig_mul(product,ig_sum(r,integer(k)));
    const IgPair lp=ig_add(ig_log(product.hi),ig_one(product.lo/product.hi));
    return ig_add(lp,ig_mul(ig_one(r),ig_one(ig_lgamma_series(r))));
}

inline Soft ig_lgamma1p_div(Soft a) {
    return a<=half64 ? ig_lgamma_series(a) : ig_div(ig_lgamma1p(a),ig_one(a)).hi;
}

inline IgPair ig_power_argument(Soft a, Soft x) {
    const IgPair logx=ig_log(x);
    if (a <= half64)
        return ig_mul(ig_one(a),ig_add(logx,ig_one(-ig_lgamma1p_div(a))));
    return ig_add(ig_mul(ig_one(a),logx),ig_neg(ig_lgamma1p(a)));
}

// Temme's uniform expansion, DLMF 8.12.3/4, evaluated on the smaller side.
// Factoring erfc(z)=exp(-z*z)*erfcx(z) avoids magnifying sqrt rounding in a tail.
inline Soft ig_temme_tail(Soft a, Soft d, IgPair deviance, bool upper) {
    const Soft z=sqrt(deviance.hi);
    const Soft eta=copysign(z*(raw(0x3ff6a09e667f3bcdul)/sqrt(a)),d);
    Soft s=0, power=1;
    for (uint k=0;k<14;++k) {
        Soft ck=raw(ig_temme[k][24]);
        for (int j=23;j>=0;--j) ck=fma64(ck,eta,raw(ig_temme[k][j]));
        const Soft term=ck*power;
        s+=term;
        if (fabs(term)<=fabs(s)*epsilon64) break;
        power/=a;
    }
    const Soft sign=upper?Soft(1):Soft(-1);
    const Soft correction=(sign*s*ig_inv_sqrt_tau)/sqrt(a);
    if (z < Soft(1))
        return half64*complementary_erf(z)+ig_exp_times(ig_neg(deviance),correction);
    return ig_exp_times(ig_neg(deviance),half64*erfcx64(z)+correction);
}

// DLMF 8.7.3, factored by a so a subnormal shape is multiplied only after
// the finite coefficient has been assembled. This avoids subtracting P from 1.
inline Soft ig_small_upper(Soft a, Soft x) {
    const IgPair ell=ig_add(ig_log(x),ig_one(-ig_lgamma1p_div(a)));
    const IgPair tp=ig_mul(ig_one(a),ell);
    const Soft t=tp.hi;
    Soft ratio=Soft(1);
    if (fabs(t)>raw(0x3e30000000000000ul)) ratio=ol_s_expm1::expm1(t)/t;
    else ratio=fma64(t,half64,Soft(1));
    Soft term=1, sum=0, low=0;
    for (uint k=1;k<=64;++k) {
        const Soft n=integer(k);
        term*= -x/n;
        const Soft add=term/(a+n), y=add-low, v=sum+y;
        low=(v-sum)-y; sum=v;
        if (fabs(add)<=fabs(sum)*epsilon64) break;
    }
    const Soft et=ig_exp_times(tp,Soft(1));
    return a*fma64(-et,sum,-(ell.hi+ell.lo)*ratio);
}

inline Soft ig_series_factor(Soft a, Soft x, device atomic_uint *err) {
    Soft term=1,sum=1,low=0;
    for (uint k=1;k<=512;++k) {
        term*=x/(a+integer(k));
        const Soft y=term-low,v=sum+y;
        low=(v-sum)-y; sum=v;
        if (term<=sum*epsilon64) return sum;
    }
    atomic_store_explicit(err,1u,memory_order_relaxed);
    return qnan;
}

// DLMF 8.9.2. Scale adjacent numerator/denominator convergents together.
// Two-component accumulation makes the convergence test reliable even when
// adjacent binary64 ratios happen to round to the same value.
inline Soft ig_fraction_factor(Soft a, Soft x, device atomic_uint *err) {
    IgPair b=ig_add(ig_sum(x,-a),ig_one(1));
    IgPair pn2=ig_one(1),pn1=b,qn2=ig_one(0),qn1=ig_one(1);
    IgPair prev=ig_div(qn1,pn1);
    for (uint k=1;k<=512;++k) {
        const Soft n=integer(k);
        const IgPair an=ig_mul(ig_one(n),ig_sum(a,-n));
        b=ig_add(b,ig_one(2));
        const IgPair pn=ig_add(ig_mul(b,pn1),ig_mul(an,pn2));
        const IgPair qn=ig_add(ig_mul(b,qn1),ig_mul(an,qn2));
        const IgPair ratio=ig_div(qn,pn);
        if (fabs(ig_add(ratio,ig_neg(prev)).hi)<=fabs(ratio.hi)*raw(0x3c60000000000000ul))
            return ratio.hi;
        pn2=pn1; qn2=qn1; pn1=pn; qn1=qn; prev=ratio;
        if (fabs(pn1.hi)>raw(0x4630000000000000ul)) {
            const IgPair scale=ig_one(raw(0x39b0000000000000ul));
            pn2=ig_mul(pn2,scale);pn1=ig_mul(pn1,scale);
            qn2=ig_mul(qn2,scale);qn1=ig_mul(qn1,scale);
        }
    }
    atomic_store_explicit(err,2u,memory_order_relaxed);
    return qnan;
}

inline Soft incomplete_gamma(Soft a,Soft x,bool upper,device atomic_uint *error) {
    const Soft one=Soft(1), zero=Soft(0);
    // ATen tests negative inputs, zero shape, zero limit, infinite shape, then
    // infinite limit before propagating all remaining NaNs.
    if (a<zero || x<zero) return qnan;
    if (a==zero) return x>zero ? (upper?zero:one) : qnan;
    if (x==zero) return upper?one:zero;
    if (a==pinf) return x==pinf ? qnan : (upper?one:zero);
    if (x==pinf) return upper?zero:one;
    if (nan(a) || nan(x)) return qnan;
    if (a==one) return upper ? exponential(-x) : -ol_s_expm1::expm1(-x);
    if (a==half64) {
        const Soft z=sqrt(x);
        if (!upper) return ol_s_erf::erf(z);
        return x<one ? complementary_erf(z) : ig_exp_times(ig_one(-x),erfcx64(z));
    }

    const bool right=x>=a;
    const Soft endpoint=upper==right ? zero : one;
    if (x>Soft(4096) && a<x*raw(0x3fd0000000000000ul)) return endpoint;
    const Soft d=x-a;
    const bool near=fabs(d)<=a*half64 && a>=Soft(8);
    if (a>=Soft(8192) && !near) return endpoint;
    if (near && fabs(d)/sqrt(a)>Soft(50)) return endpoint;

    IgPair exponent=ig_one(0);
    if (a>=Soft(8)) {
        // Chernoff bound on either small tail is exp(-D). exp(-746) is below
        // half of the smallest positive subnormal binary64.
        IgPair dev;
        if (near) dev=ig_near_deviance(a,d);
        else {
            const IgPair logratio=ig_add(ig_log(x),ig_neg(ig_log(a)));
            dev=ig_add(ig_sum(x,-a),ig_neg(ig_mul(ig_one(a),logratio)));
        }
        if (dev.hi>Soft(746)) return endpoint;
        if (near && a>=Soft(64)) {
            const Soft tail=ig_temme_tail(a,d,dev,right);
            return upper==right ? tail : one-tail;
        }
        const Soft inv=one/a;
        exponent=ig_add(ig_neg(dev),ig_one(-inv*horner(inv*inv,ig_stirling)));
    }

    const bool use_q=x>=one && right;
    const bool small_q=a<one && x<one && a*logarithm(x)>Soft(-1);
    if (small_q && x<=raw(0x3fe8000000000000ul)) {
        const Soft q=ig_small_upper(a,x);
        return upper?q:one-q;
    }
    const bool fraction=use_q || (small_q && x>raw(0x3fe8000000000000ul));
    if (a<Soft(8)) exponent=ig_add(ig_power_argument(a,x),ig_one(-x));

    Soft factor=fraction?ig_fraction_factor(a,x,error):ig_series_factor(a,x,error);
    Soft tail;
    if (a>=Soft(8)) {
        factor=ig_inv_sqrt_tau*(fraction?factor*sqrt(a):factor/sqrt(a));
        tail=ig_exp_times(exponent,factor);
    } else {
        // Include an ordinary shape before the final possible underflow.
        // For a tiny shape, keep its multiplication last so a/x stays intact.
        const bool tiny_shape=a<raw(0x07b0000000000000ul);
        if (fraction && !tiny_shape) factor*=a;
        tail=ig_exp_times(exponent,factor);
        if (fraction && tiny_shape) tail*=a;
    }
    return upper==fraction ? tail : one-tail;
}

} // namespace special64

kernel void incomplete_gamma_kernel(device ulong *out,const device ulong *shape,
                                   const device ulong *limit,constant long *layout,
                                   constant uint &upper,device atomic_uint *error,
                                   uint i [[thread_position_in_grid]]) {
    const ulong2 p=pointwise_offsets(i,layout);
    out[i]=special64::incomplete_gamma(special64::raw(shape[p.x]), special64::raw(limit[p.y]),
                                       upper!=0,error).v;
}
