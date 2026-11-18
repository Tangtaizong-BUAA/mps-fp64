"""IEEE 754 binary64 addition on Apple GPUs using integer Metal instructions.

Run with a Python environment that has NumPy and PyTorch with MPS support.
The kernel uses nearest-even rounding and returns a canonical quiet NaN.
"""

import numpy as np
import torch


SHADER = r"""
#include <metal_stdlib>
using namespace metal;

constant ulong SIGN = 1ul << 63;
constant ulong HIDDEN = 1ul << 52;
constant ulong FRACTION = HIDDEN - 1;
constant ulong INF = 0x7fful << 52;
constant ulong NAN_BITS = INF | (HIDDEN >> 1);

// Preserve any discarded nonzero bits in the least significant bit.
inline ulong shift_right_sticky(ulong value, uint shift) {
    if (shift == 0) return value;
    if (shift >= 64) return ulong(value != 0);
    return (value >> shift) | ulong((value << (64 - shift)) != 0);
}

inline ulong add_binary64(ulong a, ulong b) {
    ulong abs_a = a & ~SIGN;
    ulong abs_b = b & ~SIGN;
    if (abs_a < abs_b) {
        ulong old_a = a;
        a = b;
        b = old_a;
        ulong old_abs_a = abs_a;
        abs_a = abs_b;
        abs_b = old_abs_a;
    }
    bool subtract = ((a ^ b) & SIGN) != 0;
    ulong sign = a & SIGN;
    if (abs_a > INF || (abs_a == INF && abs_b == INF && subtract)) return NAN_BITS;
    if (abs_a == INF) return a;
    if (subtract && abs_a == abs_b) return 0;
    if (abs_b == 0) return a;

    uint exp_a = uint(abs_a >> 52);
    uint exp_b = uint(abs_b >> 52);
    ulong sig_a = (abs_a & FRACTION) | (exp_a == 0 ? 0ul : HIDDEN);
    ulong sig_b = (abs_b & FRACTION) | (exp_b == 0 ? 0ul : HIDDEN);
    uint exponent = max(exp_a, 1u);
    uint gap = exponent - max(exp_b, 1u);

    // Keep guard, round, and sticky bits below the significand.
    sig_a <<= 3;
    sig_b = shift_right_sticky(sig_b << 3, gap);
    ulong sig = subtract ? sig_a - sig_b : sig_a + sig_b;
    if (sig >= (HIDDEN << 4)) {
        sig = shift_right_sticky(sig, 1);
        ++exponent;
    } else if (sig < (HIDDEN << 3) && exponent > 1) {
        uint high = uint(sig >> 32);
        uint leading_zeros = high ? clz(high) : 32u + clz(uint(sig));
        uint shift = min(leading_zeros - 8u, exponent - 1u);
        sig <<= shift;
        exponent -= shift;
    }

    ulong rounded = sig >> 3;
    uint remainder = uint(sig & 7ul);
    if (remainder > 4 || (remainder == 4 && (rounded & 1ul))) ++rounded;
    if (rounded >= (HIDDEN << 1)) {
        rounded >>= 1;
        ++exponent;
    }
    if (exponent >= 0x7ffu) return sign | INF;
    if (exponent == 1 && rounded < HIDDEN) exponent = 0;
    return sign | (ulong(exponent) << 52) | (rounded & FRACTION);
}

kernel void fp64_add_bits(device ulong* result,
                          const device ulong* a,
                          const device ulong* b,
                          uint i [[thread_position_in_grid]]) {
    result[i] = add_binary64(a[i], b[i]);
}
"""


def upload_bits(values):
    """Reinterpret CPU float64 values as int64 before copying to MPS."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    return torch.from_numpy(values.view(np.int64)).to("mps")


def validate(kernel, a, b):
    a_gpu, b_gpu = upload_bits(a), upload_bits(b)
    result_gpu = torch.empty_like(a_gpu)
    kernel(result_gpu, a_gpu, b_gpu, threads=(result_gpu.numel(),))
    actual = result_gpu.cpu().numpy().view(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        expected = a + b
    matches = (actual.view(np.uint64) == expected.view(np.uint64)) | (
        np.isnan(actual) & np.isnan(expected)
    )
    np.testing.assert_array_equal(matches, True)
    return actual


def main():
    kernel = torch.mps.compile_shader(SHADER).fp64_add_bits
    example = validate(kernel, np.array([1.0]), np.array([2.0 ** -40]))
    print(f"1 + 2**-40 = {example[0].hex()}")

    corner = np.array([
        0x0000000000000000,  # +0
        0x8000000000000000,  # -0
        0x0000000000000001,  # smallest subnormal
        0x000fffffffffffff,  # largest subnormal
        0x0010000000000000,  # smallest normal
        0x0010000000000001,
        0x3ca0000000000000,  # half an ULP at 1
        0x3ca0000000000001,  # just over half an ULP
        0x3fefffffffffffff,
        0x3ff0000000000000,  # 1
        0x3ff0000000000001,
        0x7c90000000000000,  # half an ULP at the largest finite value
        0x7fefffffffffffff,  # largest finite
        0x7ff0000000000000,  # infinity
        0x7ff0000000000001,  # signaling NaN
        0x7ff8000000000001,  # quiet NaN
    ], dtype=np.uint64)
    corner = np.unique(np.concatenate((corner, corner ^ np.uint64(1 << 63))))
    values = corner.view(np.float64)
    validate(kernel, np.repeat(values, len(values)), np.tile(values, len(values)))

    rng = np.random.default_rng(20261117)
    bits = rng.bit_generator.random_raw((2, 1_000_000))
    validate(kernel, *bits.view(np.float64))

    # Independently sample the signs and significands, with small exponent gaps.
    # These pairs exercise rounding and cancellation more often than uniform bits.
    bits = rng.bit_generator.random_raw((2, 1_000_000))
    mask = np.uint64(0x800fffffffffffff)
    exponent = rng.integers(0, 2047, size=bits.shape[1])
    nearby = np.clip(exponent + rng.integers(-4, 5, size=exponent.size), 0, 2046)
    bits[0] = (bits[0] & mask) | (exponent.astype(np.uint64) << 52)
    bits[1] = (bits[1] & mask) | (nearby.astype(np.uint64) << 52)
    validate(kernel, *bits.view(np.float64))

    print(f"{1 + len(values) ** 2 + 2 * bits.shape[1]:,} pairs matched CPU float64.")


if __name__ == "__main__":
    main()
