"""Compare same-device binary64 FMA throughput across recurrence, dot and GEMM kernels."""

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from statistics import median
from time import perf_counter_ns

import numpy as np
import torch


CHAINS = (1, 2, 4, 8)
TABLE_WIDTH = 16
SOURCE_DIR = Path(__file__).resolve().parents[1] / "mps_fp64"

MICRO_KERNELS = r"""
kernel void bench_decode(device DotTerm *out, const device ulong *bits,
                         uint i [[thread_position_in_grid]]) {
    out[i] = unpack_dot_term(bits[i]);
}

// Every product has a changing multiplicand. Two dependent FMA updates form
// one recurrence. Separate recurrences within a thread have no dependencies.
// No explicit loads, stores or threadgroup barriers occur in the inner loop.
template<uint Chains>
inline void register_work(device ulong2 *out, const device ulong *seeds, uint patterns,
                          uint steps, uint tid) {
    DotTerm x[Chains], y[Chains], plus[Chains], minus[Chains];
    #pragma unroll
    for (uint i = 0; i < Chains; ++i) {
        uint index = (tid * Chains + i) % patterns;
        x[i]     = unpack_dot_term(seeds[3 * index]);
        y[i]     = unpack_dot_term(seeds[3 * index + 1]);
        plus[i]  = unpack_dot_term(seeds[3 * index + 2]);
        minus[i] = {-plus[i].mantissa, plus[i].exponent};
    }
    for (uint t = 0; t < steps / 2; ++t) {
        #pragma unroll
        for (uint i = 0; i < Chains; ++i)
            x[i] = dot_fma(plus[i], y[i], x[i]);
        #pragma unroll
        for (uint i = 0; i < Chains; ++i)
            y[i] = dot_fma(minus[i], x[i], y[i]);
    }
    #pragma unroll
    for (uint i = 0; i < Chains; ++i)
        out[tid * Chains + i] = ulong2(pack_dot_term(x[i]), pack_dot_term(y[i]));
}

// Repeat the dot products from the upper-left 16x16 GEMM output block.
// Inputs change at each k. Small shared device buffers are predecoded once.
template<uint Chains>
inline void table_work(device ulong *out, const device DotTerm *a, const device DotTerm *b,
                       uint steps, uint tid) {
    constexpr uint width = 16;
    DotTerm sum[Chains];
    #pragma unroll
    for (uint i = 0; i < Chains; ++i) sum[i] = unpack_dot_term(0);
    for (uint t = 0; t < steps; ++t) {
        #pragma unroll
        for (uint i = 0; i < Chains; ++i) {
            uint index = tid * Chains + i;
            sum[i] = dot_fma(a[t * width + (index / width) % width],
                             b[t * width + index % width], sum[i]);
        }
    }
    #pragma unroll
    for (uint i = 0; i < Chains; ++i) out[tid * Chains + i] = pack_dot_term(sum[i]);
}

#define ENTRY(CHAINS) \
kernel void reg_##CHAINS(device ulong2 *out, const device ulong *seeds, constant uint &patterns, \
                         constant uint &steps, uint tid [[thread_position_in_grid]]) { \
    register_work<CHAINS>(out, seeds, patterns, steps, tid); \
} \
kernel void table_##CHAINS(device ulong *out, const device DotTerm *a, const device DotTerm *b, \
                           constant uint &steps, uint tid [[thread_position_in_grid]]) { \
    table_work<CHAINS>(out, a, b, steps, tid); \
}
ENTRY(1)
ENTRY(2)
ENTRY(4)
ENTRY(8)
"""


def fma_oracle():
    if hasattr(math, "fma"):
        return math.fma
    fn = ctypes.CDLL(None).fma
    fn.argtypes = [ctypes.c_double] * 3
    fn.restype = ctypes.c_double
    return fn


def hardware_references(seeds, a, b):
    fma = fma_oracle()
    size = a.shape[1]
    recurrent = []
    for x, y, factor in seeds.tolist():
        for _ in range(size // 2):
            x = fma(factor, y, x)
            y = fma(-factor, x, y)
        recurrent.append((x, y))

    rows = a[:TABLE_WIDTH].tolist()
    cols = b[:, :TABLE_WIDTH].T.tolist()
    block = np.zeros((TABLE_WIDTH, TABLE_WIDTH), dtype=np.float64)
    for row, av in enumerate(rows):
        for col, bv in enumerate(cols):
            s = 0.0
            for ai, bi in zip(av, bv):
                s = fma(ai, bi, s)
            block[row, col] = s
    return np.asarray(recurrent).view(np.int64), block.view(np.int64)


def as_device_bits(values):
    return torch.from_numpy(np.ascontiguousarray(values).view(np.int64)).to("mps")


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def sample(inputs, output):
    """Outside the timed region, check every repeated micro result or the sampled GEMM block."""
    kind, expected = inputs
    got = output.cpu().numpy()
    if kind == "gemm":
        checked = got[:TABLE_WIDTH, :TABLE_WIDTH]
    else:
        checked = got
    if not np.array_equal(checked, expected):
        raise RuntimeError(f"{kind}: {np.count_nonzero(checked != expected)} binary64 words differ")
    return {
        "checked_words": checked.size,
        "output_sha256": digest(got),
        "expected_checked_sha256": digest(expected),
        "observed_checked_sha256": digest(checked),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="run small correctness checks without timing")
    parser.add_argument("--size", type=int, help="square GEMM size and FMA steps per micro stream, default 1024 (32 in check-only mode)")
    parser.add_argument("--repeats", type=int, default=7, help="number of independently ordered timing rounds")
    parser.add_argument("--seed", type=int, default=39917)
    parser.add_argument("--output", type=Path, help="optional JSON report")
    args = parser.parse_args()
    size = args.size if args.size is not None else (32 if args.check_only else 1024)
    if size < TABLE_WIDTH or size % 8 or size**2 > 2**32 - 1 or args.repeats < 1:
        parser.error("Choose a size divisible by 8 between 16 and 65528, and at least one repeat.")
    if not torch.backends.mps.is_available():
        parser.error("This benchmark requires MPS and torch.mps.compile_shader.")

    files = ("vendor/softfloat64.metal", "fma.metal", "gemm.metal")
    parts = [(SOURCE_DIR / name).read_text() for name in files]
    library = torch.mps.compile_shader("\n".join([parts[0], "using namespace metal;", *parts[1:], MICRO_KERNELS]))
    rng = np.random.default_rng(args.seed)
    streams = size**2
    patterns = min(512, streams)
    # Each finite, bounded rotation changes both states. The runtime factors
    # and states prevent hoisting a constant product from the recurrence.
    seeds = np.column_stack((
        rng.uniform(0.5, 1.5, patterns) * rng.choice([-1., 1.], patterns),
        rng.uniform(0.5, 1.5, patterns) * rng.choice([-1., 1.], patterns),
        rng.uniform(0.003, 0.012, patterns),
    ))
    a = rng.standard_normal((size, size))
    b = rng.standard_normal((size, size))
    recurrence, block = hardware_references(seeds, a, b)
    # Tile the independent CPU references to match the explicit GPU pattern
    # mapping. Compare the full output including every work item.
    recurrent_ref = np.resize(recurrence, (streams, 2))
    table_ref = np.resize(block.ravel(), streams)

    reg_out = torch.empty((streams, 2), dtype=torch.int64, device="mps")
    tab_out = torch.empty(streams, dtype=torch.int64, device="mps")
    gemm_out = torch.empty((size, size), dtype=torch.int64, device="mps")
    seed_bits, a_bits, b_bits = (as_device_bits(x) for x in (seeds, a, b))

    def decoded_table(values):
        packed = as_device_bits(values)
        # DotTerm has an int64 mantissa and an int32 exponent in a 16-byte
        # record. The GPU writes the same struct which the measured kernel reads.
        decoded = torch.empty((packed.numel(), 2), dtype=torch.int64, device="mps")
        library.bench_decode(decoded, packed, threads=(packed.numel(),))
        return decoded

    a_table = decoded_table(a[:TABLE_WIDTH].T)
    b_table = decoded_table(b[:, :TABLE_WIDTH])
    methods = {}
    checks = {}
    for count in CHAINS:
        jobs = streams // count
        for prefix, buffers, reference, out in (
            ("reg", (reg_out, seed_bits, patterns, size), recurrent_ref, reg_out),
            ("table", (tab_out, a_table, b_table, size), table_ref, tab_out),
        ):
            name = f"{prefix}_{count}"
            kernel = getattr(library, name)
            def launch(kernel=kernel, buffers=buffers, jobs=jobs):
                kernel(*buffers, threads=(jobs,), group_size=(min(jobs, 128),))
            methods[name] = launch
            checks[name] = ((prefix, reference), out)

    def launch_gemm():
        library.mm_staged(gemm_out, a_bits, b_bits, size, size, size,
                          threads=(((size + 15) // 16) * 16, size, 1),
                          group_size=(16, 8, 1))

    methods["gemm"] = launch_gemm
    checks["gemm"] = (("gemm", block), gemm_out)
    timings = {name: [] for name in methods}
    result = {}
    for name, launch in methods.items():
        # Start each independent validation with sentinel-filled outputs: an
        # omitted store cannot inherit the previous otherwise identical result.
        checks[name][1].fill_(-1)
        launch()
        result[name] = sample(*checks[name])

    rounding_ops = size**3
    report = {
        "device": torch.backends.mps.get_name(), "torch": torch.__version__, "macOS": platform.mac_ver()[0],
        "seed": args.seed, "size": size, "logical_streams": streams,
        "fmas_per_launch": rounding_ops, "FLOPs_per_FMA": 2, "check_only": args.check_only,
        "shader_sha256": {name: hashlib.sha256(part.encode()).hexdigest() for name, part in zip(files, parts)},
        "register_seed_patterns": patterns, "dot_reference_block": [TABLE_WIDTH, TABLE_WIDTH],
        "method": {
            "register": "1, 2, 4, or 8 independent pairs per thread. Each step is a rounded dot_fma using register-resident source-level arithmetic. Alternating updates x=fma(f,y,x), y=fma(-f,x,y) prevent a loop-invariant product. Factors and states are runtime inputs. Register allocation is compiler-dependent.",
            "table": "Same number of chains per thread. Predecoded operands change at every k. Repeated rows and columns come from the actual GEMM input. Two shared device buffers occupy 2*K*16*16 bytes and avoid GEMM tile staging. Cache behavior is device-dependent.",
            "gemm": "Shipping mm_staged, standard-normal device-resident contiguous A/B, preallocated output. All micro outputs and the upper-left 16x16 GEMM block are compared bitwise to independently iterated CPU fma.",
            "timing": "Synchronized wall-clock time per dispatch, initialized pipelines, preallocated outputs. All variants perform size cubed binary64 FMA with one nearest-even rounding per FMA. Timing order is shuffled each round.",
            "interpretation": "The maximum observed register-recurrence median throughput is a same-machine arithmetic reference for these specific inputs and kernels. Different dependency graphs, data distributions, cache traffic and compiled code affect comparisons.",
        },
        "measurements": result,
    }
    if not args.check_only:
        order = random.Random(args.seed)
        for _ in range(args.repeats):
            names = list(methods)
            order.shuffle(names)
            for name in names:
                torch.mps.synchronize()
                start = perf_counter_ns()
                methods[name]()
                torch.mps.synchronize()
                timings[name].append((perf_counter_ns() - start) / 1e6)
        for name, elapsed in timings.items():
            ms = median(elapsed)
            result[name].update(median_ms=ms, samples_ms=elapsed,
                                GFLOP_s=2 * rounding_ops / ms / 1e6)
            print(f"{name:8} {ms:8.3f} ms   {result[name]['GFLOP_s']:8.2f} GFLOP/s")
        best = max((f"reg_{c}" for c in CHAINS), key=lambda name: result[name]["GFLOP_s"])
        report["fastest_measured_recurrence"] = best
        report["gemm_over_fastest_measured_recurrence"] = result["gemm"]["GFLOP_s"] / result[best]["GFLOP_s"]
    else:
        for name, row in result.items():
            print(f"{name:8} {row['checked_words']:7} words match  SHA256 {row['observed_checked_sha256']}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
