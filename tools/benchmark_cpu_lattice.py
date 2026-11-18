"""Compare GPU from raw FP64 with a CPU that is given the exact prepacked digits for free."""

import argparse
import ctypes
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import torch
from benchmark_large_matrices import (
    FLOAT64,
    cases,
    check_case,
    enable,
    inputs,
    interleave,
)


def pack(x, scale, radix):
    # These are the *known* dyadic coefficients of the benchmark inputs.
    integer = (x * 2.0**-scale).to(torch.int32)
    assert torch.equal(integer.to(FLOAT64) * 2.0**scale, x)
    if not radix:
        return integer.float()
    high = torch.div(integer + radix // 2, radix, rounding_mode="floor")
    low = integer - radix * high
    return torch.stack((low, high, low + high)).float()


def run(c, fn, i, rounds):
    a, b = inputs(c, 1024 + i)
    m, k, n = c.m, c.k, c.n
    # Pick a smaller balanced radix when available: it reduces the number of
    # CPU binary32 K panels for 12-bit inputs compared with fixed radix 256.
    one = k * c.amplitude**2 <= 2**24
    radix = 0 if one else (64 if c.amplitude == 2047 else 256)
    planes = 1 if one else 3
    bound = (radix // 2 + (c.amplitude + radix // 2) // radix) if radix else c.amplitude
    panel = min(k, (2**24) // (bound * bound))
    assert panel >= 1 and (planes == 3 or panel == k)
    packed_a = pack(a, -13, radix)
    packed_b = pack(b, -9, radix)
    temporary = torch.empty((planes, m, n), dtype=torch.float32)
    dst = torch.empty((m, n), dtype=FLOAT64)
    direct = torch.empty_like(dst)
    pointers = [x.data_ptr() for x in (packed_a, packed_b, dst, temporary)]

    def cpu_fast():
        fn(*pointers, m, k, n, panel, radix, planes, 2.0**-22)

    cpu_fast()
    torch.mm(a, b, out=direct)
    assert dst.view(torch.int64).equal(direct.view(torch.int64)), c.name
    with enable() as scope:
        x, y = a.to("mps"), b.to("mps")
        out = torch.empty(dst.shape, dtype=FLOAT64, device="mps")
        actual = torch.mm(x, y, out=out).cpu()
        assert not scope.cpu_ops
    verify = check_case(c, direct, dst, actual)
    timed = interleave(
        {
            "cpu_prepacked_sgemm": (nullcontext, cpu_fast, False),
            "cpu_fp64_out": (nullcontext, lambda: torch.mm(a, b, out=direct), False),
            "gpu_from_fp64_out": (enable, lambda: torch.mm(x, y, out=out), True),
        },
        rounds,
        1,
        2402 + i,
    )
    best = min(
        ("cpu_prepacked_sgemm", "cpu_fp64_out"),
        key=lambda name: timed[name]["median_ms"],
    )
    gpu = timed["gpu_from_fp64_out"]["median_ms"]
    ratio = timed[best]["median_ms"] / gpu
    print(
        c.name,
        {n: round(v["median_ms"], 3) for n, v in timed.items()},
        best,
        round(ratio, 3),
        flush=True,
    )
    return {
        "case": c.name,
        "seed": 1024 + i,
        "radix": radix,
        "planes": planes,
        "max_panel_k": panel,
        "max_product_sum_bound_per_plane_panel": panel * bound**2,
        "checked_output_words": verify["checked_output_words"],
        "cpu_dgemm_vs_prepacked_vs_gpu_mismatches": 0,
        "timing": timed,
        "best_cpu": best,
        "best_cpu_over_gpu": ratio,
        "cpu_prepacked_over_gpu": timed["cpu_prepacked_sgemm"]["median_ms"] / gpu,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("disable CPU fallback")
    report = {
        "torch": torch.__version__,
        "device": torch.backends.mps.get_name(),
        "macOS": platform.mac_ver()[0],
        "cpu_reference_sha256": hashlib.sha256(
            Path(__file__).with_name("matrix_cpu_reference.c").read_bytes()
        ).hexdigest(),
        "cpu_threads": torch.get_num_threads(),
        "method": "Synchronized individual GPU calls, shuffled with both CPU paths, median wall time. All paths return a full FP64 output into a preallocated buffer. The independent CPU algorithm gets the dyadic lattice, prepacked CPU FP32 digit planes and scratch space for free. Its timings include Accelerate SGEMM panels and CPU-parallel exact double reconstruction. The GPU starts from resident raw FP64 and includes all qualification and packing. Every output compared bitwise to native CPU FP64 DGEMM.",
        "rounds": args.rounds,
        "cases": [],
    }
    with tempfile.TemporaryDirectory() as folder:
        dylib = Path(folder) / "libmatrix_cpu.dylib"
        subprocess.run(
            [
                "xcrun",
                "clang",
                "-O3",
                "-Wno-deprecated-declarations",
                "-dynamiclib",
                str(Path(__file__).with_name("matrix_cpu_reference.c")),
                "-framework",
                "Accelerate",
                "-o",
                str(dylib),
            ],
            check=True,
        )
        fn = ctypes.CDLL(str(dylib)).packed_cpu_gemm
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_double]
        fn.restype = None
        for i, c in enumerate(cases()):
            if not c.amplitude or (args.only and c.name not in args.only):
                continue
            report["cases"].append(run(c, fn, i, args.rounds))
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
