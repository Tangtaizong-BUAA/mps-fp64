"""Run Berkeley TestFloat cases against this package; use a pinned metal-softfloat checkout."""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mps_fp64 import enable
from mps_fp64.kernels import library


def build(repo, folder):
    sf = repo / "vendor/SoftFloat-3e"
    tf = repo / "vendor/TestFloat-3e"
    inc = sf / "source/include"
    spec = sf / "source/8086-SSE"
    common = ["clang", "-O2", "-dynamiclib", "-w"]
    lib = folder / "libsf.dylib"
    subprocess.run(
        [
            *common,
            "-I",
            sf,
            "-I",
            inc,
            "-I",
            spec,
            "-DSOFTFLOAT_FAST_INT64",
            "-DSOFTFLOAT_ROUND_ODD",
            "-DINLINE_LEVEL=5",
            "-DSOFTFLOAT_FAST_DIV32TO16",
            "-DSOFTFLOAT_FAST_DIV64TO32",
            *(sf / "source").glob("*.c"),
            *spec.glob("*.c"),
            "-o",
            lib,
        ],
        check=True,
    )
    bridge = folder / "bridge.dylib"
    subprocess.run(
        [
            *common,
            "-I",
            tf,
            "-I",
            tf / "source",
            "-I",
            inc,
            "-DFLOAT64",
            *(tf / "source").glob("*.c"),
            Path(__file__).with_name("testfloat_bridge.c"),
            lib,
            "-o",
            bridge,
        ],
        check=True,
    )
    return ctypes.CDLL(str(bridge))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkout", type=Path)
    p.add_argument("--level", type=int, choices=(1, 2), default=1)
    p.add_argument(
        "--ops", nargs="+", default=["add", "sub", "mul", "div", "sqrt", "fma"]
    )
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    report = {}
    with tempfile.TemporaryDirectory() as tmp, enable():
        ref = build(args.checkout.resolve(), Path(tmp))
        ref.begin_cases.restype = ctypes.c_uint64
        ref.make_cases.argtypes = [
            ctypes.c_int,
            ctypes.c_uint32,
            *([ctypes.c_void_p] * 4),
        ]
        for name in args.ops:
            code = ["add", "sub", "mul", "div", "sqrt", "fma"].index(name)
            total = ref.begin_cases(code, args.level)
            checked = 0
            mismatch = 0
            while checked < total:
                size = min(65536, total - checked)
                a, b, c, expected = [np.empty(size, dtype=np.uint64) for _ in range(4)]
                ref.make_cases(
                    code, size, *(arr.ctypes.data for arr in (a, b, c, expected))
                )
                raw = [
                    torch.from_numpy(arr.view(np.int64)).to("mps") for arr in (a, b, c)
                ]
                if name == "fma":
                    out = torch.empty_like(raw[0])
                    library().test_nearby_fma(out, *raw, threads=(size,))
                else:
                    inputs = [
                        v.view(torch.float64) for v in raw[: 1 if name == "sqrt" else 2]
                    ]
                    out = getattr(torch, name)(*inputs).view(torch.int64)
                got = out.cpu().numpy().view(np.uint64)
                same = (got == expected) | (
                    np.isnan(got.view(np.float64)) & np.isnan(expected.view(np.float64))
                )
                mismatch += int(np.count_nonzero(~same))
                checked += size
            report[name] = {
                "level": args.level,
                "cases": checked,
                "mismatches": mismatch,
                "rounding": "nearest-even",
                "NaN": "classification",
            }
            print(name, checked, mismatch, flush=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    if any(v["mismatches"] for v in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
