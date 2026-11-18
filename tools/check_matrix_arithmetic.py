"""Run Berkeley TestFloat on the arithmetic actually used by the new matrix kernels."""

import argparse
import ctypes
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from testfloat import build

ROOT = Path(__file__).resolve().parents[1] / "mps_fp64"
WRAPPERS = """
kernel void probe_add(device ulong *out,const device ulong *a,const device ulong *b,uint i [[thread_position_in_grid]]) { out[i]=gemv_fp64::add(a[i],b[i]); }
kernel void probe_mul(device ulong *out,const device ulong *a,const device ulong *b,uint i [[thread_position_in_grid]]) { out[i]=gemv_fp64::mul(a[i],b[i]); }
kernel void probe_outer(device ulong *out,const device ulong *a,const device ulong *b,uint i [[thread_position_in_grid]]) { out[i]=skinny_gemm::outer_product(a[i],b[i]); }
kernel void probe_first(device ulong *out,const device ulong *a,const device ulong *b,uint i [[thread_position_in_grid]]) { out[i]=skinny_gemm::pack(skinny_gemm::first(skinny_gemm::unpack(a[i]),skinny_gemm::unpack(b[i]))); }
kernel void probe_dense_fma(device ulong *out,const device ulong *a,const device ulong *b,const device ulong *c,uint i [[thread_position_in_grid]]) { out[i]=pack_dot_term(dense22_fma(dense22_decode(a[i]),dense22_decode(b[i]),unpack_dot_term(c[i]))); }
kernel void probe_short_fma(device ulong *out,const device ulong *a,const device ulong *b,const device ulong *c,uint i [[thread_position_in_grid]]) { out[i]=skinny_gemm::pack(skinny_gemm::step(skinny_gemm::unpack(a[i]),skinny_gemm::unpack(b[i]),skinny_gemm::unpack(c[i]))); }
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkout", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--binary-level", type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    resources = (
        "vendor/softfloat64.metal",
        "gemv_arithmetic.metal",
        "fma.metal",
        "gemm_dense.metal",
        "skinny_gemm.metal",
    )
    shader = torch.mps.compile_shader(
        "\n".join((ROOT / f).read_text() for f in resources) + "\n" + WRAPPERS
    )
    output = {
        "device": torch.backends.mps.get_name(),
        "rounding": "nearest-even",
        "NaN": "classification",
        "operations": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        ref = build(args.checkout.resolve(), Path(tmp))
        ref.begin_cases.restype = ctypes.c_uint64
        ref.make_cases.argtypes = [
            ctypes.c_int,
            ctypes.c_uint32,
            *([ctypes.c_void_p] * 4),
        ]
        for code, level, names in [
            (0, args.binary_level, ("add",)),
            (2, args.binary_level, ("mul", "first", "outer")),
            (5, 1, ("dense_fma", "short_fma")),
        ]:
            total = ref.begin_cases(code, level)
            errors = {n: 0 for n in names}
            for start in range(0, total, 65536):
                n = min(total - start, 65536)
                a, b, c, want = [np.empty(n, dtype=np.uint64) for _ in range(4)]
                ref.make_cases(code, n, *(x.ctypes.data for x in (a, b, c, want)))
                inputs = [
                    torch.from_numpy(x.view(np.int64)).to("mps")
                    for x in ((a, b, c) if code == 5 else (a, b))
                ]
                out = torch.empty_like(inputs[0])
                if code == 2:
                    # Relative to pure fmul, fma(a,b,+0) clears the sign of an
                    # exact zero product; gradual underflow of a nonzero stays signed.
                    adjusted = want.copy()
                    mag = np.uint64(0x7FFFFFFFFFFFFFFF)
                    finite = np.uint64(0x7FF0000000000000)
                    zeros = (((a & mag) == 0) & ((b & mag) < finite)) | (
                        ((b & mag) == 0) & ((a & mag) < finite)
                    )
                    adjusted[zeros] = 0
                for name in names:
                    getattr(shader, "probe_" + name)(out, *inputs, threads=n)
                    got = out.cpu().numpy().view(np.uint64)
                    expected = adjusted if name in ("first", "outer") else want
                    same = (got == expected) | (
                        np.isnan(got.view(np.float64))
                        & np.isnan(expected.view(np.float64))
                    )
                    errors[name] += int(np.count_nonzero(~same))
                    if not same.all() and errors[name] <= 16:
                        i = np.flatnonzero(~same)[0]
                        print(
                            name,
                            "first observed mismatch",
                            *[hex(int(v[i])) for v in (a, b, c, got, expected)],
                            flush=True,
                        )
            for name in names:
                output["operations"][name] = {
                    "cases": total,
                    "level": level,
                    "mismatches": errors[name],
                }
            print(total, errors, flush=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
    if any(r["mismatches"] for r in output["operations"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
