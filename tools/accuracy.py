"""Measure ULP error against MPFR, increasing precision until references agree."""

import argparse
import json
from pathlib import Path
import sys
import gmpy2 as mp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mps_fp64 import enable


def ordered(a):
    bits = a.view(np.uint64)
    return np.where((bits >> 63) != 0, ~bits, bits ^ np.uint64(1 << 63))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/mpfr.json",
    )
    report_path = cli.parse_args().report
    rng = np.random.default_rng(8472)
    raw = rng.bit_generator.random_raw(1536).view(np.float64)
    near = np.array(
        [
            np.nextafter(v, d)
            for v in (-1.0, 0.0, 1.0, np.pi / 2, 709.78, -745.0)
            for d in (np.inf, -np.inf)
        ]
    )
    values = np.r_[raw[np.isfinite(raw)], rng.uniform(-8, 8, 1024), near, 0.0, -0.0]
    ops = [
        "sin",
        "cos",
        "tan",
        "exp",
        "expm1",
        "log",
        "log1p",
        "atan",
        "asin",
        "acos",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "sinh",
        "cosh",
        "erf",
        "erfc",
        "sqrt",
        "log2",
        "log10",
        "exp2",
    ]
    output = {}
    with enable() as mode:
        gpu = torch.from_numpy(values).to("mps")
        for name in ops:
            expected = None
            for precision in (180, 340, 680):
                with mp.context(mp.get_context(), precision=precision):
                    fn = getattr(mp, name)
                    arr = np.array([float(fn(mp.mpfr(float(v)))) for v in values])
                if expected is not None and np.array_equal(
                    arr, expected, equal_nan=True
                ):
                    break
                expected = arr
            else:
                raise ArithmeticError("MPFR reference not stabilized for " + name)
            result = getattr(torch, name)(gpu).cpu().numpy()
            nan = np.isnan(expected)
            assert np.array_equal(np.isnan(result), nan), name
            assert np.array_equal(np.isinf(result), np.isinf(expected)), name
            a, b = ordered(result[~nan]), ordered(expected[~nan])
            error = np.maximum(a, b) - np.minimum(a, b)
            maxerr = int(error.max(initial=0))
            zeros = (expected == 0) & (result == 0)
            signs = int(
                np.count_nonzero(
                    np.signbit(expected[zeros]) != np.signbit(result[zeros])
                )
            )
            output[name] = {
                "samples": len(values),
                "max_ulp": maxerr,
                "different_zero_sign": signs,
                "exact_including_zero_sign": int(np.count_nonzero(error == 0)),
                "precision_bits": precision,
            }
            print(name, maxerr, signs, flush=True)
        assert not mode.cpu_ops
    report_path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
