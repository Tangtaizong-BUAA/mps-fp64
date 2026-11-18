"""Compare the original OpenLibm shaders with the certified fixed-point shaders."""

import argparse
import json
import platform
import re
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import gmpy2 as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_fp64.kernels import MATH_UNARY  # noqa: E402

NAMES = "sin cos tanh atan erf asinh exp expm1 log log2 log10".split()
MARKER = 0x7FF800000000CA51


def shader(audit):
    path = ROOT / "mps_fp64"
    base = "\n".join(
        (path / file).read_text()
        for file in (
            "vendor/softfloat64.metal",
            "soft_math.metal",
            "vendor/openlibm.metal",
            "fast_math_tables.metal",
        )
    )
    fast = (path / "fast_math.metal").read_text()
    parts = [base, fast]
    if audit:
        # Expose which inputs take the certified route, while running the original
        # uninstrumented production function separately on the same inputs.
        traced, sites = re.subn(
            r"return (?:trig|odd|log|exp)_full<[^>]+>\(x\);",
            f"return {MARKER}ul;",
            fast,
        )
        if sites != 8:
            raise RuntimeError(f"expected 8 fallback call sites, found {sites}")
        parts.append(traced.replace("namespace fast_math", "namespace traced_math"))
    for name in NAMES:
        old = (
            f"ol_e_{name}::__ieee754_{name}"
            if name in ("exp", "log", "log2", "log10")
            else f"ol_s_{name}::{name}"
        )
        expressions = {"old": old + "(Soft(x,true)).v", "new": MATH_UNARY[name]}
        if audit:
            expressions["traced"] = MATH_UNARY[name].replace(
                "fast_math::", "traced_math::"
            )
        for tag, expr in expressions.items():
            parts.append(
                f"kernel void {tag}_{name}(device ulong* out, const device ulong* a, uint i [[thread_position_in_grid]]) {{ulong x=a[i]; out[i]={expr};}}"
            )
    return torch.mps.compile_shader("\n".join(parts))


def sample(rng):
    edge = np.r_[
        np.arange(-1024, 1025) / 128,
        (-np.pi * 4, np.pi / 2, -np.pi, np.pi),
        np.power(2.0, np.arange(-1074, 1024)),
        [np.inf, -np.inf, np.nan, -0.0, 0.0],
    ]
    raw = rng.bit_generator.random_raw(8192).view(np.float64)
    return np.r_[
        rng.normal(size=17000),
        rng.uniform(-8, 8, 17000),
        raw,
        edge,
        np.nextafter(edge, np.inf),
        np.nextafter(edge, -np.inf),
    ]


def midpoint_inputs(name):
    # Construct inverse images of binary64 rounding midpoints with MPFR, then
    # test the adjoining two representable inputs on each side.
    with mp.context(mp.get_context(), precision=350):
        inv = {
            "sin": mp.asin,
            "cos": mp.acos,
            "tanh": mp.atanh,
            "atan": mp.tan,
            "asinh": mp.sinh,
            "exp": mp.log,
            "expm1": mp.log1p,
            "log": mp.exp,
            "log2": mp.exp2,
            "log10": lambda v: mp.mpfr(10) ** v,
        }
        if name not in inv:
            return np.array([])
        low, high = (
            (0.1, 0.99)
            if name in ("sin", "cos", "tanh")
            else (0.15, 1.4) if name == "atan" else (0.1, 2.0)
        )
        values = np.random.default_rng(3962).uniform(low, high, 1024)
        upper = np.nextafter(values, np.inf)
        center = np.array(
            [
                float(inv[name]((mp.mpfr(x) + mp.mpfr(y)) / 2))
                for x, y in zip(values, upper)
            ]
        )
        down, up = np.nextafter(center, -np.inf), np.nextafter(center, np.inf)
        return np.r_[
            np.nextafter(down, -np.inf), down, center, up, np.nextafter(up, np.inf)
        ]


def upload(a):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int64)).to("mps")


def ordered(a):
    u = a.view(np.uint64)
    return np.where(u >> 63, ~u, u ^ np.uint64(1 << 63))


def accuracy(lib, values):
    rows = {}
    for name in NAMES:
        host = np.r_[values, midpoint_inputs(name)]
        x = upload(host)
        out = torch.empty_like(x)
        outputs = {}
        for tag in ("old", "new", "traced"):
            getattr(lib, f"{tag}_{name}")(out, x, threads=x.numel())
            outputs[tag] = out.cpu().numpy().view(np.float64).copy()
        refs = []
        for precision in (220, 350):
            with mp.context(mp.get_context(), precision=precision):
                refs.append(
                    np.array(
                        [float(getattr(mp, name)(mp.mpfr(float(i)))) for i in host]
                    )
                )
        ref = refs[-1]
        non_nan = ~np.isnan(ref)
        for result in (refs[0], outputs["new"], outputs["old"]):
            np.testing.assert_array_equal(np.isnan(result), ~non_nan)
        np.testing.assert_array_equal(
            refs[0][non_nan].view(np.uint64), ref[non_nan].view(np.uint64)
        )
        fast = outputs["traced"].view(np.uint64) != MARKER
        np.testing.assert_array_equal(
            outputs["traced"][fast].view(np.uint64),
            outputs["new"][fast].view(np.uint64),
        )
        np.testing.assert_array_equal(
            outputs["new"][fast].view(np.uint64), ref[fast].view(np.uint64)
        )
        # An ambiguous interval or outside-domain argument must execute OpenLibm unchanged.
        np.testing.assert_array_equal(
            outputs["new"][~fast].view(np.uint64), outputs["old"][~fast].view(np.uint64)
        )
        expected = ordered(ref[non_nan])
        errors = {}
        for tag in ("old", "new"):
            actual = ordered(outputs[tag][non_nan])
            errors[tag] = np.maximum(expected, actual) - np.minimum(expected, actual)
        improved = int(np.count_nonzero(errors["new"] < errors["old"]))
        worse = int(np.count_nonzero(errors["new"] > errors["old"]))
        zero = ref == 0
        assert np.array_equal(np.signbit(outputs["new"][zero]), np.signbit(ref[zero]))
        assert worse == 0
        rows[name] = {
            "samples": len(host),
            "base_samples": len(values),
            "midpoint_neighbours": len(host) - len(values),
            "certified_bit_exact": int(fast.sum()),
            "full_range_inputs": int((~fast).sum()),
            "old_max_ulp": int(errors["old"].max()),
            "new_max_ulp": int(errors["new"].max()),
            "improved": improved,
            "worse": worse,
            "zero_sign_mismatches": 0,
        }
        print(name, rows[name], flush=True)
    return rows


def timing(lib, rng):
    n = 2**20
    normal = upload(rng.normal(size=n))
    positive = upload(np.exp(rng.normal(size=n)))
    out = torch.empty_like(normal)
    results = {}
    for name in NAMES:
        x = positive if name.startswith("log") else normal
        kernels = {tag: getattr(lib, f"{tag}_{name}") for tag in ("old", "new")}
        samples = {k: [] for k in kernels}
        for fn in kernels.values():
            for _ in range(3):
                fn(out, x, threads=n)
        torch.mps.synchronize()
        for round_no in range(7):
            order = list(kernels) if round_no % 2 else list(reversed(kernels))
            for tag in order:
                start = perf_counter()
                for _ in range(8):
                    kernels[tag](out, x, threads=n)
                torch.mps.synchronize()
                samples[tag].append((perf_counter() - start) * 1000 / 8)
        medians = {tag: median(ms) for tag, ms in samples.items()}
        results[name] = {
            "median_ms": medians,
            "speedup": medians["old"] / medians["new"],
            "samples_ms": samples,
        }
        print(name, medians, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("accuracy", "timing"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(83345)
    values = sample(rng)
    lib = shader(args.mode == "accuracy")
    data = {
        "device": torch.backends.mps.get_name(),
        "gpu_cores": torch.backends.mps.get_core_count(),
        "torch": torch.__version__,
        "macos": platform.mac_ver()[0],
        "mpfr": mp.mpfr_version(),
        "seed": 83345,
    }
    if args.mode == "accuracy":
        data.update(
            reference_precision_bits=[220, 350], operations=accuracy(lib, values)
        )
    else:
        data.update(
            elements=2**20,
            distribution="standard normal; exp(standard normal) for log/log2/log10",
            methodology="preallocated resident inputs and output, warmups 3 per variant, alternating order; 7 synchronized batches of 8 dispatches",
            operations=timing(lib, rng),
        )
    args.output.write_text(json.dumps(data, indent=2) + "\n")
