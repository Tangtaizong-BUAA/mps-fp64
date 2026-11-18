"""Compare GPU and CPU independently to high-precision real reference values."""

import argparse
import json
import math
from pathlib import Path
import random
import struct
import sys

import mpmath as mp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mps_fp64 import enable


def key(x):
    u = int.from_bytes(struct.pack(">d", x), "big")
    return (2**64 - 1 - u) if (u >> 63) else (u + 2**63)


def metrics(values, refs, inputs):
    errors = [
        (abs(v - r) / max(1.0, abs(r)), abs(key(v) - key(r)), i)
        for i, (v, r) in enumerate(zip(values, refs))
        if math.isfinite(v) and math.isfinite(r)
    ]
    err, _, idx = max(errors)
    _, max_ulp, ulp_idx = max(errors, key=lambda item: item[1])
    relative = max(
        (
            abs(v - r) / abs(r)
            for v, r in zip(values, refs)
            if math.isfinite(v) and math.isfinite(r) and abs(r) >= sys.float_info.min
        ),
        default=0.0,
    )
    mismatches = sum(
        (math.isfinite(v) != math.isfinite(r))
        or (not math.isfinite(v) and not (v == r or math.isnan(v) and math.isnan(r)))
        for v, r in zip(values, refs)
    )
    return {
        "nonfinite_mismatches": mismatches,
        "max_abs_over_max_1_abs_ref": err,
        "max_ulp_finite": max_ulp,
        "worst_ulp_input": inputs[ulp_idx],
        "max_relative_for_normal_ref": relative,
        "worst_scaled_input": inputs[idx],
        "worst_scaled_result": values[idx],
        "worst_scaled_ref": refs[idx],
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("report", type=Path)
    args = cli.parse_args()
    records = {}
    rng = random.Random(9237)

    def check(name, op, ref, inputs):
        args_list = list(zip(*[v if isinstance(v, tuple) else (v,) for v in inputs]))
        data = [torch.tensor(v, dtype=torch.float64) for v in args_list]
        if any(isinstance(v, tuple) for v in inputs):

            def reference(v):
                return ref(*map(mp.mpf, v))
        else:

            def reference(v):
                return ref(mp.mpf(v))

        def oracle(dps):
            with mp.workdps(dps):
                return [float(mp.re(reference(v))) for v in inputs]

        refs = oracle(175)
        if refs != oracle(225):
            raise AssertionError(f"{name} reference precision did not converge")
        cpu = op(*data).tolist()
        with enable() as mode:
            gpu = op(*(a.to("mps") for a in data)).cpu().tolist()
            assert not mode.cpu_ops
        records[name] = {
            "n": len(inputs),
            "gpu": metrics(gpu, refs, inputs),
            "torch_cpu": metrics(cpu, refs, inputs),
        }

    x = [rng.uniform(-8, 12) for _ in range(108)] + [
        0.25,
        0.5,
        1.0,
        1.4616321449683623,
        2.0,
        10.0,
        30.0,
        200.0,
    ]
    poles = [
        math.nextafter(float(p), toward)
        for p in (-1000, -12, -3, -1)
        for toward in (-math.inf, math.inf)
    ]
    check(
        "lgamma",
        torch.lgamma,
        lambda x: mp.log(abs(mp.gamma(x))),
        x + poles + [1e-300, 1e150],
    )
    check("digamma", torch.digamma, mp.digamma, x + poles + [1e-300, 1e150])
    # Separate the finite-domain published CPU approximation from exact-pole conventions.
    for n in (1, 2, 3, 4, 10, 50, 160, 170):
        samples = (
            x + poles
            if n <= 4
            else [
                0.49,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                5.0,
                10.0,
                40.0,
                100.0,
                1000.0,
                3000.0,
                -2.5,
                -2.49999999999999,
            ]
        )
        check(
            f"polygamma_{n}",
            lambda x: torch.polygamma(n, x),
            lambda x: mp.polygamma(n, x),
            samples,
        )
    for order in (1, 3, 5, 25):
        samples = [rng.uniform(order / 2, order + 15) for _ in range(28)]
        check(
            f"mvlgamma_{order}",
            lambda x: torch.mvlgamma(x, order),
            lambda x: mp.mpf(order * (order - 1)) / 4 * mp.log(mp.pi)
            + mp.fsum(mp.log(mp.gamma(x - mp.mpf(k) / 2)) for k in range(order)),
            samples,
        )
    inv = [rng.uniform(-1, 1) for _ in range(137)]
    inv.extend(
        s * (1.0 - 2.0**-k) for s in (-1, 1) for k in (4, 12, 20, 30, 40, 50, 52, 53)
    )
    check("erfinv", torch.erfinv, mp.erfinv, inv + [1e-90, 1e-15, 0.7, -0.7])
    samples = [rng.uniform(-26, 50) for _ in range(127)] + [
        -26.6,
        -8.0,
        -1.0,
        0.0,
        1.0,
        8.0,
        26.0,
        1e4,
        1e154,
        1e308,
    ]
    check(
        "erfcx",
        torch.special.erfcx,
        lambda x: mp.hyperu(mp.mpf(".5"), mp.mpf(".5"), x * x) / mp.sqrt(mp.pi)
        if x > 30
        else mp.exp(x * x) * mp.erfc(x),
        samples,
    )
    z = [(rng.uniform(1.01, 28), rng.uniform(0.1, 75)) for _ in range(84)]
    z.extend(
        (s, q)
        for s in (2.0, 3.0, 4.0, 7.0, 10.0)
        for q in (-100.5, -100.50000000000001, -2.25, -0.125, 0.125, 1.0)
    )
    z.extend(
        ((1.0 + 2**-52, 11.0), (1.0001, 1e200), (2.0, 1e200), (50.0, 1e9), (150.0, 1.5))
    )

    def hz(s, q):
        if q > 0:
            return mp.zeta(s, q)
        f = mp.ceil(q) - q
        return mp.zeta(s, 1 - f) + (-1) ** int(s) * (mp.zeta(s, f) - mp.zeta(s, 1 - q))

    check("zeta", torch.special.zeta, hz, z)
    result = {
        "torch": torch.__version__,
        "device": torch.backends.mps.get_name(),
        "mpmath": mp.__version__,
        "reference_decimal_precisions": [175, 225],
        "seed": 9237,
        "records": records,
    }
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for k, v in records.items():
        print(
            k,
            v["n"],
            v["gpu"]["nonfinite_mismatches"],
            v["gpu"]["max_abs_over_max_1_abs_ref"],
            flush=True,
        )


if __name__ == "__main__":
    main()
