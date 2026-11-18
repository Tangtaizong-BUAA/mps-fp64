"""Check both incomplete-gamma tails against arbitrary-precision integrals."""

import argparse
from functools import lru_cache
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


@lru_cache
def nodes(dps, degree):
    with mp.workdps(dps):
        x, w = mp.gauss_quadrature(degree, "legendre")
        return list(zip(x, w))


def log1pmx(r):
    if abs(r) > mp.mpf("0.05"):
        return mp.log1p(r) - r
    # Direct evaluation of log(1+r)-r would lose up to 154 decimal places.
    power = -(r * r) / 2
    total = power
    for k in range(3, 150):
        power *= -r * (k - 1) / k
        total += power
        if abs(power) <= mp.eps * abs(total):
            break
    return total


def reference(shape, limit, dps, degree):
    a, x = mp.mpf(shape), mp.mpf(limit)
    with mp.workdps(dps + max(0, int(math.log10(shape))) + 25):
        deviance = (x - a) - a * (mp.log(x) - mp.log(a))
        if deviance > 746:
            return (0.0, 1.0) if x < a else (1.0, 0.0)
        if a > 3000:
            scale = mp.sqrt(a) / (1 + abs(x - a) / mp.sqrt(a))
            log_density_scale = mp.log(scale) + (a - 1) * mp.log(x) - x - mp.loggamma(a)
    with mp.workdps(dps):
        if a <= 3000:
            return (
                float(mp.gammainc(a, 0, x, regularized=True)),
                float(mp.gammainc(a, x, mp.inf, regularized=True)),
            )
        # Integrate the density outward from x in units of its local tail width.
        # Scaling by density(x) keeps both an exponentially small integral and
        # a huge shape away from the numerical integration's absolute floor.
        sign = 1 if x >= a else -1
        rule = nodes(dps, degree)

        def density(u):
            r = sign * (scale / x) * u
            if r <= -1:
                return mp.mpf(0)
            return mp.exp(-(x - a + 1) * r + (a - 1) * log1pmx(r))

        edges = (0, 1, 3, 7, 15, 28, 48, 80, 128, 192)
        panels = []
        for lo, hi in zip(edges, edges[1:]):
            mid, half = mp.mpf(hi + lo) / 2, mp.mpf(hi - lo) / 2
            panels.append(half * mp.fsum(w * density(mid + half * t) for t, w in rule))
        small = mp.exp(log_density_scale) * mp.fsum(panels)
        return tuple(map(float, (1 - small, small) if sign > 0 else (small, 1 - small)))


def probes():
    g = random.Random(1182)
    cases = [(g.uniform(0.002, 25), g.uniform(0.0001, 45)) for _ in range(55)]
    cases += [(10 ** g.uniform(-305, 2), 10 ** g.uniform(-305, 3)) for _ in range(32)]
    cases += [
        (64.0, 64.0),
        (1e-300, 0.99),
        (0.5057795486219497, 0.5247630553490222),
        (68762778.13259603, 68722229.95106885),
        (2.7250777396966336e33, 2.725077739696633e33),
        (15.351824066977391, 15.671641280017298),
        (2.0, 745.0),
        (2.0, 750.0),
    ]
    for a in (
        math.ulp(0.0),
        1e-320,
        1e-300,
        0.125,
        0.5,
        1.0,
        2.0,
        7.9999,
        8.0,
        15.9999,
        16.0,
        63.9999,
        64.0,
        200.0,
        8192.0,
    ):
        cases += [(a, x) for x in (math.ulp(0.0), 0.75, 1.01, 10.0, 709.0, 738.0)]
    for a in (64.0, 128.0, 2300.0, 1e5, 1e8, 1e12, 1e20, 1e32):
        cases += [
            (a, x)
            for z in (-38.0, -9.0, -4.0, 0.0, 1.0, 5.0, 16.0, 38.0)
            if (x := a + z * math.sqrt(a)) > 0
        ]
    for a in (1e100, 1e308, sys.float_info.max):
        cases += [
            (a, x)
            for x in (math.nextafter(a, 0.0), a, math.nextafter(a, math.inf))
            if math.isfinite(x)
        ]
    return list(dict.fromkeys(cases))


def metrics(values, refs, cases):
    def bits(x):
        return int.from_bytes(struct.pack(">d", x), "big")

    def max_error(fn, selector=lambda r: True):
        idx = max(
            (i for i, r in enumerate(refs) if selector(r)),
            key=lambda j: fn(values[j], refs[j]),
        )
        return {
            "value": fn(values[idx], refs[idx]),
            "input": cases[idx],
            "result": values[idx],
            "reference": refs[idx],
        }

    return {
        "max_abs": max_error(lambda v, r: abs(v - r)),
        "max_rel_normal": max_error(
            lambda v, r: abs(v - r) / r, lambda r: r >= sys.float_info.min
        ),
        "max_ulp": max_error(lambda v, r: abs(bits(v) - bits(r))),
        "range_violations": sum(not (0 <= v <= 1) for v in values),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    args = p.parse_args()
    cases = probes()
    refs1 = [reference(a, x, 85, 24) for a, x in cases]
    refs = [reference(a, x, 135, 36) for a, x in cases]
    if refs != refs1:
        raise ArithmeticError(
            [(cases[i], r, s) for i, (r, s) in enumerate(zip(refs, refs1)) if r != s][
                :6
            ]
        )
    ac, xc = map(lambda v: torch.tensor(v, dtype=torch.float64), zip(*cases))
    output = {
        "torch": torch.__version__,
        "mpmath": mp.__version__,
        "device": torch.backends.mps.get_name(),
        "reference_decimal_digits": [85, 135],
        "quadrature_degrees": [24, 36],
        "input_pairs": len(cases),
        "tails_checked": 2 * len(cases),
        "operators": {},
    }
    with enable() as mode:
        a, x = ac.to("mps"), xc.to("mps")
        for k, op in enumerate((torch.igamma, torch.igammac)):
            gv, cv = op(a, x).cpu().tolist(), op(ac, xc).tolist()
            rr = [v[k] for v in refs]
            if not all(math.isfinite(value) for value in gv):
                raise AssertionError(
                    "GPU produced a nonfinite probability for valid inputs"
                )
            output["operators"][op.__name__] = {
                "gpu": metrics(gv, rr, cases),
                "cpu": metrics(cv, rr, cases),
            }
        assert not mode.cpu_ops
    # Explicitly retain cases for which the installed CPU approximation can
    # lose a small tail (or the relative position of x at very large a).
    show = [
        (64.0, 64.0),
        (1e-300, 0.99),
        (68762778.13259603, 68722229.95106885),
        (2.7250777396966336e33, 2.725077739696633e33),
    ]
    output["examples"] = []
    with enable():
        for a, x in show:
            j = cases.index((a, x))
            s, t = ac[j : j + 1], xc[j : j + 1]
            output["examples"].append(
                {
                    "input": [a, x],
                    "reference": refs[j],
                    "gpu": [
                        f(s.to("mps"), t.to("mps")).cpu().item()
                        for f in (torch.igamma, torch.igammac)
                    ],
                    "cpu": [f(s, t).item() for f in (torch.igamma, torch.igammac)],
                }
            )
    args.report.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["operators"], indent=2), flush=True)


if __name__ == "__main__":
    main()
