"""Compare public FP64 matrix APIs against the old wheel and same-machine CPU BLAS."""

import argparse
from contextlib import nullcontext
import ctypes as ct
import ctypes.util
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from statistics import median, quantiles
import tempfile
from time import perf_counter_ns

import torch
from compare_performance import import_baseline
from mps_fp64 import enable

ROOT = Path(__file__).resolve().parents[1]
FLOAT64, INT64 = torch.float64, torch.int64


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    k: int
    n: int
    batch: int = 1
    amplitude: int = 0
    transposed: bool = False
    vector: bool = False
    throughput: int = 1
    transfers: bool = False


def cases():
    for n in (1024, 2048, 4096):
        yield Case(f"normal_square_{n}", n, n, n, transfers=(n == 2048))
    for bits, lim in [(4, 15), (12, 2047), (16, 32767)]:
        for n in (1024, 2048, 4096):
            yield Case(
                f"dyadic_{bits}_{n}", n, n, n, amplitude=lim, transfers=(n == 2048)
            )
    for n in (4096, 8192):
        for k in (1, 4, 16, 32) if n == 4096 else (1, 4):
            yield Case(f"rank_{k}_{n}", n, k, n, transfers=(n == 4096 and k == 4))
    for m, k, n in [
        (65536, 256, 1),
        (32768, 1024, 2),
        (65536, 256, 4),
        (8192, 4096, 1),
    ]:
        yield Case(f"narrow_{m}_{k}_{n}", m, k, n, transfers=n == 2)
    yield Case("narrow_transposed", 32768, 1024, 2, transposed=True)
    for col in (False, True):
        yield Case(
            "mv_col" if col else "mv_row", 65536, 256, 1, transposed=col, vector=True
        )
    for b in (4096, 65536):
        for s in (3, 4):
            yield Case(
                f"batch_{b}_{s}", s, s, s, batch=b, throughput=12 if b == 4096 else 5
            )


def inputs(c, seed):
    g = torch.Generator().manual_seed(seed)
    batches = () if c.batch == 1 else (c.batch,)

    def draw(shape, scale):
        if c.amplitude:
            return (
                torch.randint(
                    -c.amplitude, c.amplitude + 1, shape, dtype=INT64, generator=g
                )
                .to(FLOAT64)
                .mul_(2.0**scale)
            )
        return torch.randn(shape, dtype=FLOAT64, generator=g)

    a = draw((*batches, c.k, c.m) if c.transposed else (*batches, c.m, c.k), -13)
    b = draw((c.k,) if c.vector else (*batches, c.k, c.n), -9)
    return (a.mT if c.transposed else a), b


def direct_blas(c, a, b, out):
    if c.batch != 1:
        return {}
    lib = ct.CDLL(ctypes.util.find_library("Accelerate"))
    integer, scalar, ptr = ct.c_int, ct.c_double, ct.c_void_p
    mm = lib.cblas_dgemm
    mm.argtypes = [integer] * 6 + [
        scalar,
        ptr,
        integer,
        ptr,
        integer,
        scalar,
        ptr,
        integer,
    ]
    mm.restype = None
    mv = lib.cblas_dgemv
    mv.argtypes = [integer] * 4 + [
        scalar,
        ptr,
        integer,
        ptr,
        integer,
        scalar,
        ptr,
        integer,
    ]
    mv.restype = None
    # Row-major CBLAS interprets a column-major view as its physical transpose.
    trans = 112 if c.transposed else 111
    lda = c.m if c.transposed else c.k
    ap, bp, op = (t.data_ptr() for t in (a, b, out))
    funcs = {}
    if not c.vector:
        funcs["accelerate_dgemm"] = lambda: mm(
            101, trans, 111, c.m, c.n, c.k, 1.0, ap, lda, bp, c.n, 0.0, op, c.n
        )
    if c.n == 1:
        rows, cols = (c.k, c.m) if c.transposed else (c.m, c.k)
        funcs["accelerate_dgemv"] = lambda: mv(
            101, trans, rows, cols, 1.0, ap, lda, bp, 1, 0.0, op, 1
        )
    if c.k == 1:
        funcs["cpu_outer_out"] = lambda: torch.mul(a, b, out=out)
    return funcs


def check_case(c, expected, baseline, actual):
    legacy = baseline.reshape(-1).view(INT64)
    packed = actual.reshape(-1).view(INT64)
    target = expected.reshape(-1)
    differences = unrepresentable = 0
    max_abs = ref_max = 0.0
    for i in range(0, packed.numel(), 1 << 20):
        old, v, e = (t[i : i + (1 << 20)] for t in (legacy, packed, target))
        differences += torch.count_nonzero(v != old).item()
        max_abs = max(max_abs, float((v.view(FLOAT64) - e).abs().max()))
        ref_max = max(ref_max, float(e.abs().max()))
        if c.amplitude:
            if not torch.equal(e.view(INT64), v):
                raise AssertionError(
                    f"integer-lattice result differs from CPU in {c.name}"
                )
            unrepresentable += torch.count_nonzero(e != e.float().double()).item()
    assert differences == 0, (c.name, differences)
    assert max_abs <= max(1.0, ref_max) * c.k * torch.finfo(FLOAT64).eps * 8, (
        c.name,
        max_abs,
    )
    return {
        "checked_output_words": packed.numel(),
        "differences_from_020": differences,
        "cpu_max_abs_error": max_abs,
        "cpu_max_abs_output": ref_max,
        "cpu_outputs_unrepresentable_in_fp32": unrepresentable if c.amplitude else None,
    }


def ordered_oracle(c, a, b, result):
    if c.vector:
        return {}  # Its public reduction order is independently compared to the 0.2.0 tree.
    native = ct.CDLL(None).fma
    native.argtypes = [ct.c_double] * 3
    native.restype = ct.c_double
    g = random.Random(4301)
    f64 = []
    expected = []
    for _ in range(49):
        r, t, u = g.randrange(c.m), g.randrange(c.n), g.randrange(c.batch)
        x, y = (a, b) if c.batch == 1 else (a[u], b[u])
        z = 0.0
        for lhs, rhs in zip(x[r, :].tolist(), y[:, t].tolist()):
            z = native(lhs, rhs, z)
        f64.append(z)
        expected.append((result if c.batch == 1 else result[u])[r, t].item())
    diff = (
        (
            torch.tensor(f64, dtype=FLOAT64).view(INT64)
            != torch.tensor(expected, dtype=FLOAT64).view(INT64)
        )
        .sum()
        .item()
    )
    assert not diff, (c.name, diff)
    return {
        "sampled_left_to_right_hardware_fma_dots": len(f64),
        "sampled_fma_bit_errors": diff,
    }


def interleave(paths, rounds, count, seed):
    elapsed = {key: [] for key in paths}
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(paths)
        rng.shuffle(order)
        for key in order:
            mode, fn, gpu = paths[key]
            with mode() as backend:
                if gpu:
                    torch.mps.synchronize()
                start = perf_counter_ns()
                for _ in range(count):
                    fn()
                if gpu:
                    torch.mps.synchronize()
                elapsed[key].append((perf_counter_ns() - start) / (count * 1e6))
                if backend is not None and backend.cpu_ops:
                    raise AssertionError(backend.cpu_ops)
    result = {}
    for key, values in elapsed.items():
        q = (
            quantiles(values, n=4, method="inclusive")
            if len(values) > 1
            else values * 3
        )
        result[key] = {
            "median_ms": median(values),
            "q1_ms": q[0],
            "q3_ms": q[2],
            "samples_ms": values,
        }
    return result


def run_case(c, old, rounds, seed):
    a, b = inputs(c, seed)
    op = torch.mv if c.vector else torch.mm if c.batch == 1 else torch.bmm
    cpu_ref = op(a, b)
    cpu_out = torch.empty_like(cpu_ref)
    with enable():
        x, y = a.to("mps"), b.to("mps")
        out = torch.empty(cpu_ref.shape, dtype=FLOAT64, device="mps")
        got = op(x, y).cpu()
        want_out = not c.vector
        if want_out:
            assert op(x, y, out=out) is out
            assert out.cpu().view(INT64).equal(got.view(INT64))
    with old() as scope:
        previous = op(x, y).cpu()
        assert not scope.cpu_ops
    checks = check_case(c, cpu_ref, previous, got)
    checks.update(ordered_oracle(c, a, b, got))
    del got, previous
    paths = {
        "cpu": (nullcontext, lambda: op(a, b), False),
        "cpu_out": (nullcontext, lambda: op(a, b, out=cpu_out), False),
        "mps_020": (old, lambda: op(x, y), True),
        "mps_030": (enable, lambda: op(x, y), True),
    }
    if want_out:
        paths["mps_030_out"] = (enable, lambda: op(x, y, out=out), True)
    blas = direct_blas(c, a, b, cpu_out)
    for key, fn in blas.items():
        fn()
        torch.testing.assert_close(cpu_out, cpu_ref, atol=5e-11, rtol=1e-12)
        paths[key] = (nullcontext, fn, False)
    if c.transfers:

        def roundtrip():
            return op(a.to("mps"), b.to("mps")).cpu()

        paths["mps_030_roundtrip"] = (enable, roundtrip, True)
    for mode, fn, gpu in paths.values():
        with mode() as scope:
            fn()
            if gpu:
                torch.mps.synchronize()
            if scope is not None and scope.cpu_ops:
                raise AssertionError(scope.cpu_ops)
    timed = interleave(paths, rounds, 1, seed)
    queued = (
        interleave(paths, rounds, c.throughput, seed + 206)
        if c.throughput > 1
        else None
    )
    cpu_names = [n for n in paths if n.startswith(("cpu", "accelerate"))]
    best = min(cpu_names, key=lambda n: timed[n]["median_ms"])
    main = "mps_030_out" if want_out else "mps_030"
    row = {
        "case": c.__dict__,
        "input": "independent randn float64"
        if not c.amplitude
        else "independent signed uniform integer magnitudes, A scaled by 2^-13 and B by 2^-9",
        "input_shapes": [list(a.shape), list(b.shape)],
        "input_strides": [list(a.stride()), list(b.stride())],
        "seed": seed,
        "verification": checks,
        "latency": timed,
        "queued": queued,
        "best_measured_cpu": best,
        "cpu_over_gpu": timed[best]["median_ms"] / timed[main]["median_ms"],
        "default_020_over_030": timed["mps_020"]["median_ms"]
        / timed["mps_030"]["median_ms"],
    }
    if queued:
        cpu = min(queued[n]["median_ms"] for n in cpu_names)
        row["queued_cpu_over_gpu"] = cpu / queued[main]["median_ms"]
    print(
        c.name,
        main,
        round(timed[main]["median_ms"], 4),
        best,
        round(timed[best]["median_ms"], 4),
        round(row["cpu_over_gpu"], 3),
        "x",
        flush=True,
    )
    return row


def main():
    pars = argparse.ArgumentParser(description=__doc__)
    pars.add_argument("baseline_wheel", type=Path)
    pars.add_argument("report", type=Path)
    pars.add_argument("--rounds", type=int, default=9)
    pars.add_argument("--seed", type=int, default=1024)
    pars.add_argument("--only", nargs="+")
    args = pars.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python")
    source = hashlib.sha256()
    for f in sorted(
        q for q in (ROOT / "mps_fp64").rglob("*") if q.suffix in (".py", ".metal")
    ):
        source.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes())
    report = {
        "device": torch.backends.mps.get_name(),
        "macOS": platform.mac_ver()[0],
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cpu_blas": "Apple Accelerate",
        "rounds": args.rounds,
        "old_wheel_sha256": hashlib.sha256(
            args.baseline_wheel.read_bytes()
        ).hexdigest(),
        "source_sha256": source.hexdigest(),
        "blas_build": torch.__config__.show(),
        "method": "Inputs resident except roundtrip. Each path warmed, shuffled per round. Synchronized single-call medians; batched 3x3/4x4 also timed for repeated submissions. cpu_over_gpu uses fastest measured CPU path versus GPU out= (GPU functional for mv). CPU BLAS calls reuse result; GPU qualification, packing, reconstruction, and sync included. Full outputs compared to CPU and historical GPU, plus sampled hardware FMA where applicable.",
        "cases": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        old = import_baseline(args.baseline_wheel, Path(tmp))
        for i, c in enumerate(cases()):
            if args.only and c.name not in args.only:
                continue
            report["cases"].append(run_case(c, old, args.rounds, args.seed + i))
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
