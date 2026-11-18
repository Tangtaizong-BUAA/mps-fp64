"""Compare two MPS FP64 runtimes, using a wheel or the checkout as candidate."""

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
import tempfile
from time import perf_counter_ns
import warnings
import zipfile

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_map

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

D = torch.float64


def import_baseline(wheel, folder, package_name="baseline_fp64"):
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("mps_fp64/") and ".." not in Path(name).parts:
                archive.extract(name, folder)
    spec = importlib.util.spec_from_file_location(
        package_name, folder / "mps_fp64/__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.enable


def runtime_sha256(folder):
    """Use the same relative path + NUL + bytes digest as the historical report."""
    digest = hashlib.sha256()
    files = (
        path
        for path in (folder / "mps_fp64").rglob("*")
        if path.is_file() and path.suffix in {".py", ".metal"}
    )
    for path in sorted(files):
        digest.update(
            path.relative_to(folder).as_posix().encode() + b"\0" + path.read_bytes()
        )
    return digest.hexdigest()


def tensors(obj):
    return [v for v in tree_flatten(obj)[0] if torch.is_tensor(v)]


def to_host(result):
    return tree_map(lambda x: x.detach().cpu() if torch.is_tensor(x) else x, result)


def ffi_time(fn, batch, sync):
    sync()
    start = perf_counter_ns()
    for _ in range(batch):
        fn()
    sync()
    return (perf_counter_ns() - start) / (batch * 1e6)


def compare(factories, modes, batch, rounds, rng, prepare, tolerate=False):
    records, ready = {}, {}
    for name, factory in factories.items():
        try:
            with modes[name]() as mode:
                fn, meta = prepare(name, factory)
                fn()
                if name != "cpu_fp64":
                    torch.mps.synchronize()
                if mode is not None and mode.cpu_ops:
                    raise AssertionError(f"CPU replay: {mode.cpu_ops}")
            records[name] = {"status": "pass", "samples_ms": [], **meta}
            ready[name] = fn
        except Exception as exc:
            if not tolerate and name in ("new_fp64", "cpu_fp64"):
                raise
            records[name] = {"status": type(exc).__name__, "detail": str(exc)[:280]}
    for _ in range(rounds):
        names = list(ready)
        rng.shuffle(names)
        for name in names:
            sync = (lambda: None) if name == "cpu_fp64" else torch.mps.synchronize
            with modes[name]() as mode:
                elapsed = ffi_time(ready[name], batch, sync)
                if mode is not None and mode.cpu_ops:
                    raise AssertionError(f"CPU replay: {mode.cpu_ops}")
            records[name]["samples_ms"].append(elapsed)
    for row in records.values():
        if row["status"] == "pass":
            row["median_ms"] = median(row["samples_ms"])
    return records


def workloads():
    def case(name, args, op, count):
        return name, args, op, count

    def normal(*shape):
        return torch.randn(shape, dtype=D)

    for n, reps in ((4096, 64), (262144, 32), (8388608, 8)):
        a, b = normal(n), normal(n)
        yield case(f"add_{n}", (a, b), torch.add, reps)
        if n == 8388608:
            yield case(f"mul_{n}", (a, b), torch.mul, reps)
            yield case(f"div_{n}", (a, b), torch.div, reps)
    x = normal(1048576)
    for name in ("sin", "cos", "tanh", "exp", "log"):
        yield case(
            name + "_1048576",
            (x.abs() + 0.125 if name == "log" else x,),
            getattr(torch, name),
            10,
        )
    for n, reps in ((128, 24), (256, 12), (512, 4), (1024, 2)):
        yield case(f"mm_{n}", (normal(n, n), normal(n, n)), torch.mm, reps)
    yield case(
        "bmm_16x64x128x96", (normal(16, 64, 128), normal(16, 128, 96)), torch.bmm, 12
    )
    yield case(
        "mm_strided_256", (normal(256, 256).mT, normal(256, 256).mT), torch.mm, 12
    )
    x = normal(1024, 4096)
    for name in ("sum", "amax", "max"):
        yield case(
            name + "_1024x4096_last",
            (x,),
            lambda a, name=name: getattr(a, name)(-1),
            12,
        )
    yield case("sum_1024x4096_first", (x,), lambda a: a.sum(0), 12)
    yield case("aminmax_1024x4096", (x,), torch.aminmax, 12)
    yield case("aminmax_1024x4096_last", (x,), lambda a: torch.aminmax(a, dim=-1), 12)
    yield case("argmax_262144", (normal(262144),), torch.argmax, 3)
    yield case(
        "stable_sort_256x1024",
        (normal(256, 1024),),
        lambda a: a.sort(dim=-1, stable=True),
        4,
    )
    yield case(
        "max_pool_8x32x56x56",
        (normal(8, 32, 56, 56),),
        lambda a: F.max_pool2d(a, 3, 2, 1),
        12,
    )
    yield case(
        "avg_pool_8x32x56x56",
        (normal(8, 32, 56, 56),),
        lambda a: F.avg_pool2d(a, 3, 2, 1),
        12,
    )
    for n, bs, reps in (
        (32, 256, 16),
        (256, 64, 8),
        (1024, 32, 8),
        (2048, 16, 6),
        (257, 32, 5),
        (1021, 16, 2),
    ):
        yield case(f"rfft_{bs}x{n}", (normal(bs, n),), torch.fft.rfft, reps)
    for k in (1, 3):
        yield case(
            f"conv2d_{k}x{k}",
            (normal(4, 32, 32, 32), normal(32, 32, k, k), normal(32)),
            lambda x, w, b, k=k: F.conv2d(x, w, b, padding=k // 2),
            5,
        )
    a = normal(8, 48, 48)
    pos = a @ a.mT + torch.eye(48, dtype=D)
    yield case("cholesky_8x48", (pos,), torch.linalg.cholesky_ex, 8)
    yield case("solve_8x48", (pos, normal(8, 48, 8)), torch.linalg.solve, 5)
    yield case(
        "batch_norm_eval",
        (
            normal(8, 32, 32, 32),
            normal(32),
            normal(32).abs() + 1,
            normal(32),
            normal(32),
        ),
        lambda a, m, v, w, b: F.batch_norm(a, m, v, w, b, training=False),
        10,
    )
    yield case(
        "cdist_8x96x12",
        (normal(8, 96, 12), normal(8, 96, 12)),
        lambda a, b: torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist"),
        8,
    )
    yield case(
        "scatter_reduce_mean",
        (normal(256, 128), torch.randint(128, (256, 256)), normal(256, 256)),
        lambda a, i, v: a.scatter_reduce(-1, i, v, "mean"),
        8,
    )
    yield case("cummax_8x32767", (normal(8, 32767),), lambda a: a.cummax(-1), 8)
    yield case("bessel_j0_262144", (normal(262144),), torch.special.bessel_j0, 8)
    yield case(
        "interp_bilinear_forward_backward",
        (normal(4, 8, 29, 31), normal(4, 8, 55, 63)),
        lambda a, g: interpolate_grad(a, g),
        6,
    )

    def train(a, w, v):
        w, v = w.detach().requires_grad_(), v.detach().requires_grad_()
        loss = ((a @ w).tanh() @ v).square().mean()
        return (loss, *torch.autograd.grad(loss, (w, v)))

    yield case(
        "mlp_forward_backward",
        (normal(256, 128), normal(128, 256), normal(256, 64)),
        train,
        4,
    )
    matrices = normal(8, 16, 12)
    yield case("svdvals_8x16x12", (matrices,), torch.linalg.svdvals, 4)
    yield case("pinv_8x16x12", (matrices,), torch.linalg.pinv, 4)
    yield case(
        "histc_1048576",
        (normal(1048576),),
        lambda x: torch.histc(x, bins=128, min=-6.0, max=6.0),
        8,
    )
    yield case(
        "igamma_65536",
        (normal(65536).abs() + 0.25, normal(65536).abs()),
        torch.igamma,
        6,
    )


def interpolate_grad(a, g):
    x = a.detach().requires_grad_()
    y = F.interpolate(x, size=g.shape[-2:], mode="bilinear", align_corners=False)
    return y, torch.autograd.grad(y, x, g)[0]


def make_workload(op, values, name):
    target = (
        tree_map(
            lambda x: x.float() if torch.is_tensor(x) and x.dtype == D else x, values
        )
        if name == "native_fp32"
        else values
    )
    expected = to_host(op(*target))
    if name != "cpu_fp64":
        target = tree_map(lambda x: x.to("mps") if torch.is_tensor(x) else x, target)

    def fn():
        return op(*target)

    observed = to_host(fn())
    rtol, atol = (2e-4, 2e-4) if name == "native_fp32" else (1e-10, 1e-11)
    torch.testing.assert_close(observed, expected, rtol=rtol, atol=atol, equal_nan=True)
    return fn, {}


def run_workloads(args, modes, rng):
    results = {}
    for key, values, op, batch in workloads():
        if args.only and key not in args.only:
            continue
        names = ["old_fp64", "new_fp64", "cpu_fp64", "native_fp32"]
        rows = compare(
            dict.fromkeys(names, op),
            modes,
            batch,
            args.rounds,
            rng,
            lambda name, fn: make_workload(fn, values, name),
        )
        results[key] = {
            "batch": batch,
            "shapes": [list(t.shape) for t in tensors(values)],
            "paths": rows,
        }
        print(
            key,
            {
                n: round(v["median_ms"], 5) if "median_ms" in v else v["status"]
                for n, v in rows.items()
            },
            flush=True,
        )
    cold_key = "rfft_16x1021_rebuild_plan"
    if not args.only or cold_key in args.only:
        host = torch.randn(16, 1021, dtype=D)
        ref = torch.fft.rfft(host)

        def prepare_fft(name, unused):
            if name == "new_fp64":
                package = modes[name].__module__.rsplit(".", 1)[0]
                spectral = importlib.import_module(package + ".spectral")
            x = host.to("mps")

            def call():
                if name == "new_fp64":
                    spectral._cached_plan.cache_clear()
                    spectral._cached_roots.cache_clear()
                return torch.fft.rfft(x)

            torch.testing.assert_close(call().cpu(), ref, rtol=1e-10, atol=1e-11)
            return call, {}

        rows = compare(
            dict.fromkeys(("old_fp64", "new_fp64")),
            modes,
            1,
            args.rounds,
            rng,
            prepare_fft,
        )
        results[cold_key] = {
            "batch": 1,
            "shapes": [list(host.shape)],
            "paths": rows,
            "placement": "resident input, shaders warmed; new version discards roots and chirp plan before each call",
        }
    for op in ("rand", "randn"):
        key = op + "_1048576"
        if args.only and key not in args.only:
            continue
        samples = {}
        state = torch.mps.get_rng_state()
        try:

            def prepare_random(name, unused):
                torch.mps.manual_seed(857)

                def call():
                    return getattr(torch, op)((1048576,), device="mps", dtype=D)

                first = call().cpu()
                if name == "old_fp64":
                    samples["ref"] = first
                else:
                    torch.testing.assert_close(
                        first,
                        samples["ref"],
                        rtol=2e-14 if op == "randn" else 0,
                        atol=2e-14 if op == "randn" else 0,
                    )
                return call, {}

            rows = compare(
                dict.fromkeys(("old_fp64", "new_fp64")),
                modes,
                6,
                args.rounds,
                rng,
                prepare_random,
            )
            results[key] = {
                "batch": 6,
                "shapes": [[1048576]],
                "paths": rows,
                "validation": "same seeded MPS generator as old version; RNG state restored after benchmark",
            }
        finally:
            torch.mps.set_rng_state(state)
    if not args.only or "add_cpu_roundtrip_8388608" in args.only:
        # This second traffic model intentionally places both input copies and the return inside the timer.
        a, b = (torch.randn(8388608, dtype=D) for _ in range(2))

        def prepare(name, unused):
            if name == "native_fp32":
                x, y = a.float(), b.float()
            else:
                x, y = a, b
            fn = (
                (lambda: x + y)
                if name == "cpu_fp64"
                else (lambda: (x.to("mps") + y.to("mps")).cpu())
            )
            torch.testing.assert_close(fn(), x + y, rtol=0, atol=0)
            return fn, {}

        rows = compare(dict.fromkeys(modes), modes, 4, args.rounds, rng, prepare)
        results["add_cpu_roundtrip_8388608"] = {
            "batch": 4,
            "shapes": [[8388608], [8388608]],
            "paths": rows,
            "placement": "two CPU inputs, output returned to CPU on every call",
        }
    return results


STOCHASTIC = {
    "rand",
    "randn",
    "normal",
    "uniform",
    "randn_like",
    "rand_like",
    "randint",
    "randint_like",
    "bernoulli",
    "multinomial",
    "normal_",
    "randperm",
    "random",
    "dropout",
    "nn.functional.dropout",
}


def representative(op):
    if op.name in STOCHASTIC:
        raise ValueError("stochastic input or output")
    if D not in op.supported_dtypes("cpu"):
        raise ValueError("no CPU float64 sample")
    torch.manual_seed(1024)
    samples = itertools.islice(op.sample_inputs("cpu", D, requires_grad=False), 4)
    eligible = []
    for index, sample in enumerate(samples):
        args = (sample.input, sample.args, sample.kwargs)
        ts = tensors(args)
        if (ts or args[2].get("dtype") == D) and all(
            t.layout == torch.strided and not t.is_complex() and t.numel() <= 2**20
            for t in ts
        ):
            eligible.append((sum(t.numel() for t in ts), index, sample))
    if not eligible:
        raise ValueError("no eligible dense real input among the first four samples")
    size, index, sample = max(eligible, key=lambda entry: entry[0])
    args = (sample.input, sample.args, sample.kwargs)
    ts = tensors(args)
    before = [v._version for v in ts]
    ref = op(sample.input, *sample.args, **sample.kwargs)
    if [v._version for v in ts] != before:
        raise ValueError("mutates input or running state")
    if not any(t.dtype == D for t in ts) and args[2].get("dtype") != D:
        raise ValueError("sample has no float64 input or explicit float64 output dtype")
    return (
        args,
        ref,
        {
            "sample_index": index,
            "input_elements": size,
            "shapes": [list(t.shape) for t in ts],
        },
    )


def catalog_fn(name, op, args, reference):
    def move(x):
        if not torch.is_tensor(x):
            return x
        if op.name == "as_strided" and x is args[0]:
            storage = x.untyped_storage().nbytes() // x.element_size()
            return (
                x.as_strided((storage,), (1,), 0)
                .to("mps")
                .as_strided(x.shape, x.stride(), x.storage_offset())
            )
        return x.to("mps")

    x, more, kw = tree_map(move, args)
    from opinfo_audit import target_kwargs

    kw = target_kwargs(op.name, kw)
    if op.name == "tensor_split" and more and torch.is_tensor(more[0]):
        more = (args[1][0], *more[1:])

    def fn():
        return op(x, *more, **kw)

    result = fn()
    ts = tensors(result)
    if ts and any(t.device.type != "mps" for t in ts):
        raise ValueError("output includes a CPU tensor")
    if op.name in {
        "empty_like",
        "new_empty",
        "new_empty_strided",
        "empty",
        "empty_strided",
        "empty_permuted",
    }:
        if (result.shape, result.dtype, result.stride()) != (
            reference.shape,
            reference.dtype,
            reference.stride(),
        ):
            raise AssertionError("uninitialized output metadata differs")
    else:
        torch.testing.assert_close(
            to_host(result), reference, rtol=1e-10, atol=1e-12, equal_nan=True
        )
    pointers = {t.untyped_storage().data_ptr() for t in tensors((x, more, kw))}
    category = (
        "host_scalar"
        if not ts
        else "empty"
        if not any(t.numel() for t in ts)
        else "alias"
        if all(t.untyped_storage().data_ptr() in pointers for t in ts)
        else "materialized"
    )
    return fn, {"category": category}


def run_catalog(args, modes, rng):
    from torch.testing._internal.common_methods_invocations import op_db

    rows = []
    for op in op_db:
        if args.only and op.name not in args.only:
            continue
        row = {"op": op.name, "variant": op.variant_test_name}
        try:
            values, ref, meta = representative(op)
            row.update(meta)
            row["paths"] = compare(
                dict.fromkeys(("old_fp64", "new_fp64"), op),
                modes,
                args.batch,
                args.rounds,
                rng,
                lambda name, fn: catalog_fn(name, fn, values, ref),
                True,
            )
        except Exception as exc:
            row["exclusion"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rows.append(row)
    print(
        "catalog entries",
        len(rows),
        Counter(
            "excluded" if "exclusion" in r else r["paths"]["new_fp64"]["status"]
            for r in rows
        ),
    )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wheel", type=Path, help="baseline wheel for the old_fp64 path")
    p.add_argument("report", type=Path)
    p.add_argument(
        "--candidate-wheel",
        type=Path,
        help="compare this wheel as new_fp64; defaults to the source checkout",
    )
    p.add_argument("--suite", choices=("workloads", "catalog"), default="workloads")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument(
        "--batch", type=int, default=12, help="calls per catalog timing batch"
    )
    p.add_argument("--only", nargs="+")
    args = p.parse_args()
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        p.error("set PYTORCH_ENABLE_MPS_FALLBACK=0 to measure strict GPU paths")
    warnings.filterwarnings("ignore")
    torch.manual_seed(583)
    rng = random.Random(583)
    with tempfile.TemporaryDirectory() as folder:
        baseline_root = Path(folder) / "baseline"
        old = import_baseline(args.wheel, baseline_root)
        baseline_runtime = runtime_sha256(baseline_root)
        if args.candidate_wheel:
            candidate_root = Path(folder) / "candidate"
            candidate = import_baseline(
                args.candidate_wheel, candidate_root, "candidate_fp64"
            )
        else:
            from mps_fp64 import enable as candidate

            candidate_root = ROOT
        candidate_runtime = runtime_sha256(candidate_root)
        modes = {
            "old_fp64": old,
            "new_fp64": candidate,
            "cpu_fp64": nullcontext,
            "native_fp32": nullcontext,
        }
        cases = (run_catalog if args.suite == "catalog" else run_workloads)(
            args, modes, rng
        )
    report = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "device": torch.backends.mps.get_name(),
        "baseline_wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "baseline_wheel": args.wheel.name,
        "baseline_runtime_sha256": baseline_runtime,
        "candidate_wheel": args.candidate_wheel.name if args.candidate_wheel else None,
        "candidate_wheel_sha256": (
            hashlib.sha256(args.candidate_wheel.read_bytes()).hexdigest()
            if args.candidate_wheel
            else None
        ),
        "candidate_runtime_sha256": candidate_runtime,
        "source_sha256": candidate_runtime,
        "suite": args.suite,
        "rounds": args.rounds,
        "catalog_batch": args.batch,
        "method": "resident inputs; full-output reference check; warmed batched wall time ending with MPS synchronize; seeded shuffled path order each round; built-in fallback and package CPU replay off",
        "cases": cases,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
