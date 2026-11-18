"""Compare public PyTorch CPU and emulated-MPS FP64 tensor operations.

From the repository root, run
    python tools/benchmark_tensor_ops.py --output results/public-tensor-ops.json
"""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter_ns

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
    raise RuntimeError("This benchmark requires PYTORCH_ENABLE_MPS_FALLBACK=0")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_fp64 import enable


@dataclass(frozen=True)
class Case:
    id: str
    operator: str
    shape: tuple
    kwargs: dict = field(default_factory=dict)
    transpose: bool = False


STREAM = (4096, 2048)
MEDIUM = (2048, 512)
CASES = (
    Case("add_64k", "add", (256, 256)),
    Case("add_8m", "add", STREAM),
    Case("add_transposed_1m", "add", MEDIUM, transpose=True),
    Case("mul_8m", "mul", STREAM),
    Case("div_8m", "div", STREAM),
    Case("abs_8m", "abs", STREAM),
    Case("neg_8m", "neg", STREAM),
    Case("lt_8m", "lt", STREAM),
    Case("relu_8m", "relu", STREAM),
    Case("where_8m", "where", STREAM),
    Case("exp_1m", "exp", MEDIUM),
    Case("sin_1m", "sin", MEDIUM),
    Case("sum_all_8m", "sum", STREAM, {"dim": (0, 1)}),
    Case("sum_rows_8m", "sum", STREAM, {"dim": 1}),
    Case("sum_columns_8m", "sum", STREAM, {"dim": 0}),
    Case("mean_rows_8m", "mean", STREAM, {"dim": 1}),
    Case("amax_rows_8m", "amax", STREAM, {"dim": 1}),
    Case("argmax_all_8m", "argmax", STREAM),
    Case("argmax_rows_8m", "argmax", STREAM, {"dim": 1}),
    Case("var_rows_8m", "var", STREAM, {"dim": 1, "correction": 0}),
    Case("cumsum_rows_2m", "cumsum", (4096, 512), {"dim": 1}),
    Case("sort_rows_1m", "sort", MEDIUM, {"dim": 1, "stable": True}),
    Case("topk_rows_1m", "topk", MEDIUM, {"dim": 1, "k": 32}),
    Case("softmax_rows_1m", "softmax", MEDIUM, {"dim": 1}),
)
BINARY = {"add", "mul", "div", "lt"}
EPS = np.finfo(np.float64).eps


def sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_digest():
    digest = hashlib.sha256()
    paths = sorted(
        p for p in (ROOT / "mps_fp64").rglob("*") if p.suffix in {".py", ".metal"}
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def tensors(output):
    return tuple(output) if isinstance(output, tuple) else (output,)


def make_inputs(case, seed):
    generator = torch.Generator().manual_seed(seed)

    def normal():
        shape = case.shape[::-1] if case.transpose else case.shape
        x = torch.randn(shape, dtype=torch.float64, generator=generator)
        return x.t() if case.transpose else x

    x = normal()
    if case.operator == "where":
        y = normal()
        condition = torch.randint(2, case.shape, dtype=torch.bool, generator=generator)
        return condition, x, y
    return (x, normal()) if case.operator in BINARY else (x,)


def copy_device(x):
    # Moving bits preserves double values and transposed input strides.
    return (
        x.view(torch.int64).to("mps").view(torch.float64)
        if x.dtype == torch.float64
        else x.to("mps")
    )


def input_description(x):
    prefix = x.ravel()[:1024].numpy()
    result = {
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "sample_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
    }
    if x.dtype == torch.float64:
        result["low_29_mantissa_bits_nonzero_in_sample"] = int(
            np.count_nonzero(prefix.view(np.uint64) & ((1 << 29) - 1))
        )
        result["sample_size"] = len(prefix)
    return result


def make_output(reference, gpu):
    buffers = []
    for x in tensors(reference):
        if x.dtype == torch.bool:
            buf = ~x
        elif x.dtype == torch.float64:
            buf = torch.full_like(x, math.nan)
        else:
            buf = torch.full_like(x, torch.iinfo(x.dtype).min)
        buffers.append(copy_device(buf) if gpu else buf)
    return tuple(buffers) if isinstance(reference, tuple) else buffers[0]


def reduction_scale(case, x):
    if case.operator not in {"sum", "mean", "cumsum"}:
        return None
    if case.operator == "cumsum":
        scale = torch.cumsum(x.abs(), dim=case.kwargs["dim"])
    else:
        dim = case.kwargs["dim"]
        scale = (
            x.abs().sum(dim=dim) if case.operator == "sum" else x.abs().mean(dim=dim)
        )
    return scale.numpy().ravel()


def ordered_bits(x):
    u = x.view(np.uint64)
    sign = np.uint64(1 << 63)
    return np.where((u & sign) != 0, ~u, u | sign)


def check(case, result, reference, scale):
    actual, expected = tensors(result), tensors(reference)
    if len(actual) != len(expected):
        raise AssertionError("different number of output tensors")
    mode = (
        "4 ULP"
        if case.operator in {"sin", "exp"}
        else "32eps times input absolute sum/mean/prefix sum"
        if scale is not None
        else "32eps relative"
        if case.operator in {"softmax", "var"}
        else "bitwise"
    )
    checked = {
        "criterion": mode,
        "elements": 0,
        "max_abs_error": 0.0,
        "bit_mismatches": 0,
        "max_ulp": 0,
        "max_fraction_of_error_bound": 0.0,
    }
    for y, x in zip(actual, expected):
        if (x.shape, x.dtype) != (y.shape, y.dtype):
            raise AssertionError(
                f"output mismatch: {x.shape}, {x.dtype} versus {y.shape}, {y.dtype}"
            )
        left, right = y.detach().cpu().numpy().ravel(), x.numpy().ravel()
        checked["elements"] += len(left)
        for i in range(0, len(left), 65536):
            a, b = left[i : i + 65536], right[i : i + 65536]
            if b.dtype != np.float64:
                mismatches = int(np.count_nonzero(a != b))
                checked["bit_mismatches"] += mismatches
                if mismatches:
                    raise AssertionError("unequal index or boolean output")
                continue
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise AssertionError("nonfinite result for finite benchmark data")
            mismatches = int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
            checked["bit_mismatches"] += mismatches
            if not mismatches:
                continue
            abs_error = np.abs(a - b)
            ua, ub = ordered_bits(a), ordered_bits(b)
            ulps = np.maximum(ua, ub) - np.minimum(ua, ub)
            checked["max_abs_error"] = max(
                checked["max_abs_error"], float(abs_error.max())
            )
            checked["max_ulp"] = max(checked["max_ulp"], int(ulps.max()))
            if mode == "bitwise":
                raise AssertionError(
                    f"bitwise comparison failed; example {a[np.argmax(abs_error)]} vs {b[np.argmax(abs_error)]}"
                )
            if mode == "4 ULP":
                fraction = float(ulps.max()) / 4
            else:
                bound = scale[i : i + len(a)] if scale is not None else np.abs(b)
                fraction = float(
                    np.max(
                        abs_error / np.maximum(32 * EPS * bound, np.finfo(float).tiny)
                    )
                )
            checked["max_fraction_of_error_bound"] = max(
                checked["max_fraction_of_error_bound"], fraction
            )
            if fraction > 1:
                raise AssertionError(f"tolerance exceeded by factor {fraction}")
    return checked


def fsum_check(case, x, cpu_output, gpu_output, seed):
    # Reference samples use a high-accuracy expansion sum independently of
    # both PyTorch's CPU and the GPU reduction order.
    if case.operator not in {"sum", "mean", "cumsum"}:
        return None
    inputs = x.numpy()
    cpu, gpu = cpu_output.numpy().ravel(), gpu_output.cpu().numpy().ravel()
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(cpu)), min(16, len(cpu))))
    errs = {"cpu": [], "mps": []}
    for index in chosen:
        if case.operator == "cumsum":
            row, col = divmod(index, x.shape[1])
            values = inputs[row, : col + 1]
        elif isinstance(case.kwargs["dim"], tuple):
            values = inputs.ravel()
        else:
            values = inputs[index] if case.kwargs["dim"] == 1 else inputs[:, index]
        ref = math.fsum(values)
        if case.operator == "mean":
            ref /= len(values)
        for key, y in (("cpu", cpu), ("mps", gpu)):
            errs[key].append(abs(float(y[index]) - ref))
    return {
        "sample_count": len(chosen),
        "max_abs_error": {k: max(v) for k, v in errs.items()},
    }


def error_text(exc):
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:220]}"


def run_case(case, seed, rounds, warmups):
    inputs = make_inputs(case, seed)
    fn = getattr(torch, case.operator)
    reference = fn(*inputs, **case.kwargs)
    scale = reduction_scale(case, inputs[0])
    row = {
        "id": case.id,
        "operator": f"torch.{case.operator}",
        "shape": list(case.shape),
        "input_seed": seed,
        "kwargs": case.kwargs,
        "inputs": [input_description(x) for x in inputs],
        "out_unavailable": {},
        "validation": {},
        "methods": {},
    }
    with enable(cpu_fallback=False) as backend:
        device_inputs = tuple(copy_device(x) for x in inputs)
        if any(x.stride() != y.stride() for x, y in zip(inputs, device_inputs)):
            raise AssertionError("CPU and MPS input strides differ")
        try:
            device_result = fn(*device_inputs, **case.kwargs)
        except NotImplementedError as exc:
            row.update(status="unsupported", detail=error_text(exc))
            return row
        if any(x.device.type != "mps" for x in tensors(device_result)):
            raise AssertionError("public result left the GPU")
        row["validation"]["mps_functional"] = check(
            case, device_result, reference, scale
        )
        independent = fsum_check(case, inputs[0], reference, device_result, seed)
        if independent is not None:
            row["validation"]["independent_math_fsum"] = independent
        if backend.cpu_ops:
            raise AssertionError(f"CPU replay during validation: {backend.cpu_ops}")
    del device_result

    outputs = {}
    variants = {"cpu_functional": (inputs, {}), "mps_functional": (device_inputs, {})}
    for platform_name, args in (("cpu", inputs), ("mps", device_inputs)):
        gpu = platform_name == "mps"
        with enable(cpu_fallback=False) if gpu else nullcontext() as backend:
            buf = make_output(reference, gpu)
            try:
                z = fn(*args, **case.kwargs, out=buf)
            except (NotImplementedError, TypeError) as exc:
                row["out_unavailable"][platform_name] = error_text(exc)
                continue
            if len(tensors(z)) != len(tensors(buf)) or any(
                a is not b for a, b in zip(tensors(z), tensors(buf))
            ):
                raise AssertionError(
                    f"{platform_name} out= did not return its supplied storage"
                )
            stats = check(case, buf, reference, scale)
            if gpu and backend.cpu_ops:
                raise AssertionError("CPU replay in out= dispatch")
            row["validation"][platform_name + "_out"] = stats
            outputs[platform_name] = buf
            variants[platform_name + "_out"] = (args, {"out": buf})
    del reference, scale
    samples = {key: [] for key in variants}
    order = random.Random(seed ^ 0x51F61C)
    for r in range(warmups + rounds):
        names = list(variants)
        order.shuffle(names)
        for name in names:
            args, extra = variants[name]
            gpu = name.startswith("mps")
            with enable(cpu_fallback=False) if gpu else nullcontext() as backend:
                if gpu:
                    torch.mps.synchronize()
                start = perf_counter_ns()
                result = fn(*args, **case.kwargs, **extra)
                if gpu:
                    torch.mps.synchronize()
                elapsed = (perf_counter_ns() - start) / 1e6
                if gpu and backend.cpu_ops:
                    raise AssertionError("CPU replay in timed call")
                del result
            if r >= warmups:
                samples[name].append(elapsed)
    row["methods"] = {
        name: {"median_ms": statistics.median(v), "samples_ms": v}
        for name, v in samples.items()
    }
    paired_out = "cpu" in outputs and "mps" in outputs
    cpu = "cpu_out" if paired_out else "cpu_functional"
    mps = "mps_out" if paired_out else "mps_functional"
    reuse = "cpu_out" if "cpu" in outputs else "cpu_functional"

    def ratio(a, b):
        return row["methods"][a]["median_ms"] / row["methods"][b]["median_ms"]

    row["comparison"] = {
        "matched_cpu_method": cpu,
        "matched_mps_method": mps,
        "matched_cpu_over_mps": ratio(cpu, mps),
        "cpu_reuse_method": reuse,
        "cpu_reuse_over_mps": ratio(reuse, mps),
        "cpu_reuse_over_mps_functional": ratio(reuse, "mps_functional"),
        "functional_cpu_over_mps": ratio("cpu_functional", "mps_functional"),
    }
    row["status"] = "pass"
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--rounds", type=int, default=11)
    cli.add_argument("--warmups", type=int, default=3)
    cli.add_argument("--threads", type=int, default=12)
    cli.add_argument("--seed", type=int, default=514933)
    args = cli.parse_args()
    if args.rounds < 9 or args.warmups < 1 or args.threads < 1:
        cli.error("require at least 9 rounds, 1 warmup and 1 CPU thread")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable")
    torch.set_num_threads(args.threads)
    result = {
        "schema_version": 1,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "gpu": torch.backends.mps.get_name(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "physical_cpu_cores": sysctl("hw.physicalcpu"),
        "performance_cpu_cores": sysctl("hw.perflevel0.physicalcpu"),
        "macOS": platform.mac_ver()[0],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_threads": torch.get_num_threads(),
        "seed": args.seed,
        "rounds": args.rounds,
        "warmups": args.warmups,
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "package_source_sha256": source_digest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "design": "Fixed list in CASES, all rows reported. Input tensors are standard normal binary64, with independent inputs for binary operators and a precomputed Boolean condition for where. Data are generated once on CPU per case and copied without changing values or strides. Functional outputs allocate through each backend; out= is timed separately only after checking object identity and all output elements from a deliberately different sentinel. Each sample is one public torch operation, warmed and resident. The MPS queue is synchronized before the clock starts and again before it stops. Context setup, validation, host-device transfers and Python output destruction are outside the measured interval. Order is randomly shuffled within each round. MPS tensor results stay on the GPU. Native MPS CPU fallback and package CPU replay are disabled.",
        "comparison_rule": "matched_cpu_over_mps compares out= with out= when both are available, otherwise functional with functional. cpu_reuse_over_mps gives CPU a reused output whenever the public operation permits it and retains that same MPS method. A ratio above 1 means lower MPS median latency. cpu_reuse_over_mps_functional always uses MPS functional; ratios are kept separate.",
        "precision_rule": "All output elements checked against CPU float64 or exact boolean/integer output. Exact operations compare binary64 words. Transcendentals allow four ULP. Sum/mean/scan use 32*epsilon times absolute input sum/mean/prefix sum and also report independent sampled math.fsum errors. Variance and softmax allow 32*epsilon relative difference.",
        "measurements": [],
    }
    failed = False
    with torch.inference_mode():
        for index, case in enumerate(CASES):
            seed = args.seed + index
            try:
                row = run_case(case, seed, args.rounds, args.warmups)
            except (RuntimeError, AssertionError, TypeError) as exc:
                row = {
                    "id": case.id,
                    "operator": f"torch.{case.operator}",
                    "shape": list(case.shape),
                    "input_seed": seed,
                    "kwargs": case.kwargs,
                    "status": "failed",
                    "detail": error_text(exc),
                }
                failed = True
            result["measurements"].append(row)
            print(
                case.id,
                row["status"],
                round(row.get("comparison", {}).get("cpu_reuse_over_mps", 0.0), 3),
                row.get("detail", ""),
                flush=True,
            )
    passed = [r for r in result["measurements"] if r["status"] == "pass"]
    result["summary"] = {
        "declared": len(CASES),
        "passed": len(passed),
        "matched_wins": sum(
            r["comparison"]["matched_cpu_over_mps"] > 1 for r in passed
        ),
        "cpu_reuse_wins": sum(
            r["comparison"]["cpu_reuse_over_mps"] > 1 for r in passed
        ),
        "both_out_validated": sum(
            "mps_out" in r["methods"] and "cpu_out" in r["methods"] for r in passed
        ),
        "package_cpu_replay_calls_in_validated_methods": 0,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
