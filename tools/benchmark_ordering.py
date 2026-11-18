"""Time public binary64 ordering calls against a sweep of PyTorch CPU thread counts."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import median
import subprocess
from time import perf_counter_ns

import torch

from benchmark_tensor_ops import (
    copy_device,
    make_output,
    source_digest,
    sysctl,
    tensors,
)
from mps_fp64 import enable

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((128, 128), (2048, 512), (4096, 2048))
OPERATORS = {
    "sort": {"dim": -1, "descending": False, "stable": True},
    "argmax": {},
    "topk": {"k": 32, "dim": -1, "largest": True, "sorted": True},
}
THREADS = (1, 4, 8, 12, 16)
SEED = 20261118
WARMUPS = 2
ROUNDS = 9


def exact_check(output, reference, device, buffers):
    actual = tensors(output)
    expected = tensors(reference)
    assert len(actual) == len(expected)
    if buffers is not None:
        assert all(x is b for x, b in zip(actual, tensors(buffers)))
    checked = {"value_words": 0, "indices": 0, "mismatches": 0}
    for x, y in zip(actual, expected):
        if (x.device.type, x.dtype, x.shape) != (device, y.dtype, y.shape):
            raise AssertionError("incorrect output device, dtype, or shape")
        # Copy a noncontiguous top-k view into a CPU tensor before comparing bits.
        actual_words = x.detach().cpu().contiguous().view(torch.int64)
        expected_words = y.contiguous().view(torch.int64)
        if not torch.equal(actual_words, expected_words):
            raise AssertionError("mismatch in full value or index tensor")
        checked["value_words" if x.dtype == torch.float64 else "indices"] += x.numel()
    return checked


def measure(row, x, reference, host_threads, rng):
    gpu = host_threads is None
    device = "mps" if gpu else "cpu"
    prefix = "mps" if gpu else f"cpu_t{host_threads}"
    params = row["kwargs"]
    fn = getattr(torch, row["operator"].split(".")[1])
    with enable(cpu_fallback=False) if gpu else nullcontext() as backend:
        paths = {"functional": {}, "out": {"out": make_output(reference, gpu)}}
        for allocation, options in list(paths.items()):
            try:
                output = fn(x, **params, **options)
            except NotImplementedError as exc:
                if not gpu or allocation != "out":
                    raise
                row["out_unavailable"][device] = str(exc).splitlines()[0]
                del paths[allocation]
                continue
            checked = exact_check(output, reference, device, options.get("out"))
            del output
            row["methods"][f"{prefix}_{allocation}"] = {
                "device": device,
                "threads": host_threads,
                "allocation": allocation,
                "validation": checked,
                "samples_ms": [],
            }
        path_order = list(paths)
        rng.shuffle(path_order)
        for round_index in range(WARMUPS + ROUNDS):
            # Alternate the two calls' order within the CPU thread or GPU block.
            for allocation in path_order:
                options = paths[allocation]
                start = perf_counter_ns()
                output = fn(x, **params, **options)
                if gpu:
                    torch.mps.synchronize()
                elapsed = (perf_counter_ns() - start) / 1e6
                del output
                if round_index >= WARMUPS:
                    row["methods"][f"{prefix}_{allocation}"]["samples_ms"].append(
                        elapsed
                    )
            path_order.reverse()
        if gpu and backend.cpu_ops:
            raise AssertionError(f"CPU replay detected: {backend.cpu_ops}")
    for allocation in paths:
        data = row["methods"][f"{prefix}_{allocation}"]
        assert len(data["samples_ms"]) == ROUNDS
        data["median_ms"] = median(data["samples_ms"])


def collect():
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    # Use identical contiguous host and device data across operators at each size.
    torch.set_num_threads(1)
    rng = random.Random(SEED)
    host, gpu, references, rows = {}, {}, {}, []
    for index, shape in enumerate(SHAPES):
        seed = SEED + index
        host[shape] = torch.randn(
            shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
        )
        with enable(cpu_fallback=False):
            gpu[shape] = copy_device(host[shape])
        fingerprint = sha256(memoryview(host[shape].numpy()).cast("B")).hexdigest()
        for name, kwargs in OPERATORS.items():
            item = f"{name}_{shape[0]}x{shape[1]}"
            references[item] = getattr(torch, name)(host[shape], **kwargs)
            rows.append(
                {
                    "id": item,
                    "operator": f"torch.{name}",
                    "kwargs": kwargs,
                    "shape": list(shape),
                    "input_elements": math.prod(shape),
                    "input_seed": seed,
                    "input_sha256": fingerprint,
                    "out_unavailable": {},
                    "methods": {},
                }
            )
    # Warm hardware and JIT outside timing. Each later timed GPU call has one sync at its end.
    torch.mps.synchronize()
    order = list(range(len(rows)))
    rng.shuffle(order)
    gpu_order = [rows[i]["id"] for i in order]
    for i in order:
        row = rows[i]
        measure(row, gpu[tuple(row["shape"])], references[row["id"]], None, rng)
    del gpu
    print("All GPU measurements and exact GPU checks complete.", flush=True)

    block_order = list(THREADS)
    rng.shuffle(block_order)
    cpu_orders = {}
    for n in block_order:
        torch.set_num_threads(n)
        assert torch.get_num_threads() == n
        rng.shuffle(order)
        cpu_orders[str(n)] = [rows[i]["id"] for i in order]
        for i in order:
            row = rows[i]
            measure(row, host[tuple(row["shape"])], references[row["id"]], n, rng)
        print(f"CPU block finished at {n} thread(s).", flush=True)

    for row in rows:
        methods = row["methods"]
        cpu = min(
            (n for n in methods if n.startswith("cpu_")),
            key=lambda n: methods[n]["median_ms"],
        )
        mps = "mps_functional"
        row["comparison"] = {
            "cpu_method": cpu,
            "cpu_threads": methods[cpu]["threads"],
            "cpu_median_ms": methods[cpu]["median_ms"],
            "mps_method": mps,
            "mps_median_ms": methods[mps]["median_ms"],
            "speedup": methods[cpu]["median_ms"] / methods[mps]["median_ms"],
        }
    display = json.loads(
        subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
    )["SPDisplaysDataType"][0]
    return {
        "schema_version": 1,
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "macOS": platform.mac_ver()[0],
        "python": platform.python_version(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "physical_cpu_cores": int(sysctl("hw.physicalcpu")),
        "performance_cpu_cores": int(sysctl("hw.perflevel0.physicalcpu")),
        "efficiency_cpu_cores": int(sysctl("hw.perflevel1.physicalcpu")),
        "gpu": display["_name"],
        "gpu_cores": int(display["sppci_cores"]),
        "package_source_sha256": source_digest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": "PyTorch CPU functional at 1 thread. Every returned binary64 word and int64 index checked once per operator, shape, thread count, and supported allocation mode.",
        "input": "Contiguous seeded float64 torch.randn, transferred once as raw IEEE bits; same samples for all three operators at each shape.",
        "rounds": ROUNDS,
        "warmups": WARMUPS,
        "cpu_threads": list(THREADS),
        "order_seed": SEED,
        "gpu_case_order": gpu_order,
        "cpu_thread_block_order": block_order,
        "cpu_case_order_by_threads": cpu_orders,
        "timing": "Single public call per sample on resident inputs, one MPS synchronization at the end of every measured GPU call. FP64 CPU and MPS blocks are separate. Functional and preallocated calls alternate within each block. Allocation of output buffers, validation, copies, and warmups are outside samples.",
        "comparison_rule": "Lowest median among all CPU thread counts and public allocation modes divided by public functional MPS median; ratios above one favor MPS.",
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "package_cpu_replay_calls": 0,
        "measurements": rows,
        "summary": {
            "cells": len(rows),
            "mps_faster": sum(row["comparison"]["speedup"] > 1 for row in rows),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/public-ordering.json"
    )
    args = parser.parse_args()
    with torch.no_grad():
        report = collect()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["measurements"]:
        data = row["comparison"]
        print(
            f"{row['id']:20} GPU {data['mps_median_ms']:.4f} ms  CPU {data['cpu_median_ms']:.4f} ms  {data['cpu_method']}  {data['speedup']:.3f}x"
        )


if __name__ == "__main__":
    main()
