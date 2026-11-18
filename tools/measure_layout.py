"""Compare the resident pointwise dispatch path with a previous mps-fp64 wheel."""

import argparse
import importlib.util
import json
import platform
from pathlib import Path
import sys
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_fp64 import kernels  # noqa: E402


def previous_kernels(wheel, temp):
    with ZipFile(wheel) as archive:
        # Only the previous kernel generator and its Metal source files are needed.
        for name in archive.namelist():
            if (
                name == "mps_fp64/kernels.py"
                or name.startswith("mps_fp64/")
                and name.endswith(".metal")
            ):
                archive.extract(name, temp)
    source = Path(temp) / "mps_fp64/kernels.py"
    spec = importlib.util.spec_from_file_location("reference_pointwise", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(previous):
    data = torch.randn(
        1024, 1024, dtype=torch.float64, generator=torch.Generator().manual_seed(334)
    )
    a = data.view(torch.int64).to("mps").view(torch.float64)
    row, col = a[0], a[:, :1]
    cases = {
        "4096_plus_scalar": (a[:4], 2.5),
        "1M_plus_scalar": (a, 2.5),
        "1M_plus_row": (a, row),
        "1M_plus_column": (a, col),
        "1M_transpose_plus_row": (a.T, row),
        "4096_contiguous_add": (a[:4], a[:4]),
        "1M_contiguous_add": (a, a),
        "1M_expanded_plus_transpose": (col.expand_as(a), a.T),
    }
    results = {}
    for name, inputs in cases.items():
        samples = {"old": [], "new": []}
        providers = {"old": previous.pointwise, "new": kernels.pointwise}
        reference = None
        for tag, fn in providers.items():
            cpu = fn("add", *inputs).view(torch.int64).cpu()
            if tag == "old":
                reference = cpu
            else:
                torch.testing.assert_close(cpu, reference, rtol=0, atol=0)
            for _ in range(5):
                result = fn("add", *inputs)
        torch.mps.synchronize()
        for round_no in range(7):
            order = ("old", "new") if round_no % 2 else ("new", "old")
            for tag in order:
                fn = providers[tag]
                start = perf_counter()
                for _ in range(30):
                    result = fn("add", *inputs)
                torch.mps.synchronize()
                samples[tag].append((perf_counter() - start) * 1000 / 30)
        del result
        medians = {tag: median(times) for tag, times in samples.items()}
        results[name] = {
            "median_ms": medians,
            "speedup": medians["old"] / medians["new"],
            "samples_ms": samples,
        }
        print(name, medians, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory() as tmp:
        results = measure(previous_kernels(args.baseline, tmp))
    report = {
        "device": torch.backends.mps.get_name(),
        "gpu_cores": torch.backends.mps.get_core_count(),
        "torch": torch.__version__,
        "macos": platform.mac_ver()[0],
        "seed": 334,
        "baseline_wheel": args.baseline.name,
        "methodology": "resident tensors, allocation and submission included, 7 alternating synchronized batches of 30 calls",
        "operations": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
