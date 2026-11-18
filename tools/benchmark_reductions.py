"""Time device-resident, public FP64 reductions, ordering and pooling operations."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter_ns

import torch
import torch.nn.functional as F


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--package",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing the mps_fp64 package to benchmark",
    )
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--rounds", type=int, default=7)
    args = cli.parse_args()
    sys.path.insert(0, str(args.package.resolve()))
    from mps_fp64 import enable

    rows = {}
    torch.manual_seed(183)

    def tensor(shape):
        return torch.randn(shape, dtype=torch.float64).to("mps")

    def measure(name, operation, calls):
        for _ in range(4):
            operation()
        torch.mps.synchronize()
        times = []
        for _ in range(args.rounds):
            start = perf_counter_ns()
            for _ in range(calls):
                operation()
            torch.mps.synchronize()
            times.append((perf_counter_ns() - start) / (1e6 * calls))
        rows[name] = {
            "median_ms": median(times),
            "samples_ms": times,
            "calls_per_batch": calls,
        }
        print(name, round(median(times), 5), flush=True)

    with enable() as mode:
        x = tensor((1024, 4096))
        measure("sum_all_4194304", lambda: x.sum(), 6)
        measure("sum_rows_1024x4096", lambda: x.sum(-1), 6)
        measure("sum_columns_1024x4096", lambda: x.sum(0), 6)
        short_rows = x.reshape(-1, 4)
        measure("sum_rows_1048576x4", lambda: short_rows.sum(-1), 4)
        small = x[:512, :32].contiguous()
        measure("sum_rows_512x32", lambda: small.sum(-1), 24)
        measure("prod_rows_1024x4096", lambda: x.prod(-1), 6)
        measure("amax_rows_1024x4096", lambda: x.amax(-1), 6)
        measure("amin_columns_1024x4096", lambda: x.amin(0), 6)
        measure("argmax_all_262144", lambda: x[:64].argmax(), 3)
        measure("max_rows_1024x4096", lambda: x.max(-1), 4)
        measure("max_rows_512x32", lambda: small.max(-1), 24)
        for shape in [(512, 32), (128, 257), (32, 4096), (1, 262144)]:
            data = tensor(shape)
            measure(
                f"stable_sort_{shape[0]}x{shape[1]}",
                lambda: data.sort(dim=-1, stable=True),
                4 if shape[1] < 4096 else 2,
            )
        img = tensor((2, 8, 64, 64))
        measure("avg_pool_2x8x64x64_k2", lambda: F.avg_pool2d(img, 2), 12)
        measure(
            "max_pool_2x8x64x64_k2",
            lambda: F.max_pool2d(img, 2, return_indices=True),
            12,
        )
        pooled, indices = F.max_pool2d(img, 2, return_indices=True)
        dy = tensor(pooled.shape)
        measure(
            "avg_pool_bwd_2x8x64x64_k2",
            lambda: torch.ops.aten.avg_pool2d_backward(
                dy, img, [2, 2], [2, 2], [0, 0], False, True, None
            ),
            12,
        )
        measure(
            "max_pool_bwd_2x8x64x64_k2",
            lambda: torch.ops.aten.max_pool2d_with_indices_backward(
                dy, img, [2, 2], [2, 2], [0, 0], [1, 1], False, indices
            ),
            12,
        )
        volume = tensor((2, 4, 18, 20, 22))
        measure(
            "avg_pool3d_2x4x18x20x22_k3",
            lambda: F.avg_pool3d(volume, 3, stride=2, padding=1),
            12,
        )
        measure(
            "adaptive_max_pool3d_2x4x18x20x22",
            lambda: F.adaptive_max_pool3d(volume, (5, 6, 7)),
            12,
        )
        measure("unpool_2x8x32x32_k2", lambda: F.max_unpool2d(pooled, indices, 2), 4)
        if mode.cpu_ops:
            raise RuntimeError(f"CPU replay occurred: {mode.cpu_ops}")

    names = (
        "kernels.py",
        "kernels.metal",
        "order.py",
        "ordering.metal",
        "pooling.py",
        "pooling.metal",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update((args.package / "mps_fp64" / name).read_bytes())
    report = {
        "torch": torch.__version__,
        "macOS": platform.mac_ver()[0],
        "device": torch.backends.mps.get_name(),
        "implementation_sha256": digest.hexdigest(),
        "method": (
            "Resident public FP64 API in strict mode. Four warm-up calls per case, then synchronized "
            "wall-clock per batch. Median milliseconds per operation. Seed 183; no CPU replay."
        ),
        "rounds": args.rounds,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
