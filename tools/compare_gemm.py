"""Interleave the previous and current output-allocating MPS matrix kernels."""

import argparse
import json
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter_ns

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mps_fp64 import enable
from mps_fp64.kernels import F64, I64, bits, library, mm


def previous_mm(a, b, batched):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    out = torch.zeros((*a.shape[:-2], m, n), dtype=I64, device=a.device)
    wide = min(m, k, n) >= 32
    width = 16 if wide else 8
    kernel = library().mm_near if wide else library().batch_mm
    kernel(
        out, bits(a).contiguous(), bits(b).contiguous(), m, n, k,
        threads=(((n + width - 1) // width) * width, ((m + 7) // 8) * 8,
                 a.shape[0] if batched else 1),
        group_size=(width, 8, 1),
    )
    return out.view(F64)


def inputs(batch, m, k, n, layout, generator):
    def new(*shape):
        return torch.randn(*shape, dtype=F64, generator=generator).view(I64).to("mps").view(F64)

    leading = (batch,) if batch else ()
    a = new(*leading, k, m).mT if layout == "both_transposed" else new(*leading, m, k)
    if layout == "expanded_weight":
        b = new(1, k, n).expand(batch, k, n)
    elif layout in {"both_transposed", "weight_transposed"}:
        b = new(*leading, n, k).mT
    else:
        b = new(*leading, k, n)
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(718381)
    order = random.Random(351)
    report = {
        "torch": torch.__version__, "macOS": platform.mac_ver()[0],
        "gpu": torch.backends.mps.get_name(),
        "method": "Same seeded normal input tensors. Compare kernel entry points including allocation and stride preparation, excluding dispatcher, transfer and compilation. Alternate order per round, synchronize each batch, report all call-normalized wall times. Every output bit is checked.",
        "measurements": [],
    }
    cases = [
        (0, 7, 5, 4, "contiguous"),
        (0, 32, 64, 16, "weight_transposed"),
        (0, 128, 128, 128, "contiguous"),
        (0, 256, 256, 256, "contiguous"),
        (0, 512, 512, 512, "contiguous"),
        (0, 1024, 1024, 1024, "contiguous"),
        (0, 257, 253, 259, "both_transposed"),
        (12, 32, 64, 24, "expanded_weight"),
        (64, 8, 32, 4, "contiguous"),
    ]
    for batch, m, k, n, layout in cases:
        a, b = inputs(batch, m, k, n, layout, generator)
        batched = bool(batch)
        methods = {"previous": previous_mm, "current": mm}
        reference = bits(previous_mm(a, b, batched)).cpu()
        assert torch.equal(reference, bits(mm(a, b, batched)).cpu())
        timings = {name: [] for name in methods}
        calls = 2 if max(m, k, n) >= 1024 else 4 if max(m, k, n) >= 512 else 24
        for run in range(args.repeats + 1):
            names = list(methods)
            order.shuffle(names)
            for name in names:
                fn = methods[name]
                torch.mps.synchronize()
                start = perf_counter_ns()
                for _ in range(calls):
                    fn(a, b, batched)
                torch.mps.synchronize()
                elapsed = (perf_counter_ns() - start) / 1e6 / calls
                if run:
                    timings[name].append(elapsed)
        medians = {name: median(values) for name, values in timings.items()}
        row = {
            "batch": batch if batched else None, "m": m, "n": n, "k": k,
            "layout": layout, "calls_per_sample": calls, "checked_output_bits": reference.numel(),
            "median_ms": medians, "speedup": medians["previous"] / medians["current"],
            "current_GFLOP_s": 2 * max(1, batch) * m * n * k / medians["current"] / 1e6,
            "samples_ms": timings,
        }
        report["measurements"].append(row)
        print(batch, m, k, n, layout, medians, round(row["speedup"], 2), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
