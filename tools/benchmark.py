"""Device-resident, synchronized throughput. No transfer/compilation in timed sections."""

import argparse
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mps_fp64 import enable
from mps_fp64.kernels import library


def measure(fn, synchronize, repeats, calls):
    for _ in range(3):
        fn()
    synchronize()
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        for _ in range(calls):
            fn()
        synchronize()
        samples.append((perf_counter_ns() - start) / 1e6 / calls)
    return {"median_ms": median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(173)
    rows = {}
    with enable():
        n = 2**23
        a, b = (torch.randn(n, dtype=torch.float64) for _ in range(2))
        x, y = a.to("mps"), b.to("mps")
        raw_x, raw_y = x.view(torch.int64), y.view(torch.int64)
        buffer = torch.empty_like(raw_x)
        lib = library()
        for name, fn in [
            ("matched_xor", lambda: lib.bandwidth(buffer, raw_x, raw_y, threads=(n,))),
            ("raw_add", lambda: lib.op_add(buffer, raw_x, raw_y, threads=(n,))),
            ("add", lambda: x + y),
            ("mul", lambda: x * y),
            ("div", lambda: x / y),
        ]:
            row = measure(fn, torch.mps.synchronize, 9, 12)
            row["elements"] = n
            row["GB_s"] = 24 * n / row["median_ms"] / 1e6
            rows[name] = row
        rows["cpu_add"] = measure(lambda: a + b, lambda: None, 9, 12)
        for size in [128, 256]:
            a = torch.randn(size, size, dtype=torch.float64)
            b = torch.randn_like(a)
            for device, dtype in [
                ("cpu", torch.float64),
                ("mps", torch.float64),
                ("mps", torch.float32),
            ]:
                x, y = a.to(device, dtype), b.to(device, dtype)
                row = measure(
                    lambda: x @ y,
                    torch.mps.synchronize if device == "mps" else lambda: None,
                    7,
                    4,
                )
                row["GFLOP_s"] = 2 * size**3 / row["median_ms"] / 1e6
                rows[f"mm_{size}_{device}_{dtype}"] = row
    report = {
        "torch": torch.__version__,
        "macOS": platform.mac_ver()[0],
        "cpu_threads": torch.get_num_threads(),
        "method": "FP64 streaming inputs independent standard normal; 24 logical bytes/element. Wall-clock synchronized batches; warmed up. Raw kernels reuse output; public ops allocate output.",
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, r in rows.items():
        print(
            name, round(r["median_ms"], 4), round(r.get("GB_s", r.get("GFLOP_s", 0)), 2)
        )


if __name__ == "__main__":
    main()
