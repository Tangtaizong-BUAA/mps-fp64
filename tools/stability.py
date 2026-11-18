"""Synchronized soak of a deterministic FP64 workload and allocator high-water marks."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mps_fp64 import enable


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/stability.json",
    )
    report_path = cli.parse_args().report
    values = torch.arange(65536, dtype=torch.float64) + (2**-32)
    params = torch.tensor(2**-34, dtype=torch.float64)
    samples = []
    begin = perf_counter()
    with enable() as mode:
        x = values.to("mps")
        delta = params.to("mps")
        # Exactly reversible for this input and precision. FP32 loses the low bits.
        for i in range(1, 4097):
            x = (x + delta) - delta
            if i % 256 == 0:
                torch.mps.synchronize()
                assert torch.equal(x.cpu().view(torch.int64), values.view(torch.int64))
                samples.append(
                    {
                        "step": i,
                        "allocated": torch.mps.current_allocated_memory(),
                        "driver_allocated": torch.mps.driver_allocated_memory(),
                    }
                )
        assert not mode.cpu_ops
    report = {
        "elapsed_s": perf_counter() - begin,
        "elements": values.numel(),
        "iterations": 4096,
        "scope": "one process, sequential GPU iterations, FP64 add/sub. Snapshot every 256 iterations.",
        "samples": samples,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report["elapsed_s"], samples[0], samples[-1])


if __name__ == "__main__":
    main()
