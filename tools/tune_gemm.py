"""Check output bits before comparing register tiles with the baseline."""

import json
from pathlib import Path
import torch
from benchmark import measure
from mps_fp64.kernels import library, F64, I64

tiles = [
    ("batch_mm", 1, 1, 8, 8),
    ("mm_2216", 2, 2, 8, 8),
    ("mm_4216", 4, 2, 8, 8),
    ("mm_2116", 2, 1, 16, 4),
    ("mm_118", 1, 1, 16, 8),
    ("mm_near", 1, 1, 16, 8),
    ("mm_staged", 1, 1, 16, 8),
    ("mm_1116", 1, 1, 16, 8),
]


def main():
    torch.manual_seed(80)
    rows = {}
    for m, n, k in [(61, 73, 83), (128, 128, 128), (256, 256, 256), (512, 512, 512)]:
        a = torch.randn((m, k), dtype=F64).view(I64).to("mps")
        b = torch.randn((k, n), dtype=F64).view(I64).to("mps")
        out = torch.empty((m, n), dtype=I64, device="mps")
        timing = {}
        for name, mr, nr, w, h in tiles:
            cols, rows_per_tile = w * nr, h * mr
            grid = (
                ((n + cols - 1) // cols) * w,
                ((m + rows_per_tile - 1) // rows_per_tile) * h,
                1,
            )
            kernel = getattr(library(), name)

            def run():
                kernel(out, a, b, m, n, k, threads=grid, group_size=(w, h, 1))

            run()
            if name == "batch_mm":
                reference = out.cpu()
            else:
                assert torch.equal(reference, out.cpu()), name
            timing[name] = measure(run, torch.mps.synchronize, 7, 4)["median_ms"]
        rows[f"{m}x{n}x{k}"] = timing
        print(m, timing, flush=True)
    (Path(__file__).resolve().parents[1] / "results/gemm-tiles.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
