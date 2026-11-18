# mps-fp64

Run `torch.float64` operations on Apple GPUs through PyTorch eager mode. Inside `mps_fp64.enable()`, ordinary PyTorch tensors carry the MPS device and the float64 dtype. Metal kernels implement binary64 arithmetic, and selected matrix products use native GPU matrix instructions when the original inputs admit an exact decomposition.

[Explore the benchmarks](https://tangtaizong-buaa.github.io/mps-fp64/) · [Methods and results](BENCHMARKS.md) · [Download the release](https://github.com/Tangtaizong-BUAA/mps-fp64/releases/tag/v0.3.0.post1) · [中文说明](README.zh-CN.md)

## Measured performance

On a 40-core M4 Max with PyTorch 2.11.0, the GPU won 12 of 29 synchronized, device-resident matrix cases against the fastest measured native FP64 CPU path. The table shows wall time per public call after warmup. The two dyadic rows come from a separate comparison that also gave the CPU an exact, prepacked implementation and used the faster CPU result.

| FP64 workload | CPU ms | GPU ms | CPU time / GPU time |
|---|---:|---:|---:|
| Global argmax, 4,096 × 2,048 Gaussian values, CPU thread sweep | 3.973 | 0.300 | **13.24×** |
| Stable sort, 2,048 rows of 512 Gaussian values, CPU thread sweep | 3.959 | 0.384 | **10.30×** |
| 4,096-square matrix multiply, scaled integers in ±32,767 | 171.30 | 47.57 | **3.60×** |
| 4,096-square matrix multiply, scaled integers in ±2,047 | 131.53 | 44.46 | **2.96×** |
| Gaussian matrix product, (32,768 × 1,024) @ (1,024 × 2), first operand stored transposed | 3.081 | 1.976 | **1.56×** |
| 4,096-square matrix multiply, Gaussian FP64 | 203.10 | 1,265.71 | **0.16×** |

Each dyadic operand has its own power-of-two scale. The GPU verifies all input values on each call, so eligibility checks and packing are included. All 16,777,216 outputs of the ±32,767 case matched native CPU FP64 bit for bit. In the ordinary contiguous two-column case, copying the inputs from the CPU and the result back raised GPU time from 1.91 to 7.34 ms, compared with 2.47 ms on CPU. [Matrix timings](results/v3-large-matrices.json) · [Structure-aware CPU timings](results/v3-matrix-optimized-cpu.json)

The ordering experiment measured GPU calls in one block and swept 1, 4, 8, 12 and 16 CPU threads in separate blocks. A second benchmark interleaved CPU and GPU calls with the default 12 CPU threads. At the same shapes, it measured **3.67×** for argmax and **5.33×** for stable sort. [CPU thread sweep](benchmarks/ordering.md) · [Interleaved samples](results/public-tensor-ops.json)

Use the positive-results CSV to find workloads on which the GPU finished first. The full catalog also includes CPU wins, both input seeds where available, separate transfer and queued timings, quartiles, and baseline descriptions. Some rows reuse a workload with different timing conditions or output policies. The historical 0.2.0 measurements are labeled by version. [GPU-ahead CSV](benchmarks/gpu-ahead.csv) · [All comparisons CSV](benchmarks/all-results.csv) · [Per-path samples](benchmarks/timed-paths.csv)

## Install and run

The tested configuration is Apple Silicon, macOS 15.7.1, Python 3.13 and PyTorch 2.11.0 with MPS available. The first public package is **0.3.0.post1**. Its Python and Metal runtime sources are byte-identical to the 0.3.0 used for the matrix benchmark.

```bash
python -m pip install https://github.com/Tangtaizong-BUAA/mps-fp64/releases/download/v0.3.0.post1/mps_fp64-0.3.0.post1-py3-none-any.whl
```

```python
import torch
from mps_fp64 import enable

with enable() as backend:
    a = torch.tensor([[1., 1.], [1., 1. + 2**-40]],
                     device="mps", dtype=torch.float64)
    b = torch.tensor([2., 2. + 2**-40], device="mps", dtype=torch.float64)
    x = torch.linalg.solve(a, b)

    print(x.device, x.dtype, x.cpu().tolist())  # mps:0 torch.float64 [1.0, 1.0]
    assert not backend.cpu_ops
```

Both the last matrix entry and the last right-hand-side entry lose their increment in FP32. Keep tensor creation, computation, differentiation and copies inside the context. Enter a new context on each Python thread that uses these kernels. Unsupported operations raise by default. The optional CPU replay mode records each replay in `backend.cpu_ops`.

## Operator coverage and numerical tests

The implementation includes arithmetic, casts, transcendental functions, views, indexing, sorting, reductions, matrix products, selected factorizations, convolutions and FFTs. The fixed PyTorch 2.11 catalog passed 592 of 697 entries, including 551 of the 561 that passed the matched native MPS FP32 audit. A catalog entry covers a particular set of generated examples. Native ATen integration, graph compilation, other Apple GPUs and long-running workloads have outstanding validation work. [Operator records](results/audit-v3-matrices-fp64.jsonl) · [Project roadmap](ROADMAP.md)

The matrix arithmetic checks recorded zero mismatches in 173,403,648 Berkeley TestFloat output comparisons across six kernel paths, using round-to-nearest-even and NaN classification. The installed 0.3.0 research wheel passed 88 regression tests. Reduction order, transcendental-function error bounds and factorization residuals are checked separately. [Arithmetic records](results/v3-matrix-arithmetic.json) · [Matrix verification](MATRIX_PERFORMANCE.md) · [Function error measurements](FAST_MATH.md) · [SVD validation](SVD.md)

The standalone addition example exposes the guard, round and sticky bits in a single Metal shader and validates its outputs against CPU binary64. [Standalone example](examples/metal_fp64_add.py)

## Reproduce

From a checkout, install the optional validation dependencies in a virtual environment. These commands write fresh runs outside the tracked reference records. Rebuilding the published tables from the checked-in evidence works on any machine with Python 3.10 or later. Re-running GPU timings requires an MPS-capable Mac. The specialized CPU comparison builds a small Accelerate reference with Xcode Command Line Tools.

```bash
python -m pip install -e '.[validation]'
python tools/publish_benchmarks.py --check

export PYTORCH_ENABLE_MPS_FALLBACK=0
python -m unittest discover -s tests -v
python tools/fetch_benchmark_baselines.py
python tools/benchmark_large_matrices.py .benchmarks/mps_fp64-0.2.0-1public-py3-none-any.whl .benchmarks/matrices-local.json --rounds 9
python tools/benchmark_cpu_lattice.py .benchmarks/structure-local.json --rounds 7
python tools/benchmark_tensor_ops.py --output .benchmarks/tensors-local.json
```

The baseline download verifies each public wheel's SHA-256. The historical artifact hashes and corresponding numerical source digests are recorded alongside the public hashes. [Baseline manifest](benchmarks/baselines.json) · [Benchmark methods and reproduction commands](BENCHMARKS.md)

## License

Original code, documentation and benchmark tooling are released under the MIT license. The distribution preserves separate notices for Berkeley SoftFloat, metal-softfloat, OpenLibm, and code adapted from PyTorch and its upstream authors. [MIT license](LICENSE) · [Third-party notices and pinned sources](THIRD_PARTY.md)
