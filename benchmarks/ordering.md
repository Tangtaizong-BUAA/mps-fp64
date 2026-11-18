# FP64 ordering on an Apple M4 Max

Stable row sorting finished in **0.384 ms** for a 2,048 × 512 matrix, compared with **3.959 ms** for the fastest measured PyTorch CPU configuration. At 4,096 × 2,048, global `argmax` took **0.300 ms** on the GPU and **3.973 ms** on the CPU. Row-wise `topk(32)` changed leader with size. The GPU led at 2,048 × 512, while the CPU led at 128 × 128 and 4,096 × 2,048. [Per-call samples and exact checks](../results/public-ordering.json)

## Resident-input latency

Each ratio divides the fastest CPU median by the functional GPU median. The CPU column selects from ten measured public paths at each size. An `out` call reuses a caller-provided result buffer. Times are wall-clock milliseconds per complete public call. The GPU finished sooner in six of nine comparisons.

| Operation | Input shape | GPU ms | Fastest CPU ms | CPU threads | CPU call | CPU/GPU |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Stable sort | 128 × 128 | 0.1623 | 0.2192 | 12 | out | 1.350× |
| Stable sort | 2048 × 512 | 0.3843 | 3.9587 | 12 | out | 10.300× |
| Stable sort | 4096 × 2048 | 3.4765 | 34.3484 | 12 | out | 9.880× |
| Global argmax | 128 × 128 | 0.1692 | 0.0082 | 4 | out | 0.048× |
| Global argmax | 2048 × 512 | 0.2338 | 0.4691 | 4 | out | 2.007× |
| Global argmax | 4096 × 2048 | 0.3001 | 3.9730 | 12 | functional | 13.240× |
| Row top-32 | 128 × 128 | 0.1631 | 0.0433 | 1 | out | 0.266× |
| Row top-32 | 2048 × 512 | 0.3922 | 0.7246 | 12 | out | 1.847× |
| Row top-32 | 4096 × 2048 | 3.3530 | 1.9873 | 12 | out | 0.593× |

The GPU top-k handler sorts each complete row in descending order and returns a view of its first 32 entries. For 2,048-column rows, that path took 3.353 ms and the full ascending stable sort took 3.476 ms. On the same rows the fastest CPU top-k call took 1.987 ms, compared with 34.348 ms for CPU stable sort. [GPU top-k dispatch](../mps_fp64/backend.py) · [Ordering kernels](../mps_fp64/ordering.metal)

## CPU thread sweep

Each cell gives the faster of the functional and preallocated CPU medians, in milliseconds. The machine has 12 performance cores and four efficiency cores. The column headings are PyTorch intra-op thread settings. macOS scheduled the threads.

| Operation | Shape | 1 thread | 4 threads | 8 threads | 12 threads | 16 threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable sort | 128 × 128 | 0.3297 | 0.2604 | 0.2347 | 0.2192 | 0.3674 |
| Stable sort | 2048 × 512 | 37.7683 | 9.7294 | 4.9099 | 3.9587 | 4.4242 |
| Stable sort | 4096 × 2048 | 375.0876 | 94.7986 | 48.1000 | 34.3484 | 35.5309 |
| Global argmax | 128 × 128 | 0.0084 | 0.0082 | 0.0091 | 0.0084 | 0.0092 |
| Global argmax | 2048 × 512 | 0.5252 | 0.4691 | 0.4762 | 0.5207 | 0.5310 |
| Global argmax | 4096 × 2048 | 4.2977 | 4.2532 | 4.2065 | 3.9730 | 4.2741 |
| Row top-32 | 128 × 128 | 0.0433 | 0.0714 | 0.0471 | 0.0451 | 0.0468 |
| Row top-32 | 2048 × 512 | 7.7574 | 2.0461 | 1.0678 | 0.7246 | 0.9525 |
| Row top-32 | 4096 × 2048 | 20.9582 | 5.3215 | 2.6895 | 1.9873 | 2.4955 |

## Inputs, timing and correctness

The run used PyTorch 2.11.0 on macOS 15.7.1, with a 40-core M4 Max GPU. The three shapes and operations were fixed before measurement. Each shape used a contiguous binary64 normal tensor from a seeded CPU generator and the identical bits resident on MPS. The small, medium and large seeds were 20261118, 20261119 and 20261120. The calls were `torch.sort(x, dim=-1, stable=True)`, `torch.argmax(x)` and `torch.topk(x, 32, dim=-1, largest=True, sorted=True)`.

Each supported path received two warmups and nine individually timed calls. GPU timing includes dispatch and a single `torch.mps.synchronize()` at the end of the call. Input transfers and correctness checks were outside the timer. The functional call includes allocation of the returned tensor or tensors. CPU buffers for `out=` were allocated before timing. The three MPS `out=` overloads raised `NotImplementedError` in strict mode. The raw data identifies the specific ATen overload in each case.

The GPU block finished before the CPU sweep. A seeded permutation determined case order on the GPU, thread-block order on the CPU, and the order of cases within each CPU block. CPU functional and preallocated calls alternated their order within each case. Medians compare separate blocks. The raw file retains all **891** timed calls and the recorded block orders.

Before timing, every supported path was checked against an independent call to the functional CPU operator with one thread. The GPU comparisons covered **9,654,272** full binary64 output words and **9,654,275** integer indices across the nine cases, with zero mismatches. The ten CPU configurations per case passed the same check. Strict mode recorded zero package CPU replays. [Raw measurements](../results/public-ordering.json) · [Benchmark source](../tools/benchmark_ordering.py)

From a source checkout with the project and its validation dependencies installed, run

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/benchmark_ordering.py --output results/public-ordering.json
```
