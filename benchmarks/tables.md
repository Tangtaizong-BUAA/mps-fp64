# Benchmark tables

Times are median wall-clock milliseconds per call. **CPU/GPU** is the CPU median divided by the GPU median. A ratio above one means the GPU finished sooner. A round with multiple calls measures their average. Times and ratios are rounded for display. The downloads retain full numeric precision. [Benchmark guide](../BENCHMARKS.md) · [All comparisons](all-results.csv) · [GPU-ahead comparisons](gpu-ahead.csv) · [Per-path samples](timed-paths.csv)

## 0.3.0 matrix latency

Recorded in [v3-large-matrices.json](../results/v3-large-matrices.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `normal_square_1024` | 1024×1024 · 1024×1024 | `cpu_out` | 3.047 | 17.877 | 0.17× | 9 × 1 |
| `normal_square_2048` | 2048×2048 · 2048×2048 | `accelerate_dgemm` | 21.823 | 140.88 | 0.15× | 9 × 1 |
| `normal_square_4096` | 4096×4096 · 4096×4096 | `cpu_out` | 203.10 | 1,265.71 | 0.16× | 9 × 1 |
| `dyadic_4_1024` | 1024×1024 · 1024×1024 | `cpu_out` | 3.113 | 1.085 | 2.87× | 9 × 1 |
| `dyadic_4_2048` | 2048×2048 · 2048×2048 | `cpu_out` | 23.021 | 3.261 | 7.06× | 9 × 1 |
| `dyadic_4_4096` | 4096×4096 · 4096×4096 | `accelerate_dgemm` | 180.50 | 20.647 | 8.74× | 9 × 1 |
| `dyadic_12_1024` | 1024×1024 · 1024×1024 | `accelerate_dgemm` | 3.199 | 1.515 | 2.11× | 9 × 1 |
| `dyadic_12_2048` | 2048×2048 · 2048×2048 | `cpu_out` | 22.308 | 5.650 | 3.95× | 9 × 1 |
| `dyadic_12_4096` | 4096×4096 · 4096×4096 | `cpu_out` | 183.89 | 40.413 | 4.55× | 9 × 1 |
| `dyadic_16_1024` | 1024×1024 · 1024×1024 | `accelerate_dgemm` | 3.083 | 1.773 | 1.74× | 9 × 1 |
| `dyadic_16_2048` | 2048×2048 · 2048×2048 | `cpu_out` | 21.957 | 5.896 | 3.72× | 9 × 1 |
| `dyadic_16_4096` | 4096×4096 · 4096×4096 | `accelerate_dgemm` | 184.71 | 43.734 | 4.22× | 9 × 1 |
| `rank_1_4096` | 4096×1 · 1×4096 | `cpu_outer_out` | 0.5090 | 0.6992 | 0.73× | 9 × 1 |
| `rank_4_4096` | 4096×4 · 4×4096 | `accelerate_dgemm` | 1.209 | 2.713 | 0.45× | 9 × 1 |
| `rank_16_4096` | 4096×16 · 16×4096 | `cpu_out` | 2.270 | 6.409 | 0.35× | 9 × 1 |
| `rank_32_4096` | 4096×32 · 32×4096 | `cpu_out` | 4.076 | 10.846 | 0.38× | 9 × 1 |
| `rank_1_8192` | 8192×1 · 1×8192 | `cpu_outer_out` | 1.949 | 2.410 | 0.81× | 9 × 1 |
| `rank_4_8192` | 8192×4 · 4×8192 | `accelerate_dgemm` | 4.703 | 6.728 | 0.70× | 9 × 1 |
| `narrow_65536_256_1` | 65536×256 · 256×1 | `cpu_out` | 0.7381 | 0.6733 | 1.10× | 9 × 1 |
| `narrow_32768_1024_2` | 32768×1024 · 1024×2 | `accelerate_dgemm` | 2.473 | 1.908 | 1.30× | 9 × 1 |
| `narrow_65536_256_4` | 65536×256 · 256×4 | `cpu_out` | 1.545 | 2.152 | 0.72× | 9 × 1 |
| `narrow_8192_4096_1` | 8192×4096 · 4096×1 | `cpu_out` | 1.084 | 1.828 | 0.59× | 9 × 1 |
| `narrow_transposed` | 32768×1024 · 1024×2 | `cpu_out` | 3.081 | 1.976 | 1.56× | 9 × 1 |
| `mv_row` | 65536×256 · 256 | `cpu_out` | 0.7412 | 1.001 | 0.74× | 9 × 1 |
| `mv_col` | 65536×256 · 256 | `accelerate_dgemv` | 0.5493 | 1.057 | 0.52× | 9 × 1 |
| `batch_4096_3` | 4096×3×3 · 4096×3×3 | `cpu_out` | 0.0362 | 0.1700 | 0.21× | 9 × 1 |
| `batch_4096_4` | 4096×4×4 · 4096×4×4 | `cpu_out` | 0.0414 | 0.1371 | 0.30× | 9 × 1 |
| `batch_65536_3` | 65536×3×3 · 65536×3×3 | `cpu_out` | 0.1032 | 0.2100 | 0.49× | 9 × 1 |
| `batch_65536_4` | 65536×4×4 · 65536×4×4 | `cpu_out` | 0.2558 | 0.2993 | 0.85× | 9 × 1 |

## 0.3.0 matrix round trip

Recorded in [v3-large-matrices.json](../results/v3-large-matrices.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `normal_square_2048` | 2048×2048 · 2048×2048 | `accelerate_dgemm` | 21.823 | 144.29 | 0.15× | 9 × 1 |
| `dyadic_4_2048` | 2048×2048 · 2048×2048 | `cpu_out` | 23.021 | 6.301 | 3.65× | 9 × 1 |
| `dyadic_12_2048` | 2048×2048 · 2048×2048 | `cpu_out` | 22.308 | 8.244 | 2.71× | 9 × 1 |
| `dyadic_16_2048` | 2048×2048 · 2048×2048 | `cpu_out` | 21.957 | 8.750 | 2.51× | 9 × 1 |
| `rank_4_4096` | 4096×4 · 4×4096 | `accelerate_dgemm` | 1.209 | 15.183 | 0.08× | 9 × 1 |
| `narrow_32768_1024_2` | 32768×1024 · 1024×2 | `accelerate_dgemm` | 2.473 | 7.340 | 0.34× | 9 × 1 |

## 0.3.0 matrix queue throughput

Recorded in [v3-large-matrices.json](../results/v3-large-matrices.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `batch_4096_3` | 4096×3×3 · 4096×3×3 | `cpu` | 0.0366 | 0.0326 | 1.12× | 9 × 12 |
| `batch_4096_4` | 4096×4×4 · 4096×4×4 | `cpu_out` | 0.0484 | 0.0345 | 1.40× | 9 × 12 |
| `batch_65536_3` | 65536×3×3 · 65536×3×3 | `cpu_out` | 0.1172 | 0.1004 | 1.17× | 9 × 5 |
| `batch_65536_4` | 65536×4×4 · 65536×4×4 | `cpu_out` | 0.2174 | 0.1625 | 1.34× | 9 × 5 |

## 0.3.0 matrix latency, second seed

Recorded in [v3-matrix-second-seed.json](../results/v3-matrix-second-seed.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `normal_square_4096` | 4096×4096 · 4096×4096 | `accelerate_dgemm` | 175.90 | 1,921.80 | 0.09× | 5 × 1 |
| `dyadic_12_4096` | 4096×4096 · 4096×4096 | `cpu_out` | 177.12 | 55.200 | 3.21× | 5 × 1 |
| `dyadic_16_4096` | 4096×4096 · 4096×4096 | `accelerate_dgemm` | 182.16 | 52.188 | 3.49× | 5 × 1 |
| `rank_1_8192` | 8192×1 · 1×8192 | `cpu_outer_out` | 2.005 | 2.509 | 0.80× | 5 × 1 |
| `narrow_32768_1024_2` | 32768×1024 · 1024×2 | `cpu` | 3.018 | 1.810 | 1.67× | 5 × 1 |
| `batch_65536_4` | 65536×4×4 · 65536×4×4 | `cpu_out` | 0.3191 | 0.3015 | 1.06× | 5 × 1 |

## 0.3.0 matrix round trip, second seed

Recorded in [v3-matrix-second-seed.json](../results/v3-matrix-second-seed.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `narrow_32768_1024_2` | 32768×1024 · 1024×2 | `cpu` | 3.018 | 7.773 | 0.39× | 5 × 1 |

## 0.3.0 matrix queue throughput, second seed

Recorded in [v3-matrix-second-seed.json](../results/v3-matrix-second-seed.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `batch_65536_4` | 65536×4×4 · 65536×4×4 | `cpu_out` | 0.2362 | 0.1784 | 1.32× | 5 × 5 |

## 0.3.0 structure-aware CPU comparison

Recorded in [v3-matrix-optimized-cpu.json](../results/v3-matrix-optimized-cpu.json).

| Case | Native FP64 CPU ms | Prepacked CPU ms | GPU from FP64 ms | Fastest CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|
| `dyadic_4_1024` | 3.344 | 0.9875 | 2.272 | 0.43× | 7 × 1 |
| `dyadic_4_2048` | 22.332 | 5.858 | 5.277 | 1.11× | 7 × 1 |
| `dyadic_4_4096` | 176.16 | 45.294 | 21.877 | 2.07× | 7 × 1 |
| `dyadic_12_1024` | 3.240 | 2.794 | 1.331 | 2.10× | 7 × 1 |
| `dyadic_12_2048` | 22.634 | 16.536 | 10.764 | 1.54× | 7 × 1 |
| `dyadic_12_4096` | 172.67 | 131.53 | 44.462 | 2.96× | 7 × 1 |
| `dyadic_16_1024` | 3.177 | 3.634 | 1.353 | 2.35× | 7 × 1 |
| `dyadic_16_2048` | 22.264 | 25.330 | 6.467 | 3.44× | 7 × 1 |
| `dyadic_16_4096` | 171.30 | 209.61 | 47.572 | 3.60× | 7 × 1 |

## 0.3.0 tensor latency

Recorded in [public-tensor-ops.json](../results/public-tensor-ops.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `add_64k` | 256×256 · 256×256 | `cpu_out` | 0.0352 | 0.1860 | 0.19× | 11 × 1 |
| `add_8m` | 4096×2048 · 4096×2048 | `cpu_out` | 0.7945 | 1.065 | 0.75× | 11 × 1 |
| `add_transposed_1m` | 2048×512 · 2048×512 | `cpu_out` | 0.0743 | 0.5689 | 0.13× | 11 × 1 |
| `mul_8m` | 4096×2048 · 4096×2048 | `cpu_out` | 0.7258 | 1.262 | 0.58× | 11 × 1 |
| `div_8m` | 4096×2048 · 4096×2048 | `cpu_out` | 0.8097 | 1.064 | 0.76× | 11 × 1 |
| `abs_8m` | 4096×2048 | `cpu_out` | 0.4488 | 0.5597 | 0.80× | 11 × 1 |
| `neg_8m` | 4096×2048 | `cpu_out` | 0.4517 | 0.6644 | 0.68× | 11 × 1 |
| `lt_8m` | 4096×2048 · 4096×2048 | `cpu_out` | 0.5561 | 0.8103 | 0.69× | 11 × 1 |
| `relu_8m` | 4096×2048 | `cpu_functional` | 1.296 | 0.7378 | 1.76× | 11 × 1 |
| `where_8m` | 4096×2048 · 4096×2048 · 4096×2048 | `cpu_out` | 0.7901 | 1.161 | 0.68× | 11 × 1 |
| `exp_1m` | 2048×512 | `cpu_out` | 0.2852 | 0.7106 | 0.40× | 11 × 1 |
| `sin_1m` | 2048×512 | `cpu_out` | 0.2335 | 0.7120 | 0.33× | 11 × 1 |
| `sum_all_8m` | 4096×2048 | `cpu_functional` | 0.3075 | 0.8855 | 0.35× | 11 × 1 |
| `sum_rows_8m` | 4096×2048 | `cpu_functional` | 0.2693 | 0.7278 | 0.37× | 11 × 1 |
| `sum_columns_8m` | 4096×2048 | `cpu_functional` | 0.8765 | 0.7728 | 1.13× | 11 × 1 |
| `mean_rows_8m` | 4096×2048 | `cpu_out` | 0.2837 | 0.7912 | 0.36× | 11 × 1 |
| `amax_rows_8m` | 4096×2048 | `cpu_functional` | 0.2848 | 1.083 | 0.26× | 11 × 1 |
| `argmax_all_8m` | 4096×2048 | `cpu_functional` | 3.834 | 1.046 | 3.67× | 11 × 1 |
| `argmax_rows_8m` | 4096×2048 | `cpu_out` | 0.5885 | 0.6042 | 0.97× | 11 × 1 |
| `var_rows_8m` | 4096×2048 | `cpu_out` | 6.667 | 7.217 | 0.92× | 11 × 1 |
| `cumsum_rows_2m` | 4096×512 | `cpu_functional` | 0.2322 | 173.31 | 0.00× | 11 × 1 |
| `sort_rows_1m` | 2048×512 | `cpu_out` | 4.152 | 0.7785 | 5.33× | 11 × 1 |
| `topk_rows_1m` | 2048×512 | `cpu_functional` | 0.8800 | 0.4750 | 1.85× | 11 × 1 |
| `softmax_rows_1m` | 2048×512 | `cpu_out` | 0.3052 | 0.9112 | 0.33× | 11 × 1 |

## 0.3.0 ordering with tuned CPU threads

Recorded in [public-ordering.json](../results/public-ordering.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `sort_128x128` | 128×128 | `cpu_t12_out` | 0.2192 | 0.1623 | 1.35× | 9 × 1 |
| `argmax_128x128` | 128×128 | `cpu_t4_out` | 0.0082 | 0.1692 | 0.05× | 9 × 1 |
| `topk_128x128` | 128×128 | `cpu_t1_out` | 0.0433 | 0.1631 | 0.27× | 9 × 1 |
| `sort_2048x512` | 2048×512 | `cpu_t12_out` | 3.959 | 0.3843 | 10.30× | 9 × 1 |
| `argmax_2048x512` | 2048×512 | `cpu_t4_out` | 0.4691 | 0.2338 | 2.01× | 9 × 1 |
| `topk_2048x512` | 2048×512 | `cpu_t12_out` | 0.7246 | 0.3922 | 1.85× | 9 × 1 |
| `sort_4096x2048` | 4096×2048 | `cpu_t12_out` | 34.348 | 3.476 | 9.88× | 9 × 1 |
| `argmax_4096x2048` | 4096×2048 | `cpu_t12_functional` | 3.973 | 0.3001 | 13.24× | 9 × 1 |
| `topk_4096x2048` | 4096×2048 | `cpu_t12_out` | 1.987 | 3.353 | 0.59× | 9 × 1 |

## 0.2.0 named workloads, resident

Recorded in [v2-workloads.json](../results/v2-workloads.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `add_4096` | 4096 · 4096 | `cpu_fp64` | 0.0010 | 0.0202 | 0.05× | 11 × 64 |
| `add_262144` | 262144 · 262144 | `cpu_fp64` | 0.0916 | 0.0235 | 3.89× | 11 × 32 |
| `add_8388608` | 8388608 · 8388608 | `cpu_fp64` | 1.913 | 0.4677 | 4.09× | 11 × 8 |
| `mul_8388608` | 8388608 · 8388608 | `cpu_fp64` | 2.009 | 0.4684 | 4.29× | 11 × 8 |
| `div_8388608` | 8388608 · 8388608 | `cpu_fp64` | 1.916 | 0.4657 | 4.11× | 11 × 8 |
| `sin_1048576` | 1048576 | `cpu_fp64` | 0.4670 | 0.1866 | 2.50× | 11 × 10 |
| `cos_1048576` | 1048576 | `cpu_fp64` | 0.4774 | 0.1469 | 3.25× | 11 × 10 |
| `tanh_1048576` | 1048576 | `cpu_fp64` | 0.7014 | 0.1774 | 3.95× | 11 × 10 |
| `exp_1048576` | 1048576 | `cpu_fp64` | 0.3921 | 0.1394 | 2.81× | 11 × 10 |
| `log_1048576` | 1048576 | `cpu_fp64` | 0.5206 | 0.2094 | 2.49× | 11 × 10 |
| `mm_128` | 128×128 · 128×128 | `cpu_fp64` | 0.0131 | 0.0805 | 0.16× | 11 × 24 |
| `mm_256` | 256×256 · 256×256 | `cpu_fp64` | 0.0832 | 0.3518 | 0.24× | 11 × 12 |
| `mm_512` | 512×512 · 512×512 | `cpu_fp64` | 0.5154 | 2.489 | 0.21× | 11 × 4 |
| `mm_1024` | 1024×1024 · 1024×1024 | `cpu_fp64` | 3.719 | 18.887 | 0.20× | 11 × 2 |
| `bmm_16x64x128x96` | 16×64×128 · 16×128×96 | `cpu_fp64` | 0.0952 | 0.2714 | 0.35× | 11 × 12 |
| `mm_strided_256` | 256×256 · 256×256 | `cpu_fp64` | 0.0800 | 0.3539 | 0.23× | 11 × 12 |
| `sum_1024x4096_last` | 1024×4096 | `cpu_fp64` | 0.1139 | 0.2656 | 0.43× | 11 × 12 |
| `amax_1024x4096_last` | 1024×4096 | `cpu_fp64` | 0.1111 | 0.1327 | 0.84× | 11 × 12 |
| `max_1024x4096_last` | 1024×4096 | `cpu_fp64` | 0.2045 | 0.1029 | 1.99× | 11 × 12 |
| `sum_1024x4096_first` | 1024×4096 | `cpu_fp64` | 0.2902 | 0.2798 | 1.04× | 11 × 12 |
| `aminmax_1024x4096` | 1024×4096 | `cpu_fp64` | 0.1623 | 0.1631 | 1.00× | 11 × 12 |
| `aminmax_1024x4096_last` | 1024×4096 | `cpu_fp64` | 0.4808 | 0.1850 | 2.60× | 11 × 12 |
| `argmax_262144` | 262144 | `cpu_fp64` | 0.1250 | 0.1154 | 1.08× | 11 × 3 |
| `stable_sort_256x1024` | 256×1024 | `cpu_fp64` | 1.635 | 0.1524 | 10.73× | 11 × 4 |
| `max_pool_8x32x56x56` | 8×32×56×56 | `cpu_fp64` | 0.5047 | 0.0918 | 5.50× | 11 × 12 |
| `avg_pool_8x32x56x56` | 8×32×56×56 | `cpu_fp64` | 0.1682 | 0.1203 | 1.40× | 11 × 12 |
| `rfft_256x32` | 256×32 | `cpu_fp64` | 0.0135 | 0.1805 | 0.08× | 11 × 16 |
| `rfft_64x256` | 64×256 | `cpu_fp64` | 0.0156 | 0.1591 | 0.10× | 11 × 8 |
| `rfft_32x1024` | 32×1024 | `cpu_fp64` | 0.0395 | 0.4650 | 0.08× | 11 × 8 |
| `rfft_16x2048` | 16×2048 | `cpu_fp64` | 0.0471 | 0.5313 | 0.09× | 11 × 6 |
| `rfft_32x257` | 32×257 | `cpu_fp64` | 0.0670 | 0.4688 | 0.14× | 11 × 5 |
| `rfft_16x1021` | 16×1021 | `cpu_fp64` | 0.1658 | 0.6332 | 0.26× | 11 × 2 |
| `conv2d_1x1` | 4×32×32×32 · 32×32×1×1 · 32 | `cpu_fp64` | 0.0929 | 0.1700 | 0.55× | 11 × 5 |
| `conv2d_3x3` | 4×32×32×32 · 32×32×3×3 · 32 | `cpu_fp64` | 0.3262 | 2.230 | 0.15× | 11 × 5 |
| `cholesky_8x48` | 8×48×48 | `cpu_fp64` | 0.0648 | 0.3229 | 0.20× | 11 × 8 |
| `solve_8x48` | 8×48×48 · 8×48×8 | `cpu_fp64` | 0.0963 | 1.312 | 0.07× | 11 × 5 |
| `batch_norm_eval` | 8×32×32×32 · 32 · 32 · 32 · 32 | `cpu_fp64` | 0.0790 | 0.2354 | 0.34× | 11 × 10 |
| `cdist_8x96x12` | 8×96×12 · 8×96×12 | `cpu_fp64` | 0.0774 | 0.1105 | 0.70× | 11 × 8 |
| `scatter_reduce_mean` | 256×128 · 256×256 · 256×256 | `cpu_fp64` | 0.1522 | 0.4614 | 0.33× | 11 × 8 |
| `cummax_8x32767` | 8×32767 | `cpu_fp64` | 0.3534 | 0.1107 | 3.19× | 11 × 8 |
| `bessel_j0_262144` | 262144 | `cpu_fp64` | 0.1415 | 1.107 | 0.13× | 11 × 8 |
| `interp_bilinear_forward_backward` | 4×8×29×31 · 4×8×55×63 | `cpu_fp64` | 0.2435 | 0.3065 | 0.79× | 11 × 6 |
| `mlp_forward_backward` | 256×128 · 128×256 · 256×64 | `cpu_fp64` | 0.4793 | 1.665 | 0.29× | 11 × 4 |
| `svdvals_8x16x12` | 8×16×12 | `cpu_fp64` | 0.0464 | 4.433 | 0.01× | 11 × 4 |
| `pinv_8x16x12` | 8×16×12 | `cpu_fp64` | 0.1323 | 5.111 | 0.03× | 11 × 4 |
| `histc_1048576` | 1048576 | `cpu_fp64` | 0.2656 | 0.4039 | 0.66× | 11 × 8 |
| `igamma_65536` | 65536 · 65536 | `cpu_fp64` | 2.283 | 56.616 | 0.04× | 11 × 6 |

## 0.2.0 named workloads, round trip

Recorded in [v2-workloads.json](../results/v2-workloads.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `add_cpu_roundtrip_8388608` | 8388608 · 8388608 | `cpu_fp64` | 2.097 | 9.680 | 0.22× | 11 × 4 |

## 0.2.0 streaming and small GEMM

Recorded in [v2-bandwidth.json](../results/v2-bandwidth.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `add_8388608` | 8388608 · 8388608 | `cpu_add` | 1.861 | 0.4504 | 4.13× | 9 × 12 |
| `mm_128` | 128×128 · 128×128 | `mm_128_cpu_torch.float64` | 0.0171 | 0.1128 | 0.15× | 7 × 4 |
| `mm_256` | 256×256 · 256×256 | `mm_256_cpu_torch.float64` | 0.0823 | 0.3819 | 0.22× | 7 × 4 |

## Earlier public workload archive

Recorded in [speed-20261118.json](../results/speed-20261118.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `resident_add_4096` | 4096 · 4096 | `cpu_fp64` | 0.0012 | 0.0335 | 0.03× | 9 × 64 |
| `resident_add_262144` | 262144 · 262144 | `cpu_fp64` | 0.1163 | 0.0816 | 1.43× | 9 × 48 |
| `resident_add_1048576` | 1048576 · 1048576 | `cpu_fp64` | 0.3085 | 0.0824 | 3.75× | 9 × 24 |
| `resident_add_8388608` | 8388608 · 8388608 | `cpu_fp64` | 2.055 | 0.4936 | 4.16× | 9 × 12 |
| `host_to_host_add_8388608` | 8388608 · 8388608 | `cpu_fp64` | 2.150 | 10.912 | 0.20× | 9 × 1 |
| `resident_mul_8388608` | 8388608 · 8388608 | `cpu_fp64` | 1.916 | 0.5170 | 3.71× | 9 × 10 |
| `resident_div_8388608` | 8388608 · 8388608 | `cpu_fp64` | 2.107 | 0.5843 | 3.61× | 9 × 10 |
| `resident_sin_1048576` | 1048576 | `cpu_fp64` | 0.5489 | 2.903 | 0.19× | 9 × 6 |
| `resident_mm_128` | 128×128 · 128×128 | `cpu_fp64` | 0.0145 | 0.1532 | 0.09× | 9 × 12 |
| `resident_mm_256` | 256×256 · 256×256 | `cpu_fp64` | 0.0842 | 0.6818 | 0.12× | 9 × 8 |
| `resident_mm_512` | 512×512 · 512×512 | `cpu_fp64` | 0.5431 | 4.874 | 0.11× | 9 × 4 |
| `resident_mm_1024` | 1024×1024 · 1024×1024 | `cpu_fp64` | 4.408 | 36.668 | 0.12× | 9 × 2 |
| `two_layer_forward_backward` | 256×128 · 128×256 · 256×64 | `cpu_fp64` | 0.4472 | 11.888 | 0.04× | 9 × 3 |

## First-release benchmark archive

Recorded in [benchmark.json](../results/benchmark.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `add_8388608` | 8388608 · 8388608 | `cpu_add` | 1.889 | 0.4763 | 3.97× | 9 × 12 |
| `mm_128` | 128×128 · 128×128 | `mm_128_cpu_torch.float64` | 0.0204 | 0.1650 | 0.12× | 7 × 4 |
| `mm_256` | 256×256 · 256×256 | `mm_256_cpu_torch.float64` | 0.0847 | 0.6566 | 0.13× | 7 × 4 |

## Initial benchmark archive

Recorded in [benchmark-initial.json](../results/benchmark-initial.json).

| Case | Input shapes | Selected CPU path | CPU ms | GPU ms | CPU/GPU | Rounds × calls |
|---|---|---|---|---|---|---|
| `add_8388608` | 8388608 · 8388608 | `cpu_add` | 1.920 | 0.4699 | 4.09× | 9 × 12 |
| `mm_128` | 128×128 · 128×128 | `mm_128_cpu_torch.float64` | 0.0204 | 0.3002 | 0.07× | 7 × 4 |
| `mm_256` | 256×256 · 256×256 | `mm_256_cpu_torch.float64` | 0.0854 | 1.541 | 0.06× | 7 × 4 |

## Additional measured paths

The all-results CSV includes both allocating and reusable-output GPU paths when measured, as well as historical GPU versions that shared a CPU timing session. The tables above select one GPU path per workload, placement and submission mode. Alternative paths and repeated seeds reuse case identifiers. The per-path CSV preserves each recorded raw sample, including native MPS FP32 measurements where available. [All comparisons](all-results.csv) · [Per-path samples](timed-paths.csv)

The OpInfo archive contains GPU-to-GPU timing and result categories for every entry, including exclusions. The CPU generated the numerical references. [0.1.0 / 0.2.0 comparisons](opinfo-version-comparisons.csv)
