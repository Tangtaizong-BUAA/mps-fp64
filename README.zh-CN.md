# mps-fp64 中文说明

[English](README.md) · [交互式基准](https://tangtaizong-buaa.github.io/mps-fp64/) · [GPU 领先项目与完整对照](BENCHMARKS.md) · [下载发布版](https://github.com/Tangtaizong-BUAA/mps-fp64/releases/tag/v0.3.0.post1)

在 Apple GPU 上运行软件 IEEE 754 binary64 的 PyTorch eager 研究后端。已验证的环境为 40 核 M4 Max、macOS 15.7.1、PyTorch 2.11.0。张量的公开类型是 `torch.Tensor`，设备是 `mps:0`，数据类型是 `torch.float64`。GPU 内核用整数解释双精度位模式。

**0.3.0 是面向 M4 Max 的研究预览。** `torch.mm`、`torch.bmm` 和普通 `@` 会根据形状与实际输入选择内核。一般双精度方阵、能精确拆分的二进制定点方阵、短内维乘法及批量小矩阵各有同机 CPU 对照。[0.3.0 矩阵性能与数值范围](MATRIX_PERFORMANCE.md)

默认在支持范围外抛出异常。完整原生 MPS 功能对等、长时稳定性和全算子性能目标仍待验收。[0.2.0 全库验收](PERFORMANCE.md) · [0.2.0 逐项清单](results/v2-operator-coverage-and-speed.csv) · [任务图](ROADMAP.md)

## 在这台 Mac 上运行

在支持 MPS 的 Apple Silicon Mac 上，用虚拟环境的 Python 运行下列命令即可安装。公开版本为 0.3.0.post1，其 Python 与 Metal 运算源码逐字节对应受测的 0.3.0。

```bash
python -m pip install https://github.com/Tangtaizong-BUAA/mps-fp64/releases/download/v0.3.0.post1/mps_fp64-0.3.0.post1-py3-none-any.whl
```

从仓库运行时可使用 `python -m pip install -e '.[validation]'`。

```python
import torch
from mps_fp64 import enable

with enable() as backend:
    a = torch.tensor([[1., 1.], [1., 1. + 2**-40]],
                     device="mps", dtype=torch.float64)
    b = torch.tensor([2., 2. + 2**-40], device="mps", dtype=torch.float64)
    solution = torch.linalg.solve(a, b)
    print(solution.dtype, solution.device, solution.cpu())
    assert not backend.cpu_ops
```

该例的右下元素和 1 在 FP32 中会舍入到同一数值。双精度运行得到解 `[1., 1.]`。把张量的创建、运算、反向传播和拷回 CPU 放在同一个 `with enable()` 作用域内。启动新线程时在对应线程中启用。作用域外由已安装的标准 PyTorch 处理操作。模块复用 PyTorch 自动求导；某个前向算子可用时，其反向及二阶路径仍须分别验证。

## 当前实现

大矩阵使用 `torch.mm(a, b, out=c)` 或 `torch.bmm(a, b, out=c)` 可复用已分配的 FP64 输出。带独立二进制缩放的整数输入会经过抽检和全量检查。只有整数界保证所有部分和均可用 binary64 精确表示，才在 GPU 矩阵指令上分块计算，并用整数恢复最终的双精度值。检测失败时执行有序 binary64 FMA。随机满尾数输入、端到端传输及 CPU 预分配的计时方法见矩阵性能报告。

- **张量与调度**。通过同宽存储的 dtype 视图建立真正的 MPS 双精度张量。Python 调度层接收普通 torch 调用，执行 Metal 内核、可保留位模式的存储操作，或 PyTorch 自身的算子分解。
- **数值内核**。加减乘除、平方根、最近偶数舍入的融合乘加、比较和部分 dtype 转换；OpenLibm 移植提供指数、对数、三角与部分特殊函数。GEMM 按升序逐项执行 FMA。快路径保留判定最近偶数舍入所需的尾数和低位信息，特殊输入走完整 FMA。
- **张量运算**。支持常用广播、视图、归约、稳定排序、重复索引归约、分桶、直方图、前缀极值、唯一值与随机数。线性代数包括批矩阵乘、LU、三角求解、Householder、Cholesky、矩阵逆、实数 Jacobi SVD 和广义逆。SVD 的短边上限为 128，长边上限为 8192。[分解与梯度验证](results/jacobi-svd-20261118.json)
- **神经网络与频域**。支持部分线性层、激活、归一化、池化、卷积和梯度。FFT 使用双精度实虚部，二次幂采用 radix-2，长度超过 64 的非二次幂采用 Bluestein。含极大、极小或非有限值的行保留直接 DFT 路径。
- **CPU 对照模式**。`enable(cpu_fallback=True)` 允许部分返回新张量的未实现算子在 CPU 上重放，记录在 `backend.cpu_ops`。默认关闭。验证 GPU 覆盖时同时设置 `PYTORCH_ENABLE_MPS_FALLBACK=0`。计数器用于诊断包内重放；完整设备验证还需要系统级跟踪。

## 复现

先从工程根目录运行 `python tools/fetch_benchmark_baselines.py`，将带上游授权和校验值的公开基线下载到 `.benchmarks/`。本机重测文件可另取名，以便和仓库内的实测记录比较。

| 任务 | 从工程根目录运行 |
|---|---|
| 算术、梯度、求解、卷积、频域、并发回归 | `python -m unittest discover -s tests -v` |
| MPFR 数值对照 | `python tools/accuracy.py --report results/v2-mpfr.json` |
| 大矩阵、CPU Accelerate 与 0.2.0 比较 | `PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/benchmark_large_matrices.py .benchmarks/mps_fp64-0.2.0-1public-py3-none-any.whl results/v3-large-matrices.json --rounds 9` |
| 新矩阵内核的 Berkeley TestFloat 对照 | `python tools/check_matrix_arithmetic.py /path/to/metal-softfloat results/v3-matrix-arithmetic.json` |
| 正则化不完全 Gamma 的上下尾 | `python tools/check_igamma_accuracy.py results/igamma-accuracy.json` |
| 奇异值、重构残差、广义逆与一二阶梯度 | `python tools/check_svd.py --output results/jacobi-svd-20261118.json` |
| 可重复的迭代与分配器观测 | `python tools/stability.py --report results/v2-stability.json` |
| 驻留 GPU 的公开 API、匹配访存内核与矩阵乘基准 | `python tools/benchmark.py --output results/v2-bandwidth.json` |
| GEMM 分块调优与结果逐位校验 | `python tools/tune_gemm.py` |
| 原生注册表 | `python tools/inventory.py` |
| 逐条扩展 FP64 样本审计 | `PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/opinfo_audit.py --count 1000000 results/audit-v2-current-fp64.jsonl` |
| 原生 MPS FP32 同口径基线 | `PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/opinfo_audit.py --native --count 1000000 results/audit-v2-native-fp32.jsonl` |
| 汇总审计、数值与性能记录 | `python tools/summarize_acceleration.py` |

历史 0.2.0 性能对照在同一进程加载 0.1.0 和 0.2.0 两个 wheel。下面的命令可重新生成工作负载计时和 697 条目录记录。已测量的输入形状、每轮墙钟时间以及无法计时的原因均随结果保存。

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
python tools/compare_performance.py .benchmarks/mps_fp64-0.1.0-1public-py3-none-any.whl results/v2-workloads.json --candidate-wheel .benchmarks/mps_fp64-0.2.0-1public-py3-none-any.whl --rounds 11
python tools/compare_performance.py .benchmarks/mps_fp64-0.1.0-1public-py3-none-any.whl results/v2-catalog-timing.json --candidate-wheel .benchmarks/mps_fp64-0.2.0-1public-py3-none-any.whl --suite catalog --rounds 7 --batch 12
python tools/benchmark_gemm_arithmetic.py --output results/gemm-arithmetic.json
```

MPFR、OpInfo 和测试脚本的额外 Python 依赖在 `validation` 可选依赖组。Berkeley TestFloat 验证还使用 Xcode 的 clang 和 [固定版本的 metal-softfloat 源码](https://github.com/guyfischman/metal-softfloat/tree/8b6c592e2e383040fe2778bed8dda7904df284b1)。将该 checkout 路径传给 `python tools/testfloat.py /path/to/checkout --report results/v2-testfloat-level1.json`。Level 2 的测试命令额外指定 `--level 2 --ops add sub mul div sqrt`，并使用独立的报告文件。参考 C 库在临时目录编译；运行后端本身无需这些构建依赖。

原创代码按 MIT 开源。第三方代码保留各自的许可和归属。[MIT 许可](LICENSE) · [第三方来源与授权](THIRD_PARTY.md)
