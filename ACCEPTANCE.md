# 0.1.0 历史验收

这份记录保留了首版的测试方法和数据。部分工厂样本在 CPU 上产生输出。0.2.0 的覆盖对照要求张量输出位于 MPS，并重测了旧版和原生 FP32。[当前验收与性能](PERFORMANCE.md)

**当前结论：研究预览，尚未满足完整原生 MPS 对等与全工作负载性能目标。**

环境为 macOS 15.7.1、40 核 M4 Max、48 GB 统一内存、PyTorch 2.11.0，源码提交为 `70d99e998b4955e0049d13a98d77ae1b14db1f45`。

## 算子覆盖

固定版本的 OpInfo 提供 697 个算子及变体条目。原生 MPS FP32 扩展审计通过 564 个，软件 FP64 通过 475 个，两者同时通过 **451/564**。原生基线中还有 **113 个**未匹配。两轨各自以同 dtype 的 CPU 结果作参考。

FP64 扩展审计成功比较 12,875 个样本。每个条目遇首错停止；输出类型、随机测试、输入域和原生基线失败均单独记录。FP64 通过而原生 FP32 未通过的 24 个条目不抵扣缺项。

当前 PyTorch 注册了 3203 个 ATen schema，其中 874 个有直接 MPS kernel。直接注册数包含整型和 out 变体，组合算子也可在 MPS 运行。OpInfo 和注册表是两个不同口径。

严格路径关闭了本包 CPU 重放和 PyTorch 环境回退。OpInfo 检查采用 FP64 的 rtol=1e-10、atol=1e-12 及 FP32 的 rtol=3e-4、atol=3e-5；这些阈值用于发现功能回归。逐位与 ULP 证据另见下表。审计选用 CPU 生成的稠密实数样本，每个张量限制在 2²⁰ 个元素以内。

[逐项差异清单](results/parity.csv) · [FP64 原始记录](results/opinfo-strict-expanded.jsonl) · [原生 FP32 原始记录](results/opinfo-native-expanded.jsonl) · [直接注册清单](results/mps-registrations.json)

## 精度与稳定性

| 检查 | 结果 |
|---|---|
| Berkeley TestFloat 最近偶数舍入，Level 1 | 加减乘除、平方根与 FMA 共 6,319,872 例，0 个不匹配 |
| Berkeley TestFloat 最近偶数舍入，Level 2 | 加减乘除与平方根共 161,163,264 例，0 个不匹配；FMA 的 Level 2 未执行 |
| MPFR | 22 个函数各 2573 个输入，样本最大误差 2 ULP，有符号零差异共 0 个 |
| 计算图 | 一阶及二阶有限差分、卷积与池化反传、FFT 反传、LU 与 Cholesky 检查见回归日志 |
| 重复执行 | 65,536 个元素做 4,096 轮可逆加减，每 256 轮核对所有输出位；首末快照的张量分配为 524,544 和 524,544 字节 |

TestFloat 将任意 NaN 载荷判作同类，验证输出值。浮点异常标志、长时间训练、跨系统、全部参数组合仍待验收。MPFR 样本通过两档精度收敛检查，ULP 的经验最大值只适用于这些输入。

[Level 1](results/testfloat-level1.json) · [Level 2](results/testfloat-level2.json) · [MPFR](results/mpfr.json) · [回归](results/regression.txt) · [重复执行](results/stability.json)

## 性能

输入驻留设备，先预热再计时，每批末尾同步。流式算子每元素按两读一写计 24 字节。数值是多批墙钟中位数，包含 Python 提交，排除编译和 CPU↔GPU 传输。原始记录包含每批耗时。

| 8,388,608 个元素 | 有效 GB/s | 占同机匹配访存内核 |
|---|---:|---:|
| 同流量整数 XOR | 443.0 | 100.0% |
| 预分配的 Metal 加法 | 458.8 | 103.6% |
| torch.add | 422.7 | 95.4% |
| torch.mul | 420.5 | 94.9% |
| torch.div | 430.9 | 97.3% |

256×256 矩阵乘按 2N³ 计，优化前 21.77 GFLOP/s，当前 51.10 GFLOP/s，约 2.35 倍。同次 CPU FP64 为 396.12 GFLOP/s。未给出软件 FP64 GEMM 的理论算力上界。

匹配访存内核是本机实测参照。Apple 对此芯片给出的统一内存标称带宽为 546 GB/s。有效字节数不是物理 DRAM 计数，缓存、系统负载与热状态均会影响比值。不同运算的可达上限需各自建模。[Apple 技术规格](https://support.apple.com/en-la/121554)

[初版计时](results/benchmark-initial.json) · [当前计时](results/benchmark.json) · [分块与融合乘加试验](results/gemm-tiles.json)

## 发布缺口

缺口清单按原生基线已通过、FP64 尚未通过优先排序。原生 C++ 调度、torch.compile、超大张量、多系统兼容以及未验收的原地/out/别名语义仍是发布条件。

- `cov` 的扩展样本在 33 个成功样本后出现数值或语义差异。详情见差异清单。

已复现的混合精度协方差样本中，输入为 FP64，权重为 FP32。三项权重平方和在 CPU 上为 0.628727912902832，在原生 MPS FP32 归约上为 0.628727853298187。该差异进入归一化分母，最终协方差绝对误差约 1.11e-6，保留为扩展审计失败项。
