"""Generate a version-matched native-vs-FP64 ledger and the human-readable report."""

import csv
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = root / "results"


def read(name):
    return json.loads((results / name).read_text())


def indexed(name):
    rows = [json.loads(line) for line in (results / name).read_text().splitlines()]
    data = {(r["op"], r["variant"]): r for r in rows}
    assert len(data) == len(rows)
    return data


native = indexed("opinfo-native-expanded.jsonl")
soft = indexed("opinfo-strict-expanded.jsonl")
assert native.keys() == soft.keys()
baseline = {k for k, r in native.items() if r["status"] == "pass"}
actual = {k for k, r in soft.items() if r["status"] == "pass"}
gaps = baseline - actual
with (results / "parity.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "operator",
            "variant",
            "native_float32",
            "strict_float64",
            "native_samples_passed",
            "fp64_samples_passed",
            "strict_detail",
        ]
    )
    for k in sorted(soft, key=lambda k: (k not in gaps, k)):
        writer.writerow(
            [
                *k,
                native[k]["status"],
                soft[k]["status"],
                native[k]["checked"],
                soft[k]["checked"],
                soft[k].get("detail", ""),
            ]
        )

versions = read("mps-registrations.json")
bench = read("benchmark.json")["rows"]
before = read("benchmark-initial.json")["rows"]
mpfr = read("mpfr.json")
level1 = read("testfloat-level1.json")
level2 = read("testfloat-level2.json")
stress = read("stability.json")
summary = Counter(v["status"] for v in soft.values())
failures = [
    (k, r)
    for k, r in soft.items()
    if r["status"] == "AssertionError" and not k[0].startswith("jiterator")
]
lines = [
    "# 0.1.0 历史验收",
    "",
    "这份记录保留首版口径，部分工厂样本产生了 CPU 输出。当前版本重新验证了 MPS 输出设备。[0.2.0 验收](PERFORMANCE.md)",
    "",
    "**当前结论：研究预览，尚未满足完整原生 MPS 对等与全工作负载性能目标。**",
    "",
    f"环境为 macOS 15.7.1、40 核 M4 Max、48 GB 统一内存、PyTorch {versions['torch']}，源码提交为 `{versions['commit']}`。",
    "",
    "## 算子覆盖",
    "",
    f"固定版本的 OpInfo 提供 {len(soft)} 个算子及变体条目。原生 MPS FP32 扩展审计通过 {len(baseline)} 个，软件 FP64 通过 {len(actual)} 个，两者同时通过 **{len(actual & baseline)}/{len(baseline)}**。原生基线中还有 **{len(gaps)} 个**未匹配。两轨各自以同 dtype 的 CPU 结果作参考。",
    "",
    f"FP64 扩展审计成功比较 {sum(r['checked'] for r in soft.values()):,} 个样本。每个条目遇首错停止；输出类型、随机测试、输入域和原生基线失败均单独记录。FP64 通过而原生 FP32 未通过的 {len(actual - baseline)} 个条目不抵扣缺项。",
    "",
    f"当前 PyTorch 注册了 {versions['aten_schema_count']} 个 ATen schema，其中 {versions['direct_mps_count']} 个有直接 MPS kernel。直接注册数包含整型和 out 变体，组合算子也可在 MPS 运行。OpInfo 和注册表是两个不同口径。",
    "",
    "严格路径关闭了本包 CPU 重放和 PyTorch 环境回退。OpInfo 检查采用 FP64 的 rtol=1e-10、atol=1e-12 及 FP32 的 rtol=3e-4、atol=3e-5；这些阈值用于发现功能回归。逐位与 ULP 证据另见下表。审计选用 CPU 生成的稠密实数样本，每个张量限制在 2²⁰ 个元素以内。",
    "",
    "[逐项差异清单](results/parity.csv) · [FP64 原始记录](results/opinfo-strict-expanded.jsonl) · [原生 FP32 原始记录](results/opinfo-native-expanded.jsonl) · [直接注册清单](results/mps-registrations.json)",
    "",
    "## 精度与稳定性",
    "",
    "| 检查 | 结果 |",
    "|---|---|",
    f"| Berkeley TestFloat 最近偶数舍入，Level 1 | 加减乘除、平方根与 FMA 共 {sum(x['cases'] for x in level1.values()):,} 例，{sum(x['mismatches'] for x in level1.values())} 个不匹配 |",
    f"| Berkeley TestFloat 最近偶数舍入，Level 2 | 加减乘除与平方根共 {sum(x['cases'] for x in level2.values()):,} 例，{sum(x['mismatches'] for x in level2.values())} 个不匹配；FMA 的 Level 2 未执行 |",
    f"| MPFR | {len(mpfr)} 个函数各 {next(iter(mpfr.values()))['samples']} 个输入，样本最大误差 {max(o['max_ulp'] for o in mpfr.values())} ULP，有符号零差异共 {sum(o['different_zero_sign'] for o in mpfr.values())} 个 |",
    "| 计算图 | 一阶及二阶有限差分、卷积与池化反传、FFT 反传、LU 与 Cholesky 检查见回归日志 |",
    f"| 重复执行 | {stress['elements']:,} 个元素做 {stress['iterations']:,} 轮可逆加减，每 256 轮核对所有输出位；首末快照的张量分配为 {stress['samples'][0]['allocated']:,} 和 {stress['samples'][-1]['allocated']:,} 字节 |",
    "",
    "TestFloat 将任意 NaN 载荷判作同类，验证输出值。浮点异常标志、长时间训练、跨系统、全部参数组合仍待验收。MPFR 样本通过两档精度收敛检查，ULP 的经验最大值只适用于这些输入。",
    "",
    "[Level 1](results/testfloat-level1.json) · [Level 2](results/testfloat-level2.json) · [MPFR](results/mpfr.json) · [回归](results/regression.txt) · [重复执行](results/stability.json)",
    "",
    "## 性能",
    "",
    "输入驻留设备，先预热再计时，每批末尾同步。流式算子每元素按两读一写计 24 字节。数值是多批墙钟中位数，包含 Python 提交，排除编译和 CPU↔GPU 传输。原始记录包含每批耗时。",
    "",
    "| 8,388,608 个元素 | 有效 GB/s | 占同机匹配访存内核 |",
    "|---|---:|---:|",
]
for name, label in [
    ("matched_xor", "同流量整数 XOR"),
    ("raw_add", "预分配的 Metal 加法"),
    ("add", "torch.add"),
    ("mul", "torch.mul"),
    ("div", "torch.div"),
]:
    rate = bench[name]["GB_s"]
    lines.append(
        f"| {label} | {rate:.1f} | {rate / bench['matched_xor']['GB_s']:.1%} |"
    )
g = bench["mm_256_mps_torch.float64"]["GFLOP_s"]
old = before["mm_256_mps_torch.float64"]["GFLOP_s"]
cpu = bench["mm_256_cpu_torch.float64"]["GFLOP_s"]
lines += [
    "",
    f"256×256 矩阵乘按 2N³ 计，优化前 {old:.2f} GFLOP/s，当前 {g:.2f} GFLOP/s，约 {g / old:.2f} 倍。同次 CPU FP64 为 {cpu:.2f} GFLOP/s。未给出软件 FP64 GEMM 的理论算力上界。",
    "",
    "匹配访存内核是本机实测参照。Apple 对此芯片给出的统一内存标称带宽为 546 GB/s。有效字节数不是物理 DRAM 计数，缓存、系统负载与热状态均会影响比值。不同运算的可达上限需各自建模。[Apple 技术规格](https://support.apple.com/en-la/121554)",
    "",
    "[初版计时](results/benchmark-initial.json) · [当前计时](results/benchmark.json) · [分块与融合乘加试验](results/gemm-tiles.json)",
    "",
    "## 发布缺口",
    "",
    "缺口清单按原生基线已通过、FP64 尚未通过优先排序。原生 C++ 调度、torch.compile、超大张量、多系统兼容以及未验收的原地/out/别名语义仍是发布条件。",
]
for k, r in failures:
    lines += [
        "",
        f"- `{k[0]}` 的扩展样本在 {r['checked']} 个成功样本后出现数值或语义差异。详情见差异清单。",
    ]
if any(k[0] == "cov" for k, _ in failures):
    lines += [
        "",
        "已复现的混合精度协方差样本中，输入为 FP64，权重为 FP32。三项权重平方和在 CPU 上为 0.628727912902832，在原生 MPS FP32 归约上为 0.628727853298187。该差异进入归一化分母，最终协方差绝对误差约 1.11e-6，保留为扩展审计失败项。",
    ]
(root / "ACCEPTANCE.md").write_text("\n".join(lines) + "\n")
print("overlap", len(actual & baseline), len(baseline), "open", len(gaps), summary)
