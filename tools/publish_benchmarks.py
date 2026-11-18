"""Build the public CPU/GPU benchmark catalog from the distributed raw JSON."""

import argparse
from collections import Counter
import csv
from functools import lru_cache
from dataclasses import dataclass
import io
import json
from math import isclose
from pathlib import Path
import re
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[1]
MAIN = "v3-large-matrices.json"
SECOND = "v3-matrix-second-seed.json"
MATCHED = "v3-matrix-optimized-cpu.json"
NAMED = "v2-workloads.json"
TENSOR = "public-tensor-ops.json"
ORDERING = "public-ordering.json"
CATALOG = "v2-catalog-timing.json"
RECHECK = "v2-paired-recheck.json"


@lru_cache(maxsize=None)
def read(name):
    return json.loads((ROOT / "results" / name).read_text())


def compact(obj):
    return json.dumps(obj, separators=(",", ":"))


def show_shape(shapes):
    return " · ".join("×".join(map(str, shape)) or "scalar" for shape in shapes)


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


@dataclass(frozen=True)
class Series:
    values: list
    mid: float
    q1: float
    q3: float

    @classmethod
    def parse(cls, path):
        samples = path["samples_ms"]
        if not samples or any(t <= 0 for t in samples):
            raise ValueError(f"Invalid wall times: {samples}")
        mid = median(samples)
        q1, _, q3 = (
            quantiles(samples, n=4, method="inclusive")
            if len(samples) > 1
            else samples * 3
        )
        for key, value in [("median_ms", mid), ("q1_ms", q1), ("q3_ms", q3)]:
            if key in path and not isclose(path[key], value, rel_tol=1e-12):
                raise ValueError(f"Archived {key}={path[key]} differs from {value}")
        return cls(samples, mid, q1, q3)


class Catalog:
    def __init__(self):
        self.results = []
        self.paths = []
        self.opinfo = []

    def paths_for(self, source, case, experiment, calls, methods, rounds=None):
        paths = {}
        for name, raw in methods.items():
            if "samples_ms" not in raw:
                continue
            series = Series.parse(raw)
            count = len(series.values)
            if rounds is not None and count != rounds:
                raise ValueError(f"Unexpected round count for {source}/{case}/{name}")
            paths[name] = series
            self.paths.append(
                dict(
                    source=source,
                    case=case,
                    experiment=experiment,
                    path=name,
                    calls_per_round=calls,
                    rounds=count,
                    median_ms=series.mid,
                    q1_ms=series.q1,
                    q3_ms=series.q3,
                    samples_ms=compact(series.values),
                )
            )
        return paths

    def compare(
        self,
        source,
        case,
        version,
        role,
        placement,
        calls,
        shapes,
        seed,
        paths,
        gpu_path,
        cpus,
        baseline,
        *,
        strides=None,
        cpu_threads=None,
        input_kind="",
        parameters=None,
        matched_cpu=None,
        reuse_cpu=None,
        validation="",
        reported_ratio=None,
        interleaved=True,
    ):
        cpu_path = min(cpus, key=lambda key: paths[key].mid)
        cpu, gpu = paths[cpu_path], paths[gpu_path]
        ratio = cpu.mid / gpu.mid
        if reported_ratio is not None and not isclose(
            ratio, reported_ratio, rel_tol=1e-12
        ):
            raise ValueError(
                f"Cannot reproduce source ratio for {source}/{case}/{gpu_path}"
            )
        if interleaved and len(cpu.values) != len(gpu.values):
            raise ValueError(f"Unpaired round counts for {source}/{case}")
        mode = "single" if calls == 1 else "batch-per-call"
        self.results.append(
            dict(
                source=source,
                case=case,
                version=version,
                role=role,
                timing=mode,
                placement=placement,
                calls_per_round=calls,
                input_shapes=compact(shapes),
                input_strides=compact(strides) if strides is not None else "",
                input_kind=input_kind,
                parameters=compact(parameters) if parameters is not None else "",
                seed=seed,
                cpu_baseline=baseline,
                cpu_threads=(
                    cpu_threads
                    if cpu_threads is not None
                    else read(source).get(
                        "cpu_threads", read(source).get("threads", "")
                    )
                ),
                cpu_path=cpu_path,
                gpu_path=gpu_path,
                cpu_median_ms=cpu.mid,
                gpu_median_ms=gpu.mid,
                cpu_over_gpu=ratio,
                winner=(
                    "GPU"
                    if cpu.mid > gpu.mid
                    else "CPU" if cpu.mid < gpu.mid else "tie"
                ),
                cpu_q1_ms=cpu.q1,
                cpu_q3_ms=cpu.q3,
                gpu_q1_ms=gpu.q1,
                gpu_q3_ms=gpu.q3,
                cpu_rounds=len(cpu.values),
                gpu_rounds=len(gpu.values),
                gpu_faster_paired_rounds=(
                    sum(g < c for g, c in zip(gpu.values, cpu.values))
                    if interleaved
                    else ""
                ),
                matched_cpu_path=matched_cpu or "",
                matched_cpu_over_gpu=(
                    paths[matched_cpu].mid / gpu.mid if matched_cpu else ""
                ),
                reusable_cpu_path=reuse_cpu or "",
                reusable_cpu_over_gpu=(
                    paths[reuse_cpu].mid / gpu.mid if reuse_cpu else ""
                ),
                validation=validation,
            )
        )

    def matrices(self, source):
        doc = read(source)
        for item in doc["cases"]:
            c = item["case"]
            name, shapes, seed = c["name"], item["input_shapes"], item["seed"]
            v = item["verification"]
            validation = f"{v['checked_output_words']} words, {v['differences_from_020']} mismatches vs 0.2.0"
            if c["amplitude"]:
                validation += f", bit-exact vs CPU FP64, {v['cpu_outputs_unrepresentable_in_fp32']} not exactly representable in FP32"
            for experiment, calls in [("latency", 1), ("queued", c["throughput"])]:
                if not item[experiment]:
                    continue
                paths = self.paths_for(
                    source, name, experiment, calls, item[experiment], doc["rounds"]
                )
                cpus = [k for k in paths if k.startswith(("cpu", "accelerate"))]
                primary = "mps_030" if c["vector"] else "mps_030_out"
                for path in (p for p in paths if p.startswith("mps_")):
                    transfer = path == "mps_030_roundtrip"
                    version = "0.2.0" if path == "mps_020" else "0.3.0"
                    role = (
                        "primary"
                        if transfer or path == primary
                        else "historical" if path == "mps_020" else "functional"
                    )
                    reported = (
                        item.get(
                            "queued_cpu_over_gpu"
                            if experiment == "queued"
                            else "cpu_over_gpu"
                        )
                        if path == primary
                        else None
                    )
                    self.compare(
                        source,
                        name,
                        version,
                        role,
                        "CPU round trip" if transfer else "resident",
                        calls,
                        shapes,
                        seed,
                        paths,
                        path,
                        cpus,
                        "fastest measured native FP64 CPU",
                        matched_cpu="cpu_out" if path.endswith("_out") else "cpu",
                        reuse_cpu="cpu_out",
                        strides=item["input_strides"],
                        input_kind=item["input"],
                        parameters={
                            "amplitude": c["amplitude"],
                            "transposed": c["transposed"],
                            "vector": c["vector"],
                        },
                        reported_ratio=reported,
                        validation=validation,
                    )

    def structure_aware(self):
        doc = read(MATCHED)
        main = {v["case"]["name"]: v for v in read(MAIN)["cases"]}
        for case in doc["cases"]:
            name = case["case"]
            paths = self.paths_for(
                MATCHED, name, "latency", 1, case["timing"], doc["rounds"]
            )
            self.compare(
                MATCHED,
                name,
                "0.3.0",
                "primary",
                "resident",
                1,
                main[name]["input_shapes"],
                case["seed"],
                paths,
                "gpu_from_fp64_out",
                ["cpu_prepacked_sgemm", "cpu_fp64_out"],
                "fastest of prepacked exact SGEMM and native FP64",
                strides=main[name]["input_strides"],
                input_kind=main[name]["input"],
                matched_cpu="cpu_fp64_out",
                reuse_cpu="cpu_prepacked_sgemm",
                reported_ratio=case["best_cpu_over_gpu"],
                validation=f"{case['checked_output_words']} words, {case['cpu_dgemm_vs_prepacked_vs_gpu_mismatches']} mismatches across all three paths",
            )

    def workloads(self):
        doc = read(NAMED)
        for name, case in doc["cases"].items():
            calls = case["batch"]
            paths = self.paths_for(
                NAMED, name, "public workload", calls, case["paths"], doc["rounds"]
            )
            if "cpu_fp64" not in paths:
                continue
            for path, version in [("new_fp64", "0.2.0"), ("old_fp64", "0.1.0")]:
                if path in paths:
                    self.compare(
                        NAMED,
                        name,
                        version,
                        "primary" if path == "new_fp64" else "historical",
                        (
                            "CPU round trip"
                            if name == "add_cpu_roundtrip_8388608"
                            else "resident"
                        ),
                        calls,
                        case["shapes"],
                        "",
                        paths,
                        path,
                        ["cpu_fp64"],
                        "functional CPU FP64",
                        matched_cpu="cpu_fp64",
                        validation="full output checked against CPU",
                    )

    def tensor_ops(self):
        doc = read(TENSOR)
        for case in doc["measurements"]:
            if case["status"] != "pass":
                continue
            name = case["id"]
            paths = self.paths_for(
                TENSOR, name, "latency", 1, case["methods"], doc["rounds"]
            )
            comp = case["comparison"]
            cpus = [k for k in paths if k.startswith("cpu_")]
            for g in (k for k in paths if k.startswith("mps_")):
                primary_path = g == comp["matched_mps_method"]
                matched = (
                    comp["matched_cpu_method"] if primary_path else "cpu_functional"
                )
                reuse = comp["cpu_reuse_method"]
                if primary_path and (
                    not isclose(
                        paths[matched].mid / paths[g].mid, comp["matched_cpu_over_mps"]
                    )
                    or not isclose(
                        paths[reuse].mid / paths[g].mid, comp["cpu_reuse_over_mps"]
                    )
                ):
                    raise ValueError(
                        f"Cannot reproduce allocation comparison for {name}"
                    )
                check = case["validation"][g]
                self.compare(
                    TENSOR,
                    name,
                    "0.3.0",
                    "primary" if primary_path else "functional",
                    "resident",
                    1,
                    [i["shape"] for i in case["inputs"]],
                    case["input_seed"],
                    paths,
                    g,
                    cpus,
                    "fastest measured native FP64 CPU",
                    strides=[i["stride"] for i in case["inputs"]],
                    input_kind=(
                        "Gaussian FP64 with a Boolean condition"
                        if case["operator"] == "torch.where"
                        else "Gaussian FP64"
                    ),
                    parameters=case["kwargs"],
                    matched_cpu=matched,
                    reuse_cpu=reuse,
                    validation=f"{check['elements']} output values, {check['criterion']}",
                )

    def ordering(self):
        doc = read(ORDERING)
        for case in doc["measurements"]:
            name, methods = case["id"], case["methods"]
            paths = self.paths_for(ORDERING, name, "latency", 1, methods, doc["rounds"])
            cpus = [k for k in paths if k.startswith("cpu_")]
            best = min(cpus, key=lambda k: paths[k].mid)
            best_out = min(
                (k for k in cpus if k.endswith("_out")),
                key=lambda k: paths[k].mid,
                default=None,
            )
            c = case["comparison"]
            if best != c["cpu_method"]:
                raise ValueError(f"Cannot reproduce the tuned CPU method for {name}")
            for gpu in (k for k in paths if k.startswith("mps_")):
                suffix = gpu.rsplit("_", 1)[-1]
                matched = min(
                    (k for k in cpus if k.endswith("_" + suffix)),
                    key=lambda k: paths[k].mid,
                    default=None,
                )
                self.compare(
                    ORDERING,
                    name,
                    "0.3.0",
                    "primary" if gpu == c["mps_method"] else "additional",
                    "resident",
                    1,
                    [case["shape"]],
                    case["input_seed"],
                    paths,
                    gpu,
                    cpus,
                    "fastest CPU thread count and output allocation measured",
                    strides=[[case["shape"][1], 1]],
                    input_kind=doc["input"],
                    cpu_threads=methods[best]["threads"],
                    parameters=case["kwargs"],
                    matched_cpu=matched,
                    reuse_cpu=best_out,
                    validation=f"{methods[gpu]['validation']['value_words']} value words and {methods[gpu]['validation']['indices']} indices checked, {methods[gpu]['validation']['mismatches']} mismatches",
                    reported_ratio=c["speedup"] if gpu == c["mps_method"] else None,
                    interleaved=False,
                )

    def earlier(self):
        source = "speed-20261118.json"
        for name, case in read(source)["rows"].items():
            calls = case["calls_per_batch"]
            paths = self.paths_for(
                source, name, "public workload", calls, case["paths"], case["rounds"]
            )
            if name == "two_layer_forward_backward":
                shapes = [[256, 128], [128, 256], [256, 64]]
            else:
                n = int(name.rsplit("_", 1)[1])
                shapes = (
                    [[n, n], [n, n]]
                    if "_mm_" in name
                    else [[n]] if "_sin_" in name else [[n], [n]]
                )
            self.compare(
                source,
                name,
                "unversioned archive",
                "primary",
                "CPU round trip" if name.startswith("host_to_host") else "resident",
                calls,
                shapes,
                "",
                paths,
                "mps_fp64",
                ["cpu_fp64"],
                "functional CPU FP64",
                matched_cpu="cpu_fp64",
                validation="first 16 elements of each output checked against CPU",
            )
        for source in ["v2-bandwidth.json", "benchmark.json", "benchmark-initial.json"]:
            data = read(source)["rows"]
            version = "0.2.0" if source.startswith("v2") else "unversioned archive"
            method_sets = [
                (
                    "add_8388608",
                    [[8388608], [8388608]],
                    12,
                    {
                        k: data[k]
                        for k in [
                            "matched_xor",
                            "raw_add",
                            "add",
                            "mul",
                            "div",
                            "cpu_add",
                        ]
                    },
                    "add",
                    "cpu_add",
                ),
            ]
            for size in (128, 256):
                prefix = f"mm_{size}_"
                method_sets.append(
                    (
                        f"mm_{size}",
                        [[size, size], [size, size]],
                        4,
                        {k: data[k] for k in data if k.startswith(prefix)},
                        prefix + "mps_torch.float64",
                        prefix + "cpu_torch.float64",
                    )
                )
            for name, shapes, calls, methods, g, c in method_sets:
                paths = self.paths_for(
                    source,
                    name,
                    "streaming suite" if g == "add" else "GEMM",
                    calls,
                    methods,
                )
                self.compare(
                    source,
                    name,
                    version,
                    "primary",
                    "resident",
                    calls,
                    shapes,
                    "",
                    paths,
                    g,
                    [c],
                    "functional CPU FP64",
                    matched_cpu=c,
                    interleaved=False,
                    validation="timing-only harness",
                )

    def operators(self, source):
        doc = read(source)
        for case in doc["cases"]:
            ids, name = (case["op"], case["variant"]), case["op"]
            paths = case.get("paths", {})
            methods = {}
            for path in ("old_fp64", "new_fp64"):
                meta = paths.get(path, {})
                s = Series.parse(meta) if meta.get("status") == "pass" else None
                if s and len(s.values) != doc["rounds"]:
                    raise ValueError(f"Unexpected OpInfo round count for {name}")
                methods[path] = (s, meta)
            (a, ma), (b, mb) = methods.values()
            category = mb.get("category", "")
            if not a or not b:
                status = "unpaired"
            elif category != "materialized" or ma.get("category") != "materialized":
                status = "alias, empty or host result"
            elif a.mid >= 1.10 * b.mid and a.mid - b.mid >= 0.002:
                status = "faster"
            elif b.mid >= 1.10 * a.mid and b.mid - a.mid >= 0.002:
                status = "slower"
            else:
                status = "within threshold"
            self.opinfo.append(
                dict(
                    source=source,
                    operator=ids[0],
                    variant=ids[1],
                    sample_index=case.get("sample_index", ""),
                    input_shapes=compact(case.get("shapes", [])),
                    calls_per_round=doc["catalog_batch"],
                    old_status=ma.get("status", "excluded"),
                    new_status=mb.get("status", "excluded"),
                    new_output_kind=category,
                    old_rounds=len(a.values) if a else 0,
                    new_rounds=len(b.values) if b else 0,
                    gpu_010_median_ms=a.mid if a else "",
                    gpu_020_median_ms=b.mid if b else "",
                    gpu_010_q1_ms=a.q1 if a else "",
                    gpu_010_q3_ms=a.q3 if a else "",
                    gpu_020_q1_ms=b.q1 if b else "",
                    gpu_020_q3_ms=b.q3 if b else "",
                    gpu_010_over_gpu_020=a.mid / b.mid if a and b else "",
                    timing_class=status,
                )
            )


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def ms(v):
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def factor(v):
    return f"{v:.2f}×"


def primary(catalog, source, placement=None, timing=None):
    return [
        r
        for r in catalog.results
        if r["source"] == source
        and r["role"] == "primary"
        and (not placement or r["placement"] == placement)
        and (not timing or r["timing"] == timing)
    ]


def sections(catalog):
    specs = [
        ("0.3.0 matrix latency", MAIN, "resident", "single"),
        ("0.3.0 matrix round trip", MAIN, "CPU round trip", "single"),
        ("0.3.0 matrix queue throughput", MAIN, "resident", "batch-per-call"),
        ("0.3.0 matrix latency, second seed", SECOND, "resident", "single"),
        ("0.3.0 matrix round trip, second seed", SECOND, "CPU round trip", "single"),
        (
            "0.3.0 matrix queue throughput, second seed",
            SECOND,
            "resident",
            "batch-per-call",
        ),
        ("0.3.0 structure-aware CPU comparison", MATCHED, "resident", "single"),
        ("0.3.0 tensor latency", TENSOR, "resident", "single"),
        ("0.3.0 ordering with tuned CPU threads", ORDERING, "resident", "single"),
        ("0.2.0 named workloads, resident", NAMED, "resident", None),
        ("0.2.0 named workloads, round trip", NAMED, "CPU round trip", None),
        ("0.2.0 streaming and small GEMM", "v2-bandwidth.json", None, None),
        ("Earlier public workload archive", "speed-20261118.json", None, None),
        ("First-release benchmark archive", "benchmark.json", None, None),
        ("Initial benchmark archive", "benchmark-initial.json", None, None),
    ]
    return [
        (title, src, selected)
        for title, src, place, mode in specs
        if (selected := primary(catalog, src, place, mode))
    ]


def table_rows(rows):
    return [
        (
            f"`{r['case']}`",
            show_shape(json.loads(r["input_shapes"])),
            f"`{r['cpu_path']}`",
            ms(r["cpu_median_ms"]),
            ms(r["gpu_median_ms"]),
            factor(r["cpu_over_gpu"]),
            f"{r['gpu_rounds']} × {r['calls_per_round']}",
        )
        for r in rows
    ]


def details(catalog):
    content = [
        "# Benchmark tables",
        "Times are median wall-clock milliseconds per call. **CPU/GPU** is the CPU median divided by the GPU median. A ratio above one means the GPU finished sooner. A round with multiple calls measures their average. Times and ratios are rounded for display. The downloads retain full numeric precision. [Benchmark guide](../BENCHMARKS.md) · [All comparisons](all-results.csv) · [GPU-ahead comparisons](gpu-ahead.csv) · [Per-path samples](timed-paths.csv)",
    ]
    for title, source, rows in sections(catalog):
        if source == MATCHED:
            timings = {c["case"]: c["timing"] for c in read(MATCHED)["cases"]}
            display = table(
                [
                    "Case",
                    "Native FP64 CPU ms",
                    "Prepacked CPU ms",
                    "GPU from FP64 ms",
                    "Fastest CPU/GPU",
                    "Rounds × calls",
                ],
                [
                    (
                        f"`{r['case']}`",
                        ms(timings[r["case"]]["cpu_fp64_out"]["median_ms"]),
                        ms(timings[r["case"]]["cpu_prepacked_sgemm"]["median_ms"]),
                        ms(r["gpu_median_ms"]),
                        factor(r["cpu_over_gpu"]),
                        f"{r['gpu_rounds']} × {r['calls_per_round']}",
                    )
                    for r in rows
                ],
            )
        else:
            display = table(
                [
                    "Case",
                    "Input shapes",
                    "Selected CPU path",
                    "CPU ms",
                    "GPU ms",
                    "CPU/GPU",
                    "Rounds × calls",
                ],
                table_rows(rows),
            )
        content += [
            f"## {title}",
            f"Recorded in [{source}](../results/{source}).",
            display,
        ]
    content += [
        "## Additional measured paths",
        "The all-results CSV includes both allocating and reusable-output GPU paths when measured, as well as historical GPU versions that shared a CPU timing session. The tables above select one GPU path per workload, placement and submission mode. Alternative paths and repeated seeds reuse case identifiers. The per-path CSV preserves each recorded raw sample, including native MPS FP32 measurements where available. [All comparisons](all-results.csv) · [Per-path samples](timed-paths.csv)",
        "The OpInfo archive contains GPU-to-GPU timing and result categories for every entry, including exclusions. The CPU generated the numerical references. [0.1.0 / 0.2.0 comparisons](opinfo-version-comparisons.csv)",
    ]
    return "\n\n".join(content) + "\n"


def overview(catalog):
    raw = read(MAIN)
    main = primary(catalog, MAIN, "resident", "single")
    matched = primary(catalog, MATCHED)
    by_name = {r["case"]: r for r in main}
    stricter = {r["case"]: r for r in matched}
    best = by_name["dyadic_4_4096"]
    full = by_name["normal_square_4096"]
    mm = {v["case"]["name"]: v for v in raw["cases"]}
    v = mm["dyadic_16_4096"]["verification"]
    counts = []
    for title, source, rows in sections(catalog):
        outcome = Counter(r["winner"] for r in rows)
        rounds = ", ".join(map(str, sorted({r["gpu_rounds"] for r in rows})))
        calls = sorted({r["calls_per_round"] for r in rows})
        counts.append(
            (
                title,
                len(rows),
                outcome["GPU"],
                outcome["CPU"],
                outcome["tie"],
                rounds,
                str(calls[0]) if len(calls) == 1 else f"{calls[0]}–{calls[-1]}",
            )
        )
    s = [
        "# Benchmarks on Apple M4 Max",
        f"On a 40-core M4 Max, release 0.3.0 multiplied two resident 4,096 × 4,096 FP64 matrices with scaled integer coefficients in ±15 in **{ms(best['gpu_median_ms'])} ms**. The fastest measured native FP64 CPU took **{ms(best['cpu_median_ms'])} ms**, an **{factor(best['cpu_over_gpu'])}** GPU advantage. With independent Gaussian FP64 values at the same size, the CPU took {ms(full['cpu_median_ms'])} ms and the GPU took {ms(full['gpu_median_ms'])} ms. Both timings include the wait for one completed result. The public 0.3.0.post1 distribution packages the measured 0.3.0 runtime. [Raw matrix measurements](results/v3-large-matrices.json)",
        "[All CPU/GPU comparisons (CSV)](benchmarks/all-results.csv) · [GPU ahead of the fastest measured CPU (CSV)](benchmarks/gpu-ahead.csv) · [GPU ahead at matched allocation (CSV)](benchmarks/matched-allocation-ahead.csv) · [Sample times by path (CSV)](benchmarks/timed-paths.csv) · [Complete tables](benchmarks/tables.md)",
        "## When dense matrix multiplication wins",
        "For the dyadic inputs, one operand is an integer matrix scaled by 2⁻¹³ and the other is scaled by 2⁻⁹. The GPU checks every value on each call. When the products and partial sums meet the exactness bounds, it uses native matrix instructions and reconstructs the binary64 output. The main run timed preallocated GPU output against the fastest measured CPU implementation on the same inputs. A separate experiment gave a specialized CPU implementation prepacked digits, known scales and reusable scratch space for free.",
        "![GPU speed relative to CPU for 4,096-square Gaussian and dyadic FP64 matrices. Ordinary Gaussian inputs favor CPU. Dyadic matrices favor GPU against both CPU baselines.](benchmarks/matrix-speedups.svg)",
        table(
            [
                "4,096-square inputs",
                "Native CPU ms",
                "GPU ms",
                "Native CPU/GPU",
                "Best CPU/GPU in separate structure-aware run",
            ],
            [
                (
                    label,
                    ms(r["cpu_median_ms"]),
                    ms(r["gpu_median_ms"]),
                    factor(r["cpu_over_gpu"]),
                    factor(stricter[key]["cpu_over_gpu"]) if key in stricter else "—",
                )
                for key, label in [
                    ("normal_square_4096", "Gaussian FP64"),
                    ("dyadic_4_4096", "Dyadic ±15"),
                    ("dyadic_12_4096", "Dyadic ±2,047"),
                    ("dyadic_16_4096", "Dyadic ±32,767"),
                ]
                for r in [by_name[key]]
            ],
        ),
        f"The two CPU baselines use independent interleaved runs with their own GPU times. The structure-aware result uses the faster of native CPU FP64 and prepacked Accelerate SGEMM with exact reconstruction. It favors the GPU in {sum(r['winner']=='GPU' for r in matched)} of {len(matched)} cases. The 1,024-square ±15 case favors CPU. At 4,096 square and ±32,767, all {v['checked_output_words']:,} output words matched CPU FP64 bit for bit. Of those values, {v['cpu_outputs_unrepresentable_in_fp32']:,} lose information when converted to FP32. [Structure-aware timings](results/v3-matrix-optimized-cpu.json) · [CPU reference implementation](tools/matrix_cpu_reference.c) · [Exact GPU implementation](mps_fp64/exact_gemm.py)",
        "## Placement and synchronization",
    ]
    rank = by_name["narrow_transposed"]
    q = next(
        r
        for r in primary(catalog, MAIN, "resident", "batch-per-call")
        if r["case"] == "batch_4096_4"
    )
    single = by_name[q["case"]]
    trip = next(
        r
        for r in primary(catalog, MAIN, "CPU round trip")
        if r["case"] == "dyadic_12_2048"
    )
    nt = next(
        r
        for r in primary(catalog, MAIN, "CPU round trip")
        if r["case"] == "narrow_32768_1024_2"
    )
    s += [
        f"An ordinary Gaussian 32,768 × 1,024 matrix with transposed storage multiplied by a 1,024 × 2 matrix reached **{factor(rank['cpu_over_gpu'])}** with resident inputs. For 4,096 independent 4 × 4 matrix products, a single synchronized batch call took {ms(single['gpu_median_ms'])} ms on GPU and {ms(single['cpu_median_ms'])} ms on CPU. Submitting {q['calls_per_round']} batch calls before synchronizing changed the per-call times to {ms(q['gpu_median_ms'])} and {ms(q['cpu_median_ms'])} ms, respectively.",
        f"Uploading both inputs and returning the entire output on every call preserved a **{factor(trip['cpu_over_gpu'])}** advantage for the 2,048-square ±2,047 case. The Gaussian 32,768 × 1,024 by 1,024 × 2 case fell to {factor(nt['cpu_over_gpu'])} with transfers. Each transfer-inclusive path allocates its output and finishes on the CPU. [Main-run samples](results/v3-large-matrices.json) · [Second input seed](results/v3-matrix-second-seed.json)",
    ]
    if group := primary(catalog, TENSOR):
        wins = [r for r in group if r["winner"] == "GPU"]
        s += [
            "## Current release beyond matrices",
            f"The public tensor benchmark measured {len(group)} completed CPU/GPU pairs with a wait after each call. The GPU finished sooner on **{len(wins)}** against the fastest measured native CPU path for each workload. Matching the output-allocation policy on both sides gives **{sum(r['matched_cpu_over_gpu']>1 for r in group)} of {len(group)}**. Letting the CPU reuse an output object whenever its API permits gives **{sum(r['reusable_cpu_over_gpu']>1 for r in group)} of {len(group)}**. Both devices reused outputs where both APIs supported and passed the correctness check. The CSV also gives the ratio against matched CPU allocation and CPU output reuse. [Current raw measurements](results/public-tensor-ops.json) · [Runner](tools/benchmark_tensor_ops.py)",
            table(
                [
                    "Single-call workload",
                    "GPU ms",
                    "Matched CPU/GPU",
                    "Reusable CPU/GPU",
                    "Fastest CPU/GPU",
                ],
                [
                    (
                        f"`{r['case']}`",
                        ms(r["gpu_median_ms"]),
                        factor(r["matched_cpu_over_gpu"]),
                        factor(r["reusable_cpu_over_gpu"]),
                        factor(r["cpu_over_gpu"]),
                    )
                    for r in sorted(
                        group, key=lambda r: r["cpu_over_gpu"], reverse=True
                    )
                    if r["matched_cpu_over_gpu"] > 1
                ],
            ),
        ]
    if group := primary(catalog, ORDERING):
        tensor_argmax = next(
            r for r in primary(catalog, TENSOR) if r["case"] == "argmax_all_8m"
        )
        tuned_argmax = next(r for r in group if r["case"] == "argmax_4096x2048")
        wins = [r for r in group if r["winner"] == "GPU"]
        s += [
            "## Ordering across CPU thread counts",
            f"The ordering study tested stable row sort, global argmax and row-wise top-32 on three input sizes each. It ran GPU and CPU blocks separately, with CPU tested at 1, 4, 8, 12 and 16 threads. The GPU completed **{len(wins)} of {len(group)}** cases sooner than the fastest tested CPU thread count and output policy for that case. At 4,096 × 2,048, global argmax took {ms(tuned_argmax['gpu_median_ms'])} ms on GPU in the grouped run and {ms(tensor_argmax['gpu_median_ms'])} ms in the interleaved tensor suite. These are independent single-call measurements. [Per-shape analysis](benchmarks/ordering.md) · [Thread-sweep samples](results/public-ordering.json) · [Runner](tools/benchmark_ordering.py)",
            table(
                [
                    "Workload",
                    "Shape",
                    "Fastest CPU threads",
                    "CPU ms",
                    "GPU ms",
                    "CPU/GPU",
                ],
                [
                    (
                        f"`{r['case']}`",
                        show_shape(json.loads(r["input_shapes"])),
                        r["cpu_threads"],
                        ms(r["cpu_median_ms"]),
                        ms(r["gpu_median_ms"]),
                        factor(r["cpu_over_gpu"]),
                    )
                    for r in group
                ],
            ),
        ]
    named = primary(catalog, NAMED)
    named_count = len(read(NAMED)["cases"])
    opinfo_counts = Counter(
        row["timing_class"] for row in catalog.opinfo if row["source"] == CATALOG
    )
    paired = sum(opinfo_counts[k] for k in ["faster", "slower", "within threshold"])
    recheck = [r for r in catalog.opinfo if r["source"] == RECHECK]
    ahead = sorted(
        (r for r in named if r["winner"] == "GPU"),
        key=lambda r: r["cpu_over_gpu"],
        reverse=True,
    )
    s += [
        "## Historical nonmatrix results in 0.2.0",
        f"The named workload suite measured {named_count} cases on release 0.2.0. Of {len(named)} cases with CPU timings, the GPU finished sooner in **{len(ahead)}**. All {len(ahead)} GPU wins below used resident inputs and multiple calls per round. The CPU ran the same public FP64 operation with allocating outputs. FFT plan rebuilding and the two RNG comparisons have GPU-version timings only. [Historical named workload data](results/v2-workloads.json) · [Complete wins and losses](benchmarks/tables.md)",
        table(
            ["Resident workload", "Calls per round", "CPU ms", "GPU ms", "CPU/GPU"],
            [
                (
                    f"`{r['case']}`",
                    r["calls_per_round"],
                    ms(r["cpu_median_ms"]),
                    ms(r["gpu_median_ms"]),
                    factor(r["cpu_over_gpu"]),
                )
                for r in ahead
            ],
        ),
        "## Coverage by experiment",
        "A case is one measured workload within a row of this table. A second seed, a transfer, or a new submission pattern creates another comparison for the same operator. A GPU win requires its unrounded median to be strictly lower. Small differences can fall inside the run-to-run spread. Quartiles and the number of faster paired rounds are in the CSV. The initial and first-release stream archives omit the software version in their raw data.",
        table(
            [
                "Experiment",
                "Cases",
                "GPU faster",
                "CPU faster",
                "Ties",
                "Rounds",
                "Calls per round",
            ],
            counts,
        ),
        "For 0.3.0 native matrix comparisons, CPU candidates include allocating and preallocated PyTorch plus direct Accelerate DGEMM, DGEMV or a preallocated outer product where applicable. The headline GPU path uses preallocated output for mm and bmm and the allocating public mv call. Each candidate was warmed and timed in shuffled order. GPU samples ended with a synchronization. The host-tensor and GPU-resident measurements exclude initial compilation. Source records identify the winning CPU path and the full set of candidates.",
        f"The OpInfo catalog compared the 0.1.0 and 0.2.0 GPU implementations. The CPU supplied correctness references only. Of {sum(opinfo_counts.values())} entries, {paired} materialized samples had timings in both versions. Using the original thresholds of 10% and 2 µs, {opinfo_counts['faster']} were faster in 0.2.0 and {opinfo_counts['within threshold']} fell within the threshold. {len(recheck)} samples received a separate {recheck[0]['new_rounds']}-round, {recheck[0]['calls_per_round']}-call recheck. [All catalog entries and rechecks](benchmarks/opinfo-version-comparisons.csv) · [Catalog JSON](results/v2-catalog-timing.json) · [Recheck JSON](results/v2-paired-recheck.json)",
        "## Reproduce and inspect",
        "The recorded system used Apple M4 Max, macOS 15.7.1, PyTorch 2.11.0 and 12 configured CPU threads, with additional thread counts in the ordering study. Wall times include Python dispatch and output completion under the stated timing protocol. Each ratio uses medians from the same run. Quartiles use inclusive linear interpolation. They describe dispersion across timed rounds. The historical streaming harness and the CPU-thread study measured CPU and GPU paths in separate blocks. Their CSV rows leave paired-round wins blank.",
        "Regenerate the tables and CSVs on any machine with Python 3.10 or later. The commands read the bundled measurements and check the published numbers, sample counts and quartiles. The optional `--figure` flag regenerates the SVG using Matplotlib 3.10.6.",
        "```sh\npython tools/publish_benchmarks.py\npython tools/publish_benchmarks.py --check\n```",
        "To rerun the matrix measurements on a supported Mac, install the project with its `validation` extra as described in the README. Fetch the public baseline wheels with the helper below. The historical reports record the hashes of the original measurement artifacts. The fetch helper validates the public builds. Use a fresh output filename to keep the archived samples available for comparison. The specialized CPU runner compiles its Accelerate reference with xcrun clang. [Installation](README.md) · [Matrix runner](tools/benchmark_large_matrices.py) · [Structure-aware CPU runner](tools/benchmark_cpu_lattice.py)",
        "```sh\nexport PYTORCH_ENABLE_MPS_FALLBACK=0\npython tools/fetch_benchmark_baselines.py\npython tools/benchmark_tensor_ops.py --output tensor-rerun.json\npython tools/benchmark_ordering.py --output ordering-rerun.json\npython tools/benchmark_large_matrices.py .benchmarks/mps_fp64-0.2.0-1public-py3-none-any.whl matrix-rerun.json --rounds 9\npython tools/benchmark_cpu_lattice.py structure-rerun.json --rounds 7\n```",
        "To rerun the historical named workloads or OpInfo catalog, pass the 0.2.0 public candidate wheel and the 0.1.0 public baseline from the same checkout. The runner loads both isolated packages.",
        "```sh\nexport PYTORCH_ENABLE_MPS_FALLBACK=0\npython tools/compare_performance.py .benchmarks/mps_fp64-0.1.0-1public-py3-none-any.whl named-020-rerun.json --candidate-wheel .benchmarks/mps_fp64-0.2.0-1public-py3-none-any.whl --suite workloads --rounds 11\npython tools/compare_performance.py .benchmarks/mps_fp64-0.1.0-1public-py3-none-any.whl opinfo-020-rerun.json --candidate-wheel .benchmarks/mps_fp64-0.2.0-1public-py3-none-any.whl --suite catalog --rounds 7 --batch 12\n```",
        "The all-results CSV keeps one row per GPU path and same-session CPU baseline. The first GPU-ahead file applies the fastest-CPU test. The matched-allocation file applies the `matched_cpu_over_gpu` test. Its `winner` column always describes the stricter fastest-CPU comparison. Both filters use full-precision medians. `role=primary` selects the paths in the complete tables. Source filenames resolve under `results/`. The per-path file retains alternative CPU implementations, native MPS FP32 comparisons when present, and the raw millisecond samples per call. Engineering comparisons of GPU kernels and arithmetic microbenchmarks have their own reports. [Result archive](results/) · [GPU arithmetic microbenchmark](results/gemm-arithmetic.json) · [Reduction timings](results/reduction-suite.json) · [Algorithm and validation notes](MATRIX_PERFORMANCE.md)",
    ]
    return "\n\n".join(s) + "\n"


def figure_data(catalog):
    matrix = {
        r["case"]: r["cpu_over_gpu"]
        for r in primary(catalog, MAIN, "resident", "single")
    }
    matched = {r["case"]: r["cpu_over_gpu"] for r in primary(catalog, MATCHED)}
    return [
        (label, matrix[name], matched.get(name))
        for name, label in [
            ("normal_square_4096", "Gaussian FP64"),
            ("dyadic_4_4096", "Dyadic ±15"),
            ("dyadic_12_4096", "Dyadic ±2,047"),
            ("dyadic_16_4096", "Dyadic ±32,767"),
        ]
    ]


def figure_description(data):
    return "CPU median / GPU median for 4096-square resident inputs. " + compact(data)


def render_figure(data):
    # Optional for figure authors. Normal table generation needs only the standard library.
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.ticker import NullLocator

    with mpl.rc_context(
        {
            "svg.hashsalt": "mps-fp64-matrix-030",
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    ):
        fig, ax = plt.subplots(figsize=(8, 3.45))
        fig.subplots_adjust(left=0.195, right=0.98, bottom=0.2, top=0.74)
        navy, orange = "#15566e", "#ab502a"
        for i, (_, native, optimized) in enumerate(data):
            ax.scatter(
                native,
                i - 0.12,
                c=navy,
                s=46,
                zorder=3,
                label="Native FP64 CPU" if i == 1 else None,
            )
            if optimized is not None:
                ax.scatter(
                    optimized,
                    i + 0.12,
                    c=orange,
                    marker="D",
                    s=37,
                    zorder=3,
                    label="Best of native and prepacked CPU" if i == 1 else None,
                )
        ax.set_xscale("log", base=2)
        ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        ax.set_xticks(ticks, ["⅛×", "¼×", "½×", "1×", "2×", "4×", "8×"])
        ax.set_yticks(range(len(data)), [s[0] for s in data])
        ax.set_xlim(0.11, 11.5)
        ax.set_ylim(len(data) - 0.54, -0.55)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.axvline(1, color="#525d64", linewidth=1.2, zorder=1)
        ax.grid(axis="x", color="#e0e5e8", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#88939a")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.set_xlabel("CPU median / GPU median     (above 1× favors GPU)", labelpad=8)
        fig.text(
            0.025,
            0.91,
            "4,096-square matrix multiplication",
            size=15,
            weight="bold",
            color="#192f38",
        )
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.016, 0.885),
            frameon=False,
            ncol=2,
            handletextpad=0.25,
            columnspacing=1.7,
            fontsize=9,
        )
        buf = io.StringIO()
        fig.savefig(
            buf,
            format="svg",
            metadata={
                "Date": None,
                "Description": figure_description(data),
                "Title": "4096-square FP64 matrix speed relative to CPU",
            },
        )
        plt.close(fig)
        return buf.getvalue()


def check_figure(path, data):
    from xml.etree import ElementTree as ET

    if not path.exists():
        return False
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError:
        return False
    return any(text == figure_description(data) for text in root.itertext())


def files(catalog):
    rows = catalog.results
    unique_path_cases = {
        (
            r["source"],
            r["case"],
            r["timing"],
            r["placement"],
            r["version"],
            r["gpu_path"],
        )
        for r in rows
    }
    if len(unique_path_cases) != len(rows):
        raise ValueError("Duplicate CPU/GPU path comparison")
    ahead = [r for r in rows if r["winner"] == "GPU"]
    counts = Counter(
        r["timing_class"] for r in catalog.opinfo if r["source"] == CATALOG
    )
    if counts != Counter(read("v2-summary.json")["timing_counts"]):
        raise ValueError("OpInfo counts disagree with the published acceptance report")
    return {
        "benchmarks/all-results.csv": csv_text(rows),
        "benchmarks/gpu-ahead.csv": csv_text(ahead),
        "benchmarks/matched-allocation-ahead.csv": csv_text(
            [
                r
                for r in rows
                if r["matched_cpu_over_gpu"] and r["matched_cpu_over_gpu"] > 1
            ]
        ),
        "benchmarks/timed-paths.csv": csv_text(catalog.paths),
        "benchmarks/opinfo-version-comparisons.csv": csv_text(catalog.opinfo),
        "BENCHMARKS.md": overview(catalog),
        "benchmarks/tables.md": details(catalog),
    }


def broken_document_links(outputs):
    broken = []
    for relative, value in outputs.items():
        if not relative.endswith(".md"):
            continue
        parent = (ROOT / relative).parent
        for target in re.findall(r"\]\(([^)\s]+)\)", value):
            if "://" in target or target.startswith("#"):
                continue
            target_path = Path(target.split("#", 1)[0])
            if target_path.is_absolute() or not (parent / target_path).exists():
                broken.append(f"{relative} -> {target}")
    return broken


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any published file differs from the raw records",
    )
    parser.add_argument(
        "--figure",
        action="store_true",
        help="also render the chart with Matplotlib 3.10.6",
    )
    args = parser.parse_args()
    catalog = Catalog()
    for source in (MAIN, SECOND):
        catalog.matrices(source)
    catalog.structure_aware()
    catalog.tensor_ops()
    catalog.ordering()
    catalog.workloads()
    catalog.earlier()
    for source in (CATALOG, RECHECK):
        catalog.operators(source)
    outputs = files(catalog)
    chart = "benchmarks/matrix-speedups.svg"
    data = figure_data(catalog)
    if args.figure:
        outputs[chart] = render_figure(data)
    elif not check_figure(ROOT / chart, data):
        parser.exit(
            1, "Chart data changed or the chart is missing. Regenerate with --figure.\n"
        )
    mismatches = []
    for name, generated in outputs.items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_bytes() != generated.encode("utf-8"):
                mismatches.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(generated.encode("utf-8"))
    if args.check:
        mismatches.extend(broken_document_links(outputs))
    if mismatches:
        parser.exit(1, f"Regenerate stale benchmark files: {', '.join(mismatches)}\n")
    print(
        f"{len(outputs)} publications, {len(catalog.results)} CPU/GPU comparisons, {len(catalog.opinfo)} OpInfo entries"
    )


if __name__ == "__main__":
    main()
