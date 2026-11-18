"""Join expanded correctness and paired timing records without conflating their input sets."""

import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"


def audit(name):
    rows = map(json.loads, (R / f"audit-v2-{name}.jsonl").read_text().splitlines())
    return {(r["op"], r["variant"]): r for r in rows}


def class_of(old, new):
    if old["status"] != "pass" or new["status"] != "pass":
        return "unpaired"
    if old["category"] != "materialized" or new["category"] != "materialized":
        return "alias, empty or host result"
    a, b = old["median_ms"], new["median_ms"]
    if a >= b * 1.10 and (a - b) >= 0.002:
        return "faster"
    if b >= a * 1.10 and (b - a) >= 0.002:
        return "slower"
    return "within threshold"


def main():
    native, old, new = [
        audit(n) for n in ("native-fp32", "baseline-fp64", "current-fp64")
    ]
    timing = json.loads((R / "v2-catalog-timing.json").read_text())
    cases = {(r["op"], r["variant"]): r for r in timing["cases"]}
    assert native.keys() == old.keys() == new.keys() == cases.keys()
    report = []
    for key in sorted(new):
        c = cases[key]
        a, b = (
            c.get("paths", {}).get(p, {"status": "excluded"})
            for p in ("old_fp64", "new_fp64")
        )
        row = {
            "operator": key[0],
            "variant": key[1],
            "native_fp32": native[key]["status"],
            "v01_fp64": old[key]["status"],
            "v02_fp64": new[key]["status"],
            "native_checked": native[key]["checked"],
            "v01_checked": old[key]["checked"],
            "v02_checked": new[key]["checked"],
            "v02_output_devices": ",".join(new[key].get("output_devices", [])),
            "v02_first_error": new[key].get("detail", ""),
            "timing_sample_index": c.get("sample_index"),
            "timing_shapes": json.dumps(c["shapes"]) if "shapes" in c else "",
            "v01_timing_status": a["status"],
            "v02_timing_status": b["status"],
            "v02_output_kind": b.get("category", ""),
            "old_median_us": None,
            "new_median_us": None,
            "old_over_new": None,
            "timing_class": class_of(a, b),
            "timing_detail": c.get("exclusion", b.get("detail", a.get("detail", ""))),
        }
        if "median_ms" in a:
            row["old_median_us"] = round(1000 * a["median_ms"], 4)
        if "median_ms" in b:
            row["new_median_us"] = round(1000 * b["median_ms"], 4)
        if "median_ms" in a and "median_ms" in b:
            row["old_over_new"] = round(a["median_ms"] / b["median_ms"], 5)
        report.append(row)
    with (R / "v2-operator-coverage-and-speed.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=report[0])
        writer.writeheader()
        writer.writerows(report)

    def passed(rows):
        return {k for k, v in rows.items() if v["status"] == "pass"}

    n, o, u = (passed(a) for a in (native, old, new))
    summary = {
        "entries": len(native),
        "native": len(n),
        "old": len(o),
        "new": len(u),
        "old_native": len(o & n),
        "new_native": len(u & n),
        "native_gaps": len(n - u),
        "new_passes": len(u - o),
        "lost_passes": sorted(o - u),
        "samples_compared": sum(v["checked"] for v in new.values()),
        "timing_counts": dict(Counter(r["timing_class"] for r in report)),
        "measured_new_examples": sum(r["new_median_us"] is not None for r in report),
        "unmatched_native_entries": sorted(n - u),
    }
    (R / "v2-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "unmatched_native_entries"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
