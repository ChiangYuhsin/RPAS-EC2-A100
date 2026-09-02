"""Aggregate EC-2 MMLU-57x10 controlled-subset runs.

The aggregator reads only per-seed artifacts and emits a reproducible main-table
view. It keeps test inference cost separate from search cost; a zero search
counter means that the method did not separately instrument a search phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    margin = 1.96 * stdev(values) / math.sqrt(len(values))
    center = mean(values)
    return center - margin, center + margin


def _load_seed(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    summary = result.get("summary", result)
    if manifest.get("formal_result") is not False:
        raise ValueError(f"formal_result must be false for controlled pilot: {run_dir}")
    if summary.get("num_examples") != 570:
        raise ValueError(f"expected 570 test examples for MMLU-57x10: {run_dir}")
    test_calls = int(summary.get("inference_calls", 0))
    search_calls = int(summary.get("search_calls", manifest.get("search_calls", 0)))
    test_tokens = int(summary.get("inference_tokens", 0))
    search_tokens = int(summary.get("search_tokens", manifest.get("search_tokens", 0)))
    return {
        "method": str(manifest["method"]),
        "seed": int(manifest["seed"]),
        "accuracy": float(summary["score"]),
        "valid_answer_rate": float(summary["valid_answer_rate"]),
        "test_calls": test_calls,
        "search_calls": search_calls,
        "total_calls": test_calls + search_calls,
        "test_tokens": test_tokens,
        "search_tokens": search_tokens,
        "total_tokens": test_tokens + search_tokens,
        "search_instrumented": bool(search_calls or search_tokens),
    }


def aggregate(root: str | Path, output_dir: str | Path) -> dict:
    root = Path(root)
    rows = []
    for method in ("rpas", "gdesigner"):
        for seed in (0, 1, 2):
            rows.append(_load_seed(root / method / f"seed_{seed}"))
    grouped: dict[str, list[dict]] = {method: [r for r in rows if r["method"] == method] for method in ("rpas", "gdesigner")}
    table = []
    for method, method_rows in grouped.items():
        if len(method_rows) != 3:
            raise ValueError(f"expected three seeds for {method}")
        accuracy = [r["accuracy"] for r in method_rows]
        valid = [r["valid_answer_rate"] for r in method_rows]
        fields = {key: [r[key] for r in method_rows] for key in ("test_calls", "search_calls", "total_calls", "test_tokens", "search_tokens", "total_tokens")}
        row = {
            "method": method,
            "seeds": 3,
            "test_examples": 570,
            "accuracy_mean": mean(accuracy),
            "accuracy_ci95_low": _ci95(accuracy)[0],
            "accuracy_ci95_high": _ci95(accuracy)[1],
            "valid_answer_rate_mean": mean(valid),
            **{f"{key}_mean": mean(values) for key, values in fields.items()},
            "search_cost_note": "instrumented" if any(r["search_instrumented"] for r in method_rows) else "not separately instrumented",
        }
        table.append(row)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "MMLU-57x10 controlled subset", "formal_result": False, "rows": table, "seed_rows": rows}
    (out / "main_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = list(table[0])
    with (out / "main_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help=".../ec2_gpu6/{rpas,gdesigner}")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
