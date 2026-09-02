from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from external_comparison.adapters.native_common import write_native_result
from external_comparison.adapters.native_rpas import _protocol_mmlu_candidate
from external_comparison.runners.mmlu import (
    MMLU_SUBJECTS,
    build_mmlu_manifest,
    load_mmlu_subject,
    parse_mmlu_choice,
)


def test_mmlu_choice_parser_requires_unambiguous_final_answer() -> None:
    assert parse_mmlu_choice("Reasoning\nFINAL ANSWER: C") == "C"
    assert parse_mmlu_choice("C") == "C"
    assert parse_mmlu_choice("### B\nSelected by majority") == "B"
    assert parse_mmlu_choice(r"\boxed{D}") == "D"
    assert parse_mmlu_choice("The answer is C, but maybe D") == ""


def test_rpas_mmlu_candidate_freezes_protocol_decoding() -> None:
    candidate = {
        "topology": "planner_solver_verifier",
        "temperature": 0.3,
        "planner_max_tokens": 1024,
        "agents": [{"name": "planner", "max_tokens": 1536}, {"name": "solver"}],
    }
    prepared = _protocol_mmlu_candidate(candidate)
    assert prepared["temperature"] == 0.0
    assert prepared["planner_max_tokens"] == 256
    assert all(agent["max_tokens"] == 256 for agent in prepared["agents"])
    assert candidate["temperature"] == 0.3
    assert "max_tokens" not in candidate["agents"][1]


def test_native_result_records_invalid_answer_rate(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    write_native_result(
        output_dir,
        {"method": "gdesigner", "dataset": "mmlu"},
        [{"prediction": "A", "correct": True}, {"prediction": "", "correct": False}],
        [{"total_tokens": 4, "finish_reason": "stop", "error": None}],
    )
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["summary"]["valid_answer_rate"] == 0.5
    assert result["summary"]["num_examples"] == 2


def test_mmlu_loader_rejects_malformed_rows(tmp_path: Path) -> None:
    (tmp_path / "dev").mkdir()
    path = tmp_path / "dev" / "abstract_algebra_dev.csv"
    path.write_text("question,a,b,c,d,E\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid answer"):
        load_mmlu_subject(tmp_path, "abstract_algebra", "dev")


def test_mmlu_manifest_is_deterministic(tmp_path: Path) -> None:
    for split in ("dev", "test"):
        (tmp_path / split).mkdir()
        for subject in MMLU_SUBJECTS:
            path = tmp_path / split / f"{subject}_{split}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                for index in range(3):
                    writer.writerow([f"q{index}", "a", "b", "c", "d", "A"])
    first = build_mmlu_manifest(tmp_path, search_per_subject=2, test_per_subject=2)
    second = build_mmlu_manifest(tmp_path, search_per_subject=2, test_per_subject=2)
    assert first == second
    assert first["search"]["count"] == 2 * len(MMLU_SUBJECTS)
    assert first["test"]["count"] == 2 * len(MMLU_SUBJECTS)
