import json
from pathlib import Path

from external_comparison.adapters.native_runtime import stage_checkout, stage_humaneval_data


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_aflow_seed_staging_uses_frozen_33_131_split(tmp_path: Path):
    source = tmp_path / "AFlow"
    (source / "workspace" / "HumanEval" / "workflows" / "round_1").mkdir(parents=True)
    (source / "workspace" / "HumanEval" / "workflows" / "round_1" / "graph.py").write_text("class Workflow: pass\n", encoding="utf-8")
    rows = [{"task_id": f"HumanEval/{index}", "prompt": f"p{index}", "test": "def check(x): pass", "entry_point": "f"} for index in range(164)]
    dataset = tmp_path / "humaneval.jsonl"
    public = tmp_path / "public.jsonl"
    _write_jsonl(dataset, rows)
    _write_jsonl(public, [{"entry_point": "f", "test": "assert True"}])

    workspace = stage_checkout(source, tmp_path / "outputs", "aflow", 0)
    manifest = stage_humaneval_data(workspace, "aflow", dataset, public, 2026)

    assert len(manifest["search_tasks"]) == 33
    assert len(manifest["test_tasks"]) == 131
    assert set(manifest["search_tasks"]).isdisjoint(manifest["test_tasks"])
    assert sum(1 for _ in Path(manifest["search_path"]).open(encoding="utf-8")) == 33
    assert sum(1 for _ in Path(manifest["test_path"]).open(encoding="utf-8")) == 131
    assert Path(manifest["public_test_path"]).read_bytes() == public.read_bytes()
