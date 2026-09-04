"""Native AFlow adapter backed by the official repository workflow."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time
from pathlib import Path

from external_comparison.adapters.native_common import (
    call_record,
    env_path,
    execute_humaneval,
    extract_code,
    git_commit,
    load_jsonl,
    split_rows,
    write_native_result,
)


def _root() -> Path:
    default = Path(__file__).resolve().parents[2] / "external_baselines" / "AFlow"
    return env_path("RPAS_AFLOW_ROOT", str(default))


def _config():
    from scripts.async_llm import LLMConfig

    return LLMConfig(
        {
            "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
            "key": os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"),
            "base_url": os.environ.get("RPAS_EXTERNAL_API_BASE", "http://127.0.0.1:29500/v1"),
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )


def _limit_completion_tokens(llm) -> None:
    """Apply the shared HumanEval output cap to AFlow's native client."""
    original_create = llm.aclient.chat.completions.create
    max_tokens = int(os.environ.get("RPAS_HUMANEVAL_MAX_TOKENS", "6144"))

    async def create(*args, **kwargs):
        kwargs.setdefault("max_tokens", max_tokens)
        kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
        return await original_create(*args, **kwargs)

    llm.aclient.chat.completions.create = create


async def _run(rows: list[dict], output_dir: Path, seed: int) -> None:
    root = _root()
    if not root.exists():
        raise FileNotFoundError(f"AFlow repository not found: {root}")
    os.chdir(root)
    sys.path.insert(0, str(root))
    graph_module = importlib.import_module("workspace.HumanEval.workflows.round_1.graph")
    workflow_class = graph_module.Workflow
    calls: list[dict] = []
    results: list[dict] = []
    for row in rows:
        started = time.perf_counter()
        workflow = workflow_class("native_external", _config(), "HumanEval")
        _limit_completion_tokens(workflow.llm)
        output, _ = await workflow(row["prompt"], row["entry_point"])
        elapsed = (time.perf_counter() - started) * 1000
        history = workflow.llm.get_usage_summary().get("history", [])
        for index, usage in enumerate(history):
            usage = dict(usage)
            usage["latency_ms"] = elapsed / max(1, len(history))
            calls.append(call_record(f"humaneval-aflow-seed-{seed}", "aflow", "humaneval", "test", str(row.get("task_id", row.get("id", ""))), index, usage))
        code = extract_code(str(output), row["entry_point"])
        execution = execute_humaneval(code, row)
        results.append({"task_id": row.get("task_id", row.get("id")), "passed": execution["passed"], "status": execution["status"], "output": str(output), "execution": execution})
    manifest = {
        "run_id": f"humaneval-aflow-seed-{seed}", "method": "aflow", "dataset": "humaneval", "seed": seed,
        "implementation_status": "official_workflow_artifact", "native_search": "official_round_1_workflow",
        "official_repo": str(root), "official_commit": git_commit(root), "search_calls": 0, "search_tokens": 0,
        "model": _config().model, "api_base": _config().base_url,
    }
    write_native_result(output_dir, manifest, results, calls)


def run_humaneval(args) -> None:
    rows = split_rows(load_jsonl(args.dataset_path), args.data_seed, 80, 40, 44)["test"]
    asyncio.run(_run(rows, Path(args.output_dir) / "aflow" / f"seed_{args.seed}", args.seed))
