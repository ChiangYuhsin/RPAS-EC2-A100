"""Native MaAS adapter using the official controller and HumanEval graph."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import time
import types
from enum import Enum
from pathlib import Path

from external_comparison.adapters.native_common import (
    call_record,
    env_path,
    execute_humaneval,
    extract_code,
    git_commit,
    load_jsonl,
    humaneval_external_split,
    write_native_result,
)


def _root() -> Path:
    default = Path(__file__).resolve().parents[2] / "external_baselines" / "MaAS"
    return env_path("RPAS_MAAS_ROOT", str(default))


def _llm_config():
    from maas.configs.llm_config import LLMConfig, LLMType
    return LLMConfig(
        api_type=LLMType.OPENAI,
        model=os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        base_url=os.environ.get("RPAS_EXTERNAL_API_BASE", "http://127.0.0.1:29500/v1"),
        api_key=os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"),
        max_token=6144, temperature=0.0, top_p=1.0, stream=False, calc_usage=True,
    )


def _patch_local_embedding_model() -> str:
    """Resolve MaAS's official MiniLM name to an explicitly staged local copy."""
    model_path = os.environ.get("RPAS_MAAS_EMBEDDING_MODEL", "").strip()
    if not model_path:
        return "sentence-transformers/all-MiniLM-L6-v2"

    local_path = Path(model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"MaAS embedding model directory not found: {local_path}")

    import sentence_transformers

    original_constructor = sentence_transformers.SentenceTransformer

    def local_constructor(model_name_or_path, *args, **kwargs):
        if model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2":
            model_name_or_path = str(local_path)
        return original_constructor(model_name_or_path, *args, **kwargs)

    sentence_transformers.SentenceTransformer = local_constructor
    return str(local_path)


def _prepare_runtime_config(output_dir: Path) -> Path:
    """Give MaAS a non-secret config before its import-time validation runs."""
    runtime_root = output_dir / "_maas_runtime"
    config_dir = runtime_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "llm": {
            "api_type": "openai",
            "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
            "base_url": os.environ.get("RPAS_EXTERNAL_API_BASE", "http://127.0.0.1:29500/v1"),
            "api_key": os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"),
        }
    }
    (config_dir / "config2.yaml").write_text(json.dumps(payload), encoding="utf-8")
    os.environ["METAGPT_PROJECT_ROOT"] = str(runtime_root)
    return runtime_root


def _install_import_compat(root: Path) -> None:
    """Avoid importing unused provider/tool integrations during official imports."""
    import maas

    tools_module = types.ModuleType("maas.tools")
    tools_module.__path__ = [str(root / "maas" / "tools")]

    class SearchEngineType(Enum):
        SERPAPI_GOOGLE = "serpapi"
        SERPER_GOOGLE = "serper"
        DIRECT_GOOGLE = "google"
        DUCK_DUCK_GO = "ddg"
        CUSTOM_ENGINE = "custom"
        BING = "bing"

    class WebBrowserEngineType(Enum):
        PLAYWRIGHT = "playwright"
        SELENIUM = "selenium"
        CUSTOM = "custom"

    tools_module.SearchEngineType = SearchEngineType
    tools_module.WebBrowserEngineType = WebBrowserEngineType
    sys.modules["maas.tools"] = tools_module
    maas.tools = tools_module

    provider_module = types.ModuleType("maas.provider")
    provider_module.__path__ = [str(root / "maas" / "provider")]
    sys.modules["maas.provider"] = provider_module
    maas.provider = provider_module


def _capture_usage(llm, records: list[dict]) -> None:
    original_update = llm._update_costs

    def update_costs(usage, model=None, local_calc_usage=True):
        payload = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
        records.append(payload)
        return original_update(usage, model=model, local_calc_usage=local_calc_usage)

    llm._update_costs = update_costs


async def _run(rows: list[dict], output_dir: Path, seed: int) -> None:
    root = _root()
    if not root.exists():
        raise FileNotFoundError(f"MaAS repository not found: {root}")
    _prepare_runtime_config(output_dir)
    os.chdir(root)
    sys.path.insert(0, str(root))
    embedding_model = _patch_local_embedding_model()
    _install_import_compat(root)
    import torch
    from maas.ext.maas.models import controller as controller_module
    import maas.provider.openai_api  # noqa: F401 - registers the official OpenAI provider
    from maas.ext.maas.scripts.optimized.HumanEval.test.template.operator_registry import operator_names
    graph_module = importlib.import_module("maas.ext.maas.scripts.optimized.HumanEval.test.graph")
    torch.manual_seed(seed)
    controller = controller_module.MultiLayerController(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # The upstream split `.pth` files are not consumed by the published
    # Optimizer.test() path. A valid EC-1 run must train this fresh controller
    # with Optimizer.optimize("Graph") and then load the resulting
    # HumanEval_controller_sample*.pth checkpoint before testing.
    if os.environ.get("RPAS_MAAS_FRESH_TRAINED", "0") != "1":
        raise RuntimeError(
            "MaAS EC-1 requires the official fresh train -> checkpoint -> test "
            "workflow; refusing to evaluate an untrained random controller"
        )
    encoded_names = controller_module.sentence_encoder.model.encode(operator_names)
    embeddings = torch.as_tensor(encoded_names, dtype=torch.float32)
    if tuple(embeddings.shape) != (len(operator_names), 384):
        raise ValueError(f"unexpected MaAS operator embedding shape: {tuple(embeddings.shape)}")
    calls: list[dict] = []
    results: list[dict] = []
    for row in rows:
        workflow = graph_module.Workflow("native_external", _llm_config(), "HumanEval", controller, embeddings)
        usage_records: list[dict] = []
        _capture_usage(workflow.llm, usage_records)
        started = time.perf_counter()
        output, _, _ = await workflow(row["prompt"], row["entry_point"], str(output_dir))
        elapsed = (time.perf_counter() - started) * 1000
        per_call_latency = elapsed / max(1, len(usage_records))
        for index, usage in enumerate(usage_records):
            usage = dict(usage)
            usage["model"] = _llm_config().model
            usage["latency_ms"] = per_call_latency
            calls.append(call_record(f"humaneval-maas-seed-{seed}", "maas", "humaneval", "test", str(row.get("task_id", row.get("id", ""))), index, usage))
        code = extract_code(str(output), row["entry_point"])
        execution = execute_humaneval(code, row)
        results.append({"task_id": row.get("task_id", row.get("id")), "passed": execution["passed"], "status": execution["status"], "output": str(output), "execution": execution})
    manifest = {
        "run_id": f"humaneval-maas-seed-{seed}", "method": "maas", "dataset": "humaneval", "seed": seed,
        "implementation_status": "official_controller_and_workflow", "native_search": "controller_sampling",
        "official_repo": str(root), "official_commit": git_commit(root), "search_calls": 0, "search_tokens": 0,
        "model": _llm_config().model, "api_base": _llm_config().base_url,
        "embedding_model": embedding_model, "operator_embedding_reuse": True,
        "import_compat": "lazy_official_openai_provider_and_tool_package",
    }
    write_native_result(output_dir, manifest, results, calls)


def run_humaneval(args) -> None:
    rows = humaneval_external_split(load_jsonl(args.dataset_path), args.data_seed)["test"]
    sample_limit = int(os.environ.get("RPAS_NATIVE_SAMPLE_LIMIT", "0"))
    if sample_limit > 0:
        rows = rows[:sample_limit]
    asyncio.run(_run(rows, Path(args.output_dir) / "maas" / f"seed_{args.seed}", args.seed))
