"""Isolated, reproducible staging for EC-1 native baseline runs.

The upstream AFlow and MaAS checkouts are read-only inputs.  Every run gets a
fresh copy under the ignored output directory so no seed can consume another
seed's workflow history, controller checkpoint, or generated files.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import (
    git_commit,
    humaneval_external_split,
    load_jsonl,
    sha256_file,
)


def _ignore(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {".git", "__pycache__", ".pytest_cache"}}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch before constructing an upstream optimizer."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def stage_checkout(
    source: Path,
    output_dir: Path,
    method: str,
    seed: int,
    *,
    replace: bool = False,
    require_clean_git: bool = False,
) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"official {method} checkout not found: {source}")
    if require_clean_git:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source,
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty.returncode != 0:
            raise RuntimeError(f"official {method} source is not a readable Git checkout: {source}")
        if dirty.stdout.strip():
            raise RuntimeError(
                f"official {method} source has uncommitted changes; refusing to stage an unpinned baseline: {source}"
            )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
        )
        non_cache_untracked = [
            item for item in untracked.stdout.splitlines()
            if "__pycache__/" not in item and not item.endswith(".pyc")
        ]
        if non_cache_untracked:
            raise RuntimeError(
                f"official {method} source has untracked non-cache files; refusing to stage an unpinned baseline: "
                f"{non_cache_untracked[:5]}"
            )
    workspace = output_dir / "_workspaces" / f"{method}_seed_{seed}"
    if workspace.exists():
        if not replace:
            raise FileExistsError(
                f"seed workspace already exists: {workspace}; refuse to reuse a search history. "
                "Set RPAS_EC1_REPLACE_WORKSPACE=1 only to discard this local workspace."
            )
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, workspace, ignore=_ignore)
    return workspace


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_humaneval_data(
    workspace: Path,
    method: str,
    dataset_path: Path,
    public_test_path: Path,
    data_seed: int,
) -> dict[str, Any]:
    """Materialize the frozen 33/131 split at the upstream paths it expects."""
    split = humaneval_external_split(load_jsonl(dataset_path), data_seed)
    if len(split["search"]) != 33 or len(split["test"]) != 131:
        raise AssertionError("EC-1 requires exactly 33 search and 131 test tasks")
    if method == "aflow":
        root = workspace / "data" / "datasets"
        search_path = root / "humaneval_validate.jsonl"
        test_path = root / "humaneval_test.jsonl"
    elif method == "maas":
        root = workspace / "maas" / "ext" / "maas" / "data"
        search_path = root / "humaneval_train.jsonl"
        test_path = root / "humaneval_test.jsonl"
    else:
        raise ValueError(f"unsupported native method: {method}")
    write_jsonl(search_path, split["search"])
    write_jsonl(test_path, split["test"])
    public_target = root / "humaneval_public_test.jsonl"
    shutil.copyfile(public_test_path, public_target)
    return {
        "dataset_source_sha256": sha256_file(dataset_path),
        "public_test_source_sha256": sha256_file(public_test_path),
        "search_path": str(search_path),
        "search_sha256": sha256_file(search_path),
        "search_tasks": [str(row.get("task_id", "")) for row in split["search"]],
        "test_path": str(test_path),
        "test_sha256": sha256_file(test_path),
        "test_tasks": [str(row.get("task_id", "")) for row in split["test"]],
        "public_test_path": str(public_target),
        "public_test_sha256": sha256_file(public_target),
    }


def write_aflow_config(workspace: Path, model: str, base_url: str, api_key: str) -> None:
    content = {
        "models": {
            model: {
                "api_type": "openai",
                "base_url": base_url,
                "api_key": api_key,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        }
    }
    (workspace / "config").mkdir(exist_ok=True)
    (workspace / "config" / "config2.yaml").write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def write_maas_config(workspace: Path, model: str, base_url: str, api_key: str, seed: int) -> None:
    content = {
        "llm": {"api_type": "openai", "model": model, "base_url": base_url, "api_key": api_key},
        "models": {
            model: {
                "api_type": "openai",
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_token": int(os.environ.get("RPAS_HUMANEVAL_MAX_TOKENS", "6144")),
                "stream": False,
                "calc_usage": True,
                "seed": seed,
            }
        },
    }
    (workspace / "config").mkdir(exist_ok=True)
    (workspace / "config" / "config2.yaml").write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def source_manifest(source: Path, workspace: Path, method: str, seed: int, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "seed": seed,
        "official_source": str(source),
        "official_commit": git_commit(source),
        "official_source_clean": True,
        "isolated_workspace": str(workspace),
        "data": data,
        "workspace_reused": False,
    }
