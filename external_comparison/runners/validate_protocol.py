"""Validate comparison configs without contacting an API or running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from external_comparison.common.protocol import (
    CONTROLLED_SEARCH_METHODS,
    DATASETS,
    EC1_EXTERNAL_METHODS,
    EC2_EXTERNAL_METHODS,
    SEARCH_SEEDS,
)
from external_comparison.adapters.registry import native_adapter_status


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute() or value.startswith(("sk-", "AIza"))
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return False


def validate_config(path: str | Path) -> list[str]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if _contains_absolute_path(payload):
        errors.append("config contains an absolute path or token-like value")
    if config_path.name == "common.json":
        if tuple(payload.get("search_seeds", [])) != SEARCH_SEEDS:
            errors.append(f"search_seeds must be {list(SEARCH_SEEDS)}")
        if tuple(payload.get("methods", [])) != CONTROLLED_SEARCH_METHODS:
            errors.append(f"methods must be {list(CONTROLLED_SEARCH_METHODS)}")
    elif config_path.name == "ec1_humaneval.json":
        if tuple(payload.get("methods", [])) != EC1_EXTERNAL_METHODS:
            errors.append(f"methods must be {list(EC1_EXTERNAL_METHODS)}")
    elif config_path.name == "ec2_mmlu.json":
        if tuple(payload.get("methods", [])) != EC2_EXTERNAL_METHODS:
            errors.append(f"methods must be {list(EC2_EXTERNAL_METHODS)}")
    else:
        dataset = payload.get("dataset")
        if dataset not in DATASETS:
            errors.append(f"dataset must be one of {list(DATASETS)}")
        if not payload.get("methods"):
            errors.append("methods must not be empty")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="external_comparison/configs")
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    failures = 0
    for path in sorted(config_dir.glob("*.json")):
        errors = validate_config(path)
        if errors:
            failures += 1
            print(f"FAIL {path}: {'; '.join(errors)}")
        else:
            print(f"OK   {path}")
            if args.require_native and path.name in {"ec1_humaneval.json", "ec2_mmlu.json"}:
                payload = json.loads(path.read_text(encoding="utf-8"))
                for method in payload.get("methods", []):
                    status = native_adapter_status(str(method))
                    if not status["available"]:
                        failures += 1
                        print(f"FAIL {path}: native adapter missing for {method}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
