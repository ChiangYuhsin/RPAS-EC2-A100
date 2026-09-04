"""Dispatch EC-1 to repository-local native adapters.

This module is intentionally not a proxy. An adapter must export
``run_humaneval(args: argparse.Namespace)`` and own its native search loop;
the shared evaluator/telemetry contract is the integration boundary.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

from external_comparison.adapters.registry import NATIVE_ADAPTER_MODULES, require_native_adapters
from external_comparison.runners.ec1_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--method", choices=["aflow", "maas", "rpas"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--public-test-path", help="AFlow-derived public-test fixture shared by all EC-1 methods")
    parser.add_argument("--aflow-validate-path", help="Frozen AFlow validation fixture, required for formal runs")
    parser.add_argument("--aflow-test-path", help="Frozen AFlow test fixture, required for formal runs")
    parser.add_argument("--run-kind", choices=("pilot", "formal"), default="pilot")
    args = parser.parse_args()
    if args.run_kind == "formal":
        if os.environ.get("RPAS_EC1_GPU", "") not in {"4", "5"}:
            parser.error("formal EC-1 requires RPAS_EC1_GPU=4 or RPAS_EC1_GPU=5")
        if not args.public_test_path or not args.aflow_validate_path or not args.aflow_test_path:
            parser.error("formal EC-1 requires --public-test-path, --aflow-validate-path, and --aflow-test-path")
        run_preflight(
            Path(args.dataset_path), Path(args.public_test_path),
            Path("external_comparison/configs/ec1_humaneval.json"),
            validate_path=Path(args.aflow_validate_path), test_path=Path(args.aflow_test_path),
        )
    require_native_adapters([args.method])
    module = importlib.import_module(NATIVE_ADAPTER_MODULES[args.method])
    runner = getattr(module, "run_humaneval", None)
    if runner is None:
        raise RuntimeError(f"native adapter {args.method} must export run_humaneval(args)")
    runner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
