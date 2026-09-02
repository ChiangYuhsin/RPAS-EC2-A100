"""Dispatch EC-1 to repository-local native adapters.

This module is intentionally not a proxy. An adapter must export
``run_humaneval(args: argparse.Namespace)`` and own its native search loop;
the shared evaluator/telemetry contract is the integration boundary.
"""

from __future__ import annotations

import argparse
import importlib

from external_comparison.adapters.registry import NATIVE_ADAPTER_MODULES, require_native_adapters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--method", choices=["aflow", "maas", "rpas"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    require_native_adapters([args.method])
    module = importlib.import_module(NATIVE_ADAPTER_MODULES[args.method])
    runner = getattr(module, "run_humaneval", None)
    if runner is None:
        raise RuntimeError(f"native adapter {args.method} must export run_humaneval(args)")
    runner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
