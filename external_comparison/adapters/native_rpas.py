"""Native RPAS adapter.

The repository's phase-2 runner remains the implementation authority.  This
module only supplies the external-comparison entry point and never aliases an
external baseline as RPAS.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

from external_comparison.adapters.native_common import require_valid_answer_rate, write_native_result


MMLU_MAX_TOKENS = 256


def _protocol_mmlu_candidate(candidate: dict, *, max_tokens: int = MMLU_MAX_TOKENS) -> dict:
    """Freeze the EC-2 decoding budget without changing the search candidate in place."""

    prepared = copy.deepcopy(candidate)
    for agent in prepared.get("agents", []):
        if isinstance(agent, dict):
            agent["max_tokens"] = max_tokens
    if "planner_max_tokens" in prepared:
        prepared["planner_max_tokens"] = max_tokens
    if "temperature" in prepared:
        prepared["temperature"] = 0.0
    return prepared


def run_humaneval(args) -> None:
    from external_comparison.runners.humaneval import run_experiment

    repo_root = Path(args.repo_root).resolve()
    result = run_experiment(
        repo_root=repo_root,
        dataset_path=Path(args.dataset_path).resolve(),
        model_config_path=repo_root / "experiments" / "phase2_wan_agent_config_qwen35_9b_homogeneous.json",
        output_dir=Path(args.output_dir).resolve(),
        method="rpas",
        seed=args.seed,
        dry_run=False,
    )
    run_dir = Path(result["run_dir"])
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"implementation_status": "repository_phase2_native", "native_search": "phase2_run_search", "formal_result": True})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_mmlu(args) -> None:
    from external_comparison.runners.mmlu import evaluate_candidate, load_mmlu_split
    from external_comparison.common.protocol import RPAS_MAX_ARCHIVE_SIZE
    from experiments.phase2_wan_agent_search import configure_site_penalties, load_models, load_network_profiles, load_sites, seed_architectures
    from experiments.search_adapters.base import CandidateObservation
    from experiments.search_adapters.common_space import build_common_space_adapter

    repo_root = Path(args.repo_root).resolve()
    config_path = repo_root / "experiments" / "phase2_wan_agent_config_qwen35_9b_homogeneous.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE")
    if endpoint:
        for model in raw_config.get("models", {}).values():
            model["api_base"] = endpoint
    models = load_models(raw_config["models"])
    sites = load_sites(raw_config["sites"])
    configure_site_penalties(sites, raw_config.get("defaults", {}).get("orchestrator_site", "center_a"))
    profile = load_network_profiles(raw_config["network_profiles"])["lan_homogeneous"]
    max_tokens = int(os.environ.get("RPAS_MMLU_MAX_TOKENS", str(MMLU_MAX_TOKENS)))
    if max_tokens != MMLU_MAX_TOKENS:
        raise ValueError(f"EC-2 requires RPAS_MMLU_MAX_TOKENS={MMLU_MAX_TOKENS}, got {max_tokens}")
    eval_concurrency = max(1, int(os.environ.get("RPAS_MMLU_EVAL_CONCURRENCY", "8")))
    search_per_subject = int(os.environ.get("RPAS_MMLU_SEARCH_PER_SUBJECT", "5"))
    test_per_subject = int(os.environ.get("RPAS_MMLU_TEST_PER_SUBJECT", "10"))
    search = load_mmlu_split(args.data_dir, "dev", per_subject=search_per_subject, seed=2026)
    test = load_mmlu_split(args.data_dir, "test", per_subject=test_per_subject, seed=2026)
    sample_limit = int(os.environ.get("RPAS_NATIVE_SAMPLE_LIMIT", "0"))
    if sample_limit > 0:
        search = search[:sample_limit]
        test = test[:sample_limit]
    adapter = build_common_space_adapter("rpas")
    seed_candidates = [_protocol_mmlu_candidate(candidate, max_tokens=max_tokens) for candidate in seed_architectures(raw_config)]
    adapter.initialize(seed_candidates, raw_config, random.Random(args.seed))
    candidate_list = list(seed_candidates)
    extra = max(0, int(os.environ.get("RPAS_MMLU_NEW_CANDIDATES", "0")))
    for _ in range(min(extra, RPAS_MAX_ARCHIVE_SIZE - len(candidate_list))):
        proposal = adapter.propose()
        candidate = _protocol_mmlu_candidate(proposal.architecture, max_tokens=max_tokens)
        adapter.register_candidate(candidate)
        candidate_list.append(candidate)
    search_rows = []
    for candidate in candidate_list:
        result = evaluate_candidate(
            candidate=candidate,
            examples=search,
            models=models,
            profile=profile,
            run_id=f"mmlu-rpas-seed-{args.seed}",
            method="rpas",
            split="search",
            eval_concurrency=eval_concurrency,
        )
        row = {
            "candidate_id": candidate["id"],
            "candidate": candidate,
            "accuracy": result["accuracy"],
            "valid_answer_rate": result["valid_answer_rate"],
            "valid": result["valid"],
            "total_calls": result["calls"],
            "total_tokens": result["total_tokens"],
            "communication": result["communication"],
        }
        search_rows.append(row)
        adapter.observe(
            CandidateObservation(
                candidate["id"],
                bool(result["valid"]),
                result["accuracy"] if result["valid"] else None,
                result["calls"],
                result["total_tokens"],
                0.0,
                diagnostics={"valid_answer_rate": result["valid_answer_rate"]},
            )
        )
    eligible = [row for row in search_rows if row["valid"]]
    if not eligible:
        raise RuntimeError("RPAS MMLU search produced no candidate with a valid answer rate >= 0.99")
    selected = min(
        eligible,
        key=lambda row: (
            -float(row["accuracy"]),
            int(row["total_tokens"]),
            int(row["total_calls"]),
            str(row["candidate_id"]),
        ),
    )
    test_result = evaluate_candidate(
        candidate=selected["candidate"],
        examples=test,
        models=models,
        profile=profile,
        run_id=f"mmlu-rpas-seed-{args.seed}",
        method="rpas",
        split="test",
        eval_concurrency=eval_concurrency,
    )
    valid_answer_rate = require_valid_answer_rate(
        test_result["rows"], context=f"RPAS MMLU seed {args.seed} test"
    )
    output_dir = Path(args.output_dir) / "rpas" / f"seed_{args.seed}"
    manifest = {
        "run_id": f"mmlu-rpas-seed-{args.seed}",
        "method": "rpas",
        "dataset": "mmlu",
        "seed": args.seed,
        "implementation_status": "repository_phase2_executor_with_rpas_search_policy",
        "native_search": "rpas_typed_mutation_and_pareto_parent_selection",
        "official_repo": str(repo_root),
        "search_calls": sum(int(row["total_calls"]) for row in search_rows),
        "search_tokens": sum(int(row["total_tokens"]) for row in search_rows),
        "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "search_candidates": len(search_rows),
        "search_examples": len(search),
        "test_examples": len(test),
        "eval_concurrency": eval_concurrency,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking_disabled": True,
        "answer_parser": "strict_choice_a_b_c_d",
        "valid_answer_rate": valid_answer_rate,
        "formal_result": False,
    }
    write_native_result(output_dir, manifest, test_result["rows"], test_result["calls_detail"], selected)
    (output_dir / "search_rows.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in search_rows) + "\n", encoding="utf-8")
