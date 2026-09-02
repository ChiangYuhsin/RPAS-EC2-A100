from __future__ import annotations

import json
import random

import pytest

from experiments.phase2_wan_agent_search import seed_architectures
from experiments.search_adapters.base import CandidateObservation
from experiments.search_adapters.fake import DeterministicFakeAdapter
from experiments.search_adapters.registry import build_adapter


def test_fake_adapter_receives_search_only() -> None:
    adapter = DeterministicFakeAdapter()
    adapter.initialize([], {"topologies": ["single"]}, random.Random(0))
    proposal = adapter.propose()
    adapter.observe(CandidateObservation(proposal.candidate_id, True, 1.0, 1, 1, 0.0))
    with pytest.raises(ValueError, match="search observations only"):
        adapter.observe(CandidateObservation(proposal.candidate_id, True, 1.0, 1, 1, 0.0, split="test"))


def test_fake_adapter_state_round_trip() -> None:
    first = DeterministicFakeAdapter()
    first.initialize([], {"topologies": ["single"]}, random.Random(4))
    proposal = first.propose()
    first.observe(CandidateObservation(proposal.candidate_id, True, 1.0, 1, 1, 0.0))

    second = DeterministicFakeAdapter()
    second.load_state_dict(first.state_dict())
    assert second.state_dict() == first.state_dict()


@pytest.mark.parametrize("method_id", ["random_as", "aflow_style", "adas_style", "rpas_quality", "rpas"])
def test_common_space_adapters_share_typed_contract(method_id: str) -> None:
    config = json.loads(open("experiments/phase2_wan_agent_config_qwen35_9b_homogeneous.json", encoding="utf-8").read())
    seeds = seed_architectures(config)
    adapter = build_adapter(method_id)
    adapter.initialize(seeds, config, random.Random(0))
    for seed in seeds:
        adapter.observe(CandidateObservation(seed["id"], True, 0.5, 1, 10, 0.0, diagnostics={"architecture": seed}))
    proposal = adapter.propose()
    assert proposal.candidate_id == proposal.architecture["id"]
    assert proposal.architecture["topology"] in config["allowed_topologies"]
    adapter.observe(
        CandidateObservation(
            proposal.candidate_id, True, 0.6, 1, 11, 0.0, diagnostics={"architecture": proposal.architecture}
        )
    )
    assert adapter.state_dict()["rows"][proposal.candidate_id]["score"] == 0.6
