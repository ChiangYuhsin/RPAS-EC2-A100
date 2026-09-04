# EC-3 HotpotQA Public Fixture Manifest

This directory contains only the two fixed AFlow-era HotpotQA fixtures used to
construct the EC-3 V3 formal pool. It does not contain model outputs, search
traces, evaluation predictions, credentials, or a calibration split.

| File | Role | Rows | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `source_aflow_mirror/hotpotqa_validate.jsonl` | AFlow validation fixture; split into `D_search=120` and `D_select=80` | 200 | 1,304,699 | `a2fcafe6cb44c705b48e404ab4e9a2726874cfdecffed5ddfef06e46b6c78639` |
| `source_aflow_mirror/hotpotqa_test.jsonl` | Held-out AFlow fixture; used unchanged as `D_test=800` | 800 | 5,084,520 | `9b5a171af942e75d5cb7e8746dbc6639f381900467c02e8c2e27eceb77b9a8ce` |

## Provenance and Boundary

- Upstream framework: `FoundationAgents/AFlow@3f457218fc716093fe53f6df8a5d5e6379d66346`.
- Original AFlow distribution: `aflow_data.tar.gz`, Google Drive object
  `1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e`.
- Materialized source used here: `CitrusYL/AgentSlimming@0bb1afc677e3751e09dc535e373f0316b0a8369f`, which describes these fixtures as inherited from AFlow.
- Benchmark attribution: HotpotQA, Yang et al. (EMNLP 2018), distributed under
  CC BY-SA 4.0. Preserve the attribution and share-alike terms when redistributing.

These files are AFlow-derived fixtures, not a claim that they are the canonical
upstream HotpotQA release. EC-3 uses only the provided `context` and `question`
as model input; answers and question type are retained exclusively for split
stratification and evaluation. A formal calibration split (`D_calib=40`) must
be obtained separately from the official HotpotQA distractor training data and
verified disjoint from all 1,000 IDs above before a run is unlocked.
