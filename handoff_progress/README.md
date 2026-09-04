# Minimal Resume State

This directory intentionally contains only the completed computation needed to
avoid repeating EC-1 seed-0 pilot work. It is not a result release and must not
be used as a paper main-table artifact.

## AFlow seed 0

`ec1_seed_0/aflow/workflows/` stores the generated search history, including
the selected workflow and validation-only selection record. It was produced by
the pinned official AFlow optimizer. `run_manifest.json`,
`_aflow_driver_result.json`, and `_native_aflow_calls.jsonl` preserve the
provenance and accounting needed to audit that pilot. The original held-out
evaluation had already completed; these files prevent reconstructing the
search merely to inspect it.

## MaAS seed 0

`ec1_seed_0/maas/HumanEval_controller_sample4.pth` is the compact controller
state produced after the official fresh MaAS training phase. It is not the
Qwen executor or a foundation-model checkpoint. The original process was in
its held-out test phase when this handoff was made. Its integrity is:

```text
SHA-256: cc51e468a6fce8c32053fff83de459c0e1d3f43c5adf038b3b0636e2e7f40bdf
bytes: 548673
```

Use the documented `RPAS_MAAS_TEST_ONLY=1` continuation to stage a clean
pinned MaAS checkout, copy this state to the location expected by the official
test branch, and execute only that held-out test. The continuation is a
recovery mechanism for this interrupted pilot, not a replacement for an
independent formal seed.
