# External Comparison Foundation

这是外部对比实验的基础设施层，不是实验结果。当前目标是先固定比较边界和统计口径，再接入各论文的原生搜索实现。

## HumanEval common-space runner

`runners/humaneval.py` 已提供可审计的 controlled-search 共同空间执行路径：HumanEval 数据按
`data_seed=2026` 固定切分为 `D_search/D_select/D_test`，候选由
`experiments/search_adapters/` 提议，代码由独立 Python 子进程执行并以 pass@1 评分。
runner 统一生成调用、通信、候选、选择和 provenance artifact，并支持共享 search/select cache
与 `--resume`。它默认写入 `formal_result: false`，在 G1-G9 门禁完成前不得作为正式结果。

示例（只做 split/manifest 检查，不调用模型）：

```bash
python -m external_comparison.runners.humaneval \
  --dataset-path /path/to/HumanEval.json \
  --method random_as --dry-run
```

共同空间方法名是 `random_as`、`aflow_style`、`adas_style`、`rpas_quality` 和 `rpas`。
这些方法是受控共域比较策略，不是对应论文官方代码的完整复现，不能写成 AFlow、MaAS 或 G-Designer。

正式外部实验配置是 `configs/ec1_humaneval.json` 和 `configs/ec2_mmlu.json`。
它们只接受 repository-local native adapter；`validate_protocol.py --require-native`
会在缺失时失败。

## 当前 EC-2 结果边界

仓库中已有的 EC-2 结果是 `MMLU-57x10 controlled subset`：57 个 subject、每个 subject 10 道测试题，搜索集每 subject 5 题。它们统一保留 `formal_result: false`，在 G1-G9 门禁完成前不能写成 formal result 或完整 MMLU。

RPAS 本次运行是 9 个预定义候选架构上的 controlled candidate selection，`RPAS_MMLU_NEW_CANDIDATES=0`；这不是完整 reflective mutation search。G-Designer 的 `search_calls=0` 表示没有单独 instrumented 的搜索阶段，不表示没有额外推理调用。论文表格必须同时报告 test inference calls/tokens、search calls/tokens 和 total calls/tokens。

可从每个 seed artifact 生成主表：

```bash
python -m external_comparison.runners.aggregate_mmlu \
  --root outputs/external_comparison/ec2_gpu6 \
  --output-dir outputs/external_comparison/ec2_gpu6/aggregate
```

## 实验主线

| 实验 | 主要回答的问题 | 首选方法 |
|---|---|---|
| EC-1 HumanEval | 不同架构搜索方法在统一执行器下的质量--成本 Pareto | RPAS / AFlow / MaAS |
| EC-2 MMLU | 通信拓扑是否带来可测的通信收益 | RPAS / G-Designer |
| EC-3 HotpotQA | workflow 搜索能否跨任务结构泛化 | RPAS / AFlow |
| EC-4 Transfer（可选） | 搜出的拓扑能否跨 backbone 复用 | RPAS / MaAS / AFlow |

## 当前已经准备好的内容

- `configs/`：三组实验协议配置，不包含 API key、绝对路径或结果数字。
- `common/schema.py`：统一的 call、candidate、run manifest 数据结构。
- `common/telemetry.py`：JSONL 轨迹和 tokens/calls/cost/network accounting。
- `common/manifest.py`：协议、数据、源码和配置的 SHA-256 追溯接口。
- `common/pareto.py`：统一的 Pareto、Quality operating point 和 Efficiency operating point 选择规则。
- `adapters/`：RPAS artifact reader 和外部方法的明确适配器占位。
- `runners/validate_protocol.py`：只做静态校验，不发起模型/API 调用。

## 重要边界

1. `EXPERIMENT_PROTOCOL.md` 是冻结实验协议的权威文件；附件研读笔记是外部扩展实验建议，若有冲突以仓库协议为准。
2. 不把原论文中的数字直接抄进表格；每个方法必须在同一 executor、数据划分、模型、解码、评测器和 telemetry 边界下重新跑，或明确标注为 original-paper-fidelity appendix。
3. 不强迫不同方法使用同样的迭代次数；比较 realized task-model calls/tokens 和累计预算曲线。
4. test split 不参与搜索、重排、停止、调参或选择。
5. 当前没有接入 GPTSwarm、AFlow、G-Designer、MaAS 的可运行原生实现，因此不能宣称已有外部对比结果。

## 下一步运行顺序

1. 冻结 HumanEval 数据 checksum、executor、model endpoint、decoding 和 answer evaluator。
2. 先接入 RPAS artifact normalization，再接入 EC-1 的原生 baseline adapters。
3. 用 dry-run 检查配置、manifest、telemetry 和预算账本。
4. 通过 valid-rate 和 data-leakage gates 后，才允许发起正式搜索。
