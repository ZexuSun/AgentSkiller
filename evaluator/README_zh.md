# `evaluator/`

`evaluator/` 用于对 agent rollouts 做评测：通过 **TrajectoryExecutor 执行 golden trajectories**，并结合多种 evaluator（action/environment/nl_assertions/…）对结果进行打分与诊断。

端到端用法（合成 → 采集 → 评测）请看根目录 [`README_zh.md`](../README_zh.md)。

---

## 运行评测

```bash
python -m evaluator.run_evaluation --mode all \
  --rollouts-dir rollouts/ \
  --outputs-dir outputs/ \
  --mcp-outputs-dir outputs/ \
  --output outputs/evaluation/results.jsonl
```

---

## 常用模式

```bash
# 只做 pruning（分析 golden trajectory 冗余步骤，生成 pruning index）
python -m evaluator.run_evaluation --mode prune

# 跑全部 evaluator（跳过 pruning）
python -m evaluator.run_evaluation --mode all --no-prune

# 单项评测（示例：action）
python -m evaluator.run_evaluation --mode single --evaluator action
```

---

## 评测依赖的数据

- **rollouts**：`rollouts/*.jsonl`（多轮对话 + tool calls）
- **queries**：`outputs/queries/*.jsonl`（合成工作流产出的评测任务定义）
- **mcp servers / tool lists / databases**：均在 `outputs/` 下
