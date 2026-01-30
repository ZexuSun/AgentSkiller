# Agent Skiller

Agent Skiller 是一套端到端流水线，用于**合成可调用工具的任务与查询**、**采集多轮 agent rollouts**，并**基于 golden trajectory 对 rollouts 做评测**。

如果你需要英文版，请看 [`README.md`](README.md)。

---

## ⚙️ 安装

```bash
conda create -n agentSkiller python=3.11
pip install -r requirements.txt
```

---

## 🚀 Demo（端到端）

### 1）合成任务/查询（Agent Skiller）

```bash
python -m agent_skiller run --config config.yaml
```

会在 `outputs/` 下生成评测所需产物（queries、生成的 MCP server、数据库等）。

### 2）（可选）采集 rollouts

采集模块有独立依赖与入口，见：

- `rollout/README_zh.md`
- `rollout/README.md`

### 3）评测 rollouts

```bash
python -m evaluator.run_evaluation --mode all \
  --rollouts-dir rollouts/ \
  --outputs-dir outputs/ \
  --mcp-outputs-dir outputs/ \
  --output outputs/evaluation/results.jsonl
```

---

## 📦 产物在哪里

- **合成产物**：`outputs/`（queries、生成的 MCP servers、数据库、policies 等）
- **对话 rollouts**：`rollouts/`（JSONL，多轮对话 + tool calls；由 rollout 模块产出）
- **评测结果**：`outputs/evaluation/results.jsonl`

---

## 🧩 模块入口

- **`agent_skiller/`（合成）**：生成 MCP servers、数据库、任务与 queries，写入 `outputs/`  
  见 `agent_skiller/README_zh.md` / `agent_skiller/README.md`
- **`rollout/`（采集）**：LLM Simulated User 与 Assistant 自动对话，产出多轮 rollouts  
  见 `rollout/README_zh.md` / `rollout/README.md`
- **`evaluator/`（评测）**：执行 golden trajectory 并用多 evaluator 打分  
  见 `evaluator/README_zh.md` / `evaluator/README.md`
