# Agent Skiller

Agent Skiller is an end-to-end pipeline for **synthesizing tool-using tasks & queries**, **collecting multi-turn agent rollouts**, and **evaluating rollouts against golden trajectories**.

If you prefer Chinese docs, see [`README_zh.md`](README_zh.md).

---

## ⚙️ Install

```bash
conda create -n agentSkiller python=3.11
pip install -r requirements.txt
```

---

## 🚀 Demo (end-to-end)

### 1) Synthesize tasks/queries (Agent Skiller)

From repo root:

```bash
python -m agent_skiller run --config config.yaml
```

This will generate evaluation-ready artifacts under `outputs/`.

### 2) Collect rollouts

Rollout collection has its own dependencies and entrypoints. See:

- `rollout/README.md` (English)
- `rollout/README_zh.md` (中文)

### 3) Evaluate rollouts

```bash
python -m evaluator.run_evaluation --mode all \
  --rollouts-dir rollouts/ \
  --outputs-dir outputs/ \
  --mcp-outputs-dir outputs/ \
  --output outputs/evaluation/results.jsonl
```

---

## 📦 What gets produced

- **Synthesis outputs**: `outputs/` (queries, generated MCP servers, databases, policies, etc.)
- **Collected rollouts**: `rollouts/` (JSONL conversations with tool calls; produced by the rollout module)
- **Evaluation results**: `outputs/evaluation/results.jsonl` (from the evaluator)

---

## 🧩 Modules

- **`agent_skiller/` (synthesis)**: generate MCP servers, databases, tasks, and queries into `outputs/`See `agent_skiller/README.md`.
- **`rollout/` (data collection)**: run an LLM-simulated user + assistant to produce multi-turn rolloutsSee `rollout/README.md` / `rollout/README_zh.md`.
- **`evaluator/` (evaluation)**: execute golden trajectories and score rollouts with multiple evaluators
  See `evaluator/README.md`.
