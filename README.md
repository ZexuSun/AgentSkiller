<div align="center">
  <img src="assets/logo.png" width="60%" alt="AgentSkiller" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/AgentSkiller/datasets"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-%20Dataset-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://huggingface.co/AgentSkiller/models"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="">
    <img src="https://img.shields.io/badge/Blog-WeChat-07c160?logo=wechat&logoColor=white" alt="WeChat Blog">
  </a>
  <a href="http://arxiv.org/abs/2602.09372"><img alt="Code License"
    src="https://img.shields.io/badge/arXiv-2602.09372-b31b1b.svg"/></a>
</div>

<div align="center">
  <a href="README_zh.md">中文</a> | <a href="README.md">English</a>
</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub. We greatly appreciate your support.</h5>

## 1. Introduction
![AgentSkiller Architecture](./assets/main.png)

AgentSkiller is a robust framework designed to synthesize complex, high-quality data for training next-generation generalist agents. Unlike previous ad-hoc methods, AgentSkiller employs a state-machine-driven architecture orchestrated by a Directed Acyclic Graph (DAG) to ensure determinism, recoverability, and executability.

The framework produces coherent environments with deterministic state transitions, systematically broadening the space of function-calling scenarios through a rigorous pipeline—from establishing Person-Centric Entity Graphs and standardizing Model Context Protocol (MCP) blueprints, to utilizing a Persona-Based Simulator for natural language generation.

### 🏗️ Robust Architecture
AgentSkiller is built upon three core design principles that ensure the quality of the base environment:
- 🧠 Dual-Model Architecture: Decouples semantic reasoning from syntactic implementation to ensure high-quality code generation.

- ⚙️ Granular Orchestration: Features automated checkpointing for robust long-running generation tasks.

- 🛠️ Test-Driven Self-Correction: An iterative mechanism that automatically detects and corrects errors in generated code to guarantee executability.

---

## 2. Cross-Domain Task Generation

![Cross-Domain Task Generation](./assets/cross_domain.png)

While many existing frameworks focus on atomic, single-domain tasks, AgentSkiller breaks new ground by automating the synthesis of Cross-Domain Interoperability.

Real-world tasks often span multiple service boundaries (e.g., booking a medical appointment and immediately filing an insurance claim). AgentSkiller introduces a dedicated Semantic-Driven Cross-Domain Fusion phase to simulate these high-fidelity scenarios:

1. **Trajectory Interlocking & Policy Harmonization**
Instead of simple concatenation, our system performs deep semantic fusion:
    - **Semantic Linking**: We link distinct workflows (e.g., Airline and Hotel) via shared core entities, synthesizing coherent storylines that require multi-hop reasoning.
    - **Unified Governance**: An LLM-based mediator resolves conflicting rules between domains (e.g., privacy vs. data sharing) and synthesizes "Bridge Rules" to govern the interface between services.
2. **Namespace-Isolated Context**
To support execution, we implement a Database Fusion module that aggregates entities while preventing schema collisions. By enforcing a Namespace Isolation Policy, relationships maintain their domain specificity (e.g., `Hospital_Patient` vs. `Insurance_Client`), allowing the system to verify constraints without ambiguity.
3. **Feasibility-Aware Efficiency**
To handle the combinatorial explosion of domain pairs, we employ Single Domain Feasibility Filtering. If a task segment is invalid in a single domain, the system prunes the cross-domain trajectory ex ante, ensuring computational resources are focused only on viable, high-value combinations.

## 3. Main Results Comparison

![Main Results Comparison](./assets/main_results_comparison.svg)

To rigorously validate the utility of the proposed framework, we synthesized a corpus comprising approximately 11k multi-turn interaction trajectories using AgentSkiller. Subsequent experiments across challenging function-calling benchmarks, including $\tau$-bench, $\tau^2$-bench and ACEBench, demonstrate that models trained on this dataset yield substantial performance gains. Notably, the AgentSkiller-14B exhibits exceptional capability in complex tool-use scenarios, consistently outperforming established open-source baselines and achieving parity with state-of-the-art proprietary models.

## 4. Dataset & Models

| Resource | Description |
| -------- | ----------- |
| AgentSkiller-11K | [🤗Hugging Face Dataset](https://huggingface.co/datasets/AgentSkiller/AgentSkiller-11K) |
| AgentSkiller-4B | [🤗Hugging Face Models](https://huggingface.co/AgentSkiller/AgentSkiller-4B) |
| AgentSkiller-8B | [🤗Hugging Face Models](https://huggingface.co/AgentSkiller/AgentSkiller-8B) |
| AgentSkiller-14B | [🤗Hugging Face Models](https://huggingface.co/AgentSkiller/AgentSkiller-14B) |

## ⚙️ Install

```bash
conda create -n agentSkiller python=3.11
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1) Synthesize Tasks / Queries

From repo root:

```bash
python -m agentskiller run --config config.yaml
```

This will generate evaluation-ready artifacts under `outputs/`.

### 2) Collect Rollouts

Rollout collection has its own dependencies and entrypoints. See:

- `rollout/README.md` (English)
- `rollout/README_zh.md` (中文)

### 3) Evaluate Rollouts

```bash
python -m evaluator.run_evaluation --mode all \
  --rollouts-dir rollouts/ \
  --outputs-dir outputs/ \
  --mcp-outputs-dir outputs/ \
  --output outputs/evaluation/results.jsonl
```

## 👀 AgentSkiller Workflow Overview

* **Single Domain**: Step `01` – `09` & Step `14` – `17`
* **Cross Domain**: Step `01` – `09` & Step `10` – `13` & Step `14` – `17`

### Step-by-Step Guide (Quick Reference)
|Step|Name|Function|Primary Artifacts (Default in `outputs/`)|Note|
|-|-|-|-|-|
|s01|domain_expansion|Expand seed domains|`domain_topics.json`||
|s02|entity_extraction|Extract entities|`entities.json`||
|s03|entity_graph|Construct entity graph|`entity_graph.json`||
|s04|blueprint_generation|Generate MCP blueprints|`blueprints.json`||
|s05|tool_list_formulation|Repair blueprints and export tool lists|`blueprints.json`, `tool_lists/*.json`||
|s06|database_generation|Generate entity/relationship databases and summaries|`database/`, `database_summary/`|Code generation + Execution|
|s07|policy_generation|Generate domain policy|`policies/*.md`|With structured markers (for filtering)|
|s08|tool_graph_generation|Generate tool dependency graph|`tool_graphs/*.json`||
|s09|mcp_server_implementation|Implement MCP server + tests|`mcp_servers/*.py`||
|s10|domain_combos_selection|Select cross-domain combinations|`cross_domain_templates/_combinations.json`|**Cross-domain only**|
|s11|trajectory_fusion|Cross-domain trajectory fusion|`cross_domain_templates/*.json`|**Cross-domain only**|
|s12|database_fusion|Cross-domain database fusion|`database/outputs/relationships/{fused}/*.json` `database/outputs/entities/{fused}/*.json`|**Cross-domain only**|
|s13|policy_merge|Cross-domain policy merge|`policies/{fused}.md`|**Cross-domain only**|
|s14|task_template_generation|Generate task templates|`task_templates/*.json`||
|s15|instance_combos_selection|Select/generate instance combinations for templates|`combinations/**` or `validated_tasks/**`|Single-domain: Sampling; Cross-domain: Creation-Validation|
|s16|task_filtering|Execute trajectory validation filtering|`validated_tasks/**`|Required for Single Domain only|
|s17|task_instantiation|Instantiate tasks and generate queries|`queries/*.jsonl`|Instantiation + Query generation|

## 📦 What gets produced

- **Synthesis outputs**: `outputs/` (queries, generated MCP servers, databases, policies, etc.)
- **Collected rollouts**: `rollouts/` (JSONL conversations with tool calls; produced by the rollout module)
- **Evaluation results**: `outputs/evaluation/results.jsonl` (from the evaluator)

## 🧩 Modules

- **`agentskiller/` (synthesis)**: generate MCP servers, databases, tasks, and queries into `outputs/`See `agentskiller/README.md`.
- **`rollout/` (data collection)**: run an LLM-simulated user + assistant to produce multi-turn rolloutsSee `rollout/README.md` / `rollout/README_zh.md`.
- **`evaluator/` (evaluation)**: execute golden trajectories and score rollouts with multiple evaluators
  See `evaluator/README.md`.


## 🔗 Citation

If you find this work useful, please kindly cite:
```
@misc{sun2026agentskillerscalinggeneralistagent,
      title={AgentSkiller: Scaling Generalist Agent Intelligence through Semantically Integrated Cross-Domain Data Synthesis}, 
      author={Zexu Sun and Bokai Ji and Hengyi Cai and Shuaiqiang Wang and Lei Wang and Guangxia Li and Xu Chen},
      year={2026},
      eprint={2602.09372},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.09372}, 
}
```