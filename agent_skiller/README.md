# `agent_skiller/`

`agent_skiller/` is the **synthesis workflow**: it generates MCP servers, databases, task templates, validated tasks, and evaluation queries into `outputs/`.

For the end-to-end flow (including rollouts + evaluation), see the root [`README.md`](../README.md).

---

## Run

From repo root:

```bash
python -m agent_skiller run --config config.yaml
```

Useful commands:

```bash
# Check whether each step is completed (roughly inferred by whether outputs exist)
python -m agent_skiller status --config config.yaml

# List all steps
python -m agent_skiller list-steps

# Run a single step (prefix match supported, e.g. s15)
python -m agent_skiller run --step s15 --config config.yaml

# Resume from a step (checkpoint / continue)
python -m agent_skiller run --from s14_task_template_generation --config config.yaml
```

---

## Configuration

- **`config.yaml`**: workflow switches and paths (notably `paths.outputs_dir`, default `outputs/`)
- **`models.yaml`**: model registry + defaults (used for LiteLLM registration)

---

## Key artifacts under `outputs/`

- `outputs/queries/`: JSONL tasks/queries for evaluation
- `outputs/mcp_servers/`: generated MCP server implementations
- `outputs/database/` + `outputs/database_summary/`: generated data + schema summaries
- `outputs/tool_lists/`: tool specs (OpenAI-format)
