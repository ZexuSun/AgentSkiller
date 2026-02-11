# `agentskiller/`

`agentskiller/` 是**数据合成工作流**：用于生成 MCP servers、数据库、任务模板、验证通过的任务组合，以及评测用的 queries，并写入 `outputs/`。

端到端用法（包含 rollout 采集与 evaluator 评测）请看根目录 [`README_zh.md`](../README_zh.md)。

---

## 运行

在仓库根目录执行：

```bash
python -m agentskiller run --config config.yaml
```

常用调试命令：

```bash
# 查看当前 step 完成情况（通过 outputs 是否存在粗略判断）
python -m agentskiller status --config config.yaml

# 列出所有 steps
python -m agentskiller list-steps

# 只跑某一步（支持前缀匹配，如 s15）
python -m agentskiller run --step s15 --config config.yaml

# 从某一步开始继续跑（断点续跑）
python -m agentskiller run --from s14_task_template_generation --config config.yaml
```

---

## 配置

- **`config.yaml`**：workflow 的开关与路径（重点是 `paths.outputs_dir`，默认 `outputs/`）
- **`models.yaml`**：模型注册与默认选择（用于注册到 LiteLLM）

---

## `outputs/` 下的关键产物

- `outputs/queries/`：评测任务定义（JSONL）
- `outputs/mcp_servers/`：生成的 MCP server 实现代码
- `outputs/database/` + `outputs/database_summary/`：生成的数据与 schema/约束摘要
- `outputs/tool_lists/`：工具定义（OpenAI 格式）
