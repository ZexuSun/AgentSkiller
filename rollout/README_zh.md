# Rollout

**Multi-Turn Tool Call Annotation Framework** — 自动化多轮对话工具调用数据标注框架

Rollout 是一个用于生成高质量多轮对话工具调用数据的框架，支持单域（Single Domain）和跨域（Cross Domain）场景的自动化标注。

## ✨ 特性

- 🔧 **统一 LLM API 调用** — 基于 LiteLLM，支持 100+ 模型提供商（OpenAI、Anthropic、DeepSeek 等）
- 🔄 **多轮对话支持** — Agent + User Simulator 自动对话生成
- 🛠️ **MCP Server 集成** — 支持 Mock MCP Server 工具模拟
- 📁 **单域/跨域场景** — 灵活支持单一领域和多领域组合场景
- ⚡ **并行处理** — 多线程批量处理，提升效率
- 💾 **断点续传** — Checkpoint 机制支持中断后恢复
- 🎯 **对话监控** — 智能检测对话终止条件，避免无效轮次

## 📁 项目结构

```
rollout/
├── run.py                     # 手动配置运行入口
├── configs/
│   ├── example_new.yml        # 配置文件示例
│   └── models.yml             # 自定义模型注册配置
├── scripts/
│   ├── batch_rollout_single.py   # 单域批量处理脚本
│   └── batch_rollout_cross.py    # 跨域批量处理脚本
├── rollout/
│   ├── core/
│   │   ├── agent.py           # Agent 实现（LiteLLM）
│   │   ├── user.py            # User Simulator 实现
│   │   ├── pipeline.py        # 对话 Pipeline
│   │   ├── checkpoint.py      # 断点续传管理
│   │   └── monitor.py         # 对话监控器
│   ├── tools/
│   │   ├── mcp_wrapper.py     # MCP Server 自动包装器
│   │   └── datasets/
│   │       ├── single domain/    # 单域 MCP Servers & 数据
│   │       └── cross domain/     # 跨域 MCP Servers & 数据
│   └── utils/
│       └── cross_domain.py    # 跨域组合发现工具
└── outputs/                   # 生成结果输出目录
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 方式一：手动配置运行

使用 `run.py` 配合 YAML 配置文件进行精细化控制：

```bash
# 基本运行
python run.py --config configs/example_new.yml

# 断点续传模式
python run.py --config configs/example_new.yml --resume

# 只处理指定数据集
python run.py --config configs/example_new.yml --dataset CustomerService

# 详细输出模式
python run.py --config configs/example_new.yml --verbose
```

### 方式二：批量处理（推荐）

在实际使用中，主要使用批量处理脚本：

#### 单域场景（Single Domain）

```bash
# 处理所有单域场景
python scripts/batch_rollout_single.py --all --max-workers 32 --output-dir ./outputs_cross_xxx

# 处理指定域
python scripts/batch_rollout_single.py --domains StudentAcademicPortal

# 列出所有可用域
python scripts/batch_rollout_single.py --list

# 自定义模型和参数
python scripts/batch_rollout_single.py --all \
    --agent-model openai/deepseek-v3.2-fc \
    --user-model openai/gpt-5 \
    --max-turns 20 \
    --output-dir ./outputs_single
```

#### 跨域场景（Cross Domain）

```bash
# 处理所有跨域组合
python scripts/batch_rollout_cross.py --all --max-workers 32 --output-dir ./outputs_single_xxx

# 处理指定跨域组合（顺序无关）
python scripts/batch_rollout_cross.py --domains StudentAcademicPortal StudentFinancialServices

# 列出所有可用跨域组合
python scripts/batch_rollout_cross.py --list --verbose

# 生成配置文件供手动审查
python scripts/batch_rollout_cross.py --generate-configs --config-output-dir configs/generated
```

## 📋 工作流程

### Cross Domain Workflow

1. **数据准备**：将 Cross Domain Workflow 生成的整个 `outputs`(包含MCP Server， Databases， Queries等等)放置到对应目录：

   - 单域Queries：`rollout/tools/datasets/single domain/queries/`
   - 跨域Queries：`rollout/tools/datasets/cross domain/queries/`
2. **运行批量处理**：

   ```bash
   # 单域
   python scripts/batch_rollout_single.py --all

   # 跨域
   python scripts/batch_rollout_cross.py --all
   ```
3. **输出结果**：处理结果将保存到 `--output-dir `指定的目录

### ⚠️ 重要提示：MCP Server 路径配置

如果 Single Domain 和 Cross Domain 两个生成结果中 MCP Server 和 Database 不一致，需要修改 `rollout/tools/mcp_wrapper.py` 中的默认路径：

```python
# rollout/tools/mcp_wrapper.py (lines 36-37)

# 使用 Single Domain 的 MCP Servers
DEFAULT_MCP_SERVERS_DIR = _ROLLOUT_PKG_DIR / "tools" / "datasets" / "single_domain" / "mcp_servers"
DEFAULT_TOOL_LISTS_DIR = _ROLLOUT_PKG_DIR / "tools" / "datasets" / "single_domain" / "tool_lists"

# 或者使用 Cross Domain 的 MCP Servers
DEFAULT_MCP_SERVERS_DIR = _ROLLOUT_PKG_DIR / "tools" / "datasets" / "cross_domain" / "mcp_servers"
DEFAULT_TOOL_LISTS_DIR = _ROLLOUT_PKG_DIR / "tools" / "datasets" / "cross_domain" / "tool_lists"
```

## ⚙️ 配置说明

### 主配置文件 (`configs/example_new.yml`)

```yaml
# 全局执行设置
max_workers: 48                    # 并行处理线程数
resume: false                      # 是否从上次中断处继续
use_checkpoints: true              # 启用断点机制

# 日志配置
log_level: INFO
log_file: ./logs/rollout.log

# 实时输出配置
verbose: true                      # 打印 agent/user/tool 实时输出
verbose_colors: true               # 使用彩色输出

# 对话监控配置（提前终止检测）
enable_monitor: true
monitor_max_no_tool_turns: 5       # 连续 N 轮无 tool call 后终止

# 模型配置文件
models_config_file: ./configs/models.yml

# 数据集配置
datasets:
  CustomerService:
    path: ./queries/cross_domain/Domain1_Domain2.jsonl
    output_path: ./outputs/output.jsonl
    mcp_domain: Domain1_Domain2
    tools:
      - domain1_mcp
      - domain2_mcp
    agent:
      model: openai/deepseek-v3.2-fc
      temperature: 0.2
      enable_thinking: true
    user:
      model: openai/gpt-5
      temperature: 1.0
    max_turns: 20
    max_steps_per_turn: 10
    mode: positive
```

### 模型注册配置 (`configs/models.yml`)

支持注册自定义/内部模型：

```yaml
models:
  deepseek-v3.2-fc:
    provider: openai
    api_base: http://your-api-endpoint/v1
    api_key: sk-your-api-key
    mode: chat

  gpt-5:
    provider: openai
    api_base: http://your-api-endpoint/v1
    api_key: sk-your-api-key
    mode: chat
```

## 📦 输出格式

每条处理结果为 JSONL 格式，包含：

```json
{
  "id": "unique_trajectory_id",
  "success": true,
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    ...
  ],
  "metadata": {
    "total_turns": 5,
    "stop_reason": "user_stop",
    "tool_call_count": 8
  }
}
```

## 🛠️ 高级用法

### 命令行参数

#### `batch_rollout_single.py`

| 参数              | 说明                      | 默认值                      |
| ----------------- | ------------------------- | --------------------------- |
| `--all`         | 处理所有域                | -                           |
| `--domains`     | 指定处理的域              | -                           |
| `--list`        | 列出所有可用域            | -                           |
| `--agent-model` | Agent 使用的模型          | `openai/deepseek-v3.2-fc` |
| `--user-model`  | User Simulator 使用的模型 | `openai/gpt-5`            |
| `--max-turns`   | 最大对话轮次              | 20                          |
| `--max-workers` | 并行线程数                | 8                           |
| `--output-dir`  | 输出目录                  | `./outputs_single_0114`   |
| `--no-resume`   | 不使用断点续传            | -                           |
| `--quiet`       | 静默模式                  | -                           |

#### `batch_rollout_cross.py`

| 参数                 | 说明                       | 默认值 |
| -------------------- | -------------------------- | ------ |
| `--all`            | 处理所有跨域组合           | -      |
| `--domains`        | 指定域组合（顺序无关）     | -      |
| `--list`           | 列出所有可用组合           | -      |
| `--merge-queries`  | 合并分散的 query 文件      | -      |
| `--require-policy` | 只处理有 policy 文件的组合 | -      |
| `--min-domains`    | 最少域数量                 | 2      |

## 🧠 DeepSeek V3.2 Reasoning 特性

Rollout 针对 DeepSeek V3.2 的 Thinking/Reasoning 模式进行了优化处理：

### Reasoning Content 清理机制

为了节省 Token 消耗，每个新的 Turn（从 User 发言开始计算）开始时，会自动清理上下文中**前面所有 Turn** 的 `reasoning_content`：

```
Turn 1:
  User: "帮我查询余额"
  Assistant: [reasoning_content: "用户想查余额..."] + [tool_call: check_balance]
  Tool: {"balance": 1000}
  Assistant: [reasoning_content: "余额是1000..."] + "您的余额是 1000 元"

Turn 2 开始时，上下文变为:
  User: "帮我查询余额"
  Assistant: [reasoning_content: null] + [tool_call: check_balance]  ← 清除
  Tool: {"balance": 1000}                                            ← 保留
  Assistant: [reasoning_content: null] + "您的余额是 1000 元"        ← 清除 reasoning，保留 content
  User: "再帮我转账"                                                  ← 新 Turn 开始
```

**保留的内容：**
- ✅ 所有 Tool Response（完整保留）
- ✅ 每个 Step 的 `content`（Assistant 的文本回复）
- ✅ 所有 `tool_calls` 信息

**清除的内容：**
- ❌ 前面所有 Turn 的 `reasoning_content`（思维链内容）

> **注意**：Trajectory 会在清理**之前**保存，因此最终输出文件中包含完整的 `reasoning_content`。清理仅影响后续对话的上下文输入。

## 🔧 后处理工具

### 添加 System Prompt 和 Tools

Rollout 生成的 Trajectory 默认不包含 System Prompt 和 Tools 信息。使用以下脚本可以将它们添加到输出文件中：

#### 单域场景

```bash
python add_label_single.py
```

配置项（在 `add_label_single.py` 中修改）：

```python
POLICY_ROOT = "rollout/tools/datasets/single_domain/policies"   # Policy 文件目录
TOOLS_LIST = "rollout/tools/datasets/single_domain/tool_lists"  # Tools 定义目录
OUTPUT_DIR = "outputs_single"                           # Rollout 输出目录
RESULT_FILE = "./mt_single_domain_tool_call_thinking.jsonl"     # 处理后的输出文件
```

#### 跨域场景

```bash
python add_label_cross.py
```

配置项（在 `add_label_cross.py` 中修改）：

```python
POLICY_ROOT = "rollout/tools/datasets/cross_domain/policies"    # Policy 文件目录
TOOLS_LIST = "rollout/tools/datasets/cross_domain/tool_lists"   # Tools 定义目录
OUTPUT_DIR = "outputs_cross"                            # Rollout 输出目录
RESULT_FILE = "./mt_cross_domain_tool_call_thinking.jsonl"    # 处理后的输出文件
```

> **注意**：跨域场景会自动处理域名顺序问题（如 `A_B_C.jsonl` 可以匹配 `C_B_A.md` 的 Policy 文件）。

### Tokenize 前数据预处理

`add_key.py` 提供了 Tokenize 前数据预处理的示例代码。可以参考其中的代码块结构来定制自己的预处理流程：

```python
# add_key.py 示例结构
for line in open("your_input_file.jsonl"):
    data = json.loads(line)
    messages = [
        {
            "role": msg.get("role", None),
            "content": msg.get("content", None),
            "reasoning_content": msg.get("reasoning_content", None),  # DeepSeek V3.2 思维链
            "tool_calls": msg.get("tool_calls", None),
            "tool_call_id": msg.get("tool_call_id", None)
        }
        for msg in data["messages"]
    ]
    new_data = {
        "messages": messages,
        "id": data.get("id", generate_id(data)),
        "data_source": "agent",        # 数据来源标识
        "use_cot": True,               # 是否使用 Chain-of-Thought
        "tools": data.get("tools", None)
    }
    f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
```

通过模仿上述代码块的写法，可以：

- 统一不同来源数据的格式
- 添加自定义字段（如 `data_source`、`use_cot`）
- 过滤或转换特定字段
- 合并多个数据源到单一文件
