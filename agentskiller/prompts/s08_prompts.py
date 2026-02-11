"""
Prompts for Step 08: Tool Graph Generation

Generate tool execution dependency graphs for MCP servers.
"""

# =============================================================================
# Tool Graph Construction
# =============================================================================

TOOL_GRAPH_PROMPT = """
You are an expert in reasoning about tool dependencies in modular agent systems.
Given a set of tool descriptions for {domain_name},

<tools>
{tools}
</tools>

and the policy of the system

<policy>
{policy}
</policy>

your task is to construct a **Tool Execution Graph**.

### Guidelines
1. Each node represents one tool function.
2. Each edge represents an **execution dependency**.
3. Use directed edges (`"source" → "target"`) for:
    - **State Dependencies** (e.g., authentication, record retrieval)
    - **Information Flow** (output of one tool is required by another)
    - **Storyline Flow** (a tool is usually followed by another tool in real-world scenario).
4. Never omit the nodes list — ensure every mentioned tool exists as a node.
5. **Single Source of Truth:** The node `Authorize` must be the **ONLY** node with an In-Degree of 0. It is the starting point of the entire workflow.
6. **Full Connectivity:** Every node (except `Authorize`) must have at least one incoming edge (In-Degree ≥ 1). This ensures all nodes are reachable from `Authorize`.
7. **Acyclicity:** The graph must strictly be a DAG. No circular dependencies (e.g., A->B->A is forbidden).
8. **Storyline Flow:** Consider this connection ONLY when the full connectivity constraint CANNOT be satisfied by the other two types of dependencies.
8. The output must be valid JSON that can be loaded by networkx.
9. Output JSON in the following schema without any extra explanations.
```json
{{
    "directed": true,
    "multigraph": false,
    "nodes": [
        {{"id": "tool_name"}},
        ...
    ],
    "links": [
        {{"source": "tool_a", "target": "tool_b", "type": "state dependency"}},
        {{"source": "tool_a", "target": "tool_c", "type": "information flow"}},
        {{"source": "tool_a", "target": "tool_d", "type": "storyline flow"}},
        ...
    ]
}}
```
"""
