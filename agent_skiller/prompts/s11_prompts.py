"""
Prompts for Step 11: Trajectory Fusion (Cross-Domain)

Fuse tool trajectories across domains.
"""

TRAJECTORY_FUSION_PROMPT = """
You are given several tool call trajectories and domain blueprints from several relevant domains.
<blueprint>
{blueprint}
</blueprint>

<tool_call_trajectories>
{tool_call_trajectories}
</tool_call_trajectories>

Your job is to identify cross-domain tasks that require the execution of trajectories from all provided domains.
- Each domain should contribute exactly one trajectory to the task
- The combined task should be reasonable, meaning the user should have a consistent motivation
- Exhaust all reasonable combinations

Output combinations in the format of:
```json
[
    {{
        "trajectories": [
            {{"MCP_server_name": [trajectory]}},
            ...
        ],
        "motivation": ""
    }}
]
```
"""