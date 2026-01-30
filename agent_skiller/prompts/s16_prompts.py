"""
Prompts for Step 16: Task Filtering

Filter valid tasks by executing trajectories.
Note: This step is mostly validation with minimal LLM calls.
"""

# s16 primarily executes trajectories to validate tasks
# LLM prompts are used for error analysis if needed

TRAJECTORY_VALIDATION_ERROR_ANALYSIS_PROMPT = """
A task execution failed during trajectory validation.

## Task Information
Trajectory: {trajectory}
Task Template: {task_template}
Entity Context: {entity_context}

## Error Information
Failed at tool: {failed_tool}
Error message: {error_message}
Tool output: {tool_output}

## Analysis Task
Determine the root cause of the failure:
1. Is it a constraint violation in the policy?
2. Is it missing or invalid entity data?
3. Is it an issue with the instance combination selection?
4. Is it a bug in the MCP server implementation?

Output JSON:
```json
{{
    "root_cause": "policy_violation|invalid_data|selection_error|server_bug",
    "explanation": "detailed explanation of what went wrong",
    "suggested_fix": "how to fix the issue",
    "is_recoverable": true/false
}}
```
"""
