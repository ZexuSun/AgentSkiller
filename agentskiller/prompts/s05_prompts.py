"""
Prompts for Step 05: Tool List Formulation

Fix blueprints and extract OpenAI-format tool lists.
Note: This step is mostly data processing with minimal LLM calls.
"""

TOOL_LIST_EXTRACTION_PROMPT = """
Format the functions in the following MCP server blueprint
```
{blueprint}
```
to meet the requirement of the API.

Output format:
```json
[
    {{
        "type": "function",
        "function": {{
            "name": "function_name",
            "description": "Function description",
            "parameters": {{
                "type": "object",
                "properties": {{
                    "param_name": {{
                        "type": "string",
                        "description": "Parameter description"
                    }}
                }},
                "required": ["param_name"]
            }}
        }}
    }}
]
```
"""
