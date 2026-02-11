"""
Prompts for Step 10: Domain Combos Selection (Cross-Domain)

Select cross-domain combinations for task generation.
"""

DOMAIN_COMBINATION_PROMPT = """
## MCP Server Domain Selection

You are given MCP Server blueprints that all share the same core entity: **{core_entity}**

```json
{blueprints}
```

Your task: Generate **{n_combinations}** different semantically meaningful combinations.

Requirements:
- Each combination should have 2~3 domains
- Domains in each combination should be semantically related (e.g., can form a realistic cross-domain workflow)
- Each combination should represent a unique real-world scenario
- Maximize diversity across combinations

Output format (JSON array of arrays):
```json
[
    ["DomainA", "DomainB"],
    ["DomainA", "DomainC", "DomainD"],
    ...
]
```

Output only the JSON array, no explanations.
"""