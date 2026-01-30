"""
Prompts for Step 13: Policy Merge (Cross-Domain)

Merge domain policies for cross-domain tasks.
"""

POLICY_MERGE_STRUCTURE_INSTRUCTIONS = """
IMPORTANT: When merging policies, preserve the structured markers:

1. Keep all <!-- TOOL: xxx --> and <!-- TOOL_END: xxx --> markers for each tool
2. Merge Global Rules sections from all policies and mark the combined section with <!-- GLOBAL_RULES_END -->
3. Wrap the entire merged policy with <!-- POLICY_START --> and <!-- POLICY_END -->
4. When tools from different domains have similar names, keep them distinct with their original markers
5. Add any cross-domain interaction rules in the Global Rules section
"""


POLICY_MERGE_PROMPT = """
<domain_names>
{domain_names}
</domain_names>

<policies>
{policies}
</policies>

The above describes several domains and their corresponding policies.
I'm trying to build an MCP server that maintains these domains as components.
Thus, your job is to merge these domain policies into unified policies for the combined MCP server.

Requirements:
- Keep all original policies in these domains (these guarantee completeness)
- You are allowed to add new cross-domain policies to complete the combined MCP server
- Resolve any conflicts between domains
- Add cross-domain interaction rules

{policy_merge_structure_instructions}

All tool names to preserve: {tool_names}

Directly output the merged domain policy with the structure markers preserved, no extra explanations.
"""
