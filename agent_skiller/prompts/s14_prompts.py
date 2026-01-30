"""
Prompts for Step 14: Task Template Generation

Generate task templates from tool graphs.
This unified prompt works for both single-domain and cross-domain scenarios.
"""

# =============================================================================
# Tool Preconditions Extraction
# =============================================================================

TOOL_PRECONDITIONS_PROMPT = """
Given the following policy for MCP Server:

<policy>
{policy}
</policy>

For each tool defined in the policy, analyze its Preconditions section and determine 
which other tools MUST be executed **within the same trajectory** before this tool can be called.

## What to INCLUDE as preconditions:

1. **Session/Authorization dependencies**: Tools that establish session state required by other tools
   - Example: "session must be authorized" → authorize_* tool is required
   
2. **Information flow dependencies**: Tool B needs a specific output (e.g., an ID) that is ONLY available from Tool A's response
   - Example: create_badge_credential returns badge_id, which assign_badge_to_tenant needs as input
   - This means assign_badge_to_tenant depends on create_badge_credential

## What to EXCLUDE (NOT preconditions):

1. **Data existence dependencies**: When a tool operates on records that could already exist in the database
   - Example: revoke_badge_from_tenant operates on existing badge assignments
   - The assignment could exist in the database's initial state, NOT necessarily created in this trajectory
   - So revoke_badge_from_tenant does NOT depend on assign_badge_to_tenant
   
2. **Update/Delete/Revoke operations**: These typically operate on pre-existing records
   - revoke_*, delete_*, update_*, cancel_* tools usually do NOT require the corresponding create/assign tool
   - The records they operate on may already exist in the database

## Key Principle:
Only include dependencies where Tool B **cannot possibly execute** without Tool A being called first **in the same trajectory**.
If Tool B can execute successfully when the required data already exists in the database (from initial state or previous sessions), then there is NO dependency.

Output a JSON object mapping each tool name to its list of required predecessor tools:
```json
{{
  "tool_name_1": [],
  "tool_name_2": ["tool_name_1"],
  "tool_name_3": ["tool_name_1", "tool_name_2"],
  ...
}}
```

Output only the JSON object, no explanations.
"""

# =============================================================================
# Task Template Generation
# =============================================================================

TASK_TEMPLATE_PROMPT = """
Given the following structured tool call skeleton:
<tool_call_trajectory>
{trajectory}
</tool_call_trajectory>

The required input parameters and outcomes of each tool call are specified in the policies below:
<policies>
{policies}
</policies>

Available entities and their attributes are specified in the database summary below:
<database_summary>
{database_summary}
</database_summary>

Please rewrite the tool call skeleton into a natural-language task template that sounds realistic and human-driven.

Think step by step:
1. Identify the required input parameters and outcomes of each tool call
2. Identify which of these parameters are required to be known by the user in the user's initial context
3. Identify which of these parameters can be inferred from the context of the previous tool call
4. The task template should only include the parameters that are required to be known by the user in the user's initial context

Task template requirements:
1. Have a clear or inferable motivation — the user's goal should make sense
2. Avoid obviously pointless or self-contradictory actions
3. Introduce natural variability in tone, specificity, and intent
4. Optionally include uncertainty or context
5. Can be satisfied with the instances in the database
6. The template should guarantee the execution of all tools in the trajectory
7. 'If ... then ...' is NOT allowed in the task template
8. Leave all variables as '<EntityName.attribute>' format, do not assume any values for them

Return the rewritten task template in the format of:
```json
[
    {{
        "instruction": "a short, self-contained user instruction that could plausibly trigger the tool sequence",
        "reason_for_call": "the underlying motivation"
    }}
]
```

Output only the JSON array, no explanations.
"""


TASK_TEMPLATE_JUDGE_PROMPT = """
You are evaluating the quality of a task template generated from a tool call trajectory.

<trajectory>
{trajectory}
</trajectory>

<task_template>
{task_template}
</task_template>

Please evaluate this task template on the following dimensions (1-5 scale, where 5 is best):

1. **Motivation Clarity** (1-5): Does the task have a clear, realistic user motivation? Would a real user actually want to do this?
2. **Logical Coherence** (1-5): Does the task make logical sense? Are there any contradictions or nonsensical elements?
3. **Completeness** (1-5): Does the task naturally lead to executing ALL tools in the trajectory? Would the agent need to call all these tools to fulfill the request?
4. **Naturalness** (1-5): Does this sound like something a real user would ask? Is the language natural and not overly technical?
5. **Specificity** (1-5): Is the task specific enough to be actionable? Does it provide enough context without being overly verbose?

Scoring guidelines:
- 5: Excellent - No issues, high quality
- 4: Good - Minor issues that don't affect usability
- 3: Acceptable - Some issues but still usable
- 2: Poor - Significant issues that affect quality
- 1: Unacceptable - Major issues, should be rejected

Return your evaluation in JSON format:
```json
{{
    "motivation_clarity": <score>,
    "logical_coherence": <score>,
    "completeness": <score>,
    "naturalness": <score>,
    "specificity": <score>,
    "overall_score": <average of all scores>,
    "rejection_reason": "<null if overall >= 3, otherwise brief explanation of why this template should be rejected>"
}}
```

Output only the JSON object, no explanations.
"""
