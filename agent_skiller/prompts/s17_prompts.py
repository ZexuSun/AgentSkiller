"""
Prompts for Step 17: Task Instantiation

Instantiate tasks with entity values.
"""


TASK_INSTANTIATION_PROMPT = """
## Task Instantiation

Here is the Tool Call Trajectory that the user expects the assistant to perform:
<trajectory>
{trajectory}
</trajectory>

These blueprints describe the MCP server relevant to the task:
<blueprints>
{blueprints}
</blueprints>

Here are the detailed descriptions of each tool in the trajectory:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>

Here is the task template:
<task_template>
{task_template}
</task_template>

Here are the entity instances for context:
<entity_instances>
{entity_instances}
</entity_instances>

Here are the REQUIRED parameter values that MUST ALL appear in the instruction:
<required_values>
{required_values}
</required_values>

You are a **Data-Grounded Scenario Simulator**.
Your goal is to simulate a realistic user scenario based on specific database states.
You operate in two strict modes simultaneously:
1.  **Strict Data Clerk:** You must use the EXACT values provided. **NO HALLUCINATIONS.**
2.  **Empathetic User Simulator:** You must translate function-centric parameters into natural user language.

### Phase 1: Data Grounding (Strict Rules)

**THE MARKING RULE:**
For every specific value you insert into the `instruction`, **you MUST wrap it in double brackets `[[ ]]`**.

- **Correct:** "Check reader [[30604b75-1e5a-437e-a448-7344fbd53c94]] which has status [[inactive]]."
- **Incorrect (No brackets):** "Check reader 30604b75... which has status inactive."
- **Incorrect (Hallucinated):** "Check reader [[999999-fake-id]]." (This ID does not exist).

**THE COMPLETENESS RULE (CRITICAL):**
Every single value in `<required_values>` MUST appear in your `instruction` wrapped with `[[ ]]`.
Your job is to translate function-centric parameters into natural user language.

The user does NOT know what tools will be called or their parameter names. Use the `<tool_descriptions>` to understand what each tool does, then express the parameter values in a way a user would naturally describe their request.

**How to translate (use tool descriptions to guide you):**
1. Read the tool description to understand its PURPOSE
2. Match required_values parameters to the tool's parameters
3. Express the parameter value in natural language based on what the tool does

| Tool Description | Parameter | User-Friendly Translation |
|------------------|-----------|---------------------------|
| "List rooms by floor with status filter" | `status_filter: completed` | "show me [[completed]] rooms" |
| "Update assignment status" | `new_status: revoked` | "change the status to [[revoked]]" |
| "Search with time range" | `start_after: 2025-01-20` | "from [[2025-01-20]] onwards" |

**List Parameters:**
If a value is a list, mark each element individually:
- Required value: `scheduled_times: ["18:00", "22:00"]`
- You write: "at times [[18:00]] and [[22:00]]"

**THE STRICTNESS RULE:**
- **NO VALUE INVENTION:** You must strictly use the values provided in `<required_values>`. Do not generate, modify, or substitute any values.
- **COMPLETE COVERAGE:** After writing your instruction, verify that EVERY value from `<required_values>` appears with `[[ ]]` markers.
- **NO SYNONYMS:** Do not replace values with synonyms. If the value is "completed", do not write "finished" or "done".

### Phase 2: Narrative Inference (The "Human" Layer)

Do not simply list the parameters. Instead, **reverse engineer the real-world situation** that requires these specific values.

**Examples of "Narrative Triggers" in the data:**
- **Status Filters:** If filtering for `status=completed`, the motivation might be "I need to review what's been finished"
- **Time Constraints:** If using `start_after`, the user might be saying "I only care about recent ones"
- **Boolean Flags:** If `restrict_geography=false`, the user wants a broader search

**Motivation Style Guide:**
- **Bad (Robotic/Lists parameters):** "I want to authorize with ID abc-123, then list with status completed, then update to revoked."
- **Good (Natural narrative):** "I need to check my [[completed]] assignments from my account [[abc-123]] and mark the recent one as [[revoked]] since it's no longer needed."

## User Simulation

**Start Up Query Strategy:**
The start up query should be a **single, immediate pain point** - not a technical description.

| Example Type | Instruction | User Query | Note |
|--------------|-------------|------------|------|
| Bad | Check assignments with status completed... | Hi, I want to check assignments with status completed. | Too technical |
| Good | Check assignments with status completed... | Hi, I need to review my recent work history. | Natural pain point |

## Output Format

Output in JSON format:
```json
{{
    "start_up_query": "The symptom or pain point (natural language, no technical terms).",
    "instruction": "A natural user request containing ALL values from <required_values> wrapped in [[ ]].",
    "reason_for_call": "The narrative context explaining why the user needs this."
}}
```

**FINAL CHECK:** Before submitting, count the values in `<required_values>` and verify each one appears in your `instruction` with `[[ ]]` markers.
"""


HALLUCINATION_RETRY_PROMPT = """
Your previous response contained hallucinated values that do not exist in the provided instances.

**Hallucinated values detected:** {hallucinated_values}

These values were wrapped in [[ ]] but could NOT be found in the provided data.
Please regenerate your response, using ONLY values that exist in the instances.

Remember:
- Every value in [[ ]] MUST come directly from the provided data
- Do NOT invent or modify any IDs, names, or values

Please provide a corrected response in the same JSON format.
"""


COMPLETENESS_RETRY_PROMPT = """
Your previous response is MISSING required values from the `<required_values>` section.

**Missing values that MUST appear in instruction:** {missing_values}

These values MUST appear in your `instruction` field, each wrapped with `[[ ]]` markers.

Remember:
- EVERY value from `<required_values>` must appear in the instruction
- Translate function parameters into natural user language
- Each value must be wrapped with [[ ]]

**Example:**
If missing value is `status_filter: completed`, you might write:
"...show my [[completed]] assignments..."

Please provide a corrected response with ALL required values included.
"""
