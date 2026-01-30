"""
Prompts for Step 07: Policy Generation

Generate and validate domain policies for MCP servers.
"""

# =============================================================================
# Policy Structure Instructions (for structured parsing)
# =============================================================================

POLICY_STRUCTURE_INSTRUCTIONS = """
IMPORTANT: Use these markers for structured parsing of the policy document:

1. Wrap the entire policy content with:
   <!-- POLICY_START -->
   ... policy content ...
   <!-- POLICY_END -->

2. Mark the end of global/general rules section with:
   <!-- GLOBAL_RULES_END -->

3. Wrap each tool's policy section with:
   <!-- TOOL: tool_name -->
   ### Tool: tool_name
   Purpose: ...
   Preconditions: ...
   Input validation: ...
   Permission outcomes: ...
   Effects: ...
   <!-- TOOL_END: tool_name -->

Example structure:
```
<!-- POLICY_START -->
## Global Rules
- Authorization rules...
- ID format validations...
<!-- GLOBAL_RULES_END -->

<!-- TOOL: authorize_user -->
### Tool: authorize_user
Purpose: Authorize a user session.
Preconditions: None.
Input validation: user_id must be UUID format.
Permission outcomes:
  - Permitted: user exists and is_active = true
  - Rejected: user not found or inactive
Effects: Issues auth_token bound to user_id.
<!-- TOOL_END: authorize_user -->

<!-- TOOL: get_user_profile -->
### Tool: get_user_profile
...
<!-- TOOL_END: get_user_profile -->
<!-- POLICY_END -->
```
"""


# =============================================================================
# Domain Policy Generation
# =============================================================================

DOMAIN_POLICY_PROMPT = """
## Information of Target MCP Server

I am designing an MCP Server whose blueprint is described as follows:
<blueprint>
{blueprint}
</blueprint>
where the `core_entity` is assumed to be the user that interacts with the LLM assistant.

You treat the provided
<database_summary>
{summary}
</database_summary>
as the Production Database Schema.
Such Production Database Schema summarizes all value range and generation logic of the Entity/Relationship.
You can **NOT** put constraints on features that do not exist in the schema.
You know that referencing a feature that does not exist in the schema will crash the server and get you fired.

## Example Domain Policy

Here is an example domain policy:
<example_policy>
{example_policy}
</example_policy>

## CRITERIA

Time
- The current time is {simulation_time}.
- Never use the real-world current time (e.g., time.now()) in this mock-world.
- Authorization will never expire.
- This policy is read by the Assistant that interacts with the user and the MCP Server.
- "You" refers to the assistant but not the core entity.

For each of the **tools**, define explicit rules that determine:
- The exact conditions under which the action is **permitted**, **partially permitted**, or **rejected**.
- The required **preconditions**, **input validity checks**, and **temporal or logical dependencies**.
- Make the constraints a middle complexity that is similar to the given example.

⚠️ The policies will serve as **deterministic execution logic** for the MCP Server.
This means:
- Every user request must lead to **one and only one valid system outcome**.
- No policy should contain ambiguous wording or subject interpretation.
- All exceptional cases (e.g., invalid input, boundary conditions, conflicting states) must be explicitly handled.
- The Assistant, acting as the interface to users, must be able to **decisively** determine whether to approve or deny a request based solely on these policies.

## STRICT SCHEMA ADHERENCE (CRITICAL)
You exist in a **closed-world environment**.
- You are strictly FORBIDDEN from inventing, hallucinating, or assuming the existence of any database columns, entity attributes, or flags that are not explicitly defined in the provided blueprint and database schema.
- If a specific constraint (e.g., "Check user credit score") requires an attribute (e.g., `credit_score`) that is NOT in the database schema, **YOU MUST NOT WRITE THAT CONSTRAINT**.
- Instead, rely **only** on the available features to build logic.

{policy_structure_instructions}

Tool names to include: {tool_names}

Be **clear, comprehensive, and logically complete**.
Return the enriched policy document directly — do not include any explanations or meta commentary.
Use well-structured markdown with the structure markers as specified above.
"""


# =============================================================================
# Policy Validation
# =============================================================================

POLICY_VALIDATION_PROMPT = """
You are a meticulous Policy Auditor specializing in detecting hallucinations and inconsistencies in domain policies.

## Your Task
Review the provided domain policy and identify any hallucinations or inconsistencies that contradict the **ground truth** sources (Blueprint and Database Schema).

## Ground Truth Sources

### Blueprint (MCP Server Definition)
<blueprint>
{blueprint}
</blueprint>

### Database Schema
<database_summary>
{database_summary}
</database_summary>

### Simulation Time
- The current simulation time is: **{simulation_time}**

## Policy to Validate
<policy>
{policy}
</policy>

## Hallucination Types to Check

### 1. Fabricated Entity Attributes
- Check if the policy references any entity attributes that do NOT exist in the database schema.

### 2. Fabricated Relationships
- Check if the policy references any relationships that do NOT exist in the blueprint.

### 3. Timeline Violations
- Check if the policy uses real-world time functions.

### 4. Schema Mismatches
- Check if attribute value descriptions don't match the database schema.

## Output Format

If you find hallucinations, provide SEARCH/REPLACE blocks to fix them.
If no hallucinations are found, respond with: `NO_ISSUES_FOUND`
"""


# =============================================================================
# Common Policy Wrapper
# =============================================================================

COMMON_POLICY_WRAPPER = """
## General Execution Rules

You are an LLM assistant that interacts with a user and an MCP server.
You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.
You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.
You should deny user requests that are against this policy.
You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.

<policy>
{policy}
</policy>
"""
