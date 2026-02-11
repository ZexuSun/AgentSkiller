"""
Prompts for Step 15: Instance Combos Selection

Select entity instance combinations for task instantiation.
Includes: Constraint analysis, code generation, few-shot examples.
"""

# =============================================================================
# Few-shot Examples for Instance Assignment
# =============================================================================

INSTANCE_ASSIGNMENT_EXAMPLES = """
## Example 1: Same instance (CORRECT)
Trajectory: [search_product, add_to_cart, checkout]
Entities: Product appears in search_product and add_to_cart
Analysis: User searches for a product and then adds it to cart. The product in both steps is the same one - the user found what they wanted and proceeded to add it.
Decision: Product at position 0 and 1 → SAME instance

## Example 2: Different instances (CORRECT)
Trajectory: [enroll_course, drop_course]
Entities: Course appears in enroll_course and drop_course
Analysis: Enrolling in a course and immediately dropping the same course makes little sense. A more realistic scenario is the user enrolled in Course A and then dropped Course B (perhaps to make room in their schedule).
Decision: Course at position 0 and 1 → DIFFERENT instances

## Example 3: Same instance (CORRECT)
Trajectory: [authorize_user, get_user_profile, update_user_settings]
Entities: User appears in all three tools
Analysis: Authorization, profile retrieval, and settings update are all operations on the same user within a single session. The user authenticates, views their profile, and updates their settings.
Decision: User at all positions → SAME instance

## Example 4: Different instances (CORRECT)
Trajectory: [get_product_details, compare_products]
Entities: Product appears in both tools
Analysis: User views details of one product, then compares multiple products. The comparison likely involves different products to make a meaningful comparison.
Decision: Product at position 0 and 1 → DIFFERENT instances

## Example 5: Same instance (CORRECT)
Trajectory: [create_order, add_item_to_order, confirm_order]
Entities: Order appears in all three tools
Analysis: Creating an order, adding items to it, and confirming it are sequential operations on the same order within a single transaction flow.
Decision: Order at all positions → SAME instance

## Example 6: Mixed (CORRECT)
Trajectory: [authorize_technician, get_work_order, update_work_order_status, get_next_work_order]
Entities: Technician appears at position 0; WorkOrder appears at positions 1, 2, 3
Analysis: 
- Technician: Same instance throughout (single technician session)
- WorkOrder at positions 1 and 2: Same instance (get details then update status of same order)
- WorkOrder at position 3: Different instance (getting the NEXT work order implies a different one)
Decision: 
- Technician at all positions → SAME instance
- WorkOrder at positions 1, 2 → SAME instance
- WorkOrder at position 3 → DIFFERENT instance from positions 1, 2
"""

# =============================================================================
# Constraint Analysis Prompt (Phase 1: Plan) - Ultimate Version
# =============================================================================

CONSTRAINT_ANALYSIS_PROMPT = """
Analyze this Tool Call Trajectory and list all constraints for instance/value selection.

## Input Information

Trajectory: {trajectory}

Relevant Policy (including Global Rules):
{filtered_policy}

Database Entities Available:
{available_entities}

Entity Database Summaries (detailed field specifications and constraints):
{entity_summaries}

Database Relationships Available:
{available_relationships}

Relationship Database Summaries (detailed field specifications and constraints):
{relationship_summaries}

Current Time (NOW): {current_time}

---

## Task 1: Analyze Relationship Operations in Trajectory

For each tool in the trajectory, determine its operation type on relationships:

| Operation | Description | Sampling Strategy |
|-----------|-------------|-------------------|
| **CREATE** | Creates a new relationship record | Sample entities separately, verify relationship does NOT exist |
| **READ** | Reads existing relationship(s) | May or may not require existing relationship |
| **UPDATE** | Modifies existing relationship | Relationship MUST exist, sample FROM relationship table |
| **DELETE** | Removes relationship | Relationship MUST exist, sample FROM relationship table |

---

## Task 2: Generate Sampling Requirements with Dual-Slot Constraints

For each (tool, parameter) pair, generate a complete constraint specification with TWO separate constraint slots:

### Constraint Slot 1: global_constraints
- Extract from the **Global Rules** section of the policy
- Look for definitions like "Active badge criteria", "Valid permission window", "Active assignment definition"
- **EXPAND** each definition inline with concrete conditions
- **SUBSTITUTE** NOW with the actual time value: {current_time}
- If no global constraints apply, use empty array []

### Constraint Slot 2: tool_specific_constraints  
- Extract from the **tool's own policy section**
- Look for preconditions, input validation, permission checks
- **EXPAND** any references to global definitions
- If no tool-specific constraints, use empty array []

---

## Task 3: Identify Sequential Constraints (Cross-Tool Dependencies) ⚠️ CRITICAL - ANALYZE FIRST

Analyze the trajectory for constraints that **propagate from earlier tools to later tools**. These constraints create **sparse valid combinations** where the value chosen in Tool A restricts valid values in Tool B.

### 3.1 State Transition Constraints
- Identify status/state fields that appear in multiple tools
- Extract the valid transition graph from the Global Rules
- Map: `tool_i.status_value → allowed tool_j.status_values`

Example from policy:
```
LabResult status transitions:
- "preliminary" → {{"final", "canceled", "corrected", "amended"}}
- "final" → {{"corrected", "amended"}}
- "canceled" → {{}} (terminal)
```

If trajectory is `[record_lab_result, update_lab_result_status]`:
- `record_lab_result.result_status = "preliminary"` → `update_lab_result_status.new_status` can only be `["final", "canceled", "corrected", "amended"]`
- `record_lab_result.result_status = "canceled"` → `update_lab_result_status.new_status` has NO valid options (trajectory impossible!)

### 3.2 Conditional Parameter Constraints
- Identify parameters whose requirements depend on other parameter values
- Example: `reason` is required only when `new_status ∈ {{"corrected", "amended"}}`

### 3.3 Value Propagation Constraints
- Identify when a value chosen in Tool A restricts valid values in Tool B
- Example: `abnormal_flag = "normal"` → `trigger_critical_value_alert` will be rejected (requires critical flag)

### 3.4 Entity State Mutation Constraints
- Tool A may change entity state, affecting Tool B's preconditions
- Example: `cancel_lab_order` sets `order_status = "canceled"` → subsequent tools cannot operate on canceled orders

### 3.5 Identify Trajectory-Blocking Values ⚠️ CRITICAL
For each parameter that has sequential constraints, identify values that would **block the trajectory from completing**:
- Values that lead to empty `valid_transitions` (no valid downstream options)
- Values that violate downstream tool preconditions
- Values that put entities into terminal states

Example:
- `record_lab_result.abnormal_flag = "normal"` → blocks `trigger_critical_value_alert` (requires critical_high/critical_low)
- `record_lab_result.result_status = "canceled"` → blocks all subsequent operations (terminal state)

---

## Task 4: Extract Value Domains from Policy

For value_domain parameters, extract ALL constraints:
- Enum: `status ∈ {{a, b, c}}` → {{"type": "enum", "values": ["a", "b", "c"]}}
- Range: `value in [0, 100]` → {{"type": "range", "min": 0, "max": 100, "is_int": false}}
- Array: `items 1-4 from [...]` → {{"type": "array_choice", "choices": [...], "min_count": 1, "max_count": 4, "unique": true}}
- Boolean: true/false → {{"type": "boolean"}}
- Time: future time with alignment → {{"type": "time", "constraints": ["must be > current_time", "minutes in {{0, 15, 30, 45}}"]}}
- Duration: `duration ∈ {{15, 30, 45, 60}}` → {{"type": "duration", "values_minutes": [15, 30, 45, 60]}}

**IMPORTANT**: When extracting value domains, cross-reference with `trajectory_blocking_values` to exclude blocked values.

---

## Task 5: Instance Assignment Analysis

For entity_instance parameters that appear in multiple tools:
- Determine if they should use the SAME or DIFFERENT instances
- Consider semantic meaning and real-world plausibility

{instance_assignment_examples}

---

## Sampling Strategy
- Combine BOTH constraint slots into a concrete sampling plan
- Use **Relationship-First Sampling** when the tool operates on existing relationships:
  - Sample from the Relationship table FIRST
  - Join with Entity tables to verify constraints
  - Extract entity IDs from the relationship record
  - DO NOT sample entities separately when relationship must exist


## Output Format

Output as JSON with this structure (NOTE: sequential_constraints and trajectory_blocking_values come BEFORE sampling_requirements so they can be referenced):
{{
  "relationship_operations": [
    {{
      "tool": "tool_name",
      "operation_type": "CREATE|READ|UPDATE|DELETE",
      "target_relationship": "RelationshipTableName",
      "requires_existing": true,
      "existence_condition": "SQL-like condition (e.g., revoked_at IS NULL)",
      "policy_reference": "exact quote from policy"
    }}
  ],

  "instance_assignment_plan": {{
    "EntityName": {{
      "positions": [0, 1, 2],
      "same_instance": true,
      "reason": "explanation of why same/different"
    }}
  }},
  
  "sequential_constraints": [
    {{
      "constraint_id": "seq_001",
      "constraint_type": "state_transition|conditional_required|value_propagation|entity_mutation",
      "from_tool": "tool_name_earlier",
      "from_param": "parameter_name",
      "to_tool": "tool_name_later",
      "to_param": "parameter_name",
      "valid_transitions": {{
        "value_a": ["allowed_value_1", "allowed_value_2"],
        "value_b": ["allowed_value_3"],
        "value_c": []
      }},
      "condition": {{
        "when": ["value_x", "value_y"],
        "then": "required_non_empty|must_be_null|specific_value"
      }},
      "policy_reference": "exact quote from policy"
    }}
  ],
  
  "trajectory_blocking_values": {{
    "tool_name.parameter_name": {{
      "blocked_values": ["value_that_blocks_trajectory"],
      "allowed_values": ["value_that_allows_trajectory_to_continue"],
      "reason": "Why these values block the trajectory",
      "downstream_tool": "tool_name_that_would_fail",
      "constraint_ref": "seq_001"
    }}
  }},
  
  "propagation_order": [
    "tool_0.param_a",
    "tool_1.param_b",
    "tool_1.param_c"
  ],
  
  "sampling_requirements": [
    {{
      "tool": "tool_name",
      "parameter": "param_name",
      "sampling_type": "entity_instance|value_domain|derived",
      "entity_type": "EntityName",
      
      "global_constraints": [
        {{
          "source": "Name of Global Rule (e.g., Active badge criteria)",
          "expanded": [
            "Concrete condition 1 with actual values",
            "Concrete condition 2 with actual values",
            "Nested conditions if applicable"
          ],
          "policy_reference": "exact quote from Global Rules"
        }}
      ],
      
      "tool_specific_constraints": [
        {{
          "source": "Tool policy section name",
          "expanded": [
            "Concrete condition from tool policy"
          ],
          "policy_reference": "exact quote from tool policy"
        }}
      ],

      "inter_tool_constraints": [
        {{
          "constraint_ref": "seq_001",
          "role": "source|target",
          "impact": "Brief description - MUST reference constraint from sequential_constraints above"
        }}
      ],
      
      "sampling_strategy": {{
        "approach": "direct|relationship_first|relationship_exclude",
        "source_table": "PrimaryTableToSampleFrom",
        "join_tables": ["Table1", "Table2"],
        "filter_conditions": [
          "Combined conditions from global, tool_specific, AND trajectory_blocking_values",
          "If parameter is in trajectory_blocking_values, add: value NOT IN blocked_values",
          "With actual values substituted (e.g., expiration_date >= '2025-01-23')"
        ],
        "extract_field": "field_name_to_extract"
      }},
      
      "value_domain": {{...}},
      "derived_from": "tool.output_field"
    }}
  ],
  
  "relationship_sampling_order": [
    {{
      "step": 1,
      "action": "sample_from_relationship|sample_entity|verify_no_relationship",
      "table": "TableName",
      "join_with": ["Table1", "Table2"],
      "filter_conditions": [
        "condition1 with actual values",
        "condition2 with actual values"
      ],
      "extract_fields": {{
        "param_name": "field_path"
      }},
      "reason": "why this step is needed"
    }}
  ],
  
  "example_valid_combos": [
    {{
      "tool_0.param_a": "value_1",
      "tool_1.param_b": "value_2",
      "tool_1.param_c": "value_3"
    }},
    {{
      "tool_0.param_a": "value_4",
      "tool_1.param_b": "value_5",
      "tool_1.param_c": null
    }}
  ]
}}

**IMPORTANT**: The `trajectory_blocking_values` section is CRITICAL for code generation:
- It tells the code generator which values to EXCLUDE when sampling
- Without this, the generator may sample values that make the trajectory impossible
- Example: If `abnormal_flag` has `blocked_values: ["normal", "high", "low"]`, the code should only sample from `allowed_values: ["critical_high", "critical_low"]`

---

## Critical Rules

1. **MUST fill BOTH constraint slots** for each parameter - even if one is empty []
2. **MUST expand all Global Rule references** - never just name them, always expand inline
3. **MUST substitute NOW** with actual time: {current_time}
4. **MUST use Relationship-First Sampling** when tool operates on existing relationships
5. **filter_conditions MUST combine** all global, tool_specific and sequential constraints
6. **policy_reference MUST quote** exact text from policy for traceability

---

## Example of Constraint Expansion

If policy says: "Tenant has at least one active assignment to any active badge (Active badge criteria)"

Your output should expand this to:
```json
"global_constraints": [
  {{
    "source": "Active badge criteria",
    "expanded": [
      "Tenant must have at least one TenantBadgeAssignment where:",
      "  - TenantBadgeAssignment.status == 'active'",
      "  - TenantBadgeAssignment.revoked_at IS NULL",
      "  - badgecredential_id references a BadgeCredential where:",
      "    - BadgeCredential.status == 'active'",
      "    - BadgeCredential.is_active == true",
      "    - BadgeCredential.expiration_date >= '2025-01-23'"
    ],
    "policy_reference": "Active badge criteria (all must hold at NOW): ..."
  }}
]
```

NOT just: `"global_constraints": [{{"source": "Active badge criteria", "expanded": ["must satisfy Active badge criteria"]}}]`
"""

# =============================================================================
# Selection Code Generation Prompt (Phase 2: Execution)
# =============================================================================

SELECTION_CODE_PROMPT = """
Generate Python code to select entity instances AND sample value domain parameters based on the constraint analysis.

## Constraint Analysis Result
{constraint_analysis_json}

## Database Information
Database directory: {db_dir}
Available entity files: {entity_files}
Available relationship files: {relationship_files}

## Requirements

### 1. Entity Instance Sampling
- Load entity data from JSON files in the database directory
- Apply all constraints from the analysis (both global_constraints and tool_specific_constraints)
- For entities marked same_instance=true: sample ONE instance and reuse for all positions
- For entities marked same_instance=false: sample DISTINCT instances for different positions
- **Use the sampling_strategy from the constraint analysis** - it already combines all constraints

### 2. Relationship-First Sampling (CRITICAL)
When sampling_strategy.approach is "relationship_first":
- Sample from the relationship table FIRST
- Join with entity tables to verify all filter_conditions
- Extract entity IDs from the relationship record
- DO NOT sample entities separately

### 3. Value Domain Sampling with Blocking Value Exclusion ⚠️ CRITICAL
- For "enum" type: randomly choose from the valid values list
- For "range" type: sample within [min, max] bounds (use int() if is_int=true)
- For "array_choice" type: sample between min_count and max_count items from choices
- For "boolean" type: randomly choose True or False

**IMPORTANT - Check `trajectory_blocking_values` FIRST:**
- If a parameter appears in `trajectory_blocking_values`, use ONLY the `allowed_values`
- NEVER sample from `blocked_values` - these will cause trajectory failure
- Example:
  ```python
  # If trajectory_blocking_values contains:
  # "record_lab_result.abnormal_flag": {{
  #   "blocked_values": ["normal", "high", "low", "abnormal"],
  #   "allowed_values": ["critical_high", "critical_low"]
  # }}
  # Then for abnormal_flag, ONLY sample from ["critical_high", "critical_low"]
  ```

### 4. Sequential Constraint Propagation (CRITICAL for Cross-Tool Dependencies)
When `sequential_constraints` is present in the constraint analysis:
- Follow the `propagation_order` to determine which parameters to sample first
- For each `state_transition` constraint:
  - Sample the `from_param` value first
  - Use `valid_transitions[from_value]` to get allowed values for `to_param`
  - If `valid_transitions[from_value]` is empty, this combination is INVALID - skip it
- For each `conditional_required` constraint:
  - Check if `from_param` value is in `condition.when`
  - If yes, apply `condition.then` rule to `to_param`
- For each `value_propagation` constraint:
  - The `from_param` value restricts what `to_param` can be
- Generate combinations by iterating through valid transition paths

Example implementation pattern:
```python
# For state_transition constraint
from_values = ["preliminary", "final", "corrected"]  # from value_domain
valid_combos = []
for from_val in from_values:
    allowed_to_vals = valid_transitions.get(from_val, [])
    if not allowed_to_vals:
        continue  # Skip - no valid transitions from this state
    for to_val in allowed_to_vals:
        valid_combos.append({{
            "tool_0.result_status": from_val,
            "tool_1.new_status": to_val
        }})
```

### 5. Output Format
The script must output a JSON object to stdout containing **ALL valid combinations** that satisfy the constraints:
{{
  "all_valid_combos": [
    {{
      "entity_instances": {{
        "EntityName": {{...record fields...}},
        "EntityName_2": {{...for different instance...}}
      }},
      "value_domain_samples": {{
        "tool_name.parameter": sampled_value
      }}
    }},
    // ... more valid combinations
  ],
  "metadata": {{
    "success": true,
    "total_combos_found": 15,
    "constraints_applied": 5,
    "cross_entity_verified": true
  }}
}}

**IMPORTANT**: 
- Generate ALL valid combinations, not just one
- If there are too many combinations (>100), randomly sample 100 of them
- Each combination should be independently valid for the trajectory

## Code Template

Generate a complete Python script based on this template:

```python
import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

def load_entity(entity_name: str, db_dir: Path) -> List[Dict]:
    \"\"\"Load entity records from JSON file.\"\"\"
    # Try different naming conventions
    for name_variant in [entity_name, entity_name.lower(), entity_name.replace(" ", "")]:
        path = db_dir / "entities" / f"{{name_variant}}.json"
        if path.exists():
            return json.loads(path.read_text())
    return []

def load_relationship(rel_name: str, db_dir: Path, server_name: str = None) -> List[Dict]:
    \"\"\"Load relationship records from JSON file.\"\"\"
    # Try with server subdirectory first
    if server_name:
        path = db_dir / "relationships" / server_name / f"{{rel_name}}.json"
        if path.exists():
            return json.loads(path.read_text())
    # Try direct path
    path = db_dir / "relationships" / f"{{rel_name}}.json"
    return json.loads(path.read_text()) if path.exists() else []

def filter_by_constraints(records: List[Dict], constraints: List[str]) -> List[Dict]:
    \"\"\"Filter records by constraint expressions.\"\"\"
    result = records
    for constraint in constraints:
        # Parse simple constraints like "field = value" or "field = true"
        if " = " in constraint:
            field, value = constraint.split(" = ", 1)
            field = field.strip()
            value = value.strip()
            # Handle boolean and other values
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            result = [r for r in result if r.get(field) == value]
        elif " IS NULL" in constraint.upper():
            field = constraint.upper().replace(" IS NULL", "").strip().lower()
            result = [r for r in result if r.get(field) is None]
        elif " IS NOT NULL" in constraint.upper():
            field = constraint.upper().replace(" IS NOT NULL", "").strip().lower()
            result = [r for r in result if r.get(field) is not None]
        elif " >= " in constraint:
            field, value = constraint.split(" >= ", 1)
            field = field.strip()
            value = value.strip().strip("'")
            result = [r for r in result if r.get(field) and r.get(field) >= value]
        elif " IN " in constraint.upper():
            # Handle "field IN ('a', 'b', 'c')" pattern
            parts = constraint.split(" IN ", 1)
            if len(parts) == 2:
                field = parts[0].strip()
                values_str = parts[1].strip()
                # Parse values from "(a, b, c)" or "('a', 'b', 'c')"
                values_str = values_str.strip("()")
                values = [v.strip().strip("'\"") for v in values_str.split(",")]
                result = [r for r in result if r.get(field) in values]
    return result

def sample_enum(values: List[str]) -> str:
    \"\"\"Sample from enum values.\"\"\"
    return random.choice(values)

def sample_range(min_val: float, max_val: float, is_int: bool = False) -> float:
    \"\"\"Sample from numeric range.\"\"\"
    val = random.uniform(min_val, max_val)
    return int(val) if is_int else round(val, 2)

def sample_array_choice(choices: List[str], min_count: int, max_count: int, unique: bool = True) -> List[str]:
    \"\"\"Sample array from choices.\"\"\"
    count = random.randint(min_count, min(max_count, len(choices)))
    if unique:
        return random.sample(choices, count)
    return [random.choice(choices) for _ in range(count)]

def sample_boolean() -> bool:
    \"\"\"Sample boolean value.\"\"\"
    return random.choice([True, False])

def join_tables(primary_records: List[Dict], secondary_records: List[Dict], 
                primary_key: str, foreign_key: str) -> List[Dict]:
    \"\"\"Join two tables on a key relationship.\"\"\"
    secondary_by_key = {{r[foreign_key]: r for r in secondary_records if foreign_key in r}}
    result = []
    for record in primary_records:
        if primary_key in record and record[primary_key] in secondary_by_key:
            joined = {{**record, **secondary_by_key[record[primary_key]]}}
            result.append(joined)
    return result

def select_instances(db_dir: Path) -> Dict[str, Any]:
    \"\"\"Main selection logic - implement based on constraint analysis.\"\"\"
    result = {{
        "all_valid_combos": [],
        "metadata": {{
            "success": True,
            "total_combos_found": 0,
            "constraints_applied": 0,
            "cross_entity_verified": True
        }}
    }}
    
    # TODO: Implement the actual selection logic based on constraint analysis
    # Use the sampling_strategy from each sampling_requirement
    # Follow the relationship_sampling_order for correct sampling sequence
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({{"error": "Database directory not provided"}}))
        sys.exit(1)
    
    db_dir = Path(sys.argv[1])
    if not db_dir.exists():
        print(json.dumps({{"error": f"Database directory not found: {{db_dir}}"}}))
        sys.exit(1)
    
    try:
        result = select_instances(db_dir)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(json.dumps({{"error": str(e), "success": False}}))
        sys.exit(1)
```

Now generate the complete implementation of `select_instances()` based on the constraint analysis provided above.
"""


# =============================================================================
# Constraint Proposal Prompt (for generating constraint validation code)
# =============================================================================

PROPOSE_CONSTRAINTS_PROMPT = """
You are reasoning about a tool-calling LLM agent system.
Your goal is to identify constraints that lead to successful execution of the given tool call trajectory.

<tool_call_trajectory>
{tool_call_trajectory}
</tool_call_trajectory>

<task_template>
{task_template}
</task_template>

<policy>
{policy}
</policy>

<database_summary>
{database_summary}
</database_summary>

Your job:
1. Parse the template, identify all entity/relationship mentions
2. Infer the semantic role each entity plays in this task
3. Infer what constraints they should meet to guarantee successful execution
4. Write a Python script that:
   - Samples concrete values from the database
   - Searches valid E&R combinations with constraints
     - For example, a `Student` Bob can successfully enroll a `Course` Math only when:
        - Relationship `Enrollment` does NOT have an record (Bob, Math) already. (E-R constrain)
        - `Course` Math still have available seats. (E constrain)
        - Schedule of the Math should NOT conflit with the courses that Bob has already enrolled. (E-E constrain)
   - Outputs a `.jsonl` file where each line is a combination

**CRITICAL: Memory & Performance Limits**
- Use generators/iterators instead of storing all combinations in memory
- **LIMIT output to at most 5 combinations** - stop writing after reaching this limit
- Sample entities intelligently: if there are too many entities (e.g., >100), randomly sample a subset first
- Use early termination: once you have enough valid combinations, stop searching

Database Path:
- Entity database: ./outputs/database/outputs/entities/<EntityName>.json
- Relationship database: ./outputs/database/outputs/relationships/{domain_name}/<RelationshipName>.json
- Output JSONL: ./outputs/combinations/outputs/{domain_name}/{trajectory_hash}.jsonl

Output only the Python code, no explanations.
"""


# =============================================================================
# Cross Domain Code Fusion Prompt
# =============================================================================

CODE_FUSION_PROMPT = """
You are fusing multiple Single Domain sampling codes into a unified Cross Domain sampling code.

## Cross Domain Trajectory

The Cross Domain trajectory is formed by concatenating trajectories from multiple domains:

{trajectory_structure}

Full Cross Domain Trajectory: {cross_domain_trajectory}

## Core Entity (MUST be identical across domains)

The Core Entity is the user initiating this interaction:
{core_entity}

This entity MUST use the SAME instance across ALL domains. 
For example, if "Customer" is the core entity, the customer_id used in DomainA 
must be the same customer_id used in DomainB - they represent the same user.

All other entities can be sampled independently by each domain based on its own constraints.

## Cross Domain Policy

{cross_domain_policy}

## Single Domain Sampling Codes

Below are the working sampling codes from each Single Domain. Each code has been validated to produce correct combinations for its respective domain.

{single_domain_codes}

## Your Task

Fuse these Single Domain codes into a unified Cross Domain sampling code that:

1. **Handles Core Entity Correctly (CRITICAL)**
   - The Core Entity MUST be sampled ONCE and reused across ALL domains
   - It must satisfy constraints from ALL domains that use it
   - Example: If Customer is the core entity, sample ONE Customer that satisfies both DomainA and DomainB constraints

2. **Handles Other Entities Independently**
   - Each domain can sample its other entities independently
   - Use each domain's original sampling logic for non-core entities
   - No need to share these entities across domains

3. **Merges Entity Loading Logic**
   - Use the Cross Domain database paths:
     - Entity dir: {{db_dir}}/entities/{{server_name}}/
     - Relationship dir: {{db_dir}}/relationships/{{server_name}}/

4. **Preserves Domain-Specific Constraints**
   - Keep all constraints from each Single Domain code
   - For parameters specific to one domain, use that domain's sampling logic

5. **Generates Combined Output**
   - Output format must include all entity instances and value_domain_samples
   - Use the full tool names with domain prefix: "DomainA.tool_name"

## Code Template

```python
import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

def load_entity(entity_name: str, db_dir: Path, server_name: str) -> List[Dict]:
    \"\"\"Load entity records from Cross Domain database.\"\"\"
    # Cross Domain entities are stored in: db_dir/entities/server_name/
    entity_dir = db_dir / "entities" / server_name
    for name_variant in [entity_name, entity_name.lower()]:
        path = entity_dir / f"{{name_variant}}.json"
        if path.exists():
            return json.loads(path.read_text())
    return []

def load_relationship(rel_name: str, db_dir: Path, server_name: str) -> List[Dict]:
    \"\"\"Load relationship records from Cross Domain database.\"\"\"
    rel_dir = db_dir / "relationships" / server_name
    for name_variant in [rel_name, rel_name.lower()]:
        path = rel_dir / f"{{name_variant}}.json"
        if path.exists():
            return json.loads(path.read_text())
    return []

def select_instances(db_dir: Path, server_name: str) -> Dict[str, Any]:
    \"\"\"
    Main selection logic - fuses constraints from all Single Domain codes.
    
    Key requirements:
    1. Sample the Core Entity ONCE with combined constraints from all domains
    2. Sample other entities independently per domain
    3. Generate combinations valid for the entire Cross Domain trajectory
    \"\"\"
    result = {{
        "all_valid_combos": [],
        "metadata": {{
            "success": True,
            "total_combos_found": 0,
            "constraints_applied": 0,
            "cross_entity_verified": True,
            "is_cross_domain": True
        }}
    }}
    
    # TODO: Implement fused selection logic
    # 1. Load shared entities with combined constraints
    # 2. For each valid shared entity combination:
    #    - Apply DomainA constraints
    #    - Apply DomainB constraints
    #    - If both pass, add to valid_combos
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({{"error": "Usage: python script.py <db_dir> <server_name>"}}))
        sys.exit(1)
    
    db_dir = Path(sys.argv[1])
    server_name = sys.argv[2]
    
    if not db_dir.exists():
        print(json.dumps({{"error": f"Database directory not found: {{db_dir}}"}}))
        sys.exit(1)
    
    try:
        result = select_instances(db_dir, server_name)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(json.dumps({{"error": str(e), "success": False}}))
        sys.exit(1)
```

## Critical Rules

1. **Shared Entity Consistency**: Sample shared entities FIRST, then use them across all domains
2. **Constraint Combination**: Use AND logic to combine constraints from different domains on shared entities
3. **Value Domain Namespacing**: Use full names like "DomainA.tool_name.param" for value_domain_samples
4. **Error Handling**: If any domain's constraints cannot be satisfied, skip that combination
5. **Cross-Domain Database Paths**: Use the correct paths for Cross Domain entity/relationship storage

Now generate the complete fused implementation.
"""


# =============================================================================
# Cross Domain Combination Creation Prompts
# =============================================================================

IMPLICIT_DEPENDENCY_ANALYSIS_PROMPT = """
You are analyzing a Cross Domain task template to identify IMPLICIT entity and relationship instances that must exist for the trajectory to succeed.

## Context

**Entity**: An independent data object (e.g., Student, Instructor, Vehicle, Customer)
**Relationship**: A link between entities (e.g., StudentCourseEnrollment, TenantVehicleAuthorization)

Entities exist independently. Relationships connect entities and often have additional attributes.

## Task Template

{task_template_instruction}

## Trajectory

{trajectory}

## Explicit Dependencies (Already Identified)

These are directly mentioned in placeholders - DO NOT include them again:

{explicit_dependencies_json}

## Available Relationships (per Domain)

{available_relationships}

## Your Task

Identify IMPLICIT dependencies - entities or relationships that:
1. Are NOT directly mentioned in any placeholder
2. But MUST exist for the trajectory to succeed

**Examples of implicit dependencies:**
- "my vehicle" → `TenantVehicleAuthorization` relationship must link the Tenant to the Vehicle
- "my assigned course" → `CourseInstructionAssignment` relationship must link the Instructor to the Course
- A tool that queries a relationship → that relationship record must exist
- "confirm my account is active" → An account/authorization relationship must exist

**Output Format (JSON)**:

```json
{{
  "implicit_entities": [
    {{
      "name": "EntityName",
      "reason": "Why this entity must exist",
      "fields_needed": ["field1", "field2"]
    }}
  ],
  "implicit_relationships": [
    {{
      "domain": "DomainName",
      "name": "RelationshipName",
      "reason": "Why this relationship must exist",
      "foreign_keys": {{
        "entity1_id": "references Entity1",
        "entity2_id": "references Entity2"
      }}
    }}
  ]
}}
```

Only output implicit dependencies. Do not repeat explicit ones.
If there are no implicit dependencies, return empty arrays.
"""


COMBINATION_CREATION_PROMPT = """
You are generating Python code to CREATE entity and relationship instances for a Cross Domain task.

## Key Concept

**Sampling vs Creation describe the SAME constraints:**
- Sampling Code: Searches existing database for instances satisfying constraints
- Creation Code: Generates NEW instances satisfying the same constraints

Your goal: Generate instances that would PASS all constraints in the Sampling Codes.

## Task Template

{task_template_instruction}

## Full Dependency Analysis

The following entities and relationships are needed (both explicit from placeholders and implicit from semantic analysis):

{dependency_analysis_json}

## Core Entity

{core_entity}

This entity represents the user initiating the interaction.
Generate ONE instance and reuse it across ALL domains.

## Single Domain Sampling Codes (CONSTRAINT REFERENCE)

Study these codes carefully to understand the constraints. Your Creation Code must satisfy them.
Pay attention to:
- How entities are filtered (status, dates, etc.)
- How relationships are connected (foreign keys)
- Valid value ranges for each field

{single_domain_codes}

## Filtered Policies (BUSINESS RULES)

These policies define valid state transitions and business rules. Your generated instances MUST comply with them.
Pay special attention to:
- Status field valid values and transitions
- Required preconditions for operations
- Validation rules for fields

{filtered_policy}

## Database Summary (GENERATION RULES)

Use these specifications to generate VALID field values.

### Entities
{entity_summaries}

### Relationships
{relationship_summaries}

## Output Requirements

1. **Generate Multiple Combinations** (at least 10)
2. **Each combination must include:**
   - `entity_instances`: All required Entity AND Relationship instances with complete field values
     - Entity format: "EntityName": {{...record fields...}}
     - Relationship format: "Domain.RelationshipName" or "Domain.RelationshipName_suffix": {{...record fields...}}
   - `value_domain_samples`: All placeholder field values (format: "Domain.Relationship.field": value)

3. **JSON Output Format:**
```json
{{
  "all_valid_combos": [
    {{
      "entity_instances": {{
        "EntityName": {{"field1": "value1", "field2": "value2", ...}},
        "Domain.RelationshipName": {{"field1": "value1", "field2": "value2", ...}},
        "Domain.RelationshipName_suffix": {{"field1": "value1", ...}},
        ...
      }},
      "value_domain_samples": {{
        "Domain.tool_name.param_name": "value",
        ...
      }}
    }}
  ]
}}
```

## CRITICAL: Complete Parameter Coverage

Your `value_domain_samples` MUST include entries for **ALL parameters of EVERY tool** in the trajectory.

**Time Anchor**: Current time is `2025-01-23T15:00:00-05:00 (EST)`. For parameters like `as_of_date`, `on_date`, `event_time`, or any parameter that semantically means "now" or "current time", use this date: `2025-01-23`.

**Three Valid Value Types for each parameter**:
1. **Concrete value**: Actual value (string, number, boolean, etc.) from entities or policy-defined ranges
2. **Dynamic marker**: The literal string `<From previous tool call output>` for params that depend on previous tool results (e.g., IDs created by `create_*`, `generate_*`, `add_*` tools)
3. **null**: Explicitly `null` for optional parameters that should use server defaults or be omitted

**Tools in this trajectory** (you MUST cover ALL params for each tool):
{trajectory_tool_names}

**Example value_domain_samples (complete coverage)**:
```json
{{
  "ServerA.authorize_user.user_id": "uuid-xxx",
  "ServerA.create_order.amount": 150.00,
  "ServerA.create_order.currency": "USD",
  "ServerA.create_order.notes": null,
  "ServerA.update_order_status.order_id": "<From previous tool call output>",
  "ServerA.update_order_status.new_status": "confirmed",
  "ServerA.check_availability.as_of_date": "2025-01-23",
  "ServerB.list_items.filter_status": null
}}
```

**Why this matters**: Both the Golden Trajectory Executor and Agent Rollout use these values. If a parameter is missing, each system will guess independently, causing evaluation mismatches even when the trajectory is semantically correct.

## Code Template

```python
import json
import uuid
import random
from datetime import datetime, timedelta

def generate_combinations(num_combos: int = 10) -> dict:
    all_combos = []
    
    for i in range(num_combos):
        # 1. Generate Core Entity (same structure, different values each iteration)
        core_id = str(uuid.uuid4())
        core_entity = {{
            # Fill in all required fields according to database_summary
        }}
        
        # 2. Generate other Entities
        # Each entity needs a unique ID and valid field values
        
        # 3. Generate Relationship records
        # Create actual relationship records with all required fields
        # Ensure foreign keys point to generated entity IDs
        # Follow constraints from Sampling Codes
        
        value_domain_samples = {{
            # "Domain.Relationship.field": value,
        }}
        
        combo = {{
            "entity_instances": {{
                # Include ALL entities AND relationships here!
                # Entity format: "EntityName": {{...record fields...}}
                # Relationship format: "Domain.RelationshipName": {{...record fields...}}
                # Use suffix if multiple instances: "Domain.RelationshipName_creation": {{...}}
            }},
            "value_domain_samples": value_domain_samples,
        }}
        all_combos.append(combo)
    
    return {{"all_valid_combos": all_combos}}

if __name__ == "__main__":
    result = generate_combinations()
    print(json.dumps(result, indent=2, default=str))
```

## Critical Rules

1. **Core Entity**: Generate with valid field values, reuse the SAME core_entity_id in all relationships
2. **Entity IDs**: Use `str(uuid.uuid4())` for all ID fields
3. **Field Values**: Follow the exact constraints in database_summary
4. **Foreign Keys**: Relationship foreign keys must point to generated entity IDs
5. **Status Fields**: Use valid status values (usually "active" for most cases)
6. **Date Fields**: Use realistic dates (recent past or near future)

## IMPORTANT: Pre-existing vs Created Records

**DO NOT include records that will be CREATED by the trajectory tools!**

- `entity_instances` should only contain records that must EXIST BEFORE trajectory execution
- If a tool name starts with `create_`, `generate_`, `add_`, or `register_`, it will CREATE a new record
- DO NOT add those records to `entity_instances`
- Instead, put the INPUT PARAMETERS for such tools in `value_domain_samples`

**Example:**
- Trajectory has `LabSupplyRequisition.generate_picklist_for_warehouse`
- This tool CREATES an InstructorWarehousePicklist record
- DO NOT add InstructorWarehousePicklist to entity_instances
- Instead, add input params like `LabSupplyRequisition.InstructorWarehousePicklist.scheduled_pick_date` to value_domain_samples

Now generate the complete implementation.
"""


# =============================================================================
# Error Analysis Prompt (for Plan-then-Fix approach)
# =============================================================================

ERROR_ANALYSIS_PROMPT = """
Analyze the following trajectory execution error and identify the root cause.

## CRITICAL: Understanding the Task

**What is Creation Code?**
- Creation Code generates entity/relationship instances BEFORE trajectory execution and assign tool call parameters for each tool in the trajectory
- The instances must form an initial state that leads to a successful trajectory execution (e.g., if a tool operates instances that is not created during the trajectory execution, the instance must be included in the initial state)
- The tool call parameters must assign reasonable values that follow the policy rules (e.g., valid status transition target, valid date ranges, valid values for fields, etc.)
- Creation Code produces `entity_instances` (entity/relationship) and `value_domain_samples` (tool call parameters)
- Note that some parameter for some tools may be revealed during the trajectory execution and thus not included in the `value_domain_samples`

**What is Trajectory?**
- The trajectory is a FIXED sequence of tool calls: {trajectory}
- You CANNOT modify the trajectory (add/remove/reorder tool calls)
- You can ONLY modify the instances or tool call parameters created by Creation Code

**Correct Fix Direction:**
- Do NOT suggest adding intermediate tool calls - the trajectory is IMMUTABLE
- Instead, suggest modifying entity/relationship instances or tool call parameters in Creation Code

## Error Information
{error_message}

## Execution Log
{exec_log}

## Failed Combo (the instances and tool call parameters that were created)
{failed_combo}

## Domain Policy
{filtered_policy}

## Analysis Tasks

1. **Error Category** - Identify which type:
   - MISSING_INSTANCE: Need to create additional instances that must exist BEFORE trajectory execution
   - EXTRA_INSTANCE: Created instances that should NOT exist (e.g., records that trajectory tools will CREATE)
   - POLICY_VIOLATION: Violated business rules (state transitions, preconditions, valid values)
   - INCOMPLETE_PARAMS: Missing parameters in value_domain_samples that are required for tool calls (the key "Domain.tool.param" does not exist)
   - OTHER: Parameter format, missing fields, type errors, etc.

2. **Root Cause** - Explain specifically what went wrong and WHY

3. **Policy Reference** - MANDATORY for POLICY_VIOLATION:
   - Find the EXACT rule in Policy that caused this error
   - Quote the rule VERBATIM (copy-paste from policy above)
   - If it involves calculations (e.g., date ranges), show the formula

4. **Code Fixes** - Prioritize CODE-LEVEL fixes over hardcoded values:
   - **WRONG** (hardcoded value - only works for this specific combo):
   ```python
   scheduled_pick_date = "2025-01-28T16:00:00"
   ```
   - **CORRECT** (based on policy rule - works for ALL combos):
   ```python
   # Policy: scheduled_pick_date within [generation_window_end - 3, generation_window_end + 2]
   scheduled_pick_date = generation_window_end + timedelta(days=1)
   ```
   
For each fix, provide:
- The entity/relationship instance or tool call parameter that needs to change
- Current code/logic that's wrong
- Suggested fix code/logic that works for all combos
- Quote the Policy rule that justifies this fix

Output as JSON:
{{
  "error_category": "MISSING_INSTANCE|EXTRA_INSTANCE|POLICY_VIOLATION|INCOMPLETE_PARAMS|OTHER",
  "root_cause": "Detailed explanation of what went wrong",
  "policy_rule_quoted": "EXACT quote from policy (copy-paste), empty string if not POLICY_VIOLATION",
  "missing_params": ["Domain.tool.param1", "Domain.tool.param2"],
  "code_fixes": [
    {{
      "field": "field_name",
      "current_code": "the problematic code/value",
      "fixed_code": "suggested fix code/logic that works for all combos based on policy rule",
      "policy_basis": "why this fix satisfies the policy rule"
    }}
  ]
}}
"""
