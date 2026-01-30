"""
Prompts for Step 06: Database Generation

Generate mock databases for entities and relationships through a 6-stage pipeline.
"""

# =============================================================================
# Sub-step 2: Entity Database Generation
# =============================================================================

ENTITY_DATABASE_PROMPT = """
I have an entity that has the following attributes.
<Entity>
{entity}
</Entity>

Your job is to develop a Python script that generates a mock database with good diversity on the entity attributes.

## Requirements

1. Generate a JSON file named `{entity_name}.json` at path: "{output_path}"
2. The database should contain approximately {records_count} records.
3. Strictly follow the JSON schema:
```json
[
    {{"attribute1": "", "attribute2": "", ...}},
    ...
]
```
4. You can use relevant libraries (e.g., `Faker`, `random`, `uuid`) for fake data generation.
5. If the database includes IDs, make sure the IDs are unique. Always use `uuid.uuid4()` for entity id (no prefix allowed).
6. The current time is {simulation_time}.
7. Strictly follow the value ranges of the attributes as specified in the entity definition.
8. If no value range is provided, sample the value in a reasonable range and comment the range or sample logic in the code.
9. The final output should be saved to the specified output path.

## Code Structure

```python
import json
import uuid
import random
from pathlib import Path
# ... other imports as needed

def generate_{entity_name_lower}_data(num_records: int = {records_count}) -> list:
    \"\"\"Generate mock data for {entity_name} entity.\"\"\"
    records = []
    for _ in range(num_records):
        record = {{
            # ... generate each attribute
        }}
        records.append(record)
    return records

if __name__ == "__main__":
    data = generate_{entity_name_lower}_data()
    output_path = Path("{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated {{len(data)}} records")
```

Output only the Python code, no markdown formatting.
"""


# =============================================================================
# Sub-step 3 & 6: Database Summary
# =============================================================================

DATABASE_SUMMARY_PROMPT = """
Analyze the following database generation code and create a **precise, actionable summary** of the value ranges and generation logic.

<code>
{code}
</code>

## CRITICAL REQUIREMENTS

1. **List EXACT values, not categories**:
   - ❌ Wrong: "degree_type: Master, Bachelor, Doctorate"
   - ✅ Correct: "degree_type: ['Master of Science', 'Master of Arts', 'Bachelor of Science', 'Bachelor of Arts', 'Doctor of Philosophy', ...]"

2. **For enumerated fields**: List ALL possible values exactly as they appear in the code.

3. **For pattern-based fields**: Describe the exact pattern/format (e.g., "UUID v4", "YYYY-MM-DD", "+1-XXX-XXX-XXXX").

4. **For range-based fields**: Specify exact min/max values and distribution if applicable.

5. **For derived fields**: Explain the exact derivation logic.

## OUTPUT FORMAT

```markdown
# {name} ({type})

## Generation Config
- Record count: X
- Random seed: X (if any)

## Field Specifications

### field_name_1
- **Type**: string / integer / boolean / date / etc.
- **Possible Values**: [exact list] OR pattern description
- **Generation Logic**: how it's generated

### field_name_2
...
```

**IMPORTANT**: Be exhaustive and precise. This summary will be used for constraint validation in downstream tasks.

Output only the summary in markdown format.
"""


# =============================================================================
# Sub-step 4: Constraint Identification
# =============================================================================

CONSTRAINT_IDENTIFICATION_PROMPT = """
The current time is {simulation_time}.

I'm generating a mock database for the following entity relationship:
<Relationship>
{relationship}
</Relationship>

Here are the value ranges summaries of the entities involved in this relationship:
<Entity Database Summary>
{entity_database_summary}
</Entity Database Summary>

## Problem

Randomly combining instances of entities may cause some impossible or unrealistic situations in the real world.

Examples of such conflicts:
- In relationship (`Teacher` -- Teach -- `Course`), a history teacher teaching a math course is not possible.
- In relationship (`Patient` -- Appointment -- `Doctor`), a patient with a heart disease making an appointment with a neurologist is not possible.
- In relationship (`Employee` -- WorksIn -- `Department`), an engineer working in HR department is not possible.

## Instruction

Your job is to identify all the attributes or attribute tuples that may cause logical conflicts when generating mock relationship data.

## Output Format

Return a JSON object with the following structure:
```json
{{
    "relationship_name": "string",
    "risky_attributes": [
        {{
            "attributes": ["attribute1", "attribute2"],
            "entities_involved": ["Entity1.field1", "Entity2.field2"],
            "conflict_type": "semantic_mismatch | temporal_conflict | domain_specific | logical_dependency",
            "description": "Explanation of why these attributes may conflict",
            "constraint_rule": "Rule to ensure valid combinations (e.g., 'Entity1.field1 must be compatible with Entity2.field2')"
        }}
    ],
    "generation_recommendations": [
        "Recommendation 1 for avoiding conflicts",
        "Recommendation 2..."
    ]
}}
```

If no risky attributes are identified, return:
```json
{{
    "relationship_name": "string",
    "risky_attributes": [],
    "generation_recommendations": ["No constraints needed - attributes can be combined freely"]
}}
```

Output only the JSON, no markdown formatting or explanations.
"""


# =============================================================================
# Sub-step 5: Relationship Database Generation
# =============================================================================

RELATIONSHIP_DATABASE_PROMPT = """
The current time is {simulation_time}.

## Blueprint Context
<Blueprint>
{blueprint}
</Blueprint>

## Target Relationship
<Relationship>
{relationship}
</Relationship>

## Entity Database Summaries
<Entity Database Summary>
{entity_database_summary}
</Entity Database Summary>

## Identified Constraints
<Constraints>
{constraints}
</Constraints>

## Task

Write a Python script to generate mock relationship database with rule-based constraints.

Entity databases are located at `{entity_db_path}/<EntityName>.json` with format:
```json
[
    {{"attribute1": "", "attribute2": "", ...}}
]
```

You should refer to the instances in Entity database if `value_from_entity` is not `N/A` or `Random`.
That means: load the Entity database and sample the instances with constraints.

## Requirements

1. Generate a JSON file named `{relationship_name}.json` at path: "{output_path}"
2. The database should contain approximately {records_count} records.
3. Strictly follow the JSON schema:
```json
[
    {{"attribute1": "", "attribute2": "", ...}},
    ...
]
```
4. You can use relevant libraries (e.g., `Faker`, `random`, `uuid`) for fake data generation.
5. Always use `uuid.uuid4()` for relationship id (no prefix allowed).
6. Strictly follow the value ranges of the attributes.
7. **IMPORTANT**: Apply the identified constraints to ensure valid entity combinations.
8. If no value range is provided, sample the value in a reasonable range and comment the logic.
9. The final output should be saved to the specified output path.

## IMPORTANT: Constraint-Based Sampling Strategy

**DO NOT** use random sampling + filtering approach, which is inefficient when entity instances are large.

**INSTEAD**, use constraint-driven active construction:

1. **Pre-index entities by constraint-relevant attributes**: Build lookup dictionaries/groups
2. **Design matching rules**: Based on constraints, define how to find compatible entity pairs
3. **Sample from valid groups directly**: Pick from pre-filtered compatible pools
4. **Non-Entity Attributes**: For non-entity attributes (e.g., timestamps, status, amounts), apply logical constraints:
   - **Temporal**: appointment_time > current_time, end_date > start_date, created_at < updated_at
   - **Numerical**: total_amount = unit_price * quantity, balance >= 0
   - **Status**: status must be valid for the entity state (e.g., "completed" only if all prerequisites met)
   - **Derived**: values computed from selected entity attributes (e.g., patient age from birth_date)

### Example: Teacher-Course constraint (teacher's specialty must match course subject)

```python
# BAD: Random sample + filter (O(n*m) worst case, inefficient)
for _ in range(num_records):
    teacher = random.choice(teachers)
    course = random.choice(courses)
    if teacher["specialty"] == course["subject"]:  # May fail many times!
        records.append(...)

# GOOD: Pre-index + direct sampling (O(n+m) setup, O(1) per record)
# Step 1: Group courses by subject
courses_by_subject = defaultdict(list)
for course in courses:
    courses_by_subject[course["subject"]].append(course)

# Step 2: Sample teacher, then pick from matching courses
for _ in range(num_records):
    teacher = random.choice(teachers)
    matching_courses = courses_by_subject.get(teacher["specialty"], [])
    if matching_courses:
        course = random.choice(matching_courses)
        records.append(...)
```

## Code Structure

```python
import json
import uuid
import random
from pathlib import Path
from collections import defaultdict
# ... other imports as needed

def load_entity_database(entity_name: str) -> list:
    \"\"\"Load entity database from file.\"\"\"
    path = Path("{entity_db_path}") / f"{{entity_name}}.json"
    with open(path) as f:
        return json.load(f)

def build_constraint_index(entities: list, key_field: str) -> dict:
    \"\"\"Build index for constraint-based lookups.\"\"\"
    index = defaultdict(list)
    for entity in entities:
        key = entity.get(key_field)
        if key:
            index[key].append(entity)
    return index

def generate_{relationship_name_lower}_data(num_records: int = {records_count}) -> list:
    \"\"\"Generate mock data for {relationship_name} relationship.
    
    Uses constraint-driven sampling for efficiency:
    1. Load and index entity databases by constraint-relevant fields
    2. For each record, select primary entity then find compatible secondary entities
    3. Generate relationship attributes based on selected entities
    \"\"\"
    # Load entity databases
    # entity1_data = load_entity_database("Entity1")
    # entity2_data = load_entity_database("Entity2")
    
    # Build constraint indices (based on identified constraints)
    # entity2_by_field = build_constraint_index(entity2_data, "relevant_field")
    
    records = []
    attempts = 0
    max_attempts = num_records * 3  # Safety limit
    
    while len(records) < num_records and attempts < max_attempts:
        attempts += 1
        # Select primary entity
        # entity1 = random.choice(entity1_data)
        
        # Find compatible secondary entities using index
        # compatible_entity2s = entity2_by_field.get(entity1["matching_field"], [])
        # if not compatible_entity2s:
        #     continue
        # entity2 = random.choice(compatible_entity2s)
        
        record = {{
            # ... generate each attribute using selected entities
        }}
        records.append(record)
    
    return records

if __name__ == "__main__":
    data = generate_{relationship_name_lower}_data()
    output_path = Path("{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated {{len(data)}} records")
```

Output only the Python code, no markdown formatting.
"""


# =============================================================================
# Legacy: Risky Attributes Prompt (kept for backward compatibility)
# =============================================================================

RISKY_ATTRIBUTES_PROMPT = """
The current time is {simulation_time}.
I'm generating mock database for the following entity relationship
<Relationship>
{relationship}
</Relationship>
It is the relationship of the following entities with the following value ranges summary
<Entity Database Summary>
{entity_database_summary}
</Entity Database Summary>

#### Problem
Randomly combining instances of entities may cause some impossible situations in the real-world.
For example:
- in relationship (`Teacher` -- Teach -- `Course`), a history teacher teaching a math course is not possible.
- in relationship (`Patient` -- Appointment -- `Doctor`), a patient with a heart disease appointment with a neurologist is not possible.

#### Instruction
Your job is to list all the attributes or attribute tuples of the entities that may cause conflict in the mock relationship database.
"""
