"""
Prompts for Step 04: Blueprint Generation

Generate MCP server blueprints from entities.
"""

# =============================================================================
# Stage 1: Generate Blueprint Outlines (Entity Combinations + Descriptions)
# =============================================================================

BLUEPRINT_OUTLINE_PROMPT = """
You are designing MCP (Model Context Protocol) servers for a specific core entity (user type).

<core_entity>
{core_entity}
</core_entity>

<peripheral_entities>
{peripheral_entities}
</peripheral_entities>

---

### Task
Generate exactly **{target_count}** blueprint outlines for MCP servers.

Each outline should specify:
1. A unique combination of the core entity with 1-{max_peripheral_entities} peripheral entities
2. A brief description of the MCP server's purpose and functionality

### Guidelines
- The core entity represents the user interacting with the system
- Each combination should serve a distinct, realistic use case
- Peripheral entities should have meaningful relationships with the core entity
- Cover diverse scenarios (e.g., management, scheduling, tracking, communication)
- Avoid redundant or overly similar combinations
- **IMPORTANT**: Ensure the MCP server can be built with less or equal to {max_relationships_per_blueprint} relationships between the core entity and peripheral entities.

### Output Format
Return a JSON array with exactly {target_count} outlines:
```json
[
  {{
    "entities": ["CoreEntity", "Entity1", "Entity2"],
    "server_name": "DescriptiveServerName",
    "description": "A brief description of what this MCP server does and why these entities are combined."
  }}
]
```

Output only the JSON array, no additional text.
"""


# =============================================================================
# Stage 2: Generate Full Blueprint from Outline
# =============================================================================

BLUEPRINT_DETAIL_PROMPT = """
You are implementing a detailed MCP server blueprint based on the given outline.

<outline>
Server Name: {server_name}
Description: {description}
Entities: {entities}
</outline>

<entity_definitions>
{entity_definitions}
</entity_definitions>

---

### Step by Step Generation
1. The core entity (first in the list) is the user interacting with the LLM agent.
2. Design relationships between the entities.
   - All entities and relationships are physically implemented as dataframes.
   - Relationships' attributes are columns of inter-entity interaction dataframe.
   - Relationship should have its own ID, NOT reference other relationships' IDs.
   - The core entity must be involved in all relationships. In other words, relationship that ONLY involves peripheral entities and not involve the core entity is NOT allowed.
   - Relationship attributes should ONLY reference entity attributes as foreign keys, NOT other relationships' keys. Each relationship connects entities directly (e.g., a relationship can have student_id and course_id, but should NOT have enrollment_id pointing to another relationship).
3. Design functions that operate the relationship dataframe (query records, add record, update record, etc).
4. Design auxiliary functions that complete the MCP server functionality.
   - For example, `list_all_courses()` in a `CourseManagement` MCP server.

### Requirements
- Only up to {max_relationships} relationships are allowed between two entities.
- The MCP server should include at least {min_functions} functions.
- Function parameters should NOT assume relationships or entities that do not exist.
- An authorization function (only one) is needed to authorize the user (core entity) with its ID at the beginning. The authorization function should ONLY return Success/Failed status, NOT return any tokens (e.g., auth_token). The session context will manage authentication state internally.
- You are NOT allowed to add new attributes to entities.
- You are NOT allowed to add new entities beyond those specified.
- There should be dependencies among functions to increase interaction complexity:
  - Input parameters of core functions should rely on outputs of other relevant functions.
  - For example, before enrolling a Student to a Course, check prerequisites first.

### Output Format
Return your answer in JSON format:
```json
{{
  "MCP_server_name": "{server_name}",
  "description": "{description}",
  "core_entity": "{core_entity}",
  "peripheral_entities": {peripheral_entities_json},
  "relationships": [
    {{
      "name": "",
      "description": "",
      "attributes": {{
        "attr_name": {{
          "type": "",
          "value_from_entity": "",
          "range": ""
        }}
      }}
    }}
  ],
  "functions": [
    {{
      "name": "",
      "description": "",
      "legal_accessor": [],
      "parameters": {{
        "param_name": {{
          "description": "",
          "type": "",
          "range": ""
        }}
      }}
    }}
  ]
}}
```

Output only the JSON object, no additional text.
"""


# =============================================================================
# Stage 2 Feedback: Retry with Error Context
# =============================================================================

BLUEPRINT_DETAIL_FEEDBACK_PROMPT = """
Your previous blueprint generation attempt failed validation. Please fix the issue and regenerate.

<previous_attempt>
{previous_blueprint}
</previous_attempt>

<error>
{error_message}
</error>

<outline>
Server Name: {server_name}
Description: {description}
Entities: {entities}
</outline>

<entity_definitions>
{entity_definitions}
</entity_definitions>

---

### How to Fix
1. Review the error message carefully.
2. If there are too many relationships:
   - Reduce the number of relationships to at most {max_relationships}.
   - You may need to simplify the server description to remove some entity interactions.
   - Focus on the most essential relationships for the core functionality.
3. Preserve the core functionality described in the outline.
4. peripheral_entity_a_id and peripheral_entity_b_id should NOT be merged into a unified ID field for relationship reduction.
5. Re-design the functions to match the reduced number of relationships.

### Requirements (same as before)
- Only up to {max_relationships} relationships are allowed between two entities.
- The MCP server should include at least {min_functions} functions.
- Function parameters should NOT assume relationships or entities that do not exist.
- An authorization function (only one) is needed to authorize the user (core entity) with its ID at the beginning. The authorization function should ONLY return Success/Failed status, NOT return any tokens (e.g., auth_token). The session context will manage authentication state internally.
- You are NOT allowed to add new attributes to entities.
- You are NOT allowed to add new entities beyond those specified.
- Relationship should have its own ID, NOT reference other relationships' IDs.
- Relationship attributes should ONLY reference entity attributes as foreign keys, NOT other relationships' keys. Each relationship connects entities directly (e.g., a relationship can have student_id and course_id, but should NOT have enrollment_id pointing to another relationship).
- The core entity must be involved in all relationships. In other words, relationship that ONLY involves peripheral entities and not involve the core entity is NOT allowed.

### Output Format
Return your FIXED blueprint in JSON format:
```json
{{
  "MCP_server_name": "{server_name}",
  "description": "<updated_description_if_needed>",
  "core_entity": "{core_entity}",
  "peripheral_entities": {peripheral_entities_json},
  "relationships": [
    {{
      "name": "",
      "description": "",
      "attributes": {{
        "attr_name": {{
          "type": "",
          "value_from_entity": "",
          "range": ""
        }}
      }}
    }}
  ],
  "functions": [
    {{
      "name": "",
      "description": "",
      "legal_accessor": [],
      "parameters": {{
        "param_name": {{
          "description": "",
          "type": "",
          "range": ""
        }}
      }}
    }}
  ]
}}
```

Output only the JSON object, no additional text.
"""


# =============================================================================
# Blueprint Fixup
# =============================================================================

BLUEPRINT_FIXUP_PROMPT = """
<entities>
{entities}
</entities>

<blueprint>
{blueprint}
</blueprint>

Please carefully examine the given blueprint:
1. Whether the blueprint contains entities that are misaligned with the given entities
  - For example, blueprint includes entity `Advisor` but only `AcademicAdvisor` is available in the given entities
2. Replace the non-existent entity in the blueprint with the most likely entity provided
3. The description of the blueprint mis-matches the entities and relationships, please revise the description to match the entities and relationships
4. Reference keys that are not in the given entities is NOT allowed, try reduce the number of peripheral entities and fix the blueprint description, relationships and functions to ensure valid reference keys in relationships and functions.
5. Relationship should have its own ID, NOT reference other relationships' IDs.
6. The core entity must be involved in all relationships. In other words, relationship that ONLY involves peripheral entities and not involve the core entity is NOT allowed.

Output the corrected blueprint in the same JSON format.
If nothing needs to be corrected, just repeat the given blueprint.

Output ONLY the JSON array, no extra explanations:
```json
[
  {{
    "MCP_server_name": "",
    "description": "",
    "core_entity": "",
    "peripheral_entities": [],
    "relationships": [],
    "functions": []
  }}
]
```
"""
