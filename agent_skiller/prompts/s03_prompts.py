"""
Prompts for Step 03: Entity Graph Generation

Build entity relationship graph.
"""

ENTITY_RELATION_PROMPT = """
According to the entity name and the description of the entities below
<entities>
{entities}
</entities>

You need to judge which of the entities above are relevant to entity
<entity>
{entity}
</entity>
in some possible domains.

For example:
- `Course` is relevant to `Student` in Course-Management Domain;
- `Patient` is relevant to `Clinician` in Health-Care Domain.

Return the relevant entities in the format of
```json
["entity_1", "entity_2", ...]
```
"""
