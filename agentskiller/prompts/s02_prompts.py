"""
Prompts for Step 02: Entity Extraction

Extract entities from domain topics.
"""

ENTITY_EXTRACTION_PROMPT = """
## Entity Generation Guideline

You are extracting entities for multiple related domains:
```
{domains}
```

You should:
1. Identify subfields represented by the domains (Campus, Medical, Travel, ...).
2. Extract entities for all subfields and corresponding domains.

Each entity must satisfy the *Scene-Invariant Granularity Constraint*:

1. **Entity Granularity**
  - Define entities at a mid-level, domain-specific conceptual granularity 
    - ✅ Teacher, Student, Course, etc.
    - ❌ Person, Organization, or MiddleSchoolStudent
  - The entities' existence does not depend on other Entities
    - ❌ The existence of `Enrollment` depends on two other entities, then it is not an entity but a relationship
    - ✅ The existence of `Student` does not depend on other entities, then it is an entity
  - Instances of entities should not be fundamentally different across different domains
    - ❌ Instances of `TicketType` are fundamentally different in Airline and Zoo domains.
    - ✅ Instances of `Course` are not fundamentally different across different domains.

2. **Universal Attribute Set**
  - For each entity, define a *universal attribute set* that covers all properties potentially relevant across domains.
  - Scene-specific entities can only use a *proper subset* of these attributes
  - The attributes should be stable (not frequently changed)
  - The attributes should describe the entity itself, independent of others and specific domains
    - ❌ `enrollment_date` and `grade` are NOT attributes of the entity `Student`, they are attributes of the relationship between `Student` and `Course`.
    - ✅ `blood_type` and `name` are attributes of the entity `Patient`
  - Attributes should NOT be used as category labels to split instances of the entity
    - ❌ `CampusFacility` should NOT have an attribute named `facility_type` (classroom|lab|library|dining|housing|recreation|health|administration|event_space) Facilities of different types are corresponding to different domains, and thus should be divided into different entities like `ClassRoom`, `Lab` and etc.
   
  **REQUIRED: Attribute Expansion Dimensions**
  Do not limit yourself to basic IDs and Names. You MUST explore these dimensions to populate attributes:
    - **Intrinsic Physical Specs:** (e.g., area_size for a Room, storage_capacity for a Device, material for an Item)
    - **Lifecycle Metadata:** (e.g., creation_timestamp, manufacturing_year, version_str)
    - **Configuration:** (e.g., is_active, timezone, language_code, access_level)
    - **Contact/Addressing:** (e.g., ip_address, geo_coordinates, email_alias)

Example:
- Base entity: `LabEquipment`
  - Universal attributes: {{
      "equipment_id", "serial_number", "model_name", "manufacturer",  // Identity
      "purchase_date", "warranty_expiration_date",                    // Lifecycle
      "weight_kg", "dimensions", "power_consumption_watts",           // Physical
      "firmware_version", "calibration_status", "is_portable",        // State/Config
      "asset_tag"                                                     // Administration
  }}

## Output Format
Output the entities following format
```json
{{
    "entity_name": {{
        "attributes": {{
            "attr_name": {{
                "description": "",
                "type": "",
                "range": "" 
            }},
            ...
        }},
        "description": "",
        "is_person": true/false
    }},
    ...
}}
```
Output only JSON, no explanations.
"""
