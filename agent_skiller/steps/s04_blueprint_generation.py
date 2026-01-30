"""
Step 4: Blueprint Generation

Generate MCP server blueprints from entities through a multi-stage pipeline:
1. Outline Generation: Generate core-peripheral entity combinations per Person entity
2. Detail Generation: Generate full blueprints for each combination
3. Blueprint Fixup: Fix entity name misalignments
4. ID Validation: Validate and correct foreign key ID formats

Input: entities.json, entity_graph.json
Output: blueprints.json
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import BLUEPRINT_OUTLINE_PROMPT, BLUEPRINT_DETAIL_PROMPT, BLUEPRINT_DETAIL_FEEDBACK_PROMPT, BLUEPRINT_FIXUP_PROMPT
from .base import step_handler, save_json, load_json, get_client, parallel_process

logger = logging.getLogger(__name__)


# =============================================================================
# Progress Management
# =============================================================================

def load_progress(progress_path: Path) -> Dict[str, Any]:
    """Load existing progress or return empty structure."""
    if progress_path.exists():
        try:
            return load_json(progress_path)
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}, starting fresh")
    return {}


def save_progress(progress_path: Path, progress: Dict[str, Any]) -> None:
    """Save progress to intermediate file."""
    save_json(progress, progress_path)


# =============================================================================
# Stage 1: Blueprint Outline Generation
# =============================================================================

def generate_outlines_for_person(
    client,
    person_name: str,
    person_info: Dict[str, Any],
    peripheral_entities: List[str],
    entities: Dict[str, Any],
    target_count: int,
    max_peripheral_entities: int = 3,
    max_relationships_per_blueprint: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate blueprint outlines for a single Person entity.
    
    Args:
        client: LLM client
        person_name: Name of the Person entity
        person_info: Info dict for the Person entity
        peripheral_entities: List of peripheral entity names
        entities: Full entities dict for descriptions
        target_count: Number of outlines to generate
        max_peripheral_entities: Maximum peripheral entities per outline
    
    Returns:
        List of outline dicts with {entities, server_name, description}
    """
    # Format core entity info
    core_entity_str = f"{person_name}: {person_info.get('description', '')}"
    
    # Format peripheral entities info
    peripheral_lines = []
    for name in peripheral_entities:
        info = entities.get(name, {})
        peripheral_lines.append(f"- {name}: {info.get('description', '')}")
    peripheral_entities_str = "\n".join(peripheral_lines)
    
    prompt = BLUEPRINT_OUTLINE_PROMPT.format(
        core_entity=core_entity_str,
        peripheral_entities=peripheral_entities_str,
        target_count=target_count,
        max_peripheral_entities=max_peripheral_entities,
        max_relationships_per_blueprint=max_relationships_per_blueprint,
    )
    
    response = client.chat(query=prompt, model_type="textual")
    outlines = response.parse_json()
    
    if not isinstance(outlines, list):
        raise ValueError(f"Expected list of outlines, got {type(outlines)}")
    
    # Validate and normalize outlines
    validated = []
    for outline in outlines:
        if isinstance(outline, dict) and "entities" in outline:
            # Ensure core entity is first
            entities_list = outline.get("entities", [])
            if person_name not in entities_list:
                entities_list = [person_name] + entities_list
            elif entities_list[0] != person_name:
                entities_list.remove(person_name)
                entities_list = [person_name] + entities_list
            
            validated.append({
                "entities": entities_list,
                "server_name": outline.get("server_name", f"{person_name}Manager"),
                "description": outline.get("description", ""),
                "core_entity": person_name,
            })
    
    return validated


def run_stage1_outline_generation(
    entities: Dict[str, Any],
    graph_data: Dict[str, Any],
    outputs_dir: Path,
    target_count_per_person: int = 5,
    max_peripheral_entities: int = 3,
    max_relationships_per_blueprint: int = 3,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Generate blueprint outlines for all Person entities.
    
    Parallelism: Person Entity level
    Retry granularity: Person Entity level
    """
    progress_path = outputs_dir / "blueprint_outlines_progress.json"
    progress = load_progress(progress_path)
    
    if "completed_persons" not in progress:
        progress["completed_persons"] = []
    if "outlines" not in progress:
        progress["outlines"] = []
    
    # Find person entities
    person_entities = {
        name: info for name, info in entities.items()
        if info.get("is_person", False)
    }
    
    logger.info(f"Stage 1: Generating outlines for {len(person_entities)} Person entities")
    
    # Filter pending persons
    pending_persons = [
        name for name in person_entities
        if name not in progress["completed_persons"]
    ]
    
    if not pending_persons:
        logger.info("Stage 1: All Person entities already processed")
        return progress["outlines"]
    
    client = get_client()
    progress_lock = threading.Lock()
    
    def process_person(person_name: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single Person entity to generate outlines."""
        person_info = person_entities[person_name]
        
        # Find neighbors from graph
        neighbors = []
        for link in graph_data.get("links", []):
            if link.get("source") == person_name:
                neighbors.append(link.get("target"))
            elif link.get("target") == person_name:
                neighbors.append(link.get("source"))
        
        # Remove duplicates while preserving order
        seen = set()
        peripheral = []
        for n in neighbors:
            if n not in seen and n in entities:
                seen.add(n)
                peripheral.append(n)
        
        if not peripheral:
            logger.warning(f"No peripheral entities found for {person_name}")
            with progress_lock:
                progress["completed_persons"].append(person_name)
                save_progress(progress_path, progress)
            return []
        
        try:
            outlines = generate_outlines_for_person(
                client=client,
                person_name=person_name,
                person_info=person_info,
                peripheral_entities=peripheral,
                entities=entities,
                target_count=target_count_per_person,
                max_peripheral_entities=max_peripheral_entities,
                max_relationships_per_blueprint=max_relationships_per_blueprint,
            )
            
            # Thread-safe progress update
            with progress_lock:
                progress["completed_persons"].append(person_name)
                progress["outlines"].extend(outlines)
                save_progress(progress_path, progress)
            
            logger.info(f"Stage 1: Generated {len(outlines)} outlines for {person_name}")
            return outlines
            
        except Exception as e:
            logger.warning(f"Stage 1: Failed to generate outlines for {person_name}: {e}")
            raise
    
    # Process all pending persons in parallel
    results = parallel_process(
        items=pending_persons,
        process_func=process_person,
        description="Stage 1: Generating outlines",
    )
    
    # Count successful (non-None results)
    successful = sum(1 for r in results if r is not None)
    failed = len(pending_persons) - successful
    logger.info(f"Stage 1: {successful}/{len(pending_persons)} persons processed, {failed} failed")
    
    # Check if all persons are completed
    completed_count = len(progress["completed_persons"])
    total_persons = len(person_entities)
    if completed_count < total_persons:
        raise RuntimeError(
            f"Stage 1 incomplete: {completed_count}/{total_persons} persons completed. "
            f"Re-run to retry failed items."
        )
    
    return progress["outlines"]


# =============================================================================
# Stage 2: Blueprint Detail Generation
# =============================================================================

def generate_blueprint_detail(
    client,
    outline: Dict[str, Any],
    entities: Dict[str, Any],
    min_functions: int = 10,
    max_relationships: int = 2,
    max_feedback_attempts: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Generate full blueprint from an outline with feedback-based retry.
    
    Args:
        client: LLM client
        outline: Outline dict with {entities, server_name, description, core_entity}
        entities: Full entities dict
        min_functions: Minimum number of functions required
        max_relationships: Maximum number of relationships allowed
        max_feedback_attempts: Maximum feedback attempts after initial generation fails
    
    Returns:
        Full blueprint dict, or None if all attempts (including feedback) fail
    """
    entity_names = outline["entities"]
    server_name = outline["server_name"]
    description = outline["description"]
    
    # Core entity is the first one, peripheral are the rest
    core_entity = outline.get("core_entity") or entity_names[0]
    peripheral_entities = [e for e in entity_names if e != core_entity]
    
    # Build entity definitions
    entity_definitions = {}
    for name in entity_names:
        if name in entities:
            entity_definitions[name] = entities[name]
    
    entity_definitions_json = json.dumps(entity_definitions, indent=2, ensure_ascii=False)
    peripheral_entities_json = json.dumps(peripheral_entities)
    
    # Initial generation
    prompt = BLUEPRINT_DETAIL_PROMPT.format(
        server_name=server_name,
        description=description,
        entities=", ".join(entity_names),
        entity_definitions=entity_definitions_json,
        core_entity=core_entity,
        peripheral_entities_json=peripheral_entities_json,
        min_functions=min_functions,
        max_relationships=max_relationships,
    )
    
    response = client.chat(query=prompt, model_type="textual")
    blueprint = response.parse_json()
    
    if not isinstance(blueprint, dict):
        raise ValueError(f"Expected blueprint dict, got {type(blueprint)}")
    
    # Check validation and apply feedback loop if needed
    for attempt in range(max_feedback_attempts + 1):  # +1 because first check is for initial attempt
        # Validate relationships count
        relationships = blueprint.get("relationships", [])
        if len(relationships) <= max_relationships:
            # Validation passed
            blueprint["core_entity"] = core_entity
            blueprint["peripheral_entities"] = peripheral_entities
            if attempt > 0:
                logger.info(f"Blueprint '{server_name}' fixed after {attempt} feedback attempt(s)")
            return blueprint
        
        # Validation failed
        if attempt >= max_feedback_attempts:
            # No more feedback attempts
            logger.warning(
                f"Blueprint '{server_name}' still has {len(relationships)} relationships "
                f"after {max_feedback_attempts} feedback attempts (max: {max_relationships}), giving up"
            )
            return None
        
        # Use feedback prompt to retry
        error_message = (
            f"Too many relationships: Your blueprint has {len(relationships)} relationships, "
            f"but the maximum allowed is {max_relationships}. "
            f"Please reduce the number of relationships while maintaining core functionality."
        )
        
        logger.info(
            f"Blueprint '{server_name}' has {len(relationships)} relationships "
            f"(max: {max_relationships}), sending feedback (attempt {attempt + 1}/{max_feedback_attempts})"
        )
        
        feedback_prompt = BLUEPRINT_DETAIL_FEEDBACK_PROMPT.format(
            previous_blueprint=json.dumps(blueprint, indent=2, ensure_ascii=False),
            error_message=error_message,
            server_name=server_name,
            description=description,
            entities=", ".join(entity_names),
            entity_definitions=entity_definitions_json,
            core_entity=core_entity,
            peripheral_entities_json=peripheral_entities_json,
            min_functions=min_functions,
            max_relationships=max_relationships,
        )
        
        response = client.chat(query=feedback_prompt, model_type="textual")
        blueprint = response.parse_json()
        
        if not isinstance(blueprint, dict):
            raise ValueError(f"Expected blueprint dict from feedback, got {type(blueprint)}")
    
    # Should not reach here, but just in case
    return None


def run_stage2_detail_generation(
    outlines: List[Dict[str, Any]],
    entities: Dict[str, Any],
    outputs_dir: Path,
    min_functions: int = 10,
    max_relationships: int = 2,
    max_feedback_attempts: int = 2,
) -> List[Dict[str, Any]]:
    """
    Stage 2: Generate full blueprints from all outlines.
    
    Parallelism: Tuple level (all outlines in parallel)
    Retry granularity: Tuple level
    
    Blueprints with too many relationships are treated as failures after
    feedback-based retry attempts.
    """
    progress_path = outputs_dir / "blueprint_details_progress.json"
    progress = load_progress(progress_path)
    
    if "completed_indices" not in progress:
        progress["completed_indices"] = []
    if "blueprints" not in progress:
        progress["blueprints"] = {}  # index -> blueprint
    
    logger.info(f"Stage 2: Generating details for {len(outlines)} outlines")
    
    # Filter pending outlines by index
    pending_indices = [
        i for i in range(len(outlines))
        if i not in progress["completed_indices"]
    ]
    
    if not pending_indices:
        logger.info("Stage 2: All outlines already processed")
        # Return blueprints in order
        return [progress["blueprints"][str(i)] for i in range(len(outlines)) if str(i) in progress["blueprints"]]
    
    client = get_client()
    progress_lock = threading.Lock()
    
    def process_outline(idx: int) -> Dict[str, Any]:
        """Process a single outline to generate full blueprint."""
        outline = outlines[idx]
        
        blueprint = generate_blueprint_detail(
            client=client,
            outline=outline,
            entities=entities,
            min_functions=min_functions,
            max_relationships=max_relationships,
            max_feedback_attempts=max_feedback_attempts,
        )
        
        # Blueprint is None if validation failed (e.g., too many relationships)
        # Raise exception so it will be retried
        if blueprint is None:
            raise ValueError(
                f"Blueprint validation failed for '{outline.get('server_name', 'unknown')}': "
                f"exceeded max_relationships limit ({max_relationships})"
            )
        
        # Thread-safe progress update
        with progress_lock:
            progress["completed_indices"].append(idx)
            progress["blueprints"][str(idx)] = blueprint
            save_progress(progress_path, progress)
        
        logger.debug(f"Stage 2: Generated blueprint for outline {idx}: {outline.get('server_name', 'unknown')}")
        return blueprint
    
    # Process all pending outlines in parallel
    results = parallel_process(
        items=pending_indices,
        process_func=process_outline,
        description="Stage 2: Generating blueprints",
    )
    
    # Count successful (non-None results)
    successful = sum(1 for r in results if r is not None)
    failed = len(pending_indices) - successful
    logger.info(f"Stage 2: {successful}/{len(pending_indices)} blueprints generated, {failed} failed")
    
    # Check if all outlines are completed
    completed_count = len(progress["completed_indices"])
    if completed_count < len(outlines):
        raise RuntimeError(
            f"Stage 2 incomplete: {completed_count}/{len(outlines)} outlines completed. "
            f"Re-run to retry failed items."
        )
    
    # Return all blueprints in order
    return [progress["blueprints"][str(i)] for i in range(len(outlines)) if str(i) in progress["blueprints"]]


# =============================================================================
# Stage 3: Blueprint Fixup
# =============================================================================

def fixup_blueprint(
    client,
    blueprint: Dict[str, Any],
    entity_names: Set[str],
) -> Dict[str, Any]:
    """
    Fix entity name misalignments in a blueprint.
    
    Args:
        client: LLM client
        blueprint: Blueprint dict to fix
        entity_names: Set of valid entity names
    
    Returns:
        Fixed blueprint dict
    """
    # Format entity names as simple list
    entities_str = ", ".join(sorted(entity_names))
    
    prompt = BLUEPRINT_FIXUP_PROMPT.format(
        entities=entities_str,
        blueprint=json.dumps(blueprint, indent=2, ensure_ascii=False),
    )
    
    response = client.chat(query=prompt, model_type="textual")
    result = response.parse_json()
    
    # Handle both single blueprint and array response
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    elif isinstance(result, dict):
        return result
    else:
        raise ValueError(f"Unexpected fixup response format: {type(result)}")


def run_stage3_blueprint_fixup(
    blueprints: List[Dict[str, Any]],
    entity_names: Set[str],
    outputs_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Stage 3: Fix entity name misalignments in all blueprints.
    
    Parallelism: Blueprint level (all blueprints in parallel)
    Retry granularity: Blueprint level
    """
    progress_path = outputs_dir / "blueprint_fixup_progress.json"
    progress = load_progress(progress_path)
    
    if "completed_indices" not in progress:
        progress["completed_indices"] = []
    if "fixed_blueprints" not in progress:
        progress["fixed_blueprints"] = {}  # index -> fixed blueprint
    
    logger.info(f"Stage 3: Fixing {len(blueprints)} blueprints")
    
    # Filter pending blueprints by index
    pending_indices = [
        i for i in range(len(blueprints))
        if i not in progress["completed_indices"]
    ]
    
    if not pending_indices:
        logger.info("Stage 3: All blueprints already fixed")
        return [progress["fixed_blueprints"][str(i)] for i in range(len(blueprints)) if str(i) in progress["fixed_blueprints"]]
    
    client = get_client()
    progress_lock = threading.Lock()
    
    def process_blueprint(idx: int) -> Optional[Dict[str, Any]]:
        """Fix a single blueprint."""
        blueprint = blueprints[idx]
        
        try:
            fixed = fixup_blueprint(
                client=client,
                blueprint=blueprint,
                entity_names=entity_names,
            )
            
            # Thread-safe progress update
            with progress_lock:
                progress["completed_indices"].append(idx)
                progress["fixed_blueprints"][str(idx)] = fixed
                save_progress(progress_path, progress)
            
            bp_name = blueprint.get("MCP_server_name", f"blueprint_{idx}")
            logger.debug(f"Stage 3: Fixed blueprint {bp_name}")
            return fixed
            
        except Exception as e:
            logger.warning(f"Stage 3: Failed to fix blueprint {idx}: {e}")
            raise
    
    # Process all pending blueprints in parallel
    results = parallel_process(
        items=pending_indices,
        process_func=process_blueprint,
        description="Stage 3: Fixing blueprints",
    )
    
    # Count successful (non-None results)
    successful = sum(1 for r in results if r is not None)
    failed = len(pending_indices) - successful
    logger.info(f"Stage 3: {successful}/{len(pending_indices)} blueprints fixed, {failed} failed")
    
    # Check if all blueprints are fixed
    completed_count = len(progress["completed_indices"])
    if completed_count < len(blueprints):
        raise RuntimeError(
            f"Stage 3 incomplete: {completed_count}/{len(blueprints)} blueprints fixed. "
            f"Re-run to retry failed items."
        )
    
    # Return all fixed blueprints in order
    return [progress["fixed_blueprints"][str(i)] for i in range(len(blueprints)) if str(i) in progress["fixed_blueprints"]]


# =============================================================================
# Stage 4: ID Format Validation
# =============================================================================

def get_expected_id_field(entity_name: str) -> str:
    """Get expected ID field name for an entity: {EntityName.lower()}_id"""
    return f"{entity_name.lower()}_id"


def extract_entity_from_value_from_entity(value_from_entity: str, entity_names: Set[str]) -> Optional[str]:
    """
    Extract entity name from value_from_entity field.
    
    Patterns:
    - "EntityName.entityname_id" -> EntityName
    - "EntityName" -> EntityName
    - "system_generated", "system_managed", etc. -> None
    
    Args:
        value_from_entity: The value_from_entity string
        entity_names: Set of valid entity names
    
    Returns:
        Entity name if found, None otherwise
    """
    if not value_from_entity:
        return None
    
    # Skip system values
    if value_from_entity.startswith("system_"):
        return None
    
    # Try to extract entity name from "EntityName.xxx" or "EntityName"
    if "." in value_from_entity:
        entity_part = value_from_entity.split(".")[0]
    else:
        entity_part = value_from_entity
    
    # Check if this matches a known entity (case-insensitive)
    entity_name_lower_map = {name.lower(): name for name in entity_names}
    if entity_part.lower() in entity_name_lower_map:
        return entity_name_lower_map[entity_part.lower()]
    
    return None


def validate_and_fix_id_format(
    blueprint: Dict[str, Any],
    entity_names: Set[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and fix foreign key ID formats in a blueprint.
    
    Uses value_from_entity field in relationships to determine correct ID format.
    For functions, uses known entity names to validate ID parameters.
    
    Args:
        blueprint: Blueprint dict to validate
        entity_names: Set of valid entity names
    
    Returns:
        Tuple of (fixed_blueprint, list of corrections made)
    """
    corrections = []
    
    # Deep copy blueprint
    fixed = json.loads(json.dumps(blueprint))
    
    # Build case-insensitive entity lookup
    entity_name_lower_map = {name.lower(): name for name in entity_names}
    
    # ===================
    # Fix relationships using value_from_entity
    # ===================
    if "relationships" in fixed:
        for rel in fixed["relationships"]:
            if "attributes" not in rel:
                continue
            
            rel_name = rel.get("name", "unknown")
            new_attrs = {}
            
            for attr_name, attr_info in rel["attributes"].items():
                new_attr_name = attr_name
                
                # Check if this attribute has value_from_entity pointing to an entity
                if isinstance(attr_info, dict):
                    value_from_entity = attr_info.get("value_from_entity", "")
                    entity_ref = extract_entity_from_value_from_entity(value_from_entity, entity_names)
                    
                    if entity_ref:
                        # This is a foreign key - check if name follows correct format
                        expected_id = get_expected_id_field(entity_ref)
                        
                        if attr_name != expected_id:
                            new_attr_name = expected_id
                            corrections.append(
                                f"relationship '{rel_name}': {attr_name} -> {expected_id} "
                                f"(from value_from_entity: {value_from_entity})"
                            )
                            
                            # Also fix the value_from_entity field to be consistent
                            attr_info["value_from_entity"] = f"{entity_ref}.{expected_id}"
                
                new_attrs[new_attr_name] = attr_info
            
            rel["attributes"] = new_attrs
    
    # ===================
    # Fix functions parameters
    # ===================
    if "functions" in fixed:
        for func in fixed["functions"]:
            if "parameters" not in func:
                continue
            
            func_name = func.get("name", "unknown")
            new_params = {}
            
            for param_name, param_info in func["parameters"].items():
                new_param_name = param_name
                
                # Check if this looks like an ID field
                if param_name.endswith("_id"):
                    # Extract the entity part (everything before _id)
                    id_prefix = param_name[:-3]  # Remove "_id"
                    
                    # Try to find matching entity
                    if id_prefix.lower() in entity_name_lower_map:
                        # Found matching entity - check format
                        entity_ref = entity_name_lower_map[id_prefix.lower()]
                        expected_id = get_expected_id_field(entity_ref)
                        
                        if param_name != expected_id:
                            new_param_name = expected_id
                            corrections.append(
                                f"function '{func_name}': {param_name} -> {expected_id}"
                            )
                
                new_params[new_param_name] = param_info
            
            func["parameters"] = new_params
    
    return fixed, corrections


def fix_session_id_conflicts(
    blueprint: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fix session_id conflicts by renaming to param_session_id.
    
    session_id conflicts with MCP Server's session management system,
    so we rename it by adding param_ prefix.
    
    Args:
        blueprint: Blueprint dict to fix
    
    Returns:
        Tuple of (fixed_blueprint, list of corrections made)
    """
    corrections = []
    fixed = json.loads(json.dumps(blueprint))
    
    def rename_session_id(attrs: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Rename session_id to param_session_id in attribute dict."""
        new_attrs = {}
        for key, value in attrs.items():
            new_key = key
            if key == "session_id":
                new_key = "param_session_id"
                corrections.append(f"{context}: session_id -> param_session_id")
            new_attrs[new_key] = value
        return new_attrs
    
    # Fix relationships attributes
    if "relationships" in fixed:
        for rel in fixed["relationships"]:
            if "attributes" in rel:
                rel_name = rel.get("name", "unknown")
                rel["attributes"] = rename_session_id(
                    rel["attributes"],
                    f"relationship '{rel_name}'"
                )
    
    # Fix functions parameters
    if "functions" in fixed:
        for func in fixed["functions"]:
            if "parameters" in func:
                func_name = func.get("name", "unknown")
                func["parameters"] = rename_session_id(
                    func["parameters"],
                    f"function '{func_name}'"
                )
    
    return fixed, corrections


def sanitize_server_name(name: str) -> Optional[str]:
    """
    Sanitize MCP server name to be a valid Python identifier.
    
    - Remove spaces and special characters
    - Convert to PascalCase if needed
    - Ensure it starts with a letter
    
    Args:
        name: Original server name
    
    Returns:
        Sanitized name safe for Python imports, or None if name is invalid/empty
    """
    if not name:
        return None
    
    # Replace common separators with nothing (for PascalCase)
    # First, handle spaces and hyphens by capitalizing the next letter
    result = []
    capitalize_next = True
    
    for char in name:
        if char in ' -_':
            capitalize_next = True
        elif char.isalnum():
            if capitalize_next:
                result.append(char.upper())
                capitalize_next = False
            else:
                result.append(char)
        # Skip other special characters
    
    sanitized = ''.join(result)
    
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'MCP' + sanitized
    
    # Return None if empty (will be filtered out)
    if not sanitized:
        return None
    
    return sanitized


def validate_and_fix_server_names(
    blueprints: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    """
    Validate and fix MCP server names to ensure they're valid Python identifiers.
    
    Args:
        blueprints: List of blueprint dicts
    
    Returns:
        Tuple of (fixed_blueprints, list of corrections made, number of removed blueprints)
    """
    corrections = []
    fixed_blueprints = []
    seen_names = set()
    removed_count = 0
    
    for blueprint in blueprints:
        original_name = blueprint.get("MCP_server_name", "")
        sanitized_name = sanitize_server_name(original_name)
        
        # Skip blueprints with invalid/empty names
        if sanitized_name is None:
            logger.warning(f"Removing blueprint with invalid server name: '{original_name}'")
            removed_count += 1
            continue
        
        # Ensure uniqueness by adding suffix if needed
        base_name = sanitized_name
        counter = 1
        while sanitized_name in seen_names:
            sanitized_name = f"{base_name}{counter}"
            counter += 1
        
        seen_names.add(sanitized_name)
        
        fixed = json.loads(json.dumps(blueprint))
        if sanitized_name != original_name:
            fixed["MCP_server_name"] = sanitized_name
            corrections.append(f"Server name: '{original_name}' -> '{sanitized_name}'")
        
        fixed_blueprints.append(fixed)
    
    return fixed_blueprints, corrections, removed_count


def run_stage4_id_validation(
    blueprints: List[Dict[str, Any]],
    entity_names: Set[str],
) -> List[Dict[str, Any]]:
    """
    Stage 4: Validate and fix foreign key ID formats, server names, and session_id conflicts.
    
    This is a local operation (no LLM calls).
    """
    logger.info(f"Stage 4: Validating {len(blueprints)} blueprints")
    
    # First, fix server names (may remove invalid blueprints)
    blueprints, name_corrections, removed_count = validate_and_fix_server_names(blueprints)
    if removed_count > 0:
        logger.info(f"Stage 4: Removed {removed_count} blueprints with invalid server names")
    if name_corrections:
        logger.info(f"Stage 4: Fixed {len(name_corrections)} server names")
        for corr in name_corrections:
            logger.debug(f"  - {corr}")
    
    # Then, fix ID formats and session_id conflicts
    fixed_blueprints = []
    total_id_corrections = 0
    total_session_id_fixes = 0
    
    for i, blueprint in enumerate(blueprints):
        bp_name = blueprint.get("MCP_server_name", f"blueprint_{i}")
        
        # Fix ID formats
        fixed, id_corrections = validate_and_fix_id_format(blueprint, entity_names)
        if id_corrections:
            logger.info(f"Stage 4: {bp_name} - {len(id_corrections)} ID corrections")
            for corr in id_corrections:
                logger.debug(f"  - {corr}")
            total_id_corrections += len(id_corrections)
        
        # Fix session_id conflicts
        fixed, session_corrections = fix_session_id_conflicts(fixed)
        if session_corrections:
            logger.info(f"Stage 4: {bp_name} - {len(session_corrections)} session_id fixes")
            for corr in session_corrections:
                logger.debug(f"  - {corr}")
            total_session_id_fixes += len(session_corrections)
        
        fixed_blueprints.append(fixed)
    
    logger.info(
        f"Stage 4 complete: {removed_count} removed, "
        f"{len(name_corrections)} name fixes, {total_id_corrections} ID corrections, "
        f"{total_session_id_fixes} session_id fixes"
    )
    return fixed_blueprints


# =============================================================================
# Main Step Handler
# =============================================================================

@step_handler("s04_blueprint_generation", auto_retry=True)
def blueprint_generation_step(state: WorkflowState) -> WorkflowState:
    """
    Generate MCP server blueprints through multi-stage pipeline.
    
    Process:
    1. Stage 1: Generate blueprint outlines (Person Entity parallelism)
    2. Stage 2: Generate full blueprints (Tuple parallelism)
    3. Stage 3: Fix entity name misalignments (Blueprint parallelism)
    4. Stage 4: Validate and fix ID formats (Local)
    
    Output:
    - blueprints.json: List of MCPBlueprint
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    step_config = settings.get_step_config("s04_blueprint_generation")
    
    # Config
    target_count_per_person = step_config.get("target_blueprints_per_entity", 25)
    max_peripheral_entities = step_config.get("max_peripheral_entities", 3)
    min_functions_per_blueprint = step_config.get("min_functions_per_blueprint", 10)
    max_relationships_per_blueprint = step_config.get("max_relationships_per_blueprint", 2)
    max_feedback_attempts = step_config.get("max_feedback_attempts", 2)
    
    # Load data
    entities = load_json(Path(state.entities_path))
    entity_names = set(entities.keys())
    
    # Check if blueprints.json already exists (stages 1-3 completed)
    output_path = outputs_dir / "blueprints.json"
    if output_path.exists():
        logger.info("Found existing blueprints.json, skipping stages 1-3")
        fixed_blueprints = load_json(output_path)
        logger.info(f"Loaded {len(fixed_blueprints)} existing blueprints, running stage 4 only")
    else:
        # Need to run all stages
        graph_data = load_json(Path(state.entity_graph_path))
        person_count = sum(1 for info in entities.values() if info.get("is_person", False))
        
        logger.info(f"Blueprint generation: {len(entities)} entities, {person_count} persons")
        
        # ===================
        # Stage 1: Outlines
        # ===================
        outlines = run_stage1_outline_generation(
            entities=entities,
            graph_data=graph_data,
            outputs_dir=outputs_dir,
            target_count_per_person=target_count_per_person,
            max_peripheral_entities=max_peripheral_entities,
            max_relationships_per_blueprint=max_relationships_per_blueprint
        )
        
        if not outlines:
            logger.error("Stage 1 produced no outlines")
            raise ValueError("No outlines generated")
        
        logger.info(f"Stage 1 result: {len(outlines)} outlines")
        
        # ===================
        # Stage 2: Details
        # ===================
        blueprints = run_stage2_detail_generation(
            outlines=outlines,
            entities=entities,
            outputs_dir=outputs_dir,
            min_functions=min_functions_per_blueprint,
            max_relationships=max_relationships_per_blueprint,
            max_feedback_attempts=max_feedback_attempts,
        )
        
        if not blueprints:
            logger.error("Stage 2 produced no blueprints")
            raise ValueError("No blueprints generated")
        
        logger.info(f"Stage 2 result: {len(blueprints)} blueprints")
        
        # ===================
        # Stage 3: Fixup
        # ===================
        fixed_blueprints = run_stage3_blueprint_fixup(
            blueprints=blueprints,
            entity_names=entity_names,
            outputs_dir=outputs_dir,
        )
        
        logger.info(f"Stage 3 result: {len(fixed_blueprints)} fixed blueprints")
    
    # ===================
    # Stage 4: ID Validation
    # ===================
    final_blueprints = run_stage4_id_validation(
        blueprints=fixed_blueprints,
        entity_names=entity_names,
    )
    
    logger.info(f"Stage 4 result: {len(final_blueprints)} validated blueprints")
    
    # Save final output
    save_json(final_blueprints, output_path)
    state.blueprints_path = str(output_path)
    
    # Cleanup progress files after successful completion
    progress_files = [
        outputs_dir / "blueprint_outlines_progress.json",
        outputs_dir / "blueprint_details_progress.json",
        outputs_dir / "blueprint_fixup_progress.json",
    ]
    for pf in progress_files:
        if pf.exists():
            pf.unlink()
            logger.debug(f"Cleaned up progress file: {pf.name}")
    
    logger.info("Cleaned up intermediate progress files")
    
    # Update progress
    state.update_step_progress(
        "s04_blueprint_generation",
        total=len(final_blueprints),
        completed=len(final_blueprints),
    )
    
    logger.info(f"Blueprint generation complete: {len(final_blueprints)} blueprints")
    return state
