"""
Step 15: Instance Combos Selection

Select entity instance combinations for task instantiation using a Plan-Execution approach.

Process:
1. Load task templates and filter relevant policy sections
2. Phase 1 (Plan): LLM analyzes constraints and creates selection plan
   - Must complete successfully before Phase 2
   - Triggers step-wise retry if fails
3. Phase 2 (Execute): LLM generates selection code, executed via subprocess
   - Uses BlockEditor to fix code on failure
   - Samples samples_per_template instances per trajectory

Single Domain vs Cross Domain:
- Single Domain: Standard constraint analysis + code generation
  - Sampling codes saved to outputs/sampling_codes/{server}/{hash}.py for reuse
- Cross Domain: Code fusion approach
  - Parse trajectory to extract domain-specific segments
  - Load corresponding Single Domain sampling codes
  - LLM fuses codes and adds global constraints (e.g., shared entities)

Input: task_templates/*.json, database/outputs/**, policies/*.md
Output: combinations/*.json, sampling_codes/{server}/*.py

Parallelism: Trajectory-wise
Retry granularity: Trajectory-wise
"""

import json
import logging
import subprocess
import tempfile
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pprint import pprint

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..core.policy_parser import (
    filter_policy_for_trajectory,
    has_structured_markers,
)
from ..core.block_editor import WorkflowBlockEditor
from ..prompts import (
    INSTANCE_ASSIGNMENT_EXAMPLES,
    CONSTRAINT_ANALYSIS_PROMPT,
    SELECTION_CODE_PROMPT,
    CODE_FUSION_PROMPT,
)
from .base import step_handler, save_json, load_json, ensure_dir, get_client, parallel_process

# Import trajectory hash from evaluator module (canonical implementation)
from evaluator import compute_trajectory_hash

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result for Cross Domain Code Caching
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of trajectory validation, including code information for caching.
    
    Used to pass execution results and generated codes from validation
    to the caller for saving to trajectory_code/.
    """
    success: bool
    error_message: Optional[str] = None
    execution_log: Optional[str] = None
    # Code caching fields (populated on success)
    step_param_codes: Optional[Dict[int, str]] = None
    generated_code: Optional[str] = None
    entity_context: Optional[Dict[str, Any]] = field(default_factory=dict)
    domains: Optional[List[str]] = field(default_factory=list)
    is_cross_domain: bool = True


def compute_combo_id(trajectory_hash: str, entity_instances: Dict[str, Any]) -> str:
    """
    Compute a unique ID for a combo based on trajectory_hash and entity_instances.
    
    This ID is generated at combo creation time and propagated to queries,
    allowing reverse lookup from query to its source combination.
    """
    content = json.dumps({
        "trajectory_hash": trajectory_hash,
        "entity_instances": entity_instances,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()[:12]


# =============================================================================
# Database Path Helpers (Single vs Cross Domain)
# =============================================================================

def is_cross_domain(server_name: str) -> bool:
    """Check if server is cross-domain (contains '_' in name)."""
    return "_" in server_name


def get_entity_dir(db_dir: Path, server_name: str) -> Path:
    """
    Get entity directory path based on domain type.
    
    Single Domain: outputs/database/outputs/entities/
    Cross Domain: outputs/database/outputs/entities/{ServerName}/
    """
    if is_cross_domain(server_name):
        return db_dir / "entities" / server_name
    else:
        return db_dir / "entities"


def get_relationship_dir(db_dir: Path, server_name: str) -> Path:
    """
    Get relationship directory path.
    
    Both modes: outputs/database/outputs/relationships/{ServerName}/
    """
    return db_dir / "relationships" / server_name


def get_available_entities(db_dir: Path, server_name: str = None) -> List[str]:
    """Get list of available entity types from database."""
    if server_name:
        entities_dir = get_entity_dir(db_dir, server_name)
        if not entities_dir.exists():
            # Fallback to global entities dir for single domain
            entities_dir = db_dir / "entities"
    else:
        entities_dir = db_dir / "entities"
    
    if not entities_dir.exists():
        return []
    return [f.stem for f in entities_dir.glob("*.json")]


def get_available_relationships(db_dir: Path, server_name: str) -> List[str]:
    """Get list of available relationship types from database."""
    rel_dir = get_relationship_dir(db_dir, server_name)
    if not rel_dir.exists():
        return []
    return [f.stem for f in rel_dir.glob("*.json")]


def get_relationship_summaries(summary_dir: Path, server_name: str) -> str:
    """
    Get relationship summaries from database_summary directory.
    
    Reads the .md summary files which contain detailed field specifications,
    generation logic, and constraints for each relationship.
    
    Args:
        summary_dir: Path to database_summary directory (e.g., outputs/database_summary)
        server_name: Server name
        
    Returns:
        Combined markdown string of all relationship summaries
    """
    rel_summary_dir = summary_dir / "relationships" / server_name
    if not rel_summary_dir.exists():
        return ""
    
    summaries = []
    for summary_file in sorted(rel_summary_dir.glob("*.md")):
        try:
            content = summary_file.read_text()
            summaries.append(f"### {summary_file.stem}\n\n{content}")
        except Exception:
            pass
    
    return "\n\n---\n\n".join(summaries) if summaries else ""


def get_entity_summaries(summary_dir: Path, available_entities: List[str]) -> str:
    """
    Get entity summaries from database_summary directory.
    
    Reads the .md summary files which contain detailed field specifications,
    generation logic, and constraints for each entity.
    
    Args:
        summary_dir: Path to database_summary directory (e.g., outputs/database_summary)
        available_entities: List of entity names to load summaries for
        
    Returns:
        Combined markdown string of all entity summaries
    """
    entity_summary_dir = summary_dir / "entities"
    if not entity_summary_dir.exists():
        return ""
    
    summaries = []
    for entity_name in available_entities:
        summary_file = entity_summary_dir / f"{entity_name}.md"
        if summary_file.exists():
            try:
                content = summary_file.read_text()
                summaries.append(f"### {entity_name}\n\n{content}")
            except Exception:
                pass
    
    return "\n\n---\n\n".join(summaries) if summaries else ""


def load_policy_for_server(policies_dir: Path, server_name: str) -> str:
    """Load policy file for a server."""
    policy_path = policies_dir / f"{server_name}.md"
    if policy_path.exists():
        return policy_path.read_text()
    return ""


# =============================================================================
# Sampling Code Persistence (for Cross Domain reuse)
# =============================================================================

def get_sampling_codes_dir(outputs_dir: Path, server_name: str) -> Path:
    """Get directory for storing sampling codes."""
    return outputs_dir / "sampling_codes" / server_name


def save_sampling_code(
    outputs_dir: Path,
    server_name: str,
    trajectory_hash: str,
    code: str,
    constraint_analysis: Dict[str, Any],
) -> Path:
    """
    Save sampling code for a Single Domain trajectory.
    
    This allows Cross Domain processing to reuse the code.
    
    Args:
        outputs_dir: Base outputs directory
        server_name: Server name (Single Domain only)
        trajectory_hash: Hash of the trajectory
        code: The Python sampling code
        constraint_analysis: The constraint analysis JSON
        
    Returns:
        Path to the saved code file
    """
    codes_dir = ensure_dir(get_sampling_codes_dir(outputs_dir, server_name))
    
    # Save the code
    code_path = codes_dir / f"{trajectory_hash}.py"
    code_path.write_text(code)
    
    # Save the constraint analysis as metadata
    meta_path = codes_dir / f"{trajectory_hash}.json"
    save_json({
        "trajectory_hash": trajectory_hash,
        "server_name": server_name,
        "constraint_analysis": constraint_analysis,
        "code_path": str(code_path),
        "created_at": datetime.now().isoformat(),
    }, meta_path)
    
    logger.debug(f"Saved sampling code for {server_name}/{trajectory_hash}")
    return code_path


def load_sampling_code(
    outputs_dir: Path,
    server_name: str,
    trajectory_hash: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Load saved sampling code for a Single Domain trajectory.
    
    Args:
        outputs_dir: Base outputs directory
        server_name: Server name
        trajectory_hash: Hash of the trajectory
        
    Returns:
        Tuple of (code, constraint_analysis) or None if not found
    """
    codes_dir = get_sampling_codes_dir(outputs_dir, server_name)
    
    code_path = codes_dir / f"{trajectory_hash}.py"
    meta_path = codes_dir / f"{trajectory_hash}.json"
    
    if not code_path.exists() or not meta_path.exists():
        return None
    
    try:
        code = code_path.read_text()
        meta = load_json(meta_path)
        return code, meta.get("constraint_analysis", {})
    except Exception as e:
        logger.warning(f"Failed to load sampling code for {server_name}/{trajectory_hash}: {e}")
        return None


# =============================================================================
# Cross Domain Trajectory Parsing
# =============================================================================

def parse_cross_domain_trajectory(trajectory: List[str]) -> Dict[str, List[str]]:
    """
    Parse a Cross Domain trajectory into domain-specific segments.
    
    Cross Domain trajectories have format: ["ServerA.tool1", "ServerA.tool2", "ServerB.tool3"]
    
    Args:
        trajectory: List of tool calls in format "ServerName.tool_name"
        
    Returns:
        Dict mapping server name to list of tools (without server prefix)
        Example: {"ServerA": ["tool1", "tool2"], "ServerB": ["tool3"]}
    """
    domain_segments = {}
    
    for tool_call in trajectory:
        if "." in tool_call:
            server_name, tool_name = tool_call.split(".", 1)
            if server_name not in domain_segments:
                domain_segments[server_name] = []
            domain_segments[server_name].append(tool_name)
        else:
            # Fallback for tools without server prefix
            logger.warning(f"Tool without server prefix in Cross Domain trajectory: {tool_call}")
    
    return domain_segments


def find_matching_single_domain_codes(
    outputs_dir: Path,
    domain_segments: Dict[str, List[str]],
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """
    Find matching Single Domain sampling codes for each domain segment.
    
    Args:
        outputs_dir: Base outputs directory
        domain_segments: Dict from parse_cross_domain_trajectory
        
    Returns:
        Dict mapping server name to (code, constraint_analysis) tuple
        Only includes domains where a matching code was found
    """
    found_codes = {}
    
    for server_name, tools in domain_segments.items():
        # Compute hash for this segment
        segment_hash = compute_trajectory_hash(tools)
        
        # Try to load the code
        result = load_sampling_code(outputs_dir, server_name, segment_hash)
        if result:
            found_codes[server_name] = result
            logger.debug(f"Found sampling code for {server_name}/{segment_hash}")
        else:
            logger.debug(f"No sampling code found for {server_name}/{segment_hash}")
    
    return found_codes


def get_shared_entities_from_combination(combo_data: Dict[str, Any]) -> List[str]:
    """
    Extract shared entities from a Cross Domain combination file.
    
    Args:
        combo_data: The combination data loaded from cross_domain_combinations
        
    Returns:
        List of shared entity names
    """
    combination = combo_data.get("combination", {})
    return combination.get("shared_entities", [])


def get_core_entity_from_combination(combo_data: Dict[str, Any]) -> str:
    """
    Extract the core entity from a Cross Domain combination.
    
    The core entity represents the user initiating the interaction and MUST be
    consistent across all domains.
    
    Args:
        combo_data: The combination data loaded from cross_domain_combinations
        
    Returns:
        The core entity name (e.g., "Customer"). Returns empty string if not found.
    """
    combination = combo_data.get("combination", {})
    core_entities = combination.get("core_entities", [])
    # core_entities is a list like ["Customer", "Customer"] - all should be the same
    return core_entities[0] if core_entities else ""


# =============================================================================
# Tool Parameter Extraction for Validation
# =============================================================================

def get_trajectory_tool_params(
    trajectory: List[str],
    outputs_dir: Path,
) -> Dict[str, List[str]]:
    """
    Extract parameter names for each tool in the trajectory from tool_lists.
    
    Args:
        trajectory: List of tool calls in format "ServerName.tool_name" (Cross Domain)
                   or just "tool_name" (Single Domain)
        outputs_dir: Base outputs directory containing tool_lists/
        
    Returns:
        Dict mapping "Domain.tool_name" to list of parameter names
        Example: {"ServerA.create_order": ["user_id", "amount", "currency"]}
    
    Note: Value ranges are provided by Policy, so we only need param names here.
    """
    tool_lists_dir = outputs_dir / "tool_lists"
    result = {}
    
    # Cache loaded tool lists to avoid repeated file reads
    tool_list_cache: Dict[str, List[Dict]] = {}
    
    for tool_call in trajectory:
        # Parse domain and tool name
        if "." in tool_call:
            domain, tool_name = tool_call.split(".", 1)
        else:
            # Single domain - we'd need the server_name from context
            # For now, skip (this function is mainly for Cross Domain)
            logger.warning(f"Tool without domain prefix: {tool_call}")
            continue
        
        # Load tool list for this domain if not cached
        if domain not in tool_list_cache:
            tool_list_path = tool_lists_dir / f"{domain}.json"
            if tool_list_path.exists():
                try:
                    tool_list_cache[domain] = load_json(tool_list_path)
                except Exception as e:
                    logger.warning(f"Failed to load tool list for {domain}: {e}")
                    tool_list_cache[domain] = []
            else:
                logger.warning(f"Tool list not found: {tool_list_path}")
                tool_list_cache[domain] = []
        
        # Find the tool and extract parameter names
        tool_list = tool_list_cache[domain]
        param_names = []
        
        for tool_def in tool_list:
            func_def = tool_def.get("function", {})
            if func_def.get("name") == tool_name:
                # Extract parameter names from the schema
                params_schema = func_def.get("parameters", {})
                properties = params_schema.get("properties", {})
                param_names = list(properties.keys())
                break
        
        # Store with full key "Domain.tool_name"
        result[tool_call] = param_names
        
        if not param_names:
            logger.debug(f"No parameters found for tool: {tool_call}")
    
    return result


def validate_value_domain_completeness(
    combo: Dict[str, Any],
    trajectory: List[str],
    tool_params: Dict[str, List[str]],
) -> Tuple[bool, List[str]]:
    """
    Validate that value_domain_samples covers all tool parameters.
    
    Valid values in value_domain_samples:
    1. Concrete value (string, number, boolean, etc.)
    2. "<From previous tool call output>" marker for dynamic params
    3. None (explicitly set, especially for optional params)
    
    A parameter is considered "covered" if an entry exists in value_domain_samples
    with key "Domain.tool_name.param_name", regardless of the value type.
    
    Args:
        combo: The combination dict with entity_instances and value_domain_samples
        trajectory: List of tool calls
        tool_params: Dict from get_trajectory_tool_params()
        
    Returns:
        Tuple of (is_complete, list_of_missing_params)
        Missing params format: "Domain.tool.param"
    """
    value_domain_samples = combo.get("value_domain_samples", {})
    missing_params = []
    
    for tool_call in trajectory:
        # Get expected params for this tool
        expected_params = tool_params.get(tool_call, [])
        
        for param_name in expected_params:
            # Build the expected key
            vds_key = f"{tool_call}.{param_name}"
            
            # Check if key exists in value_domain_samples
            # Note: We check if key EXISTS, not if value is truthy
            # This allows None values and empty strings as valid entries
            if vds_key not in value_domain_samples:
                missing_params.append(vds_key)
    
    is_complete = len(missing_params) == 0
    return is_complete, missing_params


# =============================================================================
# Cross Domain Code Fusion
# =============================================================================

def build_trajectory_structure_description(
    trajectory: List[str],
    domain_segments: Dict[str, List[str]],
) -> str:
    """
    Build a description of how the Cross Domain trajectory is structured.
    
    Args:
        trajectory: The full Cross Domain trajectory
        domain_segments: Parsed domain segments
        
    Returns:
        Human-readable description of trajectory structure
    """
    lines = ["The Cross Domain trajectory consists of:"]
    for server_name, tools in domain_segments.items():
        tools_str = " -> ".join(tools)
        lines.append(f"- {server_name}: [{tools_str}]")
    return "\n".join(lines)


def format_single_domain_codes_for_prompt(
    found_codes: Dict[str, Tuple[str, Dict[str, Any]]],
    domain_segments: Dict[str, List[str]],
) -> str:
    """
    Format Single Domain codes for the CODE_FUSION_PROMPT.
    
    Args:
        found_codes: Dict from find_matching_single_domain_codes
        domain_segments: Parsed domain segments
        
    Returns:
        Formatted string for the prompt
    """
    sections = []
    
    for server_name, tools in domain_segments.items():
        if server_name in found_codes:
            code, constraint_analysis = found_codes[server_name]
            segment_hash = compute_trajectory_hash(tools)
            
            section = f"""
### {server_name} (trajectory: {' -> '.join(tools)}, hash: {segment_hash})

**Constraint Analysis Summary:**
- Instance Assignment: {json.dumps(constraint_analysis.get('instance_assignment_plan', {}), indent=2)}
- Key Constraints: {len(constraint_analysis.get('sampling_requirements', []))} sampling requirements

**Sampling Code:**
```python
{code}
```
"""
            sections.append(section)
        else:
            section = f"""
### {server_name} (trajectory: {' -> '.join(tools)})

**Note:** No pre-existing sampling code found for this domain segment.
You will need to generate the sampling logic for this domain from scratch based on the policy.
"""
            sections.append(section)
    
    return "\n---\n".join(sections)


def generate_fused_selection_code(
    client,
    trajectory: List[str],
    domain_segments: Dict[str, List[str]],
    found_codes: Dict[str, Tuple[str, Dict[str, Any]]],
    core_entity: str,
    policy_content: str,
    db_dir: Path,
    server_name: str,
) -> Optional[str]:
    """
    Generate fused selection code for Cross Domain trajectory.
    
    Args:
        client: LLM client
        trajectory: Full Cross Domain trajectory
        domain_segments: Parsed domain segments
        found_codes: Single Domain codes that were found
        core_entity: The core entity that must be consistent across domains (e.g., "Customer")
        policy_content: Cross Domain policy
        db_dir: Database directory
        server_name: Cross Domain server name (e.g., "DomainA_DomainB")
        
    Returns:
        Fused Python code or None if failed
    """
    # Build prompt inputs
    trajectory_structure = build_trajectory_structure_description(trajectory, domain_segments)
    single_domain_codes = format_single_domain_codes_for_prompt(found_codes, domain_segments)
    
    prompt = CODE_FUSION_PROMPT.format(
        trajectory_structure=trajectory_structure,
        cross_domain_trajectory=json.dumps(trajectory),
        core_entity=core_entity,
        cross_domain_policy=policy_content,
        single_domain_codes=single_domain_codes,
    )
    
    try:
        response = client.chat(query=prompt, model_type="coding")
        return extract_python_code(response.content)
    except Exception as e:
        logger.warning(f"Cross Domain code fusion failed: {e}")
        return None


def generate_and_execute_cross_domain_selection(
    client,
    trajectory: List[str],
    domain_segments: Dict[str, List[str]],
    found_codes: Dict[str, Tuple[str, Dict[str, Any]]],
    core_entity: str,
    policy_content: str,
    db_dir: Path,
    server_name: str,
    max_retries: int = 3,
    timeout: int = 30,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Generate and execute fused Cross Domain selection code.
    
    Uses code fusion approach:
    1. Load Single Domain codes as reference
    2. LLM fuses codes and adds global constraints
    3. Execute fused code
    4. Apply BlockEditor fixes if needed
    
    Args:
        client: LLM client
        trajectory: Full Cross Domain trajectory
        domain_segments: Parsed domain segments
        found_codes: Single Domain codes found
        core_entity: The core entity that must be consistent across all domains
        policy_content: Cross Domain policy
        db_dir: Database directory
        server_name: Cross Domain server name
        max_retries: Maximum retry attempts
        timeout: Script execution timeout
        
    Returns:
        Tuple of (success, result_dict, generated_code)
    """
    block_editor = WorkflowBlockEditor()
    last_error = None
    final_code = None
    
    # Generate initial fused code
    code = generate_fused_selection_code(
        client,
        trajectory,
        domain_segments,
        found_codes,
        core_entity,
        policy_content,
        db_dir,
        server_name,
    )
    
    if not code:
        return False, {"error": "Failed to generate fused selection code", "attempts": 0}, None
    
    final_code = code
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        f.write(code)
        script_path = Path(f.name)
    
    try:
        for attempt in range(max_retries):
            # Execute code
            success, result, error_msg = execute_selection_script(
                script_path, db_dir, server_name, timeout
            )
            
            if success:
                final_code = script_path.read_text()
                return True, result, final_code
            
            last_error = error_msg
            logger.debug(f"Cross Domain selection attempt {attempt + 1} failed: {last_error}")
            
            if attempt < max_retries - 1:
                # Try to fix with BlockEditor
                fixed = block_editor.fix_code(
                    file_path=script_path,
                    error=error_msg,
                    llm_client=client,
                    language="python",
                    max_retries=2,
                )
                
                if not fixed:
                    logger.warning(f"BlockEditor failed to fix Cross Domain code on attempt {attempt + 1}")
                    # Regenerate fused code
                    new_code = generate_fused_selection_code(
                        client,
                        trajectory,
                        domain_segments,
                        found_codes,
                        core_entity,
                        policy_content + f"\n\n# Previous error: {last_error}",
                        db_dir,
                        server_name,
                    )
                    
                    if new_code:
                        script_path.write_text(new_code)
                        final_code = new_code
        
        return False, {"error": last_error, "attempts": max_retries}, final_code
        
    finally:
        script_path.unlink(missing_ok=True)


# =============================================================================
# Cross Domain Combination Creation (New Approach)
# =============================================================================

import re

def load_entity_and_relationship_names(
    entities_path: Path,
    blueprints_path: Path,
) -> Tuple[set, Dict[str, set]]:
    """
    Load entity names and domain->relationship mappings from entities.json and blueprints.json.
    
    Args:
        entities_path: Path to entities.json
        blueprints_path: Path to blueprints.json
        
    Returns:
        Tuple of:
        - entity_names: Set of all entity names (e.g., {"Student", "Instructor", ...})
        - domain_relationships: Dict mapping domain to set of relationship names
          (e.g., {"FerryBerthNotifier": {"PassengerVesselBooking", ...}})
    """
    # Load entity names
    entity_names = set()
    if entities_path.exists():
        entities_data = load_json(entities_path)
        entity_names = set(entities_data.keys())
    
    # Load domain -> relationship mappings from blueprints
    domain_relationships = {}
    if blueprints_path.exists():
        blueprints = load_json(blueprints_path)
        for blueprint in blueprints:
            domain = blueprint.get("MCP_server_name", "")
            if domain:
                relationships = blueprint.get("relationships", [])
                rel_names = {r.get("name", "") for r in relationships if r.get("name")}
                if rel_names:
                    domain_relationships[domain] = rel_names
    
    logger.debug(f"Loaded {len(entity_names)} entities and {len(domain_relationships)} domains with relationships")
    return entity_names, domain_relationships


def parse_explicit_placeholders(
    instruction: str,
    entity_names: set,
    domain_relationships: Dict[str, set],
) -> Dict[str, Any]:
    """
    Parse explicit placeholders from task template instruction.
    
    Handles formats:
    - <Entity.field> - e.g., <Passenger.passenger_id>
    - <Relationship.field> - e.g., <PassengerVesselBooking.route_origin>
    - <Domain.Relationship.field> - e.g., <FerryBerthNotifier.PassengerVesselBooking.booking_id>
    
    Args:
        instruction: Task template instruction string
        entity_names: Set of known entity names
        domain_relationships: Dict mapping domain to set of relationship names
        
    Returns:
        {
            "entities": {entity_name: [field1, field2, ...]},
            "relationships": {domain: {rel_name: [field1, field2, ...]}}
        }
    """
    entities = {}  # {EntityName: [field1, field2, ...]}
    relationships = {}  # {Domain: {RelName: [field1, field2, ...]}}
    
    # Build reverse lookup: relationship_name -> domain
    rel_to_domain = {}
    for domain, rels in domain_relationships.items():
        for rel in rels:
            rel_to_domain[rel] = domain
    
    # Match all placeholders: <...>
    placeholder_pattern = r'<([^>]+)>'
    matches = re.findall(placeholder_pattern, instruction)
    
    for match in matches:
        parts = match.split('.')
        
        if len(parts) == 2:
            # Format: <Foo.bar>
            name, field = parts
            
            if name in entity_names:
                # It's an Entity
                if name not in entities:
                    entities[name] = []
                if field not in entities[name]:
                    entities[name].append(field)
            elif name in rel_to_domain:
                # It's a Relationship
                domain = rel_to_domain[name]
                if domain not in relationships:
                    relationships[domain] = {}
                if name not in relationships[domain]:
                    relationships[domain][name] = []
                if field not in relationships[domain][name]:
                    relationships[domain][name].append(field)
            else:
                # Unknown - try to guess based on naming convention
                # CamelCase names ending in common suffixes are likely entities
                logger.debug(f"Unknown placeholder type: {match}")
                
        elif len(parts) == 3:
            # Format: <Domain.Relationship.field> or <Domain.field>
            domain_or_name, rel_or_field, field = parts
            
            if domain_or_name in domain_relationships:
                # It's Domain.Relationship.field
                domain = domain_or_name
                rel_name = rel_or_field
                if domain not in relationships:
                    relationships[domain] = {}
                if rel_name not in relationships[domain]:
                    relationships[domain][rel_name] = []
                if field not in relationships[domain][rel_name]:
                    relationships[domain][rel_name].append(field)
            else:
                # Could be a different format - log and skip
                logger.debug(f"Unrecognized 3-part placeholder: {match}")
    
    logger.debug(f"Parsed {len(entities)} entities, {sum(len(rels) for rels in relationships.values())} relationships")
    return {
        "entities": entities,
        "relationships": relationships,
    }


def load_required_summaries_for_creation(
    entities: Dict[str, List[str]],
    relationships: Dict[str, Dict[str, List[str]]],
    summary_dir: Path,
) -> Tuple[str, str]:
    """
    Load database_summary markdown files for required entities and relationships.
    
    Args:
        entities: {entity_name: [field1, field2, ...]}
        relationships: {domain: {rel_name: [field1, field2, ...]}}
        summary_dir: Path to database_summary directory
        
    Returns:
        (entity_summaries_markdown, relationship_summaries_markdown)
    """
    entity_summaries = []
    relationship_summaries = []
    
    # Load entity summaries
    entity_summary_dir = summary_dir / "entities"
    for entity_name in entities.keys():
        summary_path = entity_summary_dir / f"{entity_name}.md"
        if summary_path.exists():
            content = summary_path.read_text()
            entity_summaries.append(f"## {entity_name}\n\n{content}")
        else:
            logger.debug(f"Entity summary not found: {summary_path}")
    
    # Load relationship summaries
    rel_summary_dir = summary_dir / "relationships"
    for domain, rels in relationships.items():
        for rel_name in rels.keys():
            # Try domain-specific path first
            summary_path = rel_summary_dir / domain / f"{rel_name}.md"
            if summary_path.exists():
                content = summary_path.read_text()
                relationship_summaries.append(f"## {domain}.{rel_name}\n\n{content}")
            else:
                logger.debug(f"Relationship summary not found: {summary_path}")
    
    return "\n\n---\n\n".join(entity_summaries), "\n\n---\n\n".join(relationship_summaries)


def load_single_domain_sampling_codes_for_creation(
    outputs_dir: Path,
    domain_segments: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Load Single Domain sampling codes for reference during Creation.
    
    These codes contain the core constraints that must be satisfied.
    Creation Code should generate instances that would PASS these constraints.
    
    Args:
        outputs_dir: Base outputs directory
        domain_segments: {domain: [tool1, tool2, ...]}
        
    Returns:
        {domain: sampling_code_content}
    """
    codes = {}
    for domain, tools in domain_segments.items():
        traj_hash = compute_trajectory_hash(tools)
        code_path = outputs_dir / "sampling_codes" / domain / f"{traj_hash}.py"
        if code_path.exists():
            codes[domain] = code_path.read_text()
            logger.debug(f"Loaded sampling code for {domain}/{traj_hash}")
        else:
            logger.debug(f"No sampling code found for {domain}/{traj_hash}")
    return codes


def analyze_implicit_dependencies(
    client,
    task_template: Dict[str, str],
    trajectory: List[str],
    explicit_dependencies: Dict[str, Any],
    domain_relationships: Dict[str, set],
) -> Dict[str, Any]:
    """
    Use LLM to analyze implicit dependencies in a task template.
    
    Args:
        client: LLM client
        task_template: Task template with instruction
        trajectory: List of tool calls
        explicit_dependencies: Already identified explicit dependencies
        domain_relationships: {domain: {relationship_names}}
        
    Returns:
        {
            "implicit_entities": [...],
            "implicit_relationships": [...]
        }
    """
    from ..prompts import IMPLICIT_DEPENDENCY_ANALYSIS_PROMPT
    
    # Format available relationships for the prompt
    available_rels = []
    for domain, rels in domain_relationships.items():
        available_rels.append(f"- {domain}: {', '.join(sorted(rels))}")
    
    prompt = IMPLICIT_DEPENDENCY_ANALYSIS_PROMPT.format(
        task_template_instruction=task_template.get("instruction", ""),
        trajectory=json.dumps(trajectory),
        explicit_dependencies_json=json.dumps(explicit_dependencies, indent=2),
        available_relationships="\n".join(available_rels),
    )
    
    try:
        from ..core.llm_client import extract_json
        
        response = client.chat(query=prompt, model_type="textual")
        content = response.content
        
        result = extract_json(content)
        logger.debug(f"Found {len(result.get('implicit_entities', []))} implicit entities, "
                    f"{len(result.get('implicit_relationships', []))} implicit relationships")
        return result
        
    except Exception as e:
        logger.warning(f"Failed to analyze implicit dependencies: {e}")
        return {"implicit_entities": [], "implicit_relationships": []}


def merge_dependencies(
    explicit: Dict[str, Any],
    implicit: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge explicit and implicit dependencies into a unified structure.
    
    Args:
        explicit: {entities: {...}, relationships: {...}}
        implicit: {implicit_entities: [...], implicit_relationships: [...]}
        
    Returns:
        Combined dependency structure with unified entities and relationships
    """
    merged_entities = dict(explicit.get("entities", {}))
    merged_relationships = {k: dict(v) for k, v in explicit.get("relationships", {}).items()}
    
    # Add implicit entities
    for ie in implicit.get("implicit_entities", []):
        name = ie.get("name")
        if name and name not in merged_entities:
            merged_entities[name] = ie.get("fields_needed", [])
    
    # Add implicit relationships
    for ir in implicit.get("implicit_relationships", []):
        domain = ir.get("domain")
        name = ir.get("name")
        if domain and name:
            if domain not in merged_relationships:
                merged_relationships[domain] = {}
            if name not in merged_relationships[domain]:
                merged_relationships[domain][name] = list(ir.get("foreign_keys", {}).keys())
    
    return {"entities": merged_entities, "relationships": merged_relationships}


def generate_combination_creation_code(
    client,
    task_template: Dict[str, str],
    dependency_analysis: Dict[str, Any],
    core_entity: str,
    single_domain_codes: Dict[str, str],
    entity_summaries: str,
    relationship_summaries: str,
    filtered_policy: str = "",
    trajectory: List[str] = None,
    outputs_dir: Path = None,
) -> Optional[str]:
    """
    Generate Python code to create entity and relationship instances.
    
    Args:
        client: LLM client
        task_template: Task template with instruction
        dependency_analysis: Merged explicit and implicit dependencies
        core_entity: Core entity name (e.g., "Instructor")
        single_domain_codes: {domain: sampling_code}
        entity_summaries: Markdown summaries of entities
        relationship_summaries: Markdown summaries of relationships
        filtered_policy: Policy information for business rules
        trajectory: List of tool calls (for parameter extraction)
        outputs_dir: Base outputs directory (for loading tool_lists)
        
    Returns:
        Python code string or None if failed
    """
    from ..prompts import COMBINATION_CREATION_PROMPT
    
    # Format single domain codes
    codes_section = []
    for domain, code in single_domain_codes.items():
        codes_section.append(f"### {domain}\n\n```python\n{code}\n```")
    
    # Warn if reference materials are missing
    if not single_domain_codes:
        logger.warning("No Single Domain sampling codes found for reference!")
    if not entity_summaries:
        logger.warning("No entity summaries found for creation guidance!")
    if not relationship_summaries:
        logger.warning("No relationship summaries found for creation guidance!")
    if filtered_policy == "":
        logger.warning("No filtered policy information available!")

    # Build trajectory tool names list for the prompt
    trajectory_tool_names = ""
    if trajectory and outputs_dir:
        tool_params = get_trajectory_tool_params(trajectory, outputs_dir)
        tool_lines = []
        for tool_call, params in tool_params.items():
            params_str = ", ".join(params) if params else "(no parameters)"
            tool_lines.append(f"- {tool_call}: {params_str}")
        trajectory_tool_names = "\n".join(tool_lines)
    elif trajectory:
        # Fallback: just list tool names without params
        trajectory_tool_names = "\n".join([f"- {t}" for t in trajectory])
    
    prompt = COMBINATION_CREATION_PROMPT.format(
        task_template_instruction=task_template.get("instruction", ""),
        dependency_analysis_json=json.dumps(dependency_analysis, indent=2),
        core_entity=core_entity,
        single_domain_codes="\n\n---\n\n".join(codes_section) if codes_section else "(No Single Domain codes available)",
        filtered_policy=filtered_policy or "(No policy information available)",
        entity_summaries=entity_summaries or "(No entity summaries available)",
        relationship_summaries=relationship_summaries or "(No relationship summaries available)",
        trajectory_tool_names=trajectory_tool_names or "(trajectory not provided)",
    )
    
    try:
        response = client.chat(query=prompt, model_type="coding")
        return extract_python_code(response.content)
    except Exception as e:
        logger.warning(f"Failed to generate creation code: {e}")
        return None


def validate_with_trajectory_for_creation(
    combo: Dict[str, Any],
    trajectory: List[str],
    server_name: str,
    outputs_dir: Path,
) -> ValidationResult:
    """
    Validate a single combination by executing its trajectory.
    
    Args:
        combo: The combination to validate
        trajectory: List of tool calls
        server_name: Cross Domain server name
        outputs_dir: Base outputs directory
        
    Returns:
        ValidationResult with success status and code caching info
    """
    from evaluator import TrajectoryExecutor, TaskDefinition
    
    try:
        # Build TaskDefinition from combo
        task_def = TaskDefinition.from_combo(
            trajectory=trajectory,
            combo=combo,
            server_name=server_name,
        )
        
        # Create executor and run
        executor = TrajectoryExecutor(outputs_dir=str(outputs_dir))
        result = executor.execute(task=task_def, verbose=True)
        
        # Check if all steps succeeded
        all_success = result.success and all(s.success for s in result.steps)
        
        if all_success:
            # Return success with code caching information
            return ValidationResult(
                success=True,
                error_message=None,
                execution_log=None,
                # Code caching fields for TrajectoryCodeManager
                step_param_codes=result.step_param_codes,
                generated_code=result.generated_code,
                entity_context=dict(task_def.entity_context),
                domains=list(task_def.domains),
                is_cross_domain=task_def.is_cross_domain,
            )
        else:
            error_msg = result.errors[0] if result.errors else "Unknown error"
            # Collect detailed execution log
            exec_log = []
            for step in result.steps:
                status = "✓" if step.success else "✗"
                tc = step.tool_call
                if tc:
                    log_entry = f"{status} Step {step.step_index}: {tc.name}\n"
                    # Arguments (truncate if too long)
                    args_str = json.dumps(tc.arguments, indent=2, default=str)
                    if len(args_str) > 500:
                        args_str = args_str[:500] + "..."
                    log_entry += f"   Arguments: {args_str}\n"
                    # Result (truncate if too long)
                    if tc.result is not None:
                        result_str = json.dumps(tc.result, indent=2, default=str)
                        if len(result_str) > 500:
                            result_str = result_str[:500] + "..."
                        log_entry += f"   Result: {result_str}\n"
                else:
                    log_entry = f"{status} Step {step.step_index}: Unknown\n"
                # Error if present
                if step.error:
                    log_entry += f"   Error: {step.error}\n"
                exec_log.append(log_entry)
            return ValidationResult(
                success=False,
                error_message=error_msg,
                execution_log="\n".join(exec_log),
            )
            
    except Exception as e:
        logger.error(f"Trajectory validation exception: {e}")
        return ValidationResult(
            success=False,
            error_message=str(e),
            execution_log=None,
        )


def analyze_trajectory_error(
    client,
    error_message: str,
    exec_log: str,
    failed_combo: Dict[str, Any],
    filtered_policy: str,
    trajectory: List[str],
) -> Dict[str, str]:
    """
    Analyze trajectory execution error and identify root cause.
    
    Uses LLM to categorize the error and provide targeted fix suggestions.
    
    Args:
        client: LLM client
        error_message: The error message from trajectory execution
        exec_log: Execution log showing step-by-step results
        failed_combo: The combo that failed validation
        filtered_policy: Policy rules for the trajectory
        trajectory: The fixed trajectory (list of tool calls)
        
    Returns:
        Dict with error_category, root_cause, policy_reference, fix_suggestion
    """
    from ..prompts import ERROR_ANALYSIS_PROMPT
    from ..core.llm_client import extract_json
    
    prompt = ERROR_ANALYSIS_PROMPT.format(
        trajectory=json.dumps(trajectory, indent=2),
        error_message=error_message,
        exec_log=exec_log or "(No execution log)",
        failed_combo=json.dumps(failed_combo, indent=2),
        filtered_policy=filtered_policy or "(No policy available)",
    )
    
    try:
        response = client.chat(query=prompt, model_type="textual")
        result = extract_json(response.content)
        if result:
            # print(result)
            return result
    except Exception as e:
        logger.warning(f"Error analysis failed: {e}")
    
    # Fallback if analysis fails
    return {
        "error_category": "OTHER",
        "root_cause": error_message,
        "policy_rule_quoted": "",
        "code_fixes": []
    }


def validate_and_fix_creation_code(
    client,
    creation_code: str,
    trajectory: List[str],
    server_name: str,
    outputs_dir: Path,
    db_dir: Path,
    entity_names: set,
    known_relationships: set,
    filtered_policy: str = "",
    max_retries: int = 3,
    timeout: int = 60,
    tool_params: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, List[Dict], str, Optional[ValidationResult]]:
    """
    Execute creation code, validate ONE combo, fix code if needed.
    
    Uses BlockEditor to fix the creation code when validation fails.
    
    Flow for each attempt:
    1. Execute Creation Code to get combos
    2. Validate parameter completeness (if tool_params provided)
    3. Backup Cross Domain database
    4. Write combo instances to database
    5. Validate first combo via trajectory execution
    6. Success -> cleanup backup (persist data)
       Failure -> restore database from backup
    
    Args:
        client: LLM client
        creation_code: Python code to generate combinations
        trajectory: List of tool calls
        server_name: Cross Domain server name
        outputs_dir: Base outputs directory
        db_dir: Database directory
        entity_names: Set of known entity names
        known_relationships: Set of known relationship names in format "{Domain}_{RelationshipName}"
        filtered_policy: Policy information for business rules (used in error fixing)
        max_retries: Maximum retry attempts
        timeout: Script execution timeout
        tool_params: Dict mapping tool calls to their parameter names (for completeness validation)
        
    Returns:
        Tuple of (success, all_combos, final_code, validation_result)
        validation_result contains code caching info on success
    """
    block_editor = WorkflowBlockEditor()
    backup_base_dir = outputs_dir / "database" / "backups"
    
    # Create temporary file for the script
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        f.write(creation_code)
        script_path = Path(f.name)
    
    try:
        for attempt in range(max_retries):
            logger.debug(f"Creation code attempt {attempt + 1}/{max_retries}")
            
            # Execute creation code
            try:
                result = subprocess.run(
                    ["python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr or f"Script exited with code {result.returncode}"
                    logger.debug(f"Creation code execution failed: {error_msg}")
                    
                    if attempt < max_retries - 1:
                        block_editor.fix_code(script_path, error_msg, client, language="python")
                        creation_code = script_path.read_text()
                    continue
                
                # Parse output
                try:
                    output = json.loads(result.stdout)
                    combos = output.get("all_valid_combos", [])
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON output: {e}")
                    if attempt < max_retries - 1:
                        block_editor.fix_code(script_path, f"Invalid JSON output: {e}\nStdout: {result.stdout[:500]}", client, language="python")
                        creation_code = script_path.read_text()
                    continue
                
                if not combos:
                    logger.debug("No combinations generated")
                    if attempt < max_retries - 1:
                        block_editor.fix_code(script_path, "No combinations generated. Ensure generate_combinations() returns at least one combo.", client, language="python")
                        creation_code = script_path.read_text()
                    continue
                
                # ========== Step 2: Validate parameter completeness (before DB operations) ==========
                if tool_params:
                    first_combo = combos[0]

                    # pprint(first_combo)

                    is_complete, missing_params = validate_value_domain_completeness(
                        first_combo, trajectory, tool_params
                    )
                    if not is_complete:
                        logger.warning(f"Incomplete value_domain_samples. Missing {len(missing_params)} params: {missing_params[:5]}...")
                        if attempt < max_retries - 1:
                            # Build error message for fixing
                            error_msg = (
                                f"INCOMPLETE_PARAMS: value_domain_samples is missing {len(missing_params)} required parameters.\n\n"
                                f"Missing parameters (format: Domain.tool.param):\n"
                                f"{chr(10).join(f'  - {p}' for p in missing_params)}\n\n"
                                f"Every tool parameter MUST have an entry in value_domain_samples with one of:\n"
                                f"1. A concrete value (string, number, boolean)\n"
                                f"2. The literal string '<From previous tool call output>' for dynamic params\n"
                                f"3. null for optional params that should use defaults\n\n"
                                f"For time-related params indicating `now/today` (e.g., as_of_date, on_date, event_time), use '2025-01-23T15:00:00-05:00 (EST)' as the current time."
                            )
                            block_editor.fix_code(script_path, error_msg, client, language="python")
                            creation_code = script_path.read_text()
                        continue
                
                # ========== Backup -> Write -> Validate -> Restore/Cleanup ==========
                
                # Step 1: Backup database before writing new instances
                backup_cross_domain_database(db_dir, server_name, backup_base_dir)
                
                try:
                    # Step 2: Write all combo instances to database
                    written_count = write_instances_to_database(
                        combos=combos,
                        server_name=server_name,
                        db_dir=db_dir,
                        entity_names=entity_names,
                        known_relationships=known_relationships,
                    )
                    logger.info(f"Wrote {written_count} instances to database for validation")
                    
                    # Step 3: Validate FIRST combo only (Creation = 100% success rate target)
                    first_combo = combos[0]
                    validation_result = validate_with_trajectory_for_creation(
                        first_combo, trajectory, server_name, outputs_dir
                    )
                    
                    if validation_result.success:
                        # Step 4a: Success - cleanup backup (persist new data)
                        cleanup_database_backup(backup_base_dir, server_name)
                        logger.info(f"Creation code validated successfully, {len(combos)} combos generated")
                        return True, combos, script_path.read_text(), validation_result
                    
                    # Step 4b: Validation failed - restore database
                    restore_cross_domain_database(db_dir, server_name, backup_base_dir)
                    cleanup_database_backup(backup_base_dir, server_name)
                    
                    # Extract error info from validation result
                    exec_error = validation_result.error_message
                    exec_log = validation_result.execution_log
                    
                    # Trajectory validation failed - analyze error first, then fix
                    logger.warning(f"Trajectory validation failed (attempt {attempt + 1}/{max_retries}): {exec_error}")
                    if attempt < max_retries - 1:
                        # Step 1: Analyze the error with LLM
                        error_analysis = analyze_trajectory_error(
                            client, exec_error, exec_log, first_combo, filtered_policy, trajectory
                        )
                        logger.info(f"Error analysis: [{error_analysis.get('error_category')}] {error_analysis.get('root_cause', '')[:100]}...")

                        # pprint(error_analysis)
                        
                        # Step 2: Build fix context with analysis results
                        fix_context = "## Error Analysis (Use this to guide your fix)\n\n"
                        fix_context += f"**Category**: {error_analysis.get('error_category')}\n\n"
                        fix_context += f"**Root Cause**: {error_analysis.get('root_cause')}\n\n"
                        
                        # Include policy rule if quoted
                        if error_analysis.get('policy_rule_quoted'):
                            fix_context += f"**Policy Rule Violated** (from policy document):\n```\n{error_analysis.get('policy_rule_quoted')}\n```\n\n"
                        
                        # Include code fixes with specific changes
                        code_fixes = error_analysis.get('code_fixes', [])
                        if code_fixes:
                            fix_context += "## Required Code Fixes\n\n"
                            fix_context += "**IMPORTANT**: Use VARIABLES based on policy rules, NOT hardcoded values!\n\n"
                            for i, fix in enumerate(code_fixes, 1):
                                fix_context += f"### Fix {i}: `{fix.get('field', 'unknown')}`\n"
                                fix_context += f"- **Current code**: `{fix.get('current_code', '')}`\n"
                                fix_context += f"- **Fixed code**: `{fix.get('fixed_code', '')}`\n"
                                fix_context += f"- **Policy basis**: {fix.get('policy_basis', '')}\n\n"
                        
                        fix_context += "---\n\n"
                        fix_context += f"## Original Error\n{exec_error}\n\n"
                        if exec_log:
                            fix_context += f"## Execution Log\n{exec_log}\n\n"
                        fix_context += f"## Failed Combo\n{json.dumps(first_combo, indent=2)}"
                        
                        # Step 3: Fix with analysis-enriched context
                        block_editor.fix_code(script_path, fix_context, client, language="python")
                        creation_code = script_path.read_text()
                        logger.info(f"Creation code fixed by BlockEditor, retrying...")
                    
                except Exception as e:
                    # On any exception, restore database
                    restore_cross_domain_database(db_dir, server_name, backup_base_dir)
                    cleanup_database_backup(backup_base_dir, server_name)
                    raise
                
                continue
                    
            except subprocess.TimeoutExpired:
                logger.debug(f"Creation code timed out after {timeout}s")
                if attempt < max_retries - 1:
                    block_editor.fix_code(script_path, f"Script execution timed out after {timeout}s. Simplify the generation logic.", client, language="python")
                    creation_code = script_path.read_text()
        
        # All retries failed
        return False, [], script_path.read_text(), None
        
    finally:
        script_path.unlink(missing_ok=True)


# =============================================================================
# Cross Domain Database Backup/Restore
# =============================================================================

import shutil

def backup_cross_domain_database(
    db_dir: Path,
    server_name: str,
    backup_base_dir: Path,
) -> Path:
    """
    Backup Cross Domain database directories before validation.
    
    Backs up:
    - db_dir/entities/{server_name}/
    - db_dir/relationships/{server_name}/
    
    Args:
        db_dir: Database directory (outputs/database/outputs)
        server_name: Cross Domain server name
        backup_base_dir: Base directory for backups
        
    Returns:
        Path to the backup directory
    """
    backup_dir = backup_base_dir / server_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup entities
    entities_src = db_dir / "entities" / server_name
    if entities_src.exists():
        entities_dst = backup_dir / "entities" / server_name
        if entities_dst.exists():
            shutil.rmtree(entities_dst)
        shutil.copytree(entities_src, entities_dst)
        logger.debug(f"Backed up entities from {entities_src}")
    
    # Backup relationships
    relationships_src = db_dir / "relationships" / server_name
    if relationships_src.exists():
        relationships_dst = backup_dir / "relationships" / server_name
        if relationships_dst.exists():
            shutil.rmtree(relationships_dst)
        shutil.copytree(relationships_src, relationships_dst)
        logger.debug(f"Backed up relationships from {relationships_src}")
    
    logger.info(f"Database backup created at {backup_dir}")
    return backup_dir


def restore_cross_domain_database(
    db_dir: Path,
    server_name: str,
    backup_base_dir: Path,
) -> None:
    """
    Restore Cross Domain database from backup.
    
    Restores:
    - db_dir/entities/{server_name}/ from backup
    - db_dir/relationships/{server_name}/ from backup
    
    Args:
        db_dir: Database directory (outputs/database/outputs)
        server_name: Cross Domain server name
        backup_base_dir: Base directory for backups
    """
    backup_dir = backup_base_dir / server_name
    
    if not backup_dir.exists():
        logger.warning(f"Backup directory not found: {backup_dir}")
        return
    
    # Restore entities
    entities_backup = backup_dir / "entities" / server_name
    if entities_backup.exists():
        entities_dst = db_dir / "entities" / server_name
        if entities_dst.exists():
            shutil.rmtree(entities_dst)
        shutil.copytree(entities_backup, entities_dst)
        logger.debug(f"Restored entities to {entities_dst}")
    
    # Restore relationships
    relationships_backup = backup_dir / "relationships" / server_name
    if relationships_backup.exists():
        relationships_dst = db_dir / "relationships" / server_name
        if relationships_dst.exists():
            shutil.rmtree(relationships_dst)
        shutil.copytree(relationships_backup, relationships_dst)
        logger.debug(f"Restored relationships to {relationships_dst}")
    
    logger.info(f"Database restored from {backup_dir}")


def cleanup_database_backup(
    backup_base_dir: Path,
    server_name: str,
) -> None:
    """
    Delete database backup after successful validation.
    
    Args:
        backup_base_dir: Base directory for backups
        server_name: Cross Domain server name
    """
    backup_dir = backup_base_dir / server_name
    
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
        logger.debug(f"Cleaned up backup at {backup_dir}")


def write_instances_to_database(
    combos: List[Dict[str, Any]],
    server_name: str,
    db_dir: Path,
    entity_names: set,
    known_relationships: set,
) -> int:
    """
    Write new instances from validated combinations to database.
    
    Handles both Entity and Relationship instances.
    Entity instances are written to entities/{server_name}/.
    Relationship instances are written to relationships/{server_name}/.
    
    Note: UUID uniqueness is guaranteed, so no deduplication check is performed.
    
    Args:
        combos: List of validated combinations
        server_name: Cross Domain server name
        db_dir: Database directory (outputs/database/outputs)
        entity_names: Set of known entity names (to distinguish from relationships)
        known_relationships: Set of known relationship names in format "{Domain}_{RelationshipName}"
        
    Returns:
        Number of new instances written
    """
    entities_dir = db_dir / "entities" / server_name
    relationships_dir = db_dir / "relationships" / server_name
    entities_dir.mkdir(parents=True, exist_ok=True)
    relationships_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate instances by type
    entity_instances = {}  # {name: [instances]}
    relationship_instances = {}  # {name: [instances]}
    
    for combo in combos:
        # Process entity_instances field (contains BOTH entities AND relationships)
        for name, instance in combo.get("entity_instances", {}).items():
            if name in entity_names:
                # It's an Entity (exact match)
                entity_instances.setdefault(name, []).append(instance)
            else:
                # Try to match Entity with suffix removed (e.g., "Farm_creation" -> "Farm")
                base_name = name.rsplit("_", 1)[0] if "_" in name else name
                if base_name in entity_names:
                    # It's an Entity with suffix
                    entity_instances.setdefault(base_name, []).append(instance)
                else:
                    # It's a Relationship - extract pure relationship name
                    # "AgFieldLabPlanner.InstructorFarmAccess_creation" -> "InstructorFarmAccess"
                    if "." in name:
                        # Remove domain prefix: "AgFieldLabPlanner.InstructorFarmAccess_creation" -> "InstructorFarmAccess_creation"
                        rel_name = name.split(".", 1)[1]
                    else:
                        rel_name = name
                    
                    # If rel_name not in known relationships, try to remove suffix
                    if rel_name not in known_relationships:
                        # Try removing the last part after _
                        parts = rel_name.rsplit("_", 1)
                        if len(parts) == 2 and parts[0] in known_relationships:
                            rel_name = parts[0]
                    
                    relationship_instances.setdefault(rel_name, []).append(instance)
    
    written_count = 0
    
    # Write entities
    for name, instances in entity_instances.items():
        file_path = entities_dir / f"{name}.json"
        existing = load_json(file_path) if file_path.exists() else []
        existing.extend(instances)
        save_json(existing, file_path)
        written_count += len(instances)
        logger.debug(f"Wrote {len(instances)} new {name} entity instances to database")
    
    # Write relationships
    for name, instances in relationship_instances.items():
        file_path = relationships_dir / f"{name}.json"
        existing = load_json(file_path) if file_path.exists() else []
        existing.extend(instances)
        save_json(existing, file_path)
        written_count += len(instances)
        logger.debug(f"Wrote {len(instances)} new {name} relationship instances to database")
    
    return written_count


def write_validated_combos(
    combos: List[Dict[str, Any]],
    trajectory: List[str],
    trajectory_hash: str,
    task_template: Dict[str, str],
    server_name: str,
    validated_tasks_dir: Path,
) -> Path:
    """
    Write validated combinations to validated_tasks/ directory.
    
    Format matches Step 16 output for downstream compatibility.
    
    Args:
        combos: List of validated combinations
        trajectory: List of tool calls
        trajectory_hash: Hash of the trajectory
        task_template: Task template dict
        server_name: Cross Domain server name
        validated_tasks_dir: Path to validated_tasks directory
        
    Returns:
        Path to the output file
    """
    output_dir = ensure_dir(validated_tasks_dir / server_name)
    
    validated_entries = []
    for idx, combo in enumerate(combos):
        entry = {
            "sample_idx": idx,
            "combo_id": compute_combo_id(trajectory_hash, combo.get("entity_instances", {})),
            "entity_instances": combo.get("entity_instances", {}),
            "value_domain_samples": combo.get("value_domain_samples", {}),
            "validated": True,
            "task_template": task_template,
            "trajectory": trajectory,
            "trajectory_hash": trajectory_hash,
            "validation_timestamp": datetime.now().isoformat(),
        }
        validated_entries.append(entry)
    
    output_path = output_dir / "validated_combos.json"
    
    # Append to existing or create new
    if output_path.exists():
        existing = load_json(output_path)
    else:
        existing = []
    
    # Deduplicate by combo_id
    existing_ids = {e.get("combo_id") for e in existing}
    new_entries = [e for e in validated_entries if e.get("combo_id") not in existing_ids]
    
    existing.extend(new_entries)
    save_json(existing, output_path)
    
    logger.info(f"Wrote {len(new_entries)} validated combos to {output_path}")
    return output_path


def process_cross_domain_with_creation(
    trajectory_task: Dict[str, Any],
    client,
    settings: Dict[str, Any],
    outputs_dir: Path,
) -> Optional[int]:
    """
    Process Cross Domain trajectory using Combination Creation mode.
    
    Steps:
    1. Parse explicit placeholders (deterministic)
    2. LLM analyze implicit dependencies
    3. Load database_summary and Single Domain Sampling Codes
    4. LLM generate Creation Code
    5. Execute and validate (single combo)
    6. Write to Database and validated_tasks/
    
    Args:
        trajectory_task: Dict containing trajectory info
        client: LLM client
        settings: Step settings
        outputs_dir: Base outputs directory
        
    Returns:
        Number of validated combos, or None if failed
    """
    template = trajectory_task["template"]
    template_idx = trajectory_task["template_idx"]
    server_name = trajectory_task["server_name"]
    policy_content = trajectory_task.get("policy_content", "")
    db_dir = trajectory_task["db_dir"]
    summary_dir = trajectory_task.get("summary_dir", outputs_dir / "database_summary")
    
    trajectory = template.get("trajectory", [])
    task_template = template.get("task_template", {})
    
    if not trajectory:
        logger.warning(f"No trajectory in template {template_idx}")
        return None
    
    traj_hash = compute_trajectory_hash(trajectory)
    logger.info(f"Processing Cross Domain trajectory {traj_hash[:8]} with Creation mode")
    
    max_code_retries = settings.get("max_code_retries", 3)
    code_timeout = settings.get("code_timeout", 60)
    samples_per_template = settings.get("samples_per_template", 1)
    
    # Filter policy for this trajectory
    # For Cross Domain, trajectory has format "Domain.tool_name", but policy uses just "tool_name"
    # Extract pure tool names for policy filtering
    tool_names_for_policy = [t.split(".")[-1] if "." in t else t for t in trajectory]
    if policy_content and has_structured_markers(policy_content):
        filtered_policy = filter_policy_for_trajectory(policy_content, tool_names_for_policy)
    else:
        filtered_policy = policy_content
    
    # ========== Step 1: Load entity and relationship names ==========
    entities_path = outputs_dir / "entities.json"
    blueprints_path = outputs_dir / "blueprints.json"
    
    entity_names, domain_relationships = load_entity_and_relationship_names(
        entities_path, blueprints_path
    )
    
    # Build known relationship names set for database write (pure names, no domain prefix)
    known_relationships = set()
    for domain, rels in domain_relationships.items():
        for rel_name in rels:
            known_relationships.add(rel_name)
    
    # ========== Step 2: Parse explicit placeholders ==========
    instruction = task_template.get("instruction", "")
    explicit_deps = parse_explicit_placeholders(
        instruction, entity_names, domain_relationships
    )
    
    logger.debug(f"Explicit deps: {len(explicit_deps.get('entities', {}))} entities, "
                f"{sum(len(rels) for rels in explicit_deps.get('relationships', {}).values())} relationships")
    
    # ========== Step 3: Analyze implicit dependencies ==========
    implicit_deps = analyze_implicit_dependencies(
        client, task_template, trajectory, explicit_deps, domain_relationships
    )
    
    # Merge dependencies
    full_deps = merge_dependencies(explicit_deps, implicit_deps)

    # pprint(full_deps)
    
    # ========== Step 4: Get core entity ==========
    cross_domain_combos_dir = outputs_dir / "cross_domain_combinations"
    combo_file = cross_domain_combos_dir / f"{server_name}.json"
    
    core_entity = ""
    if combo_file.exists():
        try:
            combo_data = load_json(combo_file)
            core_entity = get_core_entity_from_combination(combo_data)
        except Exception as e:
            logger.warning(f"Failed to load core entity for {server_name}: {e}")
    
    if not core_entity:
        logger.warning(f"No core entity found for {server_name}")
        return None
    
    # ========== Step 5: Load reference materials ==========
    # Load database summaries
    entity_summaries, relationship_summaries = load_required_summaries_for_creation(
        full_deps.get("entities", {}),
        full_deps.get("relationships", {}),
        summary_dir,
    )
    
    # Parse domain segments and load Single Domain sampling codes
    domain_segments = parse_cross_domain_trajectory(trajectory)
    single_domain_codes = load_single_domain_sampling_codes_for_creation(
        outputs_dir, domain_segments
    )
    
    logger.info(f"Loaded {len(single_domain_codes)} Single Domain sampling codes for reference")
    
    # ========== Step 6: Generate Creation Code ==========
    creation_code = generate_combination_creation_code(
        client,
        task_template,
        full_deps,
        core_entity,
        single_domain_codes,
        entity_summaries,
        relationship_summaries,
        filtered_policy=filtered_policy,
        trajectory=trajectory,
        outputs_dir=outputs_dir,
    )

    # Save creation code
    save_sampling_code(
        outputs_dir=outputs_dir,
        server_name=f"_creation_{server_name}",
        trajectory_hash=traj_hash,
        code=creation_code,
        constraint_analysis={
            "full_deps": full_deps,
            "core_entity": core_entity,
        },
    )
    
    if not creation_code:
        logger.warning(f"Failed to generate creation code for {server_name}")
        return None
    
    # ========== Step 6.5: Get tool parameters for completeness validation ==========
    tool_params = get_trajectory_tool_params(trajectory, outputs_dir)
    logger.debug(f"Extracted parameters for {len(tool_params)} tools in trajectory")
    
    # ========== Step 7: Validate and fix ==========
    success, combos, final_code, validation_result = validate_and_fix_creation_code(
        client,
        creation_code,
        trajectory,
        server_name,
        outputs_dir,
        db_dir,
        entity_names=entity_names,
        known_relationships=known_relationships,
        filtered_policy=filtered_policy,
        max_retries=max_code_retries,
        timeout=code_timeout,
        tool_params=tool_params,
    )
    
    if not success:
        logger.warning(f"Creation code validation failed for {server_name}")
        # Save failed code for debugging
        if final_code:
            try:
                save_sampling_code(
                    outputs_dir=outputs_dir,
                    server_name=f"{server_name}_creation_failed",
                    trajectory_hash=traj_hash,
                    code=final_code,
                    constraint_analysis={
                        "error": "Creation code validation failed",
                        "full_deps": full_deps,
                        "core_entity": core_entity,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to save failed creation code: {e}")
        return None
    
    # ========== Step 8: Save Trajectory Execution Code to trajectory_code/ ==========
    # Use TrajectoryCodeManager to cache the generated param codes for reuse
    if validation_result and validation_result.step_param_codes:
        from evaluator import TrajectoryCodeManager
        
        try:
            code_manager = TrajectoryCodeManager(str(outputs_dir))
            entity_mappings = {
                k.split(".")[-1]: k 
                for k in validation_result.entity_context.keys()
            } if validation_result.entity_context else {}
            
            code_manager.save_code(
                trajectory=trajectory,
                domains=validation_result.domains or [server_name],
                is_cross_domain=validation_result.is_cross_domain,
                generated_code=validation_result.generated_code or "",
                entity_mappings=entity_mappings,
                step_param_codes=validation_result.step_param_codes,
            )
            logger.info(f"Saved trajectory execution code for {server_name}/{traj_hash[:8]}")
        except Exception as e:
            logger.warning(f"Failed to save trajectory code: {e}")
    
    # ========== Step 9: Database already updated during validation ==========
    # Instances were written to database in validate_and_fix_creation_code()
    # and persisted on validation success (backup was cleaned up)
    logger.info(f"Database updated with {len(combos)} combo instances during validation")
    
    # ========== Step 10: Limit combos to samples_per_template ==========
    if len(combos) >= samples_per_template:
        selected_combos = random.sample(combos, samples_per_template)
    else:
        selected_combos = combos
        logger.info(f"Only {len(combos)} combos available for {server_name}, requested {samples_per_template}")
    
    # ========== Step 11: Write to validated_tasks ==========
    validated_tasks_dir = outputs_dir / "validated_tasks"
    write_validated_combos(
        selected_combos, trajectory, traj_hash, task_template, server_name, validated_tasks_dir
    )
    
    # ========== Step 12: Save successful creation code ==========
    try:
        save_sampling_code(
            outputs_dir=outputs_dir,
            server_name=f"{server_name}_creation",
            trajectory_hash=traj_hash,
            code=final_code,
            constraint_analysis={
                "full_deps": full_deps,
                "core_entity": core_entity,
                "success": True,
                "combo_count": len(selected_combos),
            },
        )
    except Exception as e:
        logger.debug(f"Failed to save creation code: {e}")
    
    logger.info(f"Cross Domain Creation successful: {len(selected_combos)} combos for {server_name}")
    return len(selected_combos)


# =============================================================================
# Script Execution with BlockEditor Fix
# =============================================================================

def execute_selection_script(
    script_path: Path,
    db_dir: Path,
    server_name: str,
    timeout: int = 30
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Execute the selection script and return results.
    
    Args:
        script_path: Path to Python script file
        db_dir: Database directory path
        server_name: Server name (for path resolution)
        timeout: Execution timeout in seconds
        
    Returns:
        Tuple of (success, result_dict, error_message)
    """
    try:
        # Pass both db_dir and server_name to the script
        result = subprocess.run(
            ["python", str(script_path), str(db_dir), server_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                return True, output, None
            except json.JSONDecodeError as e:
                return False, {"error": f"Invalid JSON output: {e}", "stdout": result.stdout}, str(e)
        else:
            error_msg = result.stderr or f"Script exited with code {result.returncode}"
            return False, {"error": error_msg, "returncode": result.returncode}, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = f"Script execution timed out after {timeout}s"
        return False, {"error": error_msg}, error_msg
    except Exception as e:
        return False, {"error": str(e)}, str(e)


def extract_python_code(response_content: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in response_content:
        code = response_content.split("```python")[1].split("```")[0]
    elif "```" in response_content:
        code = response_content.split("```")[1].split("```")[0]
    else:
        code = response_content
    return code.strip()


# =============================================================================
# Phase 1: Constraint Analysis
# =============================================================================

def validate_constraint_analysis(analysis: Dict[str, Any]) -> bool:
    """
    Validate that constraint analysis result is complete.
    
    Returns True if the analysis contains all required fields.
    """
    if not analysis:
        return False
    
    # Check required top-level fields
    required_fields = ["sampling_requirements", "instance_assignment_plan"]
    for field in required_fields:
        if field not in analysis:
            logger.warning(f"Constraint analysis missing required field: {field}")
            return False
    
    # Check sampling_requirements format
    for req in analysis.get("sampling_requirements", []):
        if "tool" not in req or "parameter" not in req or "sampling_type" not in req:
            logger.warning(f"Invalid sampling requirement format: {req}")
            return False
    
    return True


def analyze_constraints(
    client,
    trajectory: List[str],
    filtered_policy: str,
    available_entities: List[str],
    entity_summaries: str,
    available_relationships: List[str],
    relationship_summaries: str,
    current_time: str,
) -> Optional[Dict[str, Any]]:
    """
    Phase 1: Analyze trajectory to identify sampling constraints.
    
    Args:
        client: LLM client
        trajectory: List of tool names
        filtered_policy: Filtered policy content (Common + Tool-specific)
        available_entities: List of available entity types
        entity_summaries: Combined markdown of entity database summaries
        available_relationships: List of available relationship types
        relationship_summaries: Combined markdown of relationship database summaries
        current_time: Current simulation time
        
    Returns:
        Constraint analysis JSON or None if failed
    """
    prompt = CONSTRAINT_ANALYSIS_PROMPT.format(
        trajectory=json.dumps(trajectory),
        filtered_policy=filtered_policy,
        available_entities=json.dumps(available_entities),
        entity_summaries=entity_summaries if entity_summaries else "(No entity summaries available)",
        available_relationships=json.dumps(available_relationships),
        relationship_summaries=relationship_summaries if relationship_summaries else "(No relationship summaries available)",
        instance_assignment_examples=INSTANCE_ASSIGNMENT_EXAMPLES,
        current_time=current_time,
    )
    
    try:
        response = client.chat(query=prompt, model_type="textual")
        
        # Parse the JSON response
        content = response.content
        
        # Try to extract JSON from the response
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            # Try to find JSON object in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
        
        # Try to parse JSON, with automatic fix on failure
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # JSON parse failed (likely due to LLM output truncation/concatenation issues)
            logger.warning(f"JSON parse error: {e}, attempting to fix...")
            
            # Use BlockEditor to fix the JSON
            block_editor = WorkflowBlockEditor()
            fixed_json = block_editor.fix_json_content(
                json_content=json_str,
                error=str(e),
                llm_client=client,
                max_retries=2,
            )
            
            if fixed_json:
                try:
                    result = json.loads(fixed_json)
                    logger.info("Successfully fixed JSON after LLM repair")
                    return result
                except json.JSONDecodeError as e2:
                    logger.warning(f"JSON still invalid after fix attempt: {e2}")
                    return None
            else:
                logger.warning(f"Failed to fix JSON: {e}")
                return None
        
    except Exception as e:
        logger.warning(f"Constraint analysis failed: {e}")
        return None


# =============================================================================
# Phase 2: Code Generation and Execution with BlockEditor
# =============================================================================

def generate_selection_code(
    client,
    constraint_analysis: Dict[str, Any],
    db_dir: Path,
    server_name: str,
    entity_files: List[str],
    relationship_files: List[str],
) -> Optional[str]:
    """
    Phase 2: Generate Python code for instance selection.
    
    Args:
        client: LLM client
        constraint_analysis: Result from Phase 1
        db_dir: Database directory path
        server_name: Server name (for path resolution)
        entity_files: List of entity file names
        relationship_files: List of relationship file names
        
    Returns:
        Python script content or None if failed
    """
    # Add path information to constraint analysis for code generation
    path_info = {
        "is_cross_domain": is_cross_domain(server_name),
        "server_name": server_name,
        "entity_dir": str(get_entity_dir(db_dir, server_name)),
        "relationship_dir": str(get_relationship_dir(db_dir, server_name)),
    }
    
    prompt = SELECTION_CODE_PROMPT.format(
        constraint_analysis_json=json.dumps({**constraint_analysis, "_path_info": path_info}, indent=2),
        db_dir=str(db_dir),
        entity_files=json.dumps(entity_files),
        relationship_files=json.dumps(relationship_files),
    )
    
    try:
        response = client.chat(query=prompt, model_type="coding")
        return extract_python_code(response.content)
    except Exception as e:
        logger.warning(f"Code generation failed: {e}")
        return None


def generate_and_execute_selection(
    client,
    constraint_analysis: Dict[str, Any],
    db_dir: Path,
    server_name: str,
    max_retries: int = 3,
    timeout: int = 30,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Generate selection code and execute it, with BlockEditor fixes on failure.
    
    Args:
        client: LLM client
        constraint_analysis: Result from Phase 1
        db_dir: Database directory path
        server_name: Server name
        max_retries: Maximum retry attempts
        timeout: Script execution timeout
        
    Returns:
        Tuple of (success, result_dict, generated_code)
        generated_code is the final working code (for saving in Single Domain mode)
    """
    entity_files = get_available_entities(db_dir, server_name)
    relationship_files = get_available_relationships(db_dir, server_name)
    
    # Create BlockEditor for code fixing
    block_editor = WorkflowBlockEditor()
    
    last_error = None
    final_code = None
    
    # Generate initial code
    code = generate_selection_code(
        client,
        constraint_analysis,
        db_dir,
        server_name,
        entity_files,
        relationship_files,
    )
    
    if not code:
        return False, {"error": "Failed to generate selection code", "attempts": 0}, None
    
    final_code = code
    
    # Create a temporary file for the script
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        f.write(code)
        script_path = Path(f.name)
    
    try:
        for attempt in range(max_retries):
            # Execute code
            success, result, error_msg = execute_selection_script(
                script_path, db_dir, server_name, timeout
            )
            
            if success:
                # Read the final working code
                final_code = script_path.read_text()
                return True, result, final_code
            
            # Store error for logging
            last_error = error_msg
            logger.debug(f"Selection attempt {attempt + 1} failed: {last_error}")
            
            # Use BlockEditor to fix the code file
            if attempt < max_retries - 1:  # Don't fix on last attempt
                fixed = block_editor.fix_code(
                    file_path=script_path,
                    error=error_msg,
                    llm_client=client,
                    language="python",
                    max_retries=2,  # Inner retry for BlockEditor
                )
                
                if not fixed:
                    logger.warning(f"BlockEditor failed to fix code on attempt {attempt + 1}")
                    # Regenerate code from scratch
                    constraint_analysis["_last_error"] = last_error
                    constraint_analysis["_retry_attempt"] = attempt + 1
                    
                    new_code = generate_selection_code(
                        client,
                        constraint_analysis,
                        db_dir,
                        server_name,
                        entity_files,
                        relationship_files,
                    )
                    
                    if new_code:
                        script_path.write_text(new_code)
                        final_code = new_code
        
        return False, {"error": last_error, "attempts": max_retries}, final_code
        
    finally:
        # Clean up temporary file
        script_path.unlink(missing_ok=True)


# =============================================================================
# Main Processing Function
# =============================================================================

def process_and_save_single_domain_trajectory(
    trajectory_task: Dict[str, Any],
    client,
    settings: Dict[str, Any],
    current_time: str,
    outputs_dir: Path,
) -> int:
    """
    Process a Single Domain trajectory and save results immediately.
    
    Also saves the sampling code for reuse by Cross Domain processing.
    
    Phase 1: Constraint Analysis (must succeed before Phase 2)
    Phase 2: Code Generation and Execution (with BlockEditor fixes)
    
    Args:
        trajectory_task: Dict containing trajectory info
        client: LLM client
        settings: Step settings
        current_time: Current simulation time
        outputs_dir: Base outputs directory (for saving sampling codes)
        
    Returns:
        Number of combos saved (0 if failed)
    """
    template = trajectory_task["template"]
    template_idx = trajectory_task["template_idx"]
    server_name = trajectory_task["server_name"]
    policy_content = trajectory_task["policy_content"]
    db_dir = trajectory_task["db_dir"]
    summary_dir = trajectory_task["summary_dir"]
    
    trajectory = template.get("trajectory", [])
    if not trajectory:
        return None
    
    enable_policy_filtering = settings.get("enable_policy_filtering", True)
    max_code_retries = settings.get("max_code_retries", 3)
    code_timeout = settings.get("code_timeout", 30)
    samples_per_template = settings.get("samples_per_template", 1)
    max_step1_retries = settings.get("max_step1_retries", 3)
    
    # Get available entities and relationships
    available_entities = get_available_entities(db_dir, server_name)
    available_relationships = get_available_relationships(db_dir, server_name)
    entity_summaries = get_entity_summaries(summary_dir, available_entities)
    relationship_summaries = get_relationship_summaries(summary_dir, server_name)
    
    # ========== Step 1: Filter Policy ==========
    if enable_policy_filtering and has_structured_markers(policy_content):
        filtered_policy = filter_policy_for_trajectory(policy_content, trajectory)
    else:
        # Use full policy if no markers or filtering disabled
        filtered_policy = policy_content
    
    # ========== Phase 1: Constraint Analysis ==========
    # Must complete successfully before Phase 2
    constraint_analysis = None
    for attempt in range(max_step1_retries):
        constraint_analysis = analyze_constraints(
            client,
            trajectory,
            filtered_policy,
            available_entities,
            entity_summaries,
            available_relationships,
            relationship_summaries,
            current_time,
        )
        
        if constraint_analysis and validate_constraint_analysis(constraint_analysis):
            logger.debug(f"Phase 1 succeeded for {server_name} template {template_idx}")
            break
        else:
            logger.warning(f"Phase 1 attempt {attempt + 1} failed for {server_name} template {template_idx}")
            constraint_analysis = None
    
    if not constraint_analysis:
        # Phase 1 completely failed - do NOT proceed to Phase 2
        logger.warning(f"Phase 1 failed after {max_step1_retries} attempts for {server_name} template {template_idx}")
        return None
    
    # ========== Phase 2: Code Generation and Execution ==========
    # Execute code ONCE to get ALL valid combinations
    success, selection_result, generated_code = generate_and_execute_selection(
        client,
        constraint_analysis,
        db_dir,
        server_name,
        max_retries=max_code_retries,
        timeout=code_timeout,
    )
    
    if not success:
        # Phase 2 failed - return None to trigger retry
        logger.warning(f"Phase 2 failed for {server_name} template {template_idx}")
        return None
    
    # ========== Save Sampling Code for Cross Domain Reuse ==========
    # Only for Single Domain trajectories
    traj_hash = compute_trajectory_hash(trajectory)
    if generated_code:
        save_sampling_code(
            outputs_dir=outputs_dir,
            server_name=server_name,
            trajectory_hash=traj_hash,
            code=generated_code,
            constraint_analysis=constraint_analysis,
        )
    
    # Extract all valid combos from the result
    all_valid_combos = selection_result.get("all_valid_combos", [])
    
    # Handle legacy format (single combo without all_valid_combos wrapper)
    if not all_valid_combos and "entity_instances" in selection_result:
        all_valid_combos = [selection_result]
    
    if not all_valid_combos:
        logger.warning(f"No valid combos found for {server_name} template {template_idx}")
        return None
    
    # Select samples_per_template combos from all valid combos
    if len(all_valid_combos) >= samples_per_template:
        selected_combos = random.sample(all_valid_combos, samples_per_template)
    else:
        # Not enough combos, use all available
        selected_combos = all_valid_combos
        logger.info(f"Only {len(all_valid_combos)} valid combos found for {server_name} template {template_idx}, requested {samples_per_template}")
    
    # ========== Save Results Immediately ==========
    # Use trajectory hash as filename, save all combos in one file
    server_combos_dir = trajectory_task["server_combos_dir"]
    
    # Get task_template info for later use in Task Instantiation
    task_template = template.get("task_template", {})
    
    # Build the output structure
    output_data = {
        "trajectory": trajectory,
        "trajectory_hash": traj_hash,
        "task_template": {
            "instruction": task_template.get("instruction", ""),
            "reason_for_call": task_template.get("reason_for_call", ""),
        },
        "combos": [],
        "metadata": {
            "server_name": server_name,
            "samples_count": len(selected_combos),
            "generated_at": datetime.now().isoformat(),
        }
    }
    
    # Add each combo to the list
    for sample_idx, combo in enumerate(selected_combos):
        entity_instances = combo.get("entity_instances", {})
        combo_id = compute_combo_id(traj_hash, entity_instances)
        combo_entry = {
            "sample_idx": sample_idx,
            "combo_id": combo_id,
            "entity_instances": entity_instances,
            "value_domain_samples": combo.get("value_domain_samples", {}),
            "constraint_analysis": constraint_analysis,
        }
        output_data["combos"].append(combo_entry)
    
    # Save to file named by trajectory hash
    combo_path = server_combos_dir / f"{traj_hash}.json"
    save_json(output_data, combo_path)
    
    logger.debug(f"Saved {len(selected_combos)} combos for {server_name} trajectory {traj_hash}")
    return len(selected_combos)


# =============================================================================
# Cross Domain Trajectory Processing
# =============================================================================

def process_and_save_cross_domain_trajectory(
    trajectory_task: Dict[str, Any],
    client,
    settings: Dict[str, Any],
    current_time: str,
    outputs_dir: Path,
) -> int:
    """
    Process a Cross Domain trajectory using the code fusion approach.
    
    Steps:
    1. Parse trajectory to extract domain-specific segments
    2. Load Single Domain sampling codes for each segment
    3. Use LLM to fuse codes with global constraints
    4. Execute fused code to generate combinations
    
    Args:
        trajectory_task: Dict containing trajectory info
        client: LLM client
        settings: Step settings
        current_time: Current simulation time
        outputs_dir: Base outputs directory
        
    Returns:
        Number of combos saved (0 if failed)
    """
    template = trajectory_task["template"]
    template_idx = trajectory_task["template_idx"]
    server_name = trajectory_task["server_name"]
    policy_content = trajectory_task["policy_content"]
    db_dir = trajectory_task["db_dir"]
    
    trajectory = template.get("trajectory", [])
    if not trajectory:
        return None
    
    max_code_retries = settings.get("max_code_retries", 3)
    code_timeout = settings.get("code_timeout", 30)
    samples_per_template = settings.get("samples_per_template", 1)
    
    traj_hash = compute_trajectory_hash(trajectory)
    
    # ========== Step 1: Parse Cross Domain Trajectory ==========
    domain_segments = parse_cross_domain_trajectory(trajectory)
    
    if not domain_segments:
        logger.warning(f"Failed to parse Cross Domain trajectory for {server_name}")
        return None
    
    logger.info(f"Cross Domain {server_name}: parsed {len(domain_segments)} domain segments")
    
    # ========== Step 2: Load Single Domain Sampling Codes ==========
    found_codes = find_matching_single_domain_codes(outputs_dir, domain_segments)
    
    # Log what was found
    found_count = len(found_codes)
    total_domains = len(domain_segments)
    logger.info(f"Cross Domain {server_name}: found {found_count}/{total_domains} Single Domain codes")
    
    if found_count == 0:
        # No Single Domain codes found - fall back to original approach
        logger.warning(f"No Single Domain codes found for {server_name}, using fallback approach")
        # Could implement fallback here, but for now return None
        return None
    
    # ========== Step 3: Get Core Entity ==========
    # The core entity is the user initiating the interaction and MUST be consistent across domains
    cross_domain_combos_dir = outputs_dir / "cross_domain_combinations"
    combo_file = cross_domain_combos_dir / f"{server_name}.json"
    
    core_entity = ""
    if combo_file.exists():
        try:
            combo_data = load_json(combo_file)
            core_entity = get_core_entity_from_combination(combo_data)
        except Exception as e:
            logger.warning(f"Failed to load core entity for {server_name}: {e}")
    
    if not core_entity:
        return None
    
    # ========== Step 4: Generate and Execute Fused Code ==========
    success, selection_result, generated_code = generate_and_execute_cross_domain_selection(
        client,
        trajectory,
        domain_segments,
        found_codes,
        core_entity,
        policy_content,
        db_dir,
        server_name,
        max_retries=max_code_retries,
        timeout=code_timeout,
    )

    # ========== Extract and Select Combos ==========
    all_valid_combos = selection_result.get("all_valid_combos", [])

    if not success:
        logger.warning(f"Cross Domain code fusion failed for {server_name} template {template_idx}")
        # Save the failed code for debugging
        if generated_code:
            try:
                save_sampling_code(
                    outputs_dir=outputs_dir,
                    server_name=f"{server_name}_failed",  # Mark as failed for debugging
                    trajectory_hash=traj_hash,
                    code=generated_code,
                    constraint_analysis={
                        "error": "Code fusion failed",
                        "selection_result": selection_result,
                        "domain_segments": domain_segments,
                        "core_entity": core_entity,
                        "found_codes": list(found_codes.keys()),
                        "combo_count": len(all_valid_combos),
                    },
                )
                logger.info(f"Saved failed Cross Domain code for debugging: {server_name}/{traj_hash}")
            except Exception as e:
                logger.warning(f"Failed to save debug code: {e}")
        return None

    # ========== Save Successful Cross Domain Code ==========
    # Save the successful fused code for future reference/debugging
    if generated_code:
        try:
            save_sampling_code(
                outputs_dir=outputs_dir,
                server_name=f"{server_name}_fused",  # Mark as fused for identification
                trajectory_hash=traj_hash,
                code=generated_code,
                constraint_analysis={
                    "domain_segments": domain_segments,
                    "core_entity": core_entity,
                    "found_codes": list(found_codes.keys()),
                    "success": True,
                    "combo_count": len(all_valid_combos),
                },
            )
            logger.debug(f"Saved successful Cross Domain code: {server_name}/{traj_hash}")
        except Exception as e:
            logger.warning(f"Failed to save successful Cross Domain code: {e}")
    
    if not all_valid_combos and "entity_instances" in selection_result:
        all_valid_combos = [selection_result]
    
    if not all_valid_combos:
        logger.warning(f"No valid combos found for Cross Domain {server_name} template {template_idx}")
        return None
    
    if len(all_valid_combos) >= samples_per_template:
        selected_combos = random.sample(all_valid_combos, samples_per_template)
    else:
        selected_combos = all_valid_combos
        logger.info(f"Only {len(all_valid_combos)} valid combos found for {server_name}, requested {samples_per_template}")
    
    # ========== Save Results ==========
    server_combos_dir = trajectory_task["server_combos_dir"]
    task_template = template.get("task_template", {})
    
    output_data = {
        "trajectory": trajectory,
        "trajectory_hash": traj_hash,
        "task_template": {
            "instruction": task_template.get("instruction", ""),
            "reason_for_call": task_template.get("reason_for_call", ""),
        },
        "combos": [],
        "metadata": {
            "server_name": server_name,
            "samples_count": len(selected_combos),
            "generated_at": datetime.now().isoformat(),
            "is_cross_domain": True,
            "domain_segments": {k: list(v) for k, v in domain_segments.items()},
            "core_entity": core_entity,
            "single_domain_codes_found": list(found_codes.keys()),
        }
    }
    
    for sample_idx, combo in enumerate(selected_combos):
        entity_instances = combo.get("entity_instances", {})
        combo_id = compute_combo_id(traj_hash, entity_instances)
        combo_entry = {
            "sample_idx": sample_idx,
            "combo_id": combo_id,
            "entity_instances": entity_instances,
            "value_domain_samples": combo.get("value_domain_samples", {}),
        }
        output_data["combos"].append(combo_entry)
    
    combo_path = server_combos_dir / f"{traj_hash}.json"
    save_json(output_data, combo_path)
    
    logger.info(f"Cross Domain {server_name}: saved {len(selected_combos)} combos for trajectory {traj_hash}")
    return len(selected_combos)


# =============================================================================
# Helper Functions for Convergence Detection
# =============================================================================

def collect_all_trajectory_tasks(
    templates_dir: Path,
    policies_dir: Path,
    db_dir: Path,
    summary_dir: Path,
    combos_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Collect all trajectory tasks from template files.
    
    Args:
        templates_dir: Path to task templates directory
        policies_dir: Path to policies directory
        db_dir: Path to database outputs directory
        summary_dir: Path to database_summary directory
        combos_dir: Path to combinations output directory
    
    Returns:
        List of all trajectory tasks (including already processed ones)
    """
    all_tasks = []
    template_files = list(templates_dir.glob("*.json"))
    
    for template_file in template_files:
        server_name = template_file.stem
        templates_data = load_json(template_file)
        templates = templates_data.get("templates", [])
        
        
        if not templates:
            continue
        
        # Load policy for this server
        policy_content = load_policy_for_server(policies_dir, server_name)
        
        # Create output directory for this server
        server_combos_dir = ensure_dir(combos_dir / server_name)
        
        # Create tasks for each template
        for i, template in enumerate(templates):
            trajectory = template.get("trajectory", [])
            if not trajectory:
                continue
            
            traj_hash = compute_trajectory_hash(trajectory)
            
            all_tasks.append({
                "template": template,
                "template_idx": i,
                "server_name": server_name,
                "policy_content": policy_content,
                "db_dir": db_dir,
                "summary_dir": summary_dir,
                "server_combos_dir": server_combos_dir,
                "trajectory_hash": traj_hash,
            })
    
    return all_tasks


def get_pending_tasks(all_tasks: List[Dict[str, Any]], combos_dir: Path) -> List[Dict[str, Any]]:
    """
    Filter tasks to get only pending (not yet processed) ones.
    
    Args:
        all_tasks: All trajectory tasks
        combos_dir: Combinations output directory
        
    Returns:
        List of pending tasks
    """
    pending_tasks = []
    
    for task in all_tasks:
        server_name = task["server_name"]
        traj_hash = task["trajectory_hash"]
        
        # Check if already processed
        server_combos_dir = combos_dir / server_name
        combo_file = server_combos_dir / f"{traj_hash}.json"
        
        if not combo_file.exists():
            pending_tasks.append(task)
    
    return pending_tasks


def save_skipped_trajectories(
    combos_dir: Path,
    skipped_tasks: List[Dict[str, Any]],
):
    """
    Save skipped trajectories to a JSON file.
    
    Args:
        combos_dir: Combinations output directory
        skipped_tasks: List of tasks that could not be processed
    """
    skipped_path = combos_dir / "_skipped_trajectories.json"
    
    # Load existing skipped if any
    if skipped_path.exists():
        existing = load_json(skipped_path)
    else:
        existing = {}
    
    # Group by server
    for task in skipped_tasks:
        server_name = task["server_name"]
        if server_name not in existing:
            existing[server_name] = []
        
        # Check if already recorded
        traj_hash = task["trajectory_hash"]
        if not any(s.get("trajectory_hash") == traj_hash for s in existing[server_name]):
            existing[server_name].append({
                "trajectory": task["template"].get("trajectory", []),
                "trajectory_hash": traj_hash,
                "reason": "No valid entity combinations found after multiple attempts",
            })
    
    save_json(existing, skipped_path)
    logger.info(f"Saved {len(skipped_tasks)} skipped trajectories to {skipped_path}")


# =============================================================================
# Step Handler
# =============================================================================

@step_handler("s15_instance_combos_selection", auto_retry=False)  # Disable auto_retry, use convergence detection
def instance_combos_selection_step(state: WorkflowState) -> WorkflowState:
    """
    Select entity instance combinations using Plan-Execution approach.
    
    Process differs for Single Domain vs Cross Domain:
    
    Single Domain:
    1. Load task templates
    2. For each trajectory (parallel):
       a. Filter relevant policy sections (Common + Tool-specific)
       b. Phase 1: Analyze constraints with LLM (must succeed)
       c. Phase 2: Generate and execute selection code (with BlockEditor fixes)
       d. Save sampling code for Cross Domain reuse
    3. Save instance combinations
    
    Cross Domain (Code Fusion Approach):
    1. Parse trajectory to extract domain-specific segments
    2. Load Single Domain sampling codes for each segment
    3. Use LLM to fuse codes with global constraints (shared entities, etc.)
    4. Execute fused code
    5. Save combinations
    
    Output:
    - combinations/{server}/*.json
    - sampling_codes/{server}/*.py (for Single Domain, reused by Cross Domain)
    - combinations/_skipped_trajectories.json (for trajectories with no valid combos)
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    step_config = settings.steps.s15_instance_combos_selection
    
    samples_per_template = step_config.get("samples_per_template", 1)
    min_progress_rate = step_config.get("min_progress_rate", 0.2)  # 20% threshold
    current_time = settings.workflow.simulation_time
    
    templates_dir = Path(state.task_templates_dir)
    db_dir = Path(state.databases_dir) / "outputs"
    summary_dir = outputs_dir / "database_summary"
    policies_dir = Path(state.policies_dir)
    combos_dir = ensure_dir(outputs_dir / "combinations")
    
    # Get LLM client
    client = get_client()
    
    # Process each template file
    template_files = list(templates_dir.glob("*.json"))
    logger.info(f"Processing {len(template_files)} template files")
    
    # Count total templates for progress
    total_templates = sum(
        len(load_json(f).get("templates", []))
        for f in template_files
    )
    
    # Collect all trajectory tasks
    all_tasks = collect_all_trajectory_tasks(
        templates_dir, policies_dir, db_dir, summary_dir, combos_dir
    )
    
    # Separate Single Domain and Cross Domain tasks
    single_domain_tasks = [t for t in all_tasks if not is_cross_domain(t["server_name"])]
    cross_domain_tasks = [t for t in all_tasks if is_cross_domain(t["server_name"])]
    
    logger.info(f"Tasks breakdown: {len(single_domain_tasks)} Single Domain, {len(cross_domain_tasks)} Cross Domain")
    
    # =========================================================================
    # Phase 1: Process Single Domain tasks first (to generate reusable codes)
    # =========================================================================
    
    logger.info("=" * 60)
    logger.info("Phase 1: Processing Single Domain trajectories")
    logger.info("=" * 60)
    
    previous_failed_count = None
    round_number = 0
    
    while True:
        round_number += 1
        
        # Get pending Single Domain tasks
        pending_tasks = get_pending_tasks(single_domain_tasks, combos_dir)
        
        if not pending_tasks:
            logger.info("All Single Domain trajectories processed!")
            break
        
        logger.info(f"Single Domain Round {round_number}: Processing {len(pending_tasks)} pending trajectories")
        
        # Process Single Domain trajectories in parallel
        def process_single_domain_task(task):
            return process_and_save_single_domain_trajectory(
                trajectory_task=task,
                client=client,
                settings=step_config,
                current_time=current_time,
                outputs_dir=outputs_dir,
            )
        
        results = parallel_process(
            items=pending_tasks,
            process_func=process_single_domain_task,
            description=f"Single Domain Round {round_number}: Processing trajectories",
        )
        
        # Count successes and failures
        success_count = sum(1 for r in results if r is not None and isinstance(r, int) and r > 0)
        current_failed_count = len(pending_tasks) - success_count
        
        logger.info(f"Single Domain Round {round_number}: {success_count} succeeded, {current_failed_count} failed")
        
        # Check convergence
        if current_failed_count == 0:
            break
        
        if previous_failed_count is not None and previous_failed_count > 0:
            delta = previous_failed_count - current_failed_count
            progress_rate = delta / previous_failed_count
            
            if progress_rate < min_progress_rate:
                logger.info(f"Single Domain converged: progress_rate {progress_rate:.1%} < {min_progress_rate:.0%}")
                break
        
        previous_failed_count = current_failed_count
    
    # =========================================================================
    # Phase 2: Process Cross Domain tasks (using Combination Creation mode)
    # Cross Domain bypasses sampling and directly creates valid instances
    # Results are written directly to validated_tasks/
    # =========================================================================
    
    logger.info("=" * 60)
    logger.info("Phase 2: Processing Cross Domain trajectories (using Combination Creation)")
    logger.info("=" * 60)
    
    validated_tasks_dir = ensure_dir(outputs_dir / "validated_tasks")
    
    # Get pending Cross Domain tasks (check both combos_dir and validated_tasks_dir)
    def is_cross_domain_completed(task):
        """Check if Cross Domain task has been completed (in validated_tasks)."""
        server_name = task["server_name"]
        template = task["template"]
        trajectory = template.get("trajectory", [])
        traj_hash = compute_trajectory_hash(trajectory)
        
        # Check validated_tasks directory
        validated_file = validated_tasks_dir / server_name / "validated_combos.json"
        if validated_file.exists():
            try:
                validated = load_json(validated_file)
                for entry in validated:
                    if entry.get("trajectory_hash") == traj_hash:
                        return True
            except Exception:
                pass
        return False
    
    pending_cross_domain = [t for t in cross_domain_tasks if not is_cross_domain_completed(t)]
    
    if not pending_cross_domain:
        logger.info("All Cross Domain trajectories already processed!")
    else:
        logger.info(f"Cross Domain: Processing {len(pending_cross_domain)} pending trajectories")
        
        # Add summary_dir to each task for Creation mode
        for task in pending_cross_domain:
            task["summary_dir"] = summary_dir
        
        # =====================================================================
        # Group tasks by Domain Combination (server_name)
        # Different Domain Combinations can be processed in parallel,
        # but trajectories within the same Domain Combination must be serial
        # to avoid database race conditions (they share the same database).
        # =====================================================================
        from collections import defaultdict
        
        combo_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for task in pending_cross_domain:
            combo_groups[task["server_name"]].append(task)
        
        logger.info(f"Cross Domain: {len(combo_groups)} Domain Combinations to process")
        
        # Process all trajectories within a single Domain Combination (serial)
        def process_domain_combination(item: Tuple[str, List[Dict[str, Any]]]) -> Tuple[str, int, int]:
            """
            Process all trajectories for a single Domain Combination serially.
            
            Args:
                item: Tuple of (server_name, list of tasks)
                
            Returns:
                Tuple of (server_name, success_count, fail_count)
            """
            server_name, tasks = item
            success_count = 0
            fail_count = 0

            print(server_name)
            
            for task in tasks:
                try:
                    result = process_cross_domain_with_creation(
                        trajectory_task=task,
                        client=client,
                        settings=step_config,
                        outputs_dir=outputs_dir,
                    )
                    if result is not None and isinstance(result, int) and result > 0:
                        success_count += result
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Error processing trajectory in {server_name}: {e}")
                    fail_count += 1
            
            logger.info(f"  {server_name}: {success_count} succeeded, {fail_count} failed")
            return server_name, success_count, fail_count
        
        # Process different Domain Combinations in parallel
        results = parallel_process(
            items=list(combo_groups.items()),
            process_func=process_domain_combination,
            description="Cross Domain: Processing combinations",
        )
        
        # Aggregate results across all Domain Combinations
        total_success = sum(r[1] for r in results if r is not None)
        total_failed = sum(r[2] for r in results if r is not None)
        
        logger.info(f"Cross Domain: {total_success} trajectories succeeded, {total_failed} failed")
    
    # =========================================================================
    # Final: Collect skipped trajectories and update state
    # =========================================================================
    
    # Collect failed Single Domain tasks (check combos_dir)
    failed_single_domain = get_pending_tasks(single_domain_tasks, combos_dir)
    
    # Collect failed Cross Domain tasks (check validated_tasks_dir)
    failed_cross_domain = [t for t in cross_domain_tasks if not is_cross_domain_completed(t)]
    
    failed_tasks = failed_single_domain + failed_cross_domain
    
    if failed_tasks:
        save_skipped_trajectories(combos_dir, failed_tasks)
        logger.info(f"Skipped {len(failed_tasks)} trajectories with no valid combos")
        logger.info(f"  - Single Domain: {len(failed_single_domain)}")
        logger.info(f"  - Cross Domain: {len(failed_cross_domain)}")
        
        # Mark step as completed despite skipped trajectories
        state.mark_step_completed("s15_instance_combos_selection")
    
    state.combinations_dir = str(combos_dir)
    
    # Count total combos generated (grouped by type)
    single_domain_combos = 0
    cross_domain_combos = 0
    
    # Count Single Domain combos from combos_dir
    for server_dir in combos_dir.iterdir():
        if server_dir.is_dir() and not server_dir.name.startswith("_"):
            if not is_cross_domain(server_dir.name):
                combo_count = len(list(server_dir.glob("*.json")))
                single_domain_combos += combo_count
    
    # Count Cross Domain combos from validated_tasks_dir
    if validated_tasks_dir.exists():
        for server_dir in validated_tasks_dir.iterdir():
            if server_dir.is_dir() and is_cross_domain(server_dir.name):
                validated_file = server_dir / "validated_combos.json"
                if validated_file.exists():
                    try:
                        validated = load_json(validated_file)
                        cross_domain_combos += len(validated)
                    except Exception:
                        pass
    
    total_combos = single_domain_combos + cross_domain_combos
    
    # Update progress
    state.update_step_progress(
        "s15_instance_combos_selection",
        total=total_templates,
        completed=total_combos,
    )
    
    logger.info("=" * 60)
    logger.info("Instance Combinations Complete")
    logger.info(f"  Single Domain: {single_domain_combos} combinations")
    logger.info(f"  Cross Domain: {cross_domain_combos} combinations (in validated_tasks/)")
    logger.info(f"  Total: {total_combos} combinations")
    logger.info("=" * 60)
    
    return state
    