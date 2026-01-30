"""
Step 17: Task Instantiation & Query Generation

Instantiate tasks with entity values and generate startup queries.
This step combines the functionality of the original S17 and S18.

Input: validated_tasks/{server}/validated_combos.json, fixed_blueprints.json
Output: queries/{server}.jsonl
"""

import json
import logging
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import TASK_INSTANTIATION_PROMPT, HALLUCINATION_RETRY_PROMPT, COMPLETENESS_RETRY_PROMPT
from .base import step_handler, save_json, load_json, ensure_dir, get_client, parallel_process

logger = logging.getLogger(__name__)


# =============================================================================
# Hallucination Detection Functions
# =============================================================================

def extract_bracketed_values(text: str) -> List[str]:
    """
    Extract all values wrapped in [[ ]] from text.
    
    Args:
        text: Text containing bracketed values like [[uuid-xxx]] or [[active]]
        
    Returns:
        List of extracted values (without brackets)
    """
    if not text:
        return []
    # Use [^\[\]]+ to prevent matching across malformed brackets
    # This ensures we only capture content without [ or ] inside
    return re.findall(r'\[\[([^\[\]]+)\]\]', text)


def collect_instance_values(combo: Dict[str, Any]) -> Set[str]:
    """
    Collect all valid values from instances for validation.
    
    Collects values from:
    - entity_instances (nested dict/list structures, recursively)
    - value_domain_samples (flat key-value pairs, values only)
    
    Args:
        combo: The combo dict containing entity_instances and value_domain_samples
        
    Returns:
        Set of all string values that are valid for use
    """
    values: Set[str] = set()
    
    def collect_recursive(obj: Any) -> None:
        """Recursively collect all string values from nested structures."""
        if obj is None:
            return
        if isinstance(obj, str):
            # Add the string value (strip whitespace)
            stripped = obj.strip()
            if stripped:
                values.add(stripped)
        elif isinstance(obj, bool):
            # Booleans: use lowercase to match LLM output format
            values.add(str(obj).lower())
        elif isinstance(obj, (int, float)):
            # Convert numeric primitives to string representation
            values.add(str(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                collect_recursive(v)
        elif isinstance(obj, list):
            for item in obj:
                collect_recursive(item)
    
    # Collect from entity_instances (recursively handles nested dicts/lists)
    if "entity_instances" in combo:
        collect_recursive(combo["entity_instances"])
    
    # Collect from value_domain_samples (values only, not keys like "tool.param")
    # Use collect_recursive to handle nested dicts/lists (e.g., updates: {dose_strength: "250mg", ...})
    if "value_domain_samples" in combo:
        for value in combo["value_domain_samples"].values():
            collect_recursive(value)
    
    return values


def validate_no_hallucinations(result: Dict[str, Any], combo: Dict[str, Any]) -> List[str]:
    """
    Validate LLM output contains no hallucinated values.
    
    Extracts all [[ ]] wrapped values from instruction, start_up_query, 
    and reason_for_call, then checks if each exists in the instance values.
    
    Args:
        result: LLM output dict with instruction, start_up_query, reason_for_call
        combo: The combo dict containing valid instances
        
    Returns:
        List of hallucinated values (values not found in instances).
        Empty list means no hallucinations detected.
    """
    # Collect all valid values from instances
    valid_values = collect_instance_values(combo)
    
    # Extract bracketed values from all relevant fields
    fields_to_check = ["instruction", "start_up_query", "reason_for_call"]
    all_bracketed: List[str] = []
    
    for field in fields_to_check:
        text = result.get(field, "")
        if text:
            all_bracketed.extend(extract_bracketed_values(text))
    
    # Helper function for substring matching
    def is_valid_value(value: str, valid_values: Set[str]) -> bool:
        """Check if value matches any valid value (exact or substring)."""
        if value in valid_values:
            return True
        # Check if value is a substring of any valid value (for comma-separated cases)
        for valid_val in valid_values:
            if value in valid_val:
                return True
        return False
    
    # Check each bracketed value against valid values
    hallucinated: List[str] = []
    for value in all_bracketed:
        value_stripped = value.strip()
        if value_stripped and not is_valid_value(value_stripped, valid_values):
            hallucinated.append(value_stripped)
    
    return hallucinated


def collect_required_values(combo: Dict[str, Any]) -> Set[str]:
    """
    Collect all values from value_domain_samples that MUST appear in instruction.
    
    Flattens lists and converts all values to strings for comparison.
    
    Args:
        combo: The combo dict containing value_domain_samples
        
    Returns:
        Set of string values that must appear in the instruction
    """
    required: Set[str] = set()
    
    if "value_domain_samples" not in combo:
        return required
    
    def add_value(value: Any) -> None:
        """Add a value to the required set, handling different types."""
        if value is None:
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                required.add(stripped)
        elif isinstance(value, bool):
            # Booleans: add lowercase string representation
            required.add(str(value).lower())
        elif isinstance(value, (int, float)):
            required.add(str(value))
        elif isinstance(value, list):
            # For lists, each element must appear individually
            for item in value:
                add_value(item)
        elif isinstance(value, dict):
            # For nested dicts, recursively collect values
            for v in value.values():
                add_value(v)
    
    for value in combo["value_domain_samples"].values():
        add_value(value)
    
    return required


def validate_completeness(result: Dict[str, Any], combo: Dict[str, Any]) -> List[str]:
    """
    Validate that ALL required values from value_domain_samples appear in instruction.
    
    This is the reverse validation: instead of checking if values in [[ ]] are valid,
    we check if all required values appear in the instruction with [[ ]] markers.
    
    Args:
        result: LLM output dict with instruction
        combo: The combo dict containing value_domain_samples
        
    Returns:
        List of missing values (required values not found in instruction).
        Empty list means all required values are present.
    """
    instruction = result.get("instruction", "")
    if not instruction:
        # If no instruction, all required values are missing
        return list(collect_required_values(combo))
    
    # Extract all bracketed values from instruction
    bracketed_values = set(extract_bracketed_values(instruction))
    # Also check without brackets (for substring matches)
    bracketed_lower = {v.lower() for v in bracketed_values}
    
    # Get all required values
    required_values = collect_required_values(combo)
    
    # Check each required value
    missing: List[str] = []
    for req_value in required_values:
        req_lower = req_value.lower()
        
        # Check if value appears in bracketed values (case-insensitive)
        found = False
        
        # Exact match
        if req_value in bracketed_values or req_lower in bracketed_lower:
            found = True
        
        # Substring match (for comma-separated or partial matches)
        if not found:
            for bracketed in bracketed_values:
                if req_value in bracketed or req_lower in bracketed.lower():
                    found = True
                    break
        
        if not found:
            missing.append(req_value)
    
    return missing



def get_combo_id(combo: Dict[str, Any]) -> str:
    """
    Get combo_id from combo.
    
    New combos (from s15+) will have combo_id pre-computed.
    For old data, fallback to computing on-the-fly.
    """
    combo_id = combo.get("combo_id")
    if combo_id:
        return combo_id
    
    # Fallback: compute on-the-fly for old data
    trajectory_hash = combo.get("trajectory_hash", "")
    entity_instances = combo.get("entity_instances", {})
    content = json.dumps({
        "trajectory_hash": trajectory_hash,
        "entity_instances": entity_instances,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def load_existing_queries(queries_dir: Path) -> Tuple[Dict[str, List[Dict]], Set[str]]:
    """
    Load existing queries from JSONL files for resume support.
    
    Returns:
        - Dict mapping server_name to list of query results
        - Set of processed combo IDs (extracted from task_info)
    """
    existing_queries: Dict[str, List[Dict]] = {}
    processed_ids: Set[str] = set()
    
    if not queries_dir.exists():
        return existing_queries, processed_ids
    
    for jsonl_file in queries_dir.glob("*.jsonl"):
        server_name = jsonl_file.stem
        queries = []
        
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        query = json.loads(line)
                        queries.append(query)
                        # Extract combo_id from task_info if present
                        combo_id = query.get("task_info", {}).get("combo_id")
                        if combo_id:
                            processed_ids.add(combo_id)
            
            existing_queries[server_name] = queries
            logger.debug(f"Loaded {len(queries)} existing queries for {server_name}")
        except Exception as e:
            logger.warning(f"Failed to load existing queries from {jsonl_file}: {e}")
    
    return existing_queries, processed_ids


def format_entity_instances(combo: Dict[str, Any]) -> str:
    """
    Format entity instances only (for known_info).
    
    This provides context about the entities involved but NOT the function parameters.
    """
    lines = []
    
    if "entity_instances" in combo:
        entity_instances = combo["entity_instances"]
        if entity_instances:
            lines.append("# Entity Instances")
            lines.append("")
            for entity_name, entity_data in entity_instances.items():
                if isinstance(entity_data, dict):
                    lines.append(f"## {entity_name}")
                    for key, value in entity_data.items():
                        lines.append(f"- {key}: {value}")
                    lines.append("")
                elif isinstance(entity_data, list):
                    lines.append(f"## {entity_name}")
                    for idx, item in enumerate(entity_data):
                        lines.append(f"### Instance {idx + 1}")
                        if isinstance(item, dict):
                            for key, value in item.items():
                                lines.append(f"- {key}: {value}")
                        else:
                            lines.append(f"- {item}")
                    lines.append("")
    
    return "\n".join(lines) if lines else "No entity instances provided."


def format_required_values(combo: Dict[str, Any]) -> str:
    """
    Format value_domain_samples as required values that MUST appear in instruction.
    
    Groups by tool name and lists all parameters with their values.
    These values must ALL be included in the generated instruction with [[ ]] markers.
    """
    lines = []
    trajectory = combo.get("trajectory", [])
    
    if "value_domain_samples" in combo:
        value_samples = combo["value_domain_samples"]
        if value_samples:
            lines.append("ALL of the following values MUST appear in your instruction wrapped with [[ ]]:")
            lines.append("")
            
            # Group by tool name
            tool_params: Dict[str, List[Tuple[str, Any]]] = {}
            for key, value in value_samples.items():
                if "." in key:
                    tool_name, param_name = key.split(".", 1)
                    if tool_name not in tool_params:
                        tool_params[tool_name] = []
                    tool_params[tool_name].append((param_name, value))
            
            # Format in trajectory order (if available), then remaining tools
            ordered_tools = list(trajectory) if trajectory else []
            for tool_name in tool_params.keys():
                if tool_name not in ordered_tools:
                    ordered_tools.append(tool_name)
            
            for tool_name in ordered_tools:
                if tool_name in tool_params:
                    lines.append(f"## {tool_name}")
                    for param_name, value in tool_params[tool_name]:
                        if isinstance(value, list):
                            # Mark list values - each element must be wrapped individually
                            lines.append(f"- {param_name}: {json.dumps(value)} (list - wrap each element)")
                        elif isinstance(value, bool):
                            # Mark boolean values
                            lines.append(f"- {param_name}: {str(value).lower()} (boolean)")
                        else:
                            lines.append(f"- {param_name}: {value}")
                    lines.append("")
    
    return "\n".join(lines) if lines else "No required values."


def format_instances(combo: Dict[str, Any]) -> str:
    """
    Format both entity instances and required values for the prompt.
    This is kept for backward compatibility but internally uses the split functions.
    """
    entity_text = format_entity_instances(combo)
    values_text = format_required_values(combo)
    
    return f"{entity_text}\n\n{values_text}"


def format_task_template(combo: Dict[str, Any]) -> str:
    """Format task template for the prompt."""
    template = combo.get("task_template", {})
    
    lines = []
    if "instruction" in template:
        lines.append(f"**Instruction:** {template['instruction']}")
    if "reason_for_call" in template:
        lines.append(f"**Reason for Call:** {template['reason_for_call']}")
    if "known_info" in template:
        known_info = template["known_info"]
        if isinstance(known_info, dict):
            lines.append(f"**Known Info:** {json.dumps(known_info)}")
        else:
            lines.append(f"**Known Info:** {known_info}")
    
    trajectory = combo.get("trajectory", [])
    if trajectory:
        lines.append(f"**Expected Trajectory:** {' -> '.join(trajectory)}")
    
    return "\n".join(lines) if lines else "No task template provided."


def format_tool_descriptions(trajectory: List[str], tool_list: List[Dict[str, Any]]) -> str:
    """
    Format tool descriptions for tools in the trajectory.
    
    Filters the full tool_list to only include tools that appear in the trajectory,
    and formats them with name, description, and parameter details.
    
    Args:
        trajectory: List of tool names in order of execution
        tool_list: Full list of tool definitions from tool_lists JSON
        
    Returns:
        Formatted string with tool descriptions
    """
    if not trajectory or not tool_list:
        return "No tool descriptions available."
    
    # Build a lookup by tool name
    tool_lookup: Dict[str, Dict[str, Any]] = {}
    for tool_def in tool_list:
        if "function" in tool_def:
            func = tool_def["function"]
            tool_name = func.get("name", "")
            if tool_name:
                tool_lookup[tool_name] = func
    
    lines = []
    for tool_name in trajectory:
        if tool_name not in tool_lookup:
            lines.append(f"## {tool_name}")
            lines.append("(No description available)")
            lines.append("")
            continue
        
        func = tool_lookup[tool_name]
        description = func.get("description", "No description")
        
        lines.append(f"## {tool_name}")
        lines.append(f"{description}")
        
        # Format parameters
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))
        
        if properties:
            lines.append("**Parameters:**")
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                req_marker = " (required)" if param_name in required else " (optional)"
                lines.append(f"- {param_name} ({param_type}){req_marker}: {param_desc}")
        
        lines.append("")
    
    return "\n".join(lines) if lines else "No tool descriptions available."


def create_instantiation_processor(
    client,
    blueprints: Dict[str, Any],
    server_name: str,
    tool_list: List[Dict[str, Any]],
    max_retries: int = 3,
):
    """Create a processor function for parallel instantiation with hallucination detection."""
    
    def process_combo(combo_with_id: Tuple[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Process a single combo to generate instantiated task and query."""
        combo_id, combo = combo_with_id
        trajectory_hash = combo.get("trajectory_hash", "unknown")
        trajectory = combo.get("trajectory", [])
        
        # Get server-specific blueprints
        server_blueprint = blueprints.get(server_name, {})
        blueprints_text = json.dumps(server_blueprint, indent=2, ensure_ascii=False)

        # Format inputs for prompt - now using split functions
        task_template_text = format_task_template(combo)
        entity_instances_text = format_entity_instances(combo)
        required_values_text = format_required_values(combo)
        trajectory_text = " -> ".join(trajectory) if trajectory else "N/A"
        
        # Format tool descriptions for only the tools in the trajectory
        tool_descriptions_text = format_tool_descriptions(trajectory, tool_list)
        
        # Build initial prompt with new structure
        base_prompt = TASK_INSTANTIATION_PROMPT.format(
            blueprints=blueprints_text,
            tool_descriptions=tool_descriptions_text,
            task_template=task_template_text,
            entity_instances=entity_instances_text,
            required_values=required_values_text,
            trajectory=trajectory_text,
        )
        
        result = None
        hallucinated: List[str] = []
        missing_values: List[str] = []
        last_error_type = None  # "hallucination" or "completeness"
        
        # Retry loop for hallucination and completeness validation
        for attempt in range(max_retries):
            try:
                # Build prompt (add retry feedback if not first attempt)
                if attempt == 0:
                    prompt = base_prompt
                elif last_error_type == "hallucination":
                    retry_feedback = HALLUCINATION_RETRY_PROMPT.format(
                        hallucinated_values=", ".join(hallucinated)
                    )
                    prompt = base_prompt + "\n\n" + retry_feedback
                elif last_error_type == "completeness":
                    retry_feedback = COMPLETENESS_RETRY_PROMPT.format(
                        missing_values=", ".join(missing_values)
                    )
                    prompt = base_prompt + "\n\n" + retry_feedback
                else:
                    prompt = base_prompt
                
                response = client.chat(query=prompt, model_type="textual")
                result = response.parse_json()
                
                if not result:
                    logger.warning(f"[{combo_id[:8]}] Empty LLM response on attempt {attempt + 1}")
                    continue
                
                # Step 1: Validate for hallucinations
                hallucinated = validate_no_hallucinations(result, combo)
                
                if hallucinated:
                    # Check if only "None" values - these can be tolerated
                    non_none_hallucinations = [h for h in hallucinated if h.lower() != "none"]
                    if non_none_hallucinations:
                        last_error_type = "hallucination"
                        logger.warning(
                            f"[{combo_id[:8]}] Hallucination on attempt {attempt + 1}: {hallucinated}"
                        )
                        continue
                
                # Step 2: Validate completeness (all required values present)
                missing_values = validate_completeness(result, combo)
                
                if missing_values:
                    last_error_type = "completeness"
                    logger.warning(
                        f"[{combo_id[:8]}] Missing values on attempt {attempt + 1}: {missing_values[:5]}..."
                    )
                    continue
                
                # Both validations passed!
                break
                
            except Exception as e:
                logger.warning(f"[{combo_id[:8]}] Error on attempt {attempt + 1}: {e}")
                continue
        
        # Check if we succeeded after all retries
        if result is None:
            logger.warning(f"[{combo_id[:8]}] Failed to get valid response after {max_retries} attempts")
            return None
        
        # Log if we still have issues after all retries
        if hallucinated:
            non_none_hallucinations = [h for h in hallucinated if h.lower() != "none"]
            if non_none_hallucinations:
                logger.warning(
                    f"[{combo_id[:8]}] Still has hallucinations after {max_retries} attempts: {hallucinated}"
                )
                return None
            else:
                logger.info(
                    f"[{combo_id[:8]}] Tolerating 'None' values after {max_retries} attempts"
                )
        
        if missing_values:
            # Log but continue - we'll use what we have
            logger.warning(
                f"[{combo_id[:8]}] Some values still missing after {max_retries} attempts: {missing_values[:3]}..."
            )
        
        try:
            # Extract fields from LLM response
            start_up_query = result.get("start_up_query", "")
            instruction = result.get("instruction", combo.get("task_template", {}).get("instruction", ""))
            reason_for_call = result.get("reason_for_call", combo.get("task_template", {}).get("reason_for_call", ""))
            
            # Set known_info to ONLY entity instances (not tool parameter mappings)
            known_info = entity_instances_text
            result["known_info"] = known_info
            
            # Generate unique task ID based on result content
            task_id = hashlib.md5(
                json.dumps(result, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:8]
            
            # Build output in S18 format
            query_output = {
                "id": task_id,
                "messages": [
                    {"role": "user", "content": start_up_query}
                ],
                "user_system_prompt": f"You are a user who needs help with: {reason_for_call}\n\nYour goal: {instruction}\n\nInformation you know: {known_info}",
                "task_info": {
                    "trajectory": trajectory,
                    "instruction": instruction,
                    "reason_for_call": reason_for_call,
                    "known_info": known_info,
                    "trajectory_hash": trajectory_hash,
                    "combo_id": combo_id,  # For resume tracking
                }
            }
            
            logger.debug(f"[{combo_id[:8]}] Generated query: {start_up_query[:50]}...")
            return query_output
            
        except Exception as e:
            logger.warning(f"[{combo_id[:8]}] Failed to process result: {e}")
            return None
    
    return process_combo


def append_queries_to_jsonl(queries: List[Dict], output_path: Path) -> None:
    """Append queries to JSONL file."""
    with open(output_path, "a") as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")


@step_handler("s17_task_instantiation", auto_retry=True)
def task_instantiation_step(state: WorkflowState) -> WorkflowState:
    """
    Instantiate tasks with entity values and generate startup queries.
    
    Supports stepwise retry:
    - Loads existing processed queries for resume
    - Tracks progress with update_step_progress()
    - Incrementally appends results to JSONL
    
    Output:
    - queries/{server}.jsonl
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    validated_dir = Path(state.validated_tasks_dir)
    queries_dir = ensure_dir(outputs_dir / "queries")
    
    # Load blueprints for context
    blueprints = {}
    # Try state path first, then fallback to default location
    blueprints_path = None
    if state.fixed_blueprints_path:
        blueprints_path = Path(state.fixed_blueprints_path)
    if not blueprints_path or not blueprints_path.exists():
        # Fallback to default location
        blueprints_path = outputs_dir / "blueprints.json"
    
    if blueprints_path.exists():
        blueprints_list = load_json(blueprints_path)
        # Index by MCP_server_name (blueprints.json is a list)
        if isinstance(blueprints_list, list):
            for bp in blueprints_list:
                server_name = bp.get("MCP_server_name")
                if server_name:
                    blueprints[server_name] = bp
        else:
            blueprints = blueprints_list  # Already a dict
        logger.info(f"Loaded {len(blueprints)} blueprints from {blueprints_path}")
    else:
        logger.warning(f"Blueprints file not found: {blueprints_path}")
    
    logger.info("=" * 60)
    logger.info("Starting Task Instantiation & Query Generation (Step 17)")
    logger.info("=" * 60)
    logger.info(f"Validated tasks dir: {validated_dir}")
    logger.info(f"Output dir: {queries_dir}")
    
    # Load existing queries for resume support
    existing_queries, processed_ids = load_existing_queries(queries_dir)
    logger.info(f"Found {len(processed_ids)} already processed combos")
    
    client = get_client()
    
    # Collect all combos across servers for progress tracking
    all_combos: List[Tuple[str, str, Dict[str, Any]]] = []  # (server_name, combo_id, combo)
    
    for server_dir in validated_dir.iterdir():
        if not server_dir.is_dir():
            continue
        
        server_name = server_dir.name
        combos_file = server_dir / "validated_combos.json"
        
        if not combos_file.exists():
            logger.warning(f"No validated_combos.json found for {server_name}")
            continue
        
        combos = load_json(combos_file)
        if not combos:
            continue
        
        for combo in combos:
            combo_id = get_combo_id(combo)
            all_combos.append((server_name, combo_id, combo))
    
    total_combos = len(all_combos)
    already_completed = len(processed_ids)
    
    # Filter out already processed combos
    to_process = [
        (server_name, combo_id, combo)
        for server_name, combo_id, combo in all_combos
        if combo_id not in processed_ids
    ]
    
    logger.info(f"Total combos: {total_combos}, Already processed: {already_completed}, To process: {len(to_process)}")
    
    if not to_process:
        logger.info("All combos already processed, skipping")
        state.queries_dir = str(queries_dir)
        state.update_step_progress(
            "s17_task_instantiation",
            total=total_combos,
            completed=total_combos,
            failed=0,
        )
        return state
    
    # Group combos by server for processing
    combos_by_server: Dict[str, List[Tuple[str, Dict]]] = {}
    for server_name, combo_id, combo in to_process:
        if server_name not in combos_by_server:
            combos_by_server[server_name] = []
        combos_by_server[server_name].append((combo_id, combo))
    
    # Process each server
    completed = already_completed
    failed = 0
    
    # Tool lists directory
    tool_lists_dir = outputs_dir / "tool_lists"
    
    for server_name, server_combos in combos_by_server.items():
        logger.info(f"Processing {server_name}: {len(server_combos)} combos to instantiate")
        
        # Load tool list for this server
        tool_list: List[Dict[str, Any]] = []
        
        # Check if Cross Domain (contains '_' in name)
        if "_" in server_name:
            # Cross Domain: split into individual domains and merge tool lists
            domains = server_name.split("_")
            for domain in domains:
                domain_tool_list_path = tool_lists_dir / f"{domain}.json"
                if domain_tool_list_path.exists():
                    domain_tools = load_json(domain_tool_list_path)
                    tool_list.extend(domain_tools)
                    logger.debug(f"  Loaded {len(domain_tools)} tools from {domain}")
                else:
                    logger.warning(f"  Tool list not found for domain: {domain}")
            logger.debug(f"  Total {len(tool_list)} tools merged for Cross Domain {server_name}")
        else:
            # Single Domain: load directly
            tool_list_path = tool_lists_dir / f"{server_name}.json"
            if tool_list_path.exists():
                tool_list = load_json(tool_list_path)
                logger.debug(f"  Loaded {len(tool_list)} tools from {tool_list_path}")
            else:
                logger.warning(f"  Tool list not found: {tool_list_path}")
        
        # Create processor with shared resources
        processor = create_instantiation_processor(
            client=client,
            blueprints=blueprints,
            server_name=server_name,
            tool_list=tool_list,
        )
        
        # Process combos in parallel
        results = parallel_process(
            items=server_combos,
            process_func=processor,
            description=f"Instantiating tasks for {server_name}",
        )
        
        # Filter valid results
        valid_results = [r for r in results if r is not None]
        failed_count = len(results) - len(valid_results)
        
        if valid_results:
            output_path = queries_dir / f"{server_name}.jsonl"
            append_queries_to_jsonl(valid_results, output_path)
            logger.info(f"  {server_name}: {len(valid_results)} queries generated, {failed_count} failed")
        else:
            logger.warning(f"  {server_name}: No queries generated")
        
        completed += len(valid_results)
        failed += failed_count
        
        # Update progress after each server
        state.update_step_progress(
            "s17_task_instantiation",
            total=total_combos,
            completed=completed,
            failed=failed,
        )
    
    # Update state
    state.queries_dir = str(queries_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Task Instantiation & Query Generation Complete")
    logger.info(f"  Total: {total_combos}, Completed: {completed}, Failed: {failed}")
    logger.info(f"  Output directory: {queries_dir}")
    logger.info("=" * 60)
    
    return state
