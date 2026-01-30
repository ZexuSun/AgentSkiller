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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
    # Phase 2: Process Cross Domain tasks (using Single Domain codes)
    # =========================================================================
    
    logger.info("=" * 60)
    logger.info("Phase 2: Processing Cross Domain trajectories (using code fusion)")
    logger.info("=" * 60)
    
    previous_failed_count = None
    round_number = 0
    
    while True:
        round_number += 1
        
        # Get pending Cross Domain tasks
        pending_tasks = get_pending_tasks(cross_domain_tasks, combos_dir)
        
        if not pending_tasks:
            logger.info("All Cross Domain trajectories processed!")
            break
        
        logger.info(f"Cross Domain Round {round_number}: Processing {len(pending_tasks)} pending trajectories")
        
        # Process Cross Domain trajectories in parallel
        def process_cross_domain_task(task):
            return process_and_save_cross_domain_trajectory(
                trajectory_task=task,
                client=client,
                settings=step_config,
                current_time=current_time,
                outputs_dir=outputs_dir,
            )
        
        results = parallel_process(
            items=pending_tasks,
            process_func=process_cross_domain_task,
            description=f"Cross Domain Round {round_number}: Processing trajectories",
        )
        
        # Count successes and failures
        success_count = sum(1 for r in results if r is not None and isinstance(r, int) and r > 0)
        current_failed_count = len(pending_tasks) - success_count
        
        logger.info(f"Cross Domain Round {round_number}: {success_count} succeeded, {current_failed_count} failed")
        
        # Check convergence
        if current_failed_count == 0:
            break
        
        if previous_failed_count is not None and previous_failed_count > 0:
            delta = previous_failed_count - current_failed_count
            progress_rate = delta / previous_failed_count
            
            if progress_rate < min_progress_rate:
                logger.info(f"Cross Domain converged: progress_rate {progress_rate:.1%} < {min_progress_rate:.0%}")
                break
        
        previous_failed_count = current_failed_count
    
    # =========================================================================
    # Final: Collect skipped trajectories and update state
    # =========================================================================
    
    # Collect all failed tasks
    failed_tasks = get_pending_tasks(all_tasks, combos_dir)
    
    if failed_tasks:
        save_skipped_trajectories(combos_dir, failed_tasks)
        logger.info(f"Skipped {len(failed_tasks)} trajectories with no valid combos")
        
        # Mark step as completed despite skipped trajectories
        state.mark_step_completed("s15_instance_combos_selection")
    
    state.combinations_dir = str(combos_dir)
    
    # Count total combos generated (grouped by type)
    single_domain_combos = 0
    cross_domain_combos = 0
    
    for server_dir in combos_dir.iterdir():
        if server_dir.is_dir() and not server_dir.name.startswith("_"):
            combo_count = len(list(server_dir.glob("*.json")))
            if is_cross_domain(server_dir.name):
                cross_domain_combos += combo_count
            else:
                single_domain_combos += combo_count
    
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
    logger.info(f"  Cross Domain: {cross_domain_combos} combinations")
    logger.info(f"  Total: {total_combos} combinations")
    logger.info("=" * 60)
    
    return state
    