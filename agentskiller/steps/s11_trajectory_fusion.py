"""
Step 11: Trajectory Fusion

Fuse tool trajectories across domains.

Input: tool_graphs/*.json, _combinations.json
Output: cross_domain_templates/*.json (trajectories)
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import TRAJECTORY_FUSION_PROMPT
from ..core.parallel import parallel_process_with_retry
from .base import step_handler, save_json, load_json, ensure_dir, get_client

logger = logging.getLogger(__name__)


# =============================================================================
# Single Domain Validation Helpers
# =============================================================================

def compute_trajectory_hash(trajectory: List[str]) -> str:
    """
    Compute a unique hash for a trajectory.
    
    Args:
        trajectory: List of tool names in order
        
    Returns:
        A short hash string (12 chars) that uniquely identifies the trajectory
    """
    trajectory_str = "_".join(trajectory)
    return hashlib.md5(trajectory_str.encode()).hexdigest()[:12]


def get_validated_trajectory_hashes(validated_tasks_dir: Path) -> Dict[str, Set[str]]:
    """
    Collect validated trajectory hashes from Single Domain validated_tasks.
    
    Args:
        validated_tasks_dir: Path to outputs/validated_tasks/
        
    Returns:
        Dict mapping domain_name to set of validated trajectory hashes
        Example: {"AgFieldLabPlanner": {"bfffd9342a5a", "a371b586c377", ...}}
    """
    result = {}
    
    if not validated_tasks_dir.exists():
        logger.warning(f"validated_tasks directory not found: {validated_tasks_dir}")
        return result
    
    for domain_dir in validated_tasks_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        # Skip Cross Domain directories (they contain underscores in name)
        if "_" in domain_dir.name:
            continue
            
        combos_file = domain_dir / "validated_combos.json"
        if combos_file.exists():
            try:
                combos = load_json(combos_file)
                hashes = {
                    c.get("trajectory_hash") 
                    for c in combos 
                    if c.get("trajectory_hash")
                }
                result[domain_dir.name] = hashes
                logger.debug(f"Loaded {len(hashes)} validated hashes for {domain_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to load {combos_file}: {e}")
    
    return result


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
    domain_segments: Dict[str, List[str]] = {}
    
    for tool_call in trajectory:
        if "." in tool_call:
            server_name, tool_name = tool_call.split(".", 1)
            domain_segments.setdefault(server_name, []).append(tool_name)
    
    return domain_segments


def is_cross_domain_feasible(
    trajectory: List[str],
    validated_hashes: Dict[str, Set[str]]
) -> Tuple[bool, str]:
    """
    Check if a Cross Domain trajectory is feasible based on Single Domain results.
    
    A Cross Domain trajectory is feasible if ALL its Single Domain segments
    have been validated (i.e., their hashes exist in validated_tasks).
    
    Args:
        trajectory: Cross Domain trajectory in format ["Server.tool", ...]
        validated_hashes: Dict mapping domain to set of validated hashes
        
    Returns:
        Tuple of (is_feasible, reason_if_not)
    """
    segments = parse_cross_domain_trajectory(trajectory)
    
    for domain, tools in segments.items():
        segment_hash = compute_trajectory_hash(tools)
        
        if domain not in validated_hashes:
            return False, f"No validated results for domain '{domain}'"
        
        if segment_hash not in validated_hashes[domain]:
            return False, f"Segment {domain}/{segment_hash} not validated"
    
    return True, ""


# =============================================================================
# Single Combination Processing
# =============================================================================

def process_single_combination(
    combo: Dict[str, Any],
    tool_graphs_dir: Path,
    cross_domain_dir: Path,
    client: Any,
    validated_hashes: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Any]:
    """
    Process a single domain combination to generate fused trajectories.
    
    Args:
        combo: Combination dict with servers, fused_name, shared_entities
        tool_graphs_dir: Directory containing tool graph JSON files
        cross_domain_dir: Output directory for fused trajectory files
        client: LLM client instance
        validated_hashes: Optional dict mapping domain to validated trajectory hashes.
                          If provided, filters out infeasible trajectories.
    
    Returns:
        Dict with processing result
    
    Raises:
        ValueError: If insufficient tool graphs available
        Exception: If LLM call or parsing fails
    """
    servers = combo.get("servers", [])
    fused_name = combo.get("fused_name", "unknown")
    shared = combo.get("shared_entities", [])
    
    # Load tool graphs for each server
    graphs = {}
    for server in servers:
        graph_path = tool_graphs_dir / f"{server}.json"
        if graph_path.exists():
            graphs[server] = load_json(graph_path)
    
    if len(graphs) < 2:
        raise ValueError(f"Insufficient tool graphs for {fused_name}: found {len(graphs)}, need >= 2")
    
    # Generate fused trajectories using LLM
    blueprint_str = json.dumps({"servers": servers, "shared_entities": shared}, indent=2)
    trajectories_str = "\n".join(f"## {s}\n{json.dumps(g, indent=2)}" for s, g in graphs.items())
    
    prompt = TRAJECTORY_FUSION_PROMPT.format(
        blueprint=blueprint_str,
        tool_call_trajectories=trajectories_str
    )
    
    response = client.chat(query=prompt, model_type="textual")
    trajectories = response.parse_json()
    
    # =========================================================================
    # Filter infeasible trajectories based on Single Domain validation results
    # =========================================================================
    if validated_hashes:
        original_count = len(trajectories)
        feasible_trajectories = []
        
        for traj_item in trajectories:
            trajectory = traj_item.get("trajectory", [])
            is_feasible, reason = is_cross_domain_feasible(trajectory, validated_hashes)
            if is_feasible:
                feasible_trajectories.append(traj_item)
            else:
                logger.debug(f"Filtered infeasible trajectory in {fused_name}: {reason}")
        
        if len(feasible_trajectories) < original_count:
            logger.info(
                f"{fused_name}: Filtered {original_count} -> {len(feasible_trajectories)} "
                f"feasible trajectories"
            )
        
        trajectories = feasible_trajectories
    
    # Save result
    output_path = cross_domain_dir / f"{fused_name}.json"
    save_json({
        "combination": combo,
        "trajectories": trajectories
    }, output_path)
    
    logger.info(f"Successfully fused trajectories for {fused_name}: {len(trajectories)} trajectories")
    
    return {
        "fused_name": fused_name,
        "servers": servers,
        "output_path": str(output_path),
        "trajectory_count": len(trajectories),
    }


# =============================================================================
# Step Handler
# =============================================================================

@step_handler("s11_trajectory_fusion", auto_retry=True)
def trajectory_fusion_step(state: WorkflowState) -> WorkflowState:
    """
    Fuse tool trajectories across domains.
    
    Process:
    1. Load combinations and tool graphs
    2. Filter out already-processed combinations
    3. Process pending combinations in parallel with retry
    4. Update progress for step-wise retry
    
    Output:
    - cross_domain_templates/{combo}.json
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # Load combinations - check state first, fallback to expected location
    if state.cross_domain_combinations_path:
        combos_path = Path(state.cross_domain_combinations_path)
    else:
        combos_path = outputs_dir / "cross_domain_combinations" / "_combinations.json"
        if combos_path.exists():
            state.cross_domain_combinations_path = str(combos_path)
        else:
            raise FileNotFoundError(
                f"Combinations file not found. Expected at {combos_path}. "
                "Please run step s10_domain_combos_selection first."
            )
    
    combos_data = load_json(combos_path)
    combinations = combos_data.get("combinations", [])
    
    # Get tool graphs directory - check state first, fallback to expected location
    if state.tool_graphs_dir:
        tool_graphs_dir = Path(state.tool_graphs_dir)
    else:
        tool_graphs_dir = outputs_dir / "tool_graphs"
        if tool_graphs_dir.exists():
            state.tool_graphs_dir = str(tool_graphs_dir)
        else:
            raise FileNotFoundError(
                f"Tool graphs directory not found. Expected at {tool_graphs_dir}. "
                "Please run earlier workflow steps first."
            )
    
    # Get/create cross domain templates directory
    if state.cross_domain_templates_dir:
        cross_domain_dir = Path(state.cross_domain_templates_dir)
    else:
        cross_domain_dir = outputs_dir / "cross_domain_combinations"
        state.cross_domain_templates_dir = str(cross_domain_dir)
    ensure_dir(cross_domain_dir)
    
    # =========================================================================
    # Incremental Progress Tracking
    # =========================================================================
    # Determine pending combinations (not yet processed)
    pending_combos = []
    already_completed = 0
    
    for combo in combinations:
        fused_name = combo.get("fused_name", "unknown")
        output_path = cross_domain_dir / f"{fused_name}.json"
        if output_path.exists():
            already_completed += 1
        else:
            pending_combos.append(combo)
    
    total_combos = len(combinations)
    logger.info(f"Trajectory fusion: {total_combos} total, {already_completed} already done, {len(pending_combos)} pending")
    
    # Early exit if all done
    if not pending_combos:
        logger.info("All combinations already processed")
        state.update_step_progress(
            "s11_trajectory_fusion",
            total=total_combos,
            completed=total_combos,
            failed=0,
        )
        return state
    
    # =========================================================================
    # Load Single Domain Validation Results for Filtering
    # =========================================================================
    validated_tasks_dir = outputs_dir / "validated_tasks"
    validated_hashes = get_validated_trajectory_hashes(validated_tasks_dir)
    
    if validated_hashes:
        total_hashes = sum(len(h) for h in validated_hashes.values())
        logger.info(
            f"Loaded {total_hashes} validated trajectory hashes from "
            f"{len(validated_hashes)} domains for feasibility filtering"
        )
    else:
        logger.warning(
            "No validated trajectory hashes found. "
            "Skipping Single Domain feasibility filtering."
        )
    
    # =========================================================================
    # Parallel Processing with Retry
    # =========================================================================
    client = get_client()
    
    # Create processor function with captured context
    def process_combo(combo: Dict[str, Any]) -> Dict[str, Any]:
        return process_single_combination(
            combo=combo,
            tool_graphs_dir=tool_graphs_dir,
            cross_domain_dir=cross_domain_dir,
            client=client,
            validated_hashes=validated_hashes,
        )
    
    # Process pending combinations in parallel with retry
    successes, failures = parallel_process_with_retry(
        items=pending_combos,
        process_func=process_combo,
        max_retries=3,
        description="Fusing trajectories",
    )
    
    # =========================================================================
    # Update Progress and Handle Failures
    # =========================================================================
    completed_count = already_completed + len(successes)
    failed_count = len(failures)
    
    state.update_step_progress(
        "s11_trajectory_fusion",
        total=total_combos,
        completed=completed_count,
        failed=failed_count,
    )
    
    logger.info(f"Trajectory fusion: {completed_count}/{total_combos} completed, {failed_count} failed")
    
    # Log failed combinations
    if failures:
        for combo, error in failures:
            fused_name = combo.get("fused_name", "unknown")
            logger.error(f"Failed to fuse {fused_name}: {error}")
        
        # Raise to trigger step-level retry
        raise RuntimeError(
            f"Failed to process {failed_count} combinations. Re-run to retry."
        )
    
    logger.info("Trajectory fusion complete")
    return state
