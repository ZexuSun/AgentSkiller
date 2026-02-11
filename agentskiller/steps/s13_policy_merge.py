"""
Step 13: Policy Merge

Merge domain policies for cross-domain tasks.

Input: policies/*.md, _combinations.json
Output: policies/{fused}.md
"""

import logging
from pathlib import Path
from typing import Dict, Any

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import POLICY_MERGE_STRUCTURE_INSTRUCTIONS, POLICY_MERGE_PROMPT
from ..core.policy_parser import get_tool_names_from_policy
from ..core.parallel import parallel_process_with_retry
from .base import step_handler, save_json, load_json, get_client

logger = logging.getLogger(__name__)


# =============================================================================
# Single Combination Processing
# =============================================================================

def process_single_combination(
    combo: Dict[str, Any],
    policies_dir: Path,
    client: Any,
) -> Dict[str, Any]:
    """
    Process a single domain combination to merge policies.
    
    Args:
        combo: Combination dict with servers, fused_name
        policies_dir: Directory containing policy markdown files
        client: LLM client instance
    
    Returns:
        Dict with processing result
    
    Raises:
        ValueError: If insufficient policies available
        Exception: If LLM call fails
    """
    fused_name = combo.get("fused_name", "unknown")
    servers = combo.get("servers", [])
    
    # Load policies for each server
    policies = {}
    for server in servers:
        policy_path = policies_dir / f"{server}.md"
        if policy_path.exists():
            policies[server] = policy_path.read_text()
    
    if len(policies) < 2:
        raise ValueError(f"Insufficient policies for {fused_name}: found {len(policies)}, need >= 2")
    
    # Get all tool names from the policies for reference
    all_tool_names = []
    for policy_content in policies.values():
        all_tool_names.extend(get_tool_names_from_policy(policy_content))
    
    # Merge policies using LLM
    domain_names = ", ".join(servers)
    policies_text = "\n\n".join(f"## {s} Policy\n{p}" for s, p in policies.items())
    tool_names = ", ".join(set(all_tool_names))
    
    prompt = POLICY_MERGE_PROMPT.format(
        domain_names=domain_names,
        policies=policies_text,
        policy_merge_structure_instructions=POLICY_MERGE_STRUCTURE_INSTRUCTIONS,
        tool_names=tool_names
    )
    
    response = client.chat(query=prompt, model_type="textual")
    
    # Save merged policy
    merged_path = policies_dir / f"{fused_name}.md"
    merged_path.write_text(response.content)
    
    logger.info(f"Successfully merged policies for {fused_name}")
    
    return {
        "fused_name": fused_name,
        "servers": servers,
        "output_path": str(merged_path),
    }


# =============================================================================
# Step Handler
# =============================================================================


@step_handler("s13_policy_merge", auto_retry=True)
def policy_merge_step(state: WorkflowState) -> WorkflowState:
    """
    Merge domain policies.
    
    Process:
    1. Load combinations and policies
    2. Filter out already-processed combinations
    3. Process pending combinations in parallel with retry
    4. Update progress for step-wise retry
    
    Output:
    - policies/{fused}.md
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # Load combinations
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
    
    # Get policies directory - check state first, fallback to expected location
    if state.policies_dir:
        policies_dir = Path(state.policies_dir)
    else:
        policies_dir = outputs_dir / "policies"
        if policies_dir.exists():
            state.policies_dir = str(policies_dir)
        else:
            raise FileNotFoundError(
                f"Policies directory not found. Expected at {policies_dir}. "
                "Please run step s07_policy_generation first."
            )
    
    # =========================================================================
    # Incremental Progress Tracking
    # =========================================================================
    # Determine pending combinations (not yet processed)
    pending_combos = []
    already_completed = 0
    
    for combo in combinations:
        fused_name = combo.get("fused_name", "unknown")
        merged_path = policies_dir / f"{fused_name}.md"
        if merged_path.exists():
            already_completed += 1
        else:
            pending_combos.append(combo)
    
    total_combos = len(combinations)
    logger.info(f"Policy merge: {total_combos} total, {already_completed} already done, {len(pending_combos)} pending")
    
    # Early exit if all done
    if not pending_combos:
        logger.info("All combinations already processed")
        state.update_step_progress(
            "s13_policy_merge",
            total=total_combos,
            completed=total_combos,
            failed=0,
        )
        state.merged_policies_dir = str(policies_dir)
        return state
    
    # =========================================================================
    # Parallel Processing with Retry
    # =========================================================================
    client = get_client()
    
    # Create processor function with captured context
    def process_combo(combo: Dict[str, Any]) -> Dict[str, Any]:
        return process_single_combination(
            combo=combo,
            policies_dir=policies_dir,
            client=client,
        )
    
    # Process pending combinations in parallel with retry
    successes, failures = parallel_process_with_retry(
        items=pending_combos,
        process_func=process_combo,
        max_retries=3,
        description="Merging policies",
    )
    
    # =========================================================================
    # Update Progress and Handle Failures
    # =========================================================================
    completed_count = already_completed + len(successes)
    failed_count = len(failures)
    
    state.update_step_progress(
        "s13_policy_merge",
        total=total_combos,
        completed=completed_count,
        failed=failed_count,
    )
    
    state.merged_policies_dir = str(policies_dir)
    
    logger.info(f"Policy merge: {completed_count}/{total_combos} completed, {failed_count} failed")
    
    # Log failed combinations
    if failures:
        for combo, error in failures:
            fused_name = combo.get("fused_name", "unknown")
            logger.error(f"Failed to merge policies for {fused_name}: {error}")
        
        # Raise to trigger step-level retry
        raise RuntimeError(
            f"Failed to process {failed_count} combinations. Re-run to retry."
        )
    
    logger.info("Policy merge complete")
    return state

