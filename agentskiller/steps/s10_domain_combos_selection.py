"""
Step 10: Domain Combos Selection

Select cross-domain combinations for task generation.

Input: blueprints.json
Output: cross_domain_templates/_combinations.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from ..models.state import WorkflowState
from ..models.tasks import CrossDomainCombination
from ..config.settings import get_settings
from ..prompts import DOMAIN_COMBINATION_PROMPT
from .base import step_handler, save_json, load_json, ensure_dir, get_client

logger = logging.getLogger(__name__)


@step_handler("s10_domain_combos_selection", auto_retry=True)
def domain_combos_selection_step(state: WorkflowState) -> WorkflowState:
    """
    Select cross-domain combinations.
    
    Process:
    1. Load blueprints
    2. Group by core_entity
    3. Select valid combinations (2-3 domains)
    4. Filter by shared entities
    
    Output:
    - cross_domain_templates/_combinations.json
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    num_combinations = settings.workflow.cross_domain_combinations
    
    # Load blueprints
    blueprints_data = load_json(Path(state.blueprints_path))
    # Handle both list format and dict with "blueprints" key
    if isinstance(blueprints_data, list):
        blueprints = blueprints_data
    else:
        blueprints = blueprints_data.get("blueprints", [])
    
    logger.info(f"Selecting combinations from {len(blueprints)} blueprints")
    
    # Setup output directory
    cross_domain_dir = ensure_dir(outputs_dir / "cross_domain_combinations")
    
    # Group blueprints by core entity
    by_core_entity = {}
    for bp in blueprints:
        core = bp.get("core_entity", "")
        if core:
            if core not in by_core_entity:
                by_core_entity[core] = []
            by_core_entity[core].append(bp)
    
    # Filter out groups with less than 2 domains
    valid_groups = {
        core: bps for core, bps in by_core_entity.items()
        if len(bps) >= 2
    }
    
    if not valid_groups:
        logger.warning("No valid groups with >= 2 domains found")
        selected = []
    else:
        # Calculate target combinations per group using largest remainder method
        total_domains = sum(len(bps) for bps in valid_groups.values())
        group_targets = {}
        remainders = {}
        
        for core, bps in valid_groups.items():
            exact = num_combinations * len(bps) / total_domains
            group_targets[core] = int(exact)
            remainders[core] = exact - int(exact)
        
        # Distribute remaining slots to groups with largest remainders
        allocated = sum(group_targets.values())
        remaining_slots = num_combinations - allocated
        sorted_cores = sorted(remainders.keys(), key=lambda c: remainders[c], reverse=True)
        for i in range(remaining_slots):
            group_targets[sorted_cores[i % len(sorted_cores)]] += 1
        
        logger.info(f"Valid groups: {len(valid_groups)}, target allocation: {group_targets}")
        
        # Get LLM client
        client = get_client()
        
        # Use LLM to select combinations within each group
        all_combinations = []
        for core, bps in valid_groups.items():
            target = group_targets[core]
            if target == 0:
                continue
            
            # Build blueprint info for LLM prompt
            blueprints_for_prompt = [
                {
                    "MCP_server_name": bp.get("MCP_server_name"),
                    "description": bp.get("description", ""),
                    "core_entity": bp.get("core_entity", ""),
                    "peripheral_entities": bp.get("peripheral_entities", []),
                }
                for bp in bps
            ]
            
            prompt = DOMAIN_COMBINATION_PROMPT.format(
                core_entity=core,
                blueprints=json.dumps(blueprints_for_prompt, indent=2, ensure_ascii=False),
                n_combinations=target,
            )
            
            try:
                response = client.chat(query=prompt, model_type="textual")
                llm_combos = response.parse_json()
                
                if not isinstance(llm_combos, list):
                    logger.warning(f"Invalid LLM response for {core}, expected list")
                    continue
                
                # Convert LLM output to CrossDomainCombination
                for combo_servers in llm_combos[:target]:
                    if not isinstance(combo_servers, list) or len(combo_servers) < 2:
                        continue
                    
                    # Find blueprints for each server in the combo
                    combo_bps = []
                    for server_name in combo_servers:
                        bp = next((b for b in bps if b.get("MCP_server_name") == server_name), None)
                        if bp:
                            combo_bps.append(bp)
                    
                    if len(combo_bps) < 2:
                        logger.warning(f"Could not find blueprints for combo: {combo_servers}")
                        continue
                    
                    # Calculate shared entities
                    entity_sets = [
                        set([bp.get("core_entity")] + bp.get("peripheral_entities", []))
                        for bp in combo_bps
                    ]
                    shared = entity_sets[0]
                    for es in entity_sets[1:]:
                        shared = shared & es
                    
                    combo = CrossDomainCombination(
                        domains=[bp.get("description", "") for bp in combo_bps],
                        core_entities=[bp.get("core_entity", "") for bp in combo_bps],
                        servers=[bp.get("MCP_server_name") for bp in combo_bps],
                        fused_name="_".join(bp.get("MCP_server_name") for bp in combo_bps),
                        shared_entities=list(shared),
                    )
                    all_combinations.append(combo.model_dump())
                
                logger.info(f"Group '{core}': LLM selected {len([c for c in llm_combos[:target] if isinstance(c, list) and len(c) >= 2])} combinations")
                
            except Exception as e:
                logger.warning(f"Failed to get LLM combinations for {core}: {e}")
                continue
        
        selected = all_combinations
    
    # Save combinations
    output_path = cross_domain_dir / "_combinations.json"
    save_json({"combinations": selected}, output_path)
    
    state.cross_domain_combinations_path = str(output_path)
    state.cross_domain_templates_dir = str(cross_domain_dir)
    
    logger.info(f"Selected {len(selected)} cross-domain combinations")
    return state

