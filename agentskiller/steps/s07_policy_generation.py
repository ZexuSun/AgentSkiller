"""
Step 7: Policy Generation

Generate and validate domain policies for MCP servers.

Input: fixed_blueprints.json, database_summary/
Output: policies/*.md
"""

import json
import logging
from pathlib import Path

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import (
    POLICY_STRUCTURE_INSTRUCTIONS,
    DOMAIN_POLICY_PROMPT,
    POLICY_VALIDATION_PROMPT,
)
from .base import (
    step_handler, save_json, load_json, get_client,
    parallel_process, ensure_dir, WorkflowBlockEditor
)

logger = logging.getLogger(__name__)


def load_database_summary(summary_dir: Path, blueprint: dict) -> str:
    """Load database summaries for all entities in the blueprint."""
    summaries = []
    
    # Get all entities
    entities = [blueprint.get("core_entity")]
    entities.extend(blueprint.get("peripheral_entities", []))
    
    # Load entity summaries
    entity_summary_dir = summary_dir / "entities"
    for entity in entities:
        if entity:
            summary_path = entity_summary_dir / f"{entity}.md"
            if summary_path.exists():
                summaries.append(f"### Entity: {entity}\n{summary_path.read_text()}")
    
    # Load relationship summaries
    server_name = blueprint.get("MCP_server_name", "")
    rel_summary_dir = summary_dir / "relationships" / server_name
    if rel_summary_dir.exists():
        for rel_file in rel_summary_dir.glob("*.md"):
            summaries.append(f"### Relationship: {rel_file.stem}\n{rel_file.read_text()}")
    
    return "\n\n".join(summaries) if summaries else "No database summary available."


@step_handler("s07_policy_generation", auto_retry=True)
def policy_generation_step(state: WorkflowState) -> WorkflowState:
    """
    Generate domain policies for MCP servers.
    
    Process:
    1. Load blueprints and database summaries
    2. Generate policy for each blueprint using LLM
    3. Validate policy against blueprint/schema (optional)
    4. Save policies
    
    Output:
    - policies/*.md: Domain policies for each server
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    step_config = settings.get_step_config("s07_policy_generation")
    
    simulation_time = step_config.get(
        "simulation_time",
        settings.workflow.simulation_time
    )
    enable_validation = step_config.get("enable_validation", True)
    
    # Load blueprints
    blueprints_path = state.blueprints_path
    blueprints_data = load_json(Path(blueprints_path))
    if isinstance(blueprints_data, list):
        blueprints = blueprints_data
    else:
        blueprints = blueprints_data.get("blueprints", [])
    
    # Setup directories
    policies_dir = ensure_dir(outputs_dir / "policies")
    database_summary_dir = Path(state.database_summary_dir) if state.database_summary_dir else None
    
    logger.info(f"Generating policies for {len(blueprints)} MCP servers")
    
    client = get_client()
    editor = WorkflowBlockEditor()
    
    # Track processed policies
    processed = set()
    for f in policies_dir.glob("*.md"):
        processed.add(f.stem)
    
    to_process = [
        bp for bp in blueprints
        if bp.get("MCP_server_name") not in processed
    ]
    
    if not to_process:
        logger.info("All policies already generated")
        state.policies_dir = str(policies_dir)
        return state
    
    def generate_policy(blueprint: dict) -> dict:
        """Generate policy for a single server."""
        server_name = blueprint.get("MCP_server_name", "Unknown")
        
        # Get function names for policy structure
        function_names = [f.get("name", "") for f in blueprint.get("functions", [])]
        
        # Load database summary if available
        summary = "N/A - Use blueprint structure"
        if database_summary_dir and database_summary_dir.exists():
            summary = load_database_summary(database_summary_dir, blueprint)
        
        # Generate policy
        policy_prompt = DOMAIN_POLICY_PROMPT.format(
            blueprint=json.dumps(blueprint, indent=2),
            summary=summary,
            example_policy=open("agentskiller/airline.md", "r").read(),
            simulation_time=simulation_time,
            policy_structure_instructions=POLICY_STRUCTURE_INSTRUCTIONS,
            tool_names=", ".join(function_names)
        )
        
        try:
            policy_response = client.chat(query=policy_prompt, model_type="textual")
            policy_content = policy_response.content
            
            # Validate policy if enabled
            if enable_validation and database_summary_dir:
                validation_prompt = POLICY_VALIDATION_PROMPT.format(
                    blueprint=json.dumps(blueprint, indent=2),
                    database_summary=summary,
                    simulation_time=simulation_time,
                    policy=policy_content
                )
                
                validation_response = client.chat(query=validation_prompt, model_type="textual")
                
                # If issues found, apply fixes
                if "NO_ISSUES_FOUND" not in validation_response.content:
                    # Apply SEARCH/REPLACE fixes
                    policy_path = policies_dir / f"{server_name}.md"
                    policy_path.write_text(policy_content)
                    
                    result = editor.apply_edits_from_response(
                        policy_path, 
                        validation_response.content
                    )
                    
                    if result.success_count > 0:
                        policy_content = policy_path.read_text()
                        logger.info(f"Applied {result.success_count} fixes to {server_name} policy")
            
            # Save policy
            policy_path = policies_dir / f"{server_name}.md"
            policy_path.write_text(policy_content)
            
            return {"success": True, "server_name": server_name}
            
        except Exception as e:
            logger.warning(f"Failed to generate policy for {server_name}: {e}")
            return {"success": False, "server_name": server_name, "error": str(e)}
    
    # Process blueprints
    results = parallel_process(
        items=to_process,
        process_func=generate_policy,
        description="Generating policies",
    )
    
    # Count successes
    success_count = sum(1 for r in results if r and r.get("success"))
    
    state.policies_dir = str(policies_dir)
    
    # Update progress
    state.update_step_progress(
        "s07_policy_generation",
        total=len(blueprints),
        completed=len(processed) + success_count
    )
    
    logger.info(f"Policy generation complete: {success_count}/{len(to_process)} policies")
    return state