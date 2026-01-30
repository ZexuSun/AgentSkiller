"""
Workflow runner with checkpointing and resume support.
"""

import logging
from typing import Optional
from datetime import datetime
import uuid
from pathlib import Path
import json

from langgraph.checkpoint.memory import MemorySaver

from .workflow import (
    create_workflow_graph,
    WORKFLOW_STEPS,
    CROSS_DOMAIN_STEPS,
    get_step_info,
)
from ..models.state import WorkflowState, StepStatus, StepResult
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


def create_workflow_with_checkpointer():
    """Create workflow graph with memory checkpointer."""
    workflow = create_workflow_graph()
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def run_workflow(
    start_step: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> WorkflowState:
    """
    Run the complete workflow.
    
    Args:
        start_step: Step to start from (default: beginning)
        checkpoint_id: Resume from a specific checkpoint
        config: Additional configuration
    
    Returns:
        Final workflow state
    """
    settings = get_settings()
    
    # Create compiled graph
    app = create_workflow_with_checkpointer()
    
    # Initialize or load state
    if checkpoint_id:
        initial_state = _load_checkpoint(checkpoint_id)
        logger.info(f"Resuming from checkpoint: {checkpoint_id}")
    else:
        initial_state = WorkflowState(
            workflow_id=str(uuid.uuid4())[:8],
            started_at=datetime.now().isoformat(),
            config=config or {},
        )
    
    # Initialize state from existing outputs
    initial_state = _initialize_state_from_outputs(initial_state)
    
    # Handle start_step
    if start_step:
        if start_step not in WORKFLOW_STEPS:
            raise ValueError(
                f"Unknown step: {start_step}. Valid steps: {WORKFLOW_STEPS}"
            )
        
        # Check dependencies
        if not initial_state.can_resume_from(start_step):
            raise ValueError(
                f"Cannot resume from {start_step} - previous steps not completed"
            )
        
        # Mark previous steps as completed
        _mark_previous_steps_completed(initial_state, start_step)
    
    # Run the workflow
    logger.info(f"Starting workflow: {initial_state.workflow_id}")
    
    try:
        thread_config = {"configurable": {"thread_id": initial_state.workflow_id}}
        final_state = app.invoke(initial_state, thread_config)
        
        logger.info("Workflow completed successfully")
        return final_state
        
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user")
        _save_checkpoint(initial_state)
        raise
    except Exception as e:
        logger.exception("Workflow failed")
        _save_checkpoint(initial_state)
        raise


def run_single_step(
    step_name: str, 
    state: Optional[WorkflowState] = None
) -> WorkflowState:
    """
    Run a single workflow step.
    
    Args:
        step_name: Name of the step to run
        state: Optional existing state to use
    
    Returns:
        Updated workflow state
    """
    if step_name not in WORKFLOW_STEPS:
        raise ValueError(f"Unknown step: {step_name}. Valid steps: {WORKFLOW_STEPS}")
    
    # Import step function
    step_func = _get_step_function(step_name)
    
    # Create state if not provided
    if state is None:
        state = WorkflowState(
            workflow_id=str(uuid.uuid4())[:8],
            started_at=datetime.now().isoformat(),
        )
        state = _initialize_state_from_outputs(state)
    
    # Run the step
    settings = get_settings()
    max_retries = settings.workflow.max_step_retries
    
    for attempt in range(max_retries):
        try:
            result_state = step_func(state)
            
            if result_state.is_step_complete(step_name):
                logger.info(f"Step {step_name} completed successfully")
                return result_state
            
            if not settings.workflow.auto_retry:
                logger.warning(f"Step {step_name} incomplete, auto_retry disabled")
                break
            
            remaining = result_state.get_remaining_items(step_name)
            logger.warning(
                f"Step {step_name} incomplete ({remaining} remaining), "
                f"retry {attempt + 1}/{max_retries}"
            )
            state = result_state
            
        except Exception as e:
            logger.exception(f"Step {step_name} failed: {e}")
            if attempt < max_retries - 1 and settings.workflow.auto_retry:
                continue
            raise
    
    return state


def get_workflow_status(state: WorkflowState) -> dict:
    """Get a summary of workflow status."""
    settings = get_settings()
    
    # Determine which steps are relevant
    task_mode = settings.workflow.task_mode
    relevant_steps = WORKFLOW_STEPS
    if task_mode == "single":
        relevant_steps = [s for s in WORKFLOW_STEPS if s not in CROSS_DOMAIN_STEPS]
    
    # Build status for each step
    step_statuses = {}
    for step in relevant_steps:
        if step in state.steps:
            result = state.steps[step]
            step_statuses[step] = {
                "status": result.status.value,
                "duration": result.duration_seconds,
                "progress": f"{result.completed_items}/{result.total_items}" 
                           if result.total_items > 0 else "N/A",
                "errors": result.errors[:3] if result.errors else [],  # First 3 errors
            }
        else:
            step_statuses[step] = {"status": "pending"}
    
    # Calculate overall progress
    completed_count = sum(
        1 for step in relevant_steps
        if step in state.steps and state.steps[step].status == StepStatus.COMPLETED
    )
    
    return {
        "workflow_id": state.workflow_id,
        "started_at": state.started_at,
        "current_step": state.current_step,
        "task_mode": task_mode,
        "progress": f"{completed_count}/{len(relevant_steps)}",
        "steps": step_statuses,
        "completed": completed_count == len(relevant_steps),
        "processed_domains": len(state.processed_domains),
        "failed_domains": len(state.failed_domains),
    }


def list_steps(include_cross_domain: bool = True) -> list:
    """List all workflow steps with descriptions."""
    steps = []
    for step_name in WORKFLOW_STEPS:
        info = get_step_info(step_name)
        
        if not include_cross_domain and info["is_cross_domain_only"]:
            continue
        
        steps.append({
            "name": step_name,
            "phase": info["phase"],
            "description": info["description"],
            "cross_domain_only": info["is_cross_domain_only"],
        })
    
    return steps


# ============================================================================
# Helper Functions
# ============================================================================

def _get_step_function(step_name: str):
    """Get the function for a workflow step."""
    from ..steps import (
        domain_expansion_step,
        entity_extraction_step,
        entity_graph_step,
        blueprint_generation_step,
        tool_list_formulation_step,
        database_generation_step,
        policy_generation_step,
        tool_graph_generation_step,
        mcp_server_implementation_step,
        domain_combos_selection_step,
        trajectory_fusion_step,
        database_fusion_step,
        policy_merge_step,
        task_template_generation_step,
        instance_combos_selection_step,
        task_filtering_step,
        task_instantiation_step,
    )
    
    step_functions = {
        "s01_domain_expansion": domain_expansion_step,
        "s02_entity_extraction": entity_extraction_step,
        "s03_entity_graph": entity_graph_step,
        "s04_blueprint_generation": blueprint_generation_step,
        "s05_tool_list_formulation": tool_list_formulation_step,
        "s06_database_generation": database_generation_step,
        "s07_policy_generation": policy_generation_step,
        "s08_tool_graph_generation": tool_graph_generation_step,
        "s09_mcp_server_implementation": mcp_server_implementation_step,
        "s10_domain_combos_selection": domain_combos_selection_step,
        "s11_trajectory_fusion": trajectory_fusion_step,
        "s12_database_fusion": database_fusion_step,
        "s13_policy_merge": policy_merge_step,
        "s14_task_template_generation": task_template_generation_step,
        "s15_instance_combos_selection": instance_combos_selection_step,
        "s16_task_filtering": task_filtering_step,
        "s17_task_instantiation": task_instantiation_step,
    }
    
    return step_functions[step_name]


def _initialize_state_from_outputs(state: WorkflowState) -> WorkflowState:
    """
    Initialize state paths from existing output files.
    
    This allows running individual steps without running the full workflow.
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # Map of state attributes to expected file paths
    path_mappings = {
        "domain_topics_path": outputs_dir / "domain_topics.json",
        "entities_path": outputs_dir / "entities.json",
        "entity_graph_path": outputs_dir / "entity_graph.json",
        "blueprints_path": outputs_dir / "blueprints.json",
        "tool_lists_dir": outputs_dir / "tool_lists",
        "databases_dir": outputs_dir / "database",
        "database_summary_dir": outputs_dir / "database_summary",
        "policies_dir": outputs_dir / "policies",
        "tool_graphs_dir": outputs_dir / "tool_graphs",
        "mcp_servers_dir": outputs_dir / "mcp_servers",
        "unit_tests_dir": outputs_dir / "unit_tests",
        "task_templates_dir": outputs_dir / "task_templates",
        "cross_domain_templates_dir": outputs_dir / "cross_domain_templates",
        "instantiated_tasks_dir": outputs_dir / "instantiated_tasks",
        "validated_tasks_dir": outputs_dir / "validated_tasks",
        "queries_dir": outputs_dir / "queries",
    }
    
    for attr, path in path_mappings.items():
        if path.exists():
            setattr(state, attr, str(path))
            logger.debug(f"Initialized {attr} from existing output: {path}")
    
    # Mark steps as completed based on existing outputs
    step_output_requirements = {
        "s01_domain_expansion": [outputs_dir / "domain_topics.json"],
        "s02_entity_extraction": [outputs_dir / "entities.json"],
        "s03_entity_graph": [outputs_dir / "entity_graph.json"],
        "s04_blueprint_generation": [outputs_dir / "blueprints.json"],
        "s05_tool_list_formulation": [outputs_dir / "tool_lists"],
        "s06_database_generation": [outputs_dir / "database"],
        "s07_policy_generation": [outputs_dir / "policies"],
        "s08_tool_graph_generation": [outputs_dir / "tool_graphs"],
        "s09_mcp_server_implementation": [outputs_dir / "mcp_servers"],
        "s10_domain_combos_selection": [outputs_dir / "cross_domain_templates" / "_combinations.json"],
        "s11_trajectory_fusion": [outputs_dir / "cross_domain_templates"],
        "s12_database_fusion": [outputs_dir / "database" / "scripts" / "cross_domain"],
        "s13_policy_merge": [outputs_dir / "policies"],
        "s14_task_template_generation": [outputs_dir / "task_templates"],
        "s15_instance_combos_selection": [outputs_dir / "combinations"],
        "s16_task_filtering": [outputs_dir / "validated_tasks"],
        "s17_task_instantiation": [outputs_dir / "queries"],
    }
    
    for step_name, required_paths in step_output_requirements.items():
        if all(p.exists() for p in required_paths):
            if step_name not in state.steps:
                state.steps[step_name] = StepResult(
                    step_name=step_name,
                    status=StepStatus.COMPLETED
                )
                logger.debug(f"Marked {step_name} as completed (outputs exist)")
    
    return state


def _mark_previous_steps_completed(state: WorkflowState, target_step: str) -> None:
    """Mark all steps before target_step as completed."""
    step_idx = WORKFLOW_STEPS.index(target_step)
    
    for prev_step in WORKFLOW_STEPS[:step_idx]:
        if prev_step not in state.steps:
            state.steps[prev_step] = StepResult(
                step_name=prev_step,
                status=StepStatus.COMPLETED
            )


def _save_checkpoint(state: WorkflowState) -> Path:
    """Save workflow state as a checkpoint."""
    settings = get_settings()
    checkpoint_dir = settings.paths.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_{state.workflow_id}_{timestamp}.json"
    
    state_dict = state.to_checkpoint()
    
    with open(checkpoint_path, "w") as f:
        json.dump(state_dict, f, indent=2, default=str)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def _load_checkpoint(checkpoint_id: str) -> WorkflowState:
    """Load workflow state from a checkpoint."""
    settings = get_settings()
    checkpoint_dir = settings.paths.checkpoints_dir
    
    # Try to find checkpoint file
    pattern = f"checkpoint_{checkpoint_id}*.json"
    matches = list(checkpoint_dir.glob(pattern))
    
    if not matches:
        raise ValueError(f"Checkpoint not found: {checkpoint_id}")
    
    # Use most recent if multiple matches
    checkpoint_path = max(matches, key=lambda p: p.stat().st_mtime)
    
    with open(checkpoint_path) as f:
        data = json.load(f)
    
    return WorkflowState.from_checkpoint(data)
