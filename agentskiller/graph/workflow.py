"""
LangGraph workflow definition with 17 steps and branching logic.

The workflow supports two paths:
- Single Domain: Steps 1-9 → 14-17
- Cross Domain: Steps 1-9 → 10-13 → 14-17
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from ..models.state import WorkflowState, StepStatus
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Step Definitions
# ============================================================================

# All workflow steps in order
WORKFLOW_STEPS = [
    # Phase 1: Domain & Entity (Steps 1-3)
    "s01_domain_expansion",
    "s02_entity_extraction",
    "s03_entity_graph",
    
    # Phase 2: Blueprint & Tools (Steps 4-5)
    "s04_blueprint_generation",
    "s05_tool_list_formulation",
    
    # Phase 3: Implementation (Steps 6-9)
    "s06_database_generation",
    "s07_policy_generation",
    "s08_tool_graph_generation",
    "s09_mcp_server_implementation",
    
    # Phase 4: Cross-Domain (Steps 10-13) - Optional
    "s10_domain_combos_selection",
    "s11_trajectory_fusion",
    "s12_database_fusion",
    "s13_policy_merge",
    
    # Phase 5: Task Generation (Steps 14-17)
    "s14_task_template_generation",
    "s15_instance_combos_selection",
    "s16_task_filtering",
    "s17_task_instantiation",
]

# Steps that are only for cross-domain mode
CROSS_DOMAIN_STEPS = [
    "s10_domain_combos_selection",
    "s11_trajectory_fusion",
    "s12_database_fusion",
    "s13_policy_merge",
]

# Step dependencies map
STEP_DEPENDENCIES = {
    "s01_domain_expansion": [],
    "s02_entity_extraction": ["s01_domain_expansion"],
    "s03_entity_graph": ["s02_entity_extraction"],
    "s04_blueprint_generation": ["s03_entity_graph"],
    "s05_tool_list_formulation": ["s04_blueprint_generation"],
    "s06_database_generation": ["s05_tool_list_formulation"],
    "s07_policy_generation": ["s06_database_generation"],
    "s08_tool_graph_generation": ["s07_policy_generation"],
    "s09_mcp_server_implementation": ["s08_tool_graph_generation"],
    "s10_domain_combos_selection": ["s09_mcp_server_implementation"],
    "s11_trajectory_fusion": ["s10_domain_combos_selection"],
    "s12_database_fusion": ["s11_trajectory_fusion"],
    "s13_policy_merge": ["s12_database_fusion"],
    "s14_task_template_generation": ["s09_mcp_server_implementation"],  # or s13 for cross-domain
    "s15_instance_combos_selection": ["s14_task_template_generation"],
    "s16_task_filtering": ["s15_instance_combos_selection"],
    "s17_task_instantiation": ["s16_task_filtering"],
}


# ============================================================================
# Routing Functions
# ============================================================================

def should_continue(state: WorkflowState) -> Literal["continue", "end"]:
    """Determine if workflow should continue or end."""
    if state.current_step:
        step_result = state.steps.get(state.current_step)
        if step_result and step_result.status == StepStatus.FAILED:
            logger.error(f"Workflow stopping due to failed step: {state.current_step}")
            return "end"
    return "continue"


def route_after_server_implementation(state: WorkflowState) -> str:
    """
    Route after MCP server implementation.
    
    Decides whether to:
    - Go to cross-domain steps (s10)
    - Skip to task generation (s14)
    """
    settings = get_settings()
    task_mode = settings.workflow.task_mode
    
    if task_mode in ("cross_domain", "both"):
        logger.info("Routing to cross-domain path (s10_domain_combos_selection)")
        return "s10_domain_combos_selection"
    else:
        logger.info("Routing to single-domain path (s14_task_template_generation)")
        return "s14_task_template_generation"


def route_after_policy_merge(state: WorkflowState) -> str:
    """Route after policy merge to task template generation."""
    return "s14_task_template_generation"


def route_next_step(state: WorkflowState) -> str:
    """Route to the next step based on current progress."""
    settings = get_settings()
    
    # Check if dry run mode
    if settings.workflow.dry_run:
        logger.info("Dry run mode - stopping after current step")
        return END
    
    # Get completed steps
    completed = [
        name for name, result in state.steps.items()
        if result.status == StepStatus.COMPLETED
    ]
    
    # Determine which steps to include based on task_mode
    task_mode = settings.workflow.task_mode
    steps_to_run = list(WORKFLOW_STEPS)
    
    if task_mode == "single":
        # Skip cross-domain steps
        steps_to_run = [s for s in steps_to_run if s not in CROSS_DOMAIN_STEPS]
    
    # Find next step
    for step in steps_to_run:
        if step not in completed:
            return step
    
    # All steps completed
    return END


# ============================================================================
# Graph Construction
# ============================================================================

def create_workflow_graph() -> StateGraph:
    """
    Create the LangGraph workflow graph with 17 steps.
    
    Workflow structure:
    
    s01 → s02 → s03 → s04 → s05 → s06 → s07 → s08 → s09
                                                      ↓
                    ┌─────────────────────────────────┴─────────────────────┐
                    ↓ (cross_domain)                                         ↓ (single)
                   s10 → s11 → s12 → s13                                    ↓
                                      ↓                                      ↓
                                      └──────────────┬───────────────────────┘
                                                     ↓
                                                    s14 → s15 → s16 → s17 → END
    """
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
    
    # Create the graph with our state type
    workflow = StateGraph(WorkflowState)
    
    # ========================================================================
    # Add nodes for each step
    # ========================================================================
    
    # Phase 1: Domain & Entity
    workflow.add_node("s01_domain_expansion", domain_expansion_step)
    workflow.add_node("s02_entity_extraction", entity_extraction_step)
    workflow.add_node("s03_entity_graph", entity_graph_step)
    
    # Phase 2: Blueprint & Tools
    workflow.add_node("s04_blueprint_generation", blueprint_generation_step)
    workflow.add_node("s05_tool_list_formulation", tool_list_formulation_step)
    
    # Phase 3: Implementation
    workflow.add_node("s06_database_generation", database_generation_step)
    workflow.add_node("s07_policy_generation", policy_generation_step)
    workflow.add_node("s08_tool_graph_generation", tool_graph_generation_step)
    workflow.add_node("s09_mcp_server_implementation", mcp_server_implementation_step)
    
    # Phase 4: Cross-Domain (optional)
    workflow.add_node("s10_domain_combos_selection", domain_combos_selection_step)
    workflow.add_node("s11_trajectory_fusion", trajectory_fusion_step)
    workflow.add_node("s12_database_fusion", database_fusion_step)
    workflow.add_node("s13_policy_merge", policy_merge_step)
    
    # Phase 5: Task Generation
    workflow.add_node("s14_task_template_generation", task_template_generation_step)
    workflow.add_node("s15_instance_combos_selection", instance_combos_selection_step)
    workflow.add_node("s16_task_filtering", task_filtering_step)
    workflow.add_node("s17_task_instantiation", task_instantiation_step)
    
    # ========================================================================
    # Set the entry point
    # ========================================================================
    workflow.set_entry_point("s01_domain_expansion")
    
    # ========================================================================
    # Add edges for sequential execution
    # ========================================================================
    
    # Phase 1: Domain & Entity (linear)
    workflow.add_edge("s01_domain_expansion", "s02_entity_extraction")
    workflow.add_edge("s02_entity_extraction", "s03_entity_graph")
    
    # Phase 2: Blueprint & Tools (linear)
    workflow.add_edge("s03_entity_graph", "s04_blueprint_generation")
    workflow.add_edge("s04_blueprint_generation", "s05_tool_list_formulation")
    
    # Phase 3: Implementation (linear)
    workflow.add_edge("s05_tool_list_formulation", "s06_database_generation")
    workflow.add_edge("s06_database_generation", "s07_policy_generation")
    workflow.add_edge("s07_policy_generation", "s08_tool_graph_generation")
    workflow.add_edge("s08_tool_graph_generation", "s09_mcp_server_implementation")
    
    # ========================================================================
    # Conditional routing after MCP server implementation
    # ========================================================================
    workflow.add_conditional_edges(
        "s09_mcp_server_implementation",
        route_after_server_implementation,
        {
            "s10_domain_combos_selection": "s10_domain_combos_selection",
            "s14_task_template_generation": "s14_task_template_generation",
        }
    )
    
    # Phase 4: Cross-Domain (linear, if taken)
    workflow.add_edge("s10_domain_combos_selection", "s11_trajectory_fusion")
    workflow.add_edge("s11_trajectory_fusion", "s12_database_fusion")
    workflow.add_edge("s12_database_fusion", "s13_policy_merge")
    workflow.add_edge("s13_policy_merge", "s14_task_template_generation")
    
    # Phase 5: Task Generation (linear)
    workflow.add_edge("s14_task_template_generation", "s15_instance_combos_selection")
    workflow.add_edge("s15_instance_combos_selection", "s16_task_filtering")
    workflow.add_edge("s16_task_filtering", "s17_task_instantiation")
    workflow.add_edge("s17_task_instantiation", END)
    
    return workflow


def get_step_info(step_name: str) -> dict:
    """Get information about a workflow step."""
    phase_map = {
        "s01": "Domain & Entity",
        "s02": "Domain & Entity",
        "s03": "Domain & Entity",
        "s04": "Blueprint & Tools",
        "s05": "Blueprint & Tools",
        "s06": "Implementation",
        "s07": "Implementation",
        "s08": "Implementation",
        "s09": "Implementation",
        "s10": "Cross-Domain",
        "s11": "Cross-Domain",
        "s12": "Cross-Domain",
        "s13": "Cross-Domain",
        "s14": "Task Generation",
        "s15": "Task Generation",
        "s16": "Task Generation",
        "s17": "Task Generation",
    }
    
    step_descriptions = {
        "s01_domain_expansion": "Expand seed domains into diverse sub-domains",
        "s02_entity_extraction": "Extract entities from domain topics",
        "s03_entity_graph": "Build entity relationship graph",
        "s04_blueprint_generation": "Generate MCP server blueprints",
        "s05_tool_list_formulation": "Formulate tool lists from blueprints",
        "s06_database_generation": "Generate mock databases",
        "s07_policy_generation": "Generate domain policies",
        "s08_tool_graph_generation": "Generate tool execution graphs",
        "s09_mcp_server_implementation": "Implement and test MCP servers",
        "s10_domain_combos_selection": "Select cross-domain combinations",
        "s11_trajectory_fusion": "Fuse tool trajectories across domains",
        "s12_database_fusion": "Fuse databases for cross-domain tasks",
        "s13_policy_merge": "Merge domain policies",
        "s14_task_template_generation": "Generate task templates",
        "s15_instance_combos_selection": "Select instance combinations",
        "s16_task_filtering": "Filter valid tasks",
        "s17_task_instantiation": "Instantiate tasks and generate queries",
    }
    
    prefix = step_name[:3]
    
    return {
        "name": step_name,
        "phase": phase_map.get(prefix, "Unknown"),
        "description": step_descriptions.get(step_name, ""),
        "dependencies": STEP_DEPENDENCIES.get(step_name, []),
        "is_cross_domain_only": step_name in CROSS_DOMAIN_STEPS,
    }
