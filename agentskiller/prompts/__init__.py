"""
LLM prompt templates for workflow steps.

Organized by step number for easy navigation.
"""

# =============================================================================
# Step 01: Domain Expansion
# =============================================================================
from .s01_prompts import (
    SEED_DOMAINS,
    DOMAIN_EXPANSION_PROMPT,
    DOMAIN_DIVERSITY_CHECK_PROMPT,
)

# =============================================================================
# Step 02: Entity Extraction
# =============================================================================
from .s02_prompts import (
    ENTITY_EXTRACTION_PROMPT,
)

# =============================================================================
# Step 03: Entity Graph
# =============================================================================
from .s03_prompts import (
    ENTITY_RELATION_PROMPT,
)

# =============================================================================
# Step 04: Blueprint Generation
# =============================================================================
from .s04_prompts import (
    BLUEPRINT_OUTLINE_PROMPT,
    BLUEPRINT_DETAIL_PROMPT,
    BLUEPRINT_DETAIL_FEEDBACK_PROMPT,
    BLUEPRINT_FIXUP_PROMPT,
)

# =============================================================================
# Step 05: Tool List Formulation
# =============================================================================
from .s05_prompts import (
    TOOL_LIST_EXTRACTION_PROMPT,
)

# =============================================================================
# Step 06: Database Generation
# =============================================================================
from .s06_prompts import (
    ENTITY_DATABASE_PROMPT,
    RISKY_ATTRIBUTES_PROMPT,
    RELATIONSHIP_DATABASE_PROMPT,
    DATABASE_SUMMARY_PROMPT,
    CONSTRAINT_IDENTIFICATION_PROMPT,
)

# =============================================================================
# Step 07: Policy Generation
# =============================================================================
from .s07_prompts import (
    POLICY_STRUCTURE_INSTRUCTIONS,
    DOMAIN_POLICY_PROMPT,
    POLICY_VALIDATION_PROMPT,
    COMMON_POLICY_WRAPPER,
)

# =============================================================================
# Step 08: Tool Graph Generation
# =============================================================================
from .s08_prompts import (
    TOOL_GRAPH_PROMPT,
)

# =============================================================================
# Step 09: MCP Server Implementation
# =============================================================================
from .s09_prompts import (
    MCP_SERVER_IMPLEMENT_PROMPT,
    UNIT_TEST_PROMPT,
    CODE_FIX_PROMPT,
    ERROR_TRACE,
)

# =============================================================================
# Step 10: Domain Combos Selection (Cross-Domain)
# =============================================================================
from .s10_prompts import (
    DOMAIN_COMBINATION_PROMPT,
)

# =============================================================================
# Step 11: Trajectory Fusion (Cross-Domain)
# =============================================================================
from .s11_prompts import (
    TRAJECTORY_FUSION_PROMPT,
)

# =============================================================================
# Step 12: Database Fusion (Cross-Domain)
# =============================================================================
# from .s12_prompts import (
#     CONSTRAINT_ENTITY_DATABASE_PROMPT,
#     CONSTRAINT_RELATIONSHIP_DATABASE_PROMPT,
# )

# =============================================================================
# Step 13: Policy Merge (Cross-Domain)
# =============================================================================
from .s13_prompts import (
    POLICY_MERGE_STRUCTURE_INSTRUCTIONS,
    POLICY_MERGE_PROMPT,
)

# =============================================================================
# Step 14: Task Template Generation
# =============================================================================
from .s14_prompts import (
    TOOL_PRECONDITIONS_PROMPT,
    TASK_TEMPLATE_PROMPT,
    TASK_TEMPLATE_JUDGE_PROMPT,
)

# =============================================================================
# Step 15: Instance Combos Selection
# =============================================================================
from .s15_prompts import (
    INSTANCE_ASSIGNMENT_EXAMPLES,
    CONSTRAINT_ANALYSIS_PROMPT,
    SELECTION_CODE_PROMPT,
    PROPOSE_CONSTRAINTS_PROMPT,
    CODE_FUSION_PROMPT,
    IMPLICIT_DEPENDENCY_ANALYSIS_PROMPT,
    COMBINATION_CREATION_PROMPT,
    ERROR_ANALYSIS_PROMPT,
)

# =============================================================================
# Step 16: Task Filtering
# =============================================================================
from .s16_prompts import (
    TRAJECTORY_VALIDATION_ERROR_ANALYSIS_PROMPT,
)

# =============================================================================
# Step 17: Task Instantiation & Query Generation
# =============================================================================
from .s17_prompts import (
    TASK_INSTANTIATION_PROMPT,
    HALLUCINATION_RETRY_PROMPT,
    COMPLETENESS_RETRY_PROMPT,
)

# =============================================================================
# Shared/Common Prompts
# =============================================================================

# Policy markers for structured parsing
POLICY_MARKERS = {
    "policy_start": "<!-- POLICY_START -->",
    "policy_end": "<!-- POLICY_END -->",
    "global_rules_end": "<!-- GLOBAL_RULES_END -->",
    "tool_start": "<!-- TOOL: {tool_name} -->",
    "tool_end": "<!-- TOOL_END: {tool_name} -->",
}

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Step 01
    "SEED_DOMAINS",
    "DOMAIN_EXPANSION_PROMPT",
    "DOMAIN_DIVERSITY_CHECK_PROMPT",
    # Step 02
    "ENTITY_EXTRACTION_PROMPT",
    # Step 03
    "ENTITY_RELATION_PROMPT",
    # Step 04
    "BLUEPRINT_OUTLINE_PROMPT",
    "BLUEPRINT_DETAIL_PROMPT",
    "BLUEPRINT_DETAIL_FEEDBACK_PROMPT",
    "BLUEPRINT_FIXUP_PROMPT",
    # Step 05
    "TOOL_LIST_EXTRACTION_PROMPT",
    # Step 06
    "ENTITY_DATABASE_PROMPT",
    "RISKY_ATTRIBUTES_PROMPT",
    "RELATIONSHIP_DATABASE_PROMPT",
    "DATABASE_SUMMARY_PROMPT",
    "CONSTRAINT_IDENTIFICATION_PROMPT",
    # Step 07
    "POLICY_STRUCTURE_INSTRUCTIONS",
    "DOMAIN_POLICY_PROMPT",
    "POLICY_VALIDATION_PROMPT",
    "COMMON_POLICY_WRAPPER",
    # Step 08
    "TOOL_GRAPH_PROMPT",
    # Step 09
    "MCP_SERVER_IMPLEMENT_PROMPT",
    "UNIT_TEST_PROMPT",
    "CODE_FIX_PROMPT",
    "ERROR_TRACE",
    # Step 10
    "DOMAIN_COMBINATION_PROMPT",
    # Step 11
    "TRAJECTORY_FUSION_PROMPT",
    "CROSS_ENTITY_CONSTRAINT_PROMPT",
    # Step 12
    "CONSTRAINT_ENTITY_DATABASE_PROMPT",
    "CONSTRAINT_RELATIONSHIP_DATABASE_PROMPT",
    # Step 13
    "POLICY_MERGE_STRUCTURE_INSTRUCTIONS",
    "POLICY_MERGE_PROMPT",
    # Step 14
    "TOOL_PRECONDITIONS_PROMPT",
    "TASK_TEMPLATE_PROMPT",
    "TASK_TEMPLATE_JUDGE_PROMPT",
    # Step 15
    "INSTANCE_ASSIGNMENT_EXAMPLES",
    "CONSTRAINT_ANALYSIS_PROMPT",
    "SELECTION_CODE_PROMPT",
    "PROPOSE_CONSTRAINTS_PROMPT",
    "CODE_FUSION_PROMPT",
    "IMPLICIT_DEPENDENCY_ANALYSIS_PROMPT",
    "COMBINATION_CREATION_PROMPT",
    # Step 16
    "TRAJECTORY_VALIDATION_ERROR_ANALYSIS_PROMPT",
    # Step 17
    "TASK_INSTANTIATION_PROMPT",
    "HALLUCINATION_RETRY_PROMPT",
    "COMPLETENESS_RETRY_PROMPT",
    # Shared
    "POLICY_MARKERS",
]
