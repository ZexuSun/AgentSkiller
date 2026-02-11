"""
CrossDomain Workflow v2 - Refactored data synthesis pipeline.

This module provides a modular, maintainable workflow for automated
data synthesis with improved LLM client, block editing, and retry mechanisms.
"""

__version__ = "2.0.0"

from .graph.workflow import WORKFLOW_STEPS
from .graph.runner import run_workflow, run_single_step

__all__ = [
    "WORKFLOW_STEPS",
    "run_workflow",
    "run_single_step",
]
