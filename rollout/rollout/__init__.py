"""
Rollout - A framework for automated Multi-Turn Tool Call data annotation using LLM APIs.

Key Features:
- Unified LLM API calls via LiteLLM (supports 100+ providers)
- Custom model registration for internal/self-hosted APIs
- Automatic tool discovery and registration
- Checkpoint-based context management for fault tolerance
- Support for both single-turn and multi-turn pipelines
"""

from rollout.core.agent import Agent
from rollout.core.user import SimulatedUser
from rollout.core.checkpoint import CheckpointManager
from rollout.core.pipeline import (
    Pipeline,
    clear_reasoning_content,
    copy_messages_without_reasoning,
)
from rollout.core.models import (
    register_model,
    register_models_from_config,
    load_models_config,
    register_internal_deepseek,
    register_internal_openai,
)
from rollout.tools import discover_tools, get_tool

__version__ = "0.2.0"
__all__ = [
    # Core classes
    "Agent",
    "SimulatedUser", 
    "CheckpointManager",
    "Pipeline",
    # Model registration
    "register_model",
    "register_models_from_config",
    "load_models_config",
    "register_internal_deepseek",
    "register_internal_openai",
    # Tool utilities
    "discover_tools",
    "get_tool",
    # DeepSeek V3.2 reasoning utilities
    "clear_reasoning_content",
    "copy_messages_without_reasoning",
]

