"""
Base utilities for workflow steps.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..config.settings import get_settings
from ..core.retry import step_handler
from ..core.parallel import parallel_process, parallel_process_with_retry
from ..core.block_editor import WorkflowBlockEditor
from ..core.llm_client import get_client

logger = logging.getLogger(__name__)

# Re-export commonly used utilities
__all__ = [
    "step_handler",
    "parallel_process",
    "parallel_process_with_retry",
    "WorkflowBlockEditor",
    "get_client",
    "save_json",
    "load_json",
    "ensure_dir",
]


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug(f"Saved JSON: {path}")


def load_json(path: Path) -> Any:
    """Load JSON from a file."""
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
