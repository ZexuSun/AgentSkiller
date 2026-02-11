"""
Step retry mechanism for agentskiller.

Provides auto-retry functionality for workflow steps:
- Automatic retry until step completely succeeds
- Incremental progress saving
- Skip already-completed items on retry
"""

import logging
from functools import wraps
from typing import Callable, Optional, Any, TYPE_CHECKING
from datetime import datetime

from ..config.settings import get_settings

if TYPE_CHECKING:
    from ..models.state import WorkflowState

logger = logging.getLogger(__name__)


def step_handler(
    step_name: str,
    auto_retry: bool = True,
    max_retries: Optional[int] = None,
):
    """
    Decorator for workflow step functions with automatic retry.
    
    Features:
    - Tracks step status (started, completed, failed)
    - Auto-retries failed items until all succeed
    - Saves checkpoint after each attempt
    - Skips already-completed items on retry
    
    Args:
        step_name: Unique name for this step
        auto_retry: Whether to auto-retry on partial failure
        max_retries: Maximum retry attempts (default from config)
    
    Usage:
        @step_handler("domain_expansion", auto_retry=True)
        def domain_expansion_step(state: WorkflowState) -> WorkflowState:
            # Process items, updating state.completed_items
            return state
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(state: "WorkflowState") -> "WorkflowState":
            settings = get_settings()
            retries = max_retries or settings.workflow.max_step_retries
            should_retry = auto_retry and settings.workflow.auto_retry
            
            logger.info(f"Starting step: {step_name}")
            state.mark_step_started(step_name)
            
            for attempt in range(retries):
                try:
                    # Run the step
                    result_state = func(state)
                    
                    # Check if step is complete
                    if result_state.is_step_complete(step_name):
                        result_state.mark_step_completed(step_name)
                        logger.info(f"Completed step: {step_name}")
                        
                        # Save checkpoint
                        if settings.workflow.enable_checkpoints:
                            _save_checkpoint(result_state)
                        
                        return result_state
                    
                    # Step incomplete - retry if enabled
                    if not should_retry:
                        logger.warning(
                            f"Step {step_name} incomplete but auto_retry disabled"
                        )
                        break
                    
                    remaining = result_state.get_remaining_items(step_name)
                    logger.warning(
                        f"Step {step_name} incomplete ({remaining} items remaining), "
                        f"retry {attempt + 1}/{retries}"
                    )
                    
                    # Save intermediate checkpoint
                    if settings.workflow.enable_checkpoints:
                        _save_checkpoint(result_state)
                    
                    # Update state for retry (keeps completed items)
                    state = result_state
                    
                except Exception as e:
                    logger.exception(f"Step {step_name} failed with error: {e}")
                    state.mark_step_failed(step_name, [str(e)])
                    
                    if attempt < retries - 1 and should_retry:
                        logger.info(f"Retrying step {step_name} after error")
                        continue
                    raise
            
            # Max retries exceeded
            logger.error(
                f"Step {step_name} did not complete after {retries} attempts"
            )
            return state
        
        # Add metadata to function
        wrapper._step_name = step_name
        wrapper._auto_retry = auto_retry
        
        return wrapper
    return decorator


def _save_checkpoint(state: "WorkflowState") -> None:
    """Save workflow state checkpoint."""
    from pathlib import Path
    import json
    
    settings = get_settings()
    checkpoint_dir = settings.paths.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_{state.workflow_id}_{timestamp}.json"
    
    # Serialize state
    state_dict = state.model_dump(exclude={"entities", "blueprints"})
    
    with open(checkpoint_path, "w") as f:
        json.dump(state_dict, f, indent=2, default=str)
    
    logger.debug(f"Saved checkpoint: {checkpoint_path}")


class StepRetryManager:
    """
    Manager for step retries across workflow execution.
    
    Tracks:
    - Which items have been processed per step
    - Which items failed and need retry
    - Overall step completion status
    """
    
    def __init__(self, step_name: str, total_items: int):
        self.step_name = step_name
        self.total_items = total_items
        self.completed_items: set = set()
        self.failed_items: dict = {}  # item_id -> error
        self.retry_counts: dict = {}  # item_id -> count
    
    def mark_completed(self, item_id: Any) -> None:
        """Mark an item as successfully completed."""
        self.completed_items.add(item_id)
        self.failed_items.pop(item_id, None)
    
    def mark_failed(self, item_id: Any, error: str) -> None:
        """Mark an item as failed."""
        self.failed_items[item_id] = error
        self.retry_counts[item_id] = self.retry_counts.get(item_id, 0) + 1
    
    def get_pending_items(self, items: list, max_item_retries: int = 3) -> list:
        """
        Get items that need processing.
        
        Returns items that:
        - Have not been completed
        - Have not exceeded max retries
        """
        return [
            item for item in items
            if item not in self.completed_items
            and self.retry_counts.get(item, 0) < max_item_retries
        ]
    
    def is_complete(self) -> bool:
        """Check if all items are completed."""
        return len(self.completed_items) >= self.total_items
    
    def get_remaining_count(self) -> int:
        """Get count of remaining items."""
        return self.total_items - len(self.completed_items)
    
    def get_summary(self) -> dict:
        """Get summary of step progress."""
        return {
            "step_name": self.step_name,
            "total": self.total_items,
            "completed": len(self.completed_items),
            "failed": len(self.failed_items),
            "remaining": self.get_remaining_count(),
            "is_complete": self.is_complete(),
        }


def with_item_retry(
    max_retries: int = 3,
    on_error: Optional[Callable[[Any, Exception], None]] = None,
):
    """
    Decorator for item-level retry within a step.
    
    Args:
        max_retries: Maximum retries per item
        on_error: Optional callback for errors
    
    Usage:
        @with_item_retry(max_retries=3)
        def process_entity(entity: EntityInfo) -> dict:
            # Process single entity
            return result
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(item: Any, *args, **kwargs) -> Optional[Any]:
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(item, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Item processing failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if on_error:
                        on_error(item, e)
            
            logger.error(f"Item processing failed after {max_retries} attempts: {last_error}")
            return None
        
        return wrapper
    return decorator
