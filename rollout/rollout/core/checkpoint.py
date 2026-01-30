"""
Checkpoint Manager for fault-tolerant trajectory processing.

Provides save/restore functionality for conversation trajectories, enabling
resume from interruptions caused by network errors or other transient failures.
"""

import os
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from filelock import FileLock

logger = logging.getLogger(__name__)


class TrajectoryStatus(str, Enum):
    """Status of a trajectory in the checkpoint system."""
    PENDING = "pending"           # Not yet started
    IN_PROGRESS = "in_progress"   # Currently being processed
    COMPLETED = "completed"       # Successfully finished
    FAILED = "failed"             # Failed with non-recoverable error
    INTERRUPTED = "interrupted"   # Interrupted (can resume)


class ErrorType(str, Enum):
    """Classification of errors for recovery decisions."""
    RECOVERABLE = "recoverable"       # Network timeout, rate limit, etc.
    NON_RECOVERABLE = "non_recoverable"  # Invalid data, logic error, etc.


@dataclass
class CheckpointData:
    """
    Data structure for a trajectory checkpoint.
    
    Stores all information needed to resume a trajectory from an interruption.
    """
    trajectory_id: str
    status: TrajectoryStatus
    messages: List[Dict[str, Any]]
    current_turn: int = 0
    current_step: int = 0
    tools_info: Optional[List[Dict[str, Any]]] = None
    user_system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_type: Optional[ErrorType] = None
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        if self.error_type:
            data["error_type"] = self.error_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = TrajectoryStatus(data["status"])
        if data.get("error_type"):
            data["error_type"] = ErrorType(data["error_type"])
        return cls(**data)


class CheckpointManager:
    """
    Manages checkpoints for trajectory processing with fault tolerance.
    
    Features:
    - Atomic save/load operations with file locking
    - Automatic retry for recoverable errors
    - Progress tracking across multiple trajectories
    - Cleanup of completed checkpoints
    
    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> 
        >>> # Save checkpoint during processing
        >>> with manager.checkpoint_context(trajectory_id) as cp:
        ...     cp.messages = messages
        ...     cp.current_turn = turn
        ...     # Processing happens here
        ...     # Checkpoint auto-saves on context exit
        >>> 
        >>> # Resume from checkpoint
        >>> if manager.has_checkpoint(trajectory_id):
        ...     cp = manager.load(trajectory_id)
        ...     messages = cp.messages
    """
    
    DEFAULT_MAX_RETRIES = 3
    CHECKPOINT_EXTENSION = ".checkpoint.json"
    LOCK_EXTENSION = ".lock"
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_retries: int = DEFAULT_MAX_RETRIES,
        auto_cleanup: bool = True
    ):
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            max_retries: Maximum retry attempts for recoverable errors
            auto_cleanup: Whether to automatically delete completed checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.auto_cleanup = auto_cleanup
    
    def _get_checkpoint_path(self, trajectory_id: str) -> Path:
        """Get the file path for a trajectory's checkpoint."""
        # Use hash prefix for better file distribution in large-scale scenarios
        prefix = hashlib.md5(trajectory_id.encode()).hexdigest()[:4]
        subdir = self.checkpoint_dir / prefix
        subdir.mkdir(exist_ok=True)
        return subdir / f"{trajectory_id}{self.CHECKPOINT_EXTENSION}"
    
    def _get_lock_path(self, trajectory_id: str) -> Path:
        """Get the lock file path for a trajectory."""
        return self._get_checkpoint_path(trajectory_id).with_suffix(self.LOCK_EXTENSION)
    
    def has_checkpoint(self, trajectory_id: str) -> bool:
        """Check if a checkpoint exists for the given trajectory."""
        return self._get_checkpoint_path(trajectory_id).exists()
    
    def can_resume(self, trajectory_id: str) -> bool:
        """
        Check if a trajectory can be resumed from checkpoint.
        
        Returns True if:
        - Checkpoint exists
        - Status is IN_PROGRESS or INTERRUPTED
        - Retry count hasn't exceeded max_retries (for recoverable errors)
        """
        if not self.has_checkpoint(trajectory_id):
            return False
        
        checkpoint = self.load(trajectory_id)
        if checkpoint is None:
            return False
        
        # Can resume if interrupted or in-progress
        if checkpoint.status in [TrajectoryStatus.IN_PROGRESS, TrajectoryStatus.INTERRUPTED]:
            # Check retry limit for recoverable errors
            if checkpoint.error_type == ErrorType.RECOVERABLE:
                return checkpoint.retry_count < self.max_retries
            return True
        
        return False
    
    def save(self, checkpoint: CheckpointData) -> None:
        """
        Atomically save a checkpoint to disk.
        
        Uses write-to-temp-then-rename pattern for atomicity.
        """
        checkpoint.updated_at = time.time()
        path = self._get_checkpoint_path(checkpoint.trajectory_id)
        lock_path = self._get_lock_path(checkpoint.trajectory_id)
        
        with FileLock(str(lock_path)):
            # Write to temp file first
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            shutil.move(str(temp_path), str(path))
        
        logger.debug(f"Saved checkpoint for {checkpoint.trajectory_id}")
    
    def load(self, trajectory_id: str) -> Optional[CheckpointData]:
        """Load a checkpoint from disk."""
        path = self._get_checkpoint_path(trajectory_id)
        lock_path = self._get_lock_path(trajectory_id)
        
        if not path.exists():
            return None
        
        try:
            with FileLock(str(lock_path)):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            return CheckpointData.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load checkpoint {trajectory_id}: {e}")
            return None
    
    def delete(self, trajectory_id: str) -> bool:
        """Delete a checkpoint."""
        path = self._get_checkpoint_path(trajectory_id)
        lock_path = self._get_lock_path(trajectory_id)
        
        try:
            if path.exists():
                path.unlink()
            if lock_path.exists():
                lock_path.unlink()
            return True
        except OSError as e:
            logger.error(f"Failed to delete checkpoint {trajectory_id}: {e}")
            return False
    
    def mark_completed(self, trajectory_id: str) -> None:
        """Mark a trajectory as completed and optionally clean up."""
        checkpoint = self.load(trajectory_id)
        if checkpoint:
            checkpoint.status = TrajectoryStatus.COMPLETED
            checkpoint.error_message = None
            checkpoint.error_type = None
            self.save(checkpoint)
            
            if self.auto_cleanup:
                self.delete(trajectory_id)
    
    def mark_failed(
        self,
        trajectory_id: str,
        error_message: str,
        error_type: ErrorType = ErrorType.NON_RECOVERABLE
    ) -> None:
        """Mark a trajectory as failed."""
        checkpoint = self.load(trajectory_id)
        if checkpoint:
            checkpoint.status = TrajectoryStatus.FAILED
            checkpoint.error_message = error_message
            checkpoint.error_type = error_type
            self.save(checkpoint)
    
    def mark_interrupted(
        self,
        trajectory_id: str,
        error_message: str,
        recoverable: bool = True
    ) -> None:
        """
        Mark a trajectory as interrupted.
        
        Args:
            trajectory_id: The trajectory ID
            error_message: Description of the error
            recoverable: Whether the error is recoverable
        """
        checkpoint = self.load(trajectory_id)
        if checkpoint:
            checkpoint.status = TrajectoryStatus.INTERRUPTED
            checkpoint.error_message = error_message
            checkpoint.error_type = (
                ErrorType.RECOVERABLE if recoverable else ErrorType.NON_RECOVERABLE
            )
            if recoverable:
                checkpoint.retry_count += 1
            self.save(checkpoint)
    
    @contextmanager
    def checkpoint_context(
        self,
        trajectory_id: str,
        initial_messages: Optional[List[Dict[str, Any]]] = None,
        **metadata
    ):
        """
        Context manager for automatic checkpoint management.
        
        Loads existing checkpoint or creates new one. Saves on normal exit.
        Marks as interrupted on exception.
        
        Example:
            >>> with manager.checkpoint_context(traj_id) as cp:
            ...     # Do processing
            ...     cp.messages.append(new_message)
            ...     cp.current_step += 1
        """
        # Load or create checkpoint
        checkpoint = self.load(trajectory_id)
        if checkpoint is None:
            checkpoint = CheckpointData(
                trajectory_id=trajectory_id,
                status=TrajectoryStatus.IN_PROGRESS,
                messages=initial_messages or [],
                metadata=metadata
            )
        else:
            checkpoint.status = TrajectoryStatus.IN_PROGRESS
        
        self.save(checkpoint)
        
        try:
            yield checkpoint
            # Normal exit - save final state
            self.save(checkpoint)
        except Exception as e:
            # Classify the error
            recoverable = self._is_recoverable_error(e)
            self.mark_interrupted(
                trajectory_id,
                str(e),
                recoverable=recoverable
            )
            raise
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable.
        
        Recoverable errors include:
        - Network timeouts
        - Rate limits
        - Temporary service unavailable
        """
        error_str = str(error).lower()
        recoverable_patterns = [
            "timeout",
            "rate limit",
            "rate_limit",
            "too many requests",
            "service unavailable",
            "503",
            "502",
            "504",
            "connection",
            "network",
            "temporarily",
        ]
        return any(pattern in error_str for pattern in recoverable_patterns)
    
    def list_checkpoints(
        self,
        status: Optional[TrajectoryStatus] = None
    ) -> List[CheckpointData]:
        """
        List all checkpoints, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of CheckpointData objects
        """
        checkpoints = []
        for subdir in self.checkpoint_dir.iterdir():
            if not subdir.is_dir():
                continue
            for path in subdir.glob(f"*{self.CHECKPOINT_EXTENSION}"):
                trajectory_id = path.stem.replace(self.CHECKPOINT_EXTENSION.replace(".json", ""), "")
                checkpoint = self.load(trajectory_id)
                if checkpoint:
                    if status is None or checkpoint.status == status:
                        checkpoints.append(checkpoint)
        
        return sorted(checkpoints, key=lambda c: c.updated_at, reverse=True)
    
    def get_resumable_trajectories(self) -> List[str]:
        """Get list of trajectory IDs that can be resumed."""
        resumable = []
        for checkpoint in self.list_checkpoints():
            if self.can_resume(checkpoint.trajectory_id):
                resumable.append(checkpoint.trajectory_id)
        return resumable
    
    def cleanup_completed(self) -> int:
        """Delete all completed checkpoints. Returns count deleted."""
        count = 0
        for checkpoint in self.list_checkpoints(status=TrajectoryStatus.COMPLETED):
            if self.delete(checkpoint.trajectory_id):
                count += 1
        return count
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about checkpoint states."""
        stats = {status.value: 0 for status in TrajectoryStatus}
        for checkpoint in self.list_checkpoints():
            stats[checkpoint.status.value] += 1
        return stats

