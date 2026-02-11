"""
Workflow state models for tracking execution progress.
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult(BaseModel):
    """Result of a workflow step execution."""
    
    step_name: str
    status: StepStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    # Progress tracking for retry
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    
    @property
    def success(self) -> bool:
        return self.status == StepStatus.COMPLETED
    
    @property
    def is_complete(self) -> bool:
        """Check if all items are processed."""
        if self.total_items == 0:
            return self.status == StepStatus.COMPLETED
        return self.completed_items >= self.total_items


class WorkflowState(BaseModel):
    """
    Complete workflow state for checkpointing and recovery.
    
    This is the central state object passed between LangGraph nodes.
    Supports:
    - Step status tracking
    - Incremental progress for retry
    - Artifact path management
    """
    
    model_config = {"extra": "allow"}  # Allow dynamic attributes
    
    # =========================================================================
    # Metadata
    # =========================================================================
    workflow_id: str
    started_at: str
    current_step: Optional[str] = None
    
    # =========================================================================
    # Step Results
    # =========================================================================
    steps: Dict[str, StepResult] = Field(default_factory=dict)
    
    # =========================================================================
    # Artifact Paths
    # =========================================================================
    # Step 1: Domain Expansion
    domain_topics_path: Optional[str] = None
    
    # Step 2-3: Entity Extraction & Graph
    entities_path: Optional[str] = None
    entity_graph_path: Optional[str] = None
    
    # Step 4-5: Blueprint Generation & Tool List
    blueprints_path: Optional[str] = None
    fixed_blueprints_path: Optional[str] = None
    tool_lists_dir: Optional[str] = None
    
    # Step 6: Database Generation
    databases_dir: Optional[str] = None
    database_summary_dir: Optional[str] = None
    
    # Step 7: MCP Server Implementation
    mcp_servers_dir: Optional[str] = None
    unit_tests_dir: Optional[str] = None
    policies_dir: Optional[str] = None
    tool_graphs_dir: Optional[str] = None
    
    # Step 8-11: Cross-Domain Steps
    cross_domain_combinations_path: Optional[str] = None
    fused_databases_dir: Optional[str] = None
    merged_policies_dir: Optional[str] = None
    
    # Step 12-16: Task Generation
    task_templates_dir: Optional[str] = None
    cross_domain_templates_dir: Optional[str] = None
    instantiated_tasks_dir: Optional[str] = None
    validated_tasks_dir: Optional[str] = None
    queries_dir: Optional[str] = None
    
    # =========================================================================
    # In-Memory Data (not serialized to checkpoint)
    # =========================================================================
    domain_topics: Optional[Set[str]] = None
    
    # =========================================================================
    # Processing State (for retry)
    # =========================================================================
    processed_domains: List[str] = Field(default_factory=list)
    failed_domains: List[str] = Field(default_factory=list)
    processed_entities: List[str] = Field(default_factory=list)
    failed_entities: Dict[str, List[str]] = Field(default_factory=dict)
    processed_blueprints: List[str] = Field(default_factory=list)
    failed_blueprints: List[str] = Field(default_factory=list)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # =========================================================================
    # Step Status Methods
    # =========================================================================
    
    def mark_step_started(self, step_name: str) -> None:
        """Mark a step as started."""
        self.current_step = step_name
        self.steps[step_name] = StepResult(
            step_name=step_name,
            status=StepStatus.RUNNING,
            started_at=datetime.now().isoformat(),
        )
    
    def mark_step_completed(
        self, 
        step_name: str, 
        outputs: Optional[Dict] = None
    ) -> None:
        """Mark a step as completed."""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now().isoformat()
            if step.started_at:
                started = datetime.fromisoformat(step.started_at)
                completed = datetime.fromisoformat(step.completed_at)
                step.duration_seconds = (completed - started).total_seconds()
            if outputs:
                step.outputs = outputs
    
    def mark_step_failed(self, step_name: str, errors: List[str]) -> None:
        """Mark a step as failed."""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.status = StepStatus.FAILED
            step.completed_at = datetime.now().isoformat()
            step.errors = errors
    
    def update_step_progress(
        self, 
        step_name: str, 
        total: int, 
        completed: int, 
        failed: int = 0
    ) -> None:
        """Update step progress counters."""
        if step_name in self.steps:
            step = self.steps[step_name]
            step.total_items = total
            step.completed_items = completed
            step.failed_items = failed
    
    # =========================================================================
    # Step Completion Checks
    # =========================================================================
    
    def is_step_complete(self, step_name: str) -> bool:
        """
        Check if a step is completely done.
        
        A step is complete if:
        - It has a COMPLETED status, OR
        - All items have been processed (remaining == 0)
        """
        if step_name not in self.steps:
            return False
        
        step = self.steps[step_name]
        
        if step.status == StepStatus.COMPLETED:
            return True
        
        # Step is complete if no items remaining
        return step.completed_items >= step.total_items
    
    def get_remaining_items(self, step_name: str) -> int:
        """Get count of remaining items for a step."""
        if step_name not in self.steps:
            return 0
        
        step = self.steps[step_name]
        return max(0, step.total_items - step.completed_items)
    
    # =========================================================================
    # Resume Support
    # =========================================================================
    
    def get_last_completed_step(self) -> Optional[str]:
        """Get the name of the last successfully completed step."""
        completed = [
            (name, step.completed_at)
            for name, step in self.steps.items()
            if step.status == StepStatus.COMPLETED and step.completed_at
        ]
        if not completed:
            return None
        completed.sort(key=lambda x: x[1])
        return completed[-1][0]
    
    def can_resume_from(self, step_name: str) -> bool:
        """Check if workflow can resume from a given step."""
        from ..graph.workflow import WORKFLOW_STEPS
        
        if step_name not in WORKFLOW_STEPS:
            return False
        
        step_idx = WORKFLOW_STEPS.index(step_name)
        
        # Check all previous steps are completed
        for prev_step in WORKFLOW_STEPS[:step_idx]:
            if prev_step not in self.steps:
                return False
            if self.steps[prev_step].status != StepStatus.COMPLETED:
                return False
        
        return True
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_checkpoint(self) -> Dict[str, Any]:
        """
        Serialize state for checkpointing.
        
        Excludes large in-memory data that should be reloaded from files.
        """
        return self.model_dump(
            exclude={"domain_topics"},
            exclude_none=True,
        )
    
    @classmethod
    def from_checkpoint(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Load state from checkpoint data."""
        return cls.model_validate(data)
