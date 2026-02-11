"""
Task models for task generation and query generation steps.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TrajectoryInfo(BaseModel):
    """Information about a tool call trajectory."""
    
    path: List[str]  # Sequence of function names
    terminal_node: str  # Final function in the trajectory
    edge_types: List[Dict[str, str]] = Field(default_factory=list)
    
    @property
    def length(self) -> int:
        return len(self.path)
    
    def to_call_sequence(self) -> str:
        """Convert to readable call sequence string."""
        return " -> ".join(self.path)


class TaskTemplate(BaseModel):
    """
    Template for a generated task.
    
    Contains the high-level task description and reasoning,
    without specific entity instances.
    """
    
    instruction: str  # What the user wants to accomplish
    reason_for_call: str  # Why the user is making this request
    known_info: Dict[str, Any] = Field(default_factory=dict)  # Info user knows
    
    # Metadata
    complexity: Optional[str] = None  # "simple", "medium", "complex"
    category: Optional[str] = None  # Task category


class TaskTemplateScores(BaseModel):
    """LLM Judge scores for a task template."""
    
    motivation_clarity: float = Field(ge=1, le=5)  # Does the task have a clear motivation?
    logical_coherence: float = Field(ge=1, le=5)   # Does the task make logical sense?
    completeness: float = Field(ge=1, le=5)        # Does it lead to executing all tools?
    naturalness: float = Field(ge=1, le=5)         # Does it sound like a real user request?
    specificity: float = Field(ge=1, le=5)         # Is the task specific enough?
    overall_score: float = Field(ge=1, le=5)       # Average of all scores
    rejection_reason: Optional[str] = None         # Reason if rejected (overall < threshold)


class TaskTemplateWithTrajectory(BaseModel):
    """Task template paired with its expected trajectory."""
    
    trajectory: List[str]  # Expected function call sequence
    task_template: TaskTemplate
    
    # LLM Judge scores (optional, populated after judging)
    scores: Optional[TaskTemplateScores] = None
    
    # Additional metadata
    domains: List[str] = Field(default_factory=list)  # Involved domains
    is_cross_domain: bool = False


class UserScenario(BaseModel):
    """User scenario for task instantiation."""
    
    persona: Optional[str] = None  # User persona description
    instructions: Dict[str, str] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class EntityContext(BaseModel):
    """
    Instantiated entity context for a task.
    
    Contains the actual entity values sampled for the task.
    """
    
    core_entity: Dict[str, Any] = Field(default_factory=dict)
    peripheral_entities: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    relationships: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class InstantiatedTask(BaseModel):
    """
    Fully instantiated task with concrete entity values.
    
    Ready for execution or query generation.
    """
    
    id: str
    template: TaskTemplateWithTrajectory
    entity_context: EntityContext
    user_scenario: Optional[UserScenario] = None
    
    # Execution metadata
    expected_trajectory: List[str] = Field(default_factory=list)
    constraint_script_path: Optional[str] = None
    
    # Validation status
    is_validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    
    def get_instantiated_instruction(self) -> str:
        """Get instruction with entity values filled in."""
        instruction = self.template.task_template.instruction
        
        # Simple template substitution
        for key, value in self.entity_context.core_entity.items():
            placeholder = f"{{{key}}}"
            if placeholder in instruction:
                instruction = instruction.replace(placeholder, str(value))
        
        return instruction


class QueryOutput(BaseModel):
    """
    Output from query generation step.
    
    Contains the user's initial query/message that starts
    the conversation with the MCP server.
    """
    
    id: str  # Task ID reference
    messages: List[Dict[str, str]]  # Conversation messages
    user_system_prompt: str  # System prompt for the user agent
    
    # Reference to original task
    task_info: Optional[Dict[str, Any]] = None
    
    # Metadata
    domains: List[str] = Field(default_factory=list)
    trajectory_length: int = 0
    is_cross_domain: bool = False


class TaskGenerationResult(BaseModel):
    """Result of task template generation."""
    
    server_name: str
    total_trajectories: int
    generated_templates: int
    output_path: str
    success: bool
    errors: List[str] = Field(default_factory=list)


class TaskInstantiationResult(BaseModel):
    """Result of task instantiation."""
    
    task_id: str
    template_id: str
    success: bool
    output_path: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class TaskFilteringResult(BaseModel):
    """Result of task filtering/validation."""
    
    total_tasks: int
    valid_tasks: int
    invalid_tasks: int
    validation_errors: Dict[str, List[str]] = Field(default_factory=dict)


class CrossDomainCombination(BaseModel):
    """
    A combination of domains for cross-domain task generation.
    """
    
    domains: List[str]  # Domain names
    core_entities: List[str]  # Core entities from each domain
    servers: List[str]  # MCP server names
    fused_name: str  # Combined name for outputs
    
    # Fusion metadata
    shared_entities: List[str] = Field(default_factory=list)
    constraint_mappings: Dict[str, str] = Field(default_factory=dict)
