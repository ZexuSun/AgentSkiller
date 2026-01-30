"""
Base classes and data models for the evaluation framework.
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime


class EvaluatorType(Enum):
    """Types of evaluators in the system."""
    ACTION = "action"
    ENVIRONMENT = "environment"
    NL_ASSERTIONS = "nl_assertions"
    SUBJECTIVE = "subjective"  # LLM-as-Judge for quality/policy compliance


class FailureType(Enum):
    """Types of failures during trajectory execution."""
    CODE_GENERATION_ERROR = "code_generation"  # LLM failed to generate valid code
    CODE_EXECUTION_ERROR = "code_execution"    # Generated code raised exception
    TOOL_CALL_ERROR = "tool_call"              # MCP tool returned error
    CONSTRAINT_VIOLATION = "constraint"         # Business constraint violated
    TIMEOUT = "timeout"                         # Execution timed out
    UNKNOWN = "unknown"                         # Unknown error


@dataclass
class ToolCall:
    """
    Represents a single tool/function call.
    
    Attributes:
        name: Function name (e.g., "authorize_student")
        arguments: Dictionary of argument name -> value
        result: The result returned by the function (if executed)
    """
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
        )
    
    def matches(
        self, 
        other: "ToolCall", 
        check_params: Optional[List[str]] = None,
        ignore_params: Optional[List[str]] = None,
    ) -> bool:
        """
        Check if this tool call matches another.
        
        Args:
            other: The tool call to compare against
            check_params: If provided, only check these specific parameters
            ignore_params: Parameters to ignore in comparison
            
        Returns:
            True if the tool calls match according to the specified criteria
        """
        # Function name must match
        if self.name != other.name:
            return False
        
        # Determine which parameters to check
        if check_params is not None:
            params_to_check = check_params
        else:
            params_to_check = list(self.arguments.keys())
        
        # Remove ignored params
        if ignore_params:
            params_to_check = [p for p in params_to_check if p not in ignore_params]
        
        # Compare parameters
        for param in params_to_check:
            if param not in self.arguments or param not in other.arguments:
                return False
            if self.arguments[param] != other.arguments[param]:
                return False
        
        return True


@dataclass
class ExecutionStep:
    """
    Represents a single step in trajectory execution.
    
    Attributes:
        step_index: Position in the trajectory (0-based)
        tool_call: The tool call made at this step
        code: Python code generated to make this call
        success: Whether execution succeeded
        error: Error message if execution failed
        failure_type: Type of failure if not successful
        retry_count: Number of retries attempted for this step
    """
    step_index: int
    tool_call: ToolCall
    code: str = ""
    success: bool = True
    error: Optional[str] = None
    failure_type: Optional[FailureType] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool_call": self.tool_call.to_dict(),
            "code": self.code,
            "success": self.success,
            "error": self.error,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "retry_count": self.retry_count,
        }


@dataclass
class DatabaseState:
    """
    Represents the state of databases at a point in time.
    
    Attributes:
        entities: Dictionary of entity_type -> list of entity records
        relationships: Dictionary of relationship_type -> list of relationship records
        hash: Cryptographic hash of the state for quick comparison
    """
    entities: Dict[str, List[Dict]] = field(default_factory=dict)
    relationships: Dict[str, List[Dict]] = field(default_factory=dict)
    _hash: Optional[str] = None
    
    @property
    def hash(self) -> str:
        """Compute a deterministic hash of the database state."""
        if self._hash is None:
            # Sort everything for deterministic serialization
            state_str = json.dumps(
                {"entities": self.entities, "relationships": self.relationships},
                sort_keys=True,
                default=str,
            )
            self._hash = hashlib.sha256(state_str.encode()).hexdigest()
        return self._hash
    
    def matches(self, other: "DatabaseState") -> bool:
        """Check if two database states are identical."""
        return self.hash == other.hash
    
    def diff(self, other: "DatabaseState") -> Dict[str, Any]:
        """
        Compute differences between two database states.
        
        Returns:
            Dictionary describing the differences
        """
        diffs = {
            "entities": {},
            "relationships": {},
        }
        
        # Compare entities
        all_entity_types = set(self.entities.keys()) | set(other.entities.keys())
        for entity_type in all_entity_types:
            self_records = self.entities.get(entity_type, [])
            other_records = other.entities.get(entity_type, [])
            
            if len(self_records) != len(other_records):
                diffs["entities"][entity_type] = {
                    "count_diff": len(other_records) - len(self_records),
                    "self_count": len(self_records),
                    "other_count": len(other_records),
                }
        
        # Compare relationships
        all_rel_types = set(self.relationships.keys()) | set(other.relationships.keys())
        for rel_type in all_rel_types:
            self_records = self.relationships.get(rel_type, [])
            other_records = other.relationships.get(rel_type, [])
            
            if len(self_records) != len(other_records):
                diffs["relationships"][rel_type] = {
                    "count_diff": len(other_records) - len(self_records),
                    "self_count": len(self_records),
                    "other_count": len(other_records),
                }
        
        return diffs


@dataclass
class TrajectoryExecution:
    """
    Complete execution result of a trajectory.
    
    Attributes:
        task_id: Unique identifier for the task
        steps: List of execution steps
        final_state: Database state after execution
        initial_state: Database state before execution
        session_id: Session ID used for execution
        generated_code: Complete Python code for all steps
        step_param_codes: Raw param extraction codes per step (for reuse)
        success: Whether all steps succeeded
        errors: List of error messages
        total_retries: Total number of step-wise retries across all steps
    """
    task_id: str
    steps: List[ExecutionStep]
    final_state: Optional[DatabaseState] = None
    initial_state: Optional[DatabaseState] = None
    session_id: str = ""
    generated_code: str = ""
    step_param_codes: List[str] = field(default_factory=list)
    success: bool = True
    errors: List[str] = field(default_factory=list)
    total_retries: int = 0
    
    @property
    def tool_calls(self) -> List[ToolCall]:
        """Get all tool calls from execution steps."""
        return [step.tool_call for step in self.steps]
    
    @property
    def failed_step_index(self) -> Optional[int]:
        """Get the index of the first failed step, if any."""
        for step in self.steps:
            if not step.success:
                return step.step_index
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "session_id": self.session_id,
            "generated_code": self.generated_code,
            "step_param_codes": self.step_param_codes,
            "success": self.success,
            "errors": self.errors,
            "total_retries": self.total_retries,
        }


@dataclass
class TaskDefinition:
    """
    Definition of an evaluation task.
    
    Attributes:
        task_id: Unique identifier
        trajectory: Expected sequence of function names
        entity_context: Initial context with entity values
        instruction: Natural language instruction
        domains: List of domains involved
        expected_actions: Expected tool calls with parameters
        communicate_requirements: Strings that must appear in responses
        nl_assertions: Natural language conditions to verify
        environment_assertions: Custom validation functions
    """
    task_id: str
    trajectory: List[str]
    entity_context: Dict[str, Any]
    instruction: str = ""
    reason_for_call: str = ""
    domains: List[str] = field(default_factory=list)
    is_cross_domain: bool = False
    
    # Evaluation criteria
    expected_actions: Optional[List[ToolCall]] = None
    check_params: Optional[List[str]] = None  # Only check these params
    ignore_params: Optional[List[str]] = None  # Ignore these params
    
    communicate_requirements: List[str] = field(default_factory=list)
    nl_assertions: List[str] = field(default_factory=list)
    environment_assertions: List[Callable[[DatabaseState, DatabaseState], bool]] = field(default_factory=list)
    
    @classmethod
    def from_task_file(cls, task_data: Dict[str, Any], use_pruned: bool = True) -> "TaskDefinition":
        """
        Create TaskDefinition from task JSON data.
        
        Args:
            task_data: Task data dictionary
            use_pruned: If True and pruned trajectory exists, use it as the golden trajectory
        """
        task_id = task_data.get("task_id", "")
        if not task_id:
            # Generate from hash of content
            task_id = hashlib.md5(
                json.dumps(task_data, sort_keys=True).encode()
            ).hexdigest()[:12]
        
        instantiated = task_data.get("instantiated_task", {})
        
        # Use pruned trajectory if available and requested
        # Pruned trajectory is stored in "trajectory" after pruning,
        # with original stored in "original_trajectory"
        trajectory = task_data.get("trajectory", [])
        original_trajectory = task_data.get("original_trajectory")
        
        # If we have an original_trajectory, it means the current trajectory is pruned
        pruning_info = task_data.get("pruning_info")
        
        # NL assertions can be in instantiated_task or at root level
        nl_assertions = instantiated.get("nl_assertions", [])
        if not nl_assertions:
            nl_assertions = task_data.get("nl_assertions", [])
        
        return cls(
            task_id=task_id,
            trajectory=trajectory,
            entity_context=task_data.get("entity_context", {}),
            instruction=instantiated.get("instruction", task_data.get("template", {}).get("instruction", "")),
            reason_for_call=instantiated.get("reason_for_call", task_data.get("template", {}).get("reason_for_call", "")),
            domains=task_data.get("domains", []) or [],
            is_cross_domain=task_data.get("is_cross_domain", False),
            nl_assertions=nl_assertions,
        )
    
    @classmethod
    def from_combo(
        cls,
        trajectory: List[str],
        combo: Dict[str, Any],
        server_name: str,
        task_template: Dict[str, Any] = None,
    ) -> "TaskDefinition":
        """
        Create TaskDefinition from a combination entry (Step 15 output).
        
        Args:
            trajectory: List of function names
            combo: Combo entry with entity_instances and value_domain_samples
            server_name: Server name (single or cross-domain)
            task_template: Optional task template with instruction/reason
        """
        # Build entity context from combo - preserve original semantic structure
        entity_context = {
            "entity_instances": combo.get("entity_instances", {}),
            "value_domain_samples": combo.get("value_domain_samples", {}),
        }
        
        # Determine domains
        if "_" in server_name:
            domains = server_name.split("_")
            is_cross_domain = True
        else:
            domains = [server_name]
            is_cross_domain = False
        
        # Generate task ID
        import hashlib
        context_hash = hashlib.md5(
            json.dumps(entity_context, sort_keys=True).encode()
        ).hexdigest()[:8]
        traj_hash = hashlib.md5("_".join(trajectory).encode()).hexdigest()[:8]
        task_id = f"{server_name}_{traj_hash}_{context_hash}"
        
        return cls(
            task_id=task_id,
            trajectory=trajectory,
            entity_context=entity_context,
            instruction=task_template.get("instruction", "") if task_template else "",
            reason_for_call=task_template.get("reason_for_call", "") if task_template else "",
            domains=domains,
            is_cross_domain=is_cross_domain,
        )


@dataclass
class EvaluationResult:
    """
    Result from a single evaluator.
    
    Attributes:
        evaluator_type: Type of evaluator that produced this result
        score: Binary score (0.0 or 1.0 for all-or-nothing evaluation)
        passed: Whether all criteria passed
        details: Detailed breakdown of evaluation
        reasoning: Explanation of the evaluation
    """
    evaluator_type: EvaluatorType
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_type": self.evaluator_type.value,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
            "reasoning": self.reasoning,
        }


@dataclass
class CompositeEvaluationResult:
    """
    Combined result from all evaluators.
    
    Attributes:
        task_id: Task identifier
        total_score: Product of all scores (all-or-nothing)
        passed: Whether all evaluators passed
        results: Individual evaluator results
        timestamp: Evaluation timestamp
    """
    task_id: str
    total_score: float
    passed: bool
    results: Dict[EvaluatorType, EvaluationResult] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_score": self.total_score,
            "passed": self.passed,
            "results": {k.value: v.to_dict() for k, v in self.results.items()},
            "timestamp": self.timestamp,
        }


class BaseEvaluator:
    """
    Base class for all evaluators.
    
    All evaluators follow the all-or-nothing scoring pattern:
    - Return 1.0 if all criteria pass
    - Return 0.0 if any criterion fails
    """
    
    evaluator_type: EvaluatorType
    
    def evaluate(
        self,
        task: TaskDefinition,
        gold_execution: TrajectoryExecution,
        agent_trajectory: List[Dict[str, Any]],
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate agent performance.
        
        Args:
            task: Task definition with evaluation criteria
            gold_execution: Execution of golden trajectory
            agent_trajectory: Agent's conversation messages
            
        Returns:
            EvaluationResult with score and details
        """
        raise NotImplementedError
    
    def _create_pass_result(self, details: Dict[str, Any] = None) -> EvaluationResult:
        """Helper to create a passing result."""
        return EvaluationResult(
            evaluator_type=self.evaluator_type,
            score=1.0,
            passed=True,
            details=details or {},
            reasoning="All criteria passed",
        )
    
    def _create_fail_result(
        self, 
        reason: str, 
        details: Dict[str, Any] = None,
    ) -> EvaluationResult:
        """Helper to create a failing result."""
        return EvaluationResult(
            evaluator_type=self.evaluator_type,
            score=0.0,
            passed=False,
            details=details or {},
            reasoning=reason,
        )
    
    def _create_skip_result(self, reason: str = "No criteria defined") -> EvaluationResult:
        """Helper to create a skipped result (returns 1.0)."""
        return EvaluationResult(
            evaluator_type=self.evaluator_type,
            score=1.0,
            passed=True,
            details={"skipped": True},
            reasoning=reason,
        )
