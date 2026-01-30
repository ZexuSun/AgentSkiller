"""
Evaluator Module - Trajectory execution and validation for multi-turn data quality.

This module provides:
1. TrajectoryExecutor: Execute golden trajectories with LLM-generated parameter code
2. TrajectoryCodeManager: Manage and cache generated trajectory execution code
3. Base data types for evaluation (TaskDefinition, TrajectoryExecution, etc.)

Usage:
    from evaluator import TrajectoryExecutor, TrajectoryCodeManager, TaskDefinition
    
    # Initialize
    executor = TrajectoryExecutor(outputs_dir="./outputs")
    code_manager = TrajectoryCodeManager(outputs_dir="./outputs")
    
    # Execute a trajectory
    task = TaskDefinition(
        task_id="test_001",
        trajectory=["authorize_student", "get_courses", "enroll_course"],
        entity_context={"Student.student_id": "..."},
        domains=["StudentAcademicPortal"],
    )
    
    result = executor.execute(task)
    
    # Save generated code for reuse
    if result.success:
        code_manager.save_code(
            trajectory=task.trajectory,
            domains=task.domains,
            step_param_codes=result.step_param_codes,
        )
"""

from .base import (
    # Enums
    EvaluatorType,
    
    # Data classes
    ToolCall,
    ExecutionStep,
    DatabaseState,
    TrajectoryExecution,
    TaskDefinition,
    EvaluationResult,
    CompositeEvaluationResult,
    
    # Base class
    BaseEvaluator,
)

from .trajectory_executor import (
    TrajectoryExecutor,
    CrossDomainServerWrapper,
)

from .trajectory_code_manager import (
    TrajectoryCodeManager,
    TrajectoryCodeEntry,
    PruningResultEntry,
    compute_trajectory_hash,
    # AST utilities for code caching
    analyze_param_code,
    fix_hardcoded_ctx,
    validate_ctx_keys,
)

from .trajectory_pruner import (
    TrajectoryPruner,
    PruningResult,
    RedundancyResult,
)

from .data_loader import (
    EvaluationDataLoader,
    EvaluationSample,
    load_evaluation_data,
)

from .run_evaluation import (
    EvaluationRunner,
    EvaluationReport,
    SampleEvaluationResult,
)

__all__ = [
    # Enums
    "EvaluatorType",
    
    # Data classes
    "ToolCall",
    "ExecutionStep",
    "DatabaseState",
    "TrajectoryExecution",
    "TaskDefinition",
    "EvaluationResult",
    "CompositeEvaluationResult",
    
    # Base class
    "BaseEvaluator",
    
    # Executor
    "TrajectoryExecutor",
    "CrossDomainServerWrapper",
    
    # Code Manager
    "TrajectoryCodeManager",
    "TrajectoryCodeEntry",
    "PruningResultEntry",
    "compute_trajectory_hash",
    # AST utilities
    "analyze_param_code",
    "fix_hardcoded_ctx",
    "validate_ctx_keys",
    
    # Pruner
    "TrajectoryPruner",
    "PruningResult",
    "RedundancyResult",
    
    # Data Loader
    "EvaluationDataLoader",
    "EvaluationSample",
    "load_evaluation_data",
    
    # Evaluation Runner
    "EvaluationRunner",
    "EvaluationReport",
    "SampleEvaluationResult",
]
