"""
Composite Evaluator - Combines all evaluators into a unified scoring system.

This is the main entry point for evaluation, orchestrating:
1. Action Evaluation (tool call sequence)
2. Environment Evaluation (database state)
3. Communication Evaluation (information delivery)
4. NL Assertions Evaluation (subjective criteria)

All scores are multiplied together (all-or-nothing approach):
- Final score = action_score × environment_score × communication_score × nl_score
- Any component failure (0.0) results in overall failure
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .base import (
    EvaluatorType,
    EvaluationResult,
    CompositeEvaluationResult,
    TaskDefinition,
    TrajectoryExecution,
)
from .action_evaluator import ActionEvaluator
from .environment_evaluator import EnvironmentEvaluator
from .nl_assertions_evaluator import NLAssertionsEvaluator
from .subjective_evaluator import SubjectiveQualityEvaluator
from .trajectory_executor import TrajectoryExecutor

logger = logging.getLogger(__name__)


class CompositeEvaluator:
    """
    Unified evaluator that combines all evaluation dimensions.
    
    Usage:
        evaluator = CompositeEvaluator()
        
        # Evaluate a single task
        result = evaluator.evaluate(
            task=task_definition,
            agent_trajectory=agent_messages,
        )
        
        print(f"Score: {result.total_score}")  # 0.0 or 1.0
        print(f"Passed: {result.passed}")
        
        # Batch evaluation
        results = evaluator.evaluate_batch(tasks, agent_trajectories)
    """
    
    def __init__(
        self,
        outputs_dir: str = "./outputs",
        enable_action: bool = True,
        enable_environment: bool = True,
        enable_communication: bool = True,
        enable_nl_assertions: bool = True,
        enable_subjective: bool = False,  # New: subjective quality evaluation
        action_config: Optional[Dict] = None,
        environment_config: Optional[Dict] = None,
        communication_config: Optional[Dict] = None,
        nl_assertions_config: Optional[Dict] = None,
        subjective_config: Optional[Dict] = None,
    ):
        """
        Initialize the composite evaluator.
        
        Args:
            outputs_dir: Directory containing MCP servers and databases
            enable_*: Whether to enable each evaluator type
            *_config: Configuration dictionaries for each evaluator
        """
        self.outputs_dir = Path(outputs_dir)
        
        # Initialize trajectory executor
        self.executor = TrajectoryExecutor(outputs_dir=outputs_dir)
        
        # Initialize individual evaluators
        self.evaluators: Dict[EvaluatorType, Any] = {}
        
        if enable_action:
            self.evaluators[EvaluatorType.ACTION] = ActionEvaluator(
                **(action_config or {})
            )
        
        if enable_environment:
            self.evaluators[EvaluatorType.ENVIRONMENT] = EnvironmentEvaluator(
                outputs_dir=outputs_dir,
                **(environment_config or {})
            )
        
        if enable_nl_assertions:
            self.evaluators[EvaluatorType.NL_ASSERTIONS] = NLAssertionsEvaluator(
                **(nl_assertions_config or {})
            )
        
        if enable_subjective:
            self.evaluators[EvaluatorType.SUBJECTIVE] = SubjectiveQualityEvaluator(
                outputs_dir=outputs_dir,
                **(subjective_config or {})
            )
    
    def evaluate(
        self,
        task: TaskDefinition,
        agent_trajectory: List[Dict[str, Any]],
        gold_execution: Optional[TrajectoryExecution] = None,
        agent_execution: Optional[TrajectoryExecution] = None,
    ) -> CompositeEvaluationResult:
        """
        Evaluate agent performance on a single task.
        
        Args:
            task: Task definition with evaluation criteria
            agent_trajectory: Agent's conversation messages
            gold_execution: Pre-computed golden execution (optional)
            agent_execution: Pre-computed agent execution (optional)
            
        Returns:
            CompositeEvaluationResult with combined scores
        """
        logger.info(f"Evaluating task: {task.task_id}")
        
        # Execute golden trajectory if not provided
        if gold_execution is None:
            logger.info("Executing golden trajectory...")
            gold_execution = self.executor.execute(task)
            
            if not gold_execution.success:
                logger.warning(f"Golden execution failed: {gold_execution.errors}")
        
        # Run each evaluator
        results: Dict[EvaluatorType, EvaluationResult] = {}
        
        for eval_type, evaluator in self.evaluators.items():
            logger.debug(f"Running {eval_type.value} evaluator...")
            
            try:
                result = evaluator.evaluate(
                    task=task,
                    gold_execution=gold_execution,
                    agent_trajectory=agent_trajectory,
                    agent_execution=agent_execution,
                )
                results[eval_type] = result
                logger.debug(f"{eval_type.value}: {'PASS' if result.passed else 'FAIL'}")
                
            except Exception as e:
                logger.error(f"{eval_type.value} evaluator failed: {e}")
                results[eval_type] = EvaluationResult(
                    evaluator_type=eval_type,
                    score=0.0,
                    passed=False,
                    reasoning=f"Evaluator error: {e}",
                )
        
        # Calculate composite score (product of all scores)
        total_score = 1.0
        all_passed = True
        
        for result in results.values():
            total_score *= result.score
            if not result.passed:
                all_passed = False
        
        return CompositeEvaluationResult(
            task_id=task.task_id,
            total_score=total_score,
            passed=all_passed,
            results=results,
        )
    
    def evaluate_batch(
        self,
        tasks: List[TaskDefinition],
        agent_trajectories: List[List[Dict[str, Any]]],
        gold_executions: Optional[List[TrajectoryExecution]] = None,
        max_workers: int = 4,
    ) -> List[CompositeEvaluationResult]:
        """
        Evaluate multiple tasks in parallel.
        
        Args:
            tasks: List of task definitions
            agent_trajectories: List of agent conversation messages
            gold_executions: Pre-computed golden executions (optional)
            max_workers: Maximum parallel workers
            
        Returns:
            List of evaluation results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if len(tasks) != len(agent_trajectories):
            raise ValueError("Number of tasks must match number of trajectories")
        
        # Execute golden trajectories if not provided
        if gold_executions is None:
            logger.info(f"Executing {len(tasks)} golden trajectories...")
            gold_executions = self.executor.execute_batch(tasks, max_workers=max_workers)
        
        # Evaluate each task
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for i, (task, agent_traj, gold_exec) in enumerate(
                zip(tasks, agent_trajectories, gold_executions)
            ):
                future = executor.submit(
                    self.evaluate,
                    task=task,
                    agent_trajectory=agent_traj,
                    gold_execution=gold_exec,
                )
                futures[future] = i
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Task {idx} evaluation failed: {e}")
                    results[idx] = CompositeEvaluationResult(
                        task_id=tasks[idx].task_id,
                        total_score=0.0,
                        passed=False,
                        results={},
                    )
        
        return results
    
    def evaluate_from_file(
        self,
        task_path: Path,
        trajectory_path: Path,
    ) -> CompositeEvaluationResult:
        """
        Evaluate from task and trajectory files.
        
        Args:
            task_path: Path to task JSON file
            trajectory_path: Path to trajectory JSONL file
            
        Returns:
            CompositeEvaluationResult
        """
        # Load task
        with open(task_path) as f:
            task_data = json.load(f)
        
        if isinstance(task_data, list):
            task_data = task_data[0]
        
        task = TaskDefinition.from_task_file(task_data)
        
        # Load trajectory
        messages = []
        with open(trajectory_path) as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
        
        return self.evaluate(task=task, agent_trajectory=messages)
    
    def generate_report(
        self,
        results: List[CompositeEvaluationResult],
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate an evaluation report from multiple results.
        
        Args:
            results: List of evaluation results
            output_path: Optional path to save the report
            
        Returns:
            Report dictionary with summary statistics
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        # Breakdown by evaluator type
        breakdown = {}
        for eval_type in EvaluatorType:
            eval_results = [
                r.results.get(eval_type) 
                for r in results 
                if eval_type in r.results
            ]
            if eval_results:
                breakdown[eval_type.value] = {
                    "total": len(eval_results),
                    "passed": sum(1 for r in eval_results if r and r.passed),
                    "failed": sum(1 for r in eval_results if r and not r.passed),
                    "skipped": sum(1 for r in eval_results if r and r.details.get("skipped")),
                }
        
        report = {
            "summary": {
                "total_tasks": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0.0,
            },
            "breakdown_by_evaluator": breakdown,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to: {output_path}")
        
        return report


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_evaluate(
    task_file: str,
    agent_messages: List[Dict[str, Any]],
    outputs_dir: str = "./outputs_cursor",
) -> CompositeEvaluationResult:
    """
    Quick evaluation function for simple use cases.
    
    Args:
        task_file: Path to task JSON file
        agent_messages: Agent's conversation messages
        outputs_dir: Outputs directory
        
    Returns:
        Evaluation result
    """
    # Load task
    with open(task_file) as f:
        task_data = json.load(f)
    
    if isinstance(task_data, list):
        task_data = task_data[0]
    
    task = TaskDefinition.from_task_file(task_data)
    
    # Evaluate
    evaluator = CompositeEvaluator(outputs_dir=outputs_dir)
    return evaluator.evaluate(task=task, agent_trajectory=agent_messages)


def evaluate_trajectory_only(
    task_file: str,
    agent_messages: List[Dict[str, Any]],
    outputs_dir: str = "./outputs_cursor",
) -> float:
    """
    Quick evaluation that only checks action matching.
    
    Returns:
        Score (0.0 or 1.0)
    """
    evaluator = CompositeEvaluator(
        outputs_dir=outputs_dir,
        enable_environment=False,
        enable_communication=False,
        enable_nl_assertions=False,
    )
    
    with open(task_file) as f:
        task_data = json.load(f)
    
    if isinstance(task_data, list):
        task_data = task_data[0]
    
    task = TaskDefinition.from_task_file(task_data)
    result = evaluator.evaluate(task=task, agent_trajectory=agent_messages)
    
    return result.total_score