#!/usr/bin/env python
"""
Run Evaluation - Unified CLI entry point for the evaluation system.

This module provides a command-line interface for running evaluations on
rollout data with support for multiple modes:

1. Prune Mode: Prune redundant tool calls from trajectories
2. All Mode: Run all evaluators and generate comprehensive report
3. Single Mode: Run a specific evaluator only

Usage:
    # Prune trajectories
    python -m evaluator.run_evaluation --mode prune --rollouts-dir rollouts/
    
    # Run all evaluations
    python -m evaluator.run_evaluation --mode all --rollouts-dir rollouts/ --outputs-dir outputs/
    
    # Run single evaluator
    python -m evaluator.run_evaluation --mode single --evaluator action
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from .data_loader import EvaluationDataLoader, EvaluationSample
from .base import TaskDefinition, EvaluatorType, ToolCall
from .trajectory_pruner import TrajectoryPruner, PruningResult
from .trajectory_code_manager import compute_trajectory_hash, TrajectoryCodeManager
from .action_evaluator import ActionEvaluator
from .environment_evaluator import EnvironmentEvaluator
from .nl_assertions_evaluator import NLAssertionsEvaluator
from .subjective_evaluator import SubjectiveQualityEvaluator
from .trajectory_executor import TrajectoryExecutor

# Suppress verbose LiteLLM logs
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Import parallel processing utilities
try:
    from agent_skiller.core.parallel import parallel_process
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False
    logger.warning("parallel_process not available, falling back to sequential processing")


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class SampleEvaluationResult:
    """Evaluation result for a single sample."""
    id: str
    passed: bool
    total_score: float
    pruning_info: Optional[Dict[str, Any]] = None
    evaluators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    task_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    samples: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "samples": self.samples,
        }
    
    def save(self, output_path: Path) -> None:
        """Save report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Report saved to: {output_path}")


# =============================================================================
# Evaluation Runner
# =============================================================================

class EvaluationRunner:
    """
    Main evaluation runner that orchestrates the evaluation process.
    
    This class handles:
    1. Loading samples from rollouts
    2. Optionally pruning trajectories
    3. Running specified evaluators
    4. Generating comprehensive reports
    """
    
    EVALUATOR_TYPES = ["action", "environment", "nl_assertions", "subjective"]
    
    def __init__(
        self,
        rollouts_dir: str = "rollouts/",
        outputs_dir: str = "outputs/",
        mcp_outputs_dir: str = "outputs/",
        verbose: bool = True,
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            rollouts_dir: Directory containing rollout JSONL files
            outputs_dir: Directory containing queries and validated_tasks
            mcp_outputs_dir: Directory containing MCP servers (for execution)
            verbose: Print progress information
        """
        self.rollouts_dir = Path(rollouts_dir)
        self.outputs_dir = Path(outputs_dir)
        self.mcp_outputs_dir = Path(mcp_outputs_dir)
        self.verbose = verbose
        
        # Initialize data loader
        self.data_loader = EvaluationDataLoader(
            rollouts_dir=str(rollouts_dir),
            outputs_dir=str(outputs_dir),
        )
        
        # Initialize pruner
        self.pruner = TrajectoryPruner(outputs_dir=str(mcp_outputs_dir))
        
        # Initialize executor
        self.executor = TrajectoryExecutor(outputs_dir=str(mcp_outputs_dir))
        
        # Initialize code manager for caching
        self.code_manager = TrajectoryCodeManager(outputs_dir=str(outputs_dir))
        
        # Evaluators will be initialized on demand
        self._evaluators: Dict[str, Any] = {}
    
    def _get_evaluator(self, evaluator_type: str):
        """Get or create an evaluator instance."""
        if evaluator_type not in self._evaluators:
            if evaluator_type == "action":
                self._evaluators[evaluator_type] = ActionEvaluator(
                    pruning_index_path=str(self.outputs_dir / "trajectory_code" / "pruning_index.json"),
                    outputs_dir=str(self.outputs_dir),
                )
            elif evaluator_type == "environment":
                self._evaluators[evaluator_type] = EnvironmentEvaluator(
                    outputs_dir=str(self.mcp_outputs_dir)
                )
            elif evaluator_type == "nl_assertions":
                self._evaluators[evaluator_type] = NLAssertionsEvaluator()
            elif evaluator_type == "subjective":
                self._evaluators[evaluator_type] = SubjectiveQualityEvaluator(
                    outputs_dir=str(self.mcp_outputs_dir)
                )
            else:
                raise ValueError(f"Unknown evaluator type: {evaluator_type}")
        
        return self._evaluators[evaluator_type]
    
    def _sample_to_task_definition(self, sample: EvaluationSample) -> TaskDefinition:
        """Convert EvaluationSample to TaskDefinition for evaluators."""
        return TaskDefinition(
            task_id=sample.id,
            trajectory=sample.trajectory,
            entity_context=sample.entity_context,
            instruction=sample.instruction,
            domains=sample.domains,
            is_cross_domain=sample.is_cross_domain,
            nl_assertions=sample.nl_assertions,
        )
    
    def _extract_agent_tool_calls(self, sample: EvaluationSample) -> List[ToolCall]:
        """Extract tool calls from agent's rollout messages."""
        tool_calls = []
        
        for msg in sample.rollout_messages:
            if msg.get("role") != "assistant":
                continue
            
            for tc in (msg.get("tool_calls") or []):
                if "function" in tc:
                    func = tc["function"]
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    
                    # Strip domain prefix
                    func_name = func["name"].split(".")[-1]
                    
                    tool_calls.append(ToolCall(
                        name=func_name,
                        arguments=args,
                    ))
        
        return tool_calls
    
    def run_prune_mode(
        self,
        output_path: Optional[Path] = None,
        use_cache: bool = True,
        max_workers: Optional[int] = None,
    ) -> EvaluationReport:
        """
        Run prune mode - analyze and prune redundant tool calls.
        
        Uses trajectory-level deduplication: samples with the same trajectory
        share the same pruning result.
        
        Args:
            output_path: Path to save pruning results
            use_cache: Whether to use cached pruning results
            max_workers: Maximum parallel workers (None = auto)
            
        Returns:
            EvaluationReport with pruning information
        """
        # Load samples
        samples = self.data_loader.load_all_samples(verbose=False)
        
        # Filter samples with trajectories
        samples_with_trajectory = [s for s in samples if s.trajectory]
        samples_without_trajectory = [s for s in samples if not s.trajectory]
        
        # Run trajectory-level pruning (handles caching, parallelism, verbose output)
        pruning_results = self.pruner.prune_trajectories(
            samples=samples_with_trajectory,
            use_cache=use_cache,
            max_workers=max_workers,
            verbose=self.verbose,
        )
        
        # Build per-sample results
        results = []
        total_original = 0
        total_pruned = 0
        
        # Add results for samples without trajectories
        for sample in samples_without_trajectory:
            results.append(SampleEvaluationResult(
                id=sample.id,
                passed=True,
                total_score=1.0,
                pruning_info={"skipped": True, "reason": "No trajectory defined"},
                task_info={
                    "instruction": sample.instruction[:100] if sample.instruction else "",
                    "domains": sample.domains,
                    "trajectory_length": 0,
                },
            ).to_dict())
        
        # Map pruning results to samples
        for sample in samples_with_trajectory:
            traj_hash = compute_trajectory_hash(sample.trajectory)
            pruning_result = pruning_results.get(traj_hash)
            
            if pruning_result:
                total_original += len(pruning_result.original_trajectory)
                total_pruned += len(pruning_result.pruned_trajectory)
                
                results.append(SampleEvaluationResult(
                    id=sample.id,
                    passed=True,
                    total_score=1.0,
                    pruning_info={
                        "trajectory_hash": traj_hash,
                        "original_trajectory": pruning_result.original_trajectory,
                        "pruned_trajectory": pruning_result.pruned_trajectory,
                        "removed_steps": [
                            {
                                "index": r.step_index,
                                "tool": r.tool_name,
                                "reason": r.reason,
                                "overlapping_source": r.overlapping_source,
                            }
                            for r in pruning_result.redundant_steps
                        ],
                        "pruning_ratio": pruning_result.pruning_ratio,
                        "from_cache": pruning_result.from_cache,
                    },
                    task_info={
                        "instruction": sample.instruction[:200] if sample.instruction else "",
                        "domains": sample.domains,
                        "trajectory_length": len(sample.trajectory),
                    },
                ).to_dict())
            else:
                # Pruning failed for this trajectory
                results.append(SampleEvaluationResult(
                    id=sample.id,
                    passed=False,
                    total_score=0.0,
                    pruning_info={"skipped": True, "reason": "Pruning analysis failed"},
                    error="Trajectory execution or analysis failed",
                    task_info={
                        "instruction": sample.instruction[:100] if sample.instruction else "",
                        "domains": sample.domains,
                        "trajectory_length": len(sample.trajectory),
                    },
                ).to_dict())
        
        # Count unique trajectories
        unique_trajectories = len(pruning_results)
        cache_hits = sum(1 for r in pruning_results.values() if r.from_cache)
        
        # Build report
        report = EvaluationReport(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "mode": "prune",
                "total_samples": len(samples),
                "unique_trajectories": unique_trajectories,
                "rollouts_dir": str(self.rollouts_dir),
                "outputs_dir": str(self.outputs_dir),
            },
            summary={
                "total": len(samples),
                "with_trajectory": len(samples_with_trajectory),
                "without_trajectory": len(samples_without_trajectory),
                "unique_trajectories": unique_trajectories,
                "cache_hits": cache_hits,
                "analyzed": unique_trajectories - cache_hits,
                "total_original_steps": total_original,
                "total_pruned_steps": total_pruned,
                "total_removed_steps": total_original - total_pruned,
                "overall_pruning_ratio": (
                    (total_original - total_pruned) / total_original 
                    if total_original > 0 else 0
                ),
            },
            samples=results,
        )
        
        if output_path:
            report.save(output_path)
        
        return report
    
    def run_all_mode(
        self,
        output_path: Optional[Path] = None,
        enable_pruning: bool = True,
        enable_subjective: bool = False,
    ) -> EvaluationReport:
        """
        Run all evaluators and generate comprehensive report.
        
        Args:
            output_path: Path to save evaluation results
            enable_pruning: Whether to prune trajectories before evaluation
            enable_subjective: Whether to include subjective quality evaluation
            
        Returns:
            EvaluationReport with all evaluation results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("📋 ALL MODE - Complete Evaluation")
            print("=" * 60)
        
        samples = self.data_loader.load_all_samples(verbose=self.verbose)
        
        if self.verbose:
            print(f"\nLoaded {len(samples)} samples for evaluation")
        
        # Get evaluator list
        evaluator_types = ["action", "environment", "nl_assertions"]
        if enable_subjective:
            evaluator_types.append("subjective")
        
        results = []
        summary_by_evaluator = {et: {"passed": 0, "failed": 0} for et in evaluator_types}
        summary_by_domain = {}
        
        for idx, sample in enumerate(samples):
            if self.verbose:
                print(f"\n[{idx + 1}/{len(samples)}] Evaluating: {sample.id[:30]}...")
            
            sample_result = self._evaluate_single_sample(
                sample=sample,
                evaluator_types=evaluator_types,
                enable_pruning=enable_pruning,
            )
            
            results.append(sample_result.to_dict())
            
            # Update summary statistics
            for et, eval_result in sample_result.evaluators.items():
                if eval_result.get("passed"):
                    summary_by_evaluator[et]["passed"] += 1
                else:
                    summary_by_evaluator[et]["failed"] += 1
            
            # Update domain statistics
            for domain in sample.domains:
                if domain not in summary_by_domain:
                    summary_by_domain[domain] = {"passed": 0, "failed": 0}
                if sample_result.passed:
                    summary_by_domain[domain]["passed"] += 1
                else:
                    summary_by_domain[domain]["failed"] += 1
        
        # Calculate overall statistics
        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        failed = total - passed
        
        # Build report
        report = EvaluationReport(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "mode": "all",
                "total_samples": total,
                "rollouts_dir": str(self.rollouts_dir),
                "outputs_dir": str(self.outputs_dir),
                "evaluators": evaluator_types,
                "pruning_enabled": enable_pruning,
            },
            summary={
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0,
                "by_evaluator": summary_by_evaluator,
                "by_domain": summary_by_domain,
            },
            samples=results,
        )
        
        if self.verbose:
            self._print_summary(report)
        
        if output_path:
            report.save(output_path)
        
        return report
    
    def run_single_mode(
        self,
        evaluator_type: str,
        output_path: Optional[Path] = None,
        max_workers: Optional[int] = None,
    ) -> EvaluationReport:
        """
        Run a single evaluator with optional parallel processing.
        
        Args:
            evaluator_type: Type of evaluator to run
            output_path: Path to save evaluation results (JSONL format)
            max_workers: Maximum parallel workers (1 = sequential, >1 = parallel)
            
        Returns:
            EvaluationReport with single evaluator results
        """
        if evaluator_type not in self.EVALUATOR_TYPES:
            raise ValueError(
                f"Unknown evaluator type: {evaluator_type}. "
                f"Valid types: {self.EVALUATOR_TYPES}"
            )
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"🎯 SINGLE MODE - {evaluator_type.upper()} Evaluator")
            print("=" * 60)
        
        samples = self.data_loader.load_all_samples(verbose=self.verbose)
        
        if self.verbose:
            print(f"\nLoaded {len(samples)} samples for evaluation")
        
        # Pre-initialize evaluator before parallel processing to avoid race condition
        # This ensures blueprints are loaded only once in the main thread
        _ = self._get_evaluator(evaluator_type)
        
        # Define evaluation function for single sample
        def evaluate_sample(sample: EvaluationSample) -> SampleEvaluationResult:
            return self._evaluate_single_sample(
                sample=sample,
                evaluator_types=[evaluator_type],
                enable_pruning=False,
            )
        
        # Use parallel if max_workers > 1 and parallel is available
        use_parallel = (max_workers is None or max_workers > 1) and HAS_PARALLEL
        
        # Run evaluation (parallel or sequential)
        if use_parallel and len(samples) > 1:
            if self.verbose:
                print(f"\nRunning parallel evaluation with {max_workers or 'auto'} workers...")
            
            raw_results = parallel_process(
                items=samples,
                process_func=evaluate_sample,
                max_workers=max_workers,
                description=f"Evaluating ({evaluator_type})",
                show_progress=self.verbose,
            )
            
            # Filter out None results and convert to list
            results = [r for r in raw_results if r is not None]
        else:
            # Sequential processing
            results = []
            for idx, sample in enumerate(samples):
                if self.verbose:
                    print(f"\n[{idx + 1}/{len(samples)}] Evaluating: {sample.id[:30]}...")
                
                sample_result = evaluate_sample(sample)
                results.append(sample_result)
        
        # Count pass/fail
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        total = len(results)
        
        # Convert to dict for report
        results_dict = [r.to_dict() for r in results]
        
        report = EvaluationReport(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "mode": "single",
                "evaluator": evaluator_type,
                "total_samples": total,
                "rollouts_dir": str(self.rollouts_dir),
                "outputs_dir": str(self.outputs_dir),
                "parallel": use_parallel,
                "max_workers": max_workers,
            },
            summary={
                "total": total,
                "passed": passed_count,
                "failed": failed_count,
                "pass_rate": passed_count / total if total > 0 else 0,
                "by_evaluator": {
                    evaluator_type: {"passed": passed_count, "failed": failed_count}
                },
            },
            samples=results_dict,
        )
        
        if self.verbose:
            self._print_summary(report)
        
        # Save results in JSONL format
        if output_path:
            self._save_results_jsonl(results, output_path)
        
        return report
    
    def _save_results_jsonl(
        self,
        results: List[SampleEvaluationResult],
        output_path: Path,
    ) -> None:
        """
        Save evaluation results to JSONL file.
        
        Each line contains one sample result with ID matching sample ID.
        
        Args:
            results: List of SampleEvaluationResult
            output_path: Path to save JSONL file
        """
        # Ensure parent directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for result in results:
                if result is not None:
                    f.write(json.dumps(result.to_dict(), default=str) + "\n")
        
        print(f"Results saved to: {output_path} ({len(results)} samples)")
    
    def _evaluate_single_sample(
        self,
        sample: EvaluationSample,
        evaluator_types: List[str],
        enable_pruning: bool = True,
    ) -> SampleEvaluationResult:
        """
        Evaluate a single sample with specified evaluators.
        
        Args:
            sample: The sample to evaluate
            evaluator_types: List of evaluator types to run
            enable_pruning: Whether to apply trajectory pruning
            
        Returns:
            SampleEvaluationResult
        """
        if not sample.trajectory:
            return SampleEvaluationResult(
                id=sample.id,
                passed=False,
                total_score=0.0,
                error="No trajectory defined for sample",
                task_info={
                    "instruction": sample.instruction[:100] if sample.instruction else "",
                    "domains": sample.domains,
                    "trajectory_length": 0,
                },
            )
        
        try:
            # Create task definition
            task = self._sample_to_task_definition(sample)
            
            # Get agent's tool calls
            agent_tool_calls = self._extract_agent_tool_calls(sample)
            
            if not agent_tool_calls:
                return SampleEvaluationResult(
                    id=sample.id,
                    passed=False,
                    total_score=0.0,
                    error="No tool calls found in agent rollout",
                    task_info={
                        "instruction": sample.instruction[:100] if sample.instruction else "",
                        "domains": sample.domains,
                        "trajectory_length": len(sample.trajectory),
                    },
                )
            
            # Execute golden trajectory with caching
            traj_hash = compute_trajectory_hash(sample.trajectory)
            cached_entry = self.code_manager.get_entry_by_hash(traj_hash)
            
            gold_execution = None
            
            # Try cached code first
            if cached_entry and cached_entry.step_param_codes:
                code_path = self.outputs_dir / cached_entry.code_path
                if code_path.exists() and code_path.stat().st_size > 0:
                    try:
                        gold_execution = self.executor.execute_with_codes(
                            task,
                            step_param_codes=cached_entry.step_param_codes,
                            verbose=False,
                        )
                    except Exception as e:
                        logger.warning(f"Cache execute failed for {sample.id}: {e}")
            
            # No cache: execute with LLM + save
            if gold_execution is None:
                gold_execution = self.executor.execute(task, verbose=False)
                
                if gold_execution.success and gold_execution.step_param_codes:
                    try:
                        self.code_manager.save_code(
                            trajectory=sample.trajectory,
                            domains=sample.domains,
                            is_cross_domain=sample.is_cross_domain,
                            generated_code=gold_execution.generated_code or "",
                            entity_mappings={},
                            step_param_codes=gold_execution.step_param_codes,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save code for {sample.id}: {e}")
            
            # Apply pruning if enabled
            pruning_info = None
            if enable_pruning and gold_execution.success:
                execution_results = [
                    {
                        "params": step.tool_call.arguments,
                        "result": step.tool_call.result,
                    }
                    for step in gold_execution.steps
                ]
                
                pruning_result = self.pruner.analyze_trajectory(
                    instruction=sample.instruction,
                    trajectory=sample.trajectory,
                    execution_results=execution_results,
                    verbose=False,
                )
                
                pruning_info = {
                    "original_trajectory": pruning_result.original_trajectory,
                    "pruned_trajectory": pruning_result.pruned_trajectory,
                    "removed_steps": [
                        {
                            "index": r.step_index,
                            "tool": r.tool_name,
                            "reason": r.reason,
                        }
                        for r in pruning_result.redundant_steps
                    ],
                    "pruning_ratio": pruning_result.pruning_ratio,
                }
                
                # Update task with pruned trajectory
                task.trajectory = pruning_result.pruned_trajectory
            
            # Run evaluators
            evaluator_results = {}
            all_passed = True
            total_score = 1.0
            
            for eval_type in evaluator_types:
                try:
                    evaluator = self._get_evaluator(eval_type)
                    
                    # Different evaluators have different signatures
                    if eval_type == "action":
                        result = evaluator.evaluate(
                            sample=sample,
                            gold_execution=gold_execution,
                        )
                    elif eval_type == "environment":
                        result = evaluator.evaluate(
                            task=task,
                            gold_execution=gold_execution,
                            agent_trajectory=sample.rollout_messages,
                        )
                    elif eval_type == "nl_assertions":
                        result = evaluator.evaluate(
                            task=task,
                            gold_execution=gold_execution,
                            agent_trajectory=sample.rollout_messages,
                        )
                    elif eval_type == "subjective":
                        result = evaluator.evaluate(
                            task=task,
                            gold_execution=gold_execution,
                            agent_trajectory=sample.rollout_messages,
                        )
                    else:
                        continue
                    
                    if result:
                        evaluator_results[eval_type] = {
                            "passed": result.passed,
                            "score": result.score,
                            "details": result.details,
                            "reasoning": result.reasoning,
                        }
                        
                        if not result.passed:
                            all_passed = False
                        total_score *= result.score
                    else:
                        # Evaluator returned None
                        evaluator_results[eval_type] = {
                            "passed": False,
                            "score": 0.0,
                            "details": {},
                            "reasoning": "Evaluator returned no result",
                        }
                        all_passed = False
                        total_score = 0.0
                        
                except Exception as e:
                    logger.warning(f"Evaluator {eval_type} failed for {sample.id}: {e}")
                    evaluator_results[eval_type] = {
                        "passed": False,
                        "score": 0.0,
                        "details": {},
                        "reasoning": f"Evaluator error: {str(e)}",
                    }
                    all_passed = False
                    total_score = 0.0
            
            if self.verbose:
                status = "✅ PASS" if all_passed else "❌ FAIL"
                print(f"  {status} (score: {total_score:.2f})")
                for et, er in evaluator_results.items():
                    et_status = "✓" if er["passed"] else "✗"
                    print(f"    {et_status} {et}: {er.get('reasoning', '')}")
                    
                    # Show detailed failure info for action evaluator
                    if et == "action" and not er["passed"]:
                        details = er.get("details", {})
                        failure_details = details.get("failure_details", "")
                        if failure_details:
                            print(failure_details)
            
            return SampleEvaluationResult(
                id=sample.id,
                passed=all_passed,
                total_score=total_score,
                pruning_info=pruning_info,
                evaluators=evaluator_results,
                task_info={
                    "instruction": sample.instruction[:200] if sample.instruction else "",
                    "domains": sample.domains,
                    "trajectory_length": len(sample.trajectory),
                    "agent_tool_calls": len(agent_tool_calls),
                },
            )
            
        except Exception as e:
            logger.exception(f"Failed to evaluate sample {sample.id}")
            return SampleEvaluationResult(
                id=sample.id,
                passed=False,
                total_score=0.0,
                error=str(e),
                task_info={
                    "instruction": sample.instruction[:100] if sample.instruction else "",
                    "domains": sample.domains,
                    "trajectory_length": len(sample.trajectory) if sample.trajectory else 0,
                },
            )
    
    def _print_summary(self, report: EvaluationReport) -> None:
        """Print evaluation summary to console."""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        s = report.summary
        print(f"Total samples:     {s['total']}")
        print(f"Passed:            {s['passed']} ({s['pass_rate']:.1%})")
        print(f"Failed:            {s['failed']}")
        
        if "by_evaluator" in s:
            print("\nBy Evaluator:")
            for et, stats in s["by_evaluator"].items():
                total = stats["passed"] + stats["failed"]
                rate = stats["passed"] / total if total > 0 else 0
                print(f"  {et:20s}: {stats['passed']:4d} ✓  {stats['failed']:4d} ✗  ({rate:.1%})")
        
        if "by_domain" in s and s["by_domain"]:
            print("\nBy Domain:")
            for domain, stats in s["by_domain"].items():
                total = stats["passed"] + stats["failed"]
                rate = stats["passed"] / total if total > 0 else 0
                print(f"  {domain:20s}: {stats['passed']:4d} ✓  {stats['failed']:4d} ✗  ({rate:.1%})")
        
        print("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run evaluations on rollout data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prune trajectories
  python -m evaluator.run_evaluation --mode prune
  
  # Run all evaluations
  python -m evaluator.run_evaluation --mode all
  
  # Run single evaluator
  python -m evaluator.run_evaluation --mode single --evaluator action
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["prune", "all", "single"],
        default="all",
        help="Evaluation mode: prune, all, or single (default: all)"
    )
    
    parser.add_argument(
        "--evaluator", "-e",
        choices=["action", "environment", "nl_assertions", "subjective"],
        help="Evaluator type for single mode"
    )
    
    parser.add_argument(
        "--rollouts-dir", "-r",
        type=str,
        default="rollouts/",
        help="Directory containing rollout JSONL files (default: rollouts/)"
    )
    
    parser.add_argument(
        "--outputs-dir", "-o",
        type=str,
        default="outputs/",
        help="Directory containing queries and validated_tasks (default: outputs/)"
    )
    
    parser.add_argument(
        "--mcp-outputs-dir",
        type=str,
        default=None,
        help="Directory containing MCP servers (default: same as outputs-dir)"
    )
    
    parser.add_argument(
        "--output", "-O",
        type=str,
        default="outputs/evaluation/results.jsonl",
        help="Output file path for evaluation results (JSONL format, default: outputs/evaluation/results.jsonl)"
    )
    
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable trajectory pruning in 'all' mode"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache for pruning (re-analyze all trajectories)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel workers (default: 8)"
    )
    
    parser.add_argument(
        "--enable-subjective",
        action="store_true",
        help="Enable subjective quality evaluation (requires LLM)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose output (default: True)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "single" and not args.evaluator:
        parser.error("--evaluator is required for single mode")
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    verbose = not args.quiet and args.verbose
    mcp_outputs_dir = args.mcp_outputs_dir or args.outputs_dir
    
    # Initialize runner
    runner = EvaluationRunner(
        rollouts_dir=args.rollouts_dir,
        outputs_dir=args.outputs_dir,
        mcp_outputs_dir=mcp_outputs_dir,
        verbose=verbose,
    )
    
    output_path = Path(args.output)
    
    # Run appropriate mode
    if args.mode == "prune":
        report = runner.run_prune_mode(
            output_path=output_path,
            use_cache=not args.no_cache,
            max_workers=args.max_workers,
        )
    elif args.mode == "all":
        report = runner.run_all_mode(
            output_path=output_path,
            enable_pruning=not args.no_prune,
            enable_subjective=args.enable_subjective,
        )
    elif args.mode == "single":
        report = runner.run_single_mode(
            evaluator_type=args.evaluator,
            output_path=output_path,
            max_workers=args.max_workers,
        )
    
    # Exit with appropriate code
    if report.summary.get("failed", 0) > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

