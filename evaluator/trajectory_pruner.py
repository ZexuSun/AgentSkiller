"""
Trajectory Pruner - Identifies and removes redundant tool calls from golden trajectories.

This module analyzes executed trajectories to find tool calls that don't provide
additional information gain, allowing for more efficient golden trajectory definitions.

Key Features:
1. Trajectory-level caching: Same trajectory_hash shares pruning result
2. Parallel processing: Multiple unique trajectories analyzed concurrently
3. Rich progress display: Clean output with progress bars

Example redundancy:
- get_student_bank_accounts returns all account info including balance
- check_account_balance is redundant if called after, as balance is already known
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .trajectory_code_manager import TrajectoryCodeManager, PruningResultEntry, compute_trajectory_hash

if TYPE_CHECKING:
    from .data_loader import EvaluationSample

logger = logging.getLogger(__name__)

# Suppress verbose LiteLLM logs
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

console = Console()


@dataclass
class RedundancyResult:
    """Result of redundancy analysis for a single tool call."""
    step_index: int
    tool_name: str
    is_redundant: bool
    reason: str
    overlapping_source: Optional[str] = None


@dataclass
class PruningResult:
    """Result of trajectory pruning."""
    trajectory_hash: str
    original_trajectory: List[str]
    pruned_trajectory: List[str]
    redundant_steps: List[RedundancyResult]
    kept_steps: List[int]  # Indices of steps kept
    pruning_ratio: float  # % of steps removed
    domains: List[str] = field(default_factory=list)
    from_cache: bool = False
    
    def to_cache_entry(self) -> Dict[str, Any]:
        """Convert to format for cache storage."""
        return {
            "trajectory_hash": self.trajectory_hash,
            "original_trajectory": self.original_trajectory,
            "pruned_trajectory": self.pruned_trajectory,
            "redundant_steps": [
                {
                    "index": r.step_index,
                    "tool": r.tool_name,
                    "reason": r.reason,
                    "overlapping_source": r.overlapping_source,
                }
                for r in self.redundant_steps
            ],
            "pruning_ratio": self.pruning_ratio,
            "domains": self.domains,
        }
    
    @classmethod
    def from_cache_entry(cls, entry: PruningResultEntry) -> "PruningResult":
        """Create from cache entry."""
        redundant_steps = [
            RedundancyResult(
                step_index=s["index"],
                tool_name=s["tool"],
                is_redundant=True,
                reason=s["reason"],
                overlapping_source=s.get("overlapping_source"),
            )
            for s in entry.redundant_steps
        ]
        
        kept_steps = entry.kept_indices
        
        return cls(
            trajectory_hash=entry.trajectory_hash,
            original_trajectory=entry.original_trajectory,
            pruned_trajectory=entry.pruned_trajectory,
            redundant_steps=redundant_steps,
            kept_steps=kept_steps,
            pruning_ratio=entry.pruning_ratio,
            domains=entry.domains,
            from_cache=True,
        )


@dataclass
class TrajectoryInfo:
    """
    Information about a unique trajectory to prune.
    
    The entity_context is sourced from EvaluationSample, which loads it
    via combo_id lookup from validated_combos.json. This includes both
    entity_instances and value_domain_samples.
    """
    trajectory_hash: str
    trajectory: List[str]
    domains: List[str]
    instruction: str  # Representative instruction
    sample_ids: List[str]  # All samples using this trajectory
    entity_context: Dict[str, Any]  # From combo: entity_instances + value_domain_samples


REDUNDANCY_CHECK_PROMPT = """## Tool Call Redundancy Analysis

### Task Instruction
The user's goal is:
```
{instruction}
```

### Execution History
The following tool calls have been executed so far:

{execution_history}

### Current Tool Call
Just executed: **{current_tool_name}**

Input parameters:
```json
{current_params}
```

Output result:
```json
{current_result}
```

### Analysis Task

Determine whether this tool call (**{current_tool_name}**) is **REDUNDANT** or **NECESSARY**.

A tool call is **REDUNDANT** if:
1. **Information already available**: The information returned by this tool was already present in a previous tool call's output.
   - Example: `check_account_balance` after `get_student_bank_accounts` - if the account list already included balance for each account, then checking balance separately is redundant.
   
2. **No new information for the task**: The tool's output doesn't contribute any new information needed to fulfill the user's instruction.
   - Example: Calling `get_course_details` twice for the same course.

A tool call is **NECESSARY** if:
1. **Provides new information**: The output contains data not available from any previous call.
2. **Performs a required action**: The tool performs a state-changing operation required by the task (e.g., `enroll_in_course`, `schedule_appointment`).
3. **Authorization**: Authorization calls (`authorize_*`) are always necessary.

### Important Notes
- Focus on **information overlap**, not just function name similarity.
- A tool can be redundant even if its name suggests it does something different, as long as the needed information was already in previous outputs.
- State-changing operations (create, update, delete, enroll, schedule, etc.) are generally NOT redundant.

### Output Format
Respond with a JSON object:
```json
{{
    "is_redundant": true/false,
    "reason": "Brief explanation of why this tool call is redundant or necessary",
    "overlapping_source": "If redundant, which previous tool call's output already contained this information (e.g., 'step_3_result from get_student_bank_accounts')"
}}
```
"""


class TrajectoryPruner:
    """
    Analyzes and prunes redundant tool calls from trajectories.
    
    Features:
    - Trajectory-level caching: Same trajectory_hash shares pruning result
    - Parallel processing: Multiple trajectories analyzed concurrently
    - Rich progress display: Clean output with progress bars
    
    Usage:
        pruner = TrajectoryPruner(outputs_dir="outputs")
        
        # Prune multiple samples (trajectory-level deduplication)
        results = pruner.prune_trajectories(
            samples=evaluation_samples,
            use_cache=True,
            max_workers=4,
        )
        
        # Or analyze a single trajectory
        result = pruner.analyze_trajectory(
            instruction="...",
            trajectory=["func1", "func2", ...],
            execution_results=[{...}, {...}, ...],
            domains=["DomainA"],
        )
    """
    
    # Tool name patterns that indicate state-changing operations (never redundant)
    STATE_CHANGING_PATTERNS = [
        "create", "add", "insert", "new",
        "update", "modify", "edit", "change",
        "delete", "remove", "cancel",
        "enroll", "unenroll", "register", "unregister",
        "schedule", "book", "reserve",
        "apply", "submit", "confirm",
        "authorize", "authenticate", "login",
        "transfer", "deposit", "withdraw",
        "assign", "unassign",
    ]
    
    def __init__(
        self,
        outputs_dir: str = "./outputs",
        llm_model: str = None,
    ):
        self.outputs_dir = Path(outputs_dir)
        self._llm_model = llm_model
        self._llm_client = None
        self._code_manager = None
        self._executor = None
    
    @property
    def llm_client(self):
        if self._llm_client is None:
            from agentskiller.core.llm_client import get_client
            self._llm_client = get_client()
        return self._llm_client
    
    @property
    def llm_model(self):
        if self._llm_model:
            return self._llm_model
        try:
            from agentskiller.config import get_settings
            return get_settings().llm.fast_model
        except Exception:
            return None
    
    @property
    def code_manager(self) -> TrajectoryCodeManager:
        if self._code_manager is None:
            self._code_manager = TrajectoryCodeManager(outputs_dir=str(self.outputs_dir))
        return self._code_manager
    
    @property
    def executor(self):
        if self._executor is None:
            from .trajectory_executor import TrajectoryExecutor
            self._executor = TrajectoryExecutor(outputs_dir=str(self.outputs_dir))
        return self._executor
    
    def is_state_changing(self, tool_name: str) -> bool:
        """Check if a tool name indicates a state-changing operation."""
        tool_lower = tool_name.lower()
        return any(pattern in tool_lower for pattern in self.STATE_CHANGING_PATTERNS)
    
    def prune_trajectories(
        self,
        samples: List["EvaluationSample"],
        use_cache: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, PruningResult]:
        """
        Prune trajectories for multiple samples.
        
        This is the main entry point. It:
        1. Groups samples by trajectory_hash (deduplication)
        2. Checks cache for existing results
        3. Processes new trajectories in parallel
        4. Returns results keyed by trajectory_hash
        
        Args:
            samples: List of EvaluationSample objects
            use_cache: Whether to use cached results
            max_workers: Maximum parallel workers (None = auto)
            verbose: Whether to show progress
            
        Returns:
            Dict mapping trajectory_hash -> PruningResult
        """
        if verbose:
            console.print("\n[bold cyan]Trajectory Pruning[/bold cyan]")
            console.print("━" * 50)
        
        # Step 1: Group samples by trajectory
        trajectory_groups = self._group_by_trajectory(samples)
        
        if verbose:
            console.print(f"Found [bold]{len(samples)}[/bold] samples → [bold]{len(trajectory_groups)}[/bold] unique trajectories")
        
        # Step 2: Check cache
        results: Dict[str, PruningResult] = {}
        to_analyze: List[TrajectoryInfo] = []
        
        for traj_hash, traj_info in trajectory_groups.items():
            if use_cache:
                cached = self.code_manager.get_pruning_result_by_hash(traj_hash)
                if cached:
                    results[traj_hash] = PruningResult.from_cache_entry(cached)
                    continue
            
            to_analyze.append(traj_info)
        
        cache_hits = len(results)
        
        if verbose:
            console.print(f"Cache hits: [green]{cache_hits}[/green] | To analyze: [yellow]{len(to_analyze)}[/yellow]")
        
        if not to_analyze:
            if verbose:
                console.print("[green]All trajectories found in cache![/green]")
            return results
        
        # Step 3: Process new trajectories in parallel
        if verbose:
            console.print()
        
        new_results = self._process_trajectories_parallel(
            to_analyze,
            max_workers=max_workers,
            verbose=verbose,
        )
        
        # Step 4: Merge results (already saved in _analyze_single_trajectory)
        for traj_hash, result in new_results.items():
            results[traj_hash] = result
        
        # Step 5: Print summary
        if verbose:
            self._print_summary(results, trajectory_groups)
        
        return results
    
    def _group_by_trajectory(
        self,
        samples: List["EvaluationSample"],
    ) -> Dict[str, TrajectoryInfo]:
        """Group samples by trajectory_hash."""
        groups: Dict[str, TrajectoryInfo] = {}
        
        for sample in samples:
            if not sample.trajectory:
                continue
            
            traj_hash = compute_trajectory_hash(sample.trajectory)
            
            if traj_hash not in groups:
                groups[traj_hash] = TrajectoryInfo(
                    trajectory_hash=traj_hash,
                    trajectory=sample.trajectory,
                    domains=sample.domains,
                    instruction=sample.instruction,
                    sample_ids=[sample.id],
                    entity_context=sample.entity_context,
                )
            else:
                groups[traj_hash].sample_ids.append(sample.id)
        
        return groups
    
    def _process_trajectories_parallel(
        self,
        trajectories: List[TrajectoryInfo],
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, PruningResult]:
        """Process multiple trajectories in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        try:
            from agentskiller.config import get_settings
            workers = max_workers or get_settings().workflow.max_workers
        except Exception:
            workers = max_workers or 4
        
        results: Dict[str, PruningResult] = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing trajectories", total=len(trajectories))
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_traj = {
                    executor.submit(self._analyze_single_trajectory, traj_info): traj_info
                    for traj_info in trajectories
                }
                
                for future in as_completed(future_to_traj):
                    traj_info = future_to_traj[future]
                    try:
                        result = future.result()
                        if result:
                            results[traj_info.trajectory_hash] = result
                            
                            # Print details for this trajectory
                            if verbose and result.redundant_steps:
                                self._print_trajectory_result(traj_info, result)
                                
                    except Exception as e:
                        logger.error(f"Failed to analyze trajectory {traj_info.trajectory_hash}: {e}")
                    
                    progress.advance(task)
        
        return results
    
    def _analyze_single_trajectory(self, traj_info: TrajectoryInfo) -> Optional[PruningResult]:
        """
        Analyze a single trajectory (called in parallel).
        
        This method implements caching logic:
        1. Check if cached execution code exists
        2. Validate/fix cached code using AST analysis
        3. Use cached code if valid, otherwise regenerate
        """
        from .base import TaskDefinition
        from .trajectory_code_manager import analyze_param_code, fix_hardcoded_ctx, validate_ctx_keys
        
        try:
            # Create task definition
            print(traj_info.domains)
            task_def = TaskDefinition(
                task_id=traj_info.trajectory_hash,
                trajectory=traj_info.trajectory,
                entity_context=traj_info.entity_context,
                domains=traj_info.domains,
                is_cross_domain=len(traj_info.domains) > 1,
            )
            
            # Step 1: Check for cached execution code
            cached_entry = self.code_manager.get_entry(traj_info.trajectory)
            execution = None
            
            # Diagnostic logging for cache lookup
            logger.info(f"🔍 Cache lookup for {traj_info.trajectory_hash}: {'FOUND' if cached_entry else 'NOT FOUND'}")
            if cached_entry:
                logger.info(f"  - step_param_codes count: {len(cached_entry.step_param_codes) if cached_entry.step_param_codes else 0}")
                logger.info(f"  - code_path: {cached_entry.code_path}")
                logger.info(f"  - domains: {cached_entry.domains}")
                logger.info(f"  - is_cross_domain: {cached_entry.is_cross_domain}")
            else:
                logger.info(f"  - No cached entry found in index")
            
            if cached_entry and cached_entry.step_param_codes:
                # Step 2: Validate and fix each step's code
                fixed_codes = []
                codes_valid = True
                
                logger.info(f"  - Validating {len(cached_entry.step_param_codes)} cached step codes...")
                for step_idx, step_code in enumerate(cached_entry.step_param_codes):
                    analysis = analyze_param_code(step_code)
                    
                    if analysis["parse_error"]:
                        # Code can't be parsed, regenerate
                        logger.warning(f"  ❌ Step {step_idx} code parse error: {analysis['parse_error']}")
                        codes_valid = False
                        break
                    
                    if analysis["has_hardcoded_ctx"]:
                        # Fix hardcoded ctx to make it reusable
                        fixed_code = fix_hardcoded_ctx(step_code, analysis["hardcoded_ctx_value"])
                        fixed_codes.append(fixed_code)
                        logger.info(f"  ✓ Step {step_idx}: Fixed hardcoded ctx")
                    else:
                        # Validate ctx keys are available
                        is_valid, missing_keys = validate_ctx_keys(
                            analysis["ctx_keys_used"],
                            traj_info.entity_context,
                        )
                        
                        if is_valid:
                            fixed_codes.append(step_code)
                            logger.debug(f"  ✓ Step {step_idx}: Code validated (ctx_keys: {analysis['ctx_keys_used']})")
                        else:
                            logger.warning(f"  ❌ Step {step_idx}: Missing ctx keys: {missing_keys}")
                            logger.warning(f"     Required keys: {analysis['ctx_keys_used']}")
                            logger.warning(f"     Available keys: {list(traj_info.entity_context.keys())[:10]}...")
                            codes_valid = False
                            break
                
                if codes_valid and len(fixed_codes) == len(cached_entry.step_param_codes):
                    # Step 3: Use fixed codes to execute (no LLM calls)
                    logger.info(f"✅ Using cached code for {traj_info.trajectory_hash} ({len(fixed_codes)} steps)")
                    execution = self.executor.execute_with_codes(
                        task_def,
                        step_param_codes=fixed_codes,
                        verbose=False,
                    )
                else:
                    logger.warning(f"❌ Cache validation failed for {traj_info.trajectory_hash}")
                    logger.warning(f"   - codes_valid: {codes_valid}")
                    logger.warning(f"   - fixed_codes count: {len(fixed_codes)}")
                    logger.warning(f"   - expected count: {len(cached_entry.step_param_codes)}")
            elif cached_entry:
                logger.warning(f"❌ Cache entry found but step_param_codes is empty for {traj_info.trajectory_hash}")
            
            # Step 4: If no cached code or invalid, generate new
            if execution is None:
                logger.info(f"🔄 Generating new code for {traj_info.trajectory_hash} (cache miss or validation failed)")
                execution = self.executor.execute(task_def, verbose=True)
                
                # Save generated codes to cache for future reuse
                if execution.success and execution.step_param_codes:
                    try:
                        self.code_manager.save_code(
                            trajectory=traj_info.trajectory,
                            domains=traj_info.domains,
                            is_cross_domain=len(traj_info.domains) > 1,
                            generated_code=execution.generated_code or "",  # Full generated code
                            entity_mappings={},
                            step_param_codes=execution.step_param_codes,
                        )
                        logger.debug(f"Saved execution code to cache")
                    except Exception as e:
                        logger.warning(f"Failed to save code to cache: {e}")
            
            if not execution.success:
                logger.warning(f"Execution failed for {traj_info.trajectory_hash}: {execution.errors}")
                return None
            
            # Build execution results
            execution_results = [
                {
                    "params": step.tool_call.arguments,
                    "result": step.tool_call.result,
                }
                for step in execution.steps
            ]
            
            # Analyze for redundancy
            result = self.analyze_trajectory(
                instruction=traj_info.instruction,
                trajectory=traj_info.trajectory,
                execution_results=execution_results,
                domains=traj_info.domains,
                verbose=False,
            )
            
            # Immediately save to disk (thread-safe via code_manager._lock)
            # Only save if execution was successful (all steps succeeded)
            if result:
                try:
                    self.code_manager.save_pruning_result(
                        trajectory=traj_info.trajectory,
                        pruned_trajectory=result.pruned_trajectory,
                        redundant_steps=[
                            {
                                "index": r.step_index,
                                "tool": r.tool_name,
                                "reason": r.reason,
                                "overlapping_source": r.overlapping_source,
                            }
                            for r in result.redundant_steps
                        ],
                        domains=traj_info.domains,
                    )
                    result.from_cache = False  # Mark as freshly analyzed
                except Exception as save_err:
                    logger.warning(f"Failed to save pruning result for {traj_info.trajectory_hash}: {save_err}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {traj_info.trajectory_hash}: {e}")
            return None
    
    def analyze_trajectory(
        self,
        instruction: str,
        trajectory: List[str],
        execution_results: List[Dict[str, Any]],
        domains: List[str] = None,
        verbose: bool = False,
    ) -> PruningResult:
        """
        Analyze a complete trajectory for redundant tool calls.
        
        Args:
            instruction: The user's task instruction
            trajectory: List of tool names in order
            execution_results: List of execution results for each tool call
                Each entry should have: {"params": {...}, "result": {...}}
            domains: List of domains involved
            verbose: Whether to print progress
            
        Returns:
            PruningResult with original/pruned trajectories and analysis
        """
        if len(trajectory) != len(execution_results):
            raise ValueError(
                f"Trajectory length ({len(trajectory)}) doesn't match "
                f"results length ({len(execution_results)})"
            )
        
        traj_hash = compute_trajectory_hash(trajectory)
        redundancy_results: List[RedundancyResult] = []
        execution_history: List[Dict[str, Any]] = []
        
        for idx, (tool_name, exec_result) in enumerate(zip(trajectory, execution_results)):
            params = exec_result.get("params", {})
            result = exec_result.get("result", {})
            
            # Check redundancy
            redundancy = self._check_redundancy(
                instruction=instruction,
                execution_history=execution_history,
                current_tool_name=tool_name,
                current_params=params,
                current_result=result,
            )
            redundancy_results.append(redundancy)
            
            # Add to history for next iteration
            execution_history.append({
                "step_index": idx,
                "tool_name": tool_name,
                "params": params,
                "result": result,
            })
        
        # Build pruned trajectory
        kept_steps = [
            r.step_index for r in redundancy_results if not r.is_redundant
        ]
        pruned_trajectory = [trajectory[i] for i in kept_steps]
        redundant_steps = [r for r in redundancy_results if r.is_redundant]
        
        # Calculate pruning ratio
        removed = len(trajectory) - len(pruned_trajectory)
        pruning_ratio = removed / len(trajectory) if trajectory else 0.0
        
        return PruningResult(
            trajectory_hash=traj_hash,
            original_trajectory=trajectory,
            pruned_trajectory=pruned_trajectory,
            redundant_steps=redundant_steps,
            kept_steps=kept_steps,
            pruning_ratio=pruning_ratio,
            domains=domains or [],
            from_cache=False,
        )
    
    def _check_redundancy(
        self,
        instruction: str,
        execution_history: List[Dict[str, Any]],
        current_tool_name: str,
        current_params: Dict[str, Any],
        current_result: Any,
    ) -> RedundancyResult:
        """Check if a single tool call is redundant given execution history."""
        step_index = len(execution_history)
        
        # Quick checks - state-changing operations are never redundant
        if self.is_state_changing(current_tool_name):
            return RedundancyResult(
                step_index=step_index,
                tool_name=current_tool_name,
                is_redundant=False,
                reason=f"State-changing operation: {current_tool_name}",
            )
        
        # If no execution history, first call is never redundant
        if not execution_history:
            return RedundancyResult(
                step_index=step_index,
                tool_name=current_tool_name,
                is_redundant=False,
                reason="First tool call in trajectory",
            )
        
        # Format execution history for prompt
        history_str = self._format_execution_history(execution_history)
        
        prompt = REDUNDANCY_CHECK_PROMPT.format(
            instruction=instruction,
            execution_history=history_str,
            current_tool_name=current_tool_name,
            current_params=json.dumps(current_params, indent=2, default=str),
            current_result=json.dumps(current_result, indent=2, default=str)[:2000],
        )
        
        try:
            response = self.llm_client.chat(
                query=prompt,
                model=self.llm_model,
            )
            
            result = response.parse_json()
            
            return RedundancyResult(
                step_index=step_index,
                tool_name=current_tool_name,
                is_redundant=result.get("is_redundant", False),
                reason=result.get("reason", ""),
                overlapping_source=result.get("overlapping_source"),
            )
            
        except Exception as e:
            logger.warning(f"Failed to check redundancy for {current_tool_name}: {e}")
            return RedundancyResult(
                step_index=step_index,
                tool_name=current_tool_name,
                is_redundant=False,
                reason=f"Analysis failed: {e}",
            )
    
    def _format_execution_history(self, history: List[Dict[str, Any]]) -> str:
        """Format execution history for the prompt."""
        if not history:
            return "(No previous tool calls)"
        
        lines = []
        for entry in history:
            idx = entry.get("step_index", 0)
            tool = entry.get("tool_name", "unknown")
            params = entry.get("params", {})
            result = entry.get("result", {})
            
            # Truncate long results
            result_str = json.dumps(result, default=str)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            
            lines.append(f"""
**Step {idx}: {tool}**
Parameters: {json.dumps(params, default=str)}
Result: {result_str}
""")
        
        return "\n".join(lines)
    
    def _print_trajectory_result(self, traj_info: TrajectoryInfo, result: PruningResult) -> None:
        """Print details for a pruned trajectory."""
        domain_str = traj_info.domains[0] if traj_info.domains else "unknown"
        removed_count = len(result.redundant_steps)
        
        console.print(f"\n[bold blue]{result.trajectory_hash}[/bold blue] ([dim]{domain_str}[/dim])")
        console.print(f"   {len(result.original_trajectory)} steps → {len(result.pruned_trajectory)} steps ", end="")
        console.print(f"[yellow](removed {removed_count})[/yellow]")
        
        for r in result.redundant_steps:
            console.print(f"   [red]✗[/red] Step {r.step_index}: [dim]{r.tool_name}[/dim]")
            reason = r.reason[:60] + "..." if len(r.reason) > 60 else r.reason
            console.print(f"     [dim]{reason}[/dim]")
    
    def _print_summary(
        self,
        results: Dict[str, PruningResult],
        trajectory_groups: Dict[str, TrajectoryInfo],
    ) -> None:
        """Print summary of pruning results."""
        console.print()
        
        total_original = sum(len(r.original_trajectory) for r in results.values())
        total_pruned = sum(len(r.pruned_trajectory) for r in results.values())
        total_removed = total_original - total_pruned
        
        cache_hits = sum(1 for r in results.values() if r.from_cache)
        analyzed = len(results) - cache_hits
        
        # Summary table
        table = Table(title="Pruning Summary", show_header=False, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")
        
        table.add_row("Trajectories", str(len(results)))
        table.add_row("  From cache", f"[green]{cache_hits}[/green]")
        table.add_row("  Analyzed", f"[yellow]{analyzed}[/yellow]")
        table.add_row("Total steps (original)", str(total_original))
        table.add_row("Total steps (pruned)", str(total_pruned))
        table.add_row("Steps removed", f"[red]{total_removed}[/red]")
        table.add_row("Pruning ratio", f"{total_removed/total_original*100:.1f}%" if total_original else "N/A")
        
        console.print(table)
        console.print()


# =============================================================================
# Legacy functions for backward compatibility
# =============================================================================

def prune_task_directory(
    tasks_dir: Path,
    outputs_dir: str = "outputs",
    output_suffix: str = "_pruned",
) -> Dict[str, Any]:
    """
    Prune all task files in a directory (legacy function).
    
    For new code, use TrajectoryPruner.prune_trajectories() instead.
    """
    from .base import TaskDefinition
    from .trajectory_executor import TrajectoryExecutor
    
    pruner = TrajectoryPruner(outputs_dir=outputs_dir)
    executor = TrajectoryExecutor(outputs_dir=outputs_dir)
    
    task_files = list(tasks_dir.glob("*.json"))
    task_files = [f for f in task_files if output_suffix not in f.stem]
    
    console.print(f"\n[bold]Trajectory Pruning[/bold]")
    console.print("━" * 50)
    console.print(f"Found {len(task_files)} task files to process")
    
    all_results = []
    
    for task_file in task_files:
        console.print(f"\n📄 Processing: {task_file.name}")
        
        try:
            with open(task_file) as f:
                tasks = json.load(f)
            
            if not isinstance(tasks, list):
                tasks = [tasks]
            
            pruned_tasks = []
            
            for task_data in tasks:
                task_def = TaskDefinition(
                    task_id=task_file.stem,
                    trajectory=task_data["trajectory"],
                    entity_context=task_data.get("entity_context", {}),
                    domains=task_data.get("domains", []),
                    is_cross_domain=task_data.get("is_cross_domain", False),
                )
                
                instruction = task_data.get("instantiated_task", {}).get("instruction", "")
                
                execution = executor.execute(task_def, verbose=False)
                
                if not execution.success:
                    pruned_tasks.append(task_data)
                    continue
                
                execution_results = [
                    {"params": step.tool_call.arguments, "result": step.tool_call.result}
                    for step in execution.steps
                ]
                
                pruning_result = pruner.analyze_trajectory(
                    instruction=instruction,
                    trajectory=task_def.trajectory,
                    execution_results=execution_results,
                    domains=task_def.domains,
                )
                
                pruned_task = task_data.copy()
                pruned_task["original_trajectory"] = task_data["trajectory"]
                pruned_task["trajectory"] = pruning_result.pruned_trajectory
                pruned_task["pruning_info"] = pruning_result.to_cache_entry()
                
                pruned_tasks.append(pruned_task)
                
                all_results.append({
                    "original_length": len(pruning_result.original_trajectory),
                    "pruned_length": len(pruning_result.pruned_trajectory),
                    "removed": len(pruning_result.redundant_steps),
                    "ratio": pruning_result.pruning_ratio,
                })
            
            output_path = task_file.parent / f"{task_file.stem}{output_suffix}.json"
            with open(output_path, 'w') as f:
                json.dump(pruned_tasks, f, indent=2)
                
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            logger.exception(f"Failed to prune {task_file}")
    
    total_original = sum(r["original_length"] for r in all_results)
    total_pruned = sum(r["pruned_length"] for r in all_results)
    total_removed = sum(r["removed"] for r in all_results)
    
    return {
        "files_processed": len(all_results),
        "total_original_steps": total_original,
        "total_pruned_steps": total_pruned,
        "total_removed": total_removed,
        "overall_pruning_ratio": total_removed / total_original if total_original else 0,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prune redundant tool calls from trajectories")
    parser.add_argument("--tasks-dir", "-d", type=str, required=True, help="Directory with task files")
    parser.add_argument("--outputs-dir", "-o", type=str, default="outputs", help="MCP outputs directory")
    parser.add_argument("--suffix", "-s", type=str, default="_pruned", help="Suffix for pruned files")
    
    args = parser.parse_args()
    
    from agentskiller.config import init_settings
    init_settings()
    
    prune_task_directory(
        tasks_dir=Path(args.tasks_dir),
        outputs_dir=args.outputs_dir,
        output_suffix=args.suffix,
    )
