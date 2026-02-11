"""
Step 16: Task Filtering

Filter valid tasks by executing trajectories against MCP servers.

This step uses the Evaluator module with trajectory-wise parallelism:
1. Collect all unique trajectories across all servers
2. Execute trajectories in parallel using parallel_process
3. Cache generated code for reuse across similar trajectories
4. Save validated tasks for downstream processing

Input: combinations/{server}/*.json, mcp_servers/*.py
Output: validated_tasks/{server}/*.json, trajectory_code/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..core.parallel import parallel_process
from .base import step_handler, save_json, load_json, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrajectoryTask:
    """A trajectory with its associated combos to validate."""
    trajectory: List[str]
    trajectory_hash: str
    task_template: Dict[str, str]
    server_name: str
    combos: List[Dict[str, Any]]
    combo_file: Path


@dataclass
class TrajectoryResult:
    """Result of validating a trajectory."""
    trajectory_hash: str
    server_name: str
    combo_file: Path
    success: bool
    validated_combos: List[Dict[str, Any]]
    total_combos: int
    error_message: Optional[str] = None
    new_code_generated: bool = False


@dataclass
class FilteringStats:
    """Statistics for the filtering process."""
    total_files: int = 0
    total_combos: int = 0
    total_trajectories: int = 0
    validated_combos: int = 0
    failed_combos: int = 0
    unique_trajectories: int = 0
    trajectories_validated: int = 0
    trajectories_failed: int = 0
    new_codes_generated: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_files": self.total_files,
            "total_combos": self.total_combos,
            "total_trajectories": self.total_trajectories,
            "validated_combos": self.validated_combos,
            "failed_combos": self.failed_combos,
            "pass_rate": f"{self.validated_combos / max(self.total_combos, 1) * 100:.1f}%",
            "unique_trajectories": self.unique_trajectories,
            "trajectories_validated": self.trajectories_validated,
            "trajectories_failed": self.trajectories_failed,
            "new_codes_generated": self.new_codes_generated,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def check_result_for_error(result: Any) -> Tuple[bool, Optional[str]]:
    """Check if a tool result contains an error."""
    if isinstance(result, dict):
        if "error" in result:
            return True, result.get("error")
        if result.get("success") == False:
            return True, result.get("message", "Operation failed")
    return False, None


def collect_trajectory_tasks(combos_dir: Path) -> List[TrajectoryTask]:
    """
    Collect all trajectory tasks from combo files.
    
    Returns a list of TrajectoryTask objects, one per unique trajectory per file.
    """
    tasks = []
    
    for server_dir in combos_dir.iterdir():
        if not server_dir.is_dir():
            continue
        
        server_name = server_dir.name
        
        # Step 15 outputs files named by trajectory hash: {hash}.json
        # e.g., a1b2c3d4e5f6.json
        for combo_file in server_dir.glob("*.json"):
            # Skip metadata files
            if combo_file.name.startswith("_"):
                continue
            try:
                data = load_json(combo_file)
                
                # Handle format: {trajectory: [...], combos: [...]}
                if isinstance(data, dict) and "trajectory" in data:
                    trajectory = data["trajectory"]
                    combos = data.get("combos", [])
                    trajectory_hash = data["trajectory_hash"]
                    task_template = data["task_template"]
                    
                    if trajectory and combos:
                        tasks.append(TrajectoryTask(
                            trajectory=trajectory,
                            trajectory_hash=trajectory_hash,
                            task_template=task_template,
                            server_name=server_name,
                            combos=combos,
                            combo_file=combo_file,
                        ))
                        
            except Exception as e:
                logger.error(f"Failed to load {combo_file}: {e}")
    
    return tasks


def create_trajectory_validator(
    outputs_dir: Path,
    code_manager,
    max_combos: int = 5,
    max_retries: int = 3,
):
    """
    Create a trajectory validation function for use with parallel_process.
    
    Returns a closure that captures the shared resources (outputs_dir, code_manager).
    """
    from evaluator import TrajectoryExecutor, TaskDefinition
    
    def validate_trajectory(task: TrajectoryTask) -> TrajectoryResult:
        """
        Validate a single trajectory with its combos.
        
        This function is designed to be called in parallel for different trajectories.
        Thread-safe: each trajectory gets its own executor instance.
        """
        trajectory = task.trajectory
        trajectory_hash = task.trajectory_hash
        server_name = task.server_name
        task_template = task.task_template
        combos = task.combos[:max_combos]
        
        logger.debug(f"[{trajectory_hash[:8]}] Starting validation with {len(combos)} combos")
        
        validated_combos = []
        new_code_generated = False
        error_message = None
        
        # Check for pre-generated codes
        has_pregenerated = False
        step_param_codes = None
        
        if code_manager:
            entry = code_manager.get_entry_by_hash(trajectory_hash)
            if entry and entry.step_param_codes:
                step_param_codes = entry.step_param_codes
                has_pregenerated = True
                logger.debug(f"[{trajectory_hash[:8]}] Found pre-generated codes")
        
        # Fast path: if trajectory already validated, skip execution entirely
        if has_pregenerated:
            logger.info(f"[{trajectory_hash[:8]}] ⏭️ SKIP - already validated, {len(combos)} combos auto-passed")
            for combo in combos:
                validated_combo = {
                    **combo,
                    "validated": True,
                    "task_template": task_template,
                    "trajectory": trajectory,
                    "trajectory_hash": trajectory_hash,
                    "validation_timestamp": datetime.now().isoformat(),
                    "skipped_execution": True,
                }
                validated_combos.append(validated_combo)
            
            return TrajectoryResult(
                trajectory_hash=trajectory_hash,
                server_name=server_name,
                combo_file=task.combo_file,
                success=True,
                validated_combos=validated_combos,
                total_combos=len(combos),
                new_code_generated=False,
            )
        
        # Slow path: need to generate and validate codes
        # Create executor for this trajectory (thread-local)
        executor = TrajectoryExecutor(outputs_dir=str(outputs_dir))
        
        trajectory_validated = False
        
        for combo_idx, combo in enumerate(combos):
            # Build TaskDefinition
            task_def = TaskDefinition.from_combo(
                trajectory=trajectory,
                combo=combo,
                server_name=server_name,
            )
            
            try:
                # Generate codes with LLM (slow path)
                result = executor.execute(task=task_def, verbose=True)
                
                # Check if all steps succeeded
                all_success = result.success and all(s.success for s in result.steps)
                
                if all_success:
                    if not trajectory_validated and not has_pregenerated:
                        # First successful execution - save codes
                        trajectory_validated = True
                        
                        if code_manager and result.step_param_codes:
                            try:
                                entity_mappings = {
                                    k.split(".")[-1]: k 
                                    for k in task_def.entity_context.keys()
                                }
                                code_manager.save_code(
                                    trajectory=trajectory,
                                    domains=task_def.domains,
                                    is_cross_domain=task_def.is_cross_domain,
                                    generated_code=result.generated_code or "",
                                    entity_mappings=entity_mappings,
                                    step_param_codes=result.step_param_codes,
                                )
                                new_code_generated = True
                                has_pregenerated = True
                                logger.info(f"[{trajectory_hash[:8]}] ✓ Saved new codes")
                            except Exception as e:
                                logger.warning(f"[{trajectory_hash[:8]}] Failed to save codes: {e}")
                    
                    # Add validated combo
                    validated_combo = {
                        **combo,
                        "validated": True,
                        "task_template": task_template,
                        "trajectory": trajectory,
                        "trajectory_hash": trajectory_hash,
                        "validation_timestamp": datetime.now().isoformat(),
                    }
                    validated_combos.append(validated_combo)
                    
                elif not trajectory_validated and combo_idx < max_retries - 1:
                    # Trajectory not yet validated, try next combo
                    error_message = result.errors[0] if result.errors else "Unknown error"
                    logger.debug(f"[{trajectory_hash[:8]}] Combo {combo_idx + 1} failed, trying next...")
                    continue
                else:
                    # Failed but trajectory already validated, or max retries reached
                    if not error_message:
                        error_message = result.errors[0] if result.errors else "Unknown error"
                        
            except Exception as e:
                logger.error(f"[{trajectory_hash[:8]}] Exception: {e}")
                if not error_message:
                    error_message = str(e)
        
        success = len(validated_combos) > 0
        
        if success:
            logger.info(f"[{trajectory_hash[:8]}] ✓ Validated {len(validated_combos)}/{len(combos)} combos")
        else:
            logger.warning(f"[{trajectory_hash[:8]}] ✗ No combos validated: {error_message}")
        
        return TrajectoryResult(
            trajectory_hash=trajectory_hash,
            server_name=server_name,
            combo_file=task.combo_file,
            success=success,
            validated_combos=validated_combos,
            total_combos=len(combos),
            error_message=error_message if not success else None,
            new_code_generated=new_code_generated,
        )
    
    return validate_trajectory


def generate_filtering_report(
    stats: FilteringStats,
    results: List[TrajectoryResult],
    output_path: Path,
) -> None:
    """Generate a markdown report of the filtering process."""
    # Group results by server
    by_server: Dict[str, List[TrajectoryResult]] = {}
    for r in results:
        if r is None:
            continue
        if r.server_name not in by_server:
            by_server[r.server_name] = []
        by_server[r.server_name].append(r)
    
    lines = [
        "# Task Filtering Report",
        f"\nGenerated at: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Total Trajectories**: {stats.total_trajectories}",
        f"- **Trajectories Validated**: {stats.trajectories_validated}",
        f"- **Trajectories Failed**: {stats.trajectories_failed}",
        f"- **Total Combos**: {stats.total_combos}",
        f"- **Validated Combos**: {stats.validated_combos}",
        f"- **Pass Rate**: {stats.validated_combos / max(stats.total_combos, 1) * 100:.1f}%",
        f"- **New Codes Generated**: {stats.new_codes_generated}",
        "",
        "## Results by Server",
        "",
    ]
    
    for server_name, server_results in sorted(by_server.items()):
        validated = sum(1 for r in server_results if r.success)
        total_validated_combos = sum(len(r.validated_combos) for r in server_results)
        total_combos = sum(r.total_combos for r in server_results)
        
        lines.append(f"### {server_name}")
        lines.append(f"- Trajectories: {validated}/{len(server_results)} validated")
        lines.append(f"- Combos: {total_validated_combos}/{total_combos} validated")
        
        # Show failed trajectories
        failed = [r for r in server_results if not r.success]
        if failed:
            lines.append("\n**Failed Trajectories:**")
            for r in failed[:5]:
                lines.append(f"- `{r.trajectory_hash[:8]}`: {r.error_message or 'Unknown'}")
            if len(failed) > 5:
                lines.append(f"- ... and {len(failed) - 5} more")
        lines.append("")
    
    output_path.write_text("\n".join(lines))
    logger.info(f"Report saved to: {output_path}")


# =============================================================================
# Main Step Handler
# =============================================================================

@step_handler("s16_task_filtering", auto_retry=True)
def task_filtering_step(state: WorkflowState) -> WorkflowState:
    """
    Filter valid tasks by trajectory execution with trajectory-wise parallelism.
    
    Process:
    1. Collect all trajectory tasks from combinations/
    2. Execute trajectories in parallel using parallel_process
    3. Each trajectory validates its combos sequentially
    4. Cache successful codes for reuse
    5. Save validated combos to validated_tasks/
    
    Output:
    - validated_tasks/{server}/*.json
    - trajectory_code/ (cached execution codes)
    - logs/task_filtering/filtering_report_*.md
    """
    from evaluator import TrajectoryCodeManager
    
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # combinations/ is the output of Step 15
    combos_dir = outputs_dir / "combinations"
    validated_dir = ensure_dir(outputs_dir / "validated_tasks")
    
    # Configuration
    max_combos_per_trajectory = 5
    max_retries_per_trajectory = 3
    
    logger.info("=" * 60)
    logger.info("Starting Task Filtering (Step 16)")
    logger.info("=" * 60)
    logger.info(f"Combos dir: {combos_dir}")
    logger.info(f"Config: max_combos={max_combos_per_trajectory}")
    
    # Initialize code manager (thread-safe for reads, writes use locks internally)
    code_manager = TrajectoryCodeManager(str(outputs_dir))
    initial_code_count = len(code_manager._index)
    logger.info(f"Loaded {initial_code_count} pre-generated trajectory codes")
    
    # Collect all trajectory tasks
    logger.info("Collecting trajectory tasks...")
    tasks = collect_trajectory_tasks(combos_dir)
    logger.info(f"Found {len(tasks)} trajectory tasks to validate")
    
    if not tasks:
        logger.warning("No trajectory tasks found!")
        state.validated_tasks_dir = str(validated_dir)
        return state
    
    # Create the validation function with captured resources
    validate_func = create_trajectory_validator(
        outputs_dir=outputs_dir,
        code_manager=code_manager,
        max_combos=max_combos_per_trajectory,
        max_retries=max_retries_per_trajectory,
    )
    
    # Execute trajectories in parallel using parallel_process
    logger.info("Starting parallel trajectory validation...")
    
    results: List[Optional[TrajectoryResult]] = parallel_process(
        items=tasks,
        process_func=validate_func,
        description="Validating trajectories",
        show_progress=True,
    )
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    # Aggregate results by server and save
    logger.info("Saving validated combos...")
    
    by_server: Dict[str, List[Dict]] = {}
    for r in valid_results:
        if r.validated_combos:
            if r.server_name not in by_server:
                by_server[r.server_name] = []
            by_server[r.server_name].extend(r.validated_combos)
    
    for server_name, combos in by_server.items():
        server_dir = ensure_dir(validated_dir / server_name)
        output_file = server_dir / "validated_combos.json"
        save_json(combos, output_file)
        logger.info(f"  {server_name}: {len(combos)} validated combos")
    
    # Calculate stats
    stats = FilteringStats(
        total_files=len(set(t.combo_file for t in tasks)),
        total_combos=sum(r.total_combos for r in valid_results),
        total_trajectories=len(valid_results),
        validated_combos=sum(len(r.validated_combos) for r in valid_results),
        failed_combos=sum(r.total_combos - len(r.validated_combos) for r in valid_results),
        unique_trajectories=len(set(r.trajectory_hash for r in valid_results)),
        trajectories_validated=sum(1 for r in valid_results if r.success),
        trajectories_failed=sum(1 for r in valid_results if not r.success),
        new_codes_generated=sum(1 for r in valid_results if r.new_code_generated),
    )
    
    # Generate report
    report_dir = ensure_dir(outputs_dir / "logs" / "task_filtering")
    report_path = report_dir / f"filtering_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_filtering_report(stats, valid_results, report_path)
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("Task Filtering Complete")
    logger.info(f"  Trajectories: {stats.trajectories_validated}/{stats.total_trajectories} validated")
    logger.info(f"  Combos: {stats.validated_combos}/{stats.total_combos} validated")
    logger.info(f"  Pass rate: {stats.validated_combos / max(stats.total_combos, 1) * 100:.1f}%")
    logger.info(f"  New codes generated: {stats.new_codes_generated}")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 60)
    
    # Update state
    state.validated_tasks_dir = str(validated_dir)
    
    state.update_step_progress(
        "s16_task_filtering",
        total=stats.total_combos,
        completed=stats.validated_combos,
        failed=stats.failed_combos,
    )
    
    return state
