"""
Step 14: Task Template Generation

Generate task templates from tool graphs.
This step supports both single-domain and cross-domain scenarios:

- Single Domain: Read valid_trajectories from tool_graphs/*.json (pre-filtered by Step 08)
- Cross Domain: Read pre-generated trajectories from cross_domain_templates/*.json (from s11)

Supports incremental processing - skips already completed trajectories on retry.

Input: 
  - Single: tool_graphs/*.json, policies/*.md, database_summary/
  - Cross: cross_domain_templates/*.json, policies/{fused}.md, database_summary/
Output: task_templates/*.json
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from ..models.state import WorkflowState
from ..models.tasks import TaskTemplate, TaskTemplateWithTrajectory, TaskTemplateScores
from ..config.settings import get_settings
from ..prompts import TASK_TEMPLATE_PROMPT, TASK_TEMPLATE_JUDGE_PROMPT
from .base import step_handler, save_json, load_json, ensure_dir, get_client, parallel_process

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_database_summary(
    database_summary_dir: Path, 
    server_names: List[str],
    blueprints: Dict[str, Dict[str, Any]]
) -> str:
    """
    Load database summaries filtered by server(s).
    
    Args:
        database_summary_dir: Path to database_summary directory
        server_names: List of server names to load summaries for
        blueprints: Dict mapping server name to blueprint dict
        
    Returns:
        Concatenated summary text for relevant entities and relationships
    """
    summary_parts = []
    
    # Collect all entities from the specified servers' blueprints
    all_entities: Set[str] = set()
    for server_name in server_names:
        bp = blueprints.get(server_name, {})
        core = bp.get("core_entity")
        if core:
            all_entities.add(core)
        peripheral = bp.get("peripheral_entities", [])
        all_entities.update(peripheral)

    print(all_entities)
    
    # Load entity summaries for collected entities
    entities_dir = database_summary_dir / "entities"
    if entities_dir.exists():
        for entity in sorted(all_entities):
            entity_file = entities_dir / f"{entity}.md"
            if entity_file.exists():
                content = entity_file.read_text()
                summary_parts.append(f"## Entity: {entity}\n{content}")
    
    # Load relationship summaries from each server's directory
    relationships_dir = database_summary_dir / "relationships"
    if relationships_dir.exists():
        for server_name in server_names:
            server_rel_dir = relationships_dir / server_name
            if server_rel_dir.exists():
                for rel_file in sorted(server_rel_dir.glob("*.md")):
                    content = rel_file.read_text()
                    summary_parts.append(f"## Relationship ({server_name}): {rel_file.stem}\n{content}")
    
    if not summary_parts:
        return "No database summary available."
    
    return "\n\n".join(summary_parts)


def load_completed_trajectories(templates_dir: Path, is_cross_domain: bool = False) -> Dict[str, Set[tuple]]:
    """
    Load already completed trajectories from existing output files.
    
    Returns:
        Dict mapping server_name/fused_name to set of completed trajectory tuples
    """
    completed: Dict[str, Set[tuple]] = {}
    
    for output_file in templates_dir.glob("*.json"):
        name = output_file.stem
        try:
            existing_data = load_json(output_file)
            
            # Filter by domain type
            file_is_cross = existing_data.get("is_cross_domain", False)
            if file_is_cross != is_cross_domain:
                continue
            
            templates = existing_data.get("templates", [])
            traj_set = set()
            for tmpl in templates:
                traj = tmpl.get("trajectory", [])
                if traj:
                    traj_set.add(tuple(traj))
            
            if traj_set:
                completed[name] = traj_set
                logger.info(f"{'Cross-domain' if is_cross_domain else 'Single-domain'} {name}: Found {len(traj_set)} completed trajectories")
        except Exception as e:
            logger.warning(f"Failed to load existing templates for {name}: {e}")
    
    return completed


def load_existing_templates(templates_dir: Path, name: str) -> List[Dict[str, Any]]:
    """Load existing templates for a server/fused_name."""
    output_path = templates_dir / f"{name}.json"
    if output_path.exists():
        try:
            data = load_json(output_path)
            return data.get("templates", [])
        except Exception:
            pass
    return []


# =============================================================================
# LLM Judge
# =============================================================================

def judge_task_template(
    client,
    trajectory: List[str],
    template: Dict[str, Any]
) -> Optional[TaskTemplateScores]:
    """Use LLM to judge the quality of a task template."""
    trajectory_text = " -> ".join(trajectory)
    template_text = f"Instruction: {template.get('instruction', '')}\nReason: {template.get('reason_for_call', '')}"
    
    prompt = TASK_TEMPLATE_JUDGE_PROMPT.format(
        trajectory=trajectory_text,
        task_template=template_text
    )
    
    try:
        response = client.chat(query=prompt, model_type="textual")
        scores_data = response.parse_json()
        
        if isinstance(scores_data, dict):
            if "overall_score" not in scores_data:
                score_fields = ["motivation_clarity", "logical_coherence", "completeness", "naturalness", "specificity"]
                valid_scores = [scores_data.get(f, 3) for f in score_fields]
                scores_data["overall_score"] = sum(valid_scores) / len(valid_scores)
            
            return TaskTemplateScores(
                motivation_clarity=scores_data.get("motivation_clarity", 3),
                logical_coherence=scores_data.get("logical_coherence", 3),
                completeness=scores_data.get("completeness", 3),
                naturalness=scores_data.get("naturalness", 3),
                specificity=scores_data.get("specificity", 3),
                overall_score=scores_data.get("overall_score", 3),
                rejection_reason=scores_data.get("rejection_reason")
            )
    except Exception as e:
        logger.warning(f"Failed to judge template: {e}")
    
    return None


# =============================================================================
# Template Generation Core
# =============================================================================

def generate_template_for_trajectory(
    client,
    trajectory: List[str],
    policy: str,
    database_summary: str,
    enable_judge: bool,
    judge_threshold: float,
    is_cross_domain: bool = False,
    domains: List[str] = None
) -> Optional[Dict[str, Any]]:
    """Generate a task template for a single trajectory."""
    trajectory_text = " -> ".join(trajectory)
    
    prompt = TASK_TEMPLATE_PROMPT.format(
        trajectory=trajectory_text,
        policies=policy,
        database_summary=database_summary
    )
    
    try:
        response = client.chat(query=prompt, model_type="textual")
        batch_templates = response.parse_json()
        
        if isinstance(batch_templates, list) and batch_templates:
            tmpl = batch_templates[0]
            
            if not isinstance(tmpl, dict):
                return None
            
            task_template = TaskTemplate(
                instruction=tmpl.get("instruction", ""),
                reason_for_call=tmpl.get("reason_for_call", "")
            )
            
            template_with_trajectory = TaskTemplateWithTrajectory(
                trajectory=trajectory,
                task_template=task_template,
                is_cross_domain=is_cross_domain,
                domains=domains or []
            )
            
            if enable_judge:
                scores = judge_task_template(client, trajectory, tmpl)
                if scores:
                    template_with_trajectory.scores = scores
                    if scores.overall_score < judge_threshold:
                        logger.debug(f"Filtered template (score={scores.overall_score:.2f})")
                        return None
            
            return template_with_trajectory.model_dump()
            
    except Exception as e:
        logger.warning(f"Failed to generate template for trajectory {trajectory_text}: {e}")
    
    return None


# =============================================================================
# Single Domain Processing
# =============================================================================

@dataclass
class TrajectoryTask:
    """A single (server, trajectory) task for parallel processing."""
    server_name: str
    trajectory: List[str]
    policy: str
    database_summary: str


def process_single_domain_templates(
    tool_graphs_dir: Path,
    policies_dir: Path,
    database_summary_dir: Optional[Path],
    templates_dir: Path,
    client,
    step_config: Dict[str, Any],
    blueprints: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process single-domain templates with incremental support.
    Skips already completed trajectories on retry.
    """
    min_trajectory_length = step_config.get("min_trajectory_length", 2)
    max_trajectory_length = step_config.get("max_trajectory_length", 5)
    enable_judge = step_config.get("enable_judge", True)
    judge_threshold = step_config.get("judge_threshold", 3.0)
    
    graph_files = list(tool_graphs_dir.glob("*.json"))
    logger.info(f"Processing {len(graph_files)} single-domain tool graphs")
    
    # Step 0: Load completed trajectories for incremental processing
    completed_trajectories = load_completed_trajectories(templates_dir, is_cross_domain=False)
    
    # Step 1: Collect tasks, skipping completed ones
    all_tasks: List[TrajectoryTask] = []
    server_trajectory_counts: Dict[str, int] = {}
    skipped_counts: Dict[str, int] = {}
    
    for graph_path in graph_files:
        server_name = graph_path.stem
        graph_data = load_json(graph_path)
        
        policy_path = policies_dir / f"{server_name}.md"
        policy = policy_path.read_text() if policy_path.exists() else "No policy available."
        
        if database_summary_dir and database_summary_dir.exists():
            database_summary = load_database_summary(database_summary_dir, [server_name], blueprints)
        else:
            database_summary = "No database summary available."
        
        # Read valid_trajectories from graph data (pre-filtered by Step 08)
        trajectories = graph_data.get("valid_trajectories", [])
        
        if not trajectories:
            logger.warning(f"Server {server_name}: No valid_trajectories found in tool graph")
            continue
        
        server_trajectory_counts[server_name] = len(trajectories)
        server_completed = completed_trajectories.get(server_name, set())
        skipped = 0
        
        for trajectory in trajectories:
            traj_tuple = tuple(trajectory)
            if traj_tuple in server_completed:
                skipped += 1
                continue
            
            all_tasks.append(TrajectoryTask(
                server_name=server_name,
                trajectory=trajectory,
                policy=policy,
                database_summary=database_summary
            ))
        
        skipped_counts[server_name] = skipped
        if skipped > 0:
            logger.info(
                f"Server {server_name}: {skipped} already completed, "
                f"{len(trajectories) - skipped} remaining"
            )
    
    total_skipped = sum(skipped_counts.values())
    logger.info(f"Total tasks: {len(all_tasks)} to process, {total_skipped} skipped (already completed)")
    
    if not all_tasks:
        logger.info("All trajectories already completed!")
        return {"total_templates": 0, "total_filtered": 0, "servers_processed": len(graph_files), "skipped": total_skipped}
    
    # Step 2: Process tasks in parallel
    def process_trajectory_task(task: TrajectoryTask) -> Optional[Dict[str, Any]]:
        template = generate_template_for_trajectory(
            client=client,
            trajectory=task.trajectory,
            policy=task.policy,
            database_summary=task.database_summary,
            enable_judge=enable_judge,
            judge_threshold=judge_threshold,
            is_cross_domain=False
        )
        if template:
            template["_server_name"] = task.server_name
            return template
        return None
    
    results = parallel_process(
        items=all_tasks,
        process_func=process_trajectory_task,
        description="Generating single-domain templates",
    )
    
    # Step 3: Group results and merge with existing
    new_templates: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        if result:
            server_name = result.pop("_server_name")
            if server_name not in new_templates:
                new_templates[server_name] = []
            new_templates[server_name].append(result)
    
    total_templates = 0
    total_filtered = 0
    
    # Process all servers (including those with only existing templates)
    all_servers = set(server_trajectory_counts.keys())
    
    for server_name in all_servers:
        existing = load_existing_templates(templates_dir, server_name)
        new = new_templates.get(server_name, [])
        
        # Merge: existing + new
        all_templates = existing + new
        
        total_trajectories = server_trajectory_counts.get(server_name, 0)
        
        stats = {
            "total_trajectories": total_trajectories,
            "generated_templates": len(all_templates),
            "new_in_this_run": len(new),
            "existing_before_run": len(existing)
        }
        
        output_path = templates_dir / f"{server_name}.json"
        save_json({
            "templates": all_templates,
            "stats": stats,
            "is_cross_domain": False
        }, output_path)
        
        total_templates += len(new)
        
        if new:
            logger.info(f"Server {server_name}: Added {len(new)} new templates (total: {len(all_templates)})")
    
    return {
        "total_templates": total_templates,
        "total_filtered": total_filtered,
        "servers_processed": len(graph_files),
        "skipped": total_skipped
    }


# =============================================================================
# Cross Domain Processing
# =============================================================================

@dataclass
class CrossDomainTrajectoryTask:
    """A single (fused_name, trajectory) task for cross-domain parallel processing."""
    fused_name: str
    trajectory: List[str]
    policy: str
    database_summary: str
    servers: List[str]
    combination: Dict[str, Any]


def process_cross_domain_templates(
    cross_domain_templates_dir: Path,
    policies_dir: Path,
    database_summary_dir: Optional[Path],
    templates_dir: Path,
    client,
    step_config: Dict[str, Any],
    blueprints: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process cross-domain templates with incremental support.
    Skips already completed trajectories on retry.
    """
    enable_judge = step_config.get("enable_judge", True)
    judge_threshold = step_config.get("judge_threshold", 3.0)
    
    trajectory_files = list(cross_domain_templates_dir.glob("*.json"))
    trajectory_files.remove(cross_domain_templates_dir / "_combinations.json")
    logger.info(f"Processing {len(trajectory_files)} cross-domain trajectory files")
    
    # Step 0: Load completed trajectories for incremental processing
    completed_trajectories = load_completed_trajectories(templates_dir, is_cross_domain=True)
    
    # Step 1: Collect tasks, skipping completed ones
    all_tasks: List[CrossDomainTrajectoryTask] = []
    fused_trajectory_counts: Dict[str, int] = {}
    fused_combinations: Dict[str, Dict[str, Any]] = {}
    skipped_counts: Dict[str, int] = {}
    
    for trajectory_file in trajectory_files:
        fused_name = trajectory_file.stem
        data = load_json(trajectory_file)
        
        trajectories = data.get("trajectories", [])
        combination = data.get("combination", {})
        servers = combination.get("servers", [])
        
        if not trajectories:
            logger.warning(f"No trajectories found in {fused_name}")
            continue
        
        fused_combinations[fused_name] = combination
        
        policy_path = policies_dir / f"{fused_name}.md"
        if policy_path.exists():
            logger.info(f"Policy found for {fused_name}")
            policy = policy_path.read_text()
        else:
            logger.info(f"No policy found for {fused_name}, using server policies")
            policy_parts = []
            for server in servers:
                server_policy_path = policies_dir / f"{server}.md"
                if server_policy_path.exists():
                    policy_parts.append(f"## {server}\n{server_policy_path.read_text()}")
            policy = "\n\n".join(policy_parts) if policy_parts else "No policy available."
        
        if database_summary_dir and database_summary_dir.exists():
            database_summary = load_database_summary(database_summary_dir, servers, blueprints)
        else:
            database_summary = "No database summary available."
        
        fused_completed = completed_trajectories.get(fused_name, set())
        skipped = 0
        valid_count = 0
        
        for traj_data in trajectories:
            trajectory = []
            
            if isinstance(traj_data, list):
                trajectory = traj_data
            elif isinstance(traj_data, dict):
                # Handle nested cross-domain format: {"trajectories": [{"ServerA": [...]}, ...], "motivation": "..."}
                nested_trajectories = traj_data.get("trajectories", [])
                if nested_trajectories and isinstance(nested_trajectories, list):
                    for server_traj in nested_trajectories:
                        if isinstance(server_traj, dict):
                            for server_name, tools in server_traj.items():
                                if isinstance(tools, list):
                                    # Add server prefix to distinguish tools from different servers
                                    trajectory.extend([f"{server_name}.{tool}" for tool in tools])
                else:
                    # Fallback to original format
                    trajectory = traj_data.get("path", traj_data.get("trajectory", []))
            
            if not trajectory:
                continue
            
            valid_count += 1
            traj_tuple = tuple(trajectory)
            
            if traj_tuple in fused_completed:
                skipped += 1
                continue
            
            all_tasks.append(CrossDomainTrajectoryTask(
                fused_name=fused_name,
                trajectory=trajectory,
                policy=policy,
                database_summary=database_summary,
                servers=servers,
                combination=combination
            ))
        
        fused_trajectory_counts[fused_name] = valid_count
        skipped_counts[fused_name] = skipped
        
        if skipped > 0:
            logger.info(f"Cross-domain {fused_name}: Skipping {skipped} completed, {valid_count - skipped} remaining")
    
    total_skipped = sum(skipped_counts.values())
    logger.info(f"Total cross-domain tasks: {len(all_tasks)} to process, {total_skipped} skipped (already completed)")
    
    if not all_tasks:
        logger.info("All cross-domain trajectories already completed!")
        return {"total_templates": 0, "total_filtered": 0, "combinations_processed": len(trajectory_files), "skipped": total_skipped}
    
    # Step 2: Process tasks in parallel
    def process_cross_domain_task(task: CrossDomainTrajectoryTask) -> Optional[Dict[str, Any]]:
        template = generate_template_for_trajectory(
            client=client,
            trajectory=task.trajectory,
            policy=task.policy,
            database_summary=task.database_summary,
            enable_judge=enable_judge,
            judge_threshold=judge_threshold,
            is_cross_domain=True,
            domains=task.servers
        )
        if template:
            template["_fused_name"] = task.fused_name
            return template
        return None
    
    results = parallel_process(
        items=all_tasks,
        process_func=process_cross_domain_task,
        description="Generating cross-domain templates"
    )
    
    # Step 3: Group results and merge with existing
    new_templates: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        if result:
            fused_name = result.pop("_fused_name")
            if fused_name not in new_templates:
                new_templates[fused_name] = []
            new_templates[fused_name].append(result)
    
    total_templates = 0
    total_filtered = 0
    
    all_fused = set(fused_trajectory_counts.keys())
    
    for fused_name in all_fused:
        existing = load_existing_templates(templates_dir, fused_name)
        new = new_templates.get(fused_name, [])
        
        all_templates = existing + new
        
        combination = fused_combinations.get(fused_name, {})
        total_trajectories = fused_trajectory_counts.get(fused_name, 0)
        
        stats = {
            "total_trajectories": total_trajectories,
            "generated_templates": len(all_templates),
            "new_in_this_run": len(new),
            "existing_before_run": len(existing),
            "combination": combination
        }
        
        output_path = templates_dir / f"{fused_name}.json"
        save_json({
            "templates": all_templates,
            "stats": stats,
            "is_cross_domain": True,
            "combination": combination
        }, output_path)
        
        total_templates += len(new)
        
        if new:
            logger.info(f"Cross-domain {fused_name}: Added {len(new)} new templates (total: {len(all_templates)})")
    
    return {
        "total_templates": total_templates,
        "total_filtered": total_filtered,
        "combinations_processed": len(trajectory_files),
        "skipped": total_skipped
    }


# =============================================================================
# Step Handler
# =============================================================================

@step_handler("s14_task_template_generation", auto_retry=True)
def task_template_generation_step(state: WorkflowState) -> WorkflowState:
    """
    Generate task templates from tool graphs.
    
    Supports incremental processing - skips already completed trajectories on retry.
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    task_mode = settings.workflow.task_mode
    step_config = settings.get_step_config("s14_task_template_generation")
    
    tool_graphs_dir = Path(state.tool_graphs_dir)
    policies_dir = Path(state.policies_dir)
    database_summary_dir = Path(state.database_summary_dir) if state.database_summary_dir else None
    templates_dir = ensure_dir(outputs_dir / "task_templates")
    
    cross_domain_templates_dir = Path(state.cross_domain_templates_dir) if hasattr(state, 'cross_domain_templates_dir') and state.cross_domain_templates_dir else None
    cross_domain_templates_dir = Path("outputs/cross_domain_combinations") if cross_domain_templates_dir is None else cross_domain_templates_dir
    
    # Load blueprints for database summary filtering
    blueprints_path = outputs_dir / "blueprints.json"
    blueprints: Dict[str, Dict[str, Any]] = {}
    if blueprints_path.exists():
        blueprints_list = load_json(blueprints_path)
        for bp in blueprints_list:
            server_name = bp.get("MCP_server_name", "")
            if server_name:
                blueprints[server_name] = bp
        logger.info(f"Loaded {len(blueprints)} blueprints for database summary filtering")
    else:
        logger.warning(f"Blueprints file not found at {blueprints_path}, database summaries will be empty")
    
    logger.info(f"Task template generation - Mode: {task_mode}")
    
    client = get_client()
    
    total_stats = {
        "single_domain": {"templates": 0, "filtered": 0, "skipped": 0},
        "cross_domain": {"templates": 0, "filtered": 0, "skipped": 0}
    }
    
    if task_mode in ["single", "both"]:
        logger.info("Processing single-domain templates...")
        single_stats = process_single_domain_templates(
            tool_graphs_dir=tool_graphs_dir,
            policies_dir=policies_dir,
            database_summary_dir=database_summary_dir,
            templates_dir=templates_dir,
            client=client,
            step_config=step_config,
            blueprints=blueprints
        )
        total_stats["single_domain"]["templates"] = single_stats.get("total_templates", 0)
        total_stats["single_domain"]["filtered"] = single_stats.get("total_filtered", 0)
        total_stats["single_domain"]["skipped"] = single_stats.get("skipped", 0)
    
    if task_mode in ["cross_domain", "both"]:
        if cross_domain_templates_dir and cross_domain_templates_dir.exists():
            logger.info("Processing cross-domain templates...")
            cross_stats = process_cross_domain_templates(
                cross_domain_templates_dir=cross_domain_templates_dir,
                policies_dir=policies_dir,
                database_summary_dir=database_summary_dir,
                templates_dir=templates_dir,
                client=client,
                step_config=step_config,
                blueprints=blueprints
            )
            total_stats["cross_domain"]["templates"] = cross_stats.get("total_templates", 0)
            total_stats["cross_domain"]["filtered"] = cross_stats.get("total_filtered", 0)
            total_stats["cross_domain"]["skipped"] = cross_stats.get("skipped", 0)
        else:
            logger.warning("Cross-domain mode enabled but no cross_domain_templates_dir found.")
    
    state.task_templates_dir = str(templates_dir)
    
    total_new = total_stats["single_domain"]["templates"] + total_stats["cross_domain"]["templates"]
    total_skipped = total_stats["single_domain"]["skipped"] + total_stats["cross_domain"]["skipped"]
    
    logger.info(f"Task template generation complete:")
    logger.info(f"  - Single domain: {total_stats['single_domain']['templates']} new, {total_stats['single_domain']['skipped']} skipped")
    logger.info(f"  - Cross domain: {total_stats['cross_domain']['templates']} new, {total_stats['cross_domain']['skipped']} skipped")
    logger.info(f"  - Total: {total_new} new templates, {total_skipped} skipped (already completed)")
    
    return state
