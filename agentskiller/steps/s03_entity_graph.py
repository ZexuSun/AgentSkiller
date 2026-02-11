"""
Step 3: Entity Graph Generation

Build entity relationship graph from extracted entities.
For each Person entity, identify relationships with other entities using batched LLM calls.

Features:
- Dynamic batching: excludes only the current Person being processed
- Parallel processing: all (Person, batch) tasks run concurrently
- Fine-grained retry at Person-batch level for resilience

Input: entities.json
Output: entity_graph.json
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

from ..models.state import WorkflowState
from ..models.entities import EntityGraph, EntityGraphNode, EntityGraphEdge
from ..config.settings import get_settings
from ..prompts import ENTITY_RELATION_PROMPT
from .base import step_handler, save_json, load_json, get_client, parallel_process

logger = logging.getLogger(__name__)


def batch_entities(entities: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    """Split entities dict into batches of specified size."""
    items = list(entities.items())
    return [dict(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]


def get_batches_for_person(
    person_name: str, 
    all_entities: Dict[str, Any], 
    batch_size: int
) -> List[Dict[str, Any]]:
    """
    Get batches of entities to check for a specific Person.
    
    Excludes only the current Person entity, includes all other entities
    (both Person and non-Person).
    """
    entities_to_check = {k: v for k, v in all_entities.items() if k != person_name}
    return batch_entities(entities_to_check, batch_size)


def process_single_batch(
    client,
    person_name: str,
    person_info: Dict[str, Any],
    batch: Dict[str, Any],
    batch_idx: int,
    valid_entity_names: Set[str],
) -> List[str]:
    """
    Process a single batch of entities for relationship detection.
    
    Args:
        client: LLM client
        person_name: Name of the Person entity
        person_info: Info dict for the Person entity
        batch: Dict of entities in this batch
        batch_idx: Index of this batch
        valid_entity_names: Set of all valid entity names for validation
    
    Returns:
        List of related entity names found in this batch
    
    Raises:
        Exception: If LLM call fails (for retry handling)
    """
    # Format batch entities for prompt
    entities_desc = "\n".join(
        f'- {name}: {info.get("description", "")}' 
        for name, info in batch.items()
    )
    
    # Format person entity for prompt
    person_desc = f'{person_name}: {person_info.get("description", "")}'
    
    prompt = ENTITY_RELATION_PROMPT.format(
        entities=entities_desc,
        entity=person_desc
    )
    
    response = client.chat(query=prompt, model_type="textual")
    batch_related = response.parse_json()
    
    # Validate and collect related entities
    related = []
    if isinstance(batch_related, list):
        for entity_name in batch_related:
            if isinstance(entity_name, str) and entity_name in valid_entity_names:
                related.append(entity_name)
            else:
                logger.debug(f"Skipping invalid entity name: {entity_name}")
    else:
        logger.warning(f"Unexpected response format for batch {batch_idx}: {type(batch_related)}")
    
    return related


def load_progress(progress_path: Path) -> Dict[str, Any]:
    """Load existing progress or return empty structure."""
    if progress_path.exists():
        try:
            return load_json(progress_path)
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}, starting fresh")
    
    return {
        "completed_tasks": {},  # {person_name: [batch_idx, ...]}
        "results": {},  # {person_name: [related_entity, ...]}
        "total_tasks": 0,
    }


def save_progress(progress_path: Path, progress: Dict[str, Any]) -> None:
    """Save progress to intermediate file."""
    save_json(progress, progress_path)


@step_handler("s03_entity_graph", auto_retry=True)
def entity_graph_step(state: WorkflowState) -> WorkflowState:
    """
    Build entity relationship graph with parallel processing and fine-grained retry.
    
    Process:
    1. Load entities and existing progress (if any)
    2. Generate all (Person, batch_idx) tasks
    3. Filter out completed tasks
    4. Process pending tasks in parallel using ThreadPoolExecutor
    5. Build edges from Person entities to their related entities
    6. Cleanup progress file after successful completion
    
    Output:
    - entity_graph.json: NetworkX JSON format graph
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    step_config = settings.get_step_config("s03_entity_graph")
    batch_size = step_config.get("relation_batch_size", 10)
    
    # Paths
    progress_path = outputs_dir / "entity_graph_progress.json"
    output_path = outputs_dir / "entity_graph.json"
    
    # Load entities
    entities_path = Path(state.entities_path)
    entities = load_json(entities_path)
    
    logger.info(f"Building graph for {len(entities)} entities with batch_size={batch_size}")
    
    # Build nodes for all entities
    nodes = []
    for name, info in entities.items():
        nodes.append(EntityGraphNode(
            id=name,
            is_person=info.get("is_person", False),
            attributes=list(info.get("attributes", {}).keys()),
            description=info.get("description", ""),
        ))
    
    # Get Person entities
    person_entities = {
        name: info for name, info in entities.items() 
        if info.get("is_person", False)
    }
    
    logger.info(f"Found {len(person_entities)} Person entities")
    
    # Handle edge case: no Person entities
    if not person_entities:
        logger.warning("No Person entities found, graph will have no edges")
        graph = EntityGraph(nodes=nodes, links=[])
        save_json(graph.to_networkx_dict(), output_path)
        state.entity_graph_path = str(output_path)
        return state
    
    # Calculate total tasks: each Person has (num_entities - 1) / batch_size batches
    # Note: batch count varies per Person since we exclude different entities
    all_tasks: List[Tuple[str, int]] = []
    for person_name in person_entities:
        batches = get_batches_for_person(person_name, entities, batch_size)
        for batch_idx in range(len(batches)):
            all_tasks.append((person_name, batch_idx))
    
    total_tasks = len(all_tasks)
    logger.info(f"Total tasks (Person * batches): {total_tasks}")
    
    # Load existing progress for incremental retry
    progress = load_progress(progress_path)
    progress["total_tasks"] = total_tasks
    
    # Filter out completed tasks
    pending_tasks = [
        (person_name, batch_idx) 
        for person_name, batch_idx in all_tasks
        if batch_idx not in progress["completed_tasks"].get(person_name, [])
    ]
    
    completed_count = total_tasks - len(pending_tasks)
    logger.info(f"Pending tasks: {len(pending_tasks)}, Already completed: {completed_count}")
    
    if pending_tasks:
        # Get LLM client
        client = get_client()
        valid_entity_names = set(entities.keys())
        
        # Thread-safe progress update
        progress_lock = threading.Lock()
        
        def process_task(task: Tuple[str, int]) -> Optional[Tuple[str, int, List[str]]]:
            """Process a single (Person, batch_idx) task."""
            person_name, batch_idx = task
            person_info = person_entities[person_name]
            
            # Get batches for this specific Person (excludes only this Person)
            batches = get_batches_for_person(person_name, entities, batch_size)
            
            if batch_idx >= len(batches):
                logger.warning(f"Invalid batch_idx {batch_idx} for Person '{person_name}'")
                return None
            
            batch = batches[batch_idx]
            
            try:
                related = process_single_batch(
                    client=client,
                    person_name=person_name,
                    person_info=person_info,
                    batch=batch,
                    batch_idx=batch_idx,
                    valid_entity_names=valid_entity_names,
                )
                
                # Thread-safe progress update
                with progress_lock:
                    progress["completed_tasks"].setdefault(person_name, []).append(batch_idx)
                    progress["results"].setdefault(person_name, []).extend(related)
                    save_progress(progress_path, progress)
                
                logger.debug(
                    f"Task ({person_name}, batch {batch_idx}): "
                    f"found {len(related)} related entities"
                )
                
                return (person_name, batch_idx, related)
                
            except Exception as e:
                logger.warning(f"Failed task ({person_name}, batch {batch_idx}): {e}")
                raise  # Let parallel_process handle the error
        
        # Process all pending tasks in parallel
        results = parallel_process(
            items=pending_tasks,
            process_func=process_task,
            description="Processing entity relations",
        )
        
        # Count successful results
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Parallel processing complete: {successful}/{len(pending_tasks)} tasks succeeded")
    
    # Update step progress
    completed_tasks_count = sum(
        len(batches) for batches in progress["completed_tasks"].values()
    )
    state.update_step_progress(
        "s03_entity_graph",
        total=total_tasks,
        completed=completed_tasks_count,
    )
    
    # Build edges from accumulated progress
    edges = []
    for person_name, related_list in progress["results"].items():
        # Deduplicate related entities while preserving order
        seen = set()
        unique_related = []
        for name in related_list:
            if name not in seen:
                seen.add(name)
                unique_related.append(name)
        
        logger.info(f"Person '{person_name}' is related to {len(unique_related)} entities")
        
        # Create edges from Person to each related entity
        for related_name in unique_related:
            edges.append(EntityGraphEdge(
                source=person_name,
                target=related_name,
                relationship="related_to",
                description="",
            ))
    
    # Create graph
    graph = EntityGraph(nodes=nodes, links=edges)
    
    # Save final output
    save_json(graph.to_networkx_dict(), output_path)
    state.entity_graph_path = str(output_path)
    
    # Cleanup progress file after successful completion
    if progress_path.exists():
        progress_path.unlink()
        logger.info("Cleaned up intermediate progress file")
    
    logger.info(f"Entity graph complete: {len(nodes)} nodes, {len(edges)} edges")
    return state
