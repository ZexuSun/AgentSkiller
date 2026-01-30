"""
Step 6: Database Generation

Generate mock databases for entities and relationships through a 6-stage pipeline:
1. Load Data: Load entities.json and blueprints.json
2. Generate Entity Database: Create Python scripts and run them to generate entity data
3. Summary Entity Database: Analyze scripts and summarize value ranges
4. Identify Relationship Constraints: Detect potential attribute conflicts
5. Generate Relationship Database: Create relationship data with constraints
6. Summary Relationship Database: Analyze and summarize relationship data

Input: entities.json, blueprints.json
Output: database/outputs/**, database_summary/**, database/constraints/**
"""

import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import (
    ENTITY_DATABASE_PROMPT,
    DATABASE_SUMMARY_PROMPT,
    CONSTRAINT_IDENTIFICATION_PROMPT,
    RELATIONSHIP_DATABASE_PROMPT,
)
from .base import (
    step_handler,
    save_json,
    load_json,
    get_client,
    parallel_process,
    ensure_dir,
    WorkflowBlockEditor,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Progress Management
# =============================================================================

def load_progress(progress_path: Path) -> Dict[str, Any]:
    """Load existing progress or return empty structure."""
    if progress_path.exists():
        try:
            return load_json(progress_path)
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}, starting fresh")
    return {}


def save_progress(progress_path: Path, progress: Dict[str, Any]) -> None:
    """Save progress to intermediate file."""
    save_json(progress, progress_path)


def cleanup_progress_files(outputs_dir: Path) -> None:
    """Remove all progress files after successful completion."""
    progress_files = [
        "entity_db_progress.json",
        "entity_summary_progress.json",
        "constraint_progress.json",
        "relationship_db_progress.json",
        "relationship_summary_progress.json",
    ]
    for filename in progress_files:
        path = outputs_dir / filename
        if path.exists():
            path.unlink()
            logger.debug(f"Cleaned up progress file: {filename}")


# =============================================================================
# Code Execution Utilities
# =============================================================================

def extract_python_code(content: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in content:
        code = content.split("```python")[1].split("```")[0]
    elif "```" in content:
        code = content.split("```")[1].split("```")[0]
    else:
        code = content
    return code.strip()


def run_script_with_retry(
    script_path: Path,
    output_path: Path,
    client,
    editor: WorkflowBlockEditor,
    max_code_fix_retries: int,
    script_timeout: int,
) -> Dict[str, Any]:
    """
    Run a Python script with retry and Block Editor fix on failure.
    
    Args:
        script_path: Path to the Python script
        output_path: Expected output JSON file path
        client: LLM client for code fixing
        editor: WorkflowBlockEditor instance
        max_code_fix_retries: Maximum number of fix attempts
        script_timeout: Script execution timeout in seconds
    
    Returns:
        Dict with 'success', 'data' or 'error' keys
    """
    last_error = None
    
    for attempt in range(max_code_fix_retries + 1):  # +1 for initial run
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=script_timeout,
            )
            
            if result.returncode == 0:
                # Check if output file was created
                if output_path.exists():
                    data = load_json(output_path)
                    return {"success": True, "data": data}
                else:
                    last_error = f"Script completed but output file not found: {output_path}"
            else:
                last_error = result.stderr or result.stdout or "Unknown error"
                
        except subprocess.TimeoutExpired:
            last_error = f"Script execution timed out after {script_timeout} seconds"
        except Exception as e:
            last_error = str(e)
        
        # If this was the last attempt, don't try to fix
        if attempt >= max_code_fix_retries:
            break
        
        # Try to fix the code using Block Editor
        logger.info(f"Attempt {attempt + 1}/{max_code_fix_retries}: Fixing script {script_path.name}")
        fixed = editor.fix_code(
            file_path=script_path,
            error=last_error,
            llm_client=client,
            language="python",
            max_retries=1,  # Single fix attempt per iteration
        )
        
        if not fixed:
            logger.warning(f"Block Editor could not fix {script_path.name}")
            # Continue to next attempt anyway, maybe LLM can fix differently
    
    return {"success": False, "error": last_error}


# =============================================================================
# Sub-step 2: Generate Entity Database
# =============================================================================

def generate_entity_script(
    entity_name: str,
    entity_info: Dict[str, Any],
    output_path: Path,
    script_path: Path,
    client,
    simulation_time: str,
    records_count: int,
) -> None:
    """Generate Python script for entity database creation."""
    prompt = ENTITY_DATABASE_PROMPT.format(
        entity=json.dumps({entity_name: entity_info}, indent=2, ensure_ascii=False),
        entity_name=entity_name,
        entity_name_lower=entity_name.lower(),
        simulation_time=simulation_time,
        output_path=str(output_path),
        records_count=records_count,
    )
    
    response = client.chat(query=prompt, model_type="coding")
    code = extract_python_code(response.content)
    
    # Ensure parent directory exists
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(code)


def run_substep2_entity_database(
    entities: Dict[str, Any],
    outputs_dir: Path,
    simulation_time: str,
    records_count: int,
    max_code_fix_retries: int,
    script_timeout: int,
) -> None:
    """
    Sub-step 2: Generate entity databases.
    
    Parallelism: Entity-wise
    Retry granularity: Entity-wise
    """
    progress_path = outputs_dir / "entity_db_progress.json"
    progress = load_progress(progress_path)
    
    if "completed" not in progress:
        progress["completed"] = []
    if "failed" not in progress:
        progress["failed"] = []
    
    # Setup directories
    db_dir = ensure_dir(outputs_dir / "database")
    scripts_dir = ensure_dir(db_dir / "scripts" / "entities")
    outputs_entities = ensure_dir(db_dir / "outputs" / "entities")
    
    # Filter pending entities
    all_entity_names = list(entities.keys())
    pending = [name for name in all_entity_names if name not in progress["completed"]]
    
    if not pending:
        logger.info("Sub-step 2: All entity databases already generated")
        return
    
    logger.info(f"Sub-step 2: Generating databases for {len(pending)} entities")
    
    client = get_client()
    editor = WorkflowBlockEditor()
    progress_lock = threading.Lock()
    
    def process_entity(entity_name: str) -> Optional[Dict[str, Any]]:
        """Process a single entity to generate its database."""
        entity_info = entities[entity_name]
        script_path = scripts_dir / f"{entity_name}.py"
        output_path = outputs_entities / f"{entity_name}.json"
        
        try:
            # Generate script
            generate_entity_script(
                entity_name=entity_name,
                entity_info=entity_info,
                output_path=output_path,
                script_path=script_path,
                client=client,
                simulation_time=simulation_time,
                records_count=records_count,
            )
            
            # Run with retry
            result = run_script_with_retry(
                script_path=script_path,
                output_path=output_path,
                client=client,
                editor=editor,
                max_code_fix_retries=max_code_fix_retries,
                script_timeout=script_timeout,
            )
            
            if result["success"]:
                with progress_lock:
                    progress["completed"].append(entity_name)
                    if entity_name in progress["failed"]:
                        progress["failed"].remove(entity_name)
                    save_progress(progress_path, progress)
                
                logger.info(f"Sub-step 2: Generated database for {entity_name} ({len(result['data'])} records)")
                return result
            else:
                with progress_lock:
                    if entity_name not in progress["failed"]:
                        progress["failed"].append(entity_name)
                    save_progress(progress_path, progress)
                
                logger.warning(f"Sub-step 2: Failed to generate database for {entity_name}: {result['error'][:200]}")
                raise RuntimeError(f"Failed to generate database for {entity_name}")
                
        except Exception as e:
            with progress_lock:
                if entity_name not in progress["failed"]:
                    progress["failed"].append(entity_name)
                save_progress(progress_path, progress)
            raise
    
    # Process all pending entities in parallel
    results = parallel_process(
        items=pending,
        process_func=process_entity,
        description="Sub-step 2: Entity databases",
    )
    
    # Check completion
    successful = sum(1 for r in results if r is not None)
    failed = len(pending) - successful
    
    if failed > 0:
        raise RuntimeError(
            f"Sub-step 2 incomplete: {failed}/{len(pending)} entities failed. "
            f"Re-run to retry failed items."
        )
    
    logger.info(f"Sub-step 2 complete: {len(progress['completed'])}/{len(all_entity_names)} entities")


# =============================================================================
# Sub-step 3: Summary Entity Database
# =============================================================================

def generate_entity_summary(
    entity_name: str,
    script_path: Path,
    client,
) -> str:
    """Generate summary for an entity database based on its generation script."""
    code = script_path.read_text()
    
    prompt = DATABASE_SUMMARY_PROMPT.format(
        code=code,
        name=entity_name,
        type="Entity",
    )
    
    response = client.chat(query=prompt, model_type="textual")
    return response.content


def run_substep3_entity_summary(
    entities: Dict[str, Any],
    outputs_dir: Path,
) -> Dict[str, str]:
    """
    Sub-step 3: Generate summaries for all entity databases.
    
    Parallelism: Entity-wise
    Retry granularity: Entity-wise
    
    Returns:
        Dict mapping entity name to summary content
    """
    progress_path = outputs_dir / "entity_summary_progress.json"
    progress = load_progress(progress_path)
    
    if "completed" not in progress:
        progress["completed"] = []
    
    # Setup directories
    scripts_dir = outputs_dir / "database" / "scripts" / "entities"
    summary_dir = ensure_dir(outputs_dir / "database_summary" / "entities")
    
    # Filter pending entities (only those with generated scripts)
    all_entity_names = list(entities.keys())
    pending = [
        name for name in all_entity_names
        if name not in progress["completed"] and (scripts_dir / f"{name}.py").exists()
    ]
    
    if not pending:
        logger.info("Sub-step 3: All entity summaries already generated")
        # Load existing summaries
        summaries = {}
        for name in all_entity_names:
            summary_path = summary_dir / f"{name}.md"
            if summary_path.exists():
                summaries[name] = summary_path.read_text()
        return summaries
    
    logger.info(f"Sub-step 3: Generating summaries for {len(pending)} entities")
    
    client = get_client()
    progress_lock = threading.Lock()
    summaries = {}
    summaries_lock = threading.Lock()
    
    def process_entity(entity_name: str) -> Optional[str]:
        """Generate summary for a single entity."""
        script_path = scripts_dir / f"{entity_name}.py"
        summary_path = summary_dir / f"{entity_name}.md"
        
        try:
            summary = generate_entity_summary(
                entity_name=entity_name,
                script_path=script_path,
                client=client,
            )
            
            # Save summary
            summary_path.write_text(summary)
            
            with summaries_lock:
                summaries[entity_name] = summary
            
            with progress_lock:
                progress["completed"].append(entity_name)
                save_progress(progress_path, progress)
            
            logger.debug(f"Sub-step 3: Generated summary for {entity_name}")
            return summary
            
        except Exception as e:
            logger.warning(f"Sub-step 3: Failed to generate summary for {entity_name}: {e}")
            raise
    
    # Process all pending entities in parallel
    results = parallel_process(
        items=pending,
        process_func=process_entity,
        description="Sub-step 3: Entity summaries",
    )
    
    # Check completion
    successful = sum(1 for r in results if r is not None)
    failed = len(pending) - successful
    
    if failed > 0:
        raise RuntimeError(
            f"Sub-step 3 incomplete: {failed}/{len(pending)} summaries failed. "
            f"Re-run to retry failed items."
        )
    
    # Load all summaries (including previously completed)
    for name in all_entity_names:
        if name not in summaries:
            summary_path = summary_dir / f"{name}.md"
            if summary_path.exists():
                summaries[name] = summary_path.read_text()
    
    logger.info(f"Sub-step 3 complete: {len(progress['completed'])}/{len(all_entity_names)} summaries")
    return summaries


# =============================================================================
# Sub-step 4: Identify Relationship Constraints
# =============================================================================

def get_relationship_tuples(blueprints: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Extract (server_name, relationship_name, relationship_info) tuples from blueprints.
    """
    tuples = []
    for bp in blueprints:
        server_name = bp.get("MCP_server_name", "Unknown")
        for rel in bp.get("relationships", []):
            rel_name = rel.get("name", "Unknown")
            tuples.append((server_name, rel_name, rel))
    return tuples


def identify_constraints(
    server_name: str,
    relationship: Dict[str, Any],
    entity_summaries: Dict[str, str],
    client,
) -> Dict[str, Any]:
    """Identify constraints for a single relationship."""
    rel_name = relationship.get("name", "Unknown")
    
    # Collect relevant entity summaries based on relationship attributes
    relevant_summaries = {}
    for attr_name, attr_info in relationship.get("attributes", {}).items():
        if isinstance(attr_info, dict):
            value_from = attr_info.get("value_from_entity", "")
            if value_from and "." in value_from:
                entity_name = value_from.split(".")[0]
                if entity_name in entity_summaries:
                    relevant_summaries[entity_name] = entity_summaries[entity_name]
    
    if not relevant_summaries:
        # No entity references, return empty constraints
        return {
            "server_name": server_name,
            "relationship_name": rel_name,
            "constraints": [],
            "notes": "No entity references found in relationship attributes",
        }
    
    prompt = CONSTRAINT_IDENTIFICATION_PROMPT.format(
        relationship=json.dumps(relationship, indent=2, ensure_ascii=False),
        entity_database_summary="\n\n---\n\n".join(
            f"## {name}\n{summary}" for name, summary in relevant_summaries.items()
        ),
        simulation_time="2024-01-01T00:00:00Z",
    )
    
    response = client.chat(query=prompt, model_type="textual")
    
    # Parse response as JSON or extract structured info
    try:
        constraints = response.parse_json()
    except Exception:
        # If not valid JSON, wrap the response
        constraints = {
            "raw_analysis": response.content,
            "constraints": [],
        }
    
    return {
        "server_name": server_name,
        "relationship_name": rel_name,
        "constraints": constraints,
    }


def run_substep4_constraint_identification(
    blueprints: List[Dict[str, Any]],
    entity_summaries: Dict[str, str],
    outputs_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Sub-step 4: Identify constraints for all relationships.
    
    Parallelism: (MCP Server, Relationship) tuple
    Retry granularity: (MCP Server, Relationship) tuple
    
    Returns:
        Dict mapping "server_name/relationship_name" to constraint info
    """
    progress_path = outputs_dir / "constraint_progress.json"
    progress = load_progress(progress_path)
    
    if "completed" not in progress:
        progress["completed"] = []
    
    # Setup directories
    constraints_dir = ensure_dir(outputs_dir / "database" / "constraints")
    
    # Get all relationship tuples
    all_tuples = get_relationship_tuples(blueprints)
    
    # Filter pending tuples
    pending = [
        t for t in all_tuples
        if [t[0], t[1]] not in progress["completed"]
    ]
    
    if not pending:
        logger.info("Sub-step 4: All constraints already identified")
        # Load existing constraints
        constraints = {}
        for server_name, rel_name, _ in all_tuples:
            key = f"{server_name}/{rel_name}"
            constraint_path = constraints_dir / server_name / f"{rel_name}.json"
            if constraint_path.exists():
                constraints[key] = load_json(constraint_path)
        return constraints
    
    logger.info(f"Sub-step 4: Identifying constraints for {len(pending)} relationships")
    
    client = get_client()
    progress_lock = threading.Lock()
    constraints = {}
    constraints_lock = threading.Lock()
    
    def process_relationship(tuple_item: Tuple[str, str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Identify constraints for a single relationship."""
        server_name, rel_name, relationship = tuple_item
        key = f"{server_name}/{rel_name}"
        
        try:
            constraint_info = identify_constraints(
                server_name=server_name,
                relationship=relationship,
                entity_summaries=entity_summaries,
                client=client,
            )
            
            # Save constraint
            server_dir = ensure_dir(constraints_dir / server_name)
            constraint_path = server_dir / f"{rel_name}.json"
            save_json(constraint_info, constraint_path)
            
            with constraints_lock:
                constraints[key] = constraint_info
            
            with progress_lock:
                progress["completed"].append([server_name, rel_name])
                save_progress(progress_path, progress)
            
            logger.debug(f"Sub-step 4: Identified constraints for {key}")
            return constraint_info
            
        except Exception as e:
            logger.warning(f"Sub-step 4: Failed to identify constraints for {key}: {e}")
            raise
    
    # Process all pending tuples in parallel
    results = parallel_process(
        items=pending,
        process_func=process_relationship,
        description="Sub-step 4: Constraint identification",
    )
    
    # Check completion
    successful = sum(1 for r in results if r is not None)
    failed = len(pending) - successful
    
    if failed > 0:
        raise RuntimeError(
            f"Sub-step 4 incomplete: {failed}/{len(pending)} constraints failed. "
            f"Re-run to retry failed items."
        )
    
    # Load all constraints (including previously completed)
    for server_name, rel_name, _ in all_tuples:
        key = f"{server_name}/{rel_name}"
        if key not in constraints:
            constraint_path = constraints_dir / server_name / f"{rel_name}.json"
            if constraint_path.exists():
                constraints[key] = load_json(constraint_path)
    
    logger.info(f"Sub-step 4 complete: {len(progress['completed'])}/{len(all_tuples)} constraints")
    return constraints


# =============================================================================
# Sub-step 5: Generate Relationship Database
# =============================================================================

def generate_relationship_script(
    server_name: str,
    relationship: Dict[str, Any],
    blueprint: Dict[str, Any],
    entity_summaries: Dict[str, str],
    constraints: Dict[str, Any],
    entity_db_path: Path,
    output_path: Path,
    script_path: Path,
    client,
    simulation_time: str,
    records_count: int,
) -> None:
    """Generate Python script for relationship database creation."""
    rel_name = relationship.get("name", "Unknown")
    
    # Collect relevant entity summaries
    relevant_summaries = {}
    for attr_name, attr_info in relationship.get("attributes", {}).items():
        if isinstance(attr_info, dict):
            value_from = attr_info.get("value_from_entity", "")
            if value_from and "." in value_from:
                entity_name = value_from.split(".")[0]
                if entity_name in entity_summaries:
                    relevant_summaries[entity_name] = entity_summaries[entity_name]
    
    prompt = RELATIONSHIP_DATABASE_PROMPT.format(
        blueprint=json.dumps(blueprint, indent=2, ensure_ascii=False),
        relationship=json.dumps(relationship, indent=2, ensure_ascii=False),
        entity_database_summary="\n\n---\n\n".join(
            f"## {name}\n{summary}" for name, summary in relevant_summaries.items()
        ),
        constraints=json.dumps(constraints, indent=2, ensure_ascii=False),
        entity_db_path=str(entity_db_path),
        relationship_name=rel_name,
        relationship_name_lower=rel_name.lower(),
        output_path=str(output_path),
        simulation_time=simulation_time,
        records_count=records_count,
    )
    
    response = client.chat(query=prompt, model_type="coding")
    code = extract_python_code(response.content)
    
    # Ensure parent directory exists
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(code)


def run_substep5_relationship_database(
    blueprints: List[Dict[str, Any]],
    entity_summaries: Dict[str, str],
    constraints: Dict[str, Dict[str, Any]],
    outputs_dir: Path,
    simulation_time: str,
    records_count: int,
    max_code_fix_retries: int,
    script_timeout: int,
) -> None:
    """
    Sub-step 5: Generate relationship databases.
    
    Parallelism: (MCP Server, Relationship) tuple
    Retry granularity: (MCP Server, Relationship) tuple
    """
    progress_path = outputs_dir / "relationship_db_progress.json"
    progress = load_progress(progress_path)
    
    if "completed" not in progress:
        progress["completed"] = []
    if "failed" not in progress:
        progress["failed"] = []
    
    # Setup directories
    db_dir = outputs_dir / "database"
    entity_db_path = db_dir / "outputs" / "entities"
    scripts_dir = ensure_dir(db_dir / "scripts" / "relationships")
    outputs_rel = ensure_dir(db_dir / "outputs" / "relationships")
    
    # Build blueprint lookup
    blueprint_lookup = {bp.get("MCP_server_name", ""): bp for bp in blueprints}
    
    # Get all relationship tuples
    all_tuples = get_relationship_tuples(blueprints)
    
    # Filter pending tuples
    pending = [
        t for t in all_tuples
        if [t[0], t[1]] not in progress["completed"]
    ]
    
    if not pending:
        logger.info("Sub-step 5: All relationship databases already generated")
        return
    
    logger.info(f"Sub-step 5: Generating databases for {len(pending)} relationships")
    
    client = get_client()
    editor = WorkflowBlockEditor()
    progress_lock = threading.Lock()
    
    def process_relationship(tuple_item: Tuple[str, str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Process a single relationship to generate its database."""
        server_name, rel_name, relationship = tuple_item
        key = f"{server_name}/{rel_name}"
        
        # Get blueprint and constraint
        blueprint = blueprint_lookup.get(server_name, {})
        constraint_info = constraints.get(key, {})
        
        # Setup paths
        server_script_dir = ensure_dir(scripts_dir / server_name)
        server_output_dir = ensure_dir(outputs_rel / server_name)
        script_path = server_script_dir / f"{rel_name}.py"
        output_path = server_output_dir / f"{rel_name}.json"
        
        try:
            # Generate script
            generate_relationship_script(
                server_name=server_name,
                relationship=relationship,
                blueprint=blueprint,
                entity_summaries=entity_summaries,
                constraints=constraint_info.get("constraints", {}),
                entity_db_path=entity_db_path,
                output_path=output_path,
                script_path=script_path,
                client=client,
                simulation_time=simulation_time,
                records_count=records_count,
            )
            
            # Run with retry
            result = run_script_with_retry(
                script_path=script_path,
                output_path=output_path,
                client=client,
                editor=editor,
                max_code_fix_retries=max_code_fix_retries,
                script_timeout=script_timeout,
            )
            
            if result["success"]:
                with progress_lock:
                    progress["completed"].append([server_name, rel_name])
                    # Remove from failed if present
                    progress["failed"] = [
                        f for f in progress["failed"]
                        if f != [server_name, rel_name]
                    ]
                    save_progress(progress_path, progress)
                
                logger.info(f"Sub-step 5: Generated database for {key} ({len(result['data'])} records)")
                return result
            else:
                with progress_lock:
                    if [server_name, rel_name] not in progress["failed"]:
                        progress["failed"].append([server_name, rel_name])
                    save_progress(progress_path, progress)
                
                logger.warning(f"Sub-step 5: Failed to generate database for {key}: {result['error'][:200]}")
                raise RuntimeError(f"Failed to generate database for {key}")
                
        except Exception as e:
            with progress_lock:
                if [server_name, rel_name] not in progress["failed"]:
                    progress["failed"].append([server_name, rel_name])
                save_progress(progress_path, progress)
            raise
    
    # Process all pending relationships in parallel
    results = parallel_process(
        items=pending,
        process_func=process_relationship,
        description="Sub-step 5: Relationship databases",
    )
    
    # Check completion
    successful = sum(1 for r in results if r is not None)
    failed = len(pending) - successful
    
    if failed > 0:
        raise RuntimeError(
            f"Sub-step 5 incomplete: {failed}/{len(pending)} relationships failed. "
            f"Re-run to retry failed items."
        )
    
    logger.info(f"Sub-step 5 complete: {len(progress['completed'])}/{len(all_tuples)} relationships")


# =============================================================================
# Sub-step 6: Summary Relationship Database
# =============================================================================

def generate_relationship_summary(
    server_name: str,
    rel_name: str,
    script_path: Path,
    client,
) -> str:
    """Generate summary for a relationship database based on its generation script."""
    code = script_path.read_text()
    
    prompt = DATABASE_SUMMARY_PROMPT.format(
        code=code,
        name=f"{server_name}/{rel_name}",
        type="Relationship",
    )
    
    response = client.chat(query=prompt, model_type="textual")
    return response.content


def run_substep6_relationship_summary(
    blueprints: List[Dict[str, Any]],
    outputs_dir: Path,
) -> Dict[str, str]:
    """
    Sub-step 6: Generate summaries for all relationship databases.
    
    Parallelism: (MCP Server, Relationship) tuple
    Retry granularity: (MCP Server, Relationship) tuple
    
    Returns:
        Dict mapping "server_name/relationship_name" to summary content
    """
    progress_path = outputs_dir / "relationship_summary_progress.json"
    progress = load_progress(progress_path)
    
    if "completed" not in progress:
        progress["completed"] = []
    
    # Setup directories
    scripts_dir = outputs_dir / "database" / "scripts" / "relationships"
    summary_dir = ensure_dir(outputs_dir / "database_summary" / "relationships")
    
    # Get all relationship tuples
    all_tuples = get_relationship_tuples(blueprints)
    
    # Filter pending tuples (only those with generated scripts)
    pending = [
        t for t in all_tuples
        if [t[0], t[1]] not in progress["completed"]
        and (scripts_dir / t[0] / f"{t[1]}.py").exists()
    ]
    
    if not pending:
        logger.info("Sub-step 6: All relationship summaries already generated")
        # Load existing summaries
        summaries = {}
        for server_name, rel_name, _ in all_tuples:
            key = f"{server_name}/{rel_name}"
            summary_path = summary_dir / server_name / f"{rel_name}.md"
            if summary_path.exists():
                summaries[key] = summary_path.read_text()
        return summaries
    
    logger.info(f"Sub-step 6: Generating summaries for {len(pending)} relationships")
    
    client = get_client()
    progress_lock = threading.Lock()
    summaries = {}
    summaries_lock = threading.Lock()
    
    def process_relationship(tuple_item: Tuple[str, str, Dict[str, Any]]) -> Optional[str]:
        """Generate summary for a single relationship."""
        server_name, rel_name, _ = tuple_item
        key = f"{server_name}/{rel_name}"
        
        script_path = scripts_dir / server_name / f"{rel_name}.py"
        server_summary_dir = ensure_dir(summary_dir / server_name)
        summary_path = server_summary_dir / f"{rel_name}.md"
        
        try:
            summary = generate_relationship_summary(
                server_name=server_name,
                rel_name=rel_name,
                script_path=script_path,
                client=client,
            )
            
            # Save summary
            summary_path.write_text(summary)
            
            with summaries_lock:
                summaries[key] = summary
            
            with progress_lock:
                progress["completed"].append([server_name, rel_name])
                save_progress(progress_path, progress)
            
            logger.debug(f"Sub-step 6: Generated summary for {key}")
            return summary
            
        except Exception as e:
            logger.warning(f"Sub-step 6: Failed to generate summary for {key}: {e}")
            raise
    
    # Process all pending tuples in parallel
    results = parallel_process(
        items=pending,
        process_func=process_relationship,
        description="Sub-step 6: Relationship summaries",
    )
    
    # Check completion
    successful = sum(1 for r in results if r is not None)
    failed = len(pending) - successful
    
    if failed > 0:
        raise RuntimeError(
            f"Sub-step 6 incomplete: {failed}/{len(pending)} summaries failed. "
            f"Re-run to retry failed items."
        )
    
    # Load all summaries (including previously completed)
    for server_name, rel_name, _ in all_tuples:
        key = f"{server_name}/{rel_name}"
        if key not in summaries:
            summary_path = summary_dir / server_name / f"{rel_name}.md"
            if summary_path.exists():
                summaries[key] = summary_path.read_text()
    
    logger.info(f"Sub-step 6 complete: {len(progress['completed'])}/{len(all_tuples)} summaries")
    return summaries


# =============================================================================
# Main Step Handler
# =============================================================================

@step_handler("s06_database_generation", auto_retry=True)
def database_generation_step(state: WorkflowState) -> WorkflowState:
    """
    Generate mock databases through a 6-stage pipeline.
    
    Process:
    1. Load Data: Load entities.json and blueprints.json
    2. Generate Entity Database: Create entity data (Entity-wise parallelism)
    3. Summary Entity Database: Summarize entity data (Entity-wise parallelism)
    4. Identify Relationship Constraints: Detect conflicts ((Server, Rel) parallelism)
    5. Generate Relationship Database: Create relationship data ((Server, Rel) parallelism)
    6. Summary Relationship Database: Summarize relationship data ((Server, Rel) parallelism)
    
    Each sub-step must fully complete before proceeding to the next.
    
    Output:
    - database/outputs/entities/*.json
    - database/outputs/relationships/{server}/*.json
    - database/constraints/{server}/*.json
    - database_summary/entities/*.md
    - database_summary/relationships/{server}/*.md
    """
    settings = get_settings()
    outputs_dir = Path(settings.paths.outputs_dir).resolve()  # Use absolute path
    step_config = settings.get_step_config("s06_database_generation")
    
    # Configuration
    records_count = step_config.get("entities_per_table", 100)
    max_code_fix_retries = step_config.get("max_code_fix_retries", 3)
    script_timeout = step_config.get("script_timeout", 120)
    simulation_time = step_config.get("simulation_time", "2024-01-01T00:00:00Z")
    
    # ===================
    # Sub-step 1: Load Data
    # ===================
    logger.info("Sub-step 1: Loading data")
    
    entities_data = load_json(Path(state.entities_path))
    # Handle both formats: {"entities": {...}} and direct dict
    if "entities" in entities_data:
        entities = entities_data["entities"]
    else:
        entities = entities_data
    
    blueprints = load_json(Path(state.blueprints_path))
    
    logger.info(f"Sub-step 1: Loaded {len(entities)} entities, {len(blueprints)} blueprints")
    
    # ===================
    # Sub-step 2: Generate Entity Database
    # ===================
    run_substep2_entity_database(
        entities=entities,
        outputs_dir=outputs_dir,
        simulation_time=simulation_time,
        records_count=records_count,
        max_code_fix_retries=max_code_fix_retries,
        script_timeout=script_timeout,
    )
    
    # ===================
    # Sub-step 3: Summary Entity Database
    # ===================
    entity_summaries = run_substep3_entity_summary(
        entities=entities,
        outputs_dir=outputs_dir,
    )
    
    # ===================
    # Sub-step 4: Identify Relationship Constraints
    # ===================
    constraints = run_substep4_constraint_identification(
        blueprints=blueprints,
        entity_summaries=entity_summaries,
        outputs_dir=outputs_dir,
    )
    
    # ===================
    # Sub-step 5: Generate Relationship Database
    # ===================
    run_substep5_relationship_database(
        blueprints=blueprints,
        entity_summaries=entity_summaries,
        constraints=constraints,
        outputs_dir=outputs_dir,
        simulation_time=simulation_time,
        records_count=records_count,
        max_code_fix_retries=max_code_fix_retries,
        script_timeout=script_timeout,
    )
    
    # ===================
    # Sub-step 6: Summary Relationship Database
    # ===================
    run_substep6_relationship_summary(
        blueprints=blueprints,
        outputs_dir=outputs_dir,
    )
    
    # ===================
    # Cleanup and Update State
    # ===================
    cleanup_progress_files(outputs_dir)
    
    db_dir = outputs_dir / "database"
    state.databases_dir = str(db_dir)
    state.database_summary_dir = str(outputs_dir / "database_summary")
    
    # Count total items for progress
    total_entities = len(entities)
    total_relationships = sum(len(bp.get("relationships", [])) for bp in blueprints)
    
    state.update_step_progress(
        "s06_database_generation",
        total=total_entities + total_relationships,
        completed=total_entities + total_relationships,
    )
    
    logger.info(
        f"Database generation complete: "
        f"{total_entities} entities, {total_relationships} relationships"
    )
    
    return state
