"""
Step 12: Database Fusion

Aggregate databases according to Domain Combinations.
This step does NOT involve any LLM requests.

Input: database/outputs/**, _combinations.json, blueprints.json
Output: database/outputs/entities/{fused_name}/*.json
        database/outputs/relationships/{fused_name}/*.json
"""
import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Set

from ..models.state import WorkflowState
from ..config.settings import get_settings
from .base import step_handler, load_json, ensure_dir

logger = logging.getLogger(__name__)


def get_entities_for_blueprint(blueprint: Dict[str, Any]) -> Set[str]:
    """
    Get all entities from a blueprint (core + peripheral).
    
    Args:
        blueprint: Blueprint dict containing core_entity and peripheral_entities
        
    Returns:
        Set of entity names
    """
    entities = set()
    
    core = blueprint.get("core_entity")
    if core:
        entities.add(core)
    
    peripheral = blueprint.get("peripheral_entities", [])
    entities.update(peripheral)
    
    return entities


def get_relationships_for_blueprint(blueprint: Dict[str, Any]) -> List[str]:
    """
    Get all relationship names from a blueprint.
    
    Args:
        blueprint: Blueprint dict containing relationships
        
    Returns:
        List of relationship names
    """
    relationships = blueprint.get("relationships", [])
    return [rel.get("name") for rel in relationships if rel.get("name")]


def copy_entity_database(
    entity_name: str,
    source_dir: Path,
    target_dir: Path,
) -> bool:
    """
    Copy an entity database file to the fused directory.
    
    Args:
        entity_name: Name of the entity
        source_dir: Source entities directory (database/outputs/entities)
        target_dir: Target fused directory (database/outputs/entities/{fused_name})
        
    Returns:
        True if copy successful, False otherwise
    """
    source_file = source_dir / f"{entity_name}.json"
    target_file = target_dir / f"{entity_name}.json"
    
    if not source_file.exists():
        logger.warning(f"Entity database not found: {source_file}")
        return False
    
    shutil.copy2(source_file, target_file)
    logger.debug(f"Copied entity database: {entity_name}")
    return True


def copy_relationship_database(
    relationship_name: str,
    server_name: str,
    source_dir: Path,
    target_dir: Path,
) -> bool:
    """
    Copy a relationship database file to the fused directory.
    
    Args:
        relationship_name: Name of the relationship
        server_name: Name of the MCP server
        source_dir: Source relationships directory (database/outputs/relationships)
        target_dir: Target fused directory (database/outputs/relationships/{fused_name})
        
    Returns:
        True if copy successful, False otherwise
    """
    source_file = source_dir / server_name / f"{relationship_name}.json"
    target_file = target_dir / f"{relationship_name}.json"
    
    if not source_file.exists():
        logger.warning(f"Relationship database not found: {source_file}")
        return False
    
    shutil.copy2(source_file, target_file)
    logger.debug(f"Copied relationship database: {server_name}/{relationship_name}")
    return True


@step_handler("s12_database_fusion", auto_retry=True)
def database_fusion_step(state: WorkflowState) -> WorkflowState:
    """
    Aggregate databases according to Domain Combinations.
    
    This step does NOT involve any LLM requests. It simply:
    1. Loads combinations and blueprints
    2. For each combination:
       a. Collects all entities involved (from all servers' blueprints)
       b. Copies entity databases to entities/{fused_name}/
       c. Copies relationship databases to relationships/{fused_name}/
    
    Output:
    - database/outputs/entities/{fused_name}/*.json
    - database/outputs/relationships/{fused_name}/*.json
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # Load combinations
    if state.cross_domain_combinations_path:
        combos_path = Path(state.cross_domain_combinations_path)
    else:
        combos_path = outputs_dir / "cross_domain_combinations" / "_combinations.json"
        if combos_path.exists():
            state.cross_domain_combinations_path = str(combos_path)
        else:
            raise FileNotFoundError(
                f"Combinations file not found. Expected at {combos_path}. "
                "Please run step s10_domain_combos_selection first."
            )
    
    combos_data = load_json(combos_path)
    combinations = combos_data.get("combinations", [])
    
    # Load blueprints
    blueprints_data = load_json(Path(state.blueprints_path))
    if isinstance(blueprints_data, list):
        blueprints = blueprints_data
    else:
        blueprints = blueprints_data.get("blueprints", [])
    
    # Build server -> blueprint mapping
    server_to_blueprint: Dict[str, Dict[str, Any]] = {}
    for bp in blueprints:
        server_name = bp.get("MCP_server_name")
        if server_name:
            server_to_blueprint[server_name] = bp
    
    # Get database directories
    if state.databases_dir:
        db_dir = Path(state.databases_dir)
    else:
        db_dir = outputs_dir / "database"
        if db_dir.exists():
            state.databases_dir = str(db_dir)
        else:
            raise FileNotFoundError(
                f"Database directory not found. Expected at {db_dir}. "
                "Please run step s06_database_generation first."
            )
    
    source_entities_dir = db_dir / "outputs" / "entities"
    source_relationships_dir = db_dir / "outputs" / "relationships"
    
    logger.info(f"Fusing databases for {len(combinations)} combinations")
    
    # Track progress
    total_combos = len(combinations)
    completed_combos = 0
    failed_combos = []
    
    for combo in combinations:
        fused_name = combo.get("fused_name", "unknown")
        servers = combo.get("servers", [])

        if os.path.exists(db_dir / "outputs" / "entities" / fused_name):
            logger.info(f"Fused database already exists for {fused_name}")
            completed_combos += 1
            continue
        
        try:
            # ================================================================
            # 1. Collect all entities for this combination
            # ================================================================
            all_entities: Set[str] = set()
            all_relationships: List[tuple] = []  # (server_name, relationship_name)
            
            for server_name in servers:
                bp = server_to_blueprint.get(server_name)
                if not bp:
                    logger.warning(f"Blueprint not found for server: {server_name}")
                    continue
                
                # Get entities from blueprint
                entities = get_entities_for_blueprint(bp)
                all_entities.update(entities)
                
                # Get relationships from blueprint
                relationships = get_relationships_for_blueprint(bp)
                for rel_name in relationships:
                    all_relationships.append((server_name, rel_name))
            
            # ================================================================
            # 2. Copy entity databases to entities/{fused_name}/
            # ================================================================
            target_entities_dir = ensure_dir(db_dir / "outputs" / "entities" / fused_name)
            
            entities_copied = 0
            for entity_name in all_entities:
                if copy_entity_database(entity_name, source_entities_dir, target_entities_dir):
                    entities_copied += 1
            
            # ================================================================
            # 3. Copy relationship databases to relationships/{fused_name}/
            # ================================================================
            target_relationships_dir = ensure_dir(db_dir / "outputs" / "relationships" / fused_name)
            
            relationships_copied = 0
            for server_name, rel_name in all_relationships:
                if copy_relationship_database(
                    rel_name, server_name, source_relationships_dir, target_relationships_dir
                ):
                    relationships_copied += 1
            
            logger.info(
                f"Fused database for {fused_name}: "
                f"{entities_copied} entities, {relationships_copied} relationships"
            )
            completed_combos += 1
            
        except Exception as e:
            logger.error(f"Failed to fuse database for {fused_name}: {e}")
            failed_combos.append(fused_name)
    
    # Update state
    state.fused_databases_dir = str(db_dir / "outputs")
    
    # Update progress
    state.update_step_progress(
        "s12_database_fusion",
        total=total_combos,
        completed=completed_combos,
        failed=len(failed_combos),
    )
    
    logger.info(
        f"Database fusion complete: {completed_combos}/{total_combos} combinations, "
        f"{len(failed_combos)} failed"
    )
    
    if failed_combos:
        logger.warning(f"Failed combinations: {failed_combos}")
    
    return state
