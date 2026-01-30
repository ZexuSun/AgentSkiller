"""
Environment Evaluator - Verifies database state after agent actions.

This evaluator builds two "parallel worlds":
1. Gold Environment: Apply golden trajectory to get expected final state
2. Predicted Environment: Apply agent's trajectory to get actual final state

Then compares them using:
- Database hash comparison (cryptographic verification)
- Custom environment assertions (business logic validation)
"""

import json
import logging
import hashlib
import re
from typing import Dict, List, Any, Optional, Callable, Set
from copy import deepcopy
from pathlib import Path

from .base import (
    BaseEvaluator,
    EvaluatorType,
    EvaluationResult,
    TaskDefinition,
    TrajectoryExecution,
    DatabaseState,
    ToolCall,
)

logger = logging.getLogger(__name__)


class EnvironmentEvaluator(BaseEvaluator):
    """
    Evaluates whether agent's actions produced the correct environment state.
    
    The evaluator:
    1. Rebuilds the "predicted environment" from agent's tool calls
    2. Compares with the "gold environment" from golden trajectory
    3. Runs custom assertions for business logic validation
    4. Returns 1.0 only if ALL checks pass
    
    This is crucial for verifying:
    - Database modifications are correct
    - No unintended side effects
    - Business logic constraints are satisfied
    """
    
    evaluator_type = EvaluatorType.ENVIRONMENT
    
    # UUID pattern for validation
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', 
        re.IGNORECASE
    )
    
    def __init__(
        self,
        outputs_dir: str = "./outputs_cursor",
        blueprints_path: str = "outputs/blueprints.json",
        check_all_tables: bool = True,
        tables_to_check: Optional[List[str]] = None,
    ):
        """
        Initialize the environment evaluator.
        
        Args:
            outputs_dir: Directory containing MCP servers
            blueprints_path: Path to blueprints.json for field info
            check_all_tables: Whether to check all database tables
            tables_to_check: Specific tables to check (if not all)
        """
        self.outputs_dir = Path(outputs_dir)
        self.check_all_tables = check_all_tables
        self.tables_to_check = tables_to_check or []
        self.blueprints = self._load_blueprints(blueprints_path)
    
    def evaluate(
        self,
        task: TaskDefinition,
        gold_execution: TrajectoryExecution,
        agent_trajectory: List[Dict[str, Any]],
        agent_execution: Optional[TrajectoryExecution] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate agent's environment state against expected state.
        
        Uses diff-based comparison:
        - Computes Gold Diff = (Gold Final) - (Initial State)
        - Computes Agent Diff = (Agent Final) - (Initial State)
        - PASS if Gold Diff ⊆ Agent Diff (Agent can have extra changes)
        
        Args:
            task: Task definition with environment assertions
            gold_execution: Golden trajectory execution result
            agent_trajectory: Agent's conversation messages
            agent_execution: Agent's execution result (if available)
            
        Returns:
            EvaluationResult with pass/fail and details
        """
        # Check if we have the necessary data
        if gold_execution.final_state is None:
            return self._create_skip_result("No gold environment state available")
        
        # Check for initial state (required for diff comparison)
        if gold_execution.initial_state is None:
            return self._create_skip_result("No initial state available for diff comparison")
        
        # Get or build agent's final state
        if agent_execution and agent_execution.final_state:
            agent_state = agent_execution.final_state
        else:
            # Try to rebuild from trajectory
            agent_state = self._rebuild_state_from_trajectory(
                task=task,
                messages=agent_trajectory,
                initial_state=gold_execution.initial_state,
            )
        
        if agent_state is None:
            return self._create_fail_result(
                "Could not determine agent's environment state",
                {"reason": "Unable to rebuild state from trajectory"}
            )
        
        # Get server name from task for blueprint lookup
        # Use fused name (e.g., "ServerA_ServerB") so blueprint lookup can search all component servers
        server_name = "_".join(task.domains) if task.domains else ""
        
        # Phase 1: Diff-based comparison (Gold Diff ⊆ Agent Diff)
        diff_result = self._compare_diffs(
            initial_state=gold_execution.initial_state,
            gold_state=gold_execution.final_state,
            agent_state=agent_state,
            server_name=server_name,
        )
        
        # If Gold Diff is empty, always PASS
        if diff_result["stats"]["gold_diff_total"] == 0:
            return self._create_pass_result({
                "diff_match": True,
                "gold_diff_count": 0,
                "agent_diff_count": diff_result["stats"]["agent_diff_total"],
                "note": "Gold trajectory made no changes, auto-pass",
            })
        
        if not diff_result["passed"]:
            return self._create_fail_result(
                f"Database diff mismatch: {diff_result['reason']}",
                {
                    "diff_comparison": diff_result,
                    "gold_diff_count": diff_result["stats"]["gold_diff_total"],
                    "agent_diff_count": diff_result["stats"]["agent_diff_total"],
                    "missing_changes": diff_result["missing_changes"],
                }
            )
        
        # Phase 2: Environment assertions (if any)
        assertions_passed = 0
        if task.environment_assertions:
            assertion_results = self._run_assertions(
                assertions=task.environment_assertions,
                gold_state=gold_execution.final_state,
                agent_state=agent_state,
            )
            
            if not assertion_results["all_passed"]:
                return self._create_fail_result(
                    f"Environment assertion failed: {assertion_results['failure_reason']}",
                    {"assertion_results": assertion_results["details"]}
                )
            assertions_passed = len(task.environment_assertions)
        
        return self._create_pass_result({
            "diff_match": True,
            "gold_diff_count": diff_result["stats"]["gold_diff_total"],
            "agent_diff_count": diff_result["stats"]["agent_diff_total"],
            "tables_checked": diff_result["stats"]["tables_checked"],
            "assertions_passed": assertions_passed,
        })
    
    def _compare_database_hashes(
        self,
        gold_state: DatabaseState,
        agent_state: DatabaseState,
        server_name: str = "",
    ) -> Dict[str, Any]:
        """
        Compare database states using cryptographic hashes.
        
        Returns:
            Dictionary with comparison result
        """
        result = {
            "passed": True,
            "reason": "",
            "details": {},
        }
        
        # Get tables to compare
        if self.check_all_tables:
            tables = set(gold_state.entities.keys()) | set(gold_state.relationships.keys())
            tables |= set(agent_state.entities.keys()) | set(agent_state.relationships.keys())
        else:
            tables = set(self.tables_to_check)
        
        # Compare each table
        mismatches = []
        
        for table in tables:
            # Get data from both states
            gold_entity = gold_state.entities.get(table, [])
            gold_rel = gold_state.relationships.get(table, [])
            agent_entity = agent_state.entities.get(table, [])
            agent_rel = agent_state.relationships.get(table, [])
            
            gold_data = gold_entity if gold_entity else gold_rel
            agent_data = agent_entity if agent_entity else agent_rel
            
            # Skip if both are empty
            if not gold_data and not agent_data:
                continue
            
            # Compare counts
            if len(gold_data) != len(agent_data):
                mismatches.append({
                    "table": table,
                    "gold_count": len(gold_data),
                    "agent_count": len(agent_data),
                    "type": "count_mismatch",
                    "gold_sample": str(gold_data[0]) if gold_data else None,
                    "agent_sample": str(agent_data[0]) if agent_data else None,
                })
                continue
            
            # Collect placeholder fields from both sides (symmetric handling)
            placeholder_fields = self._collect_placeholder_fields(gold_data, agent_data)
            
            # Compare content (sorted for determinism)
            gold_hash = self._hash_table_data(gold_data, server_name, placeholder_fields)
            agent_hash = self._hash_table_data(agent_data, server_name, placeholder_fields)
            
            # Debug: log data lengths
            logger.info(f"Table {table}: gold={len(gold_data)} records, agent={len(agent_data)} records")
            
            if gold_hash != agent_hash:
                # Find specific differences for debugging
                gold_normalized = sorted(
                    [self._normalize_record(r, server_name, placeholder_fields) for r in gold_data],
                    key=lambda x: json.dumps(x, sort_keys=True, default=str)
                )
                agent_normalized = sorted(
                    [self._normalize_record(r, server_name, placeholder_fields) for r in agent_data],
                    key=lambda x: json.dumps(x, sort_keys=True, default=str)
                )
                
                diff_fields = []
                first_diff_index = None  # Track the first record with a difference
                
                # Log the actual data for debugging
                logger.debug(f"Gold data for {table}: {gold_normalized}")
                logger.debug(f"Agent data for {table}: {agent_normalized}")
                
                # Compare records pairwise
                for i, (gr, ar) in enumerate(zip(gold_normalized, agent_normalized)):
                    for key in set(gr.keys()) | set(ar.keys()):
                        gv = gr.get(key)
                        av = ar.get(key)
                        if gv != av:
                            if first_diff_index is None:
                                first_diff_index = i
                            diff_fields.append({
                                "record_index": i,
                                "field": key,
                                "gold": str(gv),
                                "agent": str(av),
                                "gold_type": type(gv).__name__,
                                "agent_type": type(av).__name__,
                            })
                
                # Show the actual differing record, not just [0]
                diff_idx = first_diff_index if first_diff_index is not None else 0
                gold_sample = str(gold_normalized[diff_idx]) if gold_normalized else None
                agent_sample = str(agent_normalized[diff_idx]) if agent_normalized else None
                
                mismatches.append({
                    "table": table,
                    "type": "content_mismatch",
                    "diff_fields": diff_fields[:5],
                    "gold_records": len(gold_data),
                    "agent_records": len(agent_data),
                    "first_diff_index": first_diff_index,
                    "gold_sample": gold_sample,
                    "agent_sample": agent_sample,
                    "gold_hash": gold_hash[:16],
                    "agent_hash": agent_hash[:16],
                })
        
        if mismatches:
            result["passed"] = False
            result["reason"] = f"{len(mismatches)} table(s) have mismatches"
            result["details"]["mismatches"] = mismatches
        
        return result
    
    # Timestamp fields that are always dynamic
    TIMESTAMP_FIELDS = {
        'created_at', 'updated_at', 'timestamp',
    }
    
    def _load_blueprints(self, path: str) -> Dict[str, Any]:
        """Load blueprints and index by server name."""
        blueprints_path = Path(path)
        if not blueprints_path.exists():
            logger.warning(f"Blueprints file not found: {path}")
            return {}
        
        try:
            with open(blueprints_path) as f:
                blueprints_list = json.load(f)
            
            # Index by MCP_server_name
            indexed = {}
            for bp in blueprints_list:
                server_name = bp.get("MCP_server_name")
                if server_name:
                    indexed[server_name] = bp
            
            logger.info(f"Loaded {len(indexed)} blueprints for EnvironmentEvaluator")
            return indexed
        except Exception as e:
            logger.warning(f"Failed to load blueprints: {e}")
            return {}
    
    def _is_placeholder(self, value: Any) -> bool:
        """Check if a value looks like a placeholder."""
        if not isinstance(value, str):
            return False
        
        val_lower = value.lower().strip()
        
        # Keyword patterns
        placeholder_keywords = [
            'placeholder', 'tbd', 'todo', 'n/a', 'n-a', 'unknown',
            'pending', 'temp', 'dummy', 'sample', 'example', 'test-'
        ]
        for kw in placeholder_keywords:
            if kw in val_lower:
                return True
        
        # Bracket patterns: <...>, [...], {...}
        if (value.startswith('<') and value.endswith('>')) or \
           (value.startswith('[') and value.endswith(']')) or \
           (value.startswith('{') and value.endswith('}')):
            return True
        
        return False
    
    def _is_uuid(self, value: Any) -> bool:
        """Check if a string is in UUID format."""
        if not isinstance(value, str):
            return False
        return bool(self.UUID_PATTERN.match(value))
    
    def _get_entity_id_names(self, server_name: str) -> Set[str]:
        """Get set of entity ID field names for a server.
        
        Handles both single server names and fused cross-domain names.
        """
        entity_ids = set()
        
        # Get list of servers to check
        servers_to_check = [server_name]
        if "_" in server_name:
            servers_to_check.extend(server_name.split("_"))
        
        for srv in servers_to_check:
            blueprint = self.blueprints.get(srv, {})
            
            # Core entity
            core_entity = blueprint.get("core_entity", "")
            if core_entity:
                entity_ids.add(f"{core_entity.lower()}_id")
            
            # Peripheral entities
            for entity in blueprint.get("peripheral_entities", []):
                entity_ids.add(f"{entity.lower()}_id")
        
        return entity_ids
    
    def _get_free_text_fields(self, server_name: str) -> Set[str]:
        """
        Get set of field names that are free-text (range="string") for a server.
        
        Collects field names from both function parameters and relationship attributes
        where range="string".
        Handles both single server names and fused cross-domain names.
        """
        free_text_fields = set()
        
        # Get list of servers to check
        servers_to_check = [server_name]
        if "_" in server_name:
            servers_to_check.extend(server_name.split("_"))
        
        for srv in servers_to_check:
            blueprint = self.blueprints.get(srv, {})
            
            # Check function parameters
            for func in blueprint.get("functions", []):
                for param_name, param_info in func.get("parameters", {}).items():
                    if param_info.get("range") == "string":
                        free_text_fields.add(param_name)
            
            # Check relationship attributes
            for rel in blueprint.get("relationships", []):
                for attr_name, attr_info in rel.get("attributes", {}).items():
                    if attr_info.get("range") == "string":
                        free_text_fields.add(attr_name)
        
        return free_text_fields
    
    def _get_ignored_fields_from_blueprint(self, server_name: str) -> Set[str]:
        """
        Get set of field names marked with ignore_in_eval=true in blueprint.
        
        Checks both:
        1. Function parameters with ignore_in_eval=true
        2. Relationship attributes with ignore_in_eval=true
        
        Handles both single server names and fused cross-domain names.
        
        Returns:
            Set of field names that should be ignored during comparison
        """
        ignored_fields = set()
        
        # Get list of servers to check
        servers_to_check = [server_name]
        if "_" in server_name:
            servers_to_check.extend(server_name.split("_"))
        
        for srv in servers_to_check:
            blueprint = self.blueprints.get(srv, {})
            
            # Check function parameters
            for func in blueprint.get("functions", []):
                for param_name, param_info in func.get("parameters", {}).items():
                    if param_info.get("ignore_in_eval"):
                        ignored_fields.add(param_name)
            
            # Check relationship attributes
            for rel in blueprint.get("relationships", []):
                for attr_name, attr_info in rel.get("attributes", {}).items():
                    if attr_info.get("ignore_in_eval"):
                        ignored_fields.add(attr_name)
        
        return ignored_fields
    
    def _collect_placeholder_fields(self, *record_lists: List[Dict]) -> Set[str]:
        """
        Collect field names where ANY record has a placeholder value.
        
        This enables symmetric handling: if either gold or agent has a placeholder
        in a field, that field is ignored on BOTH sides during comparison.
        """
        placeholder_fields = set()
        for records in record_lists:
            for record in records:
                for key, value in record.items():
                    if self._is_placeholder(value):
                        placeholder_fields.add(key)
        return placeholder_fields
    
    def _is_dynamic_field(self, key: str, value: Any, server_name: str = "") -> bool:
        """
        Determine if a field should be ignored during comparison.
        
        Priority order:
        0. Explicit ignore_in_eval marker in blueprint (highest priority)
        1. Free-text fields (range="string" in blueprint, non-UUID value)
        2. Relationship ID fields (*_id not belonging to any entity)
        3. Timestamp fields
        4. Randomly generated business fields (like account_number)
        5. Execution-time dependent date fields
        
        Note: Placeholder values are handled separately via symmetric comparison
        to ensure fields are ignored on BOTH sides if either has a placeholder.
        """
        key_lower = key.lower()
        
        # Rule 0: Explicit ignore_in_eval marker (highest priority)
        if server_name:
            ignored_fields = self._get_ignored_fields_from_blueprint(server_name)
            if key in ignored_fields:
                return True
        
        # Rule 1: Free-text fields (range="string", non-UUID)
        if server_name:
            free_text_fields = self._get_free_text_fields(server_name)
            if key in free_text_fields and isinstance(value, str) and not self._is_uuid(value):
                return True
        
        # Rule 2: Relationship ID - any *_id not matching an entity
        if server_name:
            entity_ids = self._get_entity_id_names(server_name)
            if key_lower.endswith("_id") and key_lower not in entity_ids:
                return True
        
        # Rule 3: Timestamp fields
        if key_lower in self.TIMESTAMP_FIELDS:
            return True
        
        # Rule 4: Randomly generated number fields (e.g., account_number, reference_number)
        # Detected by: field name contains 'number' AND value is a numeric string
        if 'number' in key_lower and isinstance(value, str) and value.isdigit():
            return True
        
        # Rule 5: Execution-time dependent date fields
        # These are dates set when a record is created (opening_date, registration_date, etc.)
        if any(x in key_lower for x in ['opening_date', 'registration_date', 'creation_date']):
            return True
        
        return False
    
    def _normalize_value(self, value: Any) -> Any:
        """Normalize a value for comparison, handling floats and strings."""
        if isinstance(value, float):
            # Round floats to 2 decimal places to avoid precision issues
            return round(value, 2)
        elif isinstance(value, str):
            # Case-insensitive comparison for strings
            return value.lower()
        elif isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        return value
    
    def _normalize_record(
        self, 
        record: Dict, 
        server_name: str = "", 
        extra_ignore: Optional[Set[str]] = None,
    ) -> Dict:
        """
        Remove auto-generated fields and normalize values for comparison.
        
        Args:
            record: Record to normalize
            server_name: Server name for blueprint lookup
            extra_ignore: Additional field names to ignore (e.g., placeholder fields)
        """
        normalized = {}
        for k, v in record.items():
            # Skip extra ignore fields (e.g., placeholders on either side)
            if extra_ignore and k in extra_ignore:
                continue
            if not self._is_dynamic_field(k, v, server_name):
                normalized[k] = self._normalize_value(v)
        return normalized
    
    def _hash_table_data(
        self, 
        data: List[Dict], 
        server_name: str = "",
        extra_ignore: Optional[Set[str]] = None,
    ) -> str:
        """Generate a hash for table data, ignoring auto-generated fields."""
        # Normalize records to remove auto-generated fields and handle float precision
        normalized_data = [self._normalize_record(r, server_name, extra_ignore) for r in data]
        # Sort records for deterministic comparison
        sorted_data = sorted(normalized_data, key=lambda x: json.dumps(x, sort_keys=True, default=str))
        data_str = json.dumps(sorted_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _record_hash(
        self, 
        record: Dict, 
        server_name: str = "",
        extra_ignore: Optional[Set[str]] = None,
    ) -> str:
        """
        Compute hash for a single normalized record (excluding ID/timestamp fields).
        
        Used for efficient set-based diff comparison.
        """
        normalized = self._normalize_record(record, server_name, extra_ignore)
        return hashlib.md5(
            json.dumps(normalized, sort_keys=True, default=str).encode()
        ).hexdigest()
    
    def _get_table_hashes(
        self, 
        state: DatabaseState, 
        table: str, 
        server_name: str = "",
        extra_ignore: Optional[Set[str]] = None,
    ) -> set:
        """
        Get hash set for all records in a table.
        
        Args:
            state: DatabaseState to extract records from
            table: Table name
            server_name: Server name for blueprint lookup
            extra_ignore: Additional field names to ignore
            
        Returns:
            Set of record hashes
        """
        records = state.entities.get(table, []) or state.relationships.get(table, [])
        return {self._record_hash(r, server_name, extra_ignore) for r in records}
    
    def _build_hash_to_record_map(
        self,
        records: List[Dict],
        server_name: str = "",
        extra_ignore: Optional[Set[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Build a mapping from record hash to normalized record content.
        
        Args:
            records: List of records to process
            server_name: Server name for blueprint lookup
            extra_ignore: Additional field names to ignore
            
        Returns:
            Dict mapping hash -> normalized record
        """
        hash_to_record = {}
        for r in records:
            h = self._record_hash(r, server_name, extra_ignore)
            normalized = self._normalize_record(r, server_name, extra_ignore)
            hash_to_record[h] = normalized
        return hash_to_record
    
    def _find_closest_record(
        self,
        target_record: Dict,
        candidate_records: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """
        Find the closest matching record from candidates and analyze field differences.
        
        Args:
            target_record: The record to match against
            candidate_records: List of candidate records to search
            
        Returns:
            Dict with closest_record and differing_fields, or None if no candidates
        """
        if not candidate_records:
            return None
        
        best_match = None
        min_diff_count = float('inf')
        best_diff_fields = {}
        
        for candidate in candidate_records:
            diff_fields = {}
            all_keys = set(target_record.keys()) | set(candidate.keys())
            
            for key in all_keys:
                target_val = target_record.get(key)
                candidate_val = candidate.get(key)
                
                if target_val != candidate_val:
                    diff_fields[key] = {
                        "gold": target_val,
                        "agent": candidate_val,
                    }
            
            if len(diff_fields) < min_diff_count:
                min_diff_count = len(diff_fields)
                best_match = candidate
                best_diff_fields = diff_fields
        
        return {
            "closest_record": best_match,
            "differing_fields": best_diff_fields,
            "diff_count": min_diff_count,
        }
    
    def _compare_diffs(
        self,
        initial_state: DatabaseState,
        gold_state: DatabaseState,
        agent_state: DatabaseState,
        server_name: str = "",
    ) -> Dict[str, Any]:
        """
        Compare Gold and Agent diffs relative to initial state using Hash Sets.
        
        Pass condition: All Gold diff records must exist in Agent diff records.
        (i.e., gold_diff ⊆ agent_diff for each table)
        
        This allows Agent to make extra changes beyond what Gold requires.
        
        Args:
            initial_state: Database state before any execution
            gold_state: Database state after golden trajectory
            agent_state: Database state after agent trajectory
            server_name: Server name for blueprint lookup
            
        Returns:
            Dict with passed status, statistics, and detailed mismatch information
        """
        # Get all tables from both gold and agent states
        all_tables = (
            set(gold_state.entities.keys()) | set(gold_state.relationships.keys()) |
            set(agent_state.entities.keys()) | set(agent_state.relationships.keys())
        )
        
        missing_changes = []
        stats = {
            "gold_diff_total": 0,
            "agent_diff_total": 0,
            "tables_checked": 0,
        }
        
        for table in all_tables:
            # Get records from all states for this table
            initial_records = (
                initial_state.entities.get(table, []) or 
                initial_state.relationships.get(table, [])
            ) if initial_state else []
            gold_records = (
                gold_state.entities.get(table, []) or 
                gold_state.relationships.get(table, [])
            )
            agent_records = (
                agent_state.entities.get(table, []) or 
                agent_state.relationships.get(table, [])
            )
            
            # Collect placeholder fields from ALL records (symmetric handling)
            # If any record has a placeholder in a field, ignore that field on all sides
            placeholder_fields = self._collect_placeholder_fields(
                initial_records, gold_records, agent_records
            )
            
            # Build hash -> record mappings for lookup
            initial_hash_map = self._build_hash_to_record_map(
                initial_records, server_name, placeholder_fields
            ) if initial_records else {}
            gold_hash_map = self._build_hash_to_record_map(
                gold_records, server_name, placeholder_fields
            )
            agent_hash_map = self._build_hash_to_record_map(
                agent_records, server_name, placeholder_fields
            )
            
            # Get hash sets for each state
            initial_h = set(initial_hash_map.keys())
            gold_h = set(gold_hash_map.keys())
            agent_h = set(agent_hash_map.keys())
            
            # Compute diffs: records that are different from initial state
            gold_diff = gold_h - initial_h
            agent_diff = agent_h - initial_h
            
            stats["gold_diff_total"] += len(gold_diff)
            stats["agent_diff_total"] += len(agent_diff)
            stats["tables_checked"] += 1
            
            # Check if gold_diff is a subset of agent_diff
            missing = gold_diff - agent_diff
            extra = agent_diff - gold_diff
            
            if missing:
                # Get the actual missing records (Gold has but Agent doesn't)
                missing_records = [gold_hash_map[h] for h in missing]
                
                # Get extra records (Agent has but Gold doesn't)
                extra_records = [agent_hash_map[h] for h in extra]
                
                # Analyze field-level differences for each missing record
                field_differences = []
                for missing_rec in missing_records:
                    closest_match = self._find_closest_record(
                        target_record=missing_rec,
                        candidate_records=extra_records,
                    )
                    if closest_match:
                        field_differences.append({
                            "type": "missing_in_agent",
                            "gold_record": missing_rec,
                            "closest_agent_record": closest_match["closest_record"],
                            "differing_fields": closest_match["differing_fields"],
                            "diff_count": closest_match["diff_count"],
                        })
                    else:
                        field_differences.append({
                            "type": "missing_in_agent",
                            "gold_record": missing_rec,
                            "closest_agent_record": None,
                            "differing_fields": {},
                            "diff_count": None,
                            "note": "No candidate records in agent diff to compare",
                        })
                
                missing_changes.append({
                    "table": table,
                    "missing_count": len(missing),
                    "extra_count": len(extra),
                    "gold_diff_count": len(gold_diff),
                    "agent_diff_count": len(agent_diff),
                    "missing_records": missing_records,
                    "extra_records": extra_records,
                    "field_differences": field_differences,
                    "placeholder_fields_ignored": list(placeholder_fields) if placeholder_fields else [],
                })
        
        passed = len(missing_changes) == 0
        
        return {
            "passed": passed,
            "reason": "" if passed else f"{len(missing_changes)} table(s) missing expected changes",
            "stats": stats,
            "missing_changes": missing_changes,
        }
    
    def _run_assertions(
        self,
        assertions: List[Callable[[DatabaseState, DatabaseState], bool]],
        gold_state: DatabaseState,
        agent_state: DatabaseState,
    ) -> Dict[str, Any]:
        """
        Run custom environment assertions.
        
        Args:
            assertions: List of assertion functions
            gold_state: Expected database state
            agent_state: Actual database state
            
        Returns:
            Dictionary with assertion results
        """
        result = {
            "all_passed": True,
            "failure_reason": "",
            "details": [],
        }
        
        for i, assertion in enumerate(assertions):
            try:
                passed = assertion(gold_state, agent_state)
                result["details"].append({
                    "index": i,
                    "passed": passed,
                    "name": getattr(assertion, "__name__", f"assertion_{i}"),
                })
                
                if not passed:
                    result["all_passed"] = False
                    result["failure_reason"] = (
                        f"Assertion {i} ({getattr(assertion, '__name__', 'unnamed')}) failed"
                    )
                    
            except Exception as e:
                result["all_passed"] = False
                result["details"].append({
                    "index": i,
                    "passed": False,
                    "error": str(e),
                })
                result["failure_reason"] = f"Assertion {i} raised exception: {e}"
        
        return result
    
    def _extract_tool_results(
        self, 
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract tool call results from conversation messages.
        
        Parses the 'tool' role messages to get the returned values from each
        tool call, keyed by the tool_call_id.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Dict mapping tool_call_id to the parsed result dict
        """
        results = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                content = msg.get("content", "{}")
                if tool_call_id:
                    try:
                        results[tool_call_id] = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        # If content is not valid JSON, store as-is
                        results[tool_call_id] = {"raw_content": content}
        return results
    
    def _apply_id_mapping(
        self, 
        arguments: Dict[str, Any], 
        id_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Replace old IDs with new IDs in tool call arguments.
        
        Scans all string values in arguments and replaces any that match
        keys in the id_mapping with their mapped new values.
        
        Args:
            arguments: Original tool call arguments
            id_mapping: Mapping from old IDs to new IDs
            
        Returns:
            Arguments with IDs replaced
        """
        if not id_mapping:
            return arguments
        
        mapped_args = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value in id_mapping:
                mapped_args[key] = id_mapping[value]
                logger.debug(f"ID mapping applied: {key}: {value} -> {id_mapping[value]}")
            else:
                mapped_args[key] = value
        return mapped_args
    
    def _extract_relationship_ids(
        self, 
        result: Dict[str, Any],
        server_name: str = "",
    ) -> Dict[str, str]:
        """
        Extract relationship IDs from a tool call result.
        
        Identifies *_id fields that are NOT entity IDs (i.e., relationship IDs
        that are auto-generated during creation operations).
        
        Args:
            result: Tool call result dictionary
            server_name: Server name for blueprint lookup
            
        Returns:
            Dict of field_name -> id_value for relationship IDs
        """
        relationship_ids = {}
        entity_ids = self._get_entity_id_names(server_name) if server_name else set()
        
        for key, value in result.items():
            if key.endswith("_id") and isinstance(value, str):
                # Check if this is a relationship ID (not an entity ID)
                if key.lower() not in entity_ids:
                    relationship_ids[key] = value
        
        return relationship_ids
    
    def _extract_tool_calls_with_ids(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract tool calls from messages, preserving tool_call_id for ID mapping.
        
        Unlike ActionEvaluator._extract_tool_calls, this method preserves the
        tool_call_id which is needed to match tool calls with their results
        for building the ID mapping.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of dicts with 'name', 'arguments', and 'tool_call_id'
        """
        tool_calls = []
        
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            
            msg_tool_calls = msg.get("tool_calls")
            if not msg_tool_calls:
                continue
            
            for tc in msg_tool_calls:
                if "function" not in tc:
                    continue
                
                func = tc["function"]
                args = func.get("arguments", {})
                
                # Parse arguments if they're a JSON string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                
                # Keep the full function name with prefix for routing
                # e.g., "tcc.authorize_caregiver" stays as is for correct cross-domain routing
                func_name = func.get("name", "")
                
                tool_calls.append({
                    "name": func_name,
                    "arguments": args or {},
                    "tool_call_id": tc.get("id"),
                })
        
        return tool_calls
    
    def _rebuild_state_from_trajectory(
        self,
        task: TaskDefinition,
        messages: List[Dict[str, Any]],
        initial_state: Optional[DatabaseState],
    ) -> Optional[DatabaseState]:
        """
        Rebuild database state by replaying agent's tool calls.
        
        This is a fallback when agent_execution is not provided.
        We create a fresh server instance, replay the calls, and capture state.
        
        Uses ID mapping to handle dynamically generated relationship IDs:
        - When a tool call creates a new record (e.g., add_driver_sensor_link),
          the returned ID differs from the original conversation
        - We map old IDs to new IDs and apply this mapping to subsequent calls
        """
        try:
            # Extract tool calls from messages (with tool_call_id preserved)
            tool_calls = self._extract_tool_calls_with_ids(messages)
            
            if not tool_calls:
                return initial_state
            
            # Extract original tool results from conversation for ID mapping
            original_tool_results = self._extract_tool_results(messages)
            
            # Get domain and server - use same logic as TrajectoryExecutor
            domain_name = "_".join(task.domains) if task.is_cross_domain else task.domains[0]
            server_name = task.domains[0] if task.domains else ""
            
            # Build trajectory for cross-domain routing
            trajectory = [tc["name"] for tc in tool_calls]
            
            # Use TrajectoryExecutor's server loading logic for consistency
            from .trajectory_executor import TrajectoryExecutor
            executor = TrajectoryExecutor(outputs_dir=str(self.outputs_dir))
            server = executor._get_server(domain_name, trajectory=trajectory)
            
            # Generate a unique session
            import uuid
            session_id = f"rebuild_{uuid.uuid4().hex[:8]}"
            
            # ID mapping: old_id (from conversation) -> new_id (from replay)
            id_mapping: Dict[str, str] = {}
            
            # Replay tool calls with ID mapping
            for tc in tool_calls:
                tc_name = tc["name"]
                tc_arguments = tc["arguments"]
                tc_tool_call_id = tc.get("tool_call_id")
                
                try:
                    # Apply ID mapping to arguments before invoking
                    mapped_arguments = self._apply_id_mapping(tc_arguments, id_mapping)
                    
                    # Invoke the tool with mapped arguments
                    replay_result = server.invoke(
                        session_id=session_id,
                        tool_name=tc_name,
                        **mapped_arguments,
                    )
                    
                    # Extract relationship IDs from replay result and original result
                    # to build the mapping for subsequent calls
                    if tc_tool_call_id and tc_tool_call_id in original_tool_results:
                        original_result = original_tool_results[tc_tool_call_id]
                        
                        # Get relationship IDs from both results
                        original_rel_ids = self._extract_relationship_ids(
                            original_result, server_name
                        )
                        
                        # Parse replay result if it's a string
                        if isinstance(replay_result, str):
                            try:
                                replay_result_dict = json.loads(replay_result)
                            except (json.JSONDecodeError, TypeError):
                                replay_result_dict = {}
                        elif isinstance(replay_result, dict):
                            replay_result_dict = replay_result
                        else:
                            replay_result_dict = {}
                        
                        replay_rel_ids = self._extract_relationship_ids(
                            replay_result_dict, server_name
                        )
                        
                        # Build mapping: for each relationship ID field that exists
                        # in both results, map old -> new
                        for field_name, old_id in original_rel_ids.items():
                            if field_name in replay_rel_ids:
                                new_id = replay_rel_ids[field_name]
                                if old_id != new_id:
                                    id_mapping[old_id] = new_id
                                    logger.info(
                                        f"ID mapping created: {field_name}: "
                                        f"{old_id} -> {new_id}"
                                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to replay {tc_name}: {e}")
            
            # Capture final state from the server
            return self._capture_state_from_server(server, session_id, task.domains)
            
        except Exception as e:
            logger.error(f"Failed to rebuild state: {e}")
            return None
    
    def _capture_state(self, server, session_id: str) -> DatabaseState:
        """Capture database state from server."""
        try:
            session = server._get_session(session_id)
            db = session.databases
            
            entities = {}
            relationships = {}
            
            for key, value in db.items():
                if isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        # Determine if entity or relationship
                        if any(k.endswith("_id") and k.replace("_id", "").lower() == key.lower() 
                               for k in value[0].keys()):
                            entities[key] = deepcopy(value)
                        else:
                            relationships[key] = deepcopy(value)
            
            return DatabaseState(entities=entities, relationships=relationships)
            
        except Exception as e:
            logger.warning(f"Failed to capture state: {e}")
            return DatabaseState()
    
    def _capture_state_from_server(
        self, 
        server, 
        session_id: str, 
        domains: List[str]
    ) -> DatabaseState:
        """
        Capture database state from server, handling both single and cross-domain cases.
        """
        try:
            # Check if this is a CrossDomainServerWrapper
            if hasattr(server, '_servers') and hasattr(server, 'domains'):
                # Cross-domain case: ensure all domain servers are initialized
                entities = {}
                relationships = {}
                
                # Make sure all domain servers are initialized
                for domain in server.domains:
                    if domain not in server._servers:
                        server._get_server(domain)
                
                for domain, domain_server in server._servers.items():
                    try:
                        session = domain_server._get_session(session_id)
                        db = session.databases
                        
                        for key, value in db.items():
                            if isinstance(value, list) and value:
                                if isinstance(value[0], dict):
                                    if any(k.endswith("_id") and k.replace("_id", "").lower() == key.lower() 
                                           for k in value[0].keys()):
                                        if key not in entities:
                                            entities[key] = deepcopy(value)
                                    else:
                                        if key not in relationships:
                                            relationships[key] = deepcopy(value)
                    except Exception as e:
                        logger.warning(f"Failed to capture state from {domain}: {e}")
                
                return DatabaseState(entities=entities, relationships=relationships)
            else:
                # Single domain case
                return self._capture_state(server, session_id)
            
        except Exception as e:
            logger.warning(f"Failed to capture state from server: {e}")
            return DatabaseState()


# =============================================================================
# Common Environment Assertions
# =============================================================================

def assert_no_duplicate_records(gold: DatabaseState, agent: DatabaseState) -> bool:
    """Assert that agent didn't create duplicate records."""
    for table, records in agent.relationships.items():
        # Check for duplicates based on key fields
        seen = set()
        for record in records:
            # Use a hash of core fields as key
            key_str = json.dumps({k: v for k, v in sorted(record.items()) 
                                 if k.endswith("_id")}, sort_keys=True)
            if key_str in seen:
                return False
            seen.add(key_str)
    return True


def assert_referential_integrity(gold: DatabaseState, agent: DatabaseState) -> bool:
    """Assert that all foreign keys reference valid records."""
    entity_ids = {}
    
    # Collect all entity IDs
    for entity_type, records in agent.entities.items():
        id_field = f"{entity_type.lower()}_id"
        entity_ids[entity_type] = {
            r.get(id_field) for r in records if r.get(id_field)
        }
    
    # Check relationships
    for rel_type, records in agent.relationships.items():
        for record in records:
            for key, value in record.items():
                if key.endswith("_id") and isinstance(value, str):
                    # This looks like a foreign key
                    entity_type = key.replace("_id", "").title()
                    if entity_type in entity_ids:
                        if value not in entity_ids[entity_type]:
                            return False
    
    return True


def assert_record_count_match(gold: DatabaseState, agent: DatabaseState) -> bool:
    """Assert that record counts match between gold and agent states."""
    for table in set(gold.entities.keys()) | set(gold.relationships.keys()):
        gold_count = len(gold.entities.get(table, [])) + len(gold.relationships.get(table, []))
        agent_count = len(agent.entities.get(table, [])) + len(agent.relationships.get(table, []))
        
        if gold_count != agent_count:
            return False
    
    return True