"""
Action Evaluator - Verifies that agent executed correct tool calls.

Simplified flow:
1. Receive gold_execution from run_evaluation.py (with caching handled there)
2. Apply pruning to filter out redundant steps
3. Compare agent's tool_calls against expected
4. PASS if all expected calls are found in agent's calls
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from .base import (
    BaseEvaluator,
    EvaluatorType,
    EvaluationResult,
    ToolCall,
    TrajectoryExecution,
)
from .data_loader import EvaluationSample, compute_trajectory_hash
from .abbreviation_mappings import are_equivalent as abbrev_equivalent

logger = logging.getLogger(__name__)


class ActionEvaluator(BaseEvaluator):
    """
    Evaluates whether agent executed the correct tool calls.
    
    The evaluator:
    1. Receives gold_execution from caller (caching handled in run_evaluation.py)
    2. Applies pruning to filter redundant steps
    3. Compares with agent's tool_calls
    4. Returns 1.0 if ALL tool names are found
    
    Features:
    - Order independent: Golden actions can appear in any order
    - Extra calls allowed: Agent can make additional calls beyond golden
    - Two-level comparison: Tool names (for score) + params (for statistics)
    - No internal execution: Requires gold_execution to be passed in
    """
    
    evaluator_type = EvaluatorType.ACTION
    
    # UUID pattern for validation
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', 
        re.IGNORECASE
    )
    
    def __init__(
        self,
        pruning_index_path: str = "outputs/trajectory_code/pruning_index.json",
        outputs_dir: str = "outputs",
        blueprints_path: str = "outputs/blueprints.json",
        ignore_params: List[str] = None,
    ):
        """
        Initialize the action evaluator.
        
        Args:
            pruning_index_path: Path to pruning_index.json
            outputs_dir: Base outputs directory
            blueprints_path: Path to blueprints.json for param info
            ignore_params: Parameters to ignore in comparison (e.g., 'confirm')
        """
        self.outputs_dir = Path(outputs_dir)
        self.default_ignore_params = ignore_params or ["confirm"]
        self.pruning_index = self._load_pruning_index(pruning_index_path)
        self.blueprints = self._load_blueprints(blueprints_path)
    
    def _load_pruning_index(self, path: str) -> Dict[str, Any]:
        """Load the pruning index from disk."""
        pruning_path = Path(path)
        if pruning_path.exists():
            try:
                with open(pruning_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load pruning index: {e}")
        return {}
    
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
            
            logger.info(f"Loaded {len(indexed)} blueprints")
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
    
    def _is_uuid(self, value: str) -> bool:
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
    
    def _get_param_info(self, server_name: str, function_name: str, param_name: str) -> Optional[Dict]:
        """Get parameter info from blueprint.
        
        Handles both single server names and fused cross-domain names
        (e.g., 'ServerA_ServerB_ServerC').
        """
        # Try direct lookup first
        blueprint = self.blueprints.get(server_name, {})
        for func in blueprint.get("functions", []):
            if func.get("name") == function_name:
                return func.get("parameters", {}).get(param_name)
        
        # If not found and server_name contains underscores, try component servers
        if "_" in server_name:
            component_servers = server_name.split("_")
            for component in component_servers:
                blueprint = self.blueprints.get(component, {})
                for func in blueprint.get("functions", []):
                    if func.get("name") == function_name:
                        return func.get("parameters", {}).get(param_name)
        
        return None
    
    def _get_function_info(self, server_name: str, function_name: str) -> Optional[Dict]:
        """Get function info from blueprint.
        
        Handles both single server names and fused cross-domain names.
        """
        # Try direct lookup first
        blueprint = self.blueprints.get(server_name, {})
        for func in blueprint.get("functions", []):
            if func.get("name") == function_name:
                return func
        
        # If not found and server_name contains underscores, try component servers
        if "_" in server_name:
            component_servers = server_name.split("_")
            for component in component_servers:
                blueprint = self.blueprints.get(component, {})
                for func in blueprint.get("functions", []):
                    if func.get("name") == function_name:
                        return func
        
        return None
    
    def _is_filter_function(self, server_name: str, function_name: str) -> bool:
        """Check if a function is a filter function based on its description."""
        func_info = self._get_function_info(server_name, function_name)
        if not func_info:
            return False
        
        description = func_info.get("description", "").lower()
        # Check for filter-related keywords
        return "filter" in description
    
    def _should_ignore_param(
        self,
        param_name: str,
        exp_val: Any,
        act_val: Any,
        server_name: str,
        function_name: str,
    ) -> bool:
        """
        Check if a parameter should be ignored in comparison.
        
        Priority order:
        0. Explicit ignore_in_eval marker in blueprint (highest priority)
        1. Placeholder values (e.g., "placeholder-xxx", "TBD", "<value>")
        2. Notes - free text string fields (range="string", non-UUID value)
        3. Relationship IDs - *_id fields not belonging to any entity
        4. Filter parameters:
           a. Function description contains "filter" (e.g., "filtered by", "filtering")
           b. Parameter description contains "filter"
        """
        # Get param info from blueprint (used by multiple rules)
        param_info = self._get_param_info(server_name, function_name, param_name)
        
        # Rule 0: Explicit ignore_in_eval marker (highest priority)
        if param_info and param_info.get("ignore_in_eval"):
            return True
        
        # Rule 1: Placeholder values
        if self._is_placeholder(exp_val) or self._is_placeholder(act_val):
            return True
        
        # Rule 2: Notes - free text string (range="string", non-UUID value)
        if param_info and param_info.get("range") == "string":
            if isinstance(exp_val, str) and not self._is_uuid(exp_val):
                return True
        
        # Rule 3: Relationship ID - *_id not in entity list
        entity_ids = self._get_entity_id_names(server_name)
        if param_name.endswith("_id") and param_name not in entity_ids:
            return True
        
        # Rule 4a: Filter function - function description contains "filter"
        if self._is_filter_function(server_name, function_name):
            return True
        
        # Rule 4b: Filter parameter - parameter description contains "filter"
        if param_info:
            description = param_info.get("description", "")
            if "filter" in description.lower():
                return True
        
        return False
    
    def evaluate(
        self,
        sample: EvaluationSample,
        gold_execution: TrajectoryExecution,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate the agent's tool calls against expected trajectory.
        
        Args:
            sample: EvaluationSample containing trajectory, entity_context, rollout_messages
            gold_execution: Pre-executed golden trajectory (caching handled by caller)
        
        Returns:
            EvaluationResult: Pass/fail with details
        """
        # 1. Get expected tool calls from gold_execution
        if not gold_execution.success:
            logger.warning(f"Trajectory execution had errors: {gold_execution.errors}")
        expected_calls = gold_execution.tool_calls  # List[ToolCall] with name + arguments
        
        # Strip domain name prefix from expected calls (for consistency with agent calls)
        expected_calls = [
            ToolCall(name=tc.name.split(".")[-1], arguments=tc.arguments)
            for tc in expected_calls
        ]
        
        # 2. Apply pruning: filter to only pruned trajectory tools
        traj_hash = compute_trajectory_hash(sample.trajectory)
        pruning_info = self.pruning_index.get(traj_hash, {})
        
        if pruning_info:
            # Get indices to keep (not in redundant_steps)
            redundant_indices = set(
                s["index"] for s in pruning_info.get("redundant_steps", [])
            )
            kept_indices = [
                i for i in range(len(sample.trajectory))
                if i not in redundant_indices
            ]
            
            # Filter expected_calls to only kept indices
            if len(expected_calls) == len(sample.trajectory):
                expected_calls = [expected_calls[i] for i in kept_indices]
            else:
                # Fallback: match by function name from pruned_trajectory
                pruned_traj = pruning_info.get("pruned_trajectory", sample.trajectory)
                expected_calls = [
                    tc for tc in expected_calls 
                    if tc.name in pruned_traj
                ]
        
        # 6. Extract agent's tool calls from rollout
        agent_calls = self._extract_tool_calls(sample.rollout_messages)
        
        if not agent_calls:
            return self._create_fail_result(
                "No tool calls found in agent trajectory",
                {"expected_count": len(expected_calls), "actual_count": 0}
            )
        
        # 7. Compare: two-level check (name only + name+params)
        # Get server name from sample for blueprint lookup
        # Use fused name (e.g., "ServerA_ServerB_ServerC") so _get_param_info can search all component servers
        server_name = "_".join(sample.domains) if sample.domains else ""
        
        comparison = self._compare_actions(
            expected=expected_calls,
            actual=agent_calls,
            server_name=server_name,
        )
        
        # Build detailed info for debugging
        expected_summary = [
            {"step": i, "name": tc.name, "params": tc.arguments}
            for i, tc in enumerate(expected_calls)
        ]
        actual_summary = [
            {"step": i, "name": tc.name, "params": tc.arguments}
            for i, tc in enumerate(agent_calls)
        ]
        
        # Main score: only based on Tool Name matching
        passed = comparison["all_names_matched"]
        score = 1.0 if passed else 0.0
        
        # Build statistics for output
        total_expected = len(expected_calls)
        tool_name_stats = {
            "total_expected": total_expected,
            "names_matched": comparison["name_match_count"],
            "all_matched": comparison["all_names_matched"],
            "missing_tools": comparison["missing_tools"],
        }
        
        param_match_stats = {
            "full_matched": comparison["full_match_count"],
            "full_match_rate": (
                comparison["full_match_count"] / total_expected 
                if total_expected > 0 else 0.0
            ),
            "mismatch_cases": comparison["param_mismatch_cases"],
        }
        
        details = {
            "tool_name_stats": tool_name_stats,
            "param_match_stats": param_match_stats,
            "expected_trajectory": expected_summary,
            "actual_trajectory": actual_summary,
        }
        
        if passed:
            return self._create_pass_result(details)
        else:
            # Build detailed failure message for debugging
            failure_details = self._build_failure_details(
                expected_calls, agent_calls, comparison
            )
            details["failure_details"] = failure_details
            
            failure_reason = (
                f"Missing tools: {comparison['missing_tools']}"
                if comparison["missing_tools"]
                else "Unknown failure"
            )
            
            return self._create_fail_result(failure_reason, details)

    def _extract_tool_calls(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[ToolCall]:
        """
        Extract tool calls from the agent's messages.
        Handles OpenAI function calling format.
        """
        tool_calls = []

        for msg in messages:
            # Skip non-assistant messages
            role = msg.get("role", "")
            if role not in ["assistant", "tool"]:
                continue

            if "tool_calls" in msg and msg["tool_calls"] is not None:
                for tc in msg["tool_calls"]:
                    if "function" in tc:
                        func = tc["function"]
                        args = func.get("arguments", None)
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        
                        # Strip domain name prefix
                        func_name = func["name"].split(".")[-1]

                        tool_calls.append(ToolCall(
                            name=func_name,
                            arguments=args or {},
                        ))

        return tool_calls

    def _compare_actions(
        self,
        expected: List[ToolCall],
        actual: List[ToolCall],
        ignore_params: Optional[List[str]] = None,
        server_name: str = "",
    ) -> Dict[str, Any]:
        """
        Compare expected (golden) actions against actual tool calls.
        
        Two-level comparison:
        1. Tool Name matching (determines main score)
        2. Tool Name + Parameter matching (statistics for bonus scoring)
        
        Returns:
            Dictionary with:
            - all_names_matched: bool - whether all tool names were found
            - name_match_count: int - number of tool names matched
            - full_match_count: int - number of full matches (name + params)
            - missing_tools: list - tool names not found in agent calls
            - name_match_results: list - per-tool name match results
            - full_match_results: list - per-tool full match results
            - param_mismatch_cases: list - details of parameter mismatches
        """
        # Merge with default ignore params
        all_ignore_params = list(self.default_ignore_params)
        if ignore_params:
            all_ignore_params.extend(ignore_params)
        ignore_params = all_ignore_params
        
        # 1. Tool Name matching (determines main score)
        name_match_results = []
        missing_tools = []
        
        for gold_action in expected:
            name_found = any(a.name == gold_action.name for a in actual)
            name_match_results.append({
                "function": gold_action.name,
                "found": name_found,
            })
            if not name_found:
                missing_tools.append(gold_action.name)
        
        all_names_matched = len(missing_tools) == 0
        
        # 2. Full Match (Name + Params) for statistics
        full_match_results = []
        param_mismatch_cases = []
        
        for i, gold_action in enumerate(expected):
            matched, matched_call, mismatch_details = self._find_full_match(
                gold_action, actual, ignore_params, server_name
            )
            
            full_match_results.append({
                "step": i,
                "function": gold_action.name,
                "full_matched": matched,
            })
            
            if not matched and mismatch_details:
                param_mismatch_cases.append({
                    "step": i,
                    "function": gold_action.name,
                    "expected_params": gold_action.arguments,
                    "actual_params": matched_call.arguments if matched_call else None,
                    "mismatch_details": mismatch_details,
                })
        
        return {
            "all_names_matched": all_names_matched,
            "name_match_count": sum(1 for r in name_match_results if r["found"]),
            "full_match_count": sum(1 for r in full_match_results if r["full_matched"]),
            "missing_tools": missing_tools,
            "name_match_results": name_match_results,
            "full_match_results": full_match_results,
            "param_mismatch_cases": param_mismatch_cases,
        }
    
    def _find_full_match(
        self,
        gold_action: ToolCall,
        actual_calls: List[ToolCall],
        ignore_params: List[str],
        server_name: str = "",
    ) -> tuple[bool, Optional[ToolCall], Optional[Dict[str, Any]]]:
        """
        Find a full match (name + params) for a golden action in actual calls.
        
        Args:
            gold_action: The expected tool call
            actual_calls: List of actual tool calls from agent
            ignore_params: Parameters to ignore in comparison
            server_name: Server name for blueprint lookup
        
        Returns:
            Tuple of (matched, matched_call, mismatch_details):
            - matched: True if full match found
            - matched_call: The matching ToolCall (or closest name match)
            - mismatch_details: Details about parameter mismatches (if any)
        """
        # Find calls with matching name
        name_matches = [tc for tc in actual_calls if tc.name == gold_action.name]
        
        if not name_matches:
            return False, None, {"reason": "No matching function name"}
        
        # Collect all mismatches for debugging - check ALL name matches
        all_mismatches = []
        for actual_call in name_matches:
            param_mismatches = self._check_params(
                gold_action, actual_call, ignore_params, server_name
            )
            
            if not param_mismatches:
                # Full match found
                return True, actual_call, None
            
            all_mismatches.append({
                "call": actual_call,
                "mismatches": param_mismatches,
            })
        
        # No full match found - return the one with fewest mismatches (closest match)
        best_match = min(all_mismatches, key=lambda x: len(x["mismatches"]))
        
        return False, best_match["call"], {
            "reason": "Parameter mismatch",
            "mismatched_keys": list(best_match["mismatches"].keys()),
            "mismatches": best_match["mismatches"],
            "all_attempts": len(all_mismatches),  # Show how many were tried
        }
    
    def _check_params(
        self,
        expected: ToolCall,
        actual: ToolCall,
        ignore_params: List[str],
        server_name: str = "",
    ) -> Dict[str, Any]:
        """
        Check parameter differences between expected and actual calls.
        
        Returns:
            Dict of mismatched parameters: {param_name: {"expected": ..., "actual": ...}}
            Empty dict if all params match.
        """
        mismatches = {}
        function_name = expected.name
        
        # Get params to check (expected params minus ignored)
        params_to_check = [
            p for p in expected.arguments.keys() 
            if p not in ignore_params
        ]
        
        for param in params_to_check:
            exp_val = expected.arguments.get(param)
            act_val = actual.arguments.get(param)
            
            # Check if this param should be ignored (placeholder, note, or relationship ID)
            if server_name and self._should_ignore_param(
                param, exp_val, act_val, server_name, function_name
            ):
                continue
            
            if not self._values_match(exp_val, act_val):
                mismatches[param] = {
                    "expected": exp_val,
                    "actual": act_val,
                }
        
        return mismatches
    
    def _actions_match(
        self,
        expected: ToolCall,
        actual: ToolCall,
        ignore_params: List[str],
        server_name: str = "",
    ) -> bool:
        """Check if two tool calls match (name + params)."""
        # Function name must match
        if expected.name != actual.name:
            return False
        
        # Check parameters
        param_mismatches = self._check_params(expected, actual, ignore_params, server_name)
        return len(param_mismatches) == 0
    
    def _values_match(self, expected: Any, actual: Any) -> bool:
        """Compare two values with type flexibility.
        
        Lenient null matching rules:
        - If expected is None, any actual value is acceptable (return True)
        - If actual is None but expected is not, check if expected is "empty"
        """
        # Rule: expected is null -> any actual value is acceptable
        if expected is None:
            return True
        
        # actual is null but expected is not null
        if actual is None:
            # Consider empty values as equivalent to None
            return expected == "" or expected == [] or expected == {}
        
        # String comparison
        if isinstance(expected, str) and isinstance(actual, str):
            # Try to parse as JSON for semantic comparison (handles formatting differences)
            if expected.startswith('{') and actual.startswith('{'):
                try:
                    exp_json = json.loads(expected)
                    act_json = json.loads(actual)
                    return self._values_match(exp_json, act_json)
                except json.JSONDecodeError:
                    pass  # Fall through to string comparison
            
            # Case-insensitive string comparison
            if expected.lower() == actual.lower():
                return True
            
            # Check abbreviation equivalence (e.g., "FR" == "France")
            if abbrev_equivalent(expected, actual):
                return True
            
            return False
        
        # Numeric comparison with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if isinstance(expected, float) or isinstance(actual, float):
                return abs(expected - actual) < 0.0001
            return expected == actual
        
        # List comparison
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(
                self._values_match(e, a) 
                for e, a in zip(expected, actual)
            )
        
        # Dict comparison
        if isinstance(expected, dict) and isinstance(actual, dict):
            # Only check keys that exist in expected
            for k, exp_v in expected.items():
                if exp_v is None:
                    # expected is null - any actual value (or missing key) is acceptable
                    continue
                
                # expected is not null - actual must have key with matching value
                if k not in actual:
                    return False  # Missing required key
                if not self._values_match(exp_v, actual[k]):
                    return False  # Value mismatch
            
            # Extra keys in actual are completely ignored
            return True
        
        # Default: direct comparison
        return expected == actual
    
    def _build_failure_details(
        self,
        expected_calls: List[ToolCall],
        actual_calls: List[ToolCall],
        comparison_result: Dict[str, Any],
    ) -> str:
        """Build a detailed failure message for debugging."""
        lines = []
        
        lines.append("\n" + "="*60)
        lines.append("ACTION EVALUATION FAILURE DETAILS")
        lines.append("="*60)
        
        # Summary stats
        lines.append(f"\n📊 SUMMARY:")
        lines.append(f"  Tool Names: {comparison_result['name_match_count']}/{len(expected_calls)} matched")
        lines.append(f"  Full Match: {comparison_result['full_match_count']}/{len(expected_calls)}")
        if comparison_result["missing_tools"]:
            lines.append(f"  Missing Tools: {comparison_result['missing_tools']}")
        
        # Expected trajectory
        lines.append("\n📋 EXPECTED TRAJECTORY (Golden, after pruning):")
        for i, tc in enumerate(expected_calls):
            lines.append(f"  Step {i}: {tc.name}")
            if tc.arguments:
                for k, v in tc.arguments.items():
                    v_str = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
                    lines.append(f"    - {k}: {v_str}")
        
        # Actual trajectory
        lines.append("\n📋 ACTUAL TRAJECTORY (Agent):")
        for i, tc in enumerate(actual_calls):
            lines.append(f"  Step {i}: {tc.name}")
            if tc.arguments:
                for k, v in tc.arguments.items():
                    v_str = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
                    lines.append(f"    - {k}: {v_str}")
        
        # Missing tools
        if comparison_result["missing_tools"]:
            lines.append("\n❌ MISSING TOOLS (not found in agent trajectory):")
            for tool_name in comparison_result["missing_tools"]:
                lines.append(f"  - {tool_name}")
        
        # Parameter mismatches (for reference)
        if comparison_result["param_mismatch_cases"]:
            lines.append("\n⚠️ PARAMETER MISMATCHES (tool name matched, params differ):")
            for case in comparison_result["param_mismatch_cases"]:
                lines.append(f"\n  Step {case['step']}: {case['function']}")
                mismatch_details = case.get("mismatch_details", {})
                mismatches = mismatch_details.get("mismatches", {})
                for param, values in mismatches.items():
                    exp_str = str(values["expected"])[:50]
                    act_str = str(values["actual"])[:50]
                    lines.append(f"    - {param}:")
                    lines.append(f"        expected: {exp_str}")
                    lines.append(f"        actual:   {act_str}")
        
        lines.append("\n" + "="*60)
        
        return "\n".join(lines)
    
    def _find_closest_match(
        self,
        func_name: str,
        exp_args: Dict[str, Any],
        actual_calls: List[ToolCall],
    ) -> Optional[Dict[str, Any]]:
        """Find the closest matching call in actual calls for debugging."""
        # First, try to find by function name
        for tc in actual_calls:
            if tc.name == func_name:
                # Found matching function, identify param mismatches
                mismatches = []
                for k, exp_v in exp_args.items():
                    if k in self.default_ignore_params:
                        continue
                    act_v = tc.arguments.get(k)
                    if not self._values_match(exp_v, act_v):
                        mismatches.append(f"{k}: expected '{exp_v}', got '{act_v}'")
                
                return {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "mismatch_reason": "; ".join(mismatches) if mismatches else "Unknown mismatch",
                }
        
        return None
