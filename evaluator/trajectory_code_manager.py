"""
Trajectory Code Manager - Manages generated code for trajectory templates.

This module provides:
1. Persistent storage for generated trajectory execution code
2. Mapping between trajectory templates and their execution code
3. Quick lookup from sample -> trajectory -> code
"""
from __future__ import annotations

import ast
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
from datetime import datetime

if TYPE_CHECKING:
    from .base import TaskDefinition

logger = logging.getLogger(__name__)


# =============================================================================
# AST Analysis Utilities for Code Caching
# =============================================================================

def analyze_param_code(code: str) -> Dict[str, Any]:
    """
    Analyze param extraction code to detect hardcoded ctx and collect ctx key references.
    
    This is used to determine if cached code can be reused with different entity_context.
    
    Args:
        code: The param extraction Python code
        
    Returns:
        {
            "has_hardcoded_ctx": bool,  # True if code contains `ctx = {...}`
            "hardcoded_ctx_value": dict or None,  # The hardcoded value if found
            "ctx_keys_used": List[str],  # Top-level ctx keys referenced (e.g., ["Student", "step_0_result"])
            "parse_error": str or None,  # Error message if parsing failed
        }
    """
    result = {
        "has_hardcoded_ctx": False,
        "hardcoded_ctx_value": None,
        "ctx_keys_used": [],
        "parse_error": None,
    }
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["parse_error"] = str(e)
        return result
    
    ctx_keys = set()
    
    for node in ast.walk(tree):
        # Detect `ctx = {...}` hardcoded assignment
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ctx":
                    if isinstance(node.value, ast.Dict):
                        result["has_hardcoded_ctx"] = True
                        try:
                            result["hardcoded_ctx_value"] = ast.literal_eval(
                                ast.unparse(node.value)
                            )
                        except (ValueError, TypeError):
                            # Complex dict that can't be literal_eval'd
                            result["hardcoded_ctx_value"] = None
        
        # Collect ctx["key"] subscript references
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "ctx":
                if isinstance(node.slice, ast.Constant):
                    ctx_keys.add(node.slice.value)
        
        # Collect ctx.get("key", ...) call references
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == "ctx" and 
                    node.func.attr == "get"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        ctx_keys.add(node.args[0].value)
    
    result["ctx_keys_used"] = sorted(ctx_keys)
    return result


def fix_hardcoded_ctx(code: str, hardcoded_value: Optional[dict] = None) -> str:
    """
    Fix hardcoded ctx in code to make it reusable with different entity_context.
    
    Transforms:
        ctx = {"Student": {"id": "123"}}
        params = {"id": ctx["Student"]["id"]}
    
    Into:
        _default_ctx = {"Student": {"id": "123"}}
        ctx = _default_ctx if ctx is None else ctx
        params = {"id": ctx["Student"]["id"]}
    
    Args:
        code: The original Python code
        hardcoded_value: Optional pre-extracted hardcoded value (for efficiency)
        
    Returns:
        Fixed code that accepts ctx as parameter
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # Can't fix, return original
    
    # Find the ctx assignment
    ctx_assign_idx = None
    ctx_value_node = None
    
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ctx":
                    if isinstance(node.value, ast.Dict):
                        ctx_assign_idx = i
                        ctx_value_node = node.value
                        break
    
    if ctx_assign_idx is None:
        return code  # No hardcoded ctx found
    
    # Build the fixed code
    lines = code.split('\n')
    new_lines = []
    
    # Track if we've processed the ctx assignment
    processed_ctx = False
    
    try:
        # Re-parse to find line numbers
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if this line contains `ctx = {`
            if not processed_ctx and stripped.startswith('ctx') and '=' in stripped:
                # Check if it's assigning a dict
                try:
                    # Find the full assignment (might span multiple lines)
                    assign_start = i
                    assign_text = line
                    brace_count = assign_text.count('{') - assign_text.count('}')
                    
                    j = i + 1
                    while brace_count > 0 and j < len(lines):
                        assign_text += '\n' + lines[j]
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        j += 1
                    
                    # Parse just this assignment
                    try:
                        test_tree = ast.parse(assign_text.strip())
                        if (test_tree.body and isinstance(test_tree.body[0], ast.Assign)):
                            assign_node = test_tree.body[0]
                            if (assign_node.targets and 
                                isinstance(assign_node.targets[0], ast.Name) and
                                assign_node.targets[0].id == "ctx" and
                                isinstance(assign_node.value, ast.Dict)):
                                
                                # Get indentation
                                indent = len(line) - len(line.lstrip())
                                indent_str = line[:indent]
                                
                                # Extract the dict literal
                                dict_start = line.find('{')
                                if dict_start != -1:
                                    # Build multi-line dict if needed
                                    dict_text = line[dict_start:]
                                    for k in range(i + 1, j):
                                        dict_text += '\n' + lines[k]
                                    
                                    # Add the fixed lines
                                    new_lines.append(f"{indent_str}_default_ctx = {dict_text.strip()}")
                                    new_lines.append(f"{indent_str}ctx = _default_ctx if ctx is None else ctx")
                                    
                                    # Skip the original lines
                                    processed_ctx = True
                                    # Skip lines that were part of the multi-line dict
                                    for _ in range(i + 1, j):
                                        lines[_] = ""  # Mark as processed
                                    continue
                    except SyntaxError:
                        pass
                except Exception:
                    pass
            
            if line:  # Skip empty lines we marked
                new_lines.append(line)
        
        return '\n'.join(new_lines)
        
    except Exception as e:
        logger.warning(f"Failed to fix hardcoded ctx: {e}")
        return code


def validate_ctx_keys(ctx_keys_used: List[str], entity_context: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that all ctx keys used in code exist in the entity_context.
    
    Args:
        ctx_keys_used: List of ctx keys referenced in code
        entity_context: The entity context to validate against
        
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    # Normalize entity_context keys (handle both flat and nested formats)
    available_keys = set()
    
    for key in entity_context.keys():
        if "." in key:
            # Flat format: "Student.student_id" -> add "Student"
            entity = key.split(".")[0]
            available_keys.add(entity)
        else:
            # Nested format or direct key
            available_keys.add(key)
    
    # step_N_result keys are always valid (they're generated at runtime)
    missing_keys = []
    for key in ctx_keys_used:
        if key.startswith("step_") and "_result" in key:
            continue
        if key not in available_keys:
            missing_keys.append(key)
    
    return len(missing_keys) == 0, missing_keys


def compute_trajectory_hash(trajectory: List[str]) -> str:
    """
    Compute a unique hash for a trajectory.
    
    This is the canonical implementation used across the workflow.
    Both s15 (instance_combos_selection) and s16 (task_filtering) use this function.
    
    Args:
        trajectory: List of function names in order
        
    Returns:
        A short hash string (12 chars) that uniquely identifies the trajectory
    """
    # Use underscore separator and MD5 for consistency with s15 file naming
    trajectory_str = "_".join(trajectory)
    full_hash = hashlib.md5(trajectory_str.encode()).hexdigest()
    return full_hash[:12]


@dataclass
class PruningResultEntry:
    """
    Cached pruning result for a trajectory.
    
    Pruning is trajectory-level: same trajectory_hash shares the same result.
    
    Attributes:
        trajectory_hash: Unique hash of the trajectory
        original_trajectory: Original function call sequence
        pruned_trajectory: Trajectory after removing redundant steps
        redundant_steps: List of removed steps with reasons
        pruning_ratio: Percentage of steps removed
        created_at: When this pruning was performed
        domains: Domains involved (for organizing files)
    """
    trajectory_hash: str
    original_trajectory: List[str]
    pruned_trajectory: List[str]
    redundant_steps: List[Dict[str, Any]]  # [{index, tool, reason, overlapping_source}]
    pruning_ratio: float
    created_at: str
    domains: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PruningResultEntry":
        return cls(**data)
    
    @property
    def kept_indices(self) -> List[int]:
        """Get indices of steps that were kept."""
        removed_indices = {s["index"] for s in self.redundant_steps}
        return [i for i in range(len(self.original_trajectory)) if i not in removed_indices]


@dataclass
class TrajectoryCodeEntry:
    """
    Stored entry for a trajectory's execution code.
    
    Attributes:
        trajectory_hash: Unique hash of the trajectory
        trajectory: The function call sequence
        domains: List of domains involved
        is_cross_domain: Whether this is a cross-domain trajectory
        generated_code: The Python code to execute this trajectory
        step_param_codes: Raw param extraction code for each step (for reuse)
        code_path: Path to the saved code file
        entity_mappings: Mapping of entity context keys needed
        created_at: When this entry was created
        last_used: When this entry was last used
        sample_count: Number of samples that use this trajectory
    """
    trajectory_hash: str
    trajectory: List[str]
    domains: List[str]
    is_cross_domain: bool
    generated_code: str
    code_path: str
    entity_mappings: Dict[str, str]  # param_name -> Entity.field
    created_at: str
    last_used: str
    step_param_codes: List[str] = field(default_factory=list)
    sample_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryCodeEntry":
        # Handle missing step_param_codes for backward compatibility
        if "step_param_codes" not in data:
            data["step_param_codes"] = []
        return cls(**data)


class TrajectoryCodeManager:
    """
    Manages the lifecycle of generated trajectory code.
    
    Storage Structure:
        outputs_cursor/
          trajectory_code/
            index.json                    # Main index: trajectory_hash -> metadata
            single_domain/
              <domain>/
                <hash>.py                 # Executable code
                <hash>.json               # Metadata
            cross_domain/
              <domains_combined>/
                <hash>.py
                <hash>.json
    
    Usage:
        manager = TrajectoryCodeManager("./outputs_cursor")
        
        # Generate and save code for a trajectory
        entry = manager.generate_and_save(task_definition)
        
        # Look up code by trajectory
        code = manager.get_code_for_trajectory(trajectory)
        
        # Look up code by sample
        code = manager.get_code_for_sample(sample_data)
    """
    
    def __init__(self, outputs_dir: str = "./outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.code_dir = self.outputs_dir / "trajectory_code"
        self.index_path = self.code_dir / "index.json"
        
        # Thread lock for protecting _index and _pruning_cache access
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self.code_dir.mkdir(parents=True, exist_ok=True)
        (self.code_dir / "single_domain").mkdir(exist_ok=True)
        (self.code_dir / "cross_domain").mkdir(exist_ok=True)
        
        # Load or create index
        self._index: Dict[str, TrajectoryCodeEntry] = {}
        self._load_index()
        
        # Load or create pruning cache
        self._pruning_cache: Dict[str, PruningResultEntry] = {}
        self._load_pruning_cache()
    
    def _load_index(self):
        """Load the index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    data = json.load(f)
                self._index = {
                    k: TrajectoryCodeEntry.from_dict(v) 
                    for k, v in data.items()
                }
                logger.info(f"Loaded {len(self._index)} trajectory entries from index")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self):
        """Save the index to disk. Must be called with _lock held."""
        # Create a snapshot of the index to avoid iteration issues
        index_snapshot = dict(self._index)
        data = {k: v.to_dict() for k, v in index_snapshot.items()}
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_code_dir(self, domains: List[str], is_cross_domain: bool) -> Path:
        """Get the directory for storing code files."""
        if is_cross_domain:
            domain_name = "_".join(sorted(domains))
            code_path = self.code_dir / "cross_domain" / domain_name
        else:
            code_path = self.code_dir / "single_domain" / domains[0]
        
        code_path.mkdir(parents=True, exist_ok=True)
        return code_path
    
    def get_trajectory_hash(self, trajectory: List[str]) -> str:
        """Get the hash for a trajectory."""
        return compute_trajectory_hash(trajectory)
    
    def has_code(self, trajectory: List[str]) -> bool:
        """Check if code exists for a trajectory."""
        traj_hash = compute_trajectory_hash(trajectory)
        return traj_hash in self._index
    
    def get_entry(self, trajectory: List[str]) -> Optional[TrajectoryCodeEntry]:
        """Get the entry for a trajectory if it exists."""
        traj_hash = compute_trajectory_hash(trajectory)
        return self._index.get(traj_hash)
    
    def get_entry_by_hash(self, traj_hash: str) -> Optional[TrajectoryCodeEntry]:
        """Get the entry for a trajectory hash if it exists."""
        return self._index.get(traj_hash)
    
    def get_code(self, trajectory: List[str]) -> Optional[str]:
        """Get the generated code for a trajectory. Thread-safe."""
        entry = self.get_entry(trajectory)
        if entry:
            # Update last used with lock protection
            with self._lock:
                entry.last_used = datetime.now().isoformat()
                self._save_index()
            return entry.generated_code
        return None
    
    def save_code(
        self,
        trajectory: List[str],
        domains: List[str],
        is_cross_domain: bool,
        generated_code: str,
        entity_mappings: Dict[str, str],
        step_param_codes: List[str] = None,
    ) -> TrajectoryCodeEntry:
        """
        Save generated code for a trajectory.
        Thread-safe: uses lock to protect index updates.
        
        Args:
            trajectory: List of function names
            domains: List of domains
            is_cross_domain: Whether cross-domain
            generated_code: The Python code
            entity_mappings: Parameter to entity field mappings
            step_param_codes: Raw param extraction codes per step (for reuse)
            
        Returns:
            The created entry
        """
        traj_hash = compute_trajectory_hash(trajectory)
        code_dir = self._get_code_dir(domains, is_cross_domain)
        
        # Save code file (outside lock - file I/O is independent)
        code_path = code_dir / f"{traj_hash}.py"
        code_path.write_text(generated_code)
        
        # Save metadata file
        meta_path = code_dir / f"{traj_hash}.json"
        
        now = datetime.now().isoformat()
        entry = TrajectoryCodeEntry(
            trajectory_hash=traj_hash,
            trajectory=trajectory,
            domains=domains,
            is_cross_domain=is_cross_domain,
            generated_code=generated_code,
            code_path=str(code_path.relative_to(self.outputs_dir)),
            entity_mappings=entity_mappings,
            created_at=now,
            last_used=now,
            step_param_codes=step_param_codes or [],
            sample_count=0,
        )
        
        with open(meta_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        # Update index with lock protection
        with self._lock:
            self._index[traj_hash] = entry
            self._save_index()
        
        logger.info(f"Saved code for trajectory {traj_hash}: {code_path}")
        return entry
    
    def generate_and_save(
        self,
        task_definition: "TaskDefinition",
        force_regenerate: bool = False,
    ) -> TrajectoryCodeEntry:
        """
        Generate code for a task's trajectory and save it.
        Thread-safe: uses lock to protect index updates.
        
        Args:
            task_definition: The task with trajectory and context
            force_regenerate: If True, regenerate even if code exists
            
        Returns:
            The trajectory code entry
        """
        from .trajectory_executor import TrajectoryExecutor
        
        # Check if already exists
        if not force_regenerate and self.has_code(task_definition.trajectory):
            entry = self.get_entry(task_definition.trajectory)
            with self._lock:
                entry.sample_count += 1
                self._save_index()
            logger.info(f"Using existing code for trajectory {entry.trajectory_hash}")
            return entry
        
        # Execute trajectory to generate code
        executor = TrajectoryExecutor(
            outputs_dir=str(self.outputs_dir),
        )
        
        result = executor.execute(task_definition)
        
        if not result.success:
            logger.warning(f"Trajectory execution had errors: {result.errors}")
        
        # Extract entity mappings from the execution
        entity_mappings = self._extract_entity_mappings(
            task_definition.trajectory,
            task_definition.entity_context,
        )
        
        # Generate complete standalone code
        complete_code = self._generate_complete_code(
            task_definition,
            result.generated_code,
        )
        
        # Save (save_code already uses lock internally)
        entry = self.save_code(
            trajectory=task_definition.trajectory,
            domains=task_definition.domains,
            is_cross_domain=task_definition.is_cross_domain,
            generated_code=complete_code,
            entity_mappings=entity_mappings,
        )
        
        # Update sample count with lock
        with self._lock:
            entry.sample_count = 1
            self._save_index()
        
        return entry
    
    def _extract_entity_mappings(
        self,
        trajectory: List[str],
        entity_context: Dict[str, Any],
    ) -> Dict[str, str]:
        """Extract parameter to entity field mappings."""
        # For now, return all entity context keys
        # In practice, we'd analyze which params use which context fields
        mappings = {}
        for key in entity_context.keys():
            entity, field = key.split(".", 1) if "." in key else ("Unknown", key)
            mappings[field] = key
        return mappings
    
    def _generate_complete_code(
        self,
        task_definition: "TaskDefinition",
        generated_code: str,  # Unused, we generate our own dynamic code
    ) -> str:
        """Generate complete, standalone Python code with dynamic parameter extraction."""
        
        domains_str = ", ".join(f'"{d}"' for d in task_definition.domains)
        is_cross = task_definition.is_cross_domain
        
        # Generate the dynamic execution code for each step
        step_code = self._generate_dynamic_step_code(task_definition)
        
        complete_code = f'''#!/usr/bin/env python3
"""
Auto-generated trajectory execution code.

Trajectory: {" -> ".join(task_definition.trajectory)}
Domains: {task_definition.domains}
Is Cross-Domain: {is_cross}

Generated at: {datetime.now().isoformat()}

Usage:
    # With entity context from a specific sample:
    from pathlib import Path
    import json
    
    sample = json.loads(Path("sample.json").read_text())
    entity_context = sample["entity_context"]
    
    result = execute_trajectory(entity_context)
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration
OUTPUTS_DIR = Path(__file__).parent.parent.parent  # Adjust based on actual location
DOMAINS = [{domains_str}]
IS_CROSS_DOMAIN = {is_cross}
TRAJECTORY = {json.dumps(task_definition.trajectory, indent=4)}


def load_server():
    """Load the appropriate MCP server(s)."""
    import sys
    sys.path.insert(0, str(OUTPUTS_DIR.parent))
    
    from tools.mcp_tool_factory import load_server_class
    
    if IS_CROSS_DOMAIN:
        # For cross-domain, use the wrapper
        from evaluation.trajectory_executor import CrossDomainServerWrapper
        domain_name = "_".join(DOMAINS)
        return CrossDomainServerWrapper(
            domain_name=domain_name,
            outputs_dir=OUTPUTS_DIR,
            trajectory=TRAJECTORY,
        )
    else:
        # Single domain
        domain = DOMAINS[0]
        server_path = OUTPUTS_DIR / "mcp_servers" / f"{{domain}}.py"
        server_class = load_server_class(server_path, f"{{domain}}Server")
        return server_class(domain_name=domain)


def extract_param(ctx: Dict[str, Any], param_name: str, param_type: str = "string") -> Optional[Any]:
    """
    Extract a parameter value from entity context.
    
    Uses multiple matching strategies:
    1. Direct match: "student_id" -> "Student.student_id"
    2. Entity suffix match: "clinic_id" -> "HealthClinic.healthclinic_id"
    3. Field contains match: "clinic_id" -> field containing "clinic"
    """
    # Handle confirm - always True for automated execution
    if param_name == "confirm":
        return True
    
    # Strategy 1: Direct match
    for ctx_key, ctx_value in ctx.items():
        if ctx_key.endswith(f".{{param_name}}"):
            return ctx_value
    
    # Strategy 2: Entity name ends with param base
    param_base = param_name.replace("_id", "").lower()
    for ctx_key, ctx_value in ctx.items():
        entity_name = ctx_key.split(".")[0]
        field_name = ctx_key.split(".")[-1]
        
        if entity_name.lower().endswith(param_base):
            if field_name.endswith("_id") and param_name.endswith("_id"):
                return ctx_value
    
    # Strategy 3: Field contains entity prefix
    for ctx_key, ctx_value in ctx.items():
        entity_name = ctx_key.split(".")[0]
        field_name = ctx_key.split(".")[-1].lower()
        
        if param_base in entity_name.lower() and param_base in field_name:
            if field_name.endswith("_id") and param_name.endswith("_id"):
                return ctx_value
    
    return None


def execute_trajectory(entity_context: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
    """
    Execute the trajectory with the given entity context.
    
    Args:
        entity_context: Mapping of Entity.field -> value
        session_id: Optional session ID (generated if not provided)
        
    Returns:
        Dictionary with execution results for each step
    """
    session_id = session_id or str(uuid.uuid4())
    server = load_server()
    
    results = {{}}
    ctx = entity_context
    
{step_code}
    
    return results


def execute_with_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute using a sample's entity context.
    
    Args:
        sample: Sample data containing "entity_context" key
        
    Returns:
        Execution results
    """
    entity_context = sample.get("entity_context", {{}})
    if not entity_context and "instantiated_task" in sample:
        entity_context = sample["instantiated_task"].get("known_info", {{}})
    
    return execute_trajectory(entity_context)


if __name__ == "__main__":
    # Example usage with a test context
    test_context = {json.dumps(task_definition.entity_context, indent=8)}
    
    print("Executing trajectory...")
    results = execute_trajectory(test_context)
    
    print("\\nResults:")
    for step, result in results.items():
        print(f"  {{step}}: {{result}}")
'''
        return complete_code
    
    def _generate_dynamic_step_code(self, task_definition: "TaskDefinition") -> str:
        """Generate code that dynamically extracts parameters from entity_context."""
        
        # Load function info to know what parameters each function needs
        function_info = self._load_function_info(task_definition.domains)
        
        lines = []
        for step_idx, func_name in enumerate(task_definition.trajectory):
            func_info = function_info.get(func_name, {})
            params = func_info.get("parameters", {})
            required = func_info.get("required", [])
            
            lines.append(f"    # Step {step_idx + 1}: {func_name}")
            lines.append(f"    params_{step_idx} = {{}}")
            
            # Generate param extraction for each parameter
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "string")
                lines.append(f"    _val = extract_param(ctx, '{param_name}', '{param_type}')")
                lines.append(f"    if _val is not None:")
                lines.append(f"        params_{step_idx}['{param_name}'] = _val")
            
            # Execute the call
            lines.append(f"    step_{step_idx}_result = server.invoke(")
            lines.append(f"        session_id=session_id,")
            lines.append(f"        tool_name='{func_name}',")
            lines.append(f"        **params_{step_idx}")
            lines.append(f"    )")
            lines.append(f"    results[{step_idx}] = step_{step_idx}_result")
            lines.append(f"")
        
        return "\n".join(lines)
    
    def _load_function_info(self, domains: List[str]) -> Dict[str, Dict]:
        """Load function info from tool lists."""
        function_info = {}
        
        for domain in domains:
            tool_list_path = self.outputs_dir / "tool_lists" / f"{domain}.json"
            if tool_list_path.exists():
                with open(tool_list_path) as f:
                    tools = json.load(f)
                
                for tool in tools:
                    if "function" in tool:
                        func = tool["function"]
                        name = func["name"]
                        params = func.get("parameters", {})
                        function_info[name] = {
                            "parameters": params.get("properties", {}),
                            "required": params.get("required", []),
                        }
        
        return function_info
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by a number of spaces."""
        indent = " " * spaces
        lines = code.split("\n")
        return "\n".join(indent + line if line.strip() else line for line in lines)
    
    def get_code_for_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Get the execution code for a sample.
        
        Args:
            sample: Sample data containing trajectory info
            
        Returns:
            The generated code if available
        """
        trajectory = sample.get("trajectory", [])
        if not trajectory:
            # Try to find in instantiated_task
            if "instantiated_task" in sample:
                trajectory = sample.get("trajectory", [])
        
        if not trajectory:
            return None
        
        return self.get_code(trajectory)
    
    def generate_all_from_templates(
        self,
        template_dir: str = None,
        force_regenerate: bool = False,
    ) -> Dict[str, TrajectoryCodeEntry]:
        """
        Generate code for all trajectory templates.
        
        Args:
            template_dir: Directory containing template files
            force_regenerate: If True, regenerate all code
            
        Returns:
            Dictionary of trajectory_hash -> entry
        """
        from .base import TaskDefinition
        
        if template_dir is None:
            # Check both single and cross-domain templates
            template_dirs = [
                self.outputs_dir / "task_templates",
                self.outputs_dir / "cross_domain_templates",
            ]
        else:
            template_dirs = [Path(template_dir)]
        
        entries = {}
        
        for t_dir in template_dirs:
            if not t_dir.exists():
                continue
            
            for template_file in t_dir.glob("*.json"):
                logger.info(f"Processing template: {template_file.name}")
                
                try:
                    with open(template_file) as f:
                        templates = json.load(f)
                    
                    if not isinstance(templates, list):
                        templates = [templates]
                    
                    # Determine domains from filename
                    filename = template_file.stem
                    if "_" in filename and "cross_domain" in str(t_dir):
                        domains = filename.split("_")
                        is_cross_domain = True
                    else:
                        domains = [filename]
                        is_cross_domain = False
                    
                    for template in templates:
                        trajectory = template.get("trajectory", [])
                        if not trajectory:
                            continue
                        
                        # Create a minimal task definition
                        task = TaskDefinition(
                            task_id=compute_trajectory_hash(trajectory),
                            trajectory=trajectory,
                            entity_context={},  # Will be filled per-sample
                            domains=domains,
                            is_cross_domain=is_cross_domain,
                        )
                        
                        # Generate code
                        entry = self.generate_and_save(task, force_regenerate)
                        entries[entry.trajectory_hash] = entry
                        
                except Exception as e:
                    logger.error(f"Failed to process {template_file}: {e}")
        
        logger.info(f"Generated code for {len(entries)} trajectories")
        return entries
    
    def list_all(self) -> List[TrajectoryCodeEntry]:
        """List all trajectory code entries."""
        return list(self._index.values())
    
    # =========================================================================
    # Pruning Cache Methods
    # =========================================================================
    
    def _get_pruning_path(self, trajectory_hash: str, domains: List[str]) -> Path:
        """Get the path for a pruning result file."""
        if len(domains) > 1:
            domain_name = "_".join(sorted(domains))
            pruning_dir = self.code_dir / "cross_domain" / domain_name
        else:
            pruning_dir = self.code_dir / "single_domain" / (domains[0] if domains else "unknown")
        
        pruning_dir.mkdir(parents=True, exist_ok=True)
        return pruning_dir / f"{trajectory_hash}_pruning.json"
    
    def has_pruning_result(self, trajectory: List[str]) -> bool:
        """Check if a pruning result exists for a trajectory."""
        traj_hash = compute_trajectory_hash(trajectory)
        return traj_hash in self._pruning_cache
    
    def has_pruning_result_by_hash(self, trajectory_hash: str) -> bool:
        """Check if a pruning result exists for a trajectory hash."""
        return trajectory_hash in self._pruning_cache
    
    def get_pruning_result(self, trajectory: List[str]) -> Optional[PruningResultEntry]:
        """Get the pruning result for a trajectory if it exists."""
        traj_hash = compute_trajectory_hash(trajectory)
        return self._pruning_cache.get(traj_hash)
    
    def get_pruning_result_by_hash(self, trajectory_hash: str) -> Optional[PruningResultEntry]:
        """Get the pruning result for a trajectory hash if it exists."""
        return self._pruning_cache.get(trajectory_hash)
    
    def save_pruning_result(
        self,
        trajectory: List[str],
        pruned_trajectory: List[str],
        redundant_steps: List[Dict[str, Any]],
        domains: List[str],
    ) -> PruningResultEntry:
        """
        Save a pruning result for a trajectory.
        
        Thread-safe: uses lock to protect cache updates.
        
        Args:
            trajectory: Original trajectory
            pruned_trajectory: Trajectory after pruning
            redundant_steps: List of removed steps with details
            domains: Domains involved
            
        Returns:
            The created PruningResultEntry
        """
        traj_hash = compute_trajectory_hash(trajectory)
        
        # Calculate pruning ratio
        removed = len(trajectory) - len(pruned_trajectory)
        pruning_ratio = removed / len(trajectory) if trajectory else 0.0
        
        now = datetime.now().isoformat()
        entry = PruningResultEntry(
            trajectory_hash=traj_hash,
            original_trajectory=trajectory,
            pruned_trajectory=pruned_trajectory,
            redundant_steps=redundant_steps,
            pruning_ratio=pruning_ratio,
            created_at=now,
            domains=domains,
        )
        
        # Save to file
        pruning_path = self._get_pruning_path(traj_hash, domains)
        with open(pruning_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        # Update cache with lock
        with self._lock:
            self._pruning_cache[traj_hash] = entry
            self._save_pruning_index()
        
        logger.info(f"Saved pruning result for trajectory {traj_hash}: {len(redundant_steps)} steps removed")
        return entry
    
    def _load_pruning_cache(self) -> None:
        """Load pruning results from disk into cache."""
        pruning_index_path = self.code_dir / "pruning_index.json"
        
        if pruning_index_path.exists():
            try:
                with open(pruning_index_path) as f:
                    data = json.load(f)
                self._pruning_cache = {
                    k: PruningResultEntry.from_dict(v)
                    for k, v in data.items()
                }
                logger.info(f"Loaded {len(self._pruning_cache)} pruning entries from cache")
            except Exception as e:
                logger.warning(f"Failed to load pruning index: {e}")
                self._pruning_cache = {}
        else:
            self._pruning_cache = {}
    
    def _save_pruning_index(self) -> None:
        """Save pruning index to disk. Must be called with _lock held."""
        pruning_index_path = self.code_dir / "pruning_index.json"
        cache_snapshot = dict(self._pruning_cache)
        data = {k: v.to_dict() for k, v in cache_snapshot.items()}
        with open(pruning_index_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def list_all_pruning_results(self) -> List[PruningResultEntry]:
        """List all pruning result entries."""
        return list(self._pruning_cache.values())
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics about pruning results."""
        entries = self.list_all_pruning_results()
        
        if not entries:
            return {
                "total_entries": 0,
                "total_original_steps": 0,
                "total_pruned_steps": 0,
                "total_removed_steps": 0,
                "average_pruning_ratio": 0.0,
            }
        
        total_original = sum(len(e.original_trajectory) for e in entries)
        total_pruned = sum(len(e.pruned_trajectory) for e in entries)
        total_removed = sum(len(e.redundant_steps) for e in entries)
        
        return {
            "total_entries": len(entries),
            "total_original_steps": total_original,
            "total_pruned_steps": total_pruned,
            "total_removed_steps": total_removed,
            "average_pruning_ratio": total_removed / total_original if total_original else 0.0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored trajectory code."""
        entries = self.list_all()
        
        single_domain = [e for e in entries if not e.is_cross_domain]
        cross_domain = [e for e in entries if e.is_cross_domain]
        
        return {
            "total_entries": len(entries),
            "single_domain_count": len(single_domain),
            "cross_domain_count": len(cross_domain),
            "total_samples_covered": sum(e.sample_count for e in entries),
            "domains": list(set(d for e in entries for d in e.domains)),
        }
