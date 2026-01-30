"""
Trajectory Executor - Executes golden trajectories step by step with LLM code generation.

This module implements an "Iterative Rollout + Code Generation" approach:
1. Execute each step and observe the actual output
2. Use LLM to generate code for parameter extraction from context
3. Accumulate outputs in context for subsequent steps
4. Combine all step codes into a complete executable script
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy

from .base import (
    TaskDefinition,
    TrajectoryExecution,
    ExecutionStep,
    ToolCall,
    DatabaseState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Cross-Domain Server Wrapper
# =============================================================================

class CrossDomainServerWrapper:
    """
    Wrapper that manages multiple MCP servers for cross-domain tasks.
    
    Uses a trajectory-aware routing strategy:
    1. Pre-analyze the trajectory to determine domain for each segment
    2. Segments are delimited by authorize calls
    3. Each segment's domain is determined by unique functions in that segment
    """
    
    def __init__(self, domain_name: str, outputs_dir: Path, trajectory: List[str] = None):
        self.domain_name = domain_name
        self.outputs_dir = outputs_dir
        self.domains = domain_name.split("_")
        self.trajectory = trajectory or []
        
        self._servers: Dict[str, Any] = {}
        self._current_step = 0
        self._unique_functions: Dict[str, str] = {}
        self._shared_functions: set = set()
        self._build_function_map()
        
        # Build reverse index from all tool_lists for proactive routing
        self._tool_to_domain: Dict[str, str] = self._build_tool_to_domain_index()
        
        # Build abbreviation mapping (e.g., tcc -> TrainingComplianceCoach)
        self._abbrev_to_domain: Dict[str, str] = self._build_abbreviation_mapping()
        
        self._step_to_domain: Dict[int, str] = {}
        if trajectory:
            self._precompute_domains()
    
    def _parse_prefixed_name(self, name: str) -> tuple:
        """
        Parse a prefixed function name like 'ServerName.tool_name'.
        
        Returns:
            Tuple of (server_name, bare_name) or (None, name) if not prefixed.
        """
        if "." in name:
            parts = name.split(".", 1)
            return parts[0], parts[1]
        return None, name
    
    def _build_abbreviation_mapping(self) -> Dict[str, str]:
        """
        Build mapping from domain abbreviations to full names.
        
        Extracts uppercase letters from domain names to create abbreviations.
        Examples:
            TrainingComplianceCoach -> tcc
            MedicationAdministrationHub -> mah
        
        Returns:
            Dict mapping abbreviation -> full domain name
        """
        abbrev_map = {}
        for domain in self.domains:
            # Extract uppercase letters and lowercase them
            initials = ''.join(c for c in domain if c.isupper()).lower()
            if initials:
                abbrev_map[initials] = domain
        return abbrev_map
    
    def _build_function_map(self):
        """Build mapping from function names to their domains."""
        func_domains: Dict[str, List[str]] = {}
        
        for domain in self.domains:
            tool_list_path = self.outputs_dir / "tool_lists" / f"{domain}.json"
            if not tool_list_path.exists():
                logger.warning(f"Tool list not found for domain '{domain}': {tool_list_path}")
                continue
            
            try:
                with open(tool_list_path) as f:
                    tools = json.load(f)
                
                if not isinstance(tools, list):
                    logger.warning(f"Tool list for '{domain}' is not a list, skipping")
                    continue
                
                tool_count = 0
                for tool in tools:
                    if "function" in tool and "name" in tool.get("function", {}):
                        func_name = tool["function"]["name"]
                        if func_name not in func_domains:
                            func_domains[func_name] = []
                        func_domains[func_name].append(domain)
                        tool_count += 1
                
                logger.debug(f"Loaded {tool_count} functions from '{domain}'")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in tool list for '{domain}': {e}")
            except Exception as e:
                logger.error(f"Error loading tool list for '{domain}': {e}")
        
        for func_name, domains in func_domains.items():
            if len(domains) == 1:
                self._unique_functions[func_name] = domains[0]
            else:
                self._shared_functions.add(func_name)
        
        logger.debug(
            f"Function map built: {len(self._unique_functions)} unique, "
            f"{len(self._shared_functions)} shared functions"
        )
    
    def _build_tool_to_domain_index(self) -> Dict[str, str]:
        """
        Build reverse index: tool_name -> domain from ALL tool_lists.
        
        This enables proactive routing without iterating through domains.
        First domain wins if a tool exists in multiple domains.
        """
        tool_to_domain = {}
        tool_lists_dir = self.outputs_dir / "tool_lists"
        
        if not tool_lists_dir.exists():
            logger.warning(f"Tool lists directory not found: {tool_lists_dir}")
            return tool_to_domain
        
        for tool_list_file in tool_lists_dir.glob("*.json"):
            domain_name = tool_list_file.stem  # e.g., "MedicationAdministrationHub"
            try:
                with open(tool_list_file) as f:
                    tools = json.load(f)
                
                if not isinstance(tools, list):
                    continue
                
                for tool in tools:
                    if "function" in tool:
                        tool_name = tool["function"].get("name")
                        if tool_name and tool_name not in tool_to_domain:
                            tool_to_domain[tool_name] = domain_name
            except Exception as e:
                logger.warning(f"Failed to load tool list {tool_list_file}: {e}")
        
        logger.debug(f"Built tool-to-domain index with {len(tool_to_domain)} entries")
        return tool_to_domain
    
    def _precompute_domains(self):
        """Pre-compute which domain each step should use."""
        segments = []
        current_segment = []
        
        for i, func in enumerate(self.trajectory):
            # Parse prefixed name to get bare function name
            server_prefix, bare_func = self._parse_prefixed_name(func)
            if bare_func.startswith("authorize") and current_segment:
                segments.append(current_segment)
                current_segment = [(i, func, server_prefix, bare_func)]
            else:
                current_segment.append((i, func, server_prefix, bare_func))
        
        if current_segment:
            segments.append(current_segment)
        
        for segment in segments:
            segment_domain = None
            
            for step_idx, func_name, server_prefix, bare_name in segment:
                # If we have a server prefix that matches a known domain, use it directly
                if server_prefix and server_prefix in self.domains:
                    segment_domain = server_prefix
                    break
                # Otherwise try unique functions lookup with bare name
                if bare_name in self._unique_functions:
                    segment_domain = self._unique_functions[bare_name]
                    break
            
            if segment_domain is None:
                used_domains = set(self._step_to_domain.values())
                for domain in self.domains:
                    if domain not in used_domains:
                        segment_domain = domain
                        break
                if segment_domain is None:
                    segment_domain = self.domains[0]
            
            for step_idx, func_name, server_prefix, bare_name in segment:
                # If function has a prefix, use it for direct routing
                if server_prefix and server_prefix in self.domains:
                    self._step_to_domain[step_idx] = server_prefix
                else:
                    self._step_to_domain[step_idx] = segment_domain
        
        logger.debug(f"Pre-computed domains: {self._step_to_domain}")
    
    def _get_server(self, domain: str):
        """Get or create server for a domain."""
        if domain not in self._servers:
            from tools.mcp_tool_factory import load_server_class
            server_path = self.outputs_dir / "mcp_servers" / f"{domain}.py"
            server_class = load_server_class(server_path, f"{domain}Server")
            self._servers[domain] = server_class(domain_name=self.domain_name)
        return self._servers[domain]
    
    def _get_domain_for_step(self, step_idx: int, function_name: str) -> str:
        if step_idx in self._step_to_domain:
            return self._step_to_domain[step_idx]
        if function_name in self._unique_functions:
            return self._unique_functions[function_name]
        # Use reverse index for proactive routing
        if function_name in self._tool_to_domain:
            return self._tool_to_domain[function_name]
        return self.domains[0]
    
    def invoke(self, session_id: str, tool_name: str, **kwargs):
        """Route the call to the appropriate server with fallback."""
        # Parse prefixed name (e.g., "ServerName.tool_name" -> ("ServerName", "tool_name"))
        server_prefix, bare_name = self._parse_prefixed_name(tool_name)
        
        # Determine domain based on prefix
        if server_prefix:
            # Check full domain name first
            if server_prefix in self.domains:
                domain = server_prefix
            # Then check abbreviation mapping (tcc -> TrainingComplianceCoach)
            elif server_prefix in self._abbrev_to_domain:
                domain = self._abbrev_to_domain[server_prefix]
            else:
                domain = self._get_domain_for_step(self._current_step, bare_name)
        else:
            domain = self._get_domain_for_step(self._current_step, bare_name)
        
        server = self._get_server(domain)
        
        logger.debug(f"Step {self._current_step}: {tool_name} -> {domain} (bare: {bare_name})")
        
        # Use bare name when invoking the underlying server
        result = server.invoke(session_id=session_id, tool_name=bare_name, **kwargs)
        
        # Check for Unknown tool error and try fallback using reverse index
        if isinstance(result, dict) and "error" in result:
            error_msg = result.get("error", "")
            if "Unknown tool" in error_msg:
                logger.warning(
                    f"Unknown tool '{bare_name}' in domain '{domain}', checking reverse index..."
                )
                
                # Use reverse index for direct lookup instead of iterating
                if bare_name in self._tool_to_domain:
                    correct_domain = self._tool_to_domain[bare_name]
                    if correct_domain != domain:
                        logger.info(f"  Found '{bare_name}' in domain '{correct_domain}' via index, retrying...")
                        other_server = self._get_server(correct_domain)
                        fallback_result = other_server.invoke(
                            session_id=session_id, 
                            tool_name=bare_name, 
                            **kwargs
                        )
                        
                        # Check if fallback succeeded
                        if not (isinstance(fallback_result, dict) and "Unknown tool" in str(fallback_result.get("error", ""))):
                            logger.info(f"  Fallback to '{correct_domain}' succeeded for '{bare_name}'")
                            self._current_step += 1
                            return fallback_result
                
                # Log detailed info if fallback failed or tool not in index
                logger.error(
                    f"Tool '{bare_name}' not found in any domain. "
                    f"Current domains: {self.domains}, Index has: {bare_name in self._tool_to_domain}"
                )
        
        self._current_step += 1
        
        return result
    
    def _get_session(self, session_id: str):
        if not self._servers:
            self._get_server(self.domains[0])
        for server in self._servers.values():
            return server._get_session(session_id)
        return None


# =============================================================================
# LLM Prompts
# =============================================================================

PARAM_EXTRACTION_PROMPT = """You are generating Python code to extract parameters for a function call.

## Task Being Performed
{instruction}

## Function to Call
Name: {function_name}
Required Parameters: {required_params}
All Parameters: {all_params}

## Previous Steps Executed
{execution_history}

## Available Data Sources in `ctx`

### 1. value_domain_samples (Use this as PRIMARY RESOURCE, as long as the information is available in here, use it directly without considering the other sources)
Pre-computed parameter values in format "Server.Function.param" -> value.
Access: `ctx["value_domain_samples"]["ServerName.function_name.param_name"]`

### 2. entity_instances (Should never be used in the parameters)
Entity and relationship records as nested dicts.

### 3. step_N_result (0-indexed, user only when the information is not available in the value_domain_samples)
Results from previous tool calls. Step 1's result is ctx["step_0_result"], Step 2's is ctx["step_1_result"], etc.

## Current Context

Current time anchor: 2025-01-23T15:00:00-05:00 (EST)

```json
{context_json}
```

## Your Task: Intelligent Parameter Selection
For each required parameter, you must REASON about which data source is most appropriate:

1. **Static parameters** (IDs, names, fixed values): Usually found in `value_domain_samples` (PRIMARY RESOURCE)
2. **Dynamic parameters** (when static parameters do not fit): Must come from `step_N_result`

**Think about each parameter:**
- Is this a static parameter that should match what was specified in the `value_domain_samples`?
    - If multiple values in `value_domain_samples` match the name of the tool call parameter, decide a proper one according to the task instruction
- If not, does this parameter need a value that was just computed/returned by a previous step?
    - If multiple values in `step_N_result` match the name of the tool call parameter, decide a proper one according to the task instruction

## CRITICAL RULES
- **CURRENT TIME**: For the samples that do not specify the time, use the current time as the time anchor. e.g., some tool calls require a parameter `on_date`, which requires the current date.
- **ALWAYS FROM THE CONTEXT**: All values MUST come from `ctx`. Never invent values or directly copy from the task instruction.
- **NO PLACEHOLDERS**: Never use "example", "test", "sample", or placeholder strings.
- **If not found**: Use `None` - do NOT fabricate a value.
- **confirm parameter**: Always set to `True`.

## Code Requirements
- The code should be executable and assign to a variable named `params`. For tools do not require any parameter, set `params` to an empty dictionary.
- Handle potential missing keys gracefully with `.get()` where appropriate
- **DO NOT use import statements** - `re`, `json` modules are pre-provided
- Keep the code simple and straightforward

## Example
```python
vds = ctx.get("value_domain_samples", {{}})

params = {{
    # entity_id: Use value_domain_samples since it can be found in the value_domain_samples
    "entity_id": vds.get("ServerName.function_name.entity_id"),
    
    # created_relationship_id: This comes from the previous step's query result, not static data
    # e.g., tools with `create`, `generate` prefix genereates a relationship record with random uuid.
    # Such ID cannot be collected before the execution of the tool, thus it is a dynamic parameter that should be collected from the previous step's result.
    "created_relationship_id": ctx.get("step_0_result", {{}}).get("due_administrations", [{{}}])[0].get("created_relationship_id"),

    # entity_instances is just extra information that helps you to understand the task, do not use as parameters.
    
    "confirm": True,
}}
```

Output ONLY the Python code (no markdown, no explanations):
"""


COMPLETE_SCRIPT_TEMPLATE = '''#!/usr/bin/env python
"""
Auto-generated trajectory execution script.
Task ID: {task_id}
Generated at: {timestamp}
Trajectory: {trajectory}

This code is reusable across samples with the same trajectory.
Pass entity_context as parameter to execute_trajectory().
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.mcp_tool_factory import load_server_class


def execute_trajectory(
    entity_context: Dict[str, Any],
    session_id: str = None,
    verbose: bool = False,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Execute the trajectory with given entity context.
    
    Args:
        entity_context: Mapping of Entity.field -> value
        session_id: Optional session ID (generated if not provided)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (success, results_list)
    """
    session_id = session_id or str(uuid.uuid4())
    
    # Initial context with entity values
    ctx = entity_context.copy()
    
    # Server setup
{server_setup}
    
    # Results storage
    results = []
    success = True
    
{step_code}
    
    # Summary
    if verbose:
        print(f"\\n=== Execution Summary ===")
        print(f"Total steps: {total_steps}")
        passed = sum(1 for r in results if r.get("success"))
        print(f"Passed: {{passed}}/{total_steps}")
        print(f"Success: {{success}}")
    
    return success, results


if __name__ == "__main__":
    # Example usage - replace with actual entity context for testing
    print("This script should be imported and called with execute_trajectory(entity_context)")
    print("Example: success, results = execute_trajectory({{'Entity.field': 'value'}})")
    sys.exit(0)
'''


STEP_CODE_TEMPLATE = '''
    # Step {step_num}: {function_name}
    print(f"Step {step_num}: {function_name}")
    try:
{param_code}
        result = server.invoke(session_id=session_id, tool_name="{function_name}", **params)
        ctx["step_{step_idx}_result"] = result
        results.append({{"step": {step_num}, "function": "{function_name}", "success": True, "result": result}})
        print(f"  ✓ Success")
    except Exception as e:
        results.append({{"step": {step_num}, "function": "{function_name}", "success": False, "error": str(e)}})
        print(f"  ✗ Error: {{e}}")
        success = False
'''


@dataclass
class FunctionInfo:
    """Information about a function from the tool list."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)


class TrajectoryExecutor:
    """
    Executes golden trajectories step by step with LLM-assisted code generation.
    
    This executor uses a pure LLM-based approach:
    1. Maintains a context dict with entity values and step outputs
    2. For each step, LLM generates code to extract params from context
    3. Executes the step and adds result to context
    4. After all steps, combines code into a complete script
    """
    
    def __init__(
        self,
        outputs_dir: str = "./outputs",
        llm_model: str = None,
    ):
        self.outputs_dir = Path(outputs_dir)
        self._llm_model = llm_model
        self._llm_client = None
    
    @property
    def llm_client(self):
        if self._llm_client is None:
            from agent_skiller.core.llm_client import get_client
            self._llm_client = get_client()
        return self._llm_client
    
    @property
    def llm_model(self):
        """Get the LLM model to use, defaulting to settings.llm.coding_model."""
        if self._llm_model:
            return self._llm_model
        try:
            from agent_skiller.config import get_settings
            return get_settings().llm.coding_model
        except Exception:
            return None  # Will use default from LLMClient
    
    def _parse_prefixed_name(self, name: str) -> tuple:
        """
        Parse a prefixed function name like 'ServerName.tool_name'.
        
        Returns:
            Tuple of (server_name, bare_name) or (None, name) if not prefixed.
        """
        if "." in name:
            parts = name.split(".", 1)
            return parts[0], parts[1]
        return None, name
    
    def execute(
        self,
        task: TaskDefinition,
        session_id: Optional[str] = None,
        verbose: bool = True,
    ) -> TrajectoryExecution:
        """Execute a complete trajectory."""
        session_id = session_id or str(uuid.uuid4())
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Executing trajectory: {task.task_id}")
            print(f"   Steps: {' → '.join(task.trajectory[:5])}{'...' if len(task.trajectory) > 5 else ''}")
            print(f"{'='*60}")
        
        logger.info(f"Executing trajectory for task {task.task_id}")
        logger.info(f"Trajectory: {' -> '.join(task.trajectory)}")
        
        domain_name = self._get_domain_name(task)
        function_info = self._load_function_info(domain_name)
        
        try:
            server = self._get_server(domain_name, trajectory=task.trajectory)
        except Exception as e:
            logger.error(f"Failed to load server for {domain_name}: {e}")
            if verbose:
                print(f"❌ Failed to load server: {e}")
            return TrajectoryExecution(
                task_id=task.task_id,
                steps=[],
                session_id=session_id,
                success=False,
                errors=[f"Failed to load server: {e}"],
            )
        
        # Use entity context directly - it now has structured format with
        # entity_instances and value_domain_samples as separate keys
        ctx = dict(task.entity_context)  # Copy to avoid modifying original

        print(ctx)
        
        # Capture initial state
        initial_state = self._capture_database_state(server, session_id)
        
        steps: List[ExecutionStep] = []
        execution_history: List[Dict[str, Any]] = []
        step_codes: List[str] = []
        step_param_codes: List[str] = []  # Raw param codes for reuse
        errors: List[str] = []
        
        for step_idx, function_name in enumerate(task.trajectory):
            step_num = step_idx + 1
            total_steps = len(task.trajectory)
            
            if verbose:
                print(f"\n📍 Step {step_num}/{total_steps}: {function_name}")
            
            logger.info(f"Step {step_num}/{total_steps}: {function_name}")
            
            try:
                func_info = function_info.get(function_name)
                if not func_info:
                    raise ValueError(f"Unknown function: {function_name}")
                
                if verbose:
                    print(f"   Required params: {func_info.required}")
                
                # Generate parameter extraction code using LLM with retry
                if verbose:
                    print(f"   🤖 Generating param code...", end=" ", flush=True)
                
                param_code, params = self._generate_and_execute_param_code(
                    function_name=function_name,
                    func_info=func_info,
                    ctx=ctx,
                    execution_history=execution_history,
                    instruction=task.instruction,
                    verbose=verbose,
                )
                
                if verbose:
                    params_preview = {k: str(v)[:3000] + "..." if len(str(v)) > 3000 else v for k, v in params.items()}
                    print(f"✓")
                    print(f"   Params: {params_preview}")
                
                # Execute the function with retry on param errors
                max_invoke_retries = 3
                result = None
                is_error = False
                
                for invoke_attempt in range(max_invoke_retries):
                    if verbose:
                        if invoke_attempt == 0:
                            print(f"   📞 Invoking {function_name}...", end=" ", flush=True)
                        else:
                            print(f"   🔄 Retry {invoke_attempt + 1}/{max_invoke_retries}...", end=" ", flush=True)
                    
                    try:
                        result = server.invoke(
                            session_id=session_id,
                            tool_name=function_name,
                            **params,
                        )
                        
                        # Check result for errors
                        is_error = isinstance(result, dict) and (result.get("error") or result.get("success") == False)
                        
                        if not is_error:
                            # Success!
                            break
                        
                        # Check if it's a param-related error worth retrying
                        error_msg = str(result.get("error") or result.get("message", ""))
                        is_param_error = any(kw in error_msg.lower() for kw in 
                            ["required", "missing", "invalid", "parameter", "argument", "none", "null"])
                        
                        if is_param_error and invoke_attempt < max_invoke_retries - 1:
                            if verbose:
                                print(f"⚠️ Param error, regenerating...")
                            # Regenerate params with error context
                            param_code, params = self._generate_and_execute_param_code(
                                function_name=function_name,
                                func_info=func_info,
                                ctx=ctx,
                                execution_history=execution_history,
                                instruction=task.instruction,
                                verbose=False,
                            )
                        else:
                            # Not a param error or last attempt, stop retrying
                            break
                            
                    except Exception as invoke_exc:
                        if verbose:
                            print(f"⚠️ {invoke_exc}")
                        if invoke_attempt >= max_invoke_retries - 1:
                            raise invoke_exc
                
                if verbose:
                    if is_error:
                        error_msg = result.get("error") or result.get("message", "Unknown error") if result else "Unknown"
                        print(f"⚠️ {str(error_msg)[:3000]}")
                    else:
                        result_preview = str(result)[:3000] + "..." if len(str(result)) > 3000 else str(result)
                        print(f"✓")
                        print(f"   Result: {result_preview}")
                
                # Update context with result
                ctx[f"step_{step_idx}_result"] = result
                
                # Record step
                tool_call = ToolCall(
                    name=function_name,
                    arguments=params,
                    result=result,
                )
                
                step = ExecutionStep(
                    step_index=step_idx,
                    tool_call=tool_call,
                    code=param_code,
                    success=not is_error,
                )
                steps.append(step)
                
                execution_history.append({
                    "step_index": step_idx,
                    "function": function_name,
                    "params": params,
                    "result": result,
                })
                
                step_codes.append(
                    STEP_CODE_TEMPLATE.format(
                        step_num=step_num,
                        step_idx=step_idx,
                        function_name=function_name,
                        param_code=self._indent_code(param_code, 8),
                    )
                )
                step_param_codes.append(param_code)
                
                logger.debug(f"Step {step_num} result: {json.dumps(result, default=str)[:200]}...")
                
            except Exception as e:
                logger.error(f"Step {step_num} failed: {e}")
                if verbose:
                    print(f"❌ Exception: {e}")
                errors.append(f"Step {step_num} ({function_name}): {e}")
                
                step = ExecutionStep(
                    step_index=step_idx,
                    tool_call=ToolCall(name=function_name, arguments={}),
                    success=False,
                    error=str(e),
                )
                steps.append(step)
        
        # Capture final state
        final_state = self._capture_database_state(server, session_id)
        
        # Generate complete script
        generated_code = self._generate_complete_script(
            task=task,
            session_id=session_id,
            ctx=ctx,
            step_codes=step_codes,
            domain_name=domain_name,
        )
        
        # Summary
        success_count = sum(1 for s in steps if s.success)
        total_count = len(steps)
        all_success = len(errors) == 0 and success_count == total_count
        
        if verbose:
            print(f"\n{'='*60}")
            if all_success:
                print(f"✅ Trajectory completed successfully! ({success_count}/{total_count} steps)")
            else:
                print(f"⚠️ Trajectory completed with issues ({success_count}/{total_count} steps passed)")
                if errors:
                    print(f"   Errors: {len(errors)}")
                    for err in errors[:3]:
                        print(f"   - {err[:60]}")
            print(f"{'='*60}\n")
        
        return TrajectoryExecution(
            task_id=task.task_id,
            steps=steps,
            final_state=final_state,
            initial_state=initial_state,
            session_id=session_id,
            generated_code=generated_code,
            step_param_codes=step_param_codes,
            success=all_success,
            errors=errors,
        )
    
    def execute_with_codes(
        self,
        task: TaskDefinition,
        step_param_codes: List[str],
        session_id: Optional[str] = None,
        verbose: bool = False,
    ) -> TrajectoryExecution:
        """
        Execute trajectory using pre-generated param codes (no LLM calls).
        
        This is used to reuse previously generated param extraction codes
        across multiple task instances with the same trajectory.
        
        Args:
            task: The task definition with trajectory and entity_context
            step_param_codes: Pre-generated param extraction codes for each step
            session_id: Optional session ID
            verbose: Whether to print progress
            
        Returns:
            TrajectoryExecution result
        """
        session_id = session_id or str(uuid.uuid4())
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔄 Replaying trajectory: {task.task_id}")
            print(f"   Using pre-generated codes ({len(step_param_codes)} steps)")
            print(f"{'='*60}")
        
        if len(step_param_codes) != len(task.trajectory):
            logger.warning(
                f"Code count mismatch: {len(step_param_codes)} codes for "
                f"{len(task.trajectory)} trajectory steps"
            )
        
        domain_name = self._get_domain_name(task)
        function_info = self._load_function_info(domain_name)
        
        try:
            server = self._get_server(domain_name, trajectory=task.trajectory)
        except Exception as e:
            logger.error(f"Failed to load server for {domain_name}: {e}")
            return TrajectoryExecution(
                task_id=task.task_id,
                steps=[],
                session_id=session_id,
                success=False,
                errors=[f"Server load failed: {e}"],
            )
        
        # Use entity context directly - it now has structured format with
        # entity_instances and value_domain_samples as separate keys
        ctx = dict(task.entity_context)  # Copy to avoid modifying original

        # Capture initial state before any execution
        initial_state = self._capture_database_state(server, session_id)
        
        steps: List[ExecutionStep] = []
        errors: List[str] = []
        
        for step_idx, function_name in enumerate(task.trajectory):
            step_num = step_idx + 1
            
            if verbose:
                print(f"\n📍 Step {step_num}/{len(task.trajectory)}: {function_name}")
            
            try:
                # Get pre-generated param code
                if step_idx < len(step_param_codes):
                    param_code = step_param_codes[step_idx]
                else:
                    raise ValueError(f"No param code for step {step_idx}")
                
                # Execute the param code
                params = self._execute_param_code(param_code, ctx)
                
                if not isinstance(params, dict):
                    raise ValueError(f"Param code returned {type(params)}, expected dict")
                
                if verbose:
                    params_preview = {k: str(v)[:3000] + "..." if len(str(v)) > 3000 else v for k, v in params.items()}
                    print(f"   Params: {params_preview}")
                
                # Execute the function
                result = server.invoke(
                    session_id=session_id,
                    tool_name=function_name,
                    **params,
                )
                
                # Check result for errors
                is_error = isinstance(result, dict) and (result.get("error") or result.get("success") == False)
                
                if verbose:
                    if is_error:
                        error_msg = result.get("error") or result.get("message", "Unknown error")
                        print(f"   ⚠️ {error_msg[:50]}")
                    else:
                        print(f"   ✓ Success")
                
                # Update context with result
                ctx[f"step_{step_idx}_result"] = result
                
                # Record step
                tool_call = ToolCall(
                    name=function_name,
                    arguments=params,
                    result=result,
                )
                
                steps.append(ExecutionStep(
                    step_index=step_idx,
                    tool_call=tool_call,
                    code=param_code,
                    success=not is_error,
                ))
                
            except Exception as e:
                logger.error(f"Step {step_num} failed: {e}")
                if verbose:
                    print(f"   ❌ Error: {e}")
                errors.append(f"Step {step_num} ({function_name}): {e}")
                
                steps.append(ExecutionStep(
                    step_index=step_idx,
                    tool_call=ToolCall(name=function_name, arguments={}),
                    success=False,
                    error=str(e),
                ))
        
        success_count = sum(1 for s in steps if s.success)
        all_success = len(errors) == 0 and success_count == len(steps)
        
        # Capture final database state for environment evaluation
        final_state = self._capture_database_state(server, session_id)
        
        if verbose:
            print(f"\n{'='*60}")
            if all_success:
                print(f"✅ Replay completed! ({success_count}/{len(steps)} steps)")
            else:
                print(f"⚠️ Replay had issues ({success_count}/{len(steps)} passed)")
            print(f"{'='*60}\n")
        
        return TrajectoryExecution(
            task_id=task.task_id,
            steps=steps,
            final_state=final_state,
            initial_state=initial_state,
            session_id=session_id,
            step_param_codes=step_param_codes,
            success=all_success,
            errors=errors,
        )
    
    def _generate_and_execute_param_code(
        self,
        function_name: str,
        func_info: FunctionInfo,
        ctx: Dict,
        execution_history: List[Dict],
        instruction: str = "",
        max_retries: int = 3,
        verbose: bool = False,
    ) -> tuple:
        """
        Generate parameter extraction code and execute it with retry on failure.
        
        Args:
            instruction: The task instruction to help LLM understand context
        
        Returns:
            Tuple of (code, params)
        """
        last_code = ""
        last_error = None
        
        for attempt in range(max_retries):
            if attempt == 0:
                # First attempt: generate fresh code
                code = self._generate_param_code(
                    function_name=function_name,
                    func_info=func_info,
                    ctx=ctx,
                    execution_history=execution_history,
                    instruction=instruction,
                )
            else:
                # Retry: ask LLM to fix the code
                if verbose:
                    print(f"\n   🔧 Retry {attempt + 1}/{max_retries}: Fixing code (error: {str(last_error)[:50]})")
                code = self._fix_param_code(
                    original_code=last_code,
                    error=str(last_error),
                    function_name=function_name,
                    func_info=func_info,
                )
            
            last_code = code
            
            # Try to execute
            params = self._execute_param_code(code, ctx)
            
            # Validate params - more lenient: allow None values, let MCP server decide
            if isinstance(params, dict):
                # Check which required params are missing or None
                missing = [p for p in func_info.required if p not in params or params[p] is None]
                if missing:
                    # Fill missing params with None - let MCP server validate
                    for p in missing:
                        if p not in params:
                            params[p] = None
                    logger.debug(f"Attempt {attempt + 1}: Filled missing params with None: {missing}")
                
                # Return the params - MCP server will validate
                return code, params
            elif params is None:
                last_error = "Generated code returned None instead of params dict"
                logger.warning(f"Attempt {attempt + 1}: {last_error}")
            else:
                last_error = f"Invalid params type: {type(params).__name__}, expected dict"
                logger.warning(f"Attempt {attempt + 1}: {last_error}")
                if verbose and attempt < max_retries - 1:
                    print(f"⚠️ {last_error}")
        
        # All retries failed, return last attempt with empty params filled with None
        logger.warning(f"Failed to generate valid param code after {max_retries} attempts, using None values")
        if verbose:
            print(f"⚠️ Using None for missing params after {max_retries} attempts")
        
        # Build params with None for all required
        fallback_params = params if isinstance(params, dict) else {}
        for p in func_info.required:
            if p not in fallback_params:
                fallback_params[p] = None
        
        return last_code, fallback_params
    
    def _fix_param_code(
        self,
        original_code: str,
        error: str,
        function_name: str,
        func_info: FunctionInfo,
    ) -> str:
        """Ask LLM to fix the parameter extraction code using SEARCH/REPLACE."""
        import tempfile
        from agent_skiller.core import WorkflowBlockEditor
        
        try:
            # Write code to temp file for BlockEditor
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(original_code)
                temp_path = Path(f.name)
            
            try:
                editor = WorkflowBlockEditor()
                error_context = f"Function: {function_name}\nRequired: {func_info.required}\nError: {error}"
                success = editor.fix_code(
                    file_path=temp_path,
                    error=error_context,
                    llm_client=self.llm_client,
                    language="python",
                    max_retries=2,
                )
                if success:
                    return temp_path.read_text()
                return original_code
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to fix code: {e}")
            return original_code
    
    def _generate_param_code(
        self,
        function_name: str,
        func_info: FunctionInfo,
        ctx: Dict,
        execution_history: List[Dict],
        instruction: str = "",
    ) -> str:
        """Use LLM to generate parameter extraction code.
        
        Args:
            instruction: The task instruction to help LLM understand context
        """
        
        # Format execution history - CLEARLY specify the ctx key for each step's result
        history_str = ""
        for hist in execution_history:
            step_idx = hist['step_index']
            step_num = step_idx + 1
            result_key = f"step_{step_idx}_result"  # This is the ctx key!
            
            history_str += f"\nStep {step_num}: {hist['function']}\n"
            history_str += f"  >>> Result stored in: ctx[\"{result_key}\"]\n"
            history_str += f"  Result keys: {list(hist['result'].keys()) if isinstance(hist['result'], dict) else type(hist['result']).__name__}\n"
            
            # Include a sample of the result structure
            result_sample = json.dumps(hist['result'], default=str)
            if len(result_sample) > 500:
                result_sample = result_sample[:500] + "..."
            history_str += f"  Result: {result_sample}\n"
        
        if not history_str:
            history_str = "(No previous steps)"
        
        # Prepare context for LLM (limit size)
        ctx_for_prompt = {}
        for k, v in ctx.items():
            if k.startswith("step_"):
                # Summarize step results
                if isinstance(v, dict):
                    ctx_for_prompt[k] = {kk: type(vv).__name__ if isinstance(vv, (list, dict)) else vv 
                                         for kk, vv in list(v.items())[:5]}
                else:
                    ctx_for_prompt[k] = str(v)[:100]
            else:
                ctx_for_prompt[k] = v
        
        # Format instruction for prompt
        instruction_text = instruction if instruction else "(No instruction provided)"
        
        prompt = PARAM_EXTRACTION_PROMPT.format(
            instruction=instruction_text,
            context_json=json.dumps(ctx_for_prompt, indent=2, default=str),
            function_name=function_name,
            required_params=json.dumps(func_info.required),
            all_params=json.dumps(func_info.parameters, indent=2),
            execution_history=history_str,
        )
        
        try:
            response = self.llm_client.chat(query=prompt, model=self.llm_model)
            code = response.content.strip()
            
            # Clean up markdown
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            return code
            
        except Exception as e:
            logger.warning(f"LLM code generation failed: {e}, using fallback")
            return self._generate_fallback_code(function_name, func_info, ctx)
    
    def _generate_fallback_code(
        self,
        function_name: str,
        func_info: FunctionInfo,
        ctx: Dict,
    ) -> str:
        """Generate fallback parameter code without LLM.
        
        Uses the new structured context format:
        - ctx["value_domain_samples"]: Server.Function.param -> value
        - ctx["entity_instances"]: Entity -> {field: value}
        """
        lines = [
            "vds = ctx.get('value_domain_samples', {})",
            "entities = ctx.get('entity_instances', {})",
            "params = {}",
        ]
        
        # Extract server name from function_name (e.g., "ServerName.function" -> "ServerName")
        if "." in function_name:
            server_name, func_name = function_name.rsplit(".", 1)
        else:
            server_name, func_name = "", function_name
        
        for param_name in func_info.required:
            if param_name == "confirm":
                lines.append(f'params["confirm"] = True')
                continue
            
            # Build the value_domain_samples key
            vds_key = f"{server_name}.{func_name}.{param_name}" if server_name else f"{func_name}.{param_name}"
            
            # First: try value_domain_samples (exact match)
            lines.append(f'# Try value_domain_samples first')
            lines.append(f'params["{param_name}"] = vds.get("{vds_key}")')
            
            # Second: fallback to entity_instances
            lines.append(f'if params["{param_name}"] is None:')
            
            # Try to find in entity_instances
            found_entity = None
            entity_instances = ctx.get("entity_instances", {})
            for entity_name, entity_data in entity_instances.items():
                if isinstance(entity_data, dict):
                    if param_name in entity_data:
                        found_entity = entity_name
                        break
                    # Try fuzzy match for _id fields
                    if param_name.endswith("_id"):
                        for field in entity_data.keys():
                            if field.endswith("_id") and param_name.replace("_id", "") in field.lower():
                                found_entity = entity_name
                                lines.append(f'    params["{param_name}"] = entities.get("{entity_name}", {{}}).get("{field}")')
                                break
                        if found_entity:
                            break
            
            if found_entity and param_name in entity_instances.get(found_entity, {}):
                lines.append(f'    params["{param_name}"] = entities.get("{found_entity}", {{}}).get("{param_name}")')
            elif not found_entity:
                lines.append(f'    pass  # Could not find {param_name} in entity_instances')
        
        return "\n".join(lines)
    
    def _execute_param_code(self, code: str, ctx: Dict) -> Dict:
        """Execute generated code to extract parameters."""
        import re as re_module
        
        local_vars = {"ctx": ctx, "json": json, "re": re_module}
        
        # Safe builtins for param extraction code
        safe_builtins = {
            # Types
            "str": str, "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            # Constants
            "True": True, "False": False, "None": None,
            # Common functions
            "len": len, "isinstance": isinstance, "type": type,
            "range": range, "enumerate": enumerate, "zip": zip,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "any": any, "all": all, "next": next, "iter": iter,
            "filter": filter, "map": map,
            # String/collection methods often needed
            "sorted": sorted, "reversed": reversed,
            "getattr": getattr, "hasattr": hasattr, "setattr": setattr,
            # For print debugging if needed
            "print": print,
        }
        
        # Remove import statements from code (we pre-provide modules)
        clean_code = "\n".join(
            line for line in code.split("\n") 
            if not line.strip().startswith("import ") and not line.strip().startswith("from ")
        )
        
        try:
            exec(clean_code, {"__builtins__": safe_builtins}, local_vars)
            
            if "params" in local_vars:
                return local_vars["params"]
            else:
                print(clean_code)
                logger.warning("Generated code did not produce 'params' variable")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to execute param code: {e}\nCode:\n{clean_code}")
            return {}
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code block."""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))
    
    def _generate_complete_script(
        self,
        task: TaskDefinition,
        session_id: str,
        ctx: Dict,
        step_codes: List[str],
        domain_name: str,
    ) -> str:
        """Generate the complete executable script."""
        
        # Generate server setup code
        if "_" in domain_name:
            # Cross-domain
            domains = domain_name.split("_")
            server_setup = f'''    from evaluation.trajectory_executor import CrossDomainServerWrapper
    server = CrossDomainServerWrapper(
        domain_name="{domain_name}",
        outputs_dir=Path("outputs_cursor"),
        trajectory={task.trajectory},
    )'''
        else:
            server_setup = f'''    server_path = Path("outputs_cursor/mcp_servers/{domain_name}.py")
    server_class = load_server_class(server_path, "{domain_name}Server")
    server = server_class(domain_name="{domain_name}")'''
        
        return COMPLETE_SCRIPT_TEMPLATE.format(
            task_id=task.task_id,
            timestamp=datetime.now().isoformat(),
            trajectory=" -> ".join(task.trajectory),
            server_setup=server_setup,
            step_code="".join(step_codes),
            total_steps=len(task.trajectory),
        )
    
    def _get_domain_name(self, task: TaskDefinition) -> str:
        if task.is_cross_domain and task.domains:
            return "_".join(task.domains)
        elif task.domains:
            return task.domains[0]
        else:
            raise ValueError("Task has no domains specified")
    
    def _get_server(self, domain_name: str, trajectory: List[str] = None):
        from tools.mcp_tool_factory import load_server_class
        
        if "_" not in domain_name:
            server_path = self.outputs_dir / "mcp_servers" / f"{domain_name}.py"
            server_class = load_server_class(server_path, f"{domain_name}Server")
            return server_class(domain_name=domain_name)
        
        return CrossDomainServerWrapper(
            domain_name=domain_name,
            outputs_dir=self.outputs_dir,
            trajectory=trajectory,
        )
    
    def _load_function_info(self, domain_name: str) -> Dict[str, FunctionInfo]:
        function_info = {}
        
        domains = domain_name.split("_") if "_" in domain_name else [domain_name]
        
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
                        info = FunctionInfo(
                            name=name,
                            description=func.get("description", ""),
                            parameters=params.get("properties", {}),
                            required=params.get("required", []),
                        )
                        # Index by bare name
                        function_info[name] = info
                        # Also index by prefixed name for cross-domain trajectories
                        function_info[f"{domain}.{name}"] = info
        
        return function_info
    
    def _capture_database_state(self, server, session_id: str) -> DatabaseState:
        """Capture database state, handling both single and cross-domain servers."""
        try:
            # Check if this is a CrossDomainServerWrapper
            if hasattr(server, '_servers') and hasattr(server, 'domains'):
                # Cross-domain case: ensure all domain servers are initialized and capture from each
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
                                        # Merge entities, avoiding duplicates
                                        if key not in entities:
                                            entities[key] = deepcopy(value)
                                    else:
                                        # Merge relationships, avoiding duplicates
                                        if key not in relationships:
                                            relationships[key] = deepcopy(value)
                    except Exception as e:
                        logger.warning(f"Failed to capture state from {domain}: {e}")
                
                return DatabaseState(entities=entities, relationships=relationships)
            else:
                # Single domain case
                session = server._get_session(session_id)
                db = session.databases
                
                entities = {}
                relationships = {}
                
                for key, value in db.items():
                    if isinstance(value, list):
                        if value and isinstance(value[0], dict):
                            if any(k.endswith("_id") and k.replace("_id", "").lower() == key.lower() 
                                   for k in value[0].keys()):
                                entities[key] = deepcopy(value)
                            else:
                                relationships[key] = deepcopy(value)
                
                return DatabaseState(entities=entities, relationships=relationships)
            
        except Exception as e:
            logger.warning(f"Failed to capture database state: {e}")
            return DatabaseState()
    
    def execute_batch(
        self,
        tasks: List[TaskDefinition],
        max_workers: int = 4,
    ) -> List[TrajectoryExecution]:
        """Execute multiple trajectories in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.execute, task): idx
                for idx, task in enumerate(tasks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Task {idx} execution failed: {e}")
                    results[idx] = TrajectoryExecution(
                        task_id=tasks[idx].task_id,
                        steps=[],
                        success=False,
                        errors=[str(e)],
                    )
        
        return results
