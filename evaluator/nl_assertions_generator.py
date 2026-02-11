"""
NL Assertions Generator - Generates natural language assertions for trajectory evaluation.

This module generates specific, verifiable assertions based on:
1. The user's instruction (what they want to achieve)
2. The trajectory execution (what tools were called and their results)
3. The entity context (specific values involved)

These assertions are then used by NLAssertionsEvaluator to evaluate agent conversations.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .base import TaskDefinition, TrajectoryExecution, ExecutionStep

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts for NL Assertion Generation
# =============================================================================

NL_ASSERTIONS_SYSTEM_PROMPT = """You are an expert at generating verifiable natural language assertions for evaluating AI agent conversations.

Your task is to analyze a completed trajectory execution (tool calls and their results) along with the user's original instruction, and generate specific assertions that can be used to evaluate whether an AI assistant correctly handled the task.

## Types of Assertions to Generate

1. **Information Communication Assertions**: What specific information should the agent communicate to the user?
   - Exact values from tool outputs (balances, statuses, IDs, dates, etc.)
   - Confirmation of completed actions
   - Important details from query results

2. **Action Completion Assertions**: What actions should the agent have performed?
   - Specific tool calls with correct parameters
   - State changes that should have occurred
   - Operations that needed confirmation before execution

3. **Contextual Assertions**: Based on the task context
   - User expectations based on the instruction
   - Domain-specific requirements

## Guidelines

- Each assertion should be SPECIFIC and VERIFIABLE
- Include EXACT values from tool outputs (numbers, IDs, dates, status strings)
- Use phrases like "Agent communicates that...", "Agent should...", "Agent confirms..."
- Focus on the CRITICAL information and actions, not trivial details
- Generate 3-8 assertions depending on task complexity
- Assertions should be independent (each can be evaluated separately)

## Output Format

Return a JSON array of assertion strings:
```json
[
    "Agent communicates that the scholarship application was successfully submitted.",
    "Agent confirms the appointment is scheduled for 2025-03-18 at 10:15.",
    "Agent communicates that the new bank account balance is $63,089.51.",
    "Agent should open a bank account with product ID 7603a493-27ef-475a-b03a-b17351d95101."
]
```

ONLY output the JSON array, nothing else."""


NL_ASSERTIONS_USER_PROMPT = """## User Instruction
{instruction}

## Reason for Call
{reason_for_call}

## Entity Context
{entity_context}

## Trajectory Execution (Tool Calls and Results)
{execution_log}

## Task
Generate specific, verifiable NL assertions for this completed trajectory. Focus on:
1. Key information the agent should communicate from tool outputs
2. Actions that should be confirmed as completed
3. Specific values (numbers, dates, IDs) that must be conveyed accurately

Return ONLY a JSON array of assertion strings."""


# =============================================================================
# Helper Functions
# =============================================================================

def format_execution_log(steps: List[ExecutionStep]) -> str:
    """Format execution steps into a readable log for the LLM."""
    lines = []
    
    for step in steps:
        tc = step.tool_call
        lines.append(f"### Step {step.step_index + 1}: {tc.name}")
        
        # Format arguments
        if tc.arguments:
            args_str = json.dumps(tc.arguments, indent=2, default=str)
            lines.append(f"**Arguments:**\n```json\n{args_str}\n```")
        
        # Format result
        if tc.result is not None:
            result_str = json.dumps(tc.result, indent=2, default=str)
            # Truncate very long results
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "\n... (truncated)"
            lines.append(f"**Result:**\n```json\n{result_str}\n```")
        
        lines.append("")
    
    return "\n".join(lines)


def format_entity_context(ctx: Dict[str, Any]) -> str:
    """Format entity context for the prompt."""
    return json.dumps(ctx, indent=2, default=str)


# =============================================================================
# NL Assertions Generator
# =============================================================================

@dataclass
class GenerationResult:
    """Result of NL assertions generation."""
    task_id: str
    assertions: List[str]
    success: bool
    error: Optional[str] = None


class NLAssertionsGenerator:
    """
    Generates NL assertions for trajectory evaluation.
    
    Uses LLM to analyze completed trajectory executions and generate
    specific, verifiable assertions about what the agent should communicate
    and accomplish.
    """
    
    def __init__(
        self,
        model: str = None,
        max_retries: int = 2,
    ):
        """
        Initialize the generator.
        
        Args:
            model: LLM model to use (defaults to strong_model from settings)
            max_retries: Number of retry attempts on parsing failure
        """
        self._model = model
        self.max_retries = max_retries
        self._llm_client = None
    
    @property
    def model(self) -> str:
        """Get the model to use."""
        if self._model:
            return self._model
        try:
            from agentskiller.config import get_settings
            return get_settings().llm.strong_model
        except Exception:
            return "gpt-4o-mini"
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from agentskiller.core.llm_client import get_client
            self._llm_client = get_client()
        return self._llm_client
    
    def generate(
        self,
        task: TaskDefinition,
        execution: TrajectoryExecution,
    ) -> GenerationResult:
        """
        Generate NL assertions for a task based on its execution.
        
        Args:
            task: Task definition with instruction and entity context
            execution: Completed trajectory execution with tool results
            
        Returns:
            GenerationResult with list of assertions
        """
        # Format execution log
        execution_log = format_execution_log(execution.steps)
        
        # Get instruction and reason
        instruction = task.instruction or ""
        reason_for_call = task.reason_for_call or ""
        
        # Format entity context
        entity_context = format_entity_context(task.entity_context)
        
        # Build prompt
        user_prompt = NL_ASSERTIONS_USER_PROMPT.format(
            instruction=instruction,
            reason_for_call=reason_for_call,
            entity_context=entity_context,
            execution_log=execution_log,
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": NL_ASSERTIONS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat(
                    messages=messages,
                    model=self.model,
                )
                
                assertions = self._parse_response(response.content)
                
                if assertions:
                    return GenerationResult(
                        task_id=task.task_id,
                        assertions=assertions,
                        success=True,
                    )
                
                logger.warning(f"Attempt {attempt + 1}: Failed to parse assertions")
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
        
        return GenerationResult(
            task_id=task.task_id,
            assertions=[],
            success=False,
            error="Failed to generate assertions after retries",
        )
    
    def generate_from_task_data(
        self,
        task_data: Dict[str, Any],
        execution: TrajectoryExecution,
    ) -> GenerationResult:
        """
        Generate assertions from raw task data dictionary.
        
        Args:
            task_data: Raw task dictionary from JSON file
            execution: Completed trajectory execution
            
        Returns:
            GenerationResult with assertions
        """
        task = TaskDefinition.from_task_file(task_data)
        return self.generate(task, execution)
    
    def _parse_response(self, content: str) -> Optional[List[str]]:
        """Parse assertions from LLM response."""
        content = content.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            if content[start:start+1] == "\n":
                start += 1
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        
        try:
            result = json.loads(content)
            if isinstance(result, list) and all(isinstance(a, str) for a in result):
                return result
        except json.JSONDecodeError:
            # Try to find JSON array in content
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if isinstance(result, list):
                        return [str(a) for a in result]
                except json.JSONDecodeError:
                    pass
        
        return None


# =============================================================================
# Batch Generator
# =============================================================================

class BatchNLAssertionsGenerator:
    """
    Batch generator for processing multiple tasks.
    
    Efficiently processes validated tasks by:
    1. Loading tasks and their execution data
    2. Generating assertions for each unique trajectory
    3. Updating task files with assertions
    """
    
    def __init__(
        self,
        outputs_dir: str = "outputs_cursor",
        model: str = None,
    ):
        self.outputs_dir = Path(outputs_dir)
        self.generator = NLAssertionsGenerator(model=model)
        
        # Cache execution by trajectory hash to avoid re-execution
        self._execution_cache: Dict[str, TrajectoryExecution] = {}
    
    def process_task_file(
        self,
        task_file: Path,
        executor=None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a task file and add NL assertions.
        
        Args:
            task_file: Path to task JSON file
            executor: TrajectoryExecutor instance (created if not provided)
            verbose: Print progress
            
        Returns:
            Dictionary with processing results
        """
        from .trajectory_executor import TrajectoryExecutor
        from .trajectory_code_manager import TrajectoryCodeManager, compute_trajectory_hash
        
        if executor is None:
            executor = TrajectoryExecutor(outputs_dir=str(self.outputs_dir))
        
        code_manager = TrajectoryCodeManager(outputs_dir=self.outputs_dir)
        
        # Load tasks
        with open(task_file) as f:
            tasks = json.load(f)
        
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        results = {
            "file": str(task_file),
            "total_tasks": len(tasks),
            "generated": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }
        
        modified = False
        
        for idx, task_data in enumerate(tasks):
            # Check if assertions already exist
            if task_data.get("nl_assertions"):
                results["skipped"] += 1
                continue
            
            task = TaskDefinition.from_task_file(task_data)
            
            if verbose:
                print(f"  [{idx + 1}/{len(tasks)}] Generating assertions for {task.task_id[:8]}...", end=" ", flush=True)
            
            try:
                # Get execution (use cached codes if available)
                entry = code_manager.get_entry(task.trajectory)
                
                if entry and entry.step_param_codes:
                    execution = executor.execute_with_codes(
                        task=task,
                        step_param_codes=entry.step_param_codes,
                        verbose=False,
                    )
                else:
                    execution = executor.execute(task=task, verbose=False)
                    # Save codes for future use
                    if execution.success and execution.step_param_codes:
                        code_manager.save_code(
                            trajectory=task.trajectory,
                            domains=task.domains or [],
                            is_cross_domain=len(task.domains or []) > 1,
                            generated_code=execution.generated_code,
                            entity_mappings={},
                            step_param_codes=execution.step_param_codes,
                        )
                
                if not execution.success:
                    if verbose:
                        print(f"⚠️ Execution failed")
                    results["failed"] += 1
                    results["errors"].append(f"{task.task_id}: Execution failed")
                    continue
                
                # Generate assertions
                gen_result = self.generator.generate(task, execution)
                
                if gen_result.success and gen_result.assertions:
                    task_data["nl_assertions"] = gen_result.assertions
                    modified = True
                    results["generated"] += 1
                    if verbose:
                        print(f"✓ {len(gen_result.assertions)} assertions")
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{task.task_id}: {gen_result.error}")
                    if verbose:
                        print(f"✗ {gen_result.error}")
                        
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{task.task_id}: {str(e)}")
                if verbose:
                    print(f"✗ Exception: {e}")
        
        # Save modified tasks
        if modified:
            with open(task_file, "w") as f:
                json.dump(tasks, f, indent=2)
            if verbose:
                print(f"  → Saved {results['generated']} new assertions to {task_file.name}")
        
        return results
    
    def process_directory(
        self,
        tasks_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all task files in a directory.
        
        Args:
            tasks_dir: Directory containing task files (defaults to validated_tasks)
            verbose: Print progress
            
        Returns:
            Aggregate results
        """
        from .trajectory_executor import TrajectoryExecutor
        
        if tasks_dir is None:
            tasks_dir = self.outputs_dir / "validated_tasks"
        
        if not tasks_dir.exists():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
        
        # Find all task files
        task_files = list(tasks_dir.rglob("*.json"))
        task_files = [f for f in task_files if "_backup" not in f.stem]
        
        if verbose:
            print(f"\n{'='*60}")
            print("NL Assertions Generation")
            print(f"{'='*60}")
            print(f"Tasks directory: {tasks_dir}")
            print(f"Found {len(task_files)} task files")
            print()
        
        executor = TrajectoryExecutor(outputs_dir=str(self.outputs_dir))
        
        aggregate = {
            "total_files": len(task_files),
            "total_tasks": 0,
            "generated": 0,
            "skipped": 0,
            "failed": 0,
            "file_results": [],
        }
        
        for file_idx, task_file in enumerate(task_files):
            if verbose:
                print(f"\n📁 [{file_idx + 1}/{len(task_files)}] {task_file.relative_to(tasks_dir)}")
            
            result = self.process_task_file(task_file, executor, verbose)
            
            aggregate["total_tasks"] += result["total_tasks"]
            aggregate["generated"] += result["generated"]
            aggregate["skipped"] += result["skipped"]
            aggregate["failed"] += result["failed"]
            aggregate["file_results"].append(result)
        
        if verbose:
            print(f"\n{'='*60}")
            print("Summary")
            print(f"{'='*60}")
            print(f"Total tasks:      {aggregate['total_tasks']}")
            print(f"Generated:        {aggregate['generated']}")
            print(f"Skipped (exist):  {aggregate['skipped']}")
            print(f"Failed:           {aggregate['failed']}")
            print(f"{'='*60}\n")
        
        return aggregate

