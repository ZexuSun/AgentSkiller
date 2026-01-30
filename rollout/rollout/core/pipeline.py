"""
Pipeline for single-turn and multi-turn tool call annotation.

Integrates Agent, SimulatedUser, Tools, and CheckpointManager for
fault-tolerant trajectory generation.
"""

import json
import logging
import sys
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from rollout.core.agent import Agent
from rollout.core.user import SimulatedUser, ConversationState
from rollout.core.checkpoint import (
    CheckpointManager, 
    CheckpointData, 
    TrajectoryStatus,
    ErrorType
)
from rollout.core.monitor import (
    ConversationMonitor,
    MonitorConfig,
    StopReason
)

logger = logging.getLogger(__name__)


# ============================================================================
# Console Printer for Verbose Output
# ============================================================================

class ConsolePrinter:
    """
    Pretty prints conversation messages to console during rollout.
    
    Uses colors and formatting for better readability.
    """
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    
    # Role icons and colors
    ROLE_STYLES = {
        "user": ("👤", "cyan"),
        "assistant": ("🤖", "green"),
        "tool": ("🔧", "yellow"),
        "system": ("⚙️", "dim"),
    }
    
    def __init__(self, enabled: bool = True, use_colors: bool = True, max_content_length: int = 500):
        """
        Initialize the console printer.
        
        Args:
            enabled: Whether printing is enabled
            use_colors: Whether to use ANSI colors
            max_content_length: Max chars to show for content (0 = no limit)
        """
        self.enabled = enabled
        self.use_colors = use_colors and sys.stdout.isatty()
        self.max_content_length = max_content_length
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _truncate(self, text: str) -> str:
        """Truncate text if too long."""
        if self.max_content_length and len(text) > self.max_content_length:
            return text[:self.max_content_length] + "..."
        return text
    
    def print_separator(self, char: str = "─", length: int = 60):
        """Print a separator line."""
        if not self.enabled:
            return
        print(self._color(char * length, "dim"))
    
    def print_turn_header(self, turn: int, step: int, sample_id: str = ""):
        """Print turn/step header."""
        if not self.enabled:
            return
        header = f"Turn {turn + 1} | Step {step + 1}"
        if sample_id:
            header += f" | ID: {sample_id[:12]}..."
        print(self._color(f"\n{'═' * 60}", "bold"))
        print(self._color(f"  {header}", "bold"))
        print(self._color(f"{'═' * 60}", "bold"))
    
    def print_agent_response(self, message: Dict[str, Any]):
        """Print agent response with reasoning, content and tool calls."""
        if not self.enabled:
            return
        
        icon, color = self.ROLE_STYLES["assistant"]
        print(f"\n{icon} {self._color('Assistant:', color)}")
        
        # Print reasoning/thinking content (DeepSeek V3.2+)
        reasoning = message.get("reasoning_content")
        if reasoning:
            print(self._color("  💭 Thinking:", "magenta"))
            for line in self._truncate(reasoning).split("\n"):
                print(f"    {line}")
        
        # Print content
        content = message.get("content")
        if content:
            print(self._color("  Content:", "dim"))
            for line in self._truncate(content).split("\n"):
                print(f"    {line}")
        
        # Print tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            print(self._color(f"  Tool Calls ({len(tool_calls)}):", "yellow"))
            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                
                # Try to parse and format arguments
                try:
                    if isinstance(args, str):
                        args_dict = json.loads(args)
                    else:
                        args_dict = args
                    args_str = json.dumps(args_dict, ensure_ascii=False, indent=2)
                except:
                    args_str = str(args)
                
                print(self._color(f"    [{i+1}] {name}", "yellow"))
                # Show truncated args
                args_preview = self._truncate(args_str.replace("\n", " "))
                print(self._color(f"        Args: {args_preview}", "dim"))
    
    def print_tool_response(self, message: Dict[str, Any]):
        """Print tool response."""
        if not self.enabled:
            return
        
        icon, color = self.ROLE_STYLES["tool"]
        tool_call_id = message.get("tool_call_id", "unknown")[:12]
        content = message.get("content", "")
        
        print(f"\n{icon} {self._color(f'Tool Response (ID: {tool_call_id}...):', color)}")
        
        # Try to format as JSON
        try:
            data = json.loads(content)
            formatted = json.dumps(data, ensure_ascii=False, indent=2)
            for line in self._truncate(formatted).split("\n")[:10]:  # Max 10 lines
                print(f"    {line}")
            if formatted.count("\n") > 10:
                print(self._color("    ...(truncated)", "dim"))
        except:
            print(f"    {self._truncate(content)}")
    
    def print_user_response(self, message: Dict[str, Any], is_stop: bool = False):
        """Print user response."""
        if not self.enabled:
            return
        
        icon, color = self.ROLE_STYLES["user"]
        content = message.get("content", "")
        
        status = ""
        if is_stop:
            status = self._color(" [CONVERSATION ENDED]", "red")
        
        print(f"\n{icon} {self._color('User:', color)}{status}")
        for line in self._truncate(content).split("\n"):
            print(f"    {line}")
    
    def print_error(self, error: str):
        """Print error message."""
        if not self.enabled:
            return
        print(self._color(f"\n❌ Error: {error}", "red"))
    
    def print_summary(self, result: "TrajectoryResult"):
        """Print trajectory summary."""
        if not self.enabled:
            return
        
        print(self._color(f"\n{'─' * 60}", "dim"))
        status = "✅ Success" if result.success else "❌ Failed"
        status_color = "green" if result.success else "red"
        print(self._color(f"  {status}", status_color))
        
        if result.error:
            print(self._color(f"  Error: {result.error}", "red"))
        
        # Count messages by role
        role_counts = {}
        for msg in result.messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        counts_str = ", ".join(f"{r}: {c}" for r, c in role_counts.items())
        print(self._color(f"  Messages: {counts_str}", "dim"))
        print(self._color(f"{'─' * 60}\n", "dim"))


# ============================================================================
# Helper Functions
# ============================================================================

def clear_reasoning_content(messages: List[Dict[str, Any]]) -> None:
    """
    Clear reasoning_content from all messages in the list.
    
    This is used at the end of each turn to remove thinking content
    from the context to save tokens, following DeepSeek V3.2 conventions.
    
    Note: This modifies the messages in-place. The trajectory should be
    saved BEFORE calling this function to preserve the full reasoning content.
    
    Args:
        messages: List of message dictionaries to modify
    """
    for message in messages:
        if isinstance(message, dict) and "reasoning_content" in message:
            message["reasoning_content"] = None


def copy_messages_without_reasoning(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a deep copy of messages with reasoning_content cleared.
    
    This is useful when you want to keep the original messages intact
    but pass a version without reasoning to the API.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        New list with copied messages (reasoning_content set to None)
    """
    import copy
    copied = copy.deepcopy(messages)
    clear_reasoning_content(copied)
    return copied


class PipelineMode(str, Enum):
    """Mode of operation for the pipeline."""
    POSITIVE = "positive"      # Normal tool calling
    WRONG_TOOL = "wrong_tool"  # Provide wrong tools (agent should refuse)
    NO_TOOL = "no_tool"        # No tools provided (agent should refuse)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    max_turns: int = 1              # Max conversation turns (1 = single-turn)
    max_steps_per_turn: int = 10    # Max tool call steps per turn
    mode: PipelineMode = PipelineMode.POSITIVE
    timeout_per_step: float = 120.0  # Timeout for each step
    use_checkpoints: bool = True     # Enable checkpoint saving
    checkpoint_dir: str = "./checkpoints"
    resume: bool = True              # Whether to resume from checkpoints
    
    # Verbose output options
    verbose: bool = False            # Print messages to console in real-time
    verbose_colors: bool = True      # Use ANSI colors in output
    verbose_max_length: int = 500    # Max content length (0 = no limit)
    
    # Conversation Monitor options (early termination detection)
    enable_monitor: bool = True               # Enable conversation monitoring
    monitor_rule_detection: bool = False      # Use rule-based pattern detection (disabled by default)
    monitor_max_no_tool_turns: int = 2        # Max turns without tool calls
    monitor_use_llm_judge: bool = False       # Use LLM to confirm termination
    monitor_judge_model: str = "gpt-4o-mini"  # Model for LLM judge
    
    @property
    def is_multi_turn(self) -> bool:
        return self.max_turns > 1


@dataclass  
class TrajectoryResult:
    """Result of processing a single trajectory."""
    trajectory_id: str
    messages: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.trajectory_id,
            "messages": self.messages,
            "success": self.success,
            "error": self.error,
            **self.metadata
        }


class ToolRouter:
    """
    Routes tool calls to appropriate tool instances.
    
    Handles tool name resolution including namespaced tools (e.g., "Server.tool_name").
    """
    
    def __init__(self, tools: Dict[str, Any]):
        """
        Initialize the router.
        
        Args:
            tools: Dictionary mapping tool names to tool instances
        """
        self.tools = tools
        self._build_function_index()
    
    def _build_function_index(self):
        """Build an index of function names to tools."""
        self.function_to_tool: Dict[str, tuple] = {}
        
        for tool_name, tool in self.tools.items():
            try:
                info = tool.info
                infos = info if isinstance(info, list) else [info]
                
                for item in infos:
                    if "function" in item:
                        func_name = item["function"]["name"]
                        # Store both full name and short name
                        self.function_to_tool[func_name] = (tool_name, tool)
                        # Also store without namespace prefix
                        if "." in func_name:
                            short_name = func_name.split(".")[-1]
                            if short_name not in self.function_to_tool:
                                self.function_to_tool[short_name] = (tool_name, tool)
            except Exception as e:
                logger.warning(f"Failed to index tool {tool_name}: {e}")
    
    def route(
        self, 
        tool_call: Dict[str, Any], 
        sample_id: str
    ) -> tuple[str, str]:
        """
        Route a tool call to the appropriate tool.
        
        Args:
            tool_call: Tool call dict with 'id' and 'function' keys
            sample_id: Sample identifier for context
            
        Returns:
            Tuple of (result_content, tool_call_id)
        """
        tool_call_id = tool_call.get("id", "unknown")
        
        try:
            function = tool_call.get("function", {})
            full_name = function.get("name", "")
            arguments = function.get("arguments", "{}")
            
            # Try to find the tool
            tool_name = None
            tool = None
            
            # Check full name first
            if full_name in self.function_to_tool:
                tool_name, tool = self.function_to_tool[full_name]
            else:
                # Try without namespace
                short_name = full_name.split(".")[-1] if "." in full_name else full_name
                if short_name in self.function_to_tool:
                    tool_name, tool = self.function_to_tool[short_name]
                # Also check direct tool names
                elif full_name.split(".")[0] in self.tools:
                    server_name = full_name.split(".")[0]
                    tool = self.tools[server_name]
            
            if tool is None:
                return f"Error: Unknown tool '{full_name}'", tool_call_id
            
            # Build tool call string
            tool_call_str = json.dumps({
                "name": full_name.split(".")[-1],  # Use short name
                "arguments": arguments
            }, ensure_ascii=False)
            
            # Execute tool
            result = tool.get_observation(tool_call_str, sample_id=sample_id)
            
            # Ensure result is string
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            
            return result, tool_call_id
            
        except Exception as e:
            logger.error(f"Tool routing error: {e}")
            return f"Error: {e}", tool_call_id


class Pipeline:
    """
    Main pipeline for trajectory generation with fault tolerance.
    
    Supports both single-turn and multi-turn conversations with automatic
    checkpoint management for resuming interrupted trajectories.
    
    Example:
        >>> from rollout import Pipeline, Agent, SimulatedUser
        >>> from rollout.tools import discover_tools, instantiate_tools
        >>> 
        >>> # Setup
        >>> agent = Agent(model="gpt-4")
        >>> user = SimulatedUser(model="gpt-4")
        >>> tools = instantiate_tools(["Calculator", "Search"])
        >>> 
        >>> # Create pipeline
        >>> pipeline = Pipeline(
        ...     agent=agent,
        ...     user=user,
        ...     tools=tools,
        ...     config=PipelineConfig(max_turns=5)
        ... )
        >>> 
        >>> # Process samples
        >>> results = pipeline.process_batch(samples)
    """
    
    def __init__(
        self,
        agent: Agent,
        tools: Dict[str, Any],
        user: Optional[SimulatedUser] = None,
        config: Optional[PipelineConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            agent: The Agent instance for generating responses
            tools: Dictionary of tool instances
            user: Optional SimulatedUser for multi-turn conversations
            config: Pipeline configuration
            checkpoint_manager: Optional custom checkpoint manager
        """
        self.agent = agent
        self.tools = tools
        self.user = user
        self.config = config or PipelineConfig()
        
        self.tool_router = ToolRouter(tools)
        
        # Initialize console printer for verbose output
        self.printer = ConsolePrinter(
            enabled=self.config.verbose,
            use_colors=self.config.verbose_colors,
            max_content_length=self.config.verbose_max_length
        )
        
        # Initialize checkpoint manager
        if self.config.use_checkpoints:
            self.checkpoint_manager = checkpoint_manager or CheckpointManager(
                checkpoint_dir=self.config.checkpoint_dir
            )
        else:
            self.checkpoint_manager = None
        
        # Initialize conversation monitor for early termination detection
        if self.config.enable_monitor:
            self.monitor = ConversationMonitor(MonitorConfig(
                enable_rule_detection=self.config.monitor_rule_detection,
                max_no_tool_turns=self.config.monitor_max_no_tool_turns,
                use_llm_judge=self.config.monitor_use_llm_judge,
                judge_model=self.config.monitor_judge_model,
            ))
        else:
            self.monitor = None
    
    def _collect_tools_info(self) -> List[Dict[str, Any]]:
        """Collect tool info from all tools."""
        tools_info = []
        for tool in self.tools.values():
            info = tool.info
            if isinstance(info, list):
                tools_info.extend(info)
            else:
                tools_info.append(info)
        return tools_info
    
    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        sample_id: str,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls in parallel.
        
        Returns list of tool response messages.
        """
        responses = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.tool_router.route, tc, sample_id): tc
                for tc in tool_calls
            }
            
            for future in as_completed(futures):
                try:
                    content, tool_call_id = future.result()
                    responses.append({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id
                    })
                except Exception as e:
                    tc = futures[future]
                    responses.append({
                        "role": "tool",
                        "content": f"Error: {e}",
                        "tool_call_id": tc.get("id", "unknown")
                    })
        
        return responses
    
    def process_single_turn(
        self,
        sample: Dict[str, Any],
        tools_info: Optional[List[Dict[str, Any]]] = None
    ) -> TrajectoryResult:
        """
        Process a single-turn trajectory.
        
        Args:
            sample: Sample dict with 'id', 'messages', and optionally 'tools'
            tools_info: Optional override for tools info
            
        Returns:
            TrajectoryResult with the completed trajectory
        """
        sample_id = sample["id"]
        messages = sample["messages"].copy()
        tools_info = tools_info or sample.get("tools") or self._collect_tools_info()
        
        # Handle different modes
        if self.config.mode == PipelineMode.NO_TOOL:
            tools_info = None
        elif self.config.mode == PipelineMode.WRONG_TOOL:
            tools_info = sample.get("tools")  # Assume sample has wrong tools
        
        # Check for checkpoint (only if resume is enabled)
        checkpoint = None
        if self.config.resume and self.checkpoint_manager and self.checkpoint_manager.can_resume(sample_id):
            checkpoint = self.checkpoint_manager.load(sample_id)
            if checkpoint:
                messages = checkpoint.messages
                logger.info(f"Resuming {sample_id} from step {checkpoint.current_step}")
        
        try:
            # Use checkpoint context if available
            if self.checkpoint_manager:
                with self.checkpoint_manager.checkpoint_context(
                    sample_id, 
                    initial_messages=messages
                ) as cp:
                    result = self._run_single_turn_loop(
                        sample_id, 
                        cp.messages, 
                        tools_info, 
                        cp
                    )
                    if result.success:
                        self.checkpoint_manager.mark_completed(sample_id)
                    return result
            else:
                return self._run_single_turn_loop(
                    sample_id, 
                    messages, 
                    tools_info, 
                    None
                )
                
        except Exception as e:
            logger.error(f"Single-turn error for {sample_id}: {e}")
            return TrajectoryResult(
                trajectory_id=sample_id,
                messages=messages,
                success=False,
                error=str(e)
            )
    
    def _run_single_turn_loop(
        self,
        sample_id: str,
        messages: List[Dict[str, Any]],
        tools_info: Optional[List[Dict[str, Any]]],
        checkpoint: Optional[CheckpointData]
    ) -> TrajectoryResult:
        """
        Inner loop for single-turn processing.
        
        For single-turn, we don't clear reasoning_content between steps
        since there's no turn boundary. All reasoning_content is preserved.
        """
        start_step = checkpoint.current_step if checkpoint else 0
        
        for step in range(start_step, self.config.max_steps_per_turn):
            # Update checkpoint
            if checkpoint:
                checkpoint.current_step = step
            
            # Verbose: print step header
            self.printer.print_turn_header(0, step, sample_id)
            
            # Generate agent response (may include reasoning_content for DeepSeek V3.2+)
            agent_msg = self.agent.generate(messages, tools_info)
            messages.append(agent_msg)
            
            # Verbose: print agent response (including reasoning if present)
            self.printer.print_agent_response(agent_msg)
            
            # Update checkpoint after agent response
            if checkpoint:
                checkpoint.messages = messages
            
            # Check for tool calls
            tool_calls = agent_msg.get("tool_calls")
            if not tool_calls:
                # Agent finished (no more tool calls)
                break
            
            # Execute tool calls
            tool_responses = self._execute_tool_calls(tool_calls, sample_id)
            messages.extend(tool_responses)
            
            # Verbose: print tool responses
            for resp in tool_responses:
                self.printer.print_tool_response(resp)
            
            # Update checkpoint after tool responses
            if checkpoint:
                checkpoint.messages = messages
        
        result = TrajectoryResult(
            trajectory_id=sample_id,
            messages=messages,  # Full trajectory with reasoning_content preserved
            success=True
        )
        
        # Verbose: print summary
        self.printer.print_summary(result)
        
        return result
    
    def process_multi_turn(
        self,
        sample: Dict[str, Any],
        tools_info: Optional[List[Dict[str, Any]]] = None
    ) -> TrajectoryResult:
        """
        Process a multi-turn trajectory.
        
        Args:
            sample: Sample dict with 'id', 'messages', 'user_system_prompt'
            tools_info: Optional override for tools info
            
        Returns:
            TrajectoryResult with the completed trajectory
        """
        if self.user is None:
            raise ValueError("SimulatedUser required for multi-turn processing")
        
        sample_id = sample["id"]
        messages = sample["messages"].copy()
        tools_info = tools_info or sample.get("tools") or self._collect_tools_info()
        user_system_prompt = sample.get("user_system_prompt", "")
        
        # Register user scenario
        self.user.set_scenario(sample_id, user_system_prompt)
        
        # Check for checkpoint (only if resume is enabled)
        checkpoint = None
        start_turn = 0
        start_step = 0
        
        if self.config.resume and self.checkpoint_manager and self.checkpoint_manager.can_resume(sample_id):
            checkpoint = self.checkpoint_manager.load(sample_id)
            if checkpoint:
                messages = checkpoint.messages
                start_turn = checkpoint.current_turn
                start_step = checkpoint.current_step
                logger.info(f"Resuming {sample_id} from turn {start_turn}, step {start_step}")
        
        try:
            if self.checkpoint_manager:
                with self.checkpoint_manager.checkpoint_context(
                    sample_id,
                    initial_messages=messages
                ) as cp:
                    cp.user_system_prompt = user_system_prompt
                    result = self._run_multi_turn_loop(
                        sample_id,
                        cp.messages,
                        tools_info,
                        cp,
                        start_turn,
                        start_step
                    )
                    if result.success:
                        self.checkpoint_manager.mark_completed(sample_id)
                    return result
            else:
                return self._run_multi_turn_loop(
                    sample_id,
                    messages,
                    tools_info,
                    None,
                    start_turn,
                    start_step
                )
                
        except Exception as e:
            logger.error(f"Multi-turn error for {sample_id}: {e}")
            return TrajectoryResult(
                trajectory_id=sample_id,
                messages=messages,
                success=False,
                error=str(e)
            )
    
    def _run_multi_turn_loop(
        self,
        sample_id: str,
        messages: List[Dict[str, Any]],
        tools_info: Optional[List[Dict[str, Any]]],
        checkpoint: Optional[CheckpointData],
        start_turn: int,
        start_step: int
    ) -> TrajectoryResult:
        """
        Inner loop for multi-turn processing.
        
        Handles DeepSeek V3.2 style reasoning_content:
        - reasoning_content is preserved in trajectory for saving
        - Context is cleared of reasoning_content at turn boundaries to save tokens
        - Uses separate trajectory_messages only when reasoning_content is detected
        """
        import copy
        
        # Track if we've seen any reasoning_content (for DeepSeek V3.2 optimization)
        has_reasoning = False
        trajectory_messages = None  # Lazy init only if needed
        
        for turn in range(start_turn, self.config.max_turns):
            step_start = start_step if turn == start_turn else 0
            
            # Update checkpoint
            if checkpoint:
                checkpoint.current_turn = turn
            
            for step in range(step_start, self.config.max_steps_per_turn):
                # Update checkpoint
                if checkpoint:
                    checkpoint.current_step = step
                
                # Verbose: print turn/step header
                self.printer.print_turn_header(turn, step, sample_id)
                
                # Generate agent response (context has reasoning cleared from previous turns)
                agent_msg = self.agent.generate(messages, tools_info)
                
                # Check if this response has reasoning_content (DeepSeek V3.2)
                if agent_msg.get("reasoning_content"):
                    if not has_reasoning:
                        # First time seeing reasoning - initialize trajectory copy
                        has_reasoning = True
                        trajectory_messages = copy.deepcopy(messages)
                        logger.debug(f"[{sample_id}] Detected reasoning_content, enabling trajectory preservation")
                
                # Append to context
                messages.append(agent_msg)
                
                # If tracking trajectory separately, append there too
                if trajectory_messages is not None:
                    trajectory_messages.append(copy.deepcopy(agent_msg))
                
                # Verbose: print agent response
                self.printer.print_agent_response(agent_msg)
                
                # Update checkpoint
                if checkpoint:
                    checkpoint.messages = copy.deepcopy(trajectory_messages) if trajectory_messages else messages
                
                # Check for tool calls
                tool_calls = agent_msg.get("tool_calls")
                if not tool_calls:
                    # No tool calls - agent finished this step
                    break
                
                # Execute tool calls
                tool_responses = self._execute_tool_calls(tool_calls, sample_id)
                messages.extend(tool_responses)
                
                if trajectory_messages is not None:
                    trajectory_messages.extend(copy.deepcopy(tool_responses))
                
                # Verbose: print tool responses
                for resp in tool_responses:
                    self.printer.print_tool_response(resp)
                
                # Update checkpoint
                if checkpoint:
                    checkpoint.messages = copy.deepcopy(trajectory_messages) if trajectory_messages else messages
            
            # Check monitor BEFORE user response (to detect if assistant already completed)
            monitor_stop = False
            monitor_reason = StopReason.NONE
            # if self.monitor:
            #     monitor_stop, monitor_reason = self.monitor.check(messages, sample_id)
            #     if monitor_stop:
            #         logger.info(f"[{sample_id}] Monitor detected completion: {monitor_reason.value}")
            
            # Generate user response
            trajectory = {"id": sample_id, "messages": messages}
            state = self.user.generate(trajectory)
            messages = state.messages
            
            # Sync trajectory_messages if tracking separately
            if trajectory_messages is not None and messages and messages[-1].get("role") == "user":
                trajectory_messages.append(copy.deepcopy(messages[-1]))
            
            # Verbose: print user response
            if messages and messages[-1].get("role") == "user":
                self.printer.print_user_response(
                    messages[-1], 
                    is_stop=not state.should_continue
                )
            
            # === DeepSeek V3.2: Clear reasoning_content from context at turn boundary ===
            # Only do this if we've seen reasoning_content
            if has_reasoning:
                clear_reasoning_content(messages)
                logger.debug(f"[{sample_id}] Turn {turn + 1} ended, cleared reasoning_content from context")
            
            # Update checkpoint
            if checkpoint:
                checkpoint.messages = copy.deepcopy(trajectory_messages) if trajectory_messages else messages
            
            # Check if conversation should end (user said STOP OR monitor detected completion)
            should_end = not state.should_continue
            
            # If user didn't say STOP but monitor detected completion, trust the monitor
            if not should_end and monitor_stop:
                logger.info(f"[{sample_id}] User didn't send STOP but monitor detected completion, ending conversation")
                should_end = True
                # Override state flags based on monitor
                state.is_stopped = True
            
            if should_end:
                # Use trajectory_messages if we tracked separately, otherwise use messages
                final_messages = trajectory_messages if trajectory_messages else messages
                result = TrajectoryResult(
                    trajectory_id=sample_id,
                    messages=final_messages,
                    success=True,
                    metadata={
                        "stopped": state.is_stopped,
                        "transferred": state.is_transferred,
                        "out_of_scope": state.is_out_of_scope,
                        "total_turns": turn + 1,
                        "stop_reason": monitor_reason.value if monitor_stop else "user_token",
                        "has_reasoning": has_reasoning
                    }
                )
                # Verbose: print summary
                self.printer.print_summary(result)
                return result
        
        # Reached max turns
        final_messages = trajectory_messages if trajectory_messages else messages
        result = TrajectoryResult(
            trajectory_id=sample_id,
            messages=final_messages,
            success=True,
            metadata={
                "reached_max_turns": True, 
                "total_turns": self.config.max_turns,
                "has_reasoning": has_reasoning
            }
        )
        # Verbose: print summary
        self.printer.print_summary(result)
        return result
    
    def process(
        self,
        sample: Dict[str, Any],
        tools_info: Optional[List[Dict[str, Any]]] = None
    ) -> TrajectoryResult:
        """
        Process a sample (auto-detects single vs multi-turn).
        
        Args:
            sample: Sample dictionary
            tools_info: Optional tools info override
            
        Returns:
            TrajectoryResult
        """
        if self.config.is_multi_turn:
            return self.process_multi_turn(sample, tools_info)
        else:
            return self.process_single_turn(sample, tools_info)
    
    def process_batch(
        self,
        samples: List[Dict[str, Any]],
        max_workers: int = 4,
        tools_info: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TrajectoryResult]:
        """
        Process multiple samples in parallel.
        
        Args:
            samples: List of sample dictionaries
            max_workers: Number of parallel workers
            tools_info: Optional tools info (shared across samples)
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            List of TrajectoryResult objects
        """
        tools_info = tools_info or self._collect_tools_info()
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process, sample, tools_info): sample
                for sample in samples
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    sample = futures[future]
                    results.append(TrajectoryResult(
                        trajectory_id=sample.get("id", "unknown"),
                        messages=sample.get("messages", []),
                        success=False,
                        error=str(e)
                    ))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(samples))
        
        return results
    
    def get_resumable_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter samples to only those that can be resumed.
        
        Args:
            samples: List of all samples
            
        Returns:
            List of samples that have resumable checkpoints
        """
        if not self.checkpoint_manager:
            return []
        
        resumable_ids = set(self.checkpoint_manager.get_resumable_trajectories())
        return [s for s in samples if s.get("id") in resumable_ids]
    
    def filter_completed(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out completed samples (for resume mode).
        
        Args:
            samples: List of all samples
            
        Returns:
            List of samples that haven't been completed yet
        """
        if not self.checkpoint_manager:
            return samples
        
        completed = set()
        for cp in self.checkpoint_manager.list_checkpoints(TrajectoryStatus.COMPLETED):
            completed.add(cp.trajectory_id)
        
        return [s for s in samples if s.get("id") not in completed]

