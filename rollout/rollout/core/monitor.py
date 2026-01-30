"""
Conversation Monitor - Detects when conversations should end.

Provides multiple strategies for early termination detection to avoid
wasting tokens when the user simulator fails to output ###STOP###.

Example:
    >>> monitor = ConversationMonitor(
    ...     max_no_tool_turns=2,
    ...     use_llm_judge=True,
    ...     judge_model="gpt-4o-mini"
    ... )
    >>> 
    >>> # In the pipeline loop
    >>> if monitor.should_stop(messages):
    ...     break
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StopReason(str, Enum):
    """Reasons for stopping a conversation."""
    NONE = "none"
    USER_STOP_TOKEN = "user_stop_token"
    RULE_DETECTED = "rule_detected"
    MAX_NO_TOOL_TURNS = "max_no_tool_turns"
    LLM_JUDGE = "llm_judge"
    MAX_TURNS_REACHED = "max_turns_reached"


@dataclass
class MonitorConfig:
    """Configuration for the conversation monitor."""
    
    # Rule-based detection
    enable_rule_detection: bool = True
    
    # Max turns without tool calls before considering conversation complete
    max_no_tool_turns: int = 2
    
    # LLM Judge settings
    use_llm_judge: bool = False
    judge_model: str = "gpt-4o-mini"
    judge_api_base: Optional[str] = None
    judge_api_key: Optional[str] = None
    
    # Only use LLM judge after rule detection triggers
    llm_judge_after_rule: bool = True
    
    # Custom completion patterns (regex)
    custom_patterns: List[str] = field(default_factory=list)


class ConversationMonitor:
    """
    Monitors conversations and detects when they should end.
    
    Uses multiple strategies:
    1. Rule-based pattern matching (free)
    2. Counting turns without tool calls
    3. Optional LLM judge for confirmation
    """
    
    # Default patterns that suggest conversation completion
    DEFAULT_COMPLETION_PATTERNS = [
        # Explicit completion phrases
        r"all (?:your )?(?:requests?|tasks?) (?:have been |were )?(?:successfully )?completed",
        r"(?:successfully )?completed all",
        r"all (?:done|finished|set)",
        r"everything (?:has been |is )?(?:done|completed|processed)",
        
        # Offering further help (conversation ending)
        r"is there anything else (?:i can|you.d like|you need)",
        r"(?:let me know|feel free to ask) if (?:you need|there.s) anything else",
        r"anything else i can (?:help|assist)",
        r"do you (?:need|have) any(?:thing)? (?:else|other)",
        
        # Farewell phrases
        r"have a (?:great|nice|good|wonderful) (?:day|time)",
        r"(?:glad|happy) (?:i could|to) (?:help|assist)",
        r"take care",
        
        # Summary indicators
        r"(?:here.s|here is) a (?:quick )?summary",
        r"to (?:summarize|recap|sum up)",
        r"in summary",
    ]
    
    # Patterns in user messages that indicate continuation needed
    CONTINUATION_PATTERNS = [
        r"actually",
        r"wait",
        r"one more thing",
        r"also",
        r"can you also",
        r"i forgot",
        r"another",
        r"what about",
        r"how about",
        r"\?$",  # Ends with question mark
    ]
    
    LLM_JUDGE_PROMPT = """You are analyzing a customer service conversation to determine if it has naturally concluded.

Analyze the conversation and respond with ONLY one word: "COMPLETE" or "CONTINUE"

Rules:
- "COMPLETE" if the assistant has addressed all customer requests and the customer seems satisfied
- "COMPLETE" if the assistant is offering further help with phrases like "Is there anything else?"
- "CONTINUE" if there are unresolved requests or the customer has new questions
- "CONTINUE" if the customer is still asking for help

Conversation:
{conversation}

Your answer (COMPLETE or CONTINUE):"""

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize the conversation monitor.
        
        Args:
            config: Monitor configuration
        """
        self.config = config or MonitorConfig()
        
        # Compile patterns
        self.completion_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.DEFAULT_COMPLETION_PATTERNS
        ]
        self.continuation_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.CONTINUATION_PATTERNS
        ]
        
        # Add custom patterns
        for pattern in self.config.custom_patterns:
            try:
                self.completion_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid custom pattern '{pattern}': {e}")
        
        # Track turns without tool calls
        self._no_tool_count: Dict[str, int] = {}
    
    def reset(self, sample_id: str):
        """Reset tracking for a sample."""
        self._no_tool_count[sample_id] = 0
    
    def _check_rule_patterns(self, content: str) -> bool:
        """Check if content matches completion patterns."""
        if not content:
            return False
        
        for pattern in self.completion_patterns:
            if pattern.search(content):
                return True
        return False
    
    def _check_continuation_needed(self, user_content: str) -> bool:
        """Check if user's last message suggests conversation should continue."""
        if not user_content:
            return False
        
        for pattern in self.continuation_patterns:
            if pattern.search(user_content):
                return True
        return False
    
    def _count_recent_no_tool_turns(
        self, 
        messages: List[Dict[str, Any]], 
        sample_id: str
    ) -> int:
        """Count consecutive turns without tool calls from the end."""
        count = 0
        
        # Look at messages from the end
        for msg in reversed(messages):
            role = msg.get("role", "")
            
            if role == "assistant":
                if msg.get("tool_calls"):
                    break  # Found a tool call, stop counting
                count += 1
            elif role == "tool":
                break  # Found tool response, stop counting
        
        return count
    
    def _call_llm_judge(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Use LLM to judge if conversation should end.
        
        Returns True if conversation should end.
        """
        if not self.config.use_llm_judge:
            return False
        
        try:
            from litellm import completion
            
            # Format conversation for the judge
            conv_text = self._format_conversation(messages)
            prompt = self.LLM_JUDGE_PROMPT.format(conversation=conv_text)
            
            kwargs = {
                "model": self.config.judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 10,
            }
            
            if self.config.judge_api_base:
                kwargs["api_base"] = self.config.judge_api_base
            if self.config.judge_api_key:
                kwargs["api_key"] = self.config.judge_api_key
            
            response = completion(**kwargs)
            answer = response.choices[0].message.content.strip().upper()
            
            logger.debug(f"LLM judge response: {answer}")
            return "COMPLETE" in answer
            
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return False
    
    def _format_conversation(self, messages: List[Dict[str, Any]], max_messages: int = 10) -> str:
        """Format conversation for LLM judge (only recent messages)."""
        recent = messages[-max_messages:] if len(messages) > max_messages else messages
        
        lines = []
        for msg in recent:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            if role == "TOOL":
                # Truncate tool responses
                content = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"[TOOL RESPONSE]: {content}")
            elif role == "ASSISTANT":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    lines.append(f"ASSISTANT: {content or ''}")
                    lines.append(f"[Calls tools: {', '.join(tool_names)}]")
                else:
                    lines.append(f"ASSISTANT: {content}")
            else:
                lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def check(
        self, 
        messages: List[Dict[str, Any]], 
        sample_id: str = ""
    ) -> tuple[bool, StopReason]:
        """
        Check if conversation should stop.
        
        Args:
            messages: Current conversation messages
            sample_id: Sample identifier for tracking
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if not messages:
            return False, StopReason.NONE
        
        last_msg = messages[-1]
        last_role = last_msg.get("role", "")
        last_content = last_msg.get("content", "")
        
        # 1. Check max turns without tool calls
        no_tool_count = self._count_recent_no_tool_turns(messages, sample_id)
        if no_tool_count >= self.config.max_no_tool_turns:
            logger.info(f"[{sample_id}] Stopping: {no_tool_count} turns without tool calls")
            return True, StopReason.MAX_NO_TOOL_TURNS
        
        # 2. Rule-based detection on assistant messages
        if self.config.enable_rule_detection and last_role == "assistant":
            if self._check_rule_patterns(last_content):
                logger.debug(f"[{sample_id}] Rule pattern matched in assistant response")
                
                # If LLM judge is configured to confirm after rule detection
                if self.config.use_llm_judge and self.config.llm_judge_after_rule:
                    if self._call_llm_judge(messages):
                        logger.info(f"[{sample_id}] Stopping: Rule + LLM judge confirmed")
                        return True, StopReason.LLM_JUDGE
                    else:
                        logger.debug(f"[{sample_id}] LLM judge says continue")
                        return False, StopReason.NONE
                else:
                    # Just use rule detection
                    logger.info(f"[{sample_id}] Stopping: Rule pattern detected")
                    return True, StopReason.RULE_DETECTED
        
        # 3. Check user's last message for continuation signals
        if last_role == "user":
            if self._check_continuation_needed(last_content):
                return False, StopReason.NONE
        
        return False, StopReason.NONE
    
    def should_stop(
        self, 
        messages: List[Dict[str, Any]], 
        sample_id: str = ""
    ) -> bool:
        """
        Simple check if conversation should stop.
        
        Args:
            messages: Current conversation messages
            sample_id: Sample identifier
            
        Returns:
            True if conversation should stop
        """
        should_stop, _ = self.check(messages, sample_id)
        return should_stop
    
    def get_stop_reason(
        self, 
        messages: List[Dict[str, Any]], 
        sample_id: str = ""
    ) -> StopReason:
        """Get the reason why conversation should stop."""
        _, reason = self.check(messages, sample_id)
        return reason


def create_monitor_from_config(config: dict) -> ConversationMonitor:
    """
    Create a ConversationMonitor from a config dictionary.
    
    Args:
        config: Dictionary with monitor settings
        
    Returns:
        Configured ConversationMonitor
    """
    monitor_config = MonitorConfig(
        enable_rule_detection=config.get("enable_rule_detection", True),
        max_no_tool_turns=config.get("max_no_tool_turns", 2),
        use_llm_judge=config.get("use_llm_judge", False),
        judge_model=config.get("judge_model", "gpt-4o-mini"),
        judge_api_base=config.get("judge_api_base"),
        judge_api_key=config.get("judge_api_key"),
        llm_judge_after_rule=config.get("llm_judge_after_rule", True),
        custom_patterns=config.get("custom_patterns", []),
    )
    return ConversationMonitor(monitor_config)

