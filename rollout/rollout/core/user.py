"""
Unified Simulated User class using LiteLLM.

The SimulatedUser plays the role of a customer in multi-turn conversations,
generating realistic responses based on scenario instructions.

For custom/internal models, register them first:
    
    from rollout.core.models import register_model
    register_model(name="gpt4-internal", provider="openai", api_base="...", api_key="...")
    user = SimulatedUser(model="openai/gpt4-internal")
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import litellm
from litellm import completion

from rollout.core.models import get_model_api_base, get_model_api_key

# Control tokens for conversation state
STOP_TOKEN = "###STOP###"
TRANSFER_TOKEN = "###TRANSFER###"
OUT_OF_SCOPE_TOKEN = "###OUT-OF-SCOPE###"


@dataclass
class ConversationState:
    """Represents the current state of a conversation."""
    messages: List[Dict[str, Any]]
    is_stopped: bool = False
    is_transferred: bool = False
    is_out_of_scope: bool = False
    
    @property
    def should_continue(self) -> bool:
        return not (self.is_stopped or self.is_transferred or self.is_out_of_scope)


class SimulatedUser:
    """
    Simulated User for multi-turn conversation scenarios.
    
    The user follows scenario instructions and generates realistic customer
    responses, signaling conversation completion with special tokens.
    
    Example:
        >>> user = SimulatedUser(model="gpt-4")
        >>> user.set_scenario(sample_id, scenario_instructions)
        >>> state = user.generate(trajectory)
    """
    
#     DEFAULT_GUIDELINES = """
# # User Simulation Guidelines
# You are playing the role of a customer contacting a customer service representative. 
# Your goal is to simulate realistic customer interactions while following specific scenario instructions.

# ## Core Principles
# - Generate one message at a time, maintaining natural conversation flow.
# - Strictly follow the scenario instructions you have received.
# - Never make up or hallucinate information not provided in the scenario instructions.
# - Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language.
# - Disclose information progressively. Wait for the agent to ask for specific information.

# ## Task Completion
# - The goal is to continue the conversation until the task is complete.
# - If the instruction goal is satisfied, generate the '###STOP###' token to end the conversation.
# - If you are transferred to another agent, generate the '###TRANSFER###' token.
# - If the scenario lacks information to continue, generate the '###OUT-OF-SCOPE###' token.

# Remember: Create realistic, natural conversations while adhering to the provided instructions.
# """


# <reason_for_call>
# {reason_for_call}
# </reason_for_call>

# <instruction>
# {instruction}
# </instruction>

# <private_information>
# {known_info}
# </private_information>


    DEFAULT_GUIDELINES = """
# User Simulation Guidelines
You are simulating a realistic customer interacting with a customer service assistant.

## Role and Behavior
- You possess a private scenario (<instruction>), and stick to it strictly. Do not take advice from the assistant.
- **CRITICAL RULE:** Do NOT reveal your full agenda in the first message.
- Start with a **single, immediate pain point** or a **general vague request**.
- Only move to the next goal in your instruction after the current one is resolved or if the agent prompts specifically.
- Do not confirm before the assistant asks for it.
- **One Goal Per Turn:** Do not bundle multiple distinct requests (e.g., don't ask for booking AND banking in one sentence).

## Information Source
- Your overall goal and intent are strictly defined by the Instruction. Stick to this objective throughout the conversation.
- Do not hallucinate or invent parameter values based solely on the text of the Instruction. When the Assistant asks for specific details or clarifications (e.g., dates, names, locations, quantities), you must consult the provided known_info.
- If the requested information exists in known_info, use that exact value to answer the Assistant.
- Your goal is to guide the Assistant to the correct tool parameters by revealing the data in known_info naturally.

## Interaction Rules
1. **Internal Monologue:** Before generating the user message, you must briefly think about what strictly needs to be said *now* versus what can wait.
2. - Maintain consistency with your motivation in <reason_for_call>.
3. - Generate only one realistic user message per turn.
4. **Completion:**
   - If all goals are met -> `###STOP###`
   - If transferred -> `###TRANSFER###`
   - If impossible -> `###OUT-OF-SCOPE###`

## Demonstration (Few-Shot)

[Bad Example - DO NOT DO THIS]
Instruction: I need to reset my password and upgrade my plan.
User: Hi, I want to reset my password and also upgrade to the pro plan.
(Reason: Reveals too much at once.)

[Good Example]
Instruction: I need to reset my password and upgrade my plan.
Thought: I should start with the login issue because that's blocking me. I'll ask about the upgrade later.
User: Hi, I'm having trouble logging into my account.
"""
    
    PROMPT_TEMPLATE = """{guidelines}

<scenario>
{instructions}
</scenario>
"""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        custom_guidelines: Optional[str] = None,
        **extra_params
    ):
        """
        Initialize a SimulatedUser.
        
        Args:
            model: LiteLLM model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: API key (if None, uses environment variable)
            api_base: Custom API base URL
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            custom_guidelines: Custom user simulation guidelines (replaces default)
            **extra_params: Additional parameters passed to LiteLLM
        """
        # Auto-configure from registered model if available
        if api_base is None:
            api_base = get_model_api_base(model)
        if api_key is None:
            api_key = get_model_api_key(model)
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_params = extra_params
        self.guidelines = custom_guidelines or self.DEFAULT_GUIDELINES
        
        # Map: sample_id -> scenario instructions
        self._scenarios: Dict[str, str] = {}
    
    def set_scenario(self, sample_id: str, instructions: str):
        """
        Register scenario instructions for a specific sample.
        
        Args:
            sample_id: Unique identifier for the sample
            instructions: Scenario instructions for the user to follow
        """
        self._scenarios[sample_id] = instructions
    
    def _get_system_prompt(self, sample_id: str) -> str:
        """Build the full system prompt for the user."""
        instructions = self._scenarios.get(sample_id, "")
        return self.PROMPT_TEMPLATE.format(
            guidelines=self.guidelines,
            instructions=instructions
        )
    
    def _convert_perspective(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert message roles for user perspective.
        
        The simulated user needs to see agent messages as "user" (from their view)
        and their own previous messages as "assistant".
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "")
            
            # Skip system messages (user has their own system prompt)
            if role == "system":
                continue
            
            # Swap roles: agent's assistant -> user, original user -> assistant
            if role == "assistant":
                # Skip messages with only tool calls (no visible content)
                if msg.get("tool_calls") and not msg.get("content"):
                    continue
                converted.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })
            elif role == "user":
                converted.append({
                    "role": "assistant", 
                    "content": msg.get("content", "")
                })
            # Skip tool messages (internal to agent)
        
        return converted
    
    def _parse_control_tokens(self, content: str) -> tuple[str, bool, bool, bool]:
        """
        Check for control tokens in the response.
        
        Returns:
            Tuple of (cleaned_content, is_stopped, is_transferred, is_out_of_scope)
        """
        is_stopped = STOP_TOKEN in content
        is_transferred = TRANSFER_TOKEN in content
        is_out_of_scope = OUT_OF_SCOPE_TOKEN in content
        
        # Clean control tokens from content for display
        cleaned = content
        for token in [STOP_TOKEN, TRANSFER_TOKEN, OUT_OF_SCOPE_TOKEN]:
            cleaned = cleaned.replace(token, "").strip()
        
        return cleaned, is_stopped, is_transferred, is_out_of_scope
    
    def generate(self, trajectory: Dict[str, Any]) -> ConversationState:
        """
        Generate a user response based on the current trajectory.
        
        Args:
            trajectory: Dictionary containing:
                - id: Sample identifier
                - messages: Conversation history
                
        Returns:
            ConversationState with updated messages and status flags
        """
        sample_id = trajectory["id"]
        messages = trajectory["messages"]
        
        # Build user-perspective messages
        system_prompt = self._get_system_prompt(sample_id)
        user_messages = self._convert_perspective(messages)
        
        # Prepare full message list with system prompt
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(user_messages)
        
        # Build completion kwargs
        kwargs = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
            **self.extra_params
        }
        
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Call LiteLLM
        response = completion(**kwargs)
        content = response.choices[0].message.content or ""
        
        # Parse control tokens
        cleaned_content, is_stopped, is_transferred, is_out_of_scope = \
            self._parse_control_tokens(content)
        
        # Add user response to messages (with cleaned content)
        updated_messages = messages.copy()
        if cleaned_content:  # Only add if there's actual content
            updated_messages.append({
                "role": "user",
                "content": cleaned_content
            })
        
        return ConversationState(
            messages=updated_messages,
            is_stopped=is_stopped,
            is_transferred=is_transferred,
            is_out_of_scope=is_out_of_scope
        )
    
    def __repr__(self) -> str:
        return f"SimulatedUser(model={self.model!r})"

