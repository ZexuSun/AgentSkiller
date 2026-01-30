"""
Unified Agent class using LiteLLM for standardized LLM API calls.

LiteLLM handles the differences between various LLM providers (OpenAI, Anthropic, 
DeepSeek, Azure, etc.) automatically, including tool call format conversion.

For custom/internal models, use the models module to register them first:
    
    from rollout.core.models import register_model
    register_model(
        name="my-internal-model",
        provider="deepseek",
        api_base="http://internal-api/v1",
        api_key="sk-xxx"
    )
    
    agent = Agent(model="deepseek/my-internal-model")
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import litellm
from litellm import completion
from pprint import pprint

# Suppress LiteLLM info logs by default
litellm.set_verbose = False

# Import model utilities for auto-configuration
from rollout.core.models import get_model_api_base, get_model_api_key


@dataclass
class AgentConfig:
    """Configuration for an Agent instance."""
    model: str  # LiteLLM model string, e.g., "gpt-4", "anthropic/claude-3", "deepseek/deepseek-chat"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None  # If None, uses env var
    api_base: Optional[str] = None  # Custom API endpoint
    timeout: int = 120
    max_retries: int = 3
    enable_thinking: bool = False  # Enable thinking/reasoning for DeepSeek V3.2+
    thinking: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """
    Unified Agent class that works with any LLM provider via LiteLLM.
    
    Supported providers include:
    - OpenAI: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
    - Anthropic: "anthropic/claude-3-opus", "anthropic/claude-3-sonnet"
    - DeepSeek: "deepseek/deepseek-chat", "deepseek/deepseek-coder"
    - Azure: "azure/gpt-4", "azure/gpt-35-turbo"
    - And 100+ more providers
    
    See https://docs.litellm.ai/docs/providers for full list.
    
    Example:
        >>> agent = Agent(model="deepseek/deepseek-chat")
        >>> response = agent.generate(messages, tools)
    """
    
    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        enable_thinking: bool = False,
        thinking: Dict[str, str] = {"type": "disable"},
        **extra_params
    ):
        """
        Initialize an Agent.
        
        Args:
            model: LiteLLM model identifier (e.g., "gpt-4", "anthropic/claude-3-sonnet")
            system_prompt: Optional system prompt for the agent
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: API key (if None, uses environment variable)
            api_base: Custom API base URL
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            enable_thinking: Enable thinking/reasoning mode for DeepSeek V3.2+
            **extra_params: Additional parameters passed to LiteLLM
        """
        # Auto-configure from registered model if available
        if api_base is None:
            api_base = get_model_api_base(model)
        if api_key is None:
            api_key = get_model_api_key(model)
        
        self.config = AgentConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            enable_thinking=enable_thinking,
            thinking=thinking,
            extra_params=extra_params
        )
        self.system_prompt = system_prompt
        
        # Set API key if provided
        # if api_key:
        #     self._set_api_key(model, api_key)
    
    def _set_api_key(self, model: str, api_key: str):
        """Set the appropriate environment variable for the API key."""
        # Map model prefixes to environment variable names
        key_mapping = {
            "gpt": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "azure": "AZURE_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "cohere": "COHERE_API_KEY",
        }
        
        for prefix, env_var in key_mapping.items():
            if model.startswith(prefix) or f"/{prefix}" in model:
                os.environ[env_var] = api_key
                return
        
        # Default to OpenAI format for unknown providers
        os.environ["OPENAI_API_KEY"] = api_key
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool definitions in OpenAI format
            tool_choice: Optional tool choice directive ("auto", "none", or specific tool)
            
        Returns:
            Dictionary containing the assistant's response with keys:
            - role: "assistant"
            - content: Text content (may be None if tool_calls present)
            - tool_calls: List of tool call objects (if any)
            - reasoning_content: Thinking/reasoning content (DeepSeek V3.2+, if enabled)
            
        Raises:
            litellm.exceptions.APIError: On API errors after retries exhausted
        """
        # Prepare messages with system prompt
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)
        
        # Build completion kwargs
        kwargs = {
            "model": self.config.model,
            "messages": full_messages,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
            "num_retries": self.config.max_retries,
            **self.config.extra_params
        }
        
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        
        # Enable thinking mode for DeepSeek V3.2+ models
        if self.config.enable_thinking:
            kwargs["extra_body"] = kwargs.get("extra_body", {})
            kwargs["extra_body"]["enable_thinking"] = True

        if self.config.thinking["type"] == "enabled":
            kwargs["extra_body"] = kwargs.get("extra_body", {})
            kwargs["extra_body"]["thinking"] = {"type":"enabled"}

        
        # Call LiteLLM
        response = completion(**kwargs)
        
        # Extract and normalize the response
        choice = response.choices[0]
        message = choice.message
        
        result = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": None,
        }
        
        # Extract reasoning_content for DeepSeek V3.2+ models (only add if present)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            result["reasoning_content"] = message.reasoning_content
        
        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result
    
    def __repr__(self) -> str:
        return f"Agent(model={self.config.model!r})"

