"""
Unified LLM Client using LiteLLM with models.yaml configuration.

Provides:
- Explicit URL/KEY per model from models.yaml
- Two model types: textual (gpt-5.2) and coding (claude-opus)
- Automatic retries with exponential backoff
- Auto-continue for truncated responses
- Structured output validation with Pydantic
"""
import os
import json
import re
import logging
from typing import Optional, Any, TypeVar, Type, Literal, List, Dict
from dataclasses import dataclass
from contextlib import contextmanager
import threading

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from ..config.settings import get_settings, LLMConfig
from ..config.models_registry import (
    get_models_registry, 
    ModelsRegistry, 
    ModelInfo,
)

os.environ['LITELLM_LOG'] = 'DEBUG'
litellm.set_verbose = True

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    
    def parse_json(self) -> Any:
        """Extract and parse JSON from response."""
        return extract_json(self.content)
    
    def parse_as(self, model_class: Type[T]) -> T:
        """Parse response as a Pydantic model."""
        data = self.parse_json()
        return model_class.model_validate(data)


def extract_json(text: str) -> Any:
    """
    Extract and parse JSON from text that may contain markdown code blocks.
    Handles common LLM output issues like trailing commas, single quotes, etc.
    """
    # Try to find JSON in code blocks first
    json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```"
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find JSON array or object directly
        array_match = re.search(r'(\[[\s\S]*\])', text)
        object_match = re.search(r'(\{[\s\S]*\})', text)
        
        if array_match and object_match:
            if array_match.start() < object_match.start():
                json_str = array_match.group(1)
            else:
                json_str = object_match.group(1)
        elif array_match:
            json_str = array_match.group(1)
        elif object_match:
            json_str = object_match.group(1)
        else:
            json_str = text.strip()
    
    # Clean up common LLM output issues
    json_str = _clean_json_string(json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"First JSON parse failed: {e}, trying aggressive cleanup")
        json_str = _aggressive_json_cleanup(json_str)
        return json.loads(json_str)


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues from LLM output."""
    # Remove trailing commas before ] or }
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    # Remove any BOM or special characters at the start
    json_str = json_str.lstrip('\ufeff\u200b')
    return json_str


def _aggressive_json_cleanup(json_str: str) -> str:
    """More aggressive JSON cleanup for problematic LLM output."""
    # Remove comments
    json_str = re.sub(r'//[^\n]*\n', '\n', json_str)
    json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
    # Fix unquoted keys
    json_str = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r' "\1":', json_str)
    # Remove trailing commas
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    # Remove non-printable characters
    json_str = ''.join(c for c in json_str if c.isprintable() or c in '\n\r\t')
    return json_str


class LLMClient:
    """
    Unified LLM client with explicit model configuration.
    
    Key features:
    - Uses models.yaml for model configuration
    - Explicitly passes api_base and api_key per model
    - Distinguishes textual vs coding models
    - Automatic retries with exponential backoff
    - Auto-continue for truncated responses
    """
    
    def __init__(
        self, 
        config: Optional[LLMConfig] = None,
        models_registry: Optional[ModelsRegistry] = None,
    ):
        self.config = config or get_settings().llm
        self.registry = models_registry or get_models_registry()
        self._stop_events: Dict[int, threading.Event] = {}
        
        # Disable LiteLLM's internal caching/logging for cleaner output
        litellm.set_verbose = False
        
        # Register models with LiteLLM (required for custom model names)
        self.registry.register_with_litellm()
    
    def _get_model_info(
        self, 
        model_name: Optional[str] = None,
        model_type: Literal["textual", "coding"] = "textual",
    ) -> tuple[str, ModelInfo]:
        """
        Resolve model name and get its configuration.
        
        Priority: explicit model_name > config.yaml override > models.yaml defaults
        
        Args:
            model_name: Explicit model name, or None to use default
            model_type: Type of model ("textual" or "coding")
            
        Returns:
            Tuple of (model_name, ModelInfo)
        """
        if model_name:
            return model_name, self.registry.get_model(model_name)
        
        # Use config override if provided, otherwise use registry defaults from models.yaml
        if model_type == "coding":
            model_name = self.config.coding_model or self.registry.defaults.get("coding")
        else:
            model_name = self.config.textual_model or self.registry.defaults.get("textual")
        
        if not model_name:
            raise ValueError(f"No model configured for type: {model_type}. "
                           f"Set in config.yaml (llm.{model_type}_model) or models.yaml (defaults.{model_type})")
        
        return model_name, self.registry.get_model(model_name)
    
    def _check_stop(self, thread_id: int) -> None:
        """Check if current operation should be stopped."""
        event = self._stop_events.get(thread_id)
        if event and event.is_set():
            raise KeyboardInterrupt("Operation stopped by user")
    
    @contextmanager
    def stoppable(self):
        """Context manager for stoppable operations."""
        thread_id = threading.get_ident()
        self._stop_events[thread_id] = threading.Event()
        try:
            yield self._stop_events[thread_id]
        finally:
            self._stop_events.pop(thread_id, None)
    
    def stop(self, thread_id: Optional[int] = None):
        """Signal to stop operations."""
        if thread_id:
            event = self._stop_events.get(thread_id)
            if event:
                event.set()
        else:
            for event in self._stop_events.values():
                event.set()
    
    def _create_retry_decorator(self):
        """Create retry decorator with current config."""
        return retry(
            retry=retry_if_exception_type((Exception,)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=self.config.retry_delay,
                min=self.config.retry_delay,
                max=self.config.retry_delay * (self.config.retry_backoff ** self.config.max_retries),
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        model_info: ModelInfo,
        **kwargs
    ) -> LLMResponse:
        """
        Internal LLM call using LiteLLM with explicit api_base and api_key.
        
        Args:
            messages: Conversation messages
            model_name: Name of the model
            model_info: Model configuration from registry
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with content and metadata
        """
        thread_id = threading.get_ident()
        self._check_stop(thread_id)
        
        return self._litellm_call(
            messages=messages,
            model_name=model_name,
            model_info=model_info,
            max_continuations=self.config.max_continuations if self.config.auto_continue else 0,
        )
    
    def _litellm_call(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        model_info: ModelInfo,
        max_continuations: int = 5,
    ) -> LLMResponse:
        """
        LiteLLM call with explicit api_base and api_key.
        
        Args:
            messages: Conversation messages
            model_name: Name of the model
            model_info: Model configuration
            max_continuations: Max auto-continue attempts
            
        Returns:
            LLMResponse with content and metadata
        """
        # Build LiteLLM model name with provider prefix to avoid "Provider List" warnings
        litellm_model = f"{model_info.provider}/{model_name}"
        
        logger.debug(
            f"LiteLLM call: model={litellm_model}, api_base={model_info.api_base}, "
            f"temp={model_info.temperature}, top_p={model_info.top_p}"
        )
        
        # Build kwargs, only include top_p if explicitly set
        call_kwargs = {
            "model": litellm_model,
            "messages": messages,
            "api_base": model_info.api_base,
            "api_key": model_info.api_key,
            "temperature": model_info.temperature,
            "max_tokens": model_info.max_tokens,
            "timeout": self.config.timeout,
        }
        
        # Add top_p only if specified (allows temperature-only or top_p-only control)
        if model_info.top_p is not None:
            call_kwargs["top_p"] = model_info.top_p
            # Tell LiteLLM to allow top_p for custom/unrecognized models (like gpt-5)
            call_kwargs["allowed_openai_params"] = ["top_p"]
        
        # Call LiteLLM with explicit api_base and api_key
        response = litellm.completion(**call_kwargs)

        # 直接打印 Usage 对象
        # print(response.usage)

        # # 获取具体数值
        # input_tokens = response.usage.prompt_tokens
        # output_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print(f"输入 Token: {input_tokens}")
        # print(f"输出 Token: {output_tokens}")
        # print(f"总计 Token: {total_tokens}")
        
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # Log refusal reason if present
        if finish_reason == "refusal":
            logger.warning(response)
            logger.warning(call_kwargs)
        # Log finish_reason for debugging truncation issues
        logger.info(f"LLM finish_reason: {finish_reason}, content_length: {len(content) if content else 0}")
        
        # Convert usage to dict if present
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        # Handle truncation with auto-continue
        is_truncated = finish_reason in ("length", "max_tokens")
        needs_continuation = is_truncated and max_continuations > 0 and self.config.auto_continue
        
        if needs_continuation:
            logger.warning(
                f"Response truncated (finish_reason={finish_reason}), "
                f"requesting continuation ({max_continuations} left)..."
            )
            
            # Add continuation messages
            messages = list(messages)
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "Please continue from where you left off. Output ONLY the remaining content, no explanations or repetition."
            })
            
            try:
                continuation = self._litellm_call(
                    messages, 
                    model_name, 
                    model_info, 
                    max_continuations - 1
                )
                content = self._smart_concatenate(content, continuation.content)
            except Exception as e:
                logger.warning(f"Continuation failed ({e}), returning partial content")
        
        return LLMResponse(
            content=content,
            model=model_name,
            usage=usage,
            finish_reason=finish_reason,
            raw_response=response,
        )
    
    def _smart_concatenate(self, first: str, second: str) -> str:
        """
        Smart concatenation of response fragments.
        
        Handles:
        - JSON fragments (strips/re-adds code blocks)
        - Code fragments (preserves structure)
        - Plain text (simple join)
        """
        # Track if we need to re-wrap with code block
        had_code_block = "```" in first
        
        # Strip markdown code blocks from both
        first = first.strip()
        second = second.strip()
        
        for prefix in ["```json", "```python", "```"]:
            if first.startswith(prefix):
                first = first[len(prefix):]
                break
        if first.endswith("```"):
            first = first[:-3]
        first = first.strip()
        
        for prefix in ["```json", "```python", "```"]:
            if second.startswith(prefix):
                second = second[len(prefix):]
                break
        if second.endswith("```"):
            second = second[:-3]
        second = second.strip()
        
        # Concatenate
        result = first + second
        
        # Re-add code block if original had it
        if had_code_block:
            # Detect type
            if "{" in first or "[" in first:
                result = "```json\n" + result + "\n```"
            else:
                result = "```\n" + result + "\n```"
        
        return result
    
    def chat(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        model_type: Literal["textual", "coding"] = "textual",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat request to the LLM.
        
        Args:
            query: Simple user query (alternative to messages)
            messages: Full conversation history
            model: Model name (overrides model_type)
            model_type: "textual" or "coding" (default: textual)
            temperature: Sampling temperature (overrides model config)
            top_p: Nucleus sampling parameter (overrides model config). 
                   Lower values (e.g., 0.1) = more focused/deterministic.
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with content and metadata
        """
        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if query:
                messages.append({"role": "user", "content": query})
        
        # Resolve model
        model_name, model_info = self._get_model_info(model, model_type)
        
        # Override temperature and/or top_p if provided
        overrides = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if top_p is not None:
            overrides["top_p"] = top_p
        
        if overrides:
            model_info = ModelInfo(
                **{**model_info.model_dump(), **overrides}
            )
        
        logger.debug(
            f"Calling LLM: model={model_name}, type={model_type}, "
            f"temp={model_info.temperature}, top_p={model_info.top_p}, messages={len(messages)}"
        )
        
        # Create retry-wrapped call
        retry_decorator = self._create_retry_decorator()
        retrying_call = retry_decorator(self._call_llm)
        
        return retrying_call(
            messages=messages,
            model_name=model_name,
            model_info=model_info,
            **kwargs
        )
    
    def generate_structured(
        self,
        query: str,
        output_model: Type[T],
        model: Optional[str] = None,
        model_type: Literal["textual", "coding"] = "textual",
        max_retries: int = 3,
        **kwargs
    ) -> T:
        """
        Generate structured output validated against a Pydantic model.
        
        Args:
            query: The prompt/query
            output_model: Pydantic model class to validate against
            model: LLM model to use
            model_type: "textual" or "coding"
            max_retries: Number of retries for validation errors
            **kwargs: Additional parameters
            
        Returns:
            Validated Pydantic model instance
        """
        # Add schema hint to prompt
        schema = output_model.model_json_schema()
        enhanced_query = f"""{query}

Please respond with valid JSON matching this schema:
```json
{json.dumps(schema, indent=2)}
```
Output only the JSON, no explanations."""
        
        for attempt in range(max_retries):
            try:
                response = self.chat(
                    query=enhanced_query, 
                    model=model, 
                    model_type=model_type,
                    **kwargs
                )
                return response.parse_as(output_model)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    f"Structured output validation failed (attempt {attempt + 1}): {e}"
                )
                if attempt == max_retries - 1:
                    raise
                enhanced_query = f"""{enhanced_query}

Previous attempt failed with error: {str(e)}
Please fix the JSON format and try again."""
        
        raise RuntimeError("Failed to generate valid structured output")


# Global client instance
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def chat(
    query: str, 
    model_type: Literal["textual", "coding"] = "textual",
    **kwargs
) -> str:
    """Convenience function for simple chat queries."""
    response = get_client().chat(query=query, model_type=model_type, **kwargs)
    return response.content


def chat_json(
    query: str, 
    model_type: Literal["textual", "coding"] = "textual",
    **kwargs
) -> Any:
    """Convenience function for JSON-returning queries."""
    response = get_client().chat(query=query, model_type=model_type, **kwargs)
    return response.parse_json()
