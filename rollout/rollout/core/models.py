"""
Custom Model Registration for LiteLLM.

This module provides utilities for registering custom models with LiteLLM,
especially useful for internal/self-hosted API endpoints that are not
directly supported by LiteLLM.

Usage:
    # Option 1: Register a single model
    from rollout.core.models import register_model
    
    register_model(
        name="my-internal-model",
        provider="deepseek",
        api_base="http://internal-api.example.com/v1",
        api_key="sk-xxx",
        max_tokens=8192
    )
    
    # Option 2: Register from config dict
    from rollout.core.models import register_models_from_config
    
    register_models_from_config({
        "my-model": {
            "provider": "deepseek",
            "api_base": "http://internal-api.example.com/v1",
            "api_key": "sk-xxx"
        }
    })
    
    # Option 3: Load from YAML
    from rollout.core.models import load_models_config
    
    load_models_config("models.yml")
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import litellm
from litellm import completion

# 强制移除全局 Key，迫使 LiteLLM 读取配置文件中的 api_key
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# os.environ['LITELLM_LOG'] = 'DEBUG'

# litellm._turn_on_debug()

logger = logging.getLogger(__name__)

# Registry of custom models with their configurations
_custom_models: Dict[str, "ModelConfig"] = {}


@dataclass
class ModelConfig:
    """Configuration for a custom model."""
    name: str                          # Model name used in code
    provider: str                       # LiteLLM provider (openai, deepseek, anthropic, etc.)
    api_base: Optional[str] = None     # Custom API endpoint
    api_key: Optional[str] = None      # API key (or use env var)
    max_tokens: int = 8192              # Max context window
    input_cost_per_token: float = 0.0  # Cost tracking
    output_cost_per_token: float = 0.0
    mode: str = "chat"                  # chat, completion, etc.
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def litellm_model_name(self) -> str:
        """Get the model name in LiteLLM format (provider/name)."""
        return f"{self.provider}/{self.name}"
    
    def to_litellm_config(self) -> Dict[str, Any]:
        """Convert to LiteLLM register_model format."""
        config = {
            "max_tokens": self.max_tokens,
            "input_cost_per_token": self.input_cost_per_token,
            "output_cost_per_token": self.output_cost_per_token,
            "litellm_provider": self.provider,
            "mode": self.mode,
            **self.extra_params
        }
        return config


def register_model(
    name: str,
    provider: str = "openai",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 8192,
    input_cost_per_token: float = 0.0,
    output_cost_per_token: float = 0.0,
    mode: str = "chat",
    set_env_key: bool = True,
    **extra_params
) -> ModelConfig:
    """
    Register a custom model with LiteLLM.
    
    Args:
        name: Model name (e.g., "my-internal-gpt4")
        provider: LiteLLM provider (openai, deepseek, anthropic, etc.)
        api_base: Custom API endpoint URL
        api_key: API key for the endpoint
        max_tokens: Maximum context window size
        input_cost_per_token: Cost per input token (for tracking)
        output_cost_per_token: Cost per output token (for tracking)
        mode: Model mode (chat, completion)
        set_env_key: Whether to set environment variable for API key
        **extra_params: Additional LiteLLM parameters
        
    Returns:
        ModelConfig object
        
    Example:
        >>> register_model(
        ...     name="deepseek-v3-internal",
        ...     provider="deepseek",
        ...     api_base="http://api.internal.com/v1",
        ...     api_key="sk-xxx"
        ... )
        >>> 
        >>> # Now use it
        >>> agent = Agent(model="deepseek/deepseek-v3-internal")
    """
    config = ModelConfig(
        name=name,
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
        mode=mode,
        extra_params=extra_params
    )

    print(config)
    
    # Register with LiteLLM
    litellm.register_model({
        f"{provider}/{name}": config.to_litellm_config()
    })
    
    # Set environment variable for API key if provided
    # if api_key and set_env_key:
    #     _set_provider_api_key(provider, api_key)
    
    # Store in our registry
    _custom_models[name] = config
    _custom_models[f"{provider}/{name}"] = config
    
    logger.info(f"Registered custom model: {provider}/{name}")
    
    return config


def _set_provider_api_key(provider: str, api_key: str):
    """Set the environment variable for a provider's API key."""
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure": "AZURE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    
    env_var = provider_key_map.get(provider, f"{provider.upper()}_API_KEY")
    os.environ[env_var] = api_key


def register_models_from_config(config: Dict[str, Dict[str, Any]]) -> List[ModelConfig]:
    """
    Register multiple models from a configuration dictionary.
    
    Args:
        config: Dictionary mapping model names to their configurations
        
    Returns:
        List of registered ModelConfig objects
        
    Example:
        >>> register_models_from_config({
        ...     "gpt4-internal": {
        ...         "provider": "openai",
        ...         "api_base": "http://internal-api/v1",
        ...         "api_key": "sk-xxx"
        ...     },
        ...     "deepseek-v3": {
        ...         "provider": "deepseek",
        ...         "api_base": "http://deepseek-api/v1",
        ...         "api_key": "sk-yyy"
        ...     }
        ... })
    """
    registered = []
    
    for name, model_config in config.items():
        try:
            config_obj = register_model(name=name, **model_config)
            registered.append(config_obj)
        except Exception as e:
            logger.error(f"Failed to register model {name}: {e}")
    
    return registered


def load_models_config(config_path: str) -> List[ModelConfig]:
    """
    Load and register models from a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file
        
    Returns:
        List of registered ModelConfig objects
        
    YAML format:
        models:
          my-gpt4:
            provider: openai
            api_base: http://internal-api/v1
            api_key: sk-xxx
            max_tokens: 8192
          
          my-deepseek:
            provider: deepseek
            api_base: http://deepseek-api/v1
            api_key: sk-yyy
    """
    import yaml
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    models_config = data.get("models", data)  # Support both {"models": {...}} and {...}
    print(models_config)
    return register_models_from_config(models_config)


def get_model_config(name: str) -> Optional[ModelConfig]:
    """Get the configuration for a registered model."""
    return _custom_models.get(name)


def list_custom_models() -> List[str]:
    """List all registered custom model names."""
    return [k for k in _custom_models.keys() if "/" not in k]


def get_model_api_base(model: str) -> Optional[str]:
    """Get the API base URL for a model if it's a custom model."""
    config = _custom_models.get(model)
    return config.api_base if config else None


def get_model_api_key(model: str) -> Optional[str]:
    """Get the API key for a model if it's a custom model."""
    config = _custom_models.get(model)
    return config.api_key if config else None


# Convenience function for common internal deployment patterns
def register_internal_deepseek(
    name: str = "deepseek-v3-internal",
    api_base: str = "http://api.dbh.baidu-int.com/v1",
    api_key: Optional[str] = None,
    **kwargs
) -> ModelConfig:
    """
    Convenience function for registering internal DeepSeek deployments.
    
    Example:
        >>> register_internal_deepseek(
        ...     name="deepseek-v3-1-terminus",
        ...     api_key="sk-xxx"
        ... )
        >>> agent = Agent(model="deepseek/deepseek-v3-1-terminus", ...)
    """
    return register_model(
        name=name,
        provider="deepseek",
        api_base=api_base,
        api_key=api_key,
        max_tokens=kwargs.pop("max_tokens", 8192),
        **kwargs
    )


def register_internal_openai(
    name: str = "gpt4-internal",
    api_base: str = "http://localhost:8000/v1",
    api_key: Optional[str] = None,
    **kwargs
) -> ModelConfig:
    """
    Convenience function for registering internal OpenAI-compatible deployments.
    
    Example:
        >>> register_internal_openai(
        ...     name="gpt4-proxy",
        ...     api_base="http://internal-proxy/v1",
        ...     api_key="sk-xxx"
        ... )
        >>> agent = Agent(model="openai/gpt4-proxy", ...)
    """
    return register_model(
        name=name,
        provider="openai",
        api_base=api_base,
        api_key=api_key,
        max_tokens=kwargs.pop("max_tokens", 8192),
        **kwargs
    )


__all__ = [
    "ModelConfig",
    "register_model",
    "register_models_from_config",
    "load_models_config",
    "get_model_config",
    "list_custom_models",
    "get_model_api_base",
    "get_model_api_key",
    "register_internal_deepseek",
    "register_internal_openai",
]

