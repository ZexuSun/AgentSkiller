"""
Models Registry - Load and manage LLM model configurations from models.yaml.
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelInfo(BaseModel):
    """Configuration for a single LLM model."""
    
    provider: Literal["openai", "anthropic", "azure", "custom"] = "openai"
    api_base: str
    api_key: str = ""
    max_tokens: int = 16384
    temperature: float = 0.2
    top_p: Optional[float] = None  # Nucleus sampling: lower = more focused (e.g., 0.1 for precise)
    supports_structured_output: bool = True


class RetryConfig(BaseModel):
    """Retry configuration for LLM calls."""
    
    max_retries: int = 3
    retry_delay: float = 2.0
    retry_backoff: float = 2.0
    max_delay: float = 60.0


class ModelsRegistry:
    """
    Registry for LLM models loaded from models.yaml.
    
    All api_base and api_key values are explicit in the config file.
    Supports default model selection by task type (textual/coding).
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("models.yaml")
        self.models: Dict[str, ModelInfo] = {}
        self.defaults: Dict[str, str] = {}
        self.retry_config: RetryConfig = RetryConfig()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load models configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Models config not found: {self.config_path}")
            return
        
        with open(self.config_path) as f:
            data = yaml.safe_load(f) or {}
        
        # Load defaults
        self.defaults = data.get("defaults", {})
        
        # Load retry config
        if "retry" in data:
            self.retry_config = RetryConfig(**data["retry"])
        
        # Load models
        models_data = data.get("models", {})
        for model_name, model_config in models_data.items():
            try:
                self.models[model_name] = ModelInfo(**model_config)
                logger.debug(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
    
    def get_model(self, model_name: str) -> ModelInfo:
        """
        Get model configuration by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo for the model
            
        Raises:
            KeyError: If model is not found
        """
        if model_name not in self.models:
            raise KeyError(f"Model not found: {model_name}. Available: {list(self.models.keys())}")
        return self.models[model_name]
    
    def get_default_model(
        self, 
        model_type: Literal["textual", "coding"] = "textual"
    ) -> tuple[str, ModelInfo]:
        """
        Get the default model for a task type.
        
        Args:
            model_type: "textual" or "coding"
            
        Returns:
            Tuple of (model_name, ModelInfo)
        """
        model_name = self.defaults.get(model_type)
        if not model_name:
            # Fallback to first available model
            model_name = next(iter(self.models.keys()), None)
            if not model_name:
                raise ValueError("No models configured")
        
        return model_name, self.get_model(model_name)
    
    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def register_with_litellm(self) -> None:
        """
        Register all models with LiteLLM for cost tracking.
        
        Note: api_base and api_key are passed explicitly on each call,
        so this is only needed for cost/token tracking purposes.
        """
        try:
            import litellm
            
            # Build model cost dict in the format LiteLLM expects:
            # {model_name: {"max_tokens": ..., "litellm_provider": ...}}
            model_cost = {}
            for model_name, model_info in self.models.items():
                model_cost[model_name] = {
                    "max_tokens": model_info.max_tokens,
                    "litellm_provider": model_info.provider,
                    "mode": "chat",
                }
            
            if model_cost:
                litellm.register_model(model_cost)
                logger.debug(f"Registered {len(model_cost)} models with LiteLLM")
                
        except ImportError:
            logger.warning("LiteLLM not available, skipping registration")
        except Exception as e:
            logger.warning(f"Failed to register models with LiteLLM: {e}")


# Global registry instance
_registry: Optional[ModelsRegistry] = None


def get_models_registry(reload: bool = False) -> ModelsRegistry:
    """
    Get or create the global models registry.
    
    Args:
        reload: Force reload from config file
    """
    global _registry
    if _registry is None or reload:
        _registry = ModelsRegistry()
    return _registry


def init_models_registry(config_path: Optional[Path] = None) -> ModelsRegistry:
    """Initialize models registry from a specific config file."""
    global _registry
    _registry = ModelsRegistry(config_path)
    return _registry
