"""
Configuration management for the Rollout framework.

Provides typed configuration classes and loading from YAML files.
Supports custom model registration for internal/self-hosted APIs.
Supports auto-discovery of cross-domain combinations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from rollout.core.models import register_models_from_config


def _get_cross_domain_manager():
    """Lazy import CrossDomainManager to avoid circular imports."""
    from rollout.utils.cross_domain import CrossDomainManager
    return CrossDomainManager


@dataclass
class AgentConfig:
    """Configuration for the Agent."""
    model: str = "deepseek/deepseek-chat"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None  # If None, uses env var
    api_base: Optional[str] = None
    system_prompt_file: Optional[str] = None
    enable_thinking: bool = False  # Enable thinking/reasoning mode for DeepSeek V3.2+
    
    @property
    def system_prompt(self) -> Optional[str]:
        if self.system_prompt_file and os.path.exists(self.system_prompt_file):
            with open(self.system_prompt_file, "r") as f:
                return f.read()
        return None


@dataclass
class UserConfig:
    """Configuration for the Simulated User."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    custom_guidelines: Optional[str] = None


@dataclass
class DatasetConfig:
    """Configuration for a dataset to process."""
    name: str
    path: str
    output_path: str
    tools: List[str]
    agent: AgentConfig = field(default_factory=AgentConfig)
    user: Optional[UserConfig] = None
    max_turns: int = 1
    max_steps_per_turn: int = 10
    mode: str = "positive"  # positive, wrong_tool, no_tool
    
    # MCP Server configuration (for mock MCP tools)
    mcp_domain: Optional[str] = None  # Domain name for MCP servers (e.g., "StudentAcademicManagement_StudentAcademicPortal")


@dataclass
class RolloutConfig:
    """Main configuration for the Rollout framework."""
    # Execution settings
    max_workers: int = 4
    resume: bool = True
    use_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Verbose output (real-time printing)
    verbose: bool = False
    verbose_colors: bool = True
    verbose_max_length: int = 500
    
    # Conversation Monitor (early termination detection)
    enable_monitor: bool = True                # Enable conversation monitoring
    monitor_rule_detection: bool = False       # Use rule-based pattern detection (prone to false positives)
    monitor_max_no_tool_turns: int = 2         # Max turns without tool calls before stopping
    monitor_use_llm_judge: bool = False        # Use LLM to confirm termination
    monitor_judge_model: str = "gpt-4o-mini"   # Model for LLM judge
    
    # Custom models configuration
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models_config_file: Optional[str] = None  # Path to separate models.yml
    
    # Datasets to process
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "RolloutConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RolloutConfig":
        """Create configuration from a dictionary."""
        # Register custom models if defined
        models_config = data.get("models", {})
        if models_config:
            _resolve_env_vars(models_config)
            register_models_from_config(models_config)
        
        # Load models from external file if specified
        models_file = data.get("models_config_file")
        if models_file and os.path.exists(models_file):
            with open(models_file, "r") as f:
                external_models = yaml.safe_load(f)
            if external_models:
                ext_models = external_models.get("models", external_models)
                _resolve_env_vars(ext_models)
                register_models_from_config(ext_models)
        
        datasets = {}
        
        # Check for auto-discovery mode
        auto_discover = data.get("auto_discover")
        if auto_discover:
            datasets = cls._auto_discover_datasets(auto_discover, data)
        
        # Also process manually defined datasets
        for name, ds_data in data.get("datasets", {}).items():
            # Parse agent config
            agent_data = ds_data.get("agent", {})
            if isinstance(agent_data, str):
                # Simple model name
                agent_config = AgentConfig(model=agent_data)
            else:
                agent_config = AgentConfig(
                    model=agent_data.get("model", "deepseek/deepseek-chat"),
                    temperature=agent_data.get("temperature", 0.7),
                    max_tokens=agent_data.get("max_tokens"),
                    api_key=agent_data.get("api_key"),
                    api_base=agent_data.get("api_base"),
                    system_prompt_file=agent_data.get("system_prompt_file"),
                    enable_thinking=agent_data.get("enable_thinking", False)
                )
            
            # Parse user config (optional)
            user_data = ds_data.get("user")
            user_config = None
            if user_data:
                if isinstance(user_data, str):
                    user_config = UserConfig(model=user_data)
                else:
                    user_config = UserConfig(
                        model=user_data.get("model", "gpt-4"),
                        temperature=user_data.get("temperature", 0.7),
                        max_tokens=user_data.get("max_tokens"),
                        api_key=user_data.get("api_key"),
                        api_base=user_data.get("api_base"),
                        custom_guidelines=user_data.get("custom_guidelines")
                    )
            
            datasets[name] = DatasetConfig(
                name=name,
                path=ds_data.get("path", ""),
                output_path=ds_data.get("output_path", f"./outputs/{name}.jsonl"),
                tools=ds_data.get("tools", []),
                agent=agent_config,
                user=user_config,
                max_turns=ds_data.get("max_turns", 1),
                max_steps_per_turn=ds_data.get("max_steps_per_turn", 10),
                mode=ds_data.get("mode", "positive"),
                mcp_domain=ds_data.get("mcp_domain")
            )
        
        return cls(
            max_workers=data.get("max_workers", 4),
            resume=data.get("resume", True),
            use_checkpoints=data.get("use_checkpoints", True),
            checkpoint_dir=data.get("checkpoint_dir", "./checkpoints"),
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file"),
            verbose=data.get("verbose", False),
            verbose_colors=data.get("verbose_colors", True),
            verbose_max_length=data.get("verbose_max_length", 500),
            enable_monitor=data.get("enable_monitor", True),
            monitor_rule_detection=data.get("monitor_rule_detection", False),
            monitor_max_no_tool_turns=data.get("monitor_max_no_tool_turns", 2),
            monitor_use_llm_judge=data.get("monitor_use_llm_judge", False),
            monitor_judge_model=data.get("monitor_judge_model", "gpt-4o-mini"),
            models=models_config,
            models_config_file=models_file,
            datasets=datasets
        )


    @classmethod
    def _auto_discover_datasets(cls, auto_config: Dict[str, Any], global_config: Dict[str, Any]) -> Dict[str, "DatasetConfig"]:
        """
        Auto-discover cross-domain combinations and create DatasetConfig for each.
        
        auto_config supports:
            base_path: str - Base path for cross-domain data (default: rollout/tools/datasets/cross_domain)
            output_dir: str - Output directory for results (default: ./outputs)
            min_domains: int - Minimum domains to be cross-domain (default: 2)
            require_policy: bool - Only include combos with policy files (default: false)
            agent_model: str - Model for agent (inherits from global if not set)
            user_model: str - Model for user simulator (inherits from global if not set)
            max_turns: int - Max conversation turns (default: 20)
            filter_domains: list - Only include combos containing these domains
            exclude_domains: list - Exclude combos containing these domains
        """
        CrossDomainManager = _get_cross_domain_manager()
        
        base_path = auto_config.get("base_path", "rollout/tools/datasets/cross_domain")
        output_dir = auto_config.get("output_dir", "./outputs")
        min_domains = auto_config.get("min_domains", 2)
        require_policy = auto_config.get("require_policy", False)
        
        # Model settings (inherit from global config if not specified)
        agent_model = auto_config.get("agent_model", global_config.get("agent_model", "openai/deepseek-v3.2"))
        user_model = auto_config.get("user_model", global_config.get("user_model", "openai/gpt-5"))
        agent_temperature = auto_config.get("agent_temperature", 0.7)
        user_temperature = auto_config.get("user_temperature", 1.0)
        enable_thinking = auto_config.get("enable_thinking", True)
        
        max_turns = auto_config.get("max_turns", 20)
        max_steps_per_turn = auto_config.get("max_steps_per_turn", 10)
        mode = auto_config.get("mode", "positive")
        
        filter_domains = set(auto_config.get("filter_domains", []))
        exclude_domains = set(auto_config.get("exclude_domains", []))
        
        manager = CrossDomainManager(base_path)
        combos = manager.discover_combinations(min_domains=min_domains)
        
        datasets = {}
        
        for combo in combos:
            # Apply filters
            if require_policy and not combo.policy_file:
                continue
            
            if filter_domains and not filter_domains.intersection(combo.domain_set):
                continue
            
            if exclude_domains and exclude_domains.intersection(combo.domain_set):
                continue
            
            # Skip if no query file
            if not combo.query_file or not combo.query_file.exists():
                continue
            
            # Create agent config
            agent_config = AgentConfig(
                model=agent_model,
                temperature=agent_temperature,
                enable_thinking=enable_thinking,
                system_prompt_file=str(combo.policy_file) if combo.policy_file else None
            )
            
            # Create user config
            user_config = UserConfig(
                model=user_model,
                temperature=user_temperature
            )
            
            # Create dataset config
            datasets[combo.name] = DatasetConfig(
                name=combo.name,
                path=str(combo.query_file),
                output_path=f"{output_dir}/{combo.name}_output.jsonl",
                tools=combo.tools,
                agent=agent_config,
                user=user_config,
                max_turns=max_turns,
                max_steps_per_turn=max_steps_per_turn,
                mode=mode,
                mcp_domain=combo.name
            )
        
        return datasets


def _resolve_env_vars(config: Dict[str, Any]):
    """Resolve environment variables in config values (${VAR_NAME} syntax)."""
    import re
    env_pattern = re.compile(r'\$\{(\w+)\}')
    
    def resolve_value(value):
        if isinstance(value, str):
            match = env_pattern.search(value)
            if match:
                env_var = match.group(1)
                env_value = os.environ.get(env_var, "")
                return env_pattern.sub(env_value, value)
        return value
    
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_env_vars(value)
        else:
            config[key] = resolve_value(value)