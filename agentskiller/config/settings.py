"""
Configuration management using Pydantic Settings.
Supports environment variables and YAML config files.
"""

import os
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    # Default models (if None, uses models.yaml defaults section)
    textual_model: Optional[str] = None
    coding_model: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 3
    timeout: int = 120
    retry_delay: float = 2.0
    retry_backoff: float = 2.0
    
    # Auto-continue for truncated responses
    auto_continue: bool = True
    max_continuations: int = 5


class PathConfig(BaseModel):
    """Path configuration for inputs and outputs."""
    
    workspace: Path = Path(".")
    outputs_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")
    checkpoints_dir: Path = Path("checkpoints")
    
    # Derived paths
    @property
    def domain_topics_json(self) -> Path:
        return self.outputs_dir / "domain_topics.json"
    
    @property
    def entities_json(self) -> Path:
        return self.outputs_dir / "entities.json"
    
    @property
    def entity_graph_json(self) -> Path:
        return self.outputs_dir / "entity_graph.json"
    
    @property
    def blueprints_json(self) -> Path:
        return self.outputs_dir / "blueprints.json"
    
    @property
    def fixed_blueprints_json(self) -> Path:
        return self.outputs_dir / "fixed_blueprints.json"
    
    @property
    def database_dir(self) -> Path:
        return self.outputs_dir / "database"
    
    @property
    def mcp_servers_dir(self) -> Path:
        return self.outputs_dir / "mcp_servers"
    
    @property
    def policies_dir(self) -> Path:
        return self.outputs_dir / "policies"
    
    @property
    def tool_graphs_dir(self) -> Path:
        return self.outputs_dir / "tool_graphs"
    
    @property
    def task_templates_dir(self) -> Path:
        return self.outputs_dir / "task_templates"
    
    @property
    def cross_domain_templates_dir(self) -> Path:
        return self.outputs_dir / "cross_domain_templates"
    
    @property
    def instantiated_tasks_dir(self) -> Path:
        return self.outputs_dir / "instantiated_tasks"
    
    @property
    def validated_tasks_dir(self) -> Path:
        return self.outputs_dir / "validated_tasks"
    
    @property
    def queries_dir(self) -> Path:
        return self.outputs_dir / "queries"


class WorkflowConfig(BaseModel):
    """Workflow execution configuration."""
    
    # Parallel processing
    max_workers: int = 8
    chunk_size: int = 30
    
    # Retry mechanism
    auto_retry: bool = True
    max_step_retries: int = 3
    
    # Checkpointing
    enable_checkpoints: bool = True
    
    # Execution modes
    dry_run: bool = False
    verbose: bool = False
    
    # Task generation mode: "single", "cross_domain", "both"
    task_mode: Literal["single", "cross_domain", "both"] = "both"
    
    # Number of cross-domain combinations to generate
    cross_domain_combinations: int = 10
    
    # Simulation time (used in prompts)
    simulation_time: str = "2025-01-23 15:00:00 EST"


class BaseStepConfig(BaseModel):
    """Base configuration for a workflow step."""
    enabled: bool = True
    model: Optional[str] = None  # Override default model for this step


class StepConfig(BaseModel):
    """Configuration for individual workflow steps.
    
    Each step can have:
    - enabled: Whether to run this step
    - model: Override LLM model for this step (uses default if None)
    - Step-specific parameters
    """
    
    # Step 1: Domain Expansion
    # Uses: textual model for generating new domain names
    s01_domain_expansion: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "target_domain_number": 300,
        "expansion_batch_size": 20,
        "diversity_check_batch": 50,
    })
    
    # Step 2: Entity Extraction
    # Uses: textual model for extracting entities from domains
    s02_entity_extraction: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "entities_per_domain": 5,  # Target entities per domain
    })
    
    # Step 3: Entity Graph Generation
    # Uses: textual model for identifying entity relationships
    s03_entity_graph: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "relation_batch_size": 10,  # Entities per batch when checking relationships
    })
    
    # Step 4: Blueprint Generation
    # Uses: textual model for generating MCP blueprints
    s04_blueprint_generation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "target_blueprints_per_entity": 5,
        "min_functions_per_blueprint": 10,
        "max_peripheral_entities": 3,
    })
    
    # Step 5: Tool List Formulation
    # No LLM calls - pure data processing
    s05_tool_list_formulation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
    })
    
    # Step 6: Database Generation
    # Uses: coding model for generating Python data scripts
    s06_database_generation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default coding model
        "entities_per_table": 50,
        "max_fix_attempts": 5,
        "script_timeout": 60,  # Timeout for running data scripts
    })
    
    # Step 7: Policy Generation
    # Uses: textual model for generating domain policies
    s07_policy_generation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "enable_validation": True,
        "simulation_time": "2025-01-23 15:00:00 EST",
    })
    
    # Step 8: Tool Graph Generation
    # Uses: textual model for generating tool dependency graphs
    s08_tool_graph_generation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
    })
    
    # Step 9: MCP Server Implementation
    # Uses: coding model for code/tests
    s09_mcp_server_implementation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "code_model": None,  # Use default coding model
        "test_model": None,  # Use default coding model
        "max_test_retries": 3,
        "max_fix_attempts": 5,
        "simulation_time": "2025-01-23 15:00:00 EST",
    })
    
    # Step 10: Domain Combos Selection (Cross-Domain only)
    # No LLM calls - combinatorial selection
    s10_domain_combos_selection: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "min_shared_entities": 1,  # Minimum shared entities for valid combo
        "max_domains_per_combo": 3,  # Maximum domains in a combination
    })
    
    # Step 11: Trajectory Fusion (Cross-Domain only)
    # Uses: textual model for fusing trajectories
    s11_trajectory_fusion: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "trajectories_per_combo": 5,  # Target trajectories per combination
    })
    
    # Step 12: Database Fusion (Cross-Domain only)
    # No LLM calls - data merging
    s12_database_fusion: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
    })
    
    # Step 13: Policy Merge (Cross-Domain only)
    # Uses: textual model for merging policies
    s13_policy_merge: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "max_policy_length": 4000,  # Max chars of policy to include in prompt
    })
    
    # Step 14: Task Template Generation
    # Uses: textual model for generating task templates
    s14_task_template_generation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
        "batch_size": 10,  # Templates per LLM call
        "max_trajectories_per_server": 50,
        "max_policy_context": 2000,  # Max chars of policy context
    })
    
    # Step 15: Instance Combos Selection
    # Uses Plan-Execution approach with LLM
    s15_instance_combos_selection: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,
        "samples_per_template": 1,  # Entity instances per template
        "max_code_retries": 3,
        "code_timeout": 30,
        "enable_policy_filtering": True,
        "enable_plan_execution": True,
    })
    
    # Step 16: Task Filtering
    # No LLM calls - trajectory execution validation
    s16_task_filtering: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "execution_timeout": 30,  # Timeout for trajectory execution
    })
    
    # Step 17: Task Instantiation & Query Generation
    # Uses: textual model for instantiating tasks and generating startup queries
    s17_task_instantiation: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": None,  # Use default textual model
    })


class Settings(BaseSettings):
    """Main settings class that aggregates all configurations."""
    
    model_config = SettingsConfigDict(
        env_prefix="CROSSDOMAIN_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    steps: StepConfig = Field(default_factory=StepConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from a YAML file."""
        if not path.exists():
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)
    
    def save_yaml(self, path: Path) -> None:
        """Save current settings to a YAML file."""
        data = self.model_dump(mode="json")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """Get configuration for a specific step."""
        step_key = step_name.replace("-", "_")
        return getattr(self.steps, step_key, {})


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get or create the global settings instance.
    
    Args:
        reload: Force reload from config file
    """
    global _settings
    if _settings is None or reload:
        config_path = Path("config.yaml")
        _settings = Settings.from_yaml(config_path)
    return _settings


def init_settings(config_path: Optional[Path] = None) -> Settings:
    """Initialize settings from a config file (always reloads)."""
    global _settings
    if config_path:
        _settings = Settings.from_yaml(config_path)
    else:
        default_config = Path("config.yaml")
        if default_config.exists():
            _settings = Settings.from_yaml(default_config)
        else:
            _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
