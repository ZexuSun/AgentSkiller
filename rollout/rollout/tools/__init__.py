"""
Automatic tool discovery and registration system.

Tools are automatically discovered and registered when:
1. They are in the `tools/` directory
2. They inherit from `BaseTool`
3. They have the `@register_tool` decorator

Usage:
    # Discover all tools in the tools directory
    >>> from rollout.tools import discover_tools, get_tool
    >>> tools = discover_tools()
    >>> print(tools.keys())  # ['BingSearch', 'WikipediaTool', ...]
    
    # Get a specific tool class
    >>> WikiTool = get_tool('WikipediaTool')
    >>> tool = WikiTool()
"""

import os
import sys
import pkgutil
import importlib
import logging
from pathlib import Path
from typing import Dict, Type, Optional, List, Any

logger = logging.getLogger(__name__)

# Global registry for all discovered tools
_registered_tools: Dict[str, Type] = {}

# Track if discovery has been run
_discovery_complete = False


def register_tool(cls):
    """
    Decorator to register a tool class.
    
    The class is registered using its `tool_type` attribute if available,
    otherwise the class name is used.
    
    Example:
        >>> @register_tool
        ... class MyTool(BaseTool):
        ...     tool_type = "my_tool"
        ...     
        ...     @property
        ...     def info(self):
        ...         return {"type": "function", ...}
        ...     
        ...     def execute(self, **kwargs):
        ...         return "result"
    """
    tool_type = getattr(cls, 'tool_type', cls.__name__)
    _registered_tools[tool_type] = cls
    logger.debug(f"Registered tool: {tool_type}")
    return cls


def discover_tools(
    tools_dir: Optional[str] = None,
    force_rediscover: bool = False
) -> Dict[str, Type]:
    """
    Automatically discover and import all tool modules.
    
    Scans the tools directory for Python files and imports them,
    triggering the @register_tool decorator for each tool class.
    
    Args:
        tools_dir: Path to the tools directory. If None, uses the 
                   default 'tools/' directory relative to project root.
        force_rediscover: If True, re-scan even if discovery was already done.
    
    Returns:
        Dictionary mapping tool names to tool classes.
        
    Example:
        >>> tools = discover_tools()
        >>> BingSearch = tools['BingSearch']
        >>> search_tool = BingSearch()
    """
    global _discovery_complete
    
    if _discovery_complete and not force_rediscover:
        return _registered_tools.copy()
    
    if tools_dir is None:
        # Find the project root tools directory
        # Look for 'tools' directory relative to the package or cwd
        possible_paths = [
            Path.cwd() / "tools",
            Path(__file__).parent.parent.parent / "tools",
        ]
        for path in possible_paths:
            if path.exists() and path.is_dir():
                tools_dir = str(path)
                break
        else:
            logger.warning("Could not find tools directory")
            return _registered_tools.copy()
    
    tools_path = Path(tools_dir)
    if not tools_path.exists():
        logger.warning(f"Tools directory not found: {tools_dir}")
        return _registered_tools.copy()
    
    # Add tools directory to Python path if not already there
    tools_parent = str(tools_path.parent)
    if tools_parent not in sys.path:
        sys.path.insert(0, tools_parent)
    
    # Scan for Python files
    for item in tools_path.iterdir():
        if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
            module_name = f"tools.{item.stem}"
            try:
                importlib.import_module(module_name)
                logger.debug(f"Imported tool module: {module_name}")
            except ImportError as e:
                logger.warning(f"Failed to import {module_name}: {e}")
            except Exception as e:
                logger.error(f"Error loading {module_name}: {e}")
    
    # Also scan subdirectories (for tool packages)
    for item in tools_path.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            init_file = item / "__init__.py"
            if init_file.exists():
                module_name = f"tools.{item.name}"
                try:
                    importlib.import_module(module_name)
                    logger.debug(f"Imported tool package: {module_name}")
                except ImportError as e:
                    logger.warning(f"Failed to import {module_name}: {e}")
                except Exception as e:
                    logger.error(f"Error loading {module_name}: {e}")
    
    _discovery_complete = True
    logger.info(f"Discovered {len(_registered_tools)} tools")
    return _registered_tools.copy()


def get_tool(tool_name: str, auto_discover: bool = True) -> Optional[Type]:
    """
    Get a tool class by name.
    
    Args:
        tool_name: The name of the tool (tool_type or class name)
        auto_discover: If True, run discovery if tool not found
        
    Returns:
        The tool class, or None if not found.
        
    Example:
        >>> BingSearch = get_tool('BingSearch')
        >>> tool = BingSearch()
    """
    if tool_name in _registered_tools:
        return _registered_tools[tool_name]
    
    if auto_discover:
        discover_tools()
        return _registered_tools.get(tool_name)
    
    return None


def get_registered_tools() -> Dict[str, Type]:
    """
    Get all registered tools without triggering discovery.
    
    Returns:
        Dictionary mapping tool names to tool classes.
    """
    return _registered_tools.copy()


def list_tool_names() -> List[str]:
    """
    List all registered tool names.
    
    Returns:
        List of tool names.
    """
    return list(_registered_tools.keys())


def clear_registry():
    """Clear the tool registry. Mainly for testing."""
    global _discovery_complete
    _registered_tools.clear()
    _discovery_complete = False


def instantiate_tools(
    tool_names: List[str],
    auto_discover: bool = True,
    **init_kwargs
) -> Dict[str, Any]:
    """
    Instantiate multiple tools by name.
    
    Args:
        tool_names: List of tool names to instantiate
        auto_discover: If True, run discovery if needed
        **init_kwargs: Keyword arguments passed to tool constructors
        
    Returns:
        Dictionary mapping tool names to tool instances.
        
    Example:
        >>> tools = instantiate_tools(['BingSearch', 'Wikipedia'])
        >>> result = tools['BingSearch'].execute(query="test")
    """
    if auto_discover:
        discover_tools()
    
    instances = {}
    for name in tool_names:
        tool_cls = _registered_tools.get(name)
        if tool_cls:
            try:
                instances[name] = tool_cls(**init_kwargs)
            except Exception as e:
                logger.error(f"Failed to instantiate {name}: {e}")
        else:
            logger.warning(f"Tool not found: {name}")
    
    return instances


def collect_tools_info(tool_instances: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect tool information from instantiated tools.
    
    Args:
        tool_instances: Dictionary of tool instances
        
    Returns:
        List of tool info dictionaries (for LLM tool parameter).
    """
    tools_info = []
    for name, tool in tool_instances.items():
        try:
            info = tool.info
            if isinstance(info, list):
                tools_info.extend(info)
            elif isinstance(info, dict):
                tools_info.append(info)
        except Exception as e:
            logger.error(f"Failed to get info for {name}: {e}")
    
    return tools_info


# Re-export from base
from rollout.tools.base import BaseTool

# MCP Server wrapper utilities
from rollout.tools.mcp_wrapper import (
    create_mcp_tool,
    discover_mcp_servers,
    setup_mcp_tools,
    get_mcp_tool,
)

__all__ = [
    # Core tool utilities
    "register_tool",
    "discover_tools",
    "get_tool",
    "get_registered_tools",
    "list_tool_names",
    "instantiate_tools",
    "collect_tools_info",
    "BaseTool",
    # MCP Server utilities
    "create_mcp_tool",
    "discover_mcp_servers",
    "setup_mcp_tools",
    "get_mcp_tool",
]

