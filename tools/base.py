"""
Base classes and registry for MCP tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type


# Global tool registry
_TOOL_REGISTRY: Dict[str, Type["BaseTool"]] = {}


def register_tool(cls: Type["BaseTool"]) -> Type["BaseTool"]:
    """
    Decorator to register a tool class.
    
    Usage:
        @register_tool
        class MyTool(BaseTool):
            tool_type = "my_tool"
            ...
    """
    if hasattr(cls, 'tool_type'):
        tool_type = cls.tool_type if isinstance(cls.tool_type, str) else cls._tool_type
        _TOOL_REGISTRY[tool_type] = cls
    return cls


def get_registered_tools() -> Dict[str, Type["BaseTool"]]:
    """Get all registered tools."""
    return _TOOL_REGISTRY.copy()


def get_tool(tool_type: str) -> Type["BaseTool"]:
    """Get a registered tool by type."""
    if tool_type not in _TOOL_REGISTRY:
        raise KeyError(f"Tool '{tool_type}' not registered. Available: {list(_TOOL_REGISTRY.keys())}")
    return _TOOL_REGISTRY[tool_type]


class BaseTool(ABC):
    """
    Base class for all MCP tool wrappers.
    
    Subclasses must implement:
        - tool_type: String identifier for the tool
        - info: Property returning tool information/schema
        - execute: Method to execute tool calls
    """
    
    @property
    @abstractmethod
    def tool_type(self) -> str:
        """Unique identifier for this tool type."""
        pass
    
    @property
    @abstractmethod
    def info(self) -> List[Dict[str, Any]]:
        """
        Get tool information/schema.
        
        Returns:
            List of tool definitions in OpenAI function format.
        """
        pass
    
    @abstractmethod
    def execute(self, tool_call_str: str, sample_id: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool call.
        
        Args:
            tool_call_str: JSON string containing the tool call with "name" and "arguments".
            sample_id: Session/sample identifier.
            **kwargs: Additional arguments.
        
        Returns:
            Result dictionary from the tool execution.
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(tool_type='{self.tool_type}')>"

