"""
MCP Tool Wrappers for Multi-Turn Rollout Pipeline.

Usage:
    # Option 1: Use the factory to create individual tools
    from tools.mcp_tool_factory import create_mcp_tool, create_and_register_mcp_tool
    
    StudentPortalTool = create_mcp_tool("StudentAcademicPortal")
    tool = StudentPortalTool()
    result = tool.execute('{"name": "authorize_student", "arguments": {...}}', "session_123")
    
    # Option 2: Use the registry for lazy loading
    from tools.mcp_tool_factory import MCPToolRegistry
    
    registry = MCPToolRegistry("./outputs_cursor")
    tool = registry.get("StudentAcademicPortal")
    result = tool.execute('{"name": "authorize_student", "arguments": {...}}', "session_123")
    
    # Option 3: Register all tools at once
    from tools.mcp_tool_factory import register_all_mcp_tools
    
    tools = register_all_mcp_tools("./outputs_cursor")
"""

from .base import BaseTool, register_tool, get_registered_tools, get_tool
from .mcp_tool_factory import (
    create_mcp_tool,
    create_and_register_mcp_tool,
    register_all_mcp_tools,
    MCPToolRegistry,
)

__all__ = [
    "BaseTool",
    "register_tool",
    "get_registered_tools",
    "get_tool",
    "create_mcp_tool",
    "create_and_register_mcp_tool",
    "register_all_mcp_tools",
    "MCPToolRegistry",
]

