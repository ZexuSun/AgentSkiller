"""
Factory for dynamically creating MCP Server tool wrappers.

Usage:
    from tools.mcp_tool_factory import create_mcp_tool, register_all_mcp_tools
    
    # Create a single tool
    MyTool = create_mcp_tool(
        domain_name="StudentAcademicPortal",
        outputs_dir="./outputs_cursor",
    )
    
    # Or register all tools at once
    register_all_mcp_tools()
"""

import json
import importlib
import importlib.util
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, Type, List

from tools.base import BaseTool, register_tool


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def load_server_class(server_path: Path, class_name: str):
    """Dynamically load a server class from a Python file."""
    module_name = f"mcp_servers.{server_path.stem}"
    
    # Check if already loaded and has the expected class
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if hasattr(module, class_name):
            return getattr(module, class_name)
        # Class not found in cached module, force reload
        del sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return getattr(module, class_name)


def create_mcp_tool(
    domain_name: str,
    outputs_dir: str = "./outputs_cursor",
    tool_type: Optional[str] = None,
    is_cross_domain: bool = False,
) -> Type[BaseTool]:
    """
    Factory function to create an MCP tool wrapper class.
    
    Args:
        domain_name: Name of the domain (e.g., "StudentAcademicPortal" or 
                     "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices")
        outputs_dir: Directory containing generated outputs.
        tool_type: Optional custom tool type. If not provided, derived from domain_name.
        is_cross_domain: Whether this is a cross-domain tool (auto-detected if _ in name).
    
    Returns:
        A dynamically created tool class that wraps the MCP server.
    
    Example:
        # Create tool for single domain
        StudentTool = create_mcp_tool("StudentAcademicPortal")
        tool = StudentTool()
        result = tool.execute('{"name": "authorize_student", "arguments": {...}}', "session_1")
        
        # Create tool for cross-domain
        CrossDomainTool = create_mcp_tool(
            "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices"
        )
    """
    outputs_path = Path(outputs_dir)
    
    # Derive tool_type from domain_name if not provided
    if tool_type is None:
        tool_type = camel_to_snake(domain_name)
    
    # Auto-detect cross-domain
    if "_" in domain_name and not is_cross_domain:
        # Could be cross-domain, check if multiple capital letters exist
        parts = domain_name.split("_")
        if len(parts) > 1 and all(p[0].isupper() for p in parts if p):
            is_cross_domain = True
    
    # Determine paths
    if is_cross_domain:
        # Cross-domain: use first domain's server for now
        domains = domain_name.split("_")
        first_domain = domains[0]
        server_path = outputs_path / "mcp_servers" / f"{first_domain}.py"
        server_class_name = f"{first_domain}Server"
        tool_list_path = outputs_path / "tool_lists" / f"{first_domain}.json"
    else:
        server_path = outputs_path / "mcp_servers" / f"{domain_name}.py"
        server_class_name = f"{domain_name}Server"
        tool_list_path = outputs_path / "tool_lists" / f"{domain_name}.json"
    
    # Capture variables in closure
    _domain_name = domain_name
    _tool_type = tool_type
    _is_cross_domain = is_cross_domain
    _server_path = server_path
    _server_class_name = server_class_name
    _tool_list_path = tool_list_path
    
    class MCPToolWrapper(BaseTool):
        """Dynamically generated MCP tool wrapper."""
        
        __domain_name = _domain_name
        __tool_type = _tool_type
        __is_cross_domain = _is_cross_domain
        __server_path = _server_path
        __server_class_name = _server_class_name
        __tool_list_path = _tool_list_path
        __server_instance = None
        
        def __init__(self):
            super().__init__()
        
        @classmethod
        def _get_server(cls):
            """Lazy initialization of server instance."""
            if cls.__server_instance is None:
                if not cls.__server_path.exists():
                    raise FileNotFoundError(f"Server file not found: {cls.__server_path}")
                
                server_class = load_server_class(cls.__server_path, cls.__server_class_name)
                cls.__server_instance = server_class(domain_name=cls.__domain_name)
            
            return cls.__server_instance
        
        @property
        def tool_type(self) -> str:
            return self.__tool_type
        
        @property
        def domain_name(self) -> str:
            return self.__domain_name
        
        @property
        def info(self) -> List[Dict]:
            """Get tool information with prefixed names."""
            try:
                with open(self.__tool_list_path) as f:
                    tool_list = json.load(f)
                
                # Prefix tool names with tool_type
                for tool in tool_list:
                    if 'function' in tool:
                        original_name = tool['function']['name']
                        if not original_name.startswith(f"{self.__tool_type}."):
                            tool['function']['name'] = f"{self.__tool_type}.{original_name}"
                
                return tool_list
            except FileNotFoundError:
                return []
        
        def execute(self, tool_call_str: str, sample_id: str, **kwargs) -> Dict[str, Any]:
            """Execute a tool call."""
            tool_call = json.loads(tool_call_str) if isinstance(tool_call_str, str) else tool_call_str
            
            # Remove the tool_type prefix from the name
            tool_name = tool_call["name"]
            if tool_name.startswith(f"{self.__tool_type}."):
                tool_name = tool_name[len(f"{self.__tool_type}."):]
            
            # Parse arguments
            arguments = tool_call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            
            # Execute on server
            server = self._get_server()
            result = server.invoke(
                session_id=sample_id,
                tool_name=tool_name,
                **arguments
            )
            
            return result
    
    # Set a meaningful class name
    MCPToolWrapper.__name__ = f"{domain_name}Tool"
    MCPToolWrapper.__qualname__ = f"{domain_name}Tool"
    
    return MCPToolWrapper


def create_and_register_mcp_tool(
    domain_name: str,
    outputs_dir: str = "./outputs_cursor",
    tool_type: Optional[str] = None,
    is_cross_domain: bool = False,
) -> Type[BaseTool]:
    """
    Create an MCP tool wrapper and register it.
    
    Same as create_mcp_tool but also registers the tool.
    """
    tool_class = create_mcp_tool(
        domain_name=domain_name,
        outputs_dir=outputs_dir,
        tool_type=tool_type,
        is_cross_domain=is_cross_domain,
    )
    
    # Register the tool
    register_tool(tool_class)
    
    return tool_class


def register_all_mcp_tools(
    outputs_dir: str = "./outputs_cursor",
    include_cross_domain: bool = True,
    verbose: bool = True,
) -> Dict[str, Type[BaseTool]]:
    """
    Automatically discover and register all MCP server tools.
    
    Args:
        outputs_dir: Directory containing generated outputs.
        include_cross_domain: Whether to include cross-domain tools.
        verbose: Whether to print registration status.
    
    Returns:
        Dictionary mapping domain names to tool classes.
    """
    outputs_path = Path(outputs_dir)
    registered_tools = {}
    
    # Register single-domain tools
    mcp_servers_dir = outputs_path / "mcp_servers"
    if mcp_servers_dir.exists():
        for server_file in mcp_servers_dir.glob("*.py"):
            if server_file.name.startswith("__"):
                continue
            
            domain_name = server_file.stem
            try:
                tool_class = create_and_register_mcp_tool(
                    domain_name=domain_name,
                    outputs_dir=outputs_dir,
                    is_cross_domain=False,
                )
                registered_tools[domain_name] = tool_class
                if verbose:
                    print(f"✓ Registered: {domain_name}")
            except Exception as e:
                if verbose:
                    print(f"✗ Failed to register {domain_name}: {e}")
    
    # Register cross-domain tools
    if include_cross_domain:
        cross_domain_dir = outputs_path / "instantiated_tasks" / "cross_domain"
        if cross_domain_dir.exists():
            for combo_dir in cross_domain_dir.iterdir():
                if combo_dir.is_dir():
                    domain_name = combo_dir.name
                    try:
                        tool_class = create_and_register_mcp_tool(
                            domain_name=domain_name,
                            outputs_dir=outputs_dir,
                            is_cross_domain=True,
                        )
                        registered_tools[domain_name] = tool_class
                        if verbose:
                            print(f"✓ Registered cross-domain: {domain_name}")
                    except Exception as e:
                        if verbose:
                            print(f"✗ Failed to register cross-domain {domain_name}: {e}")
    
    return registered_tools


# ============================================================================
# Simplified Interface for Common Use Cases
# ============================================================================

class MCPToolRegistry:
    """
    Registry for MCP tools with lazy loading.
    
    Usage:
        registry = MCPToolRegistry("./outputs_cursor")
        
        # Get a specific tool
        tool = registry.get("StudentAcademicPortal")
        
        # Execute
        result = tool.execute('{"name": "authorize_student", "arguments": {...}}', "session_123")
        
        # Or use convenience method
        result = registry.execute(
            "StudentAcademicPortal",
            {"name": "authorize_student", "arguments": {"student_id": "..."}},
            "session_123"
        )
    """
    
    def __init__(self, outputs_dir: str = "./outputs_cursor"):
        self.outputs_dir = Path(outputs_dir)
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instances: Dict[str, BaseTool] = {}
    
    def get(self, domain_name: str) -> BaseTool:
        """Get or create a tool instance for the given domain."""
        if domain_name not in self._instances:
            if domain_name not in self._tools:
                # Create the tool class
                self._tools[domain_name] = create_mcp_tool(
                    domain_name=domain_name,
                    outputs_dir=str(self.outputs_dir),
                )
            
            # Instantiate
            self._instances[domain_name] = self._tools[domain_name]()
        
        return self._instances[domain_name]
    
    def list_available(self) -> List[str]:
        """List all available domain names."""
        domains = []
        
        # Single domain servers
        mcp_dir = self.outputs_dir / "mcp_servers"
        if mcp_dir.exists():
            for f in mcp_dir.glob("*.py"):
                if not f.name.startswith("__"):
                    domains.append(f.stem)
        
        return sorted(domains)
    
    def list_cross_domain(self) -> List[str]:
        """List all cross-domain combinations."""
        domains = []
        
        cross_domain_dir = self.outputs_dir / "instantiated_tasks" / "cross_domain"
        if cross_domain_dir.exists():
            for d in cross_domain_dir.iterdir():
                if d.is_dir():
                    domains.append(d.name)
        
        return sorted(domains)
    
    def execute(
        self,
        domain_name: str,
        tool_call: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Convenience method to execute a tool call.
        
        Args:
            domain_name: The domain to execute on.
            tool_call: Dictionary with "name" and "arguments" keys.
            session_id: Session identifier.
        
        Returns:
            Result from the tool execution.
        """
        tool = self.get(domain_name)
        return tool.execute(json.dumps(tool_call), session_id)
    
    def get_tool_info(self, domain_name: str) -> List[Dict]:
        """Get tool information/schema for a domain."""
        tool = self.get(domain_name)
        return tool.info
