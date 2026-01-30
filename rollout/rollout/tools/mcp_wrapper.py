"""
Automatic MCP Server Wrapper and Registration System.

This module provides automatic wrapping and registration of MCP Server classes,
eliminating the need to manually create wrapper classes for each server.

Usage:
    # Option 1: Auto-discover and register all MCP servers
    from rollout.tools.mcp_wrapper import discover_mcp_servers
    discover_mcp_servers()
    
    # Option 2: Manually wrap a specific server
    from rollout.tools.mcp_wrapper import create_mcp_tool
    ToolClass = create_mcp_tool("StudentAcademicManagement", domain_name="...")
    tool = ToolClass()
"""

import os
import sys
import json
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Type, List

from rollout.tools.base import BaseTool
from rollout.tools import register_tool, _registered_tools
from rollout.utils.domain_resolver import resolve_domain_name

logger = logging.getLogger(__name__)

# Default paths (can be overridden)
# These paths are relative to the project structure
_ROLLOUT_PKG_DIR = Path(__file__).parent.parent  # rollout/ package directory
DEFAULT_MCP_SERVERS_DIR = _ROLLOUT_PKG_DIR / "tools" / "datasets" / "single_domain" / "mcp_servers"
DEFAULT_TOOL_LISTS_DIR = _ROLLOUT_PKG_DIR / "tools" / "datasets" / "single_domain" / "tool_lists"

# Global mapping for server name abbreviations
_server_name_to_abbrev: Dict[str, str] = {}
_abbrev_to_server_name: Dict[str, str] = {}

def abbreviate_server_name(server_name: str, max_length: int = 8) -> str:
    """
    Abbreviate a CamelCase server name to a short identifier.
    
    Strategy: Take first letter of each word.
    
    Examples:
        StudentAcademicPortal -> sap
        PatientHealthcareManagement -> phm
        CaregiverChildcareManagement -> ccm
    
    Args:
        server_name: CamelCase server name
        max_length: Maximum length for abbreviation (default: 8)
        
    Returns:
        Abbreviated identifier (lowercase)
    """
    import re
    
    # Split CamelCase into words
    words = re.findall('[A-Z][a-z]*', server_name)
    
    if not words:
        # Fallback: use first few characters
        abbrev = server_name[:max_length].lower()
    else:
        # Strategy: First letter of each word
        abbrev = ''.join(w[0] for w in words).lower()
        
        # If still too long, use first 2 letters of first and last word
        if len(abbrev) > max_length and len(words) > 2:
            abbrev = (words[0][:2] + words[-1][:2]).lower()
            # Add middle word initials if space allows
            if len(words) > 2 and len(abbrev) < max_length:
                for w in words[1:-1]:
                    if len(abbrev) < max_length:
                        abbrev += w[0].lower()
                    else:
                        break
    
    # Handle conflicts: if abbreviation already exists for different server
    if abbrev in _abbrev_to_server_name:
        existing_server = _abbrev_to_server_name[abbrev]
        if existing_server != server_name:
            # Conflict detected - append number
            counter = 1
            original_abbrev = abbrev
            while abbrev in _abbrev_to_server_name:
                abbrev = f"{original_abbrev}{counter}"
                counter += 1
            logger.warning(
                f"Abbreviation conflict: {server_name} and {existing_server} "
                f"both map to '{original_abbrev}'. Using '{abbrev}' for {server_name}."
            )
    
    # Store mappings
    _server_name_to_abbrev[server_name] = abbrev
    _abbrev_to_server_name[abbrev] = server_name
    
    return abbrev


def get_server_name_from_abbrev(abbrev: str) -> Optional[str]:
    """
    Get the full server name from an abbreviation.
    
    Args:
        abbrev: Abbreviated identifier
        
    Returns:
        Full server name, or None if not found
    """
    return _abbrev_to_server_name.get(abbrev)


def get_abbrev_from_server_name(server_name: str) -> Optional[str]:
    """
    Get the abbreviation for a server name.
    
    Args:
        server_name: Full server name
        
    Returns:
        Abbreviated identifier, or None if not found
    """
    return _server_name_to_abbrev.get(server_name)

# Alternative: check workspace root
def _get_default_paths():
    """Get default paths, checking multiple possible locations."""
    possible_roots = [
        _ROLLOUT_PKG_DIR,  # rollout/ package
        _ROLLOUT_PKG_DIR.parent,  # project root
        Path.cwd(),  # current working directory
    ]
    
    for root in possible_roots:
        mcp_dir = root / "rollout" / "tools" / "datasets" / "single_domain" / "mcp_servers"
        if mcp_dir.exists():
            return (
                mcp_dir,
                root / "rollout" / "tools" / "datasets" / "single_domain" / "tool_lists"
            )
        # Also check without rollout/ prefix
        mcp_dir = root / "tools" / "datasets" / "single_domain" / "mcp_servers"
        if mcp_dir.exists():
            return (
                mcp_dir,
                root / "tools" / "datasets" / "single_domain" / "tool_lists"
            )
    
    return DEFAULT_MCP_SERVERS_DIR, DEFAULT_TOOL_LISTS_DIR


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def load_server_class(
    server_name: str,
    mcp_servers_dir: Optional[Path] = None
) -> Optional[Type]:
    """
    Dynamically load an MCP Server class from the mcp_servers directory.
    
    Args:
        server_name: Name of the server (e.g., "StudentAcademicManagement")
        mcp_servers_dir: Path to the mcp_servers directory
        
    Returns:
        The Server class, or None if not found
    """
    if mcp_servers_dir is None:
        mcp_servers_dir, _ = _get_default_paths()
    mcp_servers_dir = Path(mcp_servers_dir)
    
    server_file = mcp_servers_dir / f"{server_name}.py"
    if not server_file.exists():
        print(f"Server file not found: {server_file}")
        return None
    
    # Load the module dynamically
    module_name = f"mcp_servers.{server_name}"
    spec = importlib.util.spec_from_file_location(module_name, server_file)
    if spec is None or spec.loader is None:
        print(f"Failed to create spec for {server_file}")
        return None
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Failed to load module {server_name}: {e}")
        return None
    
    # Find the Server class (convention: {ServerName}Server)
    server_class_name = f"{server_name}Server"
    server_class = getattr(module, server_class_name, None)
    
    if server_class is None:
        print(f"Server class {server_class_name} not found in {server_file}")
        return None
    
    return server_class


def load_tool_list(
    server_name: str,
    tool_lists_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Load the tool list JSON for a server.
    
    Args:
        server_name: Name of the server
        tool_lists_dir: Path to the tool_lists directory
        
    Returns:
        List of tool definitions
    """
    if tool_lists_dir is None:
        _, tool_lists_dir = _get_default_paths()
    tool_lists_dir = Path(tool_lists_dir)
    
    tool_list_file = tool_lists_dir / f"{server_name}.json"
    if not tool_list_file.exists():
        logger.warning(f"Tool list not found: {tool_list_file}")
        return []
    
    try:
        with open(tool_list_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load tool list {tool_list_file}: {e}")
        return []


def create_mcp_tool(
    server_name: str,
    domain_name: str,
    tool_type: Optional[str] = None,
    mcp_servers_dir: Optional[Path] = None,
    tool_lists_dir: Optional[Path] = None,
    auto_register: bool = True
) -> Optional[Type[BaseTool]]:
    """
    Create a BaseTool wrapper class for an MCP Server.
    
    Args:
        server_name: Name of the MCP server (e.g., "StudentAcademicManagement")
        domain_name: Domain name for server initialization
        tool_type: Custom tool type name (default: snake_case of server_name)
        mcp_servers_dir: Path to mcp_servers directory
        tool_lists_dir: Path to tool_lists directory
        auto_register: Whether to automatically register the tool
        
    Returns:
        The created tool class, or None on failure
        
    Example:
        >>> ToolClass = create_mcp_tool(
        ...     "StudentAcademicManagement",
        ...     domain_name="StudentAcademicManagement_StudentAcademicPortal"
        ... )
        >>> tool = ToolClass()
        >>> print(tool.info)
    """
    # Load the server class
    ServerClass = load_server_class(server_name, mcp_servers_dir)

    print(f"Server Class: {ServerClass}")

    if ServerClass is None:
        print(f"Server Class not found for {server_name}")
        return None
    
    # Load the tool list
    tool_list = load_tool_list(server_name, tool_lists_dir)
    
    # Determine tool type
    _tool_type = tool_type or abbreviate_server_name(server_name)

    print(f"Tool list: {tool_list}")
    print(f"Tool type: {_tool_type}")
    
    # Create the wrapper class dynamically
    class MCPToolWrapper(BaseTool):
        """Dynamically generated wrapper for MCP Server."""
        
        # Class attributes
        tool_type = _tool_type
        _server_class = ServerClass
        _server_name = server_name
        _domain_name = domain_name
        _tool_list = tool_list
        _tool_lists_dir = tool_lists_dir or DEFAULT_TOOL_LISTS_DIR
        
        def __init__(self, num_workers: int = 1):
            # Initialize the server instance
            self._server = self._server_class(domain_name=self._domain_name)
            super().__init__(num_workers=num_workers)
        
        @property
        def info(self) -> List[Dict[str, Any]]:
            """Return tool definitions with namespaced function names."""
            # Reload tool list to ensure fresh data
            tool_list_file = self._tool_lists_dir / f"{self._server_name}.json"
            try:
                with open(tool_list_file, "r") as f:
                    tool_list = json.load(f)
            except:
                print(f"Failed to load tool list {tool_list_file}")
                tool_list = self._tool_list
            
            # Add namespace prefix to function names
            namespaced_tools = []
            for tool in tool_list:
                tool_copy = json.loads(json.dumps(tool))  # Deep copy
                original_name = tool_copy['function']['name']
                tool_copy['function']['name'] = f"{self.tool_type}.{original_name}"
                namespaced_tools.append(tool_copy)
            
            return namespaced_tools
        
        def execute(self, tool_call_str: str, sample_id: str = "", **kwargs) -> str:
            """Execute a tool call on the MCP server."""
            try:
                # Parse the tool call
                tool_call = json.loads(tool_call_str)
                tool_name = tool_call.get("name", "")
                arguments_str = tool_call.get("arguments", "{}")
                
                # Parse arguments if string
                if isinstance(arguments_str, str):
                    arguments = json.loads(arguments_str)
                else:
                    arguments = arguments_str
                
                # Remove namespace prefix if present
                if "." in tool_name:
                    tool_name = tool_name.split(".")[-1]
                
                # Invoke the server
                result = self._server.invoke(
                    session_id=sample_id or "default",
                    tool_name=tool_name,
                    **arguments
                )
                
                # Format result
                if isinstance(result, dict):
                    return json.dumps(result, ensure_ascii=False)
                return str(result)
                
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid JSON: {e}"})
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        def __repr__(self) -> str:
            return f"MCPTool({self._server_name}, tool_type={self.tool_type!r})"
    
    # Set a unique class name
    MCPToolWrapper.__name__ = f"{server_name}Tool"
    MCPToolWrapper.__qualname__ = f"{server_name}Tool"
    
    # Register the tool
    if auto_register:
        _registered_tools[_tool_type] = MCPToolWrapper
        logger.debug(f"Registered MCP tool: {_tool_type}")
    
    return MCPToolWrapper


def discover_mcp_servers(
    domain_name: str,
    mcp_servers_dir: Optional[Path] = None,
    tool_lists_dir: Optional[Path] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, Type[BaseTool]]:
    """
    Discover and register all MCP servers in a directory.
    
    Each MCP Server file is separate, but they all need the full combined domain_name
    to load the correct database. For example, if you have a cross-domain combination
    "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices", all three
    server instances (StudentAcademicPortal, StudentFinancialServices, StudentHealthServices)
    should use this full domain_name when initializing, so they can load data from the
    correct database directory.
    
    Args:
        domain_name: Full cross-domain combination name (e.g., "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices")
                    This is used by each server to load the correct database
        mcp_servers_dir: Path to mcp_servers directory
        tool_lists_dir: Path to tool_lists directory
        include: Only include these server names (if specified)
        exclude: Exclude these server names
        
    Returns:
        Dictionary mapping tool_type to tool class
        
    Example:
        >>> tools = discover_mcp_servers(
        ...     domain_name="StudentAcademicPortal_StudentFinancialServices_StudentHealthServices",
        ...     include=["StudentAcademicPortal", "StudentFinancialServices", "StudentHealthServices"]
        ... )
        >>> # All servers use the full domain_name to load the correct database
        >>> print(tools.keys())
    """
    mcp_servers_dir = Path(mcp_servers_dir or DEFAULT_MCP_SERVERS_DIR)
    tool_lists_dir = Path(tool_lists_dir or DEFAULT_TOOL_LISTS_DIR)
    exclude = exclude or []
    
    discovered = {}
    
    if not mcp_servers_dir.exists():
        logger.warning(f"MCP servers directory not found: {mcp_servers_dir}")
        return discovered
    
    # Scan for server files

    print(f"MCP Server dir: {mcp_servers_dir}")
    print(f"Include: {include}")
    print(f"Exclude: {exclude}")

    for server_file in mcp_servers_dir.glob("*.py"):
        if server_file.name.startswith("_"):
            continue
        
        server_name = server_file.stem
        
        # Apply include/exclude filters
        if include and server_name not in include:
            continue

        if server_name in exclude:    
            continue
        
        # Each MCP Server file is separate, but they all need the full combined domain_name
        # to load the correct database (e.g., "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices")
        # The domain_name parameter should be the full cross-domain combination name
        # If not provided, fall back to server_name (for single-domain scenarios)
        server_domain_name = domain_name if domain_name else server_name
        
        # Resolve the correct domain name by trying all permutations
        # The order in domain_name might not match the actual directory name
        if domain_name and "_" in domain_name:
            # Determine database base path
            db_base_path = mcp_servers_dir.parent / "database" / "outputs"
            resolved_domain = resolve_domain_name(
                server_domain_name,
                base_path=db_base_path,
                check_entities=True,
                check_relationships=True
            )
            if resolved_domain != server_domain_name:
                print(f"Resolved domain name for {server_name}: {server_domain_name} -> {resolved_domain}")
            server_domain_name = resolved_domain
        else:
            db_base_path = mcp_servers_dir.parent / "database" / "outputs"
            server_domain_name = server_name
        
        print(f"🚀 Server Name: {server_name}  |  Domain Name: {domain_name}")

        # Create and register the tool
        tool_class = create_mcp_tool(
            server_name=server_name,
            domain_name=domain_name,
            mcp_servers_dir=mcp_servers_dir,
            tool_lists_dir=tool_lists_dir,
            auto_register=False
        )
        
        if tool_class:
            # Use abbreviation as tool_type
            tool_type = abbreviate_server_name(server_name)
            discovered[tool_type] = tool_class
            print(f"Discovered MCP server: {server_name} -> {tool_type} (using domain: {server_domain_name} for database)")
    
    logger.info(f"Discovered {len(discovered)} MCP servers")
    return discovered


def get_mcp_tool(tool_type: str) -> Optional[Type[BaseTool]]:
    """
    Get a registered MCP tool by type.
    
    Args:
        tool_type: The tool type (snake_case server name)
        
    Returns:
        The tool class, or None if not found
    """
    return _registered_tools.get(tool_type)


# Convenience function for quick setup
def setup_mcp_tools(
    domain_name: str,
    tool_names: List[str],
    mcp_servers_dir: Optional[str] = None,
    tool_lists_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick setup function to discover and instantiate MCP tools.
    
    Each MCP Server file is separate, but they all need the full combined domain_name
    to load the correct database. For example, for cross-domain combination
    "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices", all three
    server instances should use this full domain_name when initializing.
    
    Args:
        domain_name: Full cross-domain combination name (e.g., "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices")
                    This is used by each server to load the correct database from database/outputs/entities/{domain_name}/
        tool_names: List of server names to include (CamelCase, e.g., "StudentAcademicPortal")
        mcp_servers_dir: Optional custom path to mcp_servers
        tool_lists_dir: Optional custom path to tool_lists
        
    Returns:
        Dictionary of instantiated tool objects (keyed by snake_case tool_type)
        
    Example:
        >>> tools = setup_mcp_tools(
        ...     domain_name="StudentAcademicPortal_StudentFinancialServices_StudentHealthServices",
        ...     tool_names=[
        ...         "StudentAcademicPortal",
        ...         "StudentFinancialServices",
        ...         "StudentHealthServices"
        ...     ]
        ... )
        >>> # All servers use the full domain_name to load the correct database
        >>> # tools is ready to use with Pipeline
    """
    if not tool_names:
        raise ValueError("tool_names is required")
    
    if not domain_name:
        raise ValueError("domain_name is required - this is the full cross-domain combination name")
    
    mcp_servers_dir = Path(mcp_servers_dir) if mcp_servers_dir else DEFAULT_MCP_SERVERS_DIR
    tool_lists_dir = Path(tool_lists_dir) if tool_lists_dir else DEFAULT_TOOL_LISTS_DIR
    
    # Discover and register
    # All servers use the full domain_name to load the correct database
    tool_classes = discover_mcp_servers(
        domain_name=domain_name,
        mcp_servers_dir=mcp_servers_dir,
        tool_lists_dir=tool_lists_dir,
        include=tool_names
    )

    print(f"Tool Classes: {tool_classes}")
    
    # Instantiate
    instances = {}
    for name in tool_names:
        # Use abbreviation as tool_type
        abbrev = abbreviate_server_name(name)
        # Also try snake_case for backward compatibility
        tool_type_snake = camel_to_snake(name)
        tool_class = tool_classes.get(abbrev) or tool_classes.get(tool_type_snake)
        if tool_class:
            instances[abbrev] = tool_class()
            logger.debug(f"Instantiated {name} -> {abbrev} with domain_name: {domain_name}")
    
    return instances


__all__ = [
    "create_mcp_tool",
    "discover_mcp_servers",
    "get_mcp_tool",
    "setup_mcp_tools",
    "load_server_class",
    "load_tool_list",
    "abbreviate_server_name",
    "get_server_name_from_abbrev",
    "get_abbrev_from_server_name",
]

