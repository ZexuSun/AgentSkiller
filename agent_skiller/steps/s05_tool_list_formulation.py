"""
Step 5: Tool List Formulation

Extract tool lists from blueprints in OpenAI Function Call format.

Input: blueprints.json
Output: tool_lists/*.json
"""

import logging
from pathlib import Path
from typing import Dict, Any

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..core.parallel import parallel_process_with_retry
from .base import step_handler, save_json, load_json, ensure_dir

logger = logging.getLogger(__name__)


@step_handler("s05_tool_list_formulation", auto_retry=True)
def tool_list_formulation_step(state: WorkflowState) -> WorkflowState:
    """
    Extract tool lists from blueprints in OpenAI Function Call format.
    
    Process:
    1. Load blueprints
    2. Extract tool lists in OpenAI format (parallel processing)
    3. Save tool lists per server
    
    Output:
    - tool_lists/*.json: OpenAI function format per server
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # Load blueprints
    blueprints_data = load_json(Path(state.blueprints_path))
    # Handle both list format and dict with "blueprints" key
    if isinstance(blueprints_data, list):
        blueprints = blueprints_data
    else:
        blueprints = blueprints_data.get("blueprints", [])
    
    logger.info(f"Processing {len(blueprints)} blueprints")
    
    tool_lists_dir = ensure_dir(outputs_dir / "tool_lists")
    
    # Process function for parallel execution
    def process_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
        return _process_single_blueprint(bp, tool_lists_dir)
    
    # Parallel processing with per-blueprint retry
    results, failures = parallel_process_with_retry(
        items=blueprints,
        process_func=process_blueprint,
        max_retries=3,
        description="Extracting tool lists",
    )
    
    # Handle failures
    if failures:
        failed_names = [bp.get("MCP_server_name", "Unknown") for bp, _ in failures]
        logger.error(f"Failed to process blueprints: {failed_names}")
        raise RuntimeError(f"Failed to process {len(failures)} blueprints: {failed_names}")
    
    # Update state
    state.tool_lists_dir = str(tool_lists_dir)
    
    # Calculate total tools
    total_tools = sum(r.get("tool_count", 0) for r in results if r)
    logger.info(f"Tool list formulation complete: {len(results)} blueprints processed, {total_tools} tools extracted")
    
    return state


def _process_single_blueprint(bp: Dict[str, Any], tool_lists_dir: Path) -> Dict[str, Any]:
    """
    Process a single blueprint: extract tool list and save.
    
    Args:
        bp: Blueprint dictionary
        tool_lists_dir: Directory to save tool lists
        
    Returns:
        Dict with server_name and tool_count
    """
    server_name = bp.get("MCP_server_name", "Unknown")
    
    # Extract tool list in OpenAI format
    tools = []
    for func in bp.get("functions", []):
        tool = _convert_to_openai_format(func)
        tools.append(tool)
    
    # Save tool list
    tool_list_path = tool_lists_dir / f"{server_name}.json"
    save_json(tools, tool_list_path)
    
    return {"server_name": server_name, "tool_count": len(tools)}


def _convert_to_openai_format(func: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a blueprint function to OpenAI Function Call format.
    
    Args:
        func: Function definition from blueprint
        
    Returns:
        OpenAI function call format dict
    """
    tool = {
        "type": "function",
        "function": {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    }
    
    # Convert parameters
    for param_name, param_info in func.get("parameters", {}).items():
        tool["function"]["parameters"]["properties"][param_name] = {
            "type": _map_type(param_info.get("type", "string")),
            "description": param_info.get("description", ""),
        }
        # All parameters are marked as required
        tool["function"]["parameters"]["required"].append(param_name)
    
    return tool


def _map_type(type_str: str) -> str:
    """Map type to JSON Schema type."""
    mapping = {
        "string": "string",
        "str": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
    }
    return mapping.get(type_str.lower(), "string")
