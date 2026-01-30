"""
Step 8: Tool Graph Generation

Generate tool execution dependency graphs for MCP servers.

Input: fixed_blueprints.json, policies/*.md
Output: tool_graphs/*.json (includes tool_preconditions and valid_trajectories)

The valid_trajectories field contains pre-filtered trajectories that satisfy
all tool preconditions. Downstream steps (s11, s14) can directly use these
trajectories without additional filtering.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set

import networkx as nx

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import TOOL_GRAPH_PROMPT, TOOL_PRECONDITIONS_PROMPT
from .base import (
    step_handler, save_json, load_json, get_client,
    parallel_process, ensure_dir
)

logger = logging.getLogger(__name__)


def extract_json_from_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to extract from markdown code block
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        json_str = content.split("```")[1].split("```")[0]
    else:
        json_str = content
    
    return json.loads(json_str.strip())


def validate_tool_graph(graph: dict, blueprint: dict) -> dict:
    """Validate and fix the tool graph structure."""
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    
    # Get function names from blueprint
    function_names = {f.get("name") for f in blueprint.get("functions", [])}
    
    # Ensure all nodes are valid functions
    valid_nodes = []
    node_ids = set()
    for node in nodes:
        node_id = node.get("id")
        if node_id and node_id in function_names:
            valid_nodes.append(node)
            node_ids.add(node_id)
    
    # Add any missing functions as nodes
    for func_name in function_names:
        if func_name not in node_ids:
            valid_nodes.append({"id": func_name})
            node_ids.add(func_name)
    
    # Validate links - only keep links between valid nodes
    valid_links = []
    for link in links:
        source = link.get("source")
        target = link.get("target")
        if source in node_ids and target in node_ids:
            valid_links.append(link)
    
    return {
        "directed": True,
        "multigraph": False,
        "nodes": valid_nodes,
        "links": valid_links
    }


# =============================================================================
# Trajectory Enumeration and Filtering
# =============================================================================

def build_graph_from_data(graph_data: Dict[str, Any]) -> nx.DiGraph:
    """Build a NetworkX DiGraph from tool graph JSON data."""
    G = nx.DiGraph()
    
    for node in graph_data.get("nodes", []):
        G.add_node(node["id"])
    
    for link in graph_data.get("links", []):
        G.add_edge(
            link["source"],
            link["target"],
            type=link.get("type", "unknown")
        )
    
    return G


def enumerate_trajectories_from_graph(
    graph_data: Dict[str, Any],
    min_length: int = 2,
    max_length: int = 5
) -> List[List[str]]:
    """Enumerate all valid trajectories from a tool graph using NetworkX."""
    G = build_graph_from_data(graph_data)
    
    if G.number_of_nodes() == 0:
        return []
    
    start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    if not start_nodes:
        start_nodes = list(G.nodes())
    if not end_nodes:
        end_nodes = list(G.nodes())
    
    trajectories: List[List[str]] = []
    
    for start in start_nodes:
        for end in end_nodes:
            if start == end:
                continue
            try:
                paths = nx.all_simple_paths(G, start, end, cutoff=max_length - 1)
                for path in paths:
                    if len(path) >= min_length:
                        trajectories.append(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    
    # Deduplicate
    seen: Set[tuple] = set()
    unique_trajectories = []
    for traj in trajectories:
        traj_tuple = tuple(traj)
        if traj_tuple not in seen:
            seen.add(traj_tuple)
            unique_trajectories.append(traj)
    
    return unique_trajectories


def validate_trajectory_preconditions(
    trajectory: List[str],
    tool_preconditions: Dict[str, List[str]]
) -> bool:
    """
    Validate that a trajectory satisfies all tool preconditions.
    
    For each tool in the trajectory, check that all its required predecessor tools
    have been executed before it in the trajectory.
    """
    seen_tools: Set[str] = set()
    
    for tool in trajectory:
        required = set(tool_preconditions.get(tool, []))
        if not required.issubset(seen_tools):
            return False
        seen_tools.add(tool)
    
    return True


def filter_trajectories_by_preconditions(
    trajectories: List[List[str]],
    tool_preconditions: Dict[str, List[str]],
    server_name: str = None
) -> List[List[str]]:
    """
    Filter trajectories that don't satisfy tool preconditions.
    """
    if not tool_preconditions:
        return trajectories
    
    valid_trajectories = []
    filtered_count = 0
    
    for traj in trajectories:
        if validate_trajectory_preconditions(traj, tool_preconditions):
            valid_trajectories.append(traj)
        else:
            filtered_count += 1
    
    if server_name and filtered_count > 0:
        logger.info(
            f"Server {server_name}: Filtered {filtered_count} invalid trajectories "
            f"({len(valid_trajectories)} remaining)"
        )
    
    return valid_trajectories


# =============================================================================
# Tool Preconditions Extraction
# =============================================================================

def extract_tool_preconditions(
    client,
    policy_content: str,
    server_name: str = None
) -> Dict[str, List[str]]:
    """
    Use LLM to extract tool preconditions from policy content.
    
    This extracts the MANDATORY dependencies for each tool - i.e., which tools
    MUST be successfully executed before a given tool can be called.
    
    Args:
        client: LLM client
        policy_content: The policy markdown content
        server_name: Optional server name for logging
    
    Returns:
        Dict mapping tool_name to list of required predecessor tools
    """
    prompt = TOOL_PRECONDITIONS_PROMPT.format(policy=policy_content)
    
    try:
        response = client.chat(query=prompt, model_type="textual")
        preconditions = extract_json_from_response(response.content)
        
        if isinstance(preconditions, dict):
            # Validate and normalize the result
            normalized = {}
            for tool, deps in preconditions.items():
                if isinstance(deps, list):
                    normalized[tool] = [d for d in deps if isinstance(d, str)]
                else:
                    normalized[tool] = []
            
            if server_name:
                logger.info(f"Extracted preconditions for {server_name}: {len(normalized)} tools")
            return normalized
        else:
            logger.warning(f"Invalid preconditions format for {server_name}: expected dict, got {type(preconditions)}")
            return {}
            
    except Exception as e:
        logger.warning(f"Failed to extract preconditions for {server_name}: {e}")
        return {}


@step_handler("s08_tool_graph_generation", auto_retry=True)
def tool_graph_generation_step(state: WorkflowState) -> WorkflowState:
    """
    Generate tool execution graphs for MCP servers.
    
    Process:
    1. Load blueprints and policies
    2. For each server, analyze tool dependencies
    3. Generate directed acyclic graph (DAG) of tool execution order
    4. Validate and save graphs
    
    Output:
    - tool_graphs/*.json: Tool execution graphs in networkx format
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # Load blueprints
    blueprints_path = state.blueprints_path
    blueprints_data = load_json(Path(blueprints_path))
    if isinstance(blueprints_data, list):
        blueprints = blueprints_data
    else:
        blueprints = blueprints_data.get("blueprints", [])
    
    # Setup directories
    tool_graphs_dir = ensure_dir(outputs_dir / "tool_graphs")
    policies_dir = Path(state.policies_dir) if state.policies_dir else None
    
    logger.info(f"Generating tool graphs for {len(blueprints)} MCP servers")
    
    client = get_client()
    
    # Track processed graphs
    processed = set()
    for f in tool_graphs_dir.glob("*.json"):
        processed.add(f.stem)
    
    to_process = [
        bp for bp in blueprints
        if bp.get("MCP_server_name") not in processed
    ]
    
    if not to_process:
        logger.info("All tool graphs already generated")
        state.tool_graphs_dir = str(tool_graphs_dir)
        return state
    
    def generate_tool_graph(blueprint: dict) -> dict:
        """Generate tool graph for a single server."""
        server_name = blueprint.get("MCP_server_name", "Unknown")
        
        # Get tools from blueprint
        functions = blueprint.get("functions", [])
        tools_description = json.dumps(functions, indent=2)
        
        # Load policy if available
        policy_content = ""
        if policies_dir:
            policy_path = policies_dir / f"{server_name}.md"
            if policy_path.exists():
                policy_content = policy_path.read_text()
        
        # Generate tool graph using LLM
        try:
            graph_prompt = TOOL_GRAPH_PROMPT.format(
                domain_name=server_name,
                tools=tools_description,
                policy=policy_content or "No policy available."
            )
            
            graph_response = client.chat(query=graph_prompt, model_type="textual")
            
            # Parse JSON response
            graph_data = extract_json_from_response(graph_response.content)
            
            # Validate and fix the graph
            graph_data = validate_tool_graph(graph_data, blueprint)
            
            # Extract tool preconditions from policy for trajectory filtering
            # This identifies which tools MUST be executed before each tool
            tool_preconditions = {}
            if policy_content:
                tool_preconditions = extract_tool_preconditions(
                    client, policy_content, server_name
                )
            
            # Add preconditions to graph data
            graph_data["tool_preconditions"] = tool_preconditions
            
            # Enumerate all trajectories from the graph
            all_trajectories = enumerate_trajectories_from_graph(
                graph_data,
                min_length=2,
                max_length=5
            )
            
            # Filter trajectories by preconditions
            valid_trajectories = filter_trajectories_by_preconditions(
                all_trajectories, tool_preconditions, server_name
            )
            
            # Add valid trajectories to graph data for downstream use
            # Downstream steps (s11, s14) can directly use these without filtering
            graph_data["valid_trajectories"] = valid_trajectories
            
            logger.info(
                f"Server {server_name}: {len(all_trajectories)} total trajectories, "
                f"{len(valid_trajectories)} valid after precondition filtering"
            )
            
            # Save tool graph with preconditions and valid trajectories
            graph_path = tool_graphs_dir / f"{server_name}.json"
            save_json(graph_data, graph_path)
            
            return {"success": True, "server_name": server_name}
            
        except Exception as e:
            logger.error(f"Failed to generate tool graph for {server_name}: {e}")
            return {"success": False, "server_name": server_name, "error": str(e)}
    
    # Process blueprints
    results = parallel_process(
        items=to_process,
        process_func=generate_tool_graph,
        description="Generating tool graphs",
    )
    
    # Count successes and failures
    success_count = sum(1 for r in results if r and r.get("success"))
    failed_count = sum(1 for r in results if r and not r.get("success"))
    
    # Log failed servers
    if failed_count > 0:
        failed_servers = [r.get("server_name") for r in results if r and not r.get("success")]
        logger.error(f"Failed to generate tool graphs for {failed_count} servers: {failed_servers}")
    
    state.tool_graphs_dir = str(tool_graphs_dir)
    
    # Update progress
    state.update_step_progress(
        "s08_tool_graph_generation",
        total=len(blueprints),
        completed=len(processed) + success_count,
        failed=failed_count
    )
    
    logger.info(f"Tool graph generation complete: {success_count}/{len(to_process)} graphs")
    
    # Raise error if any failed to trigger step-wise retry
    if failed_count > 0:
        raise RuntimeError(f"Failed to generate {failed_count} tool graphs, will retry")
    
    return state
