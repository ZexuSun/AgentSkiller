#!/usr/bin/env python3
"""
Patch blueprints.json to add evaluation markers for ignored fields.

This script uses LLM to classify parameters into three categories:
1. filter - Parameters used for filtering/narrowing results
2. freetext - Free-form text fields (notes, descriptions, content)
3. relationship_pk - Primary key fields of relationships (runtime-generated UUIDs)

Adds two markers to each identified parameter:
- param_category: "filter" | "freetext" | "relationship_pk"
- ignore_in_eval: true

Usage:
    python scripts/patch_blueprints_eval_markers.py [--dry-run] [--model MODEL]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import only what we need using direct module imports (avoid workflow_v2.__init__)
# This avoids importing langgraph which isn't needed for this script
import importlib.util

def _import_module_direct(module_path: str, module_name: str):
    """Import a module directly from its path without triggering __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load modules directly
_settings_path = project_root / "workflow_v2" / "config" / "settings.py"
_registry_path = project_root / "workflow_v2" / "config" / "models_registry.py"
_llm_path = project_root / "workflow_v2" / "core" / "llm_client.py"

# First load dependencies
_settings_mod = _import_module_direct(str(_settings_path), "workflow_v2.config.settings")
_registry_mod = _import_module_direct(str(_registry_path), "workflow_v2.config.models_registry")
_llm_mod = _import_module_direct(str(_llm_path), "workflow_v2.core.llm_client")

LLMClient = _llm_mod.LLMClient


def get_client() -> "LLMClient":
    """Get an LLM client instance."""
    return LLMClient()


CLASSIFICATION_PROMPT = """You are analyzing an MCP server blueprint to classify parameters that should be ignored during evaluation.

## Server: {server_name}
{server_description}

## Classification Categories

1. **filter**: Parameters used for filtering, narrowing, or limiting query results. Examples:
   - Time windows (window_minutes, date_range_start, date_range_end)
   - Status filters (status_filter, status)
   - Bounds/limits (price_bound, limit, offset)
   - Any parameter whose description contains "filter", "bound", "range", "limit"
   - Parameters in list_*/get_*/search_*/query_* functions that narrow results

2. **freetext**: Free-form text fields that can vary without affecting task success. Examples:
   - Notes fields (notes, comment, description when user-provided)
   - Content fields (content_text, message, reason)
   - Any string parameter meant for human-readable notes
   - NOT structured identifiers, codes, or enums

3. **relationship_pk**: Primary key fields of relationships that are runtime-generated UUIDs. Examples:
   - Fields like selection_id, method_link_id, paymentintent_id
   - Any *_id field in relationship attributes where value_from_entity is "N/A"
   - These are generated at runtime and cannot match golden trajectory

## Functions to Analyze
```json
{functions_json}
```

## Relationships to Analyze
```json
{relationships_json}
```

## Output Format
Return a JSON object with two keys:
- "function_params": Object mapping "function_name.param_name" -> category
- "relationship_attrs": Object mapping "RelationshipName.attr_name" -> category

Only include parameters/attributes that match one of the three categories.

Example:
```json
{{
  "function_params": {{
    "list_items.price_bound": "filter",
    "list_items.status_filter": "filter",
    "create_note.notes": "freetext",
    "add_record.comment": "freetext"
  }},
  "relationship_attrs": {{
    "CustomerItemSelection.selection_id": "relationship_pk",
    "CustomerItemSelection.notes": "freetext",
    "PaymentIntent.paymentintent_id": "relationship_pk"
  }}
}}
```

Analyze the server carefully and return ONLY the JSON output, no explanations.
"""


def load_blueprints(path: Path) -> List[Dict[str, Any]]:
    """Load blueprints from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_blueprints(blueprints: List[Dict[str, Any]], path: Path) -> None:
    """Save blueprints to JSON file with pretty formatting."""
    with open(path, 'w') as f:
        json.dump(blueprints, f, indent=2, ensure_ascii=False)


def backup_blueprints(path: Path) -> Path:
    """Create a timestamped backup of blueprints."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.parent / f"blueprints_backup_{timestamp}.json"
    shutil.copy(path, backup_path)
    return backup_path


def classify_server_fields(
    client: LLMClient,
    server: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Use LLM to classify a server's parameters and relationship attributes.
    
    Args:
        client: LLM client instance
        server: Server blueprint dict
        model: Optional model override
        
    Returns:
        Dict with "function_params" and "relationship_attrs" mappings
    """
    server_name = server.get("MCP_server_name", "Unknown")
    server_desc = server.get("description", "No description")
    
    # Prepare functions JSON (simplified for context length)
    functions = []
    for func in server.get("functions", []):
        func_info = {
            "name": func.get("name"),
            "description": func.get("description"),
            "parameters": func.get("parameters", {}),
        }
        functions.append(func_info)
    
    # Prepare relationships JSON
    relationships = []
    for rel in server.get("relationships", []):
        rel_info = {
            "name": rel.get("name"),
            "description": rel.get("description"),
            "attributes": rel.get("attributes", {}),
        }
        relationships.append(rel_info)
    
    prompt = CLASSIFICATION_PROMPT.format(
        server_name=server_name,
        server_description=server_desc,
        functions_json=json.dumps(functions, indent=2),
        relationships_json=json.dumps(relationships, indent=2),
    )
    
    response = client.chat(
        query=prompt,
        model=model,
        model_type="textual",
        # Note: Some models (e.g., gpt-5) don't support temperature parameter
    )
    
    try:
        result = response.parse_json()
        return {
            "function_params": result.get("function_params", {}),
            "relationship_attrs": result.get("relationship_attrs", {}),
        }
    except Exception as e:
        print(f"  Warning: Failed to parse LLM response for {server_name}: {e}")
        return {"function_params": {}, "relationship_attrs": {}}


def apply_markers_to_server(
    server: Dict[str, Any],
    classifications: Dict[str, Dict[str, str]],
) -> int:
    """
    Apply classification markers to a server's blueprint.
    
    Args:
        server: Server blueprint dict (modified in place)
        classifications: Classification results from LLM
        
    Returns:
        Number of markers applied
    """
    markers_applied = 0
    
    # Apply function parameter markers
    func_params = classifications.get("function_params", {})
    for func in server.get("functions", []):
        func_name = func.get("name", "")
        params = func.get("parameters", {})
        
        for param_name, param_info in params.items():
            key = f"{func_name}.{param_name}"
            if key in func_params:
                category = func_params[key]
                param_info["param_category"] = category
                param_info["ignore_in_eval"] = True
                markers_applied += 1
    
    # Apply relationship attribute markers
    rel_attrs = classifications.get("relationship_attrs", {})
    for rel in server.get("relationships", []):
        rel_name = rel.get("name", "")
        attrs = rel.get("attributes", {})
        
        for attr_name, attr_info in attrs.items():
            key = f"{rel_name}.{attr_name}"
            if key in rel_attrs:
                category = rel_attrs[key]
                attr_info["param_category"] = category
                attr_info["ignore_in_eval"] = True
                markers_applied += 1
    
    return markers_applied


def main():
    parser = argparse.ArgumentParser(
        description="Patch blueprints.json with evaluation markers using LLM classification"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be changed without modifying files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model to use (default: textual model from config)"
    )
    parser.add_argument(
        "--blueprints",
        type=Path,
        default=project_root / "outputs" / "blueprints.json",
        help="Path to blueprints.json (default: outputs/blueprints.json)"
    )
    parser.add_argument(
        "--servers",
        type=str,
        nargs="*",
        help="Specific server names to process (default: all servers)"
    )
    args = parser.parse_args()
    
    blueprints_path = args.blueprints
    
    if not blueprints_path.exists():
        print(f"Error: Blueprints file not found: {blueprints_path}")
        sys.exit(1)
    
    if args.dry_run:
        print("=== DRY RUN MODE - No files will be modified ===\n")
    
    # Load blueprints
    print(f"Loading blueprints from {blueprints_path}...")
    blueprints = load_blueprints(blueprints_path)
    print(f"Loaded {len(blueprints)} MCP servers\n")
    
    # Filter servers if specified
    if args.servers:
        server_set = set(args.servers)
        blueprints_to_process = [
            bp for bp in blueprints 
            if bp.get("MCP_server_name") in server_set
        ]
        print(f"Processing {len(blueprints_to_process)} specified servers\n")
    else:
        blueprints_to_process = blueprints
    
    # Initialize LLM client
    client = get_client()
    
    # Process each server
    total_markers = 0
    for i, server in enumerate(blueprints_to_process, 1):
        server_name = server.get("MCP_server_name", "Unknown")
        print(f"[{i}/{len(blueprints_to_process)}] Processing {server_name}...")
        
        try:
            # Classify fields using LLM
            classifications = classify_server_fields(client, server, args.model)
            
            func_count = len(classifications.get("function_params", {}))
            rel_count = len(classifications.get("relationship_attrs", {}))
            
            print(f"  Found {func_count} function params, {rel_count} relationship attrs to mark")
            
            if not args.dry_run:
                # Apply markers to the original blueprint in the list
                # Find the server in the original blueprints list
                for bp in blueprints:
                    if bp.get("MCP_server_name") == server_name:
                        markers = apply_markers_to_server(bp, classifications)
                        total_markers += markers
                        break
            else:
                # In dry run, just count
                total_markers += func_count + rel_count
                
                # Show what would be marked
                if func_count > 0:
                    print("    Function params:")
                    for key, cat in classifications.get("function_params", {}).items():
                        print(f"      - {key}: {cat}")
                if rel_count > 0:
                    print("    Relationship attrs:")
                    for key, cat in classifications.get("relationship_attrs", {}).items():
                        print(f"      - {key}: {cat}")
                        
        except Exception as e:
            print(f"  Error processing {server_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Total markers to apply: {total_markers}")
    
    if not args.dry_run and total_markers > 0:
        # Create backup
        backup_path = backup_blueprints(blueprints_path)
        print(f"Created backup: {backup_path}")
        
        # Save updated blueprints
        save_blueprints(blueprints, blueprints_path)
        print(f"Saved updated blueprints to {blueprints_path}")
    elif args.dry_run:
        print("\n=== DRY RUN COMPLETE - No files were modified ===")
        print("Run without --dry-run to apply changes.")
    else:
        print("No markers to apply.")


if __name__ == "__main__":
    main()

