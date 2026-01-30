"""
Utility for generating unique sample IDs.
"""

import json
import hashlib
from typing import Dict, Any


def generate_id(entry: Dict[str, Any], include_tools: bool = True) -> str:
    """
    Generate a unique ID for a sample based on its content.
    
    The ID is based on:
    - The first user message content
    - The tools (if include_tools=True)
    
    This ensures that the same question with different tools is recognized
    as different samples, which is important for tool-related experiments.
    
    Args:
        entry: Sample dictionary containing 'messages' and optionally 'tools'
        include_tools: Whether to include tools in the hash
        
    Returns:
        MD5 hash string as the sample ID
        
    Example:
        >>> sample = {"messages": [{"role": "user", "content": "Hello"}]}
        >>> generate_id(sample, include_tools=False)
        '8b1a9953c4611296a827abf8c47804d7'
    """
    messages = entry.get("messages", [])
    
    # Find the first user message
    first_user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            first_user_content = msg.get("content", "")
            break
        elif msg.get("role") == "system":
            continue
        else:
            # If first non-system message isn't user, use it
            first_user_content = msg.get("content", "")
            break
    
    # Build hash input
    if include_tools:
        tools = entry.get("tools", [])
        if tools:
            ready_to_hash = first_user_content + json.dumps(tools, sort_keys=True)
        else:
            ready_to_hash = first_user_content
    else:
        ready_to_hash = first_user_content
    
    return hashlib.md5(ready_to_hash.encode("utf-8")).hexdigest()

