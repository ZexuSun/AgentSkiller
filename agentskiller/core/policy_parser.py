"""
Policy Parser for structured policy documents.

This module provides utilities for:
1. Parsing structured policy documents with markers
2. Filtering policies to extract relevant tool sections
3. Extracting value domain definitions from policy text
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolPolicy:
    """Represents a single tool's policy section."""
    tool_name: str
    content: str
    purpose: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    input_validations: List[str] = field(default_factory=list)
    permission_outcomes: Dict[str, str] = field(default_factory=dict)
    effects: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ValueDomain:
    """Represents a value domain definition extracted from policy."""
    domain_type: str  # "enum", "range", "array_choice", "boolean"
    values: Optional[List[str]] = None  # for enum
    min_val: Optional[float] = None  # for range
    max_val: Optional[float] = None  # for range
    is_int: bool = False  # for range
    choices: Optional[List[str]] = None  # for array_choice
    min_count: Optional[int] = None  # for array_choice
    max_count: Optional[int] = None  # for array_choice
    unique: bool = True  # for array_choice
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        if self.domain_type == "enum":
            return {"type": "enum", "values": self.values or []}
        elif self.domain_type == "range":
            return {
                "type": "range",
                "min": self.min_val,
                "max": self.max_val,
                "is_int": self.is_int
            }
        elif self.domain_type == "array_choice":
            return {
                "type": "array_choice",
                "choices": self.choices or [],
                "min_count": self.min_count or 1,
                "max_count": self.max_count or len(self.choices or []),
                "unique": self.unique
            }
        elif self.domain_type == "boolean":
            return {"type": "boolean"}
        return {"type": self.domain_type}


@dataclass
class ParsedPolicy:
    """Represents a fully parsed policy document."""
    global_rules: str
    tool_policies: Dict[str, ToolPolicy]
    raw_content: str


# =============================================================================
# Policy Markers
# =============================================================================

POLICY_MARKERS = {
    "policy_start": "<!-- POLICY_START -->",
    "policy_end": "<!-- POLICY_END -->",
    "global_rules_end": "<!-- GLOBAL_RULES_END -->",
    "tool_start_pattern": r"<!-- TOOL: (\S+) -->",
    "tool_end_pattern": r"<!-- TOOL_END: (\S+) -->",
}


# =============================================================================
# Policy Parsing Functions
# =============================================================================

def parse_policy(policy_content: str) -> ParsedPolicy:
    """
    Parse a structured policy document into components.
    
    Args:
        policy_content: The full policy markdown content
        
    Returns:
        ParsedPolicy object with global rules and tool policies
    """
    global_rules = ""
    tool_policies = {}
    
    # Try to extract global rules (between POLICY_START and GLOBAL_RULES_END)
    global_match = re.search(
        r'<!-- POLICY_START -->(.*?)<!-- GLOBAL_RULES_END -->',
        policy_content,
        re.DOTALL
    )
    if global_match:
        global_rules = global_match.group(1).strip()
    else:
        # Fallback: try to extract everything before first tool section
        first_tool = re.search(r'<!-- TOOL: \S+ -->', policy_content)
        if first_tool:
            global_rules = policy_content[:first_tool.start()].strip()
        else:
            # No markers found, treat entire content as global rules
            global_rules = policy_content
    
    # Extract individual tool policies
    tool_pattern = r'<!-- TOOL: (\S+) -->(.*?)<!-- TOOL_END: \1 -->'
    for match in re.finditer(tool_pattern, policy_content, re.DOTALL):
        tool_name = match.group(1)
        tool_content = match.group(2).strip()
        
        tool_policy = ToolPolicy(
            tool_name=tool_name,
            content=tool_content
        )
        
        # Try to extract structured fields from tool content
        tool_policy.purpose = _extract_field(tool_content, "Purpose")
        tool_policy.preconditions = _extract_list_field(tool_content, "Preconditions")
        tool_policy.input_validations = _extract_list_field(tool_content, "Input validation")
        tool_policy.effects = _extract_list_field(tool_content, "Effects")
        tool_policy.dependencies = _extract_list_field(tool_content, "Dependencies")
        
        tool_policies[tool_name] = tool_policy
    
    return ParsedPolicy(
        global_rules=global_rules,
        tool_policies=tool_policies,
        raw_content=policy_content
    )


def _extract_field(content: str, field_name: str) -> Optional[str]:
    """Extract a single field value from content."""
    pattern = rf'{field_name}:\s*(.+?)(?=\n[A-Z]|\n-|\n#|\Z)'
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_list_field(content: str, field_name: str) -> List[str]:
    """Extract a list field (bullet points) from content."""
    # Find the section
    pattern = rf'{field_name}[:\s]*\n((?:\s*[-•]\s*.+\n?)+)'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        items = re.findall(r'[-•]\s*(.+)', match.group(1))
        return [item.strip() for item in items]
    return []


# =============================================================================
# Policy Filtering Functions
# =============================================================================

def filter_policy_for_trajectory(policy_content: str, trajectory: List[str]) -> str:
    """
    Extract only the relevant policy sections for a given tool call trajectory.
    
    Args:
        policy_content: The full policy markdown content
        trajectory: List of tool names in the trajectory
        
    Returns:
        Filtered policy containing only global rules and relevant tool policies
    """
    parsed = parse_policy(policy_content)
    
    # Start with global rules
    filtered_parts = []
    if parsed.global_rules:
        filtered_parts.append("## Global Rules\n\n" + parsed.global_rules)
    
    # Add relevant tool policies
    for tool_name in trajectory:
        if tool_name in parsed.tool_policies:
            tool_policy = parsed.tool_policies[tool_name]
            filtered_parts.append(f"\n\n## Tool: {tool_name}\n\n{tool_policy.content}")
        else:
            # Try case-insensitive match
            for key, policy in parsed.tool_policies.items():
                if key.lower() == tool_name.lower():
                    filtered_parts.append(f"\n\n## Tool: {tool_name}\n\n{policy.content}")
                    break
    
    return "\n".join(filtered_parts)


def filter_policy_for_trajectory_with_markers(policy_content: str, trajectory: List[str]) -> str:
    """
    Extract relevant policy sections while preserving structure markers.
    
    Args:
        policy_content: The full policy markdown content
        trajectory: List of tool names in the trajectory
        
    Returns:
        Filtered policy with markers preserved
    """
    # Extract global rules
    global_match = re.search(
        r'(<!-- POLICY_START -->.*?<!-- GLOBAL_RULES_END -->)',
        policy_content,
        re.DOTALL
    )
    global_section = global_match.group(1) if global_match else ""
    
    # Extract relevant tool sections
    tool_sections = []
    for tool_name in trajectory:
        pattern = rf'(<!-- TOOL: {re.escape(tool_name)} -->.*?<!-- TOOL_END: {re.escape(tool_name)} -->)'
        match = re.search(pattern, policy_content, re.DOTALL)
        if match:
            tool_sections.append(match.group(1))
    
    # Combine
    result = global_section + "\n\n" + "\n\n".join(tool_sections)
    if not result.strip().endswith("<!-- POLICY_END -->"):
        result += "\n<!-- POLICY_END -->"
    
    return result


# =============================================================================
# Value Domain Extraction Functions
# =============================================================================

def extract_value_domains(policy_content: str) -> Dict[str, Dict[str, ValueDomain]]:
    """
    Extract all value domain definitions from a policy document.
    
    Args:
        policy_content: The policy text content
        
    Returns:
        Dict mapping tool_name -> parameter_name -> ValueDomain
    """
    domains: Dict[str, Dict[str, ValueDomain]] = {}
    
    # Parse the policy
    parsed = parse_policy(policy_content)
    
    # Also check global rules for common patterns
    global_domains = _extract_domains_from_text(parsed.global_rules)
    if global_domains:
        domains["_global"] = global_domains
    
    # Extract from each tool policy
    for tool_name, tool_policy in parsed.tool_policies.items():
        tool_domains = _extract_domains_from_text(tool_policy.content)
        if tool_domains:
            domains[tool_name] = tool_domains
    
    return domains


def _extract_domains_from_text(text: str) -> Dict[str, ValueDomain]:
    """Extract value domain definitions from a text block."""
    domains = {}
    
    # Pattern 1: Enum with ∈ notation: "param ∈ {val1, val2, val3}"
    enum_pattern = r'(\w+)\s*[∈∊∋]\s*\{([^}]+)\}'
    for match in re.finditer(enum_pattern, text):
        param_name = match.group(1)
        values = [v.strip().strip('"\'') for v in match.group(2).split(',')]
        domains[param_name] = ValueDomain(domain_type="enum", values=values)
    
    # Pattern 2: Enum with "in" notation: "param in {val1, val2}"
    enum_in_pattern = r'(\w+)\s+in\s+\{([^}]+)\}'
    for match in re.finditer(enum_in_pattern, text, re.IGNORECASE):
        param_name = match.group(1)
        if param_name not in domains:
            values = [v.strip().strip('"\'') for v in match.group(2).split(',')]
            domains[param_name] = ValueDomain(domain_type="enum", values=values)
    
    # Pattern 3: Enum with parentheses: "param (val1|val2|val3)"
    enum_paren_pattern = r'(\w+):\s*(?:string\s*)?\(([^)]+(?:\|[^)]+)+)\)'
    for match in re.finditer(enum_paren_pattern, text):
        param_name = match.group(1)
        if param_name not in domains:
            values = [v.strip() for v in match.group(2).split('|')]
            domains[param_name] = ValueDomain(domain_type="enum", values=values)
    
    # Pattern 4: Numeric range: "param in [min, max]" or "param within [min, max]"
    range_pattern = r'(\w+)\s+(?:in|within)\s+\[([0-9.-]+),\s*([0-9.-]+)\]'
    for match in re.finditer(range_pattern, text, re.IGNORECASE):
        param_name = match.group(1)
        min_val = float(match.group(2))
        max_val = float(match.group(3))
        is_int = '.' not in match.group(2) and '.' not in match.group(3)
        domains[param_name] = ValueDomain(
            domain_type="range",
            min_val=min_val,
            max_val=max_val,
            is_int=is_int
        )
    
    # Pattern 5: Float range with type: "param float in [min, max]"
    float_range_pattern = r'(\w+)\s+(?:float|integer|int)\s+(?:in|within)\s+\[([0-9.-]+),\s*([0-9.-]+)\]'
    for match in re.finditer(float_range_pattern, text, re.IGNORECASE):
        param_name = match.group(1)
        if param_name not in domains:
            min_val = float(match.group(2))
            max_val = float(match.group(3))
            is_int = 'int' in match.group(0).lower()
            domains[param_name] = ValueDomain(
                domain_type="range",
                min_val=min_val,
                max_val=max_val,
                is_int=is_int
            )
    
    # Pattern 6: Array choice: "param must be an array of N to M ... from: list"
    array_pattern = r'(\w+)\s+(?:must be\s+)?(?:an?\s+)?array\s+of\s+(\d+)\s+to\s+(\d+).*?(?:from|chosen from)[:\s]*([^\n.]+)'
    for match in re.finditer(array_pattern, text, re.IGNORECASE):
        param_name = match.group(1)
        min_count = int(match.group(2))
        max_count = int(match.group(3))
        choices_text = match.group(4)
        # Parse choices
        choices = [c.strip().strip('"\'') for c in re.split(r'[,\s]+', choices_text) if c.strip()]
        domains[param_name] = ValueDomain(
            domain_type="array_choice",
            choices=choices,
            min_count=min_count,
            max_count=max_count,
            unique=True
        )
    
    # Pattern 7: Boolean: "param is boolean" or "param: boolean"
    bool_pattern = r'(\w+)\s*(?:is|:)\s*boolean'
    for match in re.finditer(bool_pattern, text, re.IGNORECASE):
        param_name = match.group(1)
        if param_name not in domains:
            domains[param_name] = ValueDomain(domain_type="boolean")
    
    return domains


def extract_value_domain_for_parameter(
    policy_content: str,
    tool_name: str,
    parameter_name: str
) -> Optional[ValueDomain]:
    """
    Extract a specific parameter's value domain from policy.
    
    Args:
        policy_content: The policy text content
        tool_name: Name of the tool
        parameter_name: Name of the parameter
        
    Returns:
        ValueDomain if found, None otherwise
    """
    all_domains = extract_value_domains(policy_content)
    
    # Check tool-specific domains first
    if tool_name in all_domains:
        if parameter_name in all_domains[tool_name]:
            return all_domains[tool_name][parameter_name]
    
    # Check global domains
    if "_global" in all_domains:
        if parameter_name in all_domains["_global"]:
            return all_domains["_global"][parameter_name]
    
    return None


# =============================================================================
# Utility Functions
# =============================================================================

def has_structured_markers(policy_content: str) -> bool:
    """Check if a policy document has structured markers."""
    return (
        "<!-- POLICY_START -->" in policy_content or
        "<!-- TOOL:" in policy_content
    )


def add_markers_to_policy(policy_content: str, tool_names: List[str]) -> str:
    """
    Attempt to add structured markers to an unmarked policy document.
    
    This is a best-effort function for legacy policies without markers.
    
    Args:
        policy_content: The policy content without markers
        tool_names: List of expected tool names
        
    Returns:
        Policy content with markers added
    """
    if has_structured_markers(policy_content):
        return policy_content
    
    result = ["<!-- POLICY_START -->"]
    
    # Try to find tool sections using common patterns
    tool_sections = {}
    remaining = policy_content
    
    for tool_name in tool_names:
        # Common patterns for tool section headers
        patterns = [
            rf'(###?\s*(?:Tool[:\s]*)?{re.escape(tool_name)}.*?)(?=###?\s*(?:Tool)?|\Z)',
            rf'({re.escape(tool_name)}\s*\n[-=]+\n.*?)(?=\n[A-Z].*?\n[-=]+|\Z)',
            rf'(\d+\)\s*{re.escape(tool_name)}.*?)(?=\d+\)|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, remaining, re.DOTALL | re.IGNORECASE)
            if match:
                tool_sections[tool_name] = match.group(1).strip()
                break
    
    # Find global rules (content before first tool section)
    first_tool_pos = len(policy_content)
    for tool_name, section in tool_sections.items():
        pos = policy_content.find(section)
        if pos < first_tool_pos:
            first_tool_pos = pos
    
    global_rules = policy_content[:first_tool_pos].strip()
    if global_rules:
        result.append(global_rules)
        result.append("<!-- GLOBAL_RULES_END -->")
    
    # Add tool sections with markers
    for tool_name in tool_names:
        if tool_name in tool_sections:
            result.append(f"\n<!-- TOOL: {tool_name} -->")
            result.append(tool_sections[tool_name])
            result.append(f"<!-- TOOL_END: {tool_name} -->")
    
    result.append("\n<!-- POLICY_END -->")
    
    return "\n".join(result)


def get_tool_names_from_policy(policy_content: str) -> List[str]:
    """Extract all tool names from a policy document."""
    tool_names = []
    
    # From markers
    marker_pattern = r'<!-- TOOL: (\S+) -->'
    tool_names.extend(re.findall(marker_pattern, policy_content))
    
    if not tool_names:
        # Try to find from headers
        header_patterns = [
            r'###?\s*Tool[:\s]*(\w+)',
            r'###?\s*\d+\)\s*(\w+)',
            r'^(\w+)\s*\n[-=]+',
        ]
        for pattern in header_patterns:
            tool_names.extend(re.findall(pattern, policy_content, re.MULTILINE))
    
    return list(dict.fromkeys(tool_names))  # Remove duplicates, preserve order
