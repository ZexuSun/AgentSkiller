"""
Domain Name Resolver

Resolves the correct domain name by trying all possible permutations
and checking which one exists in the database directory.

This is needed because the domain name order in configuration might not
match the actual directory name in database/outputs/entities/.
"""

import os
from pathlib import Path
from typing import Optional, List
from itertools import permutations


def _check_domain_exists(
    domain_name: str,
    base_path: Path,
    check_entities: bool,
    check_relationships: bool
) -> bool:
    """Check if a domain directory exists."""
    if check_entities:
        entities_dir = base_path / "entities" / domain_name
        if not entities_dir.exists():
            return False
    
    if check_relationships:
        relationships_dir = base_path / "relationships" / domain_name
        if not relationships_dir.exists():
            return False
    
    return True


def resolve_domain_name(
    domain_name: str,
    base_path: Optional[Path] = None,
    check_entities: bool = True,
    check_relationships: bool = True
) -> Optional[str]:
    """
    Resolve the correct domain name by trying all permutations.
    
    Args:
        domain_name: Domain name (may be in wrong order, e.g., "A_B_C")
        base_path: Base path to database directory (default: relative to mcp_servers)
        check_entities: Check entities directory
        check_relationships: Check relationships directory
        
    Returns:
        The correct domain name that exists, or None if not found
        
    Example:
        >>> resolve_domain_name("StudentAcademicPortal_StudentFinancialServices_StudentHealthServices")
        "StudentFinancialServices_StudentHealthServices_StudentAcademicPortal"  # if this is the actual directory
    """
    if "_" not in domain_name:
        # Single domain, no permutation needed
        return domain_name
    
    # Split into individual domains
    domains = domain_name.split("_")
    
    if len(domains) <= 1:
        return domain_name
    
    # Generate all permutations (but avoid duplicates if domains are repeated)
    # Use set to avoid duplicate permutations
    seen = set()
    perms = []
    for perm in permutations(domains):
        perm_str = "_".join(perm)
        if perm_str not in seen:
            seen.add(perm_str)
            perms.append(perm_str)
    
    # Determine base path
    if base_path is None:
        # Default: relative to mcp_servers directory
        script_dir = Path(__file__).parent.parent / "tools" / "datasets" / "cross_domain"
        base_path = script_dir / "database" / "outputs"
    else:
        base_path = Path(base_path)
    
    # First, check if the original domain_name exists (most common case)
    if _check_domain_exists(domain_name, base_path, check_entities, check_relationships):
        return domain_name
    
    # Try each permutation (skip the original since we already checked it)
    for perm_domain in perms:
        if perm_domain == domain_name:
            continue  # Already checked
        
        if _check_domain_exists(perm_domain, base_path, check_entities, check_relationships):
            return perm_domain
    
    # If no exact match found, return original (might be single-domain or wrong path)
    return domain_name


def find_matching_domain_dirs(
    domains: List[str],
    base_path: Optional[Path] = None
) -> List[str]:
    """
    Find all matching domain directories by trying all permutations.
    
    Args:
        domains: List of individual domain names
        base_path: Base path to database directory
        
    Returns:
        List of matching domain directory names
        
    Example:
        >>> find_matching_domain_dirs(["StudentAcademicPortal", "StudentFinancialServices", "StudentHealthServices"])
        ["StudentAcademicPortal_StudentFinancialServices_StudentHealthServices",
         "StudentFinancialServices_StudentHealthServices_StudentAcademicPortal"]
    """
    if not domains or len(domains) <= 1:
        return domains if domains else []
    
    # Determine base path
    if base_path is None:
        script_dir = Path(__file__).parent.parent / "tools" / "datasets" / "cross_domain"
        base_path = script_dir / "database" / "outputs"
    else:
        base_path = Path(base_path)
    
    entities_dir = base_path / "entities"
    if not entities_dir.exists():
        return []
    
    # Generate all permutations
    seen = set()
    perms = []
    for perm in permutations(domains):
        perm_str = "_".join(perm)
        if perm_str not in seen:
            seen.add(perm_str)
            perms.append(perm_str)
    
    # Check which ones exist
    matching = []
    for perm_domain in perms:
        domain_dir = entities_dir / perm_domain
        if domain_dir.exists() and domain_dir.is_dir():
            matching.append(perm_domain)
    
    return matching


def resolve_domain_name_from_domains(
    domains: List[str],
    base_path: Optional[Path] = None
) -> Optional[str]:
    """
    Resolve domain name from a list of individual domains.
    
    This is a convenience function that tries all permutations and returns
    the first matching one.
    
    Args:
        domains: List of individual domain names
        base_path: Base path to database directory
        
    Returns:
        The first matching domain name, or None if not found
    """
    matching = find_matching_domain_dirs(domains, base_path)
    return matching[0] if matching else None

