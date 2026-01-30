#!/usr/bin/env python3
"""
Cross-Domain Discovery and Matching Utilities

Handles automatic discovery of cross-domain combinations and order-agnostic
matching of policies, queries, and MCP servers.

Example:
    >>> from rollout.utils.cross_domain import CrossDomainManager
    >>> manager = CrossDomainManager(base_path="rollout/tools/datasets/cross_domain")
    >>> combos = manager.discover_combinations()
    >>> for combo in combos:
    ...     print(combo.name, combo.policy_file, combo.query_dir)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def normalize_domain_set(domain_string: str) -> frozenset:
    """
    Extract domains from a combined domain string and return as a frozenset.
    Order-agnostic comparison.
    
    Example:
        "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices"
        -> frozenset({'StudentAcademicPortal', 'StudentFinancialServices', 'StudentHealthServices'})
    """
    # Split by underscore but handle cases where domain names contain underscores
    # Domain names are CamelCase, so we split on the pattern where lowercase/digit
    # is followed by uppercase
    parts = domain_string.split('_')
    
    # Reconstruct domain names by joining parts that form valid CamelCase names
    domains = []
    current = []
    
    for part in parts:
        if part and part[0].isupper():
            if current:
                domains.append('_'.join(current))
            current = [part]
        else:
            current.append(part)
    
    if current:
        domains.append('_'.join(current))
    
    return frozenset(domains)


def extract_domains_smart(name: str) -> List[str]:
    """
    Smart extraction of domain names from a combined string.
    Handles both underscore-separated and potential variations.
    
    Returns list of domain names in original order.
    """
    # Known domain patterns (can be extended)
    known_domains = {
        'StudentAcademicPortal', 'StudentFinancialServices', 'StudentHealthServices',
        'StudentAcademicManagement', 'StudentAcademicAdvisor', 'StudentLearningPortal',
        'PatientHealthcareManagement', 'PatientInsuranceManagement', 'PatientBankingServices',
        'EmployeeAcademicManagement', 'EmployeeFinancialManagement', 'EmployeeHealthcareManagement',
        'CaregiverChildcareManagement', 'CaregiverEmploymentServices', 'CaregiverHealthcareManagement',
        'ChildHealthcareManagement', 'ChildcareManagement', 'ChildPersonalDevelopment',
        'ClinicalEducationManagement', 'ClinicalPracticeManagement', 'ClinicalResearchManagement',
        'InstructorCourseManagement', 'InstructorMentorshipProgram', 'InstructorResearchManagement',
        'HealthcareAppointmentSystem', 'HealthcarePatientPortal',
        'FitnessTrackingSystem', 'FitnessTrainingManager',
        'JobApplicationManager', 'JobApplicationSystem',
        'MentorshipManagementSystem', 'MentorshipPlatform',
        'EventManagementSystem', 'EventSpeakerManagement',
        'SpeakerMentorshipPlatform', 'SpeakerTravelBooking',
        'StylistEventManager', 'StylistPortfolioManager',
        'SalonStylistBooking', 'RestaurantBookingSystem',
        'PersonalFinanceManager', 'PersonalWellnessTracker',
        'PropertyTaxManager', 'TaxPreparationService',
        'ResearchMentorshipHub', 'SkillCertificationTracker',
        'TechnicianHealthcareEquipment', 'TechnicianJobPlacement', 'TechnicianWorkOrderManagement',
        'AcademicAdvisingSystem', 'AcademicAdvisorySystem', 'CareerAdvisoryPlatform',
    }
    
    # Try to match known domains greedily
    remaining = name
    domains = []
    
    while remaining:
        matched = False
        # Try longest match first
        for domain in sorted(known_domains, key=len, reverse=True):
            if remaining.startswith(domain):
                domains.append(domain)
                remaining = remaining[len(domain):]
                # Remove leading underscore if present
                if remaining.startswith('_'):
                    remaining = remaining[1:]
                matched = True
                break
        
        if not matched:
            # Fallback: take until next underscore or end
            if '_' in remaining:
                part, remaining = remaining.split('_', 1)
                domains.append(part)
            else:
                if remaining:
                    domains.append(remaining)
                break
    
    return domains


@dataclass
class CrossDomainCombo:
    """Represents a cross-domain combination with all associated resources."""
    name: str  # Original directory/file name
    domains: List[str]  # Individual domain names
    domain_set: frozenset  # For order-agnostic comparison
    
    # Paths (may be None if not found)
    query_dir: Optional[Path] = None
    query_file: Optional[Path] = None  # queries.jsonl
    policy_file: Optional[Path] = None
    
    # Individual query files (before merging)
    individual_query_files: List[Path] = field(default_factory=list)
    
    # Tool names in snake_case
    tools: List[str] = field(default_factory=list)
    
    @property
    def needs_merge(self) -> bool:
        """Check if this combo has individual files but no merged queries.jsonl."""
        return len(self.individual_query_files) > 0 and (
            self.query_file is None or not self.query_file.exists()
        )
    
    @property
    def query_count(self) -> int:
        """Estimate total query count from individual files."""
        if self.query_file and self.query_file.exists():
            return sum(1 for line in open(self.query_file) if line.strip())
        
        count = 0
        for f in self.individual_query_files:
            count += sum(1 for line in open(f) if line.strip())
        return count
    
    def __hash__(self):
        return hash(self.domain_set)
    
    def __eq__(self, other):
        if isinstance(other, CrossDomainCombo):
            return self.domain_set == other.domain_set
        return False


class CrossDomainManager:
    """
    Manages discovery and matching of cross-domain combinations.
    
    Handles:
    - Auto-discovery of query directories
    - Order-agnostic policy file matching
    - Tool name generation
    """
    
    def __init__(self, base_path: str = "rollout/tools/datasets/cross_domain"):
        self.base_path = Path(base_path)
        self.queries_dir = self.base_path / "queries"
        self.policies_dir = self.base_path / "policies"
        self.mcp_servers_dir = self.base_path / "mcp_servers"
        self.tool_lists_dir = self.base_path / "tool_lists"
        
        # Cache for policy file mapping
        self._policy_cache: Dict[frozenset, Path] = {}
        self._build_policy_cache()
    
    def _build_policy_cache(self):
        """Build a cache mapping domain sets to policy files."""
        if not self.policies_dir.exists():
            return
        
        for policy_file in self.policies_dir.glob("*.md"):
            name = policy_file.stem
            # Skip single-domain policies for cross-domain matching
            if '_' not in name:
                continue
            
            domains = extract_domains_smart(name)
            domain_set = frozenset(domains)
            self._policy_cache[domain_set] = policy_file
    
    def find_policy_file(self, domain_set: frozenset) -> Optional[Path]:
        """
        Find policy file for a domain set, ignoring order.
        
        Returns None if no matching policy file found.
        """
        return self._policy_cache.get(domain_set)
    
    def discover_combinations(self, min_domains: int = 2) -> List[CrossDomainCombo]:
        """
        Discover all cross-domain combinations from the queries directory.
        
        Args:
            min_domains: Minimum number of domains to be considered cross-domain
        
        Returns:
            List of CrossDomainCombo objects
        """
        if not self.queries_dir.exists():
            return []
        
        combos = []
        
        for query_file in self.queries_dir.iterdir():
            if not ("_" in query_file.name):
                continue

            name = query_file.name.split(".")[0]
            domains = extract_domains_smart(name)
            
            if len(domains) < min_domains:
                continue
            
            domain_set = frozenset(domains)
            
            # Find associated resources
            policy_file = self.find_policy_file(domain_set)
            
            # Generate tool names
            tools = [to_snake_case(d) for d in domains]
            
            combo = CrossDomainCombo(
                name=name,
                domains=domains,
                domain_set=domain_set,
                query_dir=None,
                query_file=query_file if query_file.exists() else None,
                policy_file=policy_file,
                individual_query_files=None,
                tools=tools
            )
            
            combos.append(combo)
        
        return combos
    
    def get_combo_by_name(self, name: str) -> Optional[CrossDomainCombo]:
        """Get a specific combination by name (order-agnostic)."""
        domains = extract_domains_smart(name)
        target_set = frozenset(domains)
        
        for combo in self.discover_combinations():
            if combo.domain_set == target_set:
                return combo
        
        return None
    
    def generate_dataset_config(
        self,
        combo: CrossDomainCombo,
        output_dir: str = "./outputs",
        agent_model: str = "openai/deepseek-v3.2",
        user_model: str = "openai/gpt-5",
        max_turns: int = 20,
        mode: str = "positive"
    ) -> Dict:
        """
        Generate a dataset configuration dict for a cross-domain combination.
        """
        config = {
            "path": str(combo.query_file) if combo.query_file else f"./queries/{combo.name}/queries.jsonl",
            "output_path": f"{output_dir}/{combo.name}_output.jsonl",
            "mcp_domain": combo.name,
            "tools": combo.tools,
            "agent": {
                "model": agent_model,
                "temperature": 0.7,
                "enable_thinking": True
            },
            "user": {
                "model": user_model,
                "temperature": 1.0
            },
            "max_turns": max_turns,
            "max_steps_per_turn": 10,
            "mode": mode
        }
        
        if combo.policy_file:
            config["agent"]["system_prompt_file"] = str(combo.policy_file)
        
        return config
    
    def list_all_combinations(self) -> List[Dict]:
        """
        List all discovered combinations with their status.
        
        Returns list of dicts with combo info and resource availability.
        """
        results = []
        
        for combo in self.discover_combinations():
            results.append({
                "name": combo.name,
                "domains": combo.domains,
                "tools": combo.tools,
                "has_queries": combo.query_file is not None and combo.query_file.exists(),
                "has_policy": combo.policy_file is not None,
                "policy_file": str(combo.policy_file) if combo.policy_file else None,
                "query_dir": str(combo.query_dir),
                "individual_files": len(combo.individual_query_files),
                "needs_merge": combo.needs_merge
            })
        
        return results
    
    def merge_queries(self, combo: CrossDomainCombo, force: bool = False) -> Optional[Path]:
        """
        Merge individual query files into a single queries.jsonl.
        
        Args:
            combo: CrossDomainCombo to merge queries for
            force: Overwrite existing queries.jsonl if present
        
        Returns:
            Path to merged file, or None if nothing to merge
        """
        import json
        
        if not combo.individual_query_files:
            return combo.query_file  # Nothing to merge
        
        output_path = combo.query_dir / "queries.jsonl"
        
        if output_path.exists() and not force:
            return output_path  # Already exists
        
        seen_ids = set()
        total = 0
        
        with open(output_path, "w", encoding="utf-8") as outfile:
            for f in combo.individual_query_files:
                with open(f, "r", encoding="utf-8") as infile:
                    for line in infile:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                query_id = data.get("id", "")
                                
                                # Skip duplicates
                                if query_id in seen_ids:
                                    continue
                                
                                seen_ids.add(query_id)
                                outfile.write(line if line.endswith("\n") else line + "\n")
                                total += 1
                            except json.JSONDecodeError:
                                outfile.write(line if line.endswith("\n") else line + "\n")
                                total += 1
        
        return output_path
    
    def merge_all_queries(self, force: bool = False, verbose: bool = True) -> Dict[str, int]:
        """
        Merge queries for all discovered combinations that need it.
        
        Returns statistics about the merge operation.
        """
        stats = {"merged": 0, "skipped": 0, "total_queries": 0}
        
        for combo in self.discover_combinations():
            if not combo.individual_query_files:
                continue
            
            output_path = combo.query_dir / "queries.jsonl"
            
            if output_path.exists() and not force:
                stats["skipped"] += 1
                if verbose:
                    print(f"  - Skipped {combo.name} (queries.jsonl exists)")
                continue
            
            result = self.merge_queries(combo, force=force)
            if result:
                stats["merged"] += 1
                count = sum(1 for _ in open(result))
                stats["total_queries"] += count
                if verbose:
                    print(f"  ✓ Merged {combo.name}: {len(combo.individual_query_files)} files → {count} queries")
        
        return stats


def discover_all_query_files(base_path: str = "rollout/tools/datasets/cross_domain") -> List[Dict]:
    """
    Discover all query files including per-task files.
    
    Returns list of dicts with query file info.
    """
    queries_dir = Path(base_path) / "queries" / "cross_domain"
    results = []
    
    if not queries_dir.exists():
        return results
    
    for domain_dir in queries_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        
        for query_file in domain_dir.glob("*.jsonl"):
            results.append({
                "domain": domain_dir.name,
                "file": query_file.name,
                "path": str(query_file),
                "is_combined": query_file.name == "queries.jsonl"
            })
    
    return results


if __name__ == "__main__":
    # Demo usage
    manager = CrossDomainManager()
    
    print("=" * 60)
    print("Discovered Cross-Domain Combinations")
    print("=" * 60)
    
    combos = manager.discover_combinations()
    print(f"Found {len(combos)} combinations\n")
    
    for combo in combos[:5]:  # Show first 5
        print(f"Name: {combo.name}")
        print(f"  Domains: {', '.join(combo.domains)}")
        print(f"  Tools: {', '.join(combo.tools)}")
        print(f"  Has queries: {combo.query_file is not None}")
        print(f"  Has policy: {combo.policy_file is not None}")
        if combo.policy_file:
            print(f"  Policy: {combo.policy_file.name}")
        print()

