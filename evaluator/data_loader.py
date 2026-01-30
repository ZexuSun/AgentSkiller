"""
Data Loader - Unified data loading for evaluation from rollouts and queries.

This module provides:
1. EvaluationSample: Unified data structure for evaluation
2. EvaluationDataLoader: Load and correlate data from multiple sources

Data Flow:
    rollouts/*.jsonl  →  sample ID  →  queries/*.jsonl  →  validated_tasks/
                                           ↓
                                    EvaluationSample
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def compute_combo_id(trajectory_hash: str, entity_context: Dict[str, Any]) -> str:
    """
    Compute a unique ID for a combo based on its content.
    
    This matches the formula used in s17_task_instantiation.py.
    
    Args:
        trajectory_hash: The trajectory hash string
        entity_context: The entity context dict (entity_instances)
        
    Returns:
        12-character hex string combo_id
    """
    content = json.dumps({
        "trajectory_hash": trajectory_hash,
        "entity_context": entity_context,
    }, sort_keys=True, ensure_ascii=False)
    
    return hashlib.md5(content.encode()).hexdigest()[:12]


def extract_uuids_from_text(text: str) -> set:
    """Extract all UUIDs from a text string."""
    import re
    if not isinstance(text, str):
        # Handle case where known_info might be a dict
        text = json.dumps(text) if text else ""
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    return set(re.findall(uuid_pattern, text.lower()))


def build_entity_context_from_combo(combo: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete entity_context from a combo entry.
    
    Preserves the original semantic structure:
    - entity_instances: Nested dict of entities/relationships with their fields
    - value_domain_samples: Direct Server.Function.param -> value mappings
    
    This structure is clearer for the LLM to understand:
    - value_domain_samples contains exact parameter values for tool calls
    - entity_instances contains full entity records for reference
    """
    return {
        "entity_instances": combo.get("entity_instances", {}),
        "value_domain_samples": combo.get("value_domain_samples", {}),
    }


def find_combo_by_known_info(
    combos: List[Dict[str, Any]],
    known_info: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a combo by matching entity IDs from known_info text.
    
    Deprecated: Use find_combo_by_combo_id() instead for direct lookup.
    This function is kept for backward compatibility with old data that
    doesn't have combo_id field.
    
    This function extracts UUIDs from known_info and finds the combo whose
    entity_instances contain the same UUIDs.
    
    Args:
        combos: List of combo dicts with entity_instances
        known_info: Text containing entity info with IDs
        
    Returns:
        The matching combo dict or None
    """
    if not known_info:
        return combos[0] if combos else None
    
    # Extract UUIDs from known_info
    target_uuids = extract_uuids_from_text(known_info)
    
    if not target_uuids:
        return combos[0] if combos else None
    
    # Find combo with matching UUIDs
    for combo in combos:
        entity_instances = combo.get("entity_instances", {})
        
        # Collect all UUIDs from this combo's entity_instances
        combo_uuids = set()
        for entity_data in entity_instances.values():
            if isinstance(entity_data, dict):
                for value in entity_data.values():
                    if isinstance(value, str) and len(value) == 36 and '-' in value:
                        combo_uuids.add(value.lower())
        
        # Check if all combo UUIDs are in target UUIDs
        if combo_uuids and combo_uuids.issubset(target_uuids):
            return combo
    
    return None


def find_combo_by_id(
    combos: List[Dict[str, Any]],
    target_combo_id: str,
    trajectory_hash: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a combo in the combos array by computing its combo_id.
    
    Deprecated: Use find_combo_by_combo_id() instead for direct lookup.
    This function computes combo_id on-the-fly which is slower.
    """
    for combo in combos:
        entity_instances = combo.get("entity_instances", {})
        computed_id = compute_combo_id(trajectory_hash, entity_instances)
        if computed_id == target_combo_id:
            return combo
    
    return None


def find_combo_by_combo_id(
    combos: List[Dict[str, Any]],
    combo_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a combo by its pre-computed combo_id field.
    
    This is the preferred method after the combo_id fix (s15/s17 updates).
    Each combo now has a unique combo_id stored directly in the combo dict.
    
    Args:
        combos: List of combo dicts from validated_combos.json
        combo_id: The combo_id to search for
        
    Returns:
        The matching combo dict, or None if not found
    """
    if not combo_id:
        return None
    
    for combo in combos:
        if combo.get("combo_id") == combo_id:
            return combo
    
    return None


def compute_trajectory_hash(trajectory: List[str]) -> str:
    """
    Compute a unique hash for a trajectory.
    
    Args:
        trajectory: List of function names in order
        
    Returns:
        A short hash string (12 chars) that uniquely identifies the trajectory
    """
    trajectory_str = "_".join(trajectory)
    full_hash = hashlib.md5(trajectory_str.encode()).hexdigest()
    return full_hash[:12]


@dataclass
class EvaluationSample:
    """
    Unified data structure for evaluation samples.
    
    Data Sources:
    - rollouts/*.jsonl: Agent conversation/tool call records
    - outputs/queries/: Task definitions with trajectory and context
    - outputs/validated_tasks/: Entity context for the task
    
    Attributes:
        id: Unique identifier for the sample
        rollout_messages: Agent's conversation messages (OpenAI format)
        query: Query definition from queries file
        trajectory: Expected tool call sequence
        entity_context: Entity context values for the task
        domains: List of domains involved
        instruction: Task instruction text
        nl_assertions: Natural language assertions for evaluation
        is_cross_domain: Whether this is a cross-domain task
    """
    id: str
    rollout_messages: List[Dict[str, Any]] = field(default_factory=list)
    query: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[str] = field(default_factory=list)
    entity_context: Dict[str, Any] = field(default_factory=dict)
    domains: List[str] = field(default_factory=list)
    instruction: str = ""
    nl_assertions: List[str] = field(default_factory=list)
    is_cross_domain: bool = False
    
    # Additional metadata from rollout
    rollout_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    rollout_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "rollout_messages": self.rollout_messages,
            "query": self.query,
            "trajectory": self.trajectory,
            "entity_context": self.entity_context,
            "domains": self.domains,
            "instruction": self.instruction,
            "nl_assertions": self.nl_assertions,
            "is_cross_domain": self.is_cross_domain,
            "rollout_tool_calls": self.rollout_tool_calls,
            "rollout_metadata": self.rollout_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSample":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            rollout_messages=data.get("rollout_messages", []),
            query=data.get("query", {}),
            trajectory=data.get("trajectory", []),
            entity_context=data.get("entity_context", {}),
            domains=data.get("domains", []),
            instruction=data.get("instruction", ""),
            nl_assertions=data.get("nl_assertions", []),
            is_cross_domain=data.get("is_cross_domain", False),
            rollout_tool_calls=data.get("rollout_tool_calls", []),
            rollout_metadata=data.get("rollout_metadata", {}),
        )
    
    def get_agent_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Extract tool calls from agent's rollout messages.
        
        Returns:
            List of tool call dictionaries with name and arguments
        """
        if self.rollout_tool_calls:
            return self.rollout_tool_calls
        
        tool_calls = []
        for msg in self.rollout_messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    if "function" in tc:
                        func = tc["function"]
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        
                        # Strip domain prefix if present
                        func_name = func["name"].split(".")[-1]
                        
                        tool_calls.append({
                            "name": func_name,
                            "arguments": args,
                        })
        
        return tool_calls


class EvaluationDataLoader:
    """
    Loads and correlates evaluation data from multiple sources.
    
    Data Sources:
    - rollouts_dir: Contains JSONL files with agent rollout data
    - outputs_dir: Contains queries/ and validated_tasks/ subdirectories
    
    File Structure Expected:
        rollouts/
            *.jsonl         # Each line is a rollout sample with 'id' field
        outputs/
            queries/
                <domain>.jsonl  # Query files named by domain (single or fused cross-domain)
                                # e.g., AccessControlServer.jsonl (single)
                                # e.g., ServerA_ServerB_ServerC.jsonl (cross-domain)
            validated_tasks/
                <domain>/
                    validated_combos.json  # Entity instances for queries
    
    Data Flow:
        1. Query files provide: domain_key, domains list, combo_id, trajectory
        2. domain_key is used to lookup validated_tasks/<domain_key>/validated_combos.json
        3. combo_id is used to find the specific entity_context
        4. len(domains) > 1 determines is_cross_domain
    
    Usage:
        loader = EvaluationDataLoader(
            rollouts_dir="rollouts/",
            outputs_dir="outputs/"
        )
        
        # Load all samples
        samples = loader.load_all_samples()
        
        # Or iterate through samples
        for sample in loader.iter_samples():
            print(sample.id, sample.trajectory, sample.is_cross_domain)
    """
    
    def __init__(
        self,
        rollouts_dir: str = "rollouts/",
        outputs_dir: str = "outputs/",
    ):
        """
        Initialize the data loader.
        
        Args:
            rollouts_dir: Directory containing rollout JSONL files
            outputs_dir: Directory containing queries and validated_tasks
        """
        self.rollouts_dir = Path(rollouts_dir)
        self.outputs_dir = Path(outputs_dir)
        self.queries_dir = self.outputs_dir / "queries"
        self.validated_tasks_dir = self.outputs_dir / "validated_tasks"
        
        # Caches for loaded data
        self._queries_cache: Dict[str, Dict] = {}
        self._validated_combos_cache: Dict[str, List[Dict]] = {}  # domain_key -> combos
        
        logger.info(f"Initialized EvaluationDataLoader:")
        logger.info(f"  Rollouts: {self.rollouts_dir}")
        logger.info(f"  Queries: {self.queries_dir}")
        logger.info(f"  Validated Tasks: {self.validated_tasks_dir}")
    
    def _load_queries(self) -> None:
        """Load all query files into cache.
        
        Query files are .jsonl files named by domain:
        - Single domain: "AccessControlServer.jsonl"
        - Cross-domain: "ServerA_ServerB_ServerC.jsonl" (fused name)
        
        For each query item, we set:
        - domain_key: The fused name (for validated_tasks lookup)
        - domains: List of individual server names (split by "_")
        """
        if self._queries_cache:
            return
        
        if not self.queries_dir.exists():
            logger.warning(f"Queries directory not found: {self.queries_dir}")
            return
        
        # Load JSONL files
        for query_file in self.queries_dir.glob("*.jsonl"):
            try:
                # Extract domain key from filename
                # e.g., "ServerA_ServerB_ServerC.jsonl" -> "ServerA_ServerB_ServerC"
                domain_key = query_file.stem
                
                # Split to get individual domains for is_cross_domain detection
                # e.g., "ServerA_ServerB_ServerC" -> ["ServerA", "ServerB", "ServerC"]
                domains = domain_key.split("_")
                
                with open(query_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            if "id" in item:
                                # Merge task_info into the item for easier access
                                if "task_info" in item:
                                    task_info = item["task_info"]
                                    item["trajectory"] = task_info.get("trajectory", [])
                                    item["instruction"] = task_info.get("instruction", "")
                                    item["trajectory_hash"] = task_info.get("trajectory_hash", "")
                                    item["combo_id"] = task_info.get("combo_id", "")
                                
                                # Set domain_key (fused name for validated_tasks lookup)
                                item["domain_key"] = domain_key
                                
                                # Set domains list (for is_cross_domain detection)
                                item["domains"] = domains
                                
                                self._queries_cache[item["id"]] = item
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to load query file {query_file}: {e}")
        
        logger.info(f"Loaded {len(self._queries_cache)} queries")
    
    def _get_query_by_id(self, sample_id: str) -> Optional[Dict]:
        """Get query data by sample ID."""
        self._load_queries()
        return self._queries_cache.get(sample_id)
    
    def _load_validated_combos(self, domain: str) -> List[Dict]:
        """
        Load validated_combos.json for a domain.
        
        This is the source of truth for entity_instances that were used
        to generate the queries.
        """
        if domain in self._validated_combos_cache:
            return self._validated_combos_cache[domain]
        
        combos_file = self.validated_tasks_dir / domain / "validated_combos.json"
        
        if not combos_file.exists():
            logger.debug(f"No validated_combos.json for domain: {domain}")
            self._validated_combos_cache[domain] = []
            return []
        
        try:
            with open(combos_file) as f:
                combos = json.load(f)
            self._validated_combos_cache[domain] = combos
            logger.debug(f"Loaded {len(combos)} validated combos for {domain}")
            return combos
        except Exception as e:
            logger.warning(f"Failed to load validated_combos for {domain}: {e}")
            self._validated_combos_cache[domain] = []
            return []
    
    def _parse_rollout_line(self, line: str) -> Optional[Dict]:
        """Parse a single line from rollout JSONL."""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse rollout line: {e}")
            return None
    
    def _create_sample_from_rollout(self, rollout_data: Dict) -> Optional[EvaluationSample]:
        """
        Create an EvaluationSample from rollout data.
        
        Data flow:
        1. Get query data by sample_id (contains domain_key, domains, combo_id)
        2. Use domain_key for validated_tasks lookup
        3. Use combo_id to find entity_context from validated_combos.json
        4. Use len(domains) > 1 for is_cross_domain detection
        
        Args:
            rollout_data: Raw rollout data from JSONL
            
        Returns:
            EvaluationSample or None if data is incomplete
        """
        sample_id = rollout_data.get("id", "")
        if not sample_id:
            logger.warning("Rollout missing 'id' field")
            return None
        
        # Extract messages from rollout
        messages = rollout_data.get("messages", [])
        if not messages and "conversation" in rollout_data:
            messages = rollout_data["conversation"]
        
        # Extract tool calls if provided directly
        tool_calls = rollout_data.get("tool_calls", [])
        
        # Get query data (primary source of truth)
        query = self._get_query_by_id(sample_id) or {}
        
        # Extract trajectory from query or rollout
        trajectory = (
            query.get("trajectory", []) or
            rollout_data.get("trajectory", []) or
            rollout_data.get("golden_trajectory", [])
        )
        
        # Get domain_key (fused name for validated_tasks lookup)
        # and domains (list for is_cross_domain detection)
        domain_key = query.get("domain_key", "")
        domains = query.get("domains", [])
        
        # Fallback: construct from domains list if domain_key not set
        if not domain_key and domains:
            domain_key = "_".join(domains)
        
        # Extract entity context using combo_id from validated_combos.json
        entity_context = {}
        combo_id = query.get("combo_id") or query.get("task_info", {}).get("combo_id")
        
        if combo_id and domain_key:
            validated_combos = self._load_validated_combos(domain_key)
            if validated_combos:
                matched_combo = find_combo_by_combo_id(validated_combos, combo_id)
                if matched_combo:
                    entity_context = build_entity_context_from_combo(matched_combo)
                    logger.debug(f"Matched combo by combo_id={combo_id} with {len(entity_context)} keys")
        
        # Fallback: direct entity_instances in rollout data
        if not entity_context and rollout_data.get("entity_context"):
            entity_context = rollout_data["entity_context"]
        
        # Extract instruction from query
        instruction = query.get("instruction", "") or rollout_data.get("instruction", "")
        
        # Extract NL assertions
        nl_assertions = query.get("nl_assertions", []) or rollout_data.get("nl_assertions", [])
        
        # Determine if cross-domain based on number of domains
        is_cross_domain = len(domains) > 1
        
        # Additional metadata
        metadata = {
            k: v for k, v in rollout_data.items()
            if k not in ["id", "messages", "conversation", "tool_calls", 
                        "trajectory", "entity_context", "domains", "instruction"]
        }
        
        return EvaluationSample(
            id=sample_id,
            rollout_messages=messages,
            query=query,
            trajectory=trajectory,
            entity_context=entity_context,
            domains=domains,
            instruction=instruction,
            nl_assertions=nl_assertions,
            is_cross_domain=is_cross_domain,
            rollout_tool_calls=tool_calls,
            rollout_metadata=metadata,
        )
    
    def iter_rollout_files(self) -> Iterator[Path]:
        """Iterate through all rollout JSONL files."""
        if not self.rollouts_dir.exists():
            logger.warning(f"Rollouts directory not found: {self.rollouts_dir}")
            return
        
        for path in self.rollouts_dir.glob("*.jsonl"):
            yield path
        
        # Also check for .json files (single sample per file)
        for path in self.rollouts_dir.glob("*.json"):
            yield path
    
    def iter_samples(self, verbose: bool = False) -> Iterator[EvaluationSample]:
        """
        Iterate through all evaluation samples.
        
        Args:
            verbose: Print progress information
            
        Yields:
            EvaluationSample instances
        """
        sample_count = 0
        
        for rollout_file in self.iter_rollout_files():
            if verbose:
                print(f"Loading rollouts from: {rollout_file.name}")
            
            try:
                with open(rollout_file) as f:
                    if rollout_file.suffix == ".jsonl":
                        # JSONL format - one sample per line
                        for line in f:
                            if not line.strip():
                                continue
                            
                            rollout_data = self._parse_rollout_line(line)
                            if rollout_data:
                                sample = self._create_sample_from_rollout(rollout_data)
                                if sample:
                                    sample_count += 1
                                    yield sample
                    else:
                        # JSON format - single sample or list
                        data = json.load(f)
                        if isinstance(data, list):
                            for rollout_data in data:
                                sample = self._create_sample_from_rollout(rollout_data)
                                if sample:
                                    sample_count += 1
                                    yield sample
                        elif isinstance(data, dict):
                            sample = self._create_sample_from_rollout(data)
                            if sample:
                                sample_count += 1
                                yield sample
                                
            except Exception as e:
                logger.error(f"Failed to load rollout file {rollout_file}: {e}")
        
        if verbose:
            print(f"Total samples loaded: {sample_count}")
    
    def load_all_samples(self, verbose: bool = False) -> List[EvaluationSample]:
        """
        Load all evaluation samples into a list.
        
        Args:
            verbose: Print progress information
            
        Returns:
            List of EvaluationSample instances
        """
        return list(self.iter_samples(verbose=verbose))
    
    def load_sample_by_id(self, sample_id: str) -> Optional[EvaluationSample]:
        """
        Load a specific sample by ID.
        
        Note: This iterates through all rollouts to find the sample.
        For better performance with many lookups, use load_all_samples()
        and build your own index.
        
        Args:
            sample_id: The sample ID to find
            
        Returns:
            EvaluationSample or None if not found
        """
        for sample in self.iter_samples():
            if sample.id == sample_id:
                return sample
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary with statistics
        """
        samples = self.load_all_samples()
        
        domains = set()
        trajectory_lengths = []
        cross_domain_count = 0
        
        for sample in samples:
            domains.update(sample.domains)
            trajectory_lengths.append(len(sample.trajectory))
            if sample.is_cross_domain:
                cross_domain_count += 1
        
        return {
            "total_samples": len(samples),
            "unique_domains": list(domains),
            "cross_domain_samples": cross_domain_count,
            "single_domain_samples": len(samples) - cross_domain_count,
            "avg_trajectory_length": sum(trajectory_lengths) / len(trajectory_lengths) if trajectory_lengths else 0,
            "min_trajectory_length": min(trajectory_lengths) if trajectory_lengths else 0,
            "max_trajectory_length": max(trajectory_lengths) if trajectory_lengths else 0,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def load_evaluation_data(
    rollouts_dir: str = "rollouts/",
    outputs_dir: str = "outputs/",
    verbose: bool = False,
) -> List[EvaluationSample]:
    """
    Convenience function to load all evaluation samples.
    
    Args:
        rollouts_dir: Directory containing rollout JSONL files
        outputs_dir: Directory containing queries and validated_tasks
        verbose: Print progress information
        
    Returns:
        List of EvaluationSample instances
    """
    loader = EvaluationDataLoader(
        rollouts_dir=rollouts_dir,
        outputs_dir=outputs_dir,
    )
    return loader.load_all_samples(verbose=verbose)


def create_sample_from_task_data(
    task_data: Dict[str, Any],
    rollout_messages: List[Dict[str, Any]],
) -> EvaluationSample:
    """
    Create an EvaluationSample from task data and rollout messages.
    
    This is useful when you have task data from a task file and want to
    combine it with agent rollout messages for evaluation.
    
    Args:
        task_data: Task data dictionary (from task JSON files)
        rollout_messages: Agent's conversation messages
        
    Returns:
        EvaluationSample instance
    """
    # Generate ID if not present
    sample_id = task_data.get("task_id", "")
    if not sample_id:
        traj_str = "_".join(task_data.get("trajectory", []))
        sample_id = hashlib.md5(traj_str.encode()).hexdigest()[:12]
    
    # Extract instruction
    instruction = ""
    if task_data.get("instantiated_task"):
        instruction = task_data["instantiated_task"].get("instruction", "")
    elif task_data.get("template"):
        instruction = task_data["template"].get("instruction", "")
    
    return EvaluationSample(
        id=sample_id,
        rollout_messages=rollout_messages,
        query=task_data,
        trajectory=task_data.get("trajectory", []),
        entity_context=task_data.get("entity_context", {}),
        domains=task_data.get("domains", []),
        instruction=instruction,
        nl_assertions=task_data.get("nl_assertions", []),
        is_cross_domain=task_data.get("is_cross_domain", False),
    )

