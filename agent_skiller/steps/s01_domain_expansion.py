"""
Step 1: Domain Expansion

Expand seed domains into diverse sub-domains using LLM.

Input: SEED_DOMAINS (built-in constant)
Output: domain_topics.json
"""

import logging
from pathlib import Path
from typing import Set

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.console import Console

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import SEED_DOMAINS, DOMAIN_EXPANSION_PROMPT
from .base import step_handler, save_json, load_json, get_client

logger = logging.getLogger(__name__)
console = Console()


@step_handler("s01_domain_expansion", auto_retry=True)
def domain_expansion_step(state: WorkflowState) -> WorkflowState:
    """
    Expand seed domains into diverse sub-domains.
    
    Process:
    1. Start with seed domains
    2. Iteratively expand each domain using LLM
    3. Check diversity and deduplicate
    4. Continue until target number reached
    
    Output:
    - domain_topics.json: List of unique domain topics
    """
    settings = get_settings()
    step_config = settings.steps.s01_domain_expansion
    
    target_count = step_config.get("target_domain_number", 300)
    batch_size = step_config.get("expansion_batch_size", 20)
    
    outputs_dir = settings.paths.outputs_dir
    output_path = outputs_dir / "domain_topics.json"
    
    # Load existing topics if resuming
    if output_path.exists():
        existing = load_json(output_path)
        topics: Set[str] = set(existing.get("domains", []))
        logger.info(f"Resuming with {len(topics)} existing topics")
    else:
        topics = set(SEED_DOMAINS)
    
    client = get_client()
    
    # Initialize step progress
    state.update_step_progress(
        "s01_domain_expansion",
        total=target_count,
        completed=len(topics)
    )
    
    # Expand until we reach target with progress bar
    progress_columns = [
        TextColumn("[cyan]Domain Expansion"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ]
    
    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("Expanding", total=target_count, completed=len(topics))
        
        while len(topics) < target_count:
            # Generate expansion prompt using template
            prompt = DOMAIN_EXPANSION_PROMPT.format(
                existing_domains="\n".join(f"- {d}" for d in topics),
                num_new_domains=batch_size
            )
            
            try:
                response = client.chat(query=prompt, model_type="textual")
                result = response.parse_json()
                
                # Handle both formats: {"domains": [...]} or [...]
                if isinstance(result, dict) and "domains" in result:
                    new_domains = result["domains"]
                elif isinstance(result, list):
                    new_domains = result
                else:
                    new_domains = []
                
                for domain in new_domains:
                    if isinstance(domain, str) and domain not in topics:
                        topics.add(domain)
            except Exception as e:
                logger.warning(f"Expansion failed: {e}")
            
            # Save progress
            save_json({"domains": sorted(topics)}, output_path)
            
            # Update progress bar and state
            progress.update(task, completed=len(topics))
            state.update_step_progress(
                "s01_domain_expansion",
                total=target_count,
                completed=len(topics)
            )
    
    # Final save
    save_json({"domains": sorted(topics)}, output_path)
    state.domain_topics_path = str(output_path)
    state.domain_topics = topics
    
    logger.info(f"Domain expansion complete: {len(topics)} topics")
    return state
