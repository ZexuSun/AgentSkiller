#!/usr/bin/env python3
"""
Batch Rollout Script for Cross-Domain Combinations

Automatically discovers and processes all cross-domain combinations
without requiring manual configuration for each one.

Usage:
    # Process all combinations
    python scripts/batch_rollout.py --all

    # Process specific combinations (order doesn't matter)
    python scripts/batch_rollout.py --domains StudentAcademicPortal StudentFinancialServices StudentHealthServices

    # List all available combinations
    python scripts/batch_rollout.py --list

    # Generate config files for manual review
    python scripts/batch_rollout.py --generate-configs --output-dir configs/generated

    # Process with specific model
    python scripts/batch_rollout.py --all --agent-model openai/deepseek-v3.2 --user-model openai/gpt-5
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rollout.utils.cross_domain import (
    CrossDomainManager,
    CrossDomainCombo,
    extract_domains_smart,
    to_snake_case
)

from concurrent.futures import ThreadPoolExecutor, as_completed


def list_combinations(manager: CrossDomainManager, verbose: bool = False):
    """List all discovered combinations."""
    combos = manager.discover_combinations()
    
    print(f"\n{'=' * 70}")
    print(f"  Found {len(combos)} Cross-Domain Combinations")
    print(f"{'=' * 70}\n")
    
    # Group by domain count
    by_count = {}
    for combo in combos:
        n = len(combo.domains)
        if n not in by_count:
            by_count[n] = []
        by_count[n].append(combo)
    
    needs_merge_count = 0
    
    for count in sorted(by_count.keys()):
        print(f"  {count}-Domain Combinations ({len(by_count[count])} total):")
        print(f"  {'-' * 50}")
        
        for combo in by_count[count]:
            status = []
            if combo.query_file and combo.query_file.exists():
                status.append("✓ queries")
            elif combo.individual_query_files:
                status.append(f"⚠ {len(combo.individual_query_files)} files (needs merge)")
                needs_merge_count += 1
            else:
                status.append("✗ queries")
            
            if combo.policy_file:
                status.append("✓ policy")
            else:
                status.append("✗ policy")
            
            status_str = " | ".join(status)
            
            if verbose:
                print(f"    {combo.name}")
                print(f"      Domains: {', '.join(combo.domains)}")
                print(f"      Tools:   {', '.join(combo.tools)}")
                print(f"      Status:  {status_str}")
                if combo.policy_file:
                    print(f"      Policy:  {combo.policy_file.name}")
                if combo.individual_query_files:
                    print(f"      Files:   {', '.join(f.name for f in combo.individual_query_files[:3])}{'...' if len(combo.individual_query_files) > 3 else ''}")
                print()
            else:
                print(f"    {combo.name} [{status_str}]")
        
        print()
    
    if needs_merge_count > 0:
        print(f"  💡 {needs_merge_count} combination(s) need query merging.")
        print(f"     Run with --merge-queries to merge them automatically.\n")


def generate_config_file(
    combos: List[CrossDomainCombo],
    output_path: str,
    base_config: dict
):
    """Generate a YAML config file for the given combinations."""
    import yaml
    
    config = {
        **base_config,
        "datasets": {}
    }
    
    for combo in combos:
        dataset_config = {
            "path": str(combo.query_file) if combo.query_file else f"./queries/{combo.name}/queries.jsonl",
            "output_path": f"./outputs_tmp/{combo.name}_output.jsonl",
            "mcp_domain": combo.name,
            "tools": combo.tools,
            "agent": {
                "model": base_config.get("agent_model", "openai/deepseek-v3.2"),
                "temperature": 0.7,
                # "enable_thinking": True,
                "thinking": {
                    "type": "enabled"
                }
            },
            "user": {
                "model": base_config.get("user_model", "openai/gpt-5"),
                "temperature": 1.0
            },
            "max_turns": 20,
            "max_steps_per_turn": 10,
            "mode": "positive"
        }
        
        if combo.policy_file:
            dataset_config["agent"]["system_prompt_file"] = str(combo.policy_file)
        
        # Use a sanitized name as key
        key = combo.name.replace("-", "_")
        config["datasets"][key] = dataset_config
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Generated config: {output_path}")


def run_single_combination(
    combo: CrossDomainCombo,
    agent_model: str,
    user_model: str,
    output_dir: str,
    max_turns: int = 20,
    max_workers: int = 4,
    resume: bool = True,
    verbose: bool = True,
    models_config_file: Optional[str] = None
):
    """Run rollout for a single combination."""
    from rollout import Agent, SimulatedUser, Pipeline
    from rollout.core.models import load_models_config
    from rollout.tools import setup_mcp_tools, collect_tools_info
    from rollout.core.checkpoint import CheckpointManager
    from rollout.core.pipeline import PipelineConfig
    
    # Load models config
    models_config_path = Path(models_config_file or "configs/models.yml")
    if models_config_path.exists():
        load_models_config(str(models_config_path))
        if verbose:
            print(f"  ✓ Loaded models config: {models_config_path}")
    elif models_config_file:
        print(f"  ⚠ Warning: Models config file not found: {models_config_path}")
    
    # Setup output path
    # timestamp = datetime.now().strftime("%m%d%H%M")
    output_path = Path(output_dir) / f"{combo.name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print(f"Processing: {combo.name}")
    print(f"  Domains: {', '.join(combo.domains)}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")
    
    # Check prerequisites
    if not combo.query_file or not combo.query_file.exists():
        print(f"  ✗ Skipping: No query file found")
        return False
    
    if not combo.policy_file:
        print(f"  ⚠ Warning: No policy file found, using default")
    
    # Setup MCP tools
    # Each MCP Server file is separate, but they all need the full combined domain_name
    # (e.g., "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices")
    # to load the correct database from database/outputs/entities/{domain_name}/
    # combo.domains contains CamelCase server names (e.g., "StudentAcademicPortal")
    # combo.name is the full cross-domain combination name
    print(f"Domain Name: {combo.name} | Query Name: {combo.query_file}")
    tools = setup_mcp_tools(domain_name=combo.name, tool_names=combo.domains)
    if not tools:
        print(f"  ✗ Skipping: Failed to setup MCP tools")
        return False
    
    tools_info = collect_tools_info(tools)
    print(f"  ✓ Loaded {len(tools_info)} tools")
    
    # Load system prompt
    system_prompt = ""
    if combo.policy_file and combo.policy_file.exists():
        system_prompt = combo.policy_file.read_text()
        print(f"  ✓ Loaded policy: {combo.policy_file.name}")
    
    # Initialize agent and user
    agent = Agent(
        model=agent_model,
        system_prompt=system_prompt,
        tools=tools_info,
        temperature=0.7,
        enable_thinking=True,
        thinking={"type": "enabled"}
    )
    
    user = SimulatedUser(
        model=user_model,
        temperature=1.0
    )
    
    # Setup checkpoint manager
    checkpoint_dir = Path("./checkpoints") / combo.name
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        max_turns=max_turns,
        max_steps_per_turn=20,
        mode="positive",
        enable_monitor=True,
        monitor_max_no_tool_turns=5,
        monitor_rule_detection=False,
        verbose=verbose
    )
    
    pipeline = Pipeline(
        agent=agent,
        user=user,
        tools=tools,
        config=pipeline_config,
        checkpoint_manager=checkpoint_manager
    )
    
    # Load queries
    queries = []
    with open(combo.query_file, "r") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    print(f"  ✓ Loaded {len(queries)} queries")
    
    # Filter for resume
    if resume:
        completed_ids = set()
        if output_path.exists():
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        completed_ids.add(data.get("id"))
        
        queries = [q for q in queries if q.get("id") not in completed_ids]
        if completed_ids:
            print(f"  ✓ Resuming: {len(completed_ids)} already completed, {len(queries)} remaining")
    
    if not queries:
        print(f"  ✓ All queries already completed")
        return True
    
    # Process queries
    print(f"\n  Processing {len(queries)} queries...\n")
    
    with open(output_path, "a") as f:
        for i, query in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {query.get('id', 'unknown')[:16]}...")
            
            try:
                result = pipeline.process(query)
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
                
                status = "✓" if result.success else "✗"
                total_turns = result.metadata.get("total_turns", 0)
                stop_reason = result.metadata.get("stop_reason", "unknown")
                print(f"    {status} {total_turns} turns, stop_reason: {stop_reason}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
    
    print(f"\n  ✓ Completed: {output_path}")
    return True


def find_combos_by_domains(
    manager: CrossDomainManager,
    domain_names: List[str]
) -> List[CrossDomainCombo]:
    """Find combinations matching the given domains (order-agnostic)."""
    target_set = frozenset(domain_names)
    
    results = []
    for combo in manager.discover_combinations():
        if combo.domain_set == target_set:
            results.append(combo)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch Rollout for Cross-Domain Combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Action modes
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--all", action="store_true",
                              help="Process all discovered combinations")
    action_group.add_argument("--domains", nargs="+",
                              help="Process specific domain combination (order doesn't matter)")
    action_group.add_argument("--list", action="store_true",
                              help="List all available combinations")
    action_group.add_argument("--generate-configs", action="store_true",
                              help="Generate config files for all combinations")
    action_group.add_argument("--merge-queries", action="store_true",
                              help="Merge individual query files into queries.jsonl")
    
    # Model settings
    parser.add_argument("--agent-model", default="openai/deepseek-v3.2-fc",
                        help="Agent model (default: openai/deepseek-v3.2)")
    parser.add_argument("--user-model", default="openai/gpt-5",
                        help="User simulator model (default: openai/gpt-5)")
    parser.add_argument("--models-config", default="configs/models.yml",
                        help="Path to models configuration file (default: configs/models.yml)")
    parser.add_argument("--query-dir", type=str, default="rollout/tools/datasets/cross_domain/queries",
                        help="The query directory to use")
    
    # Output settings
    parser.add_argument("--output-dir", default="./outputs_cross_domain_0114",
                        help="Output directory for results")
    parser.add_argument("--config-output-dir", default="./configs/generated",
                        help="Output directory for generated configs")
    
    # Processing settings
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max conversation turns")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Parallel workers")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from previous runs")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose listing output")
    
    # Data path
    parser.add_argument("--base-path", default="rollout/tools/datasets/cross_domain",
                        help="Base path for cross-domain data")
    
    # Filtering
    parser.add_argument("--min-domains", type=int, default=2,
                        help="Minimum domains for cross-domain")
    parser.add_argument("--require-policy", action="store_true",
                        help="Only process combinations with policy files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of combinations to process")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force overwrite existing files")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = CrossDomainManager(args.base_path)
    
    if args.list:
        list_combinations(manager, verbose=args.verbose)
        return
    
    if args.generate_configs:
        combos = manager.discover_combinations(min_domains=args.min_domains)
        
        if args.require_policy:
            combos = [c for c in combos if c.policy_file]
        
        if args.limit > 0:
            combos = combos[:args.limit]
        
        # Create output directory
        config_dir = Path(args.config_output_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Base config template
        base_config = {
            "max_workers": args.max_workers,
            "resume": not args.no_resume,
            "use_checkpoints": True,
            "checkpoint_dir": "./checkpoints",
            "verbose": not args.quiet,
            "verbose_colors": True,
            "enable_monitor": True,
            "monitor_rule_detection": False,
            "monitor_max_no_tool_turns": 5,
            "models_config_file": args.models_config,
            "agent_model": args.agent_model,
            "user_model": args.user_model
        }
        
        # Generate individual configs or one combined config
        # Combined config
        output_path = config_dir / "all_cross_domain.yml"
        generate_config_file(combos, str(output_path), base_config)
        
        print(f"\nGenerated configs for {len(combos)} combinations")
        return
    
    # Get combinations to process
    if args.all:
        combos = manager.discover_combinations(min_domains=args.min_domains)
    else:
        combos = find_combos_by_domains(manager, args.domains)
        if not combos:
            print(f"No combination found for domains: {args.domains}")
            print("Note: Domain order doesn't matter")
            return
    
    if args.require_policy:
        combos = [c for c in combos if c.policy_file]
    
    if args.limit > 0:
        combos = combos[:args.limit]
    
    print(f"\nWill process {len(combos)} combinations")
    
    # Process each combination
    success_count = 0
    fail_count = 0

    print(args.max_workers)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        print(combos)
        futures = [
            executor.submit(
                run_single_combination,
                combo,
                args.agent_model,
                args.user_model,
                args.output_dir,
                args.max_turns,
                args.max_workers,
                not args.no_resume,
                not args.quiet,
                args.models_config
            )
            for combo in combos
        ]

        for future in as_completed(futures):
            success = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"  Batch Complete: {success_count} success, {fail_count} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

