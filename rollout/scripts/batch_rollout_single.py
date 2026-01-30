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

from rollout.utils.cross_domain import to_snake_case
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def list_domains(query_dir: Path):
    domains = [
        d.stem for d in query_dir.iterdir() # DomainName.jsonl
    ]

    print(f"\n{'=' * 70}")
    print(f"  Found {len(domains)} Combinations")
    print(f"{'=' * 70}\n")

    return domains


def generate_config_file(
    domains: List[str],
    output_path: str,
    query_dir: Path,
    policy_dir: Path,
    base_config: dict
):
    """Generate a YAML config file for the given combinations."""
    import yaml
    
    config = {
        **base_config,
        "datasets": {}
    }
    
    for domain in domains:
        dataset_config = {
            "path": str(query_dir / f"{domain}.jsonl"),
            "output_path": f"./outputs/{domain}.jsonl",
            "mcp_domain": domain,
            "tools": [to_snake_case(domain)],
            "agent": {
                "model": base_config.get("agent_model", "openai/deepseek-v3.2"),
                "temperature": base_config.get("agent_temperature", 0.7),
                "enable_thinking": base_config.get("enable_thinking", True)
            },
            "user": {
                "model": base_config.get("user_model", "openai/gpt-5"),
                "temperature": base_config.get("user_temperature", 1.0)
            },
            "max_turns": base_config.get("max_turns", 20),
            "max_steps_per_turn": base_config.get("max_steps_per_turn", 10),
            "mode": "positive"
        }
        
        policy_file = policy_dir / f"{domain}.md"
        if policy_file.exists():
            dataset_config["agent"]["system_prompt_file"] = str(policy_file)
        
        config["datasets"][domain] = dataset_config
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Generated config: {output_path}")


def run_single_domain(
    domain: str,
    agent_model: str,
    user_model: str,
    output_path: Path,
    query_file: Path,
    policy_file: Path,
    tools: List[str],
    max_turns: int = 20,
    max_steps_per_turn: int = 10,
    agent_temperature: float = 0.7,
    user_temperature: float = 1.0,
    enable_thinking: bool = True,
    resume: bool = True,
    verbose: bool = True,
    models_config_file: Optional[str] = None
):
    """Run rollout for a single domain."""
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing: {domain}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")

    if not query_file.exists():
        print(f"  ✗ Skipping: No query file found")
        return False
    if not policy_file.exists():
        print(f"  ✗ Skipping: No policy file found")

    print(f"Domain: {domain}")
    print(f"Tools: {tools}")
    
    tools = setup_mcp_tools(domain_name=domain, tool_names=tools)
    if not tools:
        print(f"  ✗ Skipping: Failed to setup MCP tools")
        return False

    tools_info = collect_tools_info(tools)
    print(f"  ✓ Loaded {len(tools_info)} tools")

    system_prompt = """
⚠️ **Do not autonomously infer or generate parameters for tool calls. When a tool usage is identified, you must first ask the User to provide or confirm the specific values for the required arguments.**
**You may only determine parameters based on the User's overall goals if (and only if) the User explicitly states they are unsure, asks you to decide, or declines to provide specific details.**"
"""
    if policy_file.exists():
        system_prompt = system_prompt + policy_file.read_text()
        print(f"  ✓ Loaded policy: {policy_file.name}")

    agent = Agent(
        model=agent_model,
        system_prompt=system_prompt,
        tools=tools_info,
        temperature=agent_temperature,
        enable_thinking=enable_thinking
    )
    
    user = SimulatedUser(
        model=user_model,
        temperature=user_temperature
    )

    pipeline_config = PipelineConfig(
        max_turns=max_turns,
        max_steps_per_turn=max_steps_per_turn,
        use_checkpoints=False,
        resume=resume,
        enable_monitor=False,
        verbose=verbose
    )

    pipeline = Pipeline(
        agent=agent,
        user=user,
        tools=tools,
        config=pipeline_config,
        checkpoint_manager=None
    )

    queries = []
    with open(query_file, "r") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"  ✓ Loaded {len(queries)} queries")

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


def main():
    parser = argparse.ArgumentParser(description="Batch Rollout Script for Single Domain")

    # Action modes
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--all", action="store_true", help="Process all domains")
    action_group.add_argument("--domains", nargs="+", help="Process specific domains")
    action_group.add_argument("--list", action="store_true", help="List all available domains")
    action_group.add_argument("--generate-configs", action="store_true", help="Generate config files for all domains")

    # Model settings
    parser.add_argument("--agent-model", type=str, default="openai/deepseek-v3.2-fc", help="The agent model to use")
    parser.add_argument("--user-model", type=str, default="openai/gpt-5", help="The user model to use")
    parser.add_argument("--models-config", type=str, default="configs/models.yml", help="The models configuration file to use")
    parser.add_argument("--policy-dir", type=str, default="rollout/tools/datasets/single_domain/policies", help="The policy directory to use")
    parser.add_argument("--query-dir", type=str, default="rollout/tools/datasets/single_domain/queries", help="The query directory to use")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="./outputs_single_0114", help="The output directory to use")
    parser.add_argument("--config-output-dir", type=str, default="./configs/generated", help="The output directory for generated configs")

    # Processing settings
    parser.add_argument("--max-turns", type=int, default=20, help="The maximum number of turns to process")
    parser.add_argument("--max-steps-per-turn", type=int, default=10, help="The maximum number of steps per turn")
    parser.add_argument("--agent-temperature", type=float, default=0.7, help="The temperature for the agent model")
    parser.add_argument("--user-temperature", type=float, default=1.0, help="The temperature for the user model")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking for the agent model")
    parser.add_argument("--max-workers", type=int, default=8, help="The maximum number of workers to use")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous runs")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose listing output")

    # Data path
    parser.add_argument("--base-path", type=str, default="rollout/tools/datasets/single_domain", help="The base path for the datasets")

    args = parser.parse_args()

    if args.list:
        list_domains(Path(args.query_dir))

    if args.generate_configs:
        domains = list_domains(Path(args.query_dir))
        base_config = {
            "max_workers": args.max_workers,
            "resume": not args.no_resume,
            "use_checkpoints": False,
            "verbose": not args.quiet,
            "verbose_colors": True,
            "enable_monitor": False,
            "monitor_rule_detection": False,
            "monitor_max_no_tool_turns": 5,
            "models_config_file": args.models_config,
            "agent_model": args.agent_model,
            "user_model": args.user_model,
            "agent_temperature": args.agent_temperature,
            "user_temperature": args.user_temperature,
            "enable_thinking": not args.no_thinking,
            "max_turns": args.max_turns,
            "max_steps_per_turn": args.max_steps_per_turn
        }
        generate_config_file(
            domains=domains,
            output_path=args.config_output_dir,
            query_dir=Path(args.query_dir),
            policy_dir=Path(args.policy_dir),
            base_config=base_config
        )

    if args.all:
        domains = list_domains(Path(args.query_dir))
        print(f"Processing {len(domains)} domains.")
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    run_single_domain,
                    domain,
                    args.agent_model,
                    args.user_model,
                    Path(args.output_dir) / f"{domain}.jsonl",
                    Path(args.query_dir) / f"{domain}.jsonl",
                    Path(args.policy_dir) / f"{domain}.md",
                    [domain],
                    args.max_turns,
                    args.max_steps_per_turn,
                    args.agent_temperature,
                    args.user_temperature,
                    not args.no_thinking,
                    not args.no_resume,
                    not args.quiet,
                    args.models_config
                )
                for domain in domains
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

    if args.domains:
        success_count = 0
        fail_count = 0
        for domain in args.domains:
            success = run_single_domain(
                domain=domain,
                agent_model=args.agent_model,
                user_model=args.user_model,
                output_path=Path(args.output_dir) / f"{domain}.jsonl",
                query_file=Path(args.query_dir) / f"{domain}.jsonl",
                policy_file=Path(args.policy_dir) / f"{domain}.md",
                tools=[domain],
                max_turns=args.max_turns,
                max_steps_per_turn=args.max_steps_per_turn,
                agent_temperature=args.agent_temperature,
                user_temperature=args.user_temperature,
                enable_thinking=not args.no_thinking,
                resume=not args.no_resume,
                verbose=not args.quiet,
                models_config_file=args.models_config
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        print(f"\n{'=' * 60}")
        print(f"  Batch Complete: {success_count} success, {fail_count} failed")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    main()