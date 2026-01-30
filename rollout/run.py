#!/usr/bin/env python3
"""
Rollout - Multi-Turn Tool Call Annotation Framework

Usage:
    python run.py --config configs/example.yml
    python run.py --config configs/example.yml --resume
    python run.py --config configs/example.yml --dataset Mix
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rollout import Agent, SimulatedUser, Pipeline
from rollout.config import RolloutConfig
from rollout.tools import discover_tools, instantiate_tools, collect_tools_info, setup_mcp_tools
from rollout.core.pipeline import PipelineConfig, PipelineMode
from rollout.core.checkpoint import CheckpointManager
from rollout.utils.generate_id import generate_id


def setup_logging(config: RolloutConfig):
    """Configure logging based on config."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def load_samples(path: str) -> list:
    """Load samples from JSONL or JSON file."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Rollout - Multi-Turn Tool Call Annotation Framework"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Process only this dataset (by name)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from checkpoints"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and count samples without processing"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print model responses in real-time"
    )
    parser.add_argument(
        "--no-colors",
        action="store_true",
        help="Disable colored output in verbose mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = RolloutConfig.from_yaml(args.config)
    if args.resume:
        config.resume = True
    if args.verbose:
        config.verbose = True
    if args.no_colors:
        config.verbose_colors = False
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger("rollout")
    
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Resume mode: {config.resume}")
    logger.info(f"Checkpoints: {config.use_checkpoints}")
    logger.info(f"Verbose: {config.verbose}")
    
    # Discover all tools
    logger.info("Discovering tools...")
    all_tools = discover_tools()
    logger.info(f"Found {len(all_tools)} tools: {list(all_tools.keys())}")
    
    # Filter datasets if specified
    datasets_to_process = config.datasets
    if args.dataset:
        if args.dataset not in config.datasets:
            logger.error(f"Dataset '{args.dataset}' not found in config")
            sys.exit(1)
        datasets_to_process = {args.dataset: config.datasets[args.dataset]}
    
    # Process each dataset
    for name, ds_config in datasets_to_process.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {name}")
        logger.info(f"{'='*60}")
        
        # Load samples
        samples = load_samples(ds_config.path)
        logger.info(f"Loaded {len(samples)} samples from {ds_config.path}")
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would process {len(samples)} samples")
            continue
        
        # Initialize tools for this dataset
        if ds_config.mcp_domain:
            # Use MCP Server tools
            logger.info(f"Loading MCP tools with domain: {ds_config.mcp_domain}")
            # Convert tool names: snake_case -> CamelCase for MCP server names
            # Convert snake_case tool names to CamelCase MCP server names
            mcp_tool_names = [
                ''.join(word.capitalize() for word in name.split('_'))
                for name in ds_config.tools
            ]
            # Each MCP Server file is separate, but they all need the full combined domain_name
            # (e.g., "StudentAcademicPortal_StudentFinancialServices_StudentHealthServices")
            # to load the correct database from database/outputs/entities/{domain_name}/
            if not ds_config.mcp_domain:
                raise ValueError(f"mcp_domain is required for MCP tools in dataset {ds_config.name}")
            tools = setup_mcp_tools(domain_name=ds_config.mcp_domain, tool_names=mcp_tool_names)
        else:
            # Use regular tools
            tools = instantiate_tools(ds_config.tools)
        
        tools_info = collect_tools_info(tools)
        logger.info(f"Initialized {len(tools)} tools with {len(tools_info)} functions")
        
        # Initialize agent
        agent = Agent(
            model=ds_config.agent.model,
            system_prompt=ds_config.agent.system_prompt,
            temperature=ds_config.agent.temperature,
            max_tokens=ds_config.agent.max_tokens,
            api_key=ds_config.agent.api_key,
            api_base=ds_config.agent.api_base,
            enable_thinking=ds_config.agent.enable_thinking
        )
        logger.info(f"Initialized agent: {agent}")
        if ds_config.agent.enable_thinking:
            logger.info("DeepSeek V3.2 thinking mode enabled")
        
        # Initialize user (for multi-turn)
        user = None
        if ds_config.max_turns > 1 and ds_config.user:
            user = SimulatedUser(
                model=ds_config.user.model,
                temperature=ds_config.user.temperature,
                max_tokens=ds_config.user.max_tokens,
                api_key=ds_config.user.api_key,
                api_base=ds_config.user.api_base,
                custom_guidelines=ds_config.user.custom_guidelines
            )
            logger.info(f"Initialized user simulator: {user}")
        
        # Initialize checkpoint manager
        checkpoint_manager = None
        if config.use_checkpoints:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=config.checkpoint_dir
            )
            
            # If not resuming, clear existing checkpoints for samples to reprocess
            if not config.resume:
                logger.info("Clearing existing checkpoints (resume=false)...")
                cleared = 0
                for sample in samples:
                    sample_id = sample.get("id") or generate_id(sample, include_tools=True)
                    if checkpoint_manager.has_checkpoint(sample_id):
                        checkpoint_manager.delete(sample_id)
                        cleared += 1
                if cleared > 0:
                    logger.info(f"Cleared {cleared} checkpoints")
        
        # Create pipeline
        pipeline_config = PipelineConfig(
            max_turns=ds_config.max_turns,
            max_steps_per_turn=ds_config.max_steps_per_turn,
            mode=PipelineMode(ds_config.mode),
            use_checkpoints=config.use_checkpoints,
            checkpoint_dir=config.checkpoint_dir,
            resume=config.resume,
            verbose=config.verbose,
            verbose_colors=config.verbose_colors,
            verbose_max_length=config.verbose_max_length,
            # Conversation monitor for early termination
            enable_monitor=config.enable_monitor,
            monitor_rule_detection=config.monitor_rule_detection,
            monitor_max_no_tool_turns=config.monitor_max_no_tool_turns,
            monitor_use_llm_judge=config.monitor_use_llm_judge,
            monitor_judge_model=config.monitor_judge_model
        )
        
        pipeline = Pipeline(
            agent=agent,
            tools=tools,
            user=user,
            config=pipeline_config,
            checkpoint_manager=checkpoint_manager
        )
        
        # Prepare samples (add IDs, filter completed)
        for sample in samples:
            if "id" not in sample:
                sample["id"] = generate_id(sample, include_tools=True)
            if ds_config.mode == "positive":
                sample["tools"] = tools_info
        
        # Filter completed samples in resume mode
        if config.resume and os.path.exists(ds_config.output_path):
            processed_ids = set()
            with open(ds_config.output_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed_ids.add(data.get("id"))
            
            samples = [s for s in samples if s["id"] not in processed_ids]
            logger.info(f"Resume mode: {len(processed_ids)} already processed, {len(samples)} remaining")
        
        if not samples:
            logger.info("No samples to process")
            continue
        
        # Process samples with immediate write
        write_mode = "a" if config.resume else "w"
        os.makedirs(os.path.dirname(ds_config.output_path) or ".", exist_ok=True)
        
        # Use a lock for thread-safe file writing
        import threading
        write_lock = threading.Lock()
        success_count = 0
        
        # Open file for immediate writing
        output_file = open(ds_config.output_path, write_mode)
        
        def process_and_write(sample):
            """Process a single sample and write result immediately."""
            nonlocal success_count
            try:
                result = pipeline.process(sample, tools_info)
                
                # Write immediately with lock
                with write_lock:
                    if result.success:
                        output_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                        output_file.flush()  # Ensure data is written to disk
                        success_count += 1
                    else:
                        logger.warning(f"Failed: {result.trajectory_id} - {result.error}")
                
                return result
            except Exception as e:
                logger.error(f"Error processing {sample.get('id', 'unknown')}: {e}")
                return None
        
        try:
            pbar = tqdm(total=len(samples), desc=f"Processing {name}", ncols=100)
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {executor.submit(process_and_write, sample): sample for sample in samples}
                
                for future in as_completed(futures):
                    pbar.update(1)
            
            pbar.close()
            logger.info(f"Completed: {success_count}/{len(samples)} successful")
        finally:
            output_file.close()
    
    logger.info("\nAll datasets processed!")


if __name__ == "__main__":
    main()

