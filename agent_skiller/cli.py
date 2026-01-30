"""
Command-line interface for agent_skiller.

Usage:
    python -m agent_skiller run [OPTIONS]
    python -m agent_skiller status
    python -m agent_skiller list-steps
"""

import logging
import sys
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from .config.settings import get_settings, init_settings
from .config.models_registry import get_models_registry
from .graph.workflow import WORKFLOW_STEPS, get_step_info
from .graph.runner import run_workflow, run_single_step, list_steps as get_step_list

app = typer.Typer(
    name="agent_skiller",
    help="Data Synthesis Workflow",
    add_completion=False,
)
console = Console()


def setup_logging(verbose: bool = False):
    """Configure rich logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    
    # Suppress verbose LiteLLM logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


@app.command()
def run(
    step: Optional[str] = typer.Option(
        None, "--step", "-s",
        help="Run specific step (default: all)"
    ),
    from_step: Optional[str] = typer.Option(
        None, "--from", "-f",
        help="Start from this step"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to config.yaml"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Run without making changes"
    ),
):
    """
    Run the workflow.
    
    Examples:
        agent_skiller run                    # Run all steps
        agent_skiller run -s s01             # Run step 1 only
        agent_skiller run -f s04             # Start from step 4
        agent_skiller run --dry-run          # Dry run mode
    """
    setup_logging(verbose)
    
    # Initialize settings
    if config:
        init_settings(config)
    settings = get_settings()
    
    if dry_run:
        settings.workflow.dry_run = True
    
    # Initialize models registry
    get_models_registry()
    
    try:
        if step:
            # Run single step
            if step not in WORKFLOW_STEPS:
                # Try matching by prefix
                matches = [s for s in WORKFLOW_STEPS if s.startswith(step)]
                if len(matches) == 1:
                    step = matches[0]
                else:
                    console.print(f"[red]Unknown step: {step}[/red]")
                    console.print(f"Available steps: {', '.join(WORKFLOW_STEPS)}")
                    raise typer.Exit(1)
            
            console.print(f"[bold cyan]Running step: {step}[/bold cyan]")
            run_single_step(step)
        else:
            # Run full workflow
            console.print("[bold cyan]Running full workflow[/bold cyan]")
            run_workflow(start_step=from_step)
        
        console.print("[bold green]✓ Workflow completed successfully[/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Workflow interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[bold red]✗ Workflow failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def status(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to config.yaml"
    ),
):
    """Show workflow status."""
    if config:
        init_settings(config)
    settings = get_settings()
    
    console.print("[bold]Workflow Status[/bold]\n")
    
    # Show configuration
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("Task Mode", settings.workflow.task_mode)
    table.add_row("Max Workers", str(settings.workflow.max_workers))
    table.add_row("Auto Retry", str(settings.workflow.auto_retry))
    table.add_row("Outputs Dir", str(settings.paths.outputs_dir))
    
    console.print(table)
    console.print()
    
    # Show step status
    steps_table = Table(title="Step Status")
    steps_table.add_column("Step", style="cyan")
    steps_table.add_column("Phase")
    steps_table.add_column("Status")
    steps_table.add_column("Output")
    
    for step_name in WORKFLOW_STEPS:
        info = get_step_info(step_name)
        
        # Check if output exists
        output_exists = "✓" if _check_step_output(step_name, settings) else ""
        
        # Determine status
        if info["is_cross_domain_only"] and settings.workflow.task_mode == "single":
            status = "[dim]skipped[/dim]"
        elif output_exists:
            status = "[green]completed[/green]"
        else:
            status = "[yellow]pending[/yellow]"
        
        steps_table.add_row(
            step_name,
            info["phase"],
            status,
            output_exists,
        )
    
    console.print(steps_table)


@app.command("list-steps")
def list_steps_cmd(
    include_cross_domain: bool = typer.Option(
        True, "--cross-domain/--single",
        help="Include cross-domain steps"
    ),
):
    """List all workflow steps."""
    steps = get_step_list(include_cross_domain)
    
    table = Table(title="Workflow Steps")
    table.add_column("#", style="dim")
    table.add_column("Step Name", style="cyan")
    table.add_column("Phase")
    table.add_column("Description")
    table.add_column("CD", justify="center")
    
    for i, step in enumerate(steps, 1):
        cd_marker = "●" if step["cross_domain_only"] else ""
        table.add_row(
            str(i),
            step["name"],
            step["phase"],
            step["description"],
            cd_marker,
        )
    
    console.print(table)
    console.print("\n[dim]CD = Cross-Domain only[/dim]")


@app.command("init-config")
def init_config(
    output: Path = typer.Option(
        Path("config.yaml"),
        "--output", "-o",
        help="Output path for config file"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing file"
    ),
):
    """Create a sample configuration file."""
    if output.exists() and not force:
        console.print(f"[yellow]Config already exists: {output}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    settings = get_settings()
    settings.save_yaml(output)
    console.print(f"[green]Created config file: {output}[/green]")


def _check_step_output(step_name: str, settings) -> bool:
    """Check if step output exists."""
    outputs_dir = settings.paths.outputs_dir
    
    output_files = {
        "s01_domain_expansion": outputs_dir / "domain_topics.json",
        "s02_entity_extraction": outputs_dir / "entities.json",
        "s03_entity_graph": outputs_dir / "entity_graph.json",
        "s04_blueprint_generation": outputs_dir / "blueprints.json",
        "s05_tool_list_formulation": outputs_dir / "tool_lists",
        "s06_database_generation": outputs_dir / "database",
        "s07_policy_generation": outputs_dir / "policies",
        "s08_tool_graph_generation": outputs_dir / "tool_graphs",
        "s09_mcp_server_implementation": outputs_dir / "mcp_servers",
        "s10_domain_combos_selection": outputs_dir / "cross_domain_templates" / "_combinations.json",
        "s11_trajectory_fusion": outputs_dir / "cross_domain_templates",
        "s12_database_fusion": outputs_dir / "database" / "scripts" / "cross_domain",
        "s13_policy_merge": outputs_dir / "policies",
        "s14_task_template_generation": outputs_dir / "task_templates",
        "s15_instance_combos_selection": outputs_dir / "combinations",
        "s16_task_filtering": outputs_dir / "validated_tasks",
        "s17_task_instantiation": outputs_dir / "queries",
    }
    
    output_path = output_files.get(step_name)
    return output_path.exists() if output_path else False


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
