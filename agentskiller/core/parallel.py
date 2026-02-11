"""
Parallel processing utilities for agentskiller.

Provides reusable parallel processing with:
- ThreadPoolExecutor-based parallelism
- Rich progress bar integration
- Error handling and result collection
"""

import logging
from typing import List, Callable, Optional, Any, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import (
    Progress,
    TaskID,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()

T = TypeVar("T")
R = TypeVar("R")


def parallel_process(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: Optional[int] = None,
    description: str = "Processing",
    show_progress: bool = True,
    on_error: Optional[Callable[[T, Exception], None]] = None,
) -> List[Optional[R]]:
    """
    Process items in parallel with progress tracking.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers (default from config)
        description: Progress bar description
        show_progress: Whether to show progress bar
        on_error: Optional callback for errors (item, exception)
    
    Returns:
        List of results in the same order as inputs.
        Failed items will have None as result.
    
    Usage:
        def process_entity(entity: EntityInfo) -> dict:
            # Process single entity
            return {"id": entity.id, "result": ...}
        
        results = parallel_process(
            items=entities,
            process_func=process_entity,
            description="Processing entities",
        )
    """
    settings = get_settings()
    workers = max_workers or settings.workflow.max_workers
    
    if not items:
        return []
    
    results: List[Optional[R]] = [None] * len(items)
    
    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    
    with Progress(*progress_columns, console=console, disable=not show_progress) as progress:
        task = progress.add_task(f"[cyan]{description}", total=len(items))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_func, item): idx
                for idx, item in enumerate(items)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    if on_error:
                        on_error(items[idx], e)
                    results[idx] = None
                
                progress.advance(task)
    
    return results


def parallel_process_with_retry(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: Optional[int] = None,
    max_retries: int = 3,
    description: str = "Processing",
    show_progress: bool = True,
) -> tuple[List[R], List[tuple[T, Exception]]]:
    """
    Process items in parallel with automatic retry for failures.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers
        max_retries: Maximum retries for failed items
        description: Progress bar description
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (successful_results, failed_items_with_errors)
    
    Usage:
        successes, failures = parallel_process_with_retry(
            items=entities,
            process_func=process_entity,
            max_retries=3,
        )
    """
    settings = get_settings()
    workers = max_workers or settings.workflow.max_workers
    
    if not items:
        return [], []
    
    # Track results and failures
    results: dict[int, R] = {}
    pending_indices = list(range(len(items)))
    failures: List[tuple[T, Exception]] = []
    
    for attempt in range(max_retries):
        if not pending_indices:
            break
        
        current_items = [(idx, items[idx]) for idx in pending_indices]
        new_pending = []
        
        desc = f"{description} (attempt {attempt + 1})" if attempt > 0 else description
        
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]
        
        with Progress(*progress_columns, console=console, disable=not show_progress) as progress:
            task = progress.add_task(f"[cyan]{desc}", total=len(current_items))
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(process_func, item): idx
                    for idx, item in current_items
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        if attempt < max_retries - 1:
                            new_pending.append(idx)
                            logger.warning(
                                f"Item {idx} failed, will retry: {e}"
                            )
                        else:
                            failures.append((items[idx], e))
                            logger.error(
                                f"Item {idx} failed after {max_retries} attempts: {e}"
                            )
                    
                    progress.advance(task)
        
        pending_indices = new_pending
    
    # Build ordered result list
    ordered_results = [results[i] for i in sorted(results.keys())]
    
    return ordered_results, failures


def chunked_parallel_process(
    items: List[T],
    process_func: Callable[[List[T]], List[R]],
    chunk_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    description: str = "Processing chunks",
    show_progress: bool = True,
) -> List[R]:
    """
    Process items in parallel chunks.
    
    Useful when:
    - Items should be processed in batches (e.g., for LLM calls)
    - Individual items are too small for parallelization overhead
    
    Args:
        items: List of items to process
        process_func: Function that processes a chunk and returns results
        chunk_size: Size of each chunk (default from config)
        max_workers: Maximum number of parallel workers
        description: Progress bar description
        show_progress: Whether to show progress bar
    
    Returns:
        Flattened list of results from all chunks
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.workflow.chunk_size
    
    if not items:
        return []
    
    # Create chunks
    chunks = [
        items[i:i + chunk_size]
        for i in range(0, len(items), chunk_size)
    ]
    
    # Process chunks in parallel
    chunk_results = parallel_process(
        items=chunks,
        process_func=process_func,
        max_workers=max_workers,
        description=description,
        show_progress=show_progress,
    )
    
    # Flatten results
    all_results = []
    for result in chunk_results:
        if result is not None:
            all_results.extend(result)
    
    return all_results


class ParallelProcessor:
    """
    Reusable parallel processor with configuration.
    
    Provides a stateful wrapper around parallel_process functions
    with consistent configuration and error tracking.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        max_retries: int = 3,
    ):
        settings = get_settings()
        self.max_workers = max_workers or settings.workflow.max_workers
        self.show_progress = show_progress
        self.max_retries = max_retries
        self.errors: List[tuple[Any, Exception]] = []
    
    def process(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        description: str = "Processing",
    ) -> List[Optional[R]]:
        """Process items with error tracking."""
        self.errors = []
        
        def on_error(item: T, e: Exception):
            self.errors.append((item, e))
        
        return parallel_process(
            items=items,
            process_func=process_func,
            max_workers=self.max_workers,
            description=description,
            show_progress=self.show_progress,
            on_error=on_error,
        )
    
    def process_with_retry(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        description: str = "Processing",
    ) -> tuple[List[R], List[tuple[T, Exception]]]:
        """Process items with retry and error tracking."""
        results, failures = parallel_process_with_retry(
            items=items,
            process_func=process_func,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            description=description,
            show_progress=self.show_progress,
        )
        self.errors = failures
        return results, failures
    
    def get_error_summary(self) -> dict:
        """Get summary of processing errors."""
        return {
            "total_errors": len(self.errors),
            "error_types": list(set(type(e).__name__ for _, e in self.errors)),
        }
