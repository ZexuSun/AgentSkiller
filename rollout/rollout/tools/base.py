"""
Base Tool class for the Rollout framework.

All custom tools should inherit from BaseTool and implement the required methods.
"""

import json
import time
import queue
import hashlib
import threading
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the Rollout framework.
    
    A tool is defined as a function that takes arguments and returns a string response.
    Subclasses must implement:
    - `info` property: Returns tool schema in OpenAI function calling format
    - `execute()` method: Performs the actual tool operation
    
    The base class provides:
    - Thread pool for concurrent request handling
    - Request queue management
    - Automatic ID generation
    - Error handling
    
    Example:
        >>> from rollout.tools import BaseTool, register_tool
        >>> 
        >>> @register_tool
        ... class Calculator(BaseTool):
        ...     tool_type = "calculator"
        ...     
        ...     @property
        ...     def info(self):
        ...         return {
        ...             "type": "function",
        ...             "function": {
        ...                 "name": "calculate",
        ...                 "description": "Perform arithmetic calculation",
        ...                 "parameters": {
        ...                     "type": "object",
        ...                     "properties": {
        ...                         "expression": {"type": "string"}
        ...                     },
        ...                     "required": ["expression"]
        ...                 }
        ...             }
        ...         }
        ...     
        ...     def execute(self, tool_call_str: str, **kwargs):
        ...         args = json.loads(tool_call_str)
        ...         return str(eval(args["expression"]))
    """
    
    # Override in subclasses to set the tool identifier
    tool_type: str = "base_tool"
    
    def __init__(self, num_workers: int = 1) -> None:
        """
        Initialize the tool with a worker thread pool.
        
        Args:
            num_workers: Number of concurrent workers for handling requests.
                        Adjust based on the backend service's capacity.
        """
        self.num_workers = num_workers
        self.request_queue: queue.Queue = queue.Queue(maxsize=-1)
        self.results: Dict[str, Any] = {}
        self.results_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._workers: List[threading.Thread] = []
        self._start_workers()
    
    def _generate_id(self, request: str) -> str:
        """Generate a unique ID for a request."""
        return hashlib.md5(request.encode("utf-8")).hexdigest()
    
    def _start_workers(self):
        """Start the worker threads."""
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self):
        """Main loop for worker threads."""
        while not self._stop_event.is_set():
            try:
                task_data = self.request_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            result_id = task_data["result_id"]
            try:
                tool_response = self.execute(**task_data)
                with self.results_lock:
                    self.results[result_id] = tool_response
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                with self.results_lock:
                    self.results[result_id] = f"Error: {e}"
            finally:
                self.request_queue.task_done()
    
    def _submit_request(self, tool_call_str: str, **kwargs) -> str:
        """Submit a request to the worker queue."""
        result_id = self._generate_id(tool_call_str)
        task_data = {
            "tool_call_str": tool_call_str,
            "result_id": result_id,
            **kwargs
        }
        self.request_queue.put(task_data)
        return result_id
    
    @property
    @abstractmethod
    def info(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Return the tool's schema in OpenAI function calling format.
        
        Can return a single dict for one function, or a list of dicts
        for tools that expose multiple functions.
        
        Returns:
            Tool schema dict or list of dicts with structure:
            {
                "type": "function",
                "function": {
                    "name": "function_name",
                    "description": "What the function does",
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                    }
                }
            }
        """
        raise NotImplementedError("Subclasses must implement the info property")
    
    @abstractmethod
    def execute(self, tool_call_str: str, **kwargs) -> str:
        """
        Execute the tool with the given arguments.
        
        Args:
            tool_call_str: JSON string containing the tool call arguments
            **kwargs: Additional context (e.g., sample_id)
            
        Returns:
            String result of the tool execution
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def get_observation(
        self,
        tool_call_str: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Submit a tool call and wait for the result.
        
        Args:
            tool_call_str: JSON string containing the tool call
            timeout: Maximum time to wait for result (None = wait forever)
            **kwargs: Additional context passed to execute()
            
        Returns:
            The tool's response string, or error message on timeout
        """
        result_id = self._submit_request(tool_call_str, **kwargs)
        start_time = time.time()
        
        while timeout is None or time.time() - start_time < timeout:
            with self.results_lock:
                if result_id in self.results:
                    result = self.results.pop(result_id)
                    return result
            time.sleep(0.1)
        
        return "Error: Request timeout"
    
    def execute_sync(self, tool_call_str: str, **kwargs) -> str:
        """
        Execute tool synchronously (bypasses worker queue).
        
        Useful for simple tools or debugging.
        """
        return self.execute(tool_call_str=tool_call_str, **kwargs)
    
    def parse_arguments(self, tool_call_str: str) -> Dict[str, Any]:
        """
        Parse tool call arguments from JSON string.
        
        Helper method for subclasses.
        """
        try:
            data = json.loads(tool_call_str)
            # Handle both raw args and function call format
            if "arguments" in data:
                if isinstance(data["arguments"], str):
                    return json.loads(data["arguments"])
                return data["arguments"]
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool call: {e}")
    
    def format_response(self, result: Any) -> str:
        """
        Format the result for return.
        
        Helper method for subclasses.
        """
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def shutdown(self):
        """Stop all worker threads."""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=5.0)
    
    def __del__(self):
        """Clean up workers on deletion."""
        self.shutdown()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tool_type={self.tool_type!r})"

