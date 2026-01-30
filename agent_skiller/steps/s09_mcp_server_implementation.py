"""
Step 9: MCP Server Implementation

Implement and test MCP servers from blueprints.

Input: fixed_blueprints.json, policies/*.md
Output: mcp_servers/*.py, unit_tests/*.py

Process:
1. Generate MCP server Python code
2. Validate syntax and fix if needed
3. Generate unit tests
4. Validate test syntax and fix if needed
5. Run tests and fix errors in a loop
6. Track progress for step-wise retry
"""

import ast
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from ..models.state import WorkflowState
from ..config.settings import get_settings
from ..prompts import (
    MCP_SERVER_IMPLEMENT_PROMPT,
    UNIT_TEST_PROMPT,
    CODE_FIX_PROMPT,
    ERROR_TRACE,
)
from .base import (
    step_handler, save_json, load_json, get_client,
    parallel_process, ensure_dir, WorkflowBlockEditor
)

logger = logging.getLogger(__name__)


def ensure_package_init(directory: Path) -> None:
    """Ensure __init__.py exists in directory to make it a Python package."""
    init_file = directory / "__init__.py"
    if not init_file.exists():
        init_file.write_text(f"# Auto-generated __init__.py for {directory.name} package\n")
        logger.debug(f"Created {init_file}")


def fix_server_class_names(servers_dir: Path) -> int:
    """
    检查所有 MCP Server 文件，确保类名为 {domain_name}Server 以匹配 mcp_tool_factory.py 的期望。
    
    例如：
    - CourseCompass.py 中的类名应为 CourseCompassServer
    - AccessControlServer.py 中的类名应为 AccessControlServerServer
    
    Args:
        servers_dir: MCP servers 目录路径
        
    Returns:
        修复的文件数量
    """
    fixed_count = 0
    
    for server_file in servers_dir.glob("*.py"):
        if server_file.name.startswith("__"):
            continue
        
        domain_name = server_file.stem  # e.g., "CourseCompass" or "AccessControlServer"
        expected_class_name = f"{domain_name}Server"  # e.g., "CourseCompassServer" or "AccessControlServerServer"
        
        content = server_file.read_text()
        
        # 检查是否已经是正确的类名
        if f"class {expected_class_name}" in content:
            continue
        
        # 检查是否存在需要修复的类名（与文件名相同但没有 Server 后缀）
        current_class_name = domain_name  # e.g., "CourseCompass" or "AccessControlServer"
        
        if f"class {current_class_name}" in content:
            # 替换类名定义
            new_content = content.replace(
                f"class {current_class_name}:",
                f"class {expected_class_name}:"
            )
            # 同时替换类名的其他引用（如果有的话）
            new_content = new_content.replace(
                f"class {current_class_name}(",
                f"class {expected_class_name}("
            )
            
            server_file.write_text(new_content)
            logger.info(f"Fixed class name: {current_class_name} -> {expected_class_name} in {server_file.name}")
            fixed_count += 1
    
    return fixed_count

# Template directory
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "code_templates"


def load_template(template_name: str) -> str:
    """Load template file content."""
    template_path = TEMPLATES_DIR / template_name
    if template_path.exists():
        return template_path.read_text()
    logger.warning(f"Template not found: {template_path}")
    return ""


def extract_python_code(content: str) -> str:
    """Extract Python code from LLM response, handling markdown code blocks."""
    if "```python" in content:
        code = content.split("```python")[1].split("```")[0]
    elif "```" in content:
        code = content.split("```")[1].split("```")[0]
    else:
        code = content
    return code.strip()


def validate_python(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def run_tests(test_file: Path) -> Tuple[bool, str]:
    """Run pytest on a test file until first failure."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "--maxfail=1", "-v"],
        capture_output=True,
        text=True,
    )
    
    all_passed = result.returncode == 0
    output = result.stdout if result.stdout else result.stderr
    
    return all_passed, output


def parse_pytest_summary(summary_line: str) -> Optional[Dict]:
    """Parse pytest summary line to extract statistics."""
    # Remove ANSI escape codes
    clean_line = re.sub(r'\x1B\[[0-9;]*[a-zA-Z]', '', summary_line)
    
    patterns = [
        r'(\d+)\s*failed,\s*(\d+)\s*passed',
        r'(\d+)\s*passed,\s*(\d+)\s*failed',
        r'(\d+)\s*passed',
        r'(\d+)\s*failed',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_line)
        if match:
            groups = match.groups()
            if 'failed' in pattern and 'passed' in pattern:
                if pattern.startswith(r'(\d+)\s*failed'):
                    return {'failed': int(groups[0]), 'passed': int(groups[1])}
                else:
                    return {'passed': int(groups[0]), 'failed': int(groups[1])}
            elif 'passed' in pattern:
                return {'passed': int(groups[0]), 'failed': 0}
            elif 'failed' in pattern:
                return {'failed': int(groups[0]), 'passed': 0}
    
    return None


def get_test_statistics(test_file: Path) -> Optional[Any]:
    """Run all tests and return pass/fail statistics."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        return "All passed"
    
    if not result.stdout:
        logger.error(f"Pytest failed with no stdout. Stderr: {result.stderr}")
        return None
    
    lines = result.stdout.splitlines()
    if not lines:
        return None
    
    # Find the summary line (usually the last non-empty line)
    for line in reversed(lines):
        if line.strip():
            statistics = parse_pytest_summary(line)
            if statistics:
                return statistics
    
    return None


def extract_test_failures(pytest_output: str) -> str:
    """Extract failure information from pytest output."""
    failures_pattern = r"={20,}\s*FAILURES\s*={20,}(.*?)(?:={20,}|$)"
    match = re.search(failures_pattern, pytest_output, re.DOTALL)
    
    if match:
        failures_section = match.group(1).strip()
        # Also extract short test summary info
        summary_pattern = r"={20,}\s*short test summary info\s*={20,}(.*?)(?:={20,}|$)"
        summary_match = re.search(summary_pattern, pytest_output, re.DOTALL)
        if summary_match:
            return f"FAILURES\n{failures_section}\n\nSHORT TEST SUMMARY INFO\n{summary_match.group(1).strip()}"
        return failures_section
    
    return pytest_output


def extract_test_data_generator(test_code: str) -> str:
    """Extract TestDataBuilder class from test code."""
    if "class TestDataBuilder:" in test_code:
        parts = test_code.split("class TestDataBuilder:")
        if len(parts) > 1:
            # Find the end of the class
            rest = parts[1]
            if "class Test" in rest:
                return "class TestDataBuilder:" + rest.split("class Test")[0]
            return "class TestDataBuilder:" + rest
    return ""


def analyze_test_failure(
    error: str,
    policy: str,
    server_code: str,
    test_code: str,
    client: Any,
) -> Optional[Dict]:
    """Analyze a test failure to determine the cause."""
    test_data_gen = extract_test_data_generator(test_code)
    
    query = ERROR_TRACE.format(
        policy=policy,
        mcp_server_code=server_code,
        unit_test_failure=error,
        test_data_generator=test_data_gen,
    )
    
    try:
        response = client.chat(query=query, model_type="coding")
        
        # Try to parse JSON from response
        content = response.content
        
        # Look for JSON block in markdown
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON object in text
        json_match = re.search(r'\{[^{}]*"likely_bug_location"[^{}]*\}', content)
        if json_match:
            return json.loads(json_match.group(0))
        
        # Fallback: infer from text
        content_lower = content.lower()
        if "unit test" in content_lower and any(w in content_lower for w in ["wrong", "incorrect", "bug"]):
            return {
                "failed_test_name": "unknown",
                "likely_bug_location": "unit test",
                "explanation": content
            }
        elif "server" in content_lower or "function" in content_lower:
            return {
                "failed_test_name": "unknown",
                "likely_bug_location": "function",
                "explanation": content
            }
        
        # Default to unit test
        return {
            "failed_test_name": "unknown",
            "likely_bug_location": "unit test",
            "explanation": "Could not parse LLM response"
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze test failure: {e}")
        return None


def fix_code_with_editor(
    file_path: Path,
    error: str,
    client: Any,
    editor: WorkflowBlockEditor,
    max_retries: int = 5,
) -> bool:
    """Use LLM and BlockEditor to fix code errors."""
    content = file_path.read_text()
    
    for attempt in range(max_retries):
        query = CODE_FIX_PROMPT.format(
            file_path=str(file_path),
            output_or_error=error,
            full_code_content=content,
        )
        
        response = client.chat(query=query, model_type="coding")
        result = editor.apply_edits_from_response(file_path, response.content)
        
        if result.success_count > 0 and result.fail_count == 0:
            logger.info(f"Fixed code in {file_path} after {attempt + 1} attempt(s)")
            return True
        
        if result.error_message:
            error = result.error_message
            content = file_path.read_text()
    
    logger.warning(f"Failed to fix code in {file_path} after {max_retries} attempts")
    return False


@step_handler("s09_mcp_server_implementation", auto_retry=True)
def mcp_server_implementation_step(state: WorkflowState) -> WorkflowState:
    """
    Implement MCP servers from blueprints.
    
    Process:
    1. Load blueprints and policies
    2. Generate MCP server Python code with syntax validation
    3. Generate unit tests with syntax validation
    4. Run tests and fix errors in a loop
    5. Track progress for step-wise retry
    
    Output:
    - mcp_servers/*.py: Server implementations
    - unit_tests/*.py: Unit tests
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    step_config = settings.get_step_config("s09_mcp_server_implementation")
    
    # Configuration
    max_fix_attempts = step_config.get("max_fix_attempts", 5)
    max_syntax_retries = step_config.get("max_syntax_retries", 5)
    simulation_time = step_config.get(
        "simulation_time",
        settings.workflow.simulation_time
    )
    
    # Load templates
    mcp_server_template = load_template("mcp_server_template.py")
    pytest_template = load_template("unit_test_template.py")
    
    # Load blueprints
    blueprints_path = state.blueprints_path
    blueprints_data = load_json(Path(blueprints_path))
    if isinstance(blueprints_data, list):
        blueprints = blueprints_data
    else:
        blueprints = blueprints_data.get("blueprints", [])
    
    # Setup directories
    servers_dir = ensure_dir(outputs_dir / "mcp_servers")
    tests_dir = ensure_dir(outputs_dir / "unit_tests")
    policies_dir = Path(state.policies_dir) if state.policies_dir else None
    
    # Ensure __init__.py files exist for Python package imports
    ensure_package_init(outputs_dir)
    ensure_package_init(servers_dir)
    ensure_package_init(tests_dir)
    
    logger.info(f"Implementing {len(blueprints)} MCP servers")
    logger.info(f"Max fix attempts per test: {max_fix_attempts}")
    
    client = get_client()
    editor = WorkflowBlockEditor()
    
    # Load global record for resuming
    global_record_path = outputs_dir / "mcp_global_record.json"
    global_record: Dict[str, Any] = {}
    if global_record_path.exists():
        try:
            global_record = load_json(global_record_path)
            passed_domains = [k for k, v in global_record.items() if v == "All passed"]
            logger.info(f"Loaded global record: {len(passed_domains)} domains already passed")
        except Exception as e:
            logger.warning(f"Failed to load global record: {e}")
    
    # Filter out already passed domains
    passed_domains = {k for k, v in global_record.items() if v == "All passed"}
    to_process = [
        bp for bp in blueprints
        if bp.get("MCP_server_name") not in passed_domains
    ]
    
    if not to_process:
        logger.info("All servers already implemented and passed")
        state.mcp_servers_dir = str(servers_dir)
        state.unit_tests_dir = str(tests_dir)
        return state
    
    logger.info(f"Processing {len(to_process)} servers (skipping {len(passed_domains)} already passed)")
    
    def implement_server(blueprint: dict) -> dict:
        """Implement a single MCP server with full test-fix loop."""
        server_name = blueprint.get("MCP_server_name", "Unknown")
        server_path = servers_dir / f"{server_name}.py"
        test_path = tests_dir / f"test_{server_name}.py"
        
        # Load policy
        policy_content = ""
        if policies_dir:
            policy_path = policies_dir / f"{server_name}.md"
            if policy_path.exists():
                policy_content = policy_path.read_text()
        
        try:
            # ========================================
            # Step 1: Generate MCP Server Code
            # ========================================
            logger.info(f"[{server_name}] Generating MCP server code...")
            
            code_prompt = MCP_SERVER_IMPLEMENT_PROMPT.format(
                domain_policy=policy_content,
                blueprint=json.dumps(blueprint, indent=2),
                mcp_server_code_template=mcp_server_template,
                simulation_time=simulation_time
            )
            
            code_response = client.chat(query=code_prompt, model_type="coding")
            server_code = extract_python_code(code_response.content)
            server_path.write_text(server_code)
            
            # Validate and fix server syntax
            logger.info(f"[{server_name}] Validating server syntax...")
            syntax_valid = False
            for attempt in range(max_syntax_retries):
                valid, error = validate_python(server_code)
                if valid:
                    logger.info(f"[{server_name}] Server syntax valid")
                    syntax_valid = True
                    break
                logger.warning(f"[{server_name}] Syntax error (attempt {attempt + 1}): {error}")
                if fix_code_with_editor(server_path, error, client, editor):
                    server_code = server_path.read_text()
                else:
                    break
            
            if not syntax_valid:
                logger.error(f"[{server_name}] Failed to fix server syntax")
                return {
                    "success": False,
                    "server_name": server_name,
                    "error": "Server syntax error",
                    "status": "FAILED"
                }
            
            # ========================================
            # Step 2: Generate Unit Tests
            # ========================================
            logger.info(f"[{server_name}] Generating unit tests...")
            
            test_prompt = UNIT_TEST_PROMPT.format(
                server_code=server_code,
                policy=policy_content,
                pytest_template=pytest_template,
                outputs_dir="outputs",
                domain_name=server_name
            )
            
            test_response = client.chat(query=test_prompt, model_type="coding")
            test_code = extract_python_code(test_response.content)
            test_path.write_text(test_code)
            
            # Validate and fix test syntax
            logger.info(f"[{server_name}] Validating test syntax...")
            test_syntax_valid = False
            for attempt in range(max_syntax_retries):
                valid, error = validate_python(test_code)
                if valid:
                    logger.info(f"[{server_name}] Test syntax valid")
                    test_syntax_valid = True
                    break
                logger.warning(f"[{server_name}] Test syntax error (attempt {attempt + 1}): {error}")
                if fix_code_with_editor(test_path, error, client, editor):
                    test_code = test_path.read_text()
                else:
                    break
            
            if not test_syntax_valid:
                logger.error(f"[{server_name}] Failed to fix test syntax")
                return {
                    "success": False,
                    "server_name": server_name,
                    "error": "Test syntax error",
                    "status": "FAILED"
                }
            
            # ========================================
            # Step 3: Get Initial Test Statistics
            # ========================================
            logger.info(f"[{server_name}] Running initial tests...")
            statistics = get_test_statistics(test_path)
            
            if statistics is None:
                logger.error(f"[{server_name}] Failed to get test statistics")
                return {
                    "success": False,
                    "server_name": server_name,
                    "error": "Test execution error",
                    "status": "FAILED"
                }
            
            if statistics == "All passed":
                logger.info(f"[{server_name}] All tests passed on first run!")
                return {
                    "success": True,
                    "server_name": server_name,
                    "status": "All passed"
                }
            
            logger.info(f"[{server_name}] Initial statistics: {statistics}")
            
            # ========================================
            # Step 4: Test-Fix Loop
            # ========================================
            failed_count = statistics.get("failed", 0)
            expected_num_repair = failed_count * max_fix_attempts
            
            logger.info(f"[{server_name}] Starting fix loop ({expected_num_repair} max attempts)")
            
            attempts_per_test = defaultdict(int)
            server_code_bak = server_path.read_text()
            test_code_bak = test_path.read_text()
            
            for attempt in range(expected_num_repair):
                # Run tests until first failure
                passed, output = run_tests(test_path)
                
                if passed:
                    logger.info(f"[{server_name}] All tests passed after {attempt} fixes!")
                    return {
                        "success": True,
                        "server_name": server_name,
                        "status": "All passed"
                    }
                
                # Extract failure info
                failure_info = extract_test_failures(output)
                
                # Analyze failure
                server_code = server_path.read_text()
                test_code = test_path.read_text()
                
                analysis = analyze_test_failure(
                    failure_info, policy_content, server_code, test_code, client
                )
                
                if not analysis:
                    logger.warning(f"[{server_name}] Failed to analyze test failure")
                    break
                
                test_name = analysis.get("failed_test_name", "unknown")
                error_location = analysis.get("likely_bug_location", "unit test")
                reason = analysis.get("explanation", "")
                
                logger.info(f"[{server_name}] Analysis: {error_location} - {test_name}")
                
                attempts_per_test[test_name] += 1
                
                # Check if max retries exceeded for this test
                if attempts_per_test[test_name] > max_fix_attempts:
                    logger.warning(f"[{server_name}] Max retries exceeded for {test_name}, stopping")
                    # Rollback to backup
                    server_path.write_text(server_code_bak)
                    test_path.write_text(test_code_bak)
                    break
                
                # Fix based on error location
                if error_location == "function":
                    logger.info(f"[{server_name}] Fixing MCP server code...")
                    server_code_bak = server_code
                    fix_code_with_editor(server_path, failure_info, client, editor, max_retries=3)
                elif error_location in ["unit test", "test data"]:
                    logger.info(f"[{server_name}] Fixing unit test code...")
                    test_code_bak = test_code
                    fix_code_with_editor(test_path, failure_info, client, editor, max_retries=3)
                else:
                    logger.warning(f"[{server_name}] Unknown error location: {error_location}")
                    continue
            
            # ========================================
            # Step 5: Final Statistics
            # ========================================
            final_stats = get_test_statistics(test_path)
            
            if final_stats == "All passed":
                logger.info(f"[{server_name}] Final: All tests passed!")
                return {
                    "success": True,
                    "server_name": server_name,
                    "status": "All passed"
                }
            elif isinstance(final_stats, dict):
                logger.info(f"[{server_name}] Final statistics: {final_stats}")
                return {
                    "success": False,
                    "server_name": server_name,
                    "status": final_stats,
                    "error": f"Tests failed: {final_stats}"
                }
            else:
                return {
                    "success": False,
                    "server_name": server_name,
                    "status": "FAILED",
                    "error": "Failed to get final statistics"
                }
            
        except Exception as e:
            logger.warning(f"[{server_name}] Exception: {e}")
            return {
                "success": False,
                "server_name": server_name,
                "error": str(e),
                "status": "FAILED"
            }
    
    # Process servers
    results = parallel_process(
        items=to_process,
        process_func=implement_server,
        description="Implementing servers",
    )
    
    # Update global record
    for result in results:
        if result:
            server_name = result.get("server_name")
            status = result.get("status")
            if server_name:
                global_record[server_name] = status
    
    # Save global record
    save_json(global_record, global_record_path)
    
    # Count results
    success_count = sum(1 for r in results if r and r.get("success"))
    all_passed_count = sum(
        1 for v in global_record.values() 
        if v == "All passed"
    )
    
    state.mcp_servers_dir = str(servers_dir)
    state.unit_tests_dir = str(tests_dir)
    
    # Update progress
    state.update_step_progress(
        "s09_mcp_server_implementation",
        total=len(blueprints),
        completed=all_passed_count
    )
    
    logger.info(f"MCP server implementation complete: {success_count}/{len(to_process)} succeeded this run")
    logger.info(f"Total progress: {all_passed_count}/{len(blueprints)} all passed")
    
    # ========================================
    # Post-processing: Fix Server Class Names
    # ========================================
    # 如果 MCP Server Name 以 Server 结尾但类名不是 xxxServerServer，
    # 则将类名改为 xxxServerServer 以匹配 mcp_tool_factory.py 的期望
    fixed_count = fix_server_class_names(servers_dir)
    if fixed_count > 0:
        logger.info(f"Fixed {fixed_count} server class names (added 'Server' suffix)")
    
    return state
