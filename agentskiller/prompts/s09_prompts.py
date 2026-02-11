"""
Prompts for Step 09: MCP Server Implementation

Implement and test MCP servers from blueprints.
"""

# =============================================================================
# MCP Server Implementation
# =============================================================================

MCP_SERVER_IMPLEMENT_PROMPT = """
You must implement an MCP server as a complete Python module.

Produce the complete MCP server implementation according to the domain policy,
database blueprint, MCP server code template, and the behavioral constraints described as follows.

=====================
## 0. Inputs
=====================

### 0.1 Domain Policy
{domain_policy}

### 0.2 MCP Server Blueprint
{blueprint}

### 0.3 MCP Server Code Template
{mcp_server_code_template}

=====================
## 1. Implementation Target
=====================

Generate a Python file containing:
- A session context @dataclass
  - Session state management
  - Session database isolation (independent database per session_id)
- A full MCP server class
  - All tools implemented as class methods
  - A unified router entrypoint: invoke(session_id, tool_name, **kwargs)
    - `session_id` helps you find the session context and database copies
    - `tool_name` is the tool you would like to call
    - `**kwargs` are the parameters of the tool you would like to call
  - Domain-policy violation checking
- All "xx_id" must satisfy {{entity_name.lower()}}_id.

### 1.1 Interface
The implemented MCP server must be compatible with the standard wrapper interface.

### 1.2 Load Database
Get database root by
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
db_root = os.path.abspath(os.path.join(script_dir, '..', 'database', 'outputs'))
```

**NOTE**
- `__init__(self, domain_name)` should call `_load_database(domain_name)` during initialization.
- Never mock new databases instead of reading from the given files.
- The current time is {simulation_time}.

### 1.3 Output Rules
- All tools must return JSON-serializable Python objects
- No printing
- No placeholder methods

=====================
## 2. CRITICAL IMPLEMENTATION RULES
=====================

### 2.1 Confirmation Pattern (REQUIRED for data-changing functions)
If the policy mentions that a function requires "user confirmation" before execution:
- Add a `confirm: bool = False` parameter to the function
- When `confirm=False`: Return a preview of what will happen with `{{"needs_confirmation": True, "action_preview": "...description..."}}`
- When `confirm=True`: Execute the actual action and return the result
- The LLM Assistant will first call with confirm=False, show the preview to user, then call again with confirm=True after user says "yes"

### 2.2 Accept Extra Parameters (REQUIRED for ALL functions)
Every function MUST accept `**kwargs` as the last parameter to gracefully ignore unexpected parameters.

### 2.3 Core Entity ID Handling
- Authorization binds the core entity ID to the session
- After authorization, functions should use `session.authorized_{{core_entity}}_id` instead of requiring it as a parameter

=====================
## 3. Output Constraints
=====================

1. Do NOT include unit tests.
2. The final output must be valid Python code (no markdown).
3. All imports must be included.
4. The module must execute without modification.
5. EVERY function must end with `**kwargs` in its signature.
6. Data-changing functions MUST implement the confirm pattern if policy requires confirmation.

Output only the Python code, no markdown formatting.
"""


# =============================================================================
# Unit Test Generation
# =============================================================================

UNIT_TEST_PROMPT = """
Your job is to generate comprehensive unit tests for the MCP server below.

## 1.1 Tested MCP Server
```python
{server_code}
```

## 1.2 Domain Policy
{policy}

## 2. Test Behavior Details

### 2.1 Router Tests
- invoke() must dispatch the correct tool
- invoke() must reject unknown tool_name

### 2.2 Session Tests
- Two sessions with different sample_id must not influence each other
- Modifications in one session must not appear in the other

### 2.3 Policy Violation Tests
Use the rules in domain policy to construct:
  - Valid calls (no violation)
  - Invalid calls (expected to return an error)

### 2.4 Tool Tests
For each tool in the tool list:
- Test normal execution

### 2.5 Database Tests
- Ensure temporary DB is created
- Ensure global DB is untouched
- Ensure each session DB is isolated

## 3. Template
```python
{pytest_template}
```

## 4. Import
```python
from {outputs_dir}.mcp_servers.{domain_name} import {domain_name}Server, SessionContext
```

Output only the Python code, no markdown formatting.
"""


# =============================================================================
# Code Fix Prompts
# =============================================================================

CODE_FIX_PROMPT = """
You are an expert software engineer specializing in debugging Python code.
Your task is to analyze a bug and provide a fix in the form of SEARCH/REPLACE blocks.

I am debugging the file `{file_path}`. When I run the tests, I encounter:
<output_or_error>
{output_or_error}
</output_or_error>

Here is the full content of {file_path}:
```python
{full_code_content}
```

Please analyze the error and provide fixes using *SEARCH/REPLACE block* format:
```python
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Rules:
- Every *SEARCH* section must *EXACTLY MATCH* the existing file content
- Include enough context lines to uniquely identify the location
- Keep blocks concise
"""


# =============================================================================
# Error Trace Analysis
# =============================================================================

ERROR_TRACE = """
I'm building an MCP server following the policy below
<policy>
{policy}
</policy>

Here is the current implementation:
```python
{mcp_server_code}
```

Such code failed in the following unit test:
<unit_test_failure>
{unit_test_failure}
</unit_test_failure>

The test data generator:
<test_data_generator>
{test_data_generator}
</test_data_generator>

## Task
Identify the reason that caused the failure:
- Is it because of wrong implementation of the MCP server?
- Is it simply a wrong unit test?
- Is it related to improper test data selection?

IMPORTANT: You MUST respond with ONLY a valid JSON object. No additional text before or after.

```json
{{
    "failed_test_name": "<exact test function name>",
    "likely_bug_location": "<one of: function | unit test | test data>",
    "explanation": "<brief explanation of what's wrong>"
}}
```

Respond with ONLY the JSON object above, nothing else.
"""
