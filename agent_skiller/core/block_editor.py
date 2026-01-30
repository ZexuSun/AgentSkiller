"""
Block Editor for SEARCH/REPLACE operations.

This module provides the BlockEditor class for parsing and applying
SEARCH/REPLACE blocks from LLM responses. Used for:
- Modifying code (MCP Server, Unit Tests, Database scripts, etc.)
- Modifying Markdown (Domain Policies)

The implementation is based on the original block_editor.py with
workflow-specific enhancements.
"""

import re
import logging
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Generator, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EditResult:
    """Result of applying edit blocks to a file."""
    
    success_count: int
    fail_count: int
    failed_edits: List[Tuple[str, str]]
    error_message: Optional[str] = None


class BlockEditor:
    """
    Editor for SEARCH/REPLACE block operations.
    
    Parses and applies edit blocks in the format:
    
    <<<<<<< SEARCH
    original text to find
    =======
    replacement text
    >>>>>>> REPLACE
    
    Features:
    - Exact match replacement
    - Whitespace-tolerant matching
    - Ellipsis (...) support for partial matches
    - Similar line detection for failed matches
    """
    
    # Regex patterns for block markers
    HEAD = r"^<{5,9} SEARCH>?\s*$"
    DIVIDER = r"^={5,9}\s*$"
    UPDATED = r"^>{5,9} REPLACE\s*$"
    
    # Error message markers
    HEAD_ERR = "<<<<<<< SEARCH"
    DIVIDER_ERR = "======="
    UPDATED_ERR = ">>>>>>> REPLACE"
    
    def __init__(self):
        pass
    
    def _find_original_update_blocks(
        self, 
        content: str
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Find all SEARCH/REPLACE blocks in content.
        
        Args:
            content: Content containing edit blocks
            
        Yields:
            Tuples of (original_text, updated_text)
        """
        lines = content.splitlines(keepends=True)
        
        head_pattern = re.compile(self.HEAD)
        divider_pattern = re.compile(self.DIVIDER)
        updated_pattern = re.compile(self.UPDATED)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if head_pattern.match(line.strip()):
                try:
                    original_text = []
                    i += 1
                    while i < len(lines) and not divider_pattern.match(lines[i].strip()):
                        original_text.append(lines[i])
                        i += 1
                    
                    if i >= len(lines) or not divider_pattern.match(lines[i].strip()):
                        raise ValueError(f"Expected `{self.DIVIDER_ERR}`")
                    
                    updated_text = []
                    i += 1
                    while i < len(lines) and not (
                        updated_pattern.match(lines[i].strip()) or 
                        divider_pattern.match(lines[i].strip())
                    ):
                        updated_text.append(lines[i])
                        i += 1
                    
                    if i >= len(lines) or not (
                        updated_pattern.match(lines[i].strip()) or 
                        divider_pattern.match(lines[i].strip())
                    ):
                        raise ValueError(f"Expected `{self.UPDATED_ERR}` or `{self.DIVIDER_ERR}`")
                    
                    yield "".join(original_text), "".join(updated_text)
                    
                except ValueError as e:
                    processed = "".join(lines[:i + 1])
                    err = e.args[0]
                    raise ValueError(f"{processed}\n^^^ {err}")
            i += 1
    
    def _get_edits(self, response_content: str) -> List[Tuple[str, str]]:
        """
        Parse edit blocks from LLM response content.
        
        Args:
            response_content: LLM response containing edit blocks
            
        Returns:
            List of (original, updated) tuples
        """
        return list(self._find_original_update_blocks(response_content))
    
    @staticmethod
    def _strip_quoted_wrapping(res: str) -> str:
        """
        Remove code block wrapping from content.
        
        Handles:
        ```
        content
        ```
        """
        if not res:
            return res
        
        res = res.splitlines()
        
        if res[0].startswith("```") and res[-1].startswith("```"):
            res = res[1:-1]
        
        res = "\n".join(res)
        if res and res[-1] != "\n":
            res += "\n"
        
        return res
    
    def _prepare_content(self, content: str) -> Tuple[str, List[str]]:
        """Prepare content for matching."""
        if content and not content.endswith("\n"):
            content += "\n"
        lines = content.splitlines(keepends=True)
        return content, lines
    
    def _perfect_replace(
        self, 
        whole_lines: List[str], 
        part_lines: List[str], 
        replace_lines: List[str]
    ) -> Optional[str]:
        """Try exact match replacement."""
        part_tup = tuple(part_lines)
        part_len = len(part_lines)
        
        for i in range(len(whole_lines) - part_len + 1):
            whole_tup = tuple(whole_lines[i:i + part_len])
            if part_tup == whole_tup:
                res = whole_lines[:i] + replace_lines + whole_lines[i + part_len:]
                return "".join(res)
        
        return None
    
    def _match_but_for_leading_whitespace(
        self, 
        whole_lines: List[str], 
        part_lines: List[str]
    ) -> Optional[str]:
        """Check if lines match ignoring leading whitespace."""
        num = len(whole_lines)
        
        # Does the non-whitespace all agree?
        if not all(
            whole_lines[i].lstrip() == part_lines[i].lstrip() 
            for i in range(num)
        ):
            return None
        
        # Are they all offset the same?
        add = set(
            whole_lines[i][:len(whole_lines[i]) - len(part_lines[i])]
            for i in range(num)
            if whole_lines[i].strip()
        )
        
        if len(add) != 1:
            return None
        
        return add.pop()
    
    def _replace_part_with_missing_leading_whitespace(
        self, 
        whole_lines: List[str], 
        part_lines: List[str], 
        replace_lines: List[str]
    ) -> Optional[str]:
        """Handle LLM-induced whitespace issues."""
        # Outdent everything by the max fixed amount possible
        leading = [
            len(p) - len(p.lstrip()) 
            for p in part_lines if p.strip()
        ] + [
            len(p) - len(p.lstrip()) 
            for p in replace_lines if p.strip()
        ]
        
        if leading and min(leading):
            num_leading = min(leading)
            part_lines = [
                p[num_leading:] if p.strip() else p 
                for p in part_lines
            ]
            replace_lines = [
                p[num_leading:] if p.strip() else p 
                for p in replace_lines
            ]
        
        num_part_lines = len(part_lines)
        
        for i in range(len(whole_lines) - num_part_lines + 1):
            add_leading = self._match_but_for_leading_whitespace(
                whole_lines[i:i + num_part_lines], 
                part_lines
            )
            
            if add_leading is None:
                continue
            
            replace_lines = [
                add_leading + rline if rline.strip() else rline 
                for rline in replace_lines
            ]
            whole_lines = (
                whole_lines[:i] + 
                replace_lines + 
                whole_lines[i + num_part_lines:]
            )
            return "".join(whole_lines)
        
        return None
    
    def _perfect_or_whitespace(
        self, 
        whole_lines: List[str], 
        part_lines: List[str], 
        replace_lines: List[str]
    ) -> Optional[str]:
        """Try perfect match, then whitespace-tolerant match."""
        # Try for a perfect match
        res = self._perfect_replace(whole_lines, part_lines, replace_lines)
        if res:
            return res
        
        # Try being flexible about leading whitespace
        res = self._replace_part_with_missing_leading_whitespace(
            whole_lines, part_lines, replace_lines
        )
        return res
    
    def _try_dotdotdots(
        self, 
        whole: str, 
        part: str, 
        replace: str
    ) -> Optional[str]:
        """
        Handle edit blocks with ... (ellipsis) lines.
        
        Allows matching and replacing partial content.
        """
        dots_re = re.compile(r"(^\s*\.\.\.\n)", re.MULTILINE | re.DOTALL)
        
        part_pieces = re.split(dots_re, part)
        replace_pieces = re.split(dots_re, replace)
        
        if len(part_pieces) != len(replace_pieces):
            raise ValueError("Unpaired ... in SEARCH/REPLACE block")
        
        if len(part_pieces) == 1:
            return None
        
        # Compare odd strings (the ... markers)
        all_dots_match = all(
            part_pieces[i] == replace_pieces[i] 
            for i in range(1, len(part_pieces), 2)
        )
        
        if not all_dots_match:
            raise ValueError("Unmatched ... in SEARCH/REPLACE block")
        
        # Get the actual content pieces
        part_pieces = [part_pieces[i] for i in range(0, len(part_pieces), 2)]
        replace_pieces = [replace_pieces[i] for i in range(0, len(replace_pieces), 2)]
        
        for part, replace in zip(part_pieces, replace_pieces):
            if not part and not replace:
                continue
            
            if not part and replace:
                if not whole.endswith("\n"):
                    whole += "\n"
                whole += replace
                continue
            
            if whole.count(part) == 0:
                raise ValueError(f"Part not found: {part[:50]}...")
            if whole.count(part) > 1:
                raise ValueError(f"Part found multiple times: {part[:50]}...")
            
            whole = whole.replace(part, replace, 1)
        
        return whole
    
    def _replace_most_similar_chunk(
        self, 
        whole: str, 
        part: str, 
        replace: str
    ) -> Optional[str]:
        """
        Best effort to find and replace content.
        
        Tries multiple matching strategies.
        """
        whole, whole_lines = self._prepare_content(whole)
        part, part_lines = self._prepare_content(part)
        replace, replace_lines = self._prepare_content(replace)
        
        # Try exact or whitespace-tolerant match
        res = self._perfect_or_whitespace(whole_lines, part_lines, replace_lines)
        if res:
            return res
        
        # Try skipping leading blank line
        if len(part_lines) > 2 and not part_lines[0].strip():
            skip_blank = part_lines[1:]
            res = self._perfect_or_whitespace(whole_lines, skip_blank, replace_lines)
            if res:
                return res
        
        # Try ellipsis handling
        try:
            res = self._try_dotdotdots(whole, part, replace)
            if res:
                return res
        except ValueError:
            pass
        
        return None
    
    def _do_replace(
        self, 
        content: str, 
        before_text: str, 
        after_text: str
    ) -> Optional[str]:
        """Apply a single replacement."""
        before_text = self._strip_quoted_wrapping(before_text)
        after_text = self._strip_quoted_wrapping(after_text)
        
        if not before_text.strip():
            # Empty search = append
            return content + after_text
        
        return self._replace_most_similar_chunk(content, before_text, after_text)
    
    def _find_similar_lines(
        self, 
        search_lines: str, 
        content_lines: str, 
        threshold: float = 0.6
    ) -> str:
        """Find similar lines for error reporting."""
        search_lines_list = search_lines.splitlines()
        content_lines_list = content_lines.splitlines()
        
        best_ratio = 0
        best_match = None
        best_match_i = 0
        
        for i in range(len(content_lines_list) - len(search_lines_list) + 1):
            chunk = content_lines_list[i:i + len(search_lines_list)]
            ratio = SequenceMatcher(None, search_lines_list, chunk).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = chunk
                best_match_i = i
        
        if best_ratio < threshold:
            return ""
        
        if (
            best_match[0] == search_lines_list[0] and 
            best_match[-1] == search_lines_list[-1]
        ):
            return "\n".join(best_match)
        
        # Expand context
        N = 5
        best_match_end = min(
            len(content_lines_list), 
            best_match_i + len(search_lines_list) + N
        )
        best_match_i = max(0, best_match_i - N)
        
        return "\n".join(content_lines_list[best_match_i:best_match_end])
    
    def apply_edits(
        self, 
        full_path: Path, 
        edits: List[Tuple[str, str]], 
        edit_logger: Optional[logging.Logger] = None
    ) -> Optional[str]:
        """
        Apply edit blocks to a file.
        
        Args:
            full_path: Path to file to modify
            edits: List of (original, updated) tuples
            edit_logger: Logger for edit operations
            
        Returns:
            Error message if any edits failed, None if all succeeded
        """
        if edit_logger is None:
            edit_logger = logger
        
        failed = []
        passed = []
        
        for edit in edits:
            edit_logger.info(f"""
{"!" * 50} Applying Edits {"!" * 50}
<<<<<<< SEARCH
{edit[0]}
=======
{edit[1]}
>>>>>>> REPLACE
""")
            original, updated = edit
            content = open(full_path, "r").read()
            new_content = self._do_replace(content, original, updated)
            
            if new_content:
                open(full_path, "w").write(new_content)
                passed.append(edit)
            else:
                failed.append(edit)
        
        if not failed:
            return None
        
        # Build error message
        blocks = "block" if len(failed) == 1 else "blocks"
        res = f"# {len(failed)} SEARCH/REPLACE {blocks} failed to match!\n"
        
        for edit in failed:
            original, updated = edit
            content = open(full_path, "r").read()
            
            res += f"""
## SearchReplaceNoExactMatch: This SEARCH block failed to exactly match lines in {full_path}
<<<<<<< SEARCH
{original}=======
{updated}>>>>>>> REPLACE

"""
            did_you_mean = self._find_similar_lines(original, content)
            if did_you_mean:
                res += f"""Did you mean to match some of these actual lines from {full_path}?

```
{did_you_mean}
```

"""
            if updated in content and updated:
                res += f"""Are you sure you need this SEARCH/REPLACE block?
The REPLACE lines are already in {full_path}!

"""
        
        res += (
            "The SEARCH section must exactly match an existing block of lines "
            "including all white space, comments, indentation, docstrings, etc\n"
        )
        
        if passed:
            pblocks = "block" if len(passed) == 1 else "blocks"
            res += f"""
# The other {len(passed)} SEARCH/REPLACE {pblocks} were applied successfully.
Don't re-send them.
Just reply with fixed versions of the {blocks} above that failed to match.
"""
        
        return res
    
    def apply_edits_from_response(
        self, 
        file_path: Path, 
        llm_response: str,
        edit_logger: Optional[logging.Logger] = None
    ) -> EditResult:
        """
        Parse LLM response and apply edits to a file.
        
        Args:
            file_path: Path to file to modify
            llm_response: LLM response containing SEARCH/REPLACE blocks
            edit_logger: Optional logger for edit operations
            
        Returns:
            EditResult with success/fail counts
        """
        edits = self._get_edits(llm_response)
        
        if not edits:
            return EditResult(
                success_count=0,
                fail_count=0,
                failed_edits=[],
                error_message="No edit blocks found in response"
            )
        
        error_msg = self.apply_edits(file_path, edits, edit_logger)
        
        if error_msg:
            # Count failures from error message
            fail_count = len([e for e in edits if e[0] in error_msg])
            success_count = len(edits) - fail_count
            failed_edits = [e for e in edits if e[0] in error_msg]
        else:
            success_count = len(edits)
            fail_count = 0
            failed_edits = []
        
        return EditResult(
            success_count=success_count,
            fail_count=fail_count,
            failed_edits=failed_edits,
            error_message=error_msg
        )


class WorkflowBlockEditor(BlockEditor):
    """
    Workflow-specific block editor with LLM integration.
    
    Extends BlockEditor with methods for:
    - LLM-driven code fixing
    - LLM-driven markdown fixing
    - Automatic retry with error feedback
    """
    
    # Prompts for different fix types
    CODE_FIX_PROMPT = """
The following code has an error:

Error: {error}

File: {file_path}

```{language}
{content}
```

Please fix the error using SEARCH/REPLACE blocks:

<<<<<<< SEARCH
original text to find
=======
replacement text
>>>>>>> REPLACE

Only output the SEARCH/REPLACE blocks, no explanations.
"""
    
    MARKDOWN_FIX_PROMPT = """
The following markdown has issues:

Issues:
{issues}

File: {file_path}

```markdown
{content}
```

Please fix the issues using SEARCH/REPLACE blocks:

<<<<<<< SEARCH
original text to find
=======
replacement text
>>>>>>> REPLACE

Only output the SEARCH/REPLACE blocks, no explanations.
"""
    
    JSON_FIX_PROMPT = """
The following JSON has a parsing error:

Error: {error}

```json
{content}
```

Please fix the JSON syntax error using SEARCH/REPLACE blocks:

<<<<<<< SEARCH
original text with error
=======
corrected text
>>>>>>> REPLACE

Common JSON errors to fix:
- Trailing commas before ] or }}
- Missing commas between elements
- Unescaped quotes in strings
- Incomplete JSON (add missing closing brackets)

Only output the SEARCH/REPLACE blocks, no explanations.
"""
    
    def fix_code(
        self,
        file_path: Path,
        error: str,
        llm_client: Any,
        language: str = "python",
        max_retries: int = 3,
    ) -> bool:
        """
        Use LLM to fix code file.
        
        Args:
            file_path: Path to code file
            error: Error message
            llm_client: LLM client instance
            language: Programming language
            max_retries: Maximum fix attempts
            
        Returns:
            True if file was fixed, False otherwise
        """
        content = file_path.read_text()
        original_content = content
        
        for attempt in range(max_retries):
            query = self.CODE_FIX_PROMPT.format(
                error=error,
                file_path=str(file_path),
                language=language,
                content=content,
            )
            
            response = llm_client.chat(
                query=query,
                model_type="coding",
            )
            
            result = self.apply_edits_from_response(file_path, response.content)
            
            if result.fail_count == 0 and result.success_count > 0:
                logger.info(f"Fixed code in {file_path} after {attempt + 1} attempt(s)")
                return True
            
            if result.error_message:
                error = result.error_message
                content = file_path.read_text()
        
        logger.warning(f"Failed to fix code in {file_path} after {max_retries} attempts")
        return False
    
    def fix_markdown(
        self,
        file_path: Path,
        issues: List[str],
        llm_client: Any,
        max_retries: int = 3,
    ) -> bool:
        """
        Use LLM to fix markdown file.
        
        Args:
            file_path: Path to markdown file
            issues: List of issues to fix
            llm_client: LLM client instance
            max_retries: Maximum fix attempts
            
        Returns:
            True if file was fixed, False otherwise
        """
        content = file_path.read_text()
        
        for attempt in range(max_retries):
            query = self.MARKDOWN_FIX_PROMPT.format(
                issues="\n".join(f"- {issue}" for issue in issues),
                file_path=str(file_path),
                content=content,
            )
            
            response = llm_client.chat(
                query=query,
                model_type="textual",
            )
            
            result = self.apply_edits_from_response(file_path, response.content)
            
            if result.fail_count == 0 and result.success_count > 0:
                logger.info(f"Fixed markdown in {file_path} after {attempt + 1} attempt(s)")
                return True
            
            if result.error_message:
                issues = [result.error_message]
                content = file_path.read_text()
        
        logger.warning(f"Failed to fix markdown in {file_path} after {max_retries} attempts")
        return False
    
    def fix_json_content(
        self,
        json_content: str,
        error: str,
        llm_client: Any,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Use LLM to fix JSON content.
        
        Args:
            json_content: JSON string with errors
            error: Error message
            llm_client: LLM client instance
            max_retries: Maximum fix attempts
            
        Returns:
            Fixed JSON string, or None if unfixable
        """
        import tempfile
        
        for attempt in range(max_retries):
            query = self.JSON_FIX_PROMPT.format(
                error=error,
                content=json_content,
            )
            
            response = llm_client.chat(
                query=query,
                model_type="coding",
            )
            
            edits = self._get_edits(response.content)
            
            if not edits:
                logger.warning("No edit blocks found in LLM response")
                continue
            
            # Apply edits to content string
            for original, updated in edits:
                original = self._strip_quoted_wrapping(original)
                updated = self._strip_quoted_wrapping(updated)
                
                if original in json_content:
                    json_content = json_content.replace(original, updated, 1)
            
            # Try to parse
            import json
            try:
                json.loads(json_content)
                logger.info(f"Fixed JSON after {attempt + 1} attempt(s)")
                return json_content
            except json.JSONDecodeError as e:
                error = str(e)
        
        logger.warning(f"Failed to fix JSON after {max_retries} attempts")
        return None
