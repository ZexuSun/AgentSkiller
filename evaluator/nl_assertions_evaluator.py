"""
Natural Language Assertions Evaluator - LLM-as-Judge for subjective criteria.

This evaluator uses an LLM to evaluate subjective conditions that require
understanding context and nuance, such as:
- "Agent did not offer compensation before user requested it"
- "Agent verified the flight delay before proceeding"
- "Agent maintained a professional tone throughout"
"""

import json
import logging
from typing import Dict, List, Any, Optional

from .base import (
    BaseEvaluator,
    EvaluatorType,
    EvaluationResult,
    TaskDefinition,
    TrajectoryExecution,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts for LLM-as-Judge
# =============================================================================

NL_ASSERTION_PROMPT = """
You are an expert evaluator assessing whether an AI assistant's conversation satisfies specific criteria.

## Conversation Transcript
{conversation}

## Criteria to Evaluate
{assertions}

## Task Context
Instruction: {instruction}
Reason for call: {reason}

## Your Task
For each criterion, determine if it is SATISFIED or NOT SATISFIED based on the conversation.

## Output Format
Return a JSON object with this structure:
```json
{{
    "evaluations": [
        {{
            "criterion": "the criterion text",
            "satisfied": true/false,
            "reasoning": "brief explanation of why"
        }}
    ],
    "all_satisfied": true/false
}}
```

Be strict but fair. Only mark a criterion as satisfied if there is clear evidence in the conversation.
Output ONLY the JSON, no additional text.
"""


SINGLE_ASSERTION_PROMPT = """
You are evaluating whether an AI assistant's conversation satisfies a specific criterion.

## Conversation
{conversation}

## Criterion to Evaluate
{assertion}

## Context
The user's instruction was: {instruction}
The reason for calling was: {reason}

## Your Task
Determine if this criterion is SATISFIED based on the conversation evidence.

## Output Format
Return a JSON object:
```json
{{
    "satisfied": true/false,
    "reasoning": "your explanation",
    "evidence": "specific quote or observation from the conversation"
}}
```

Be strict but fair. Only mark satisfied if there is clear evidence.
Output ONLY the JSON.
"""


class NLAssertionsEvaluator(BaseEvaluator):
    """
    Evaluates subjective criteria using LLM-as-Judge.
    
    The evaluator:
    1. Formats the conversation into a readable transcript
    2. Sends it to an LLM with the assertions to evaluate
    3. Parses the LLM's structured judgment
    4. Returns 1.0 only if ALL assertions are satisfied
    
    Configuration:
    - model: Which model to use for evaluation
    - batch_assertions: Whether to evaluate all assertions in one call
    - require_evidence: Whether to require specific evidence
    """
    
    evaluator_type = EvaluatorType.NL_ASSERTIONS
    
    def __init__(
        self,
        model: str = None,
        batch_assertions: bool = True,
        require_evidence: bool = False,
        max_retries: int = 2,
    ):
        """
        Initialize the NL assertions evaluator.
        
        Args:
            model: LLM model to use for evaluation (defaults to strong_model from settings)
            batch_assertions: Evaluate all assertions in one call
            require_evidence: Require specific evidence for each judgment
            max_retries: Retry attempts on parsing failure
        """
        self._model = model
        self.batch_assertions = batch_assertions
        self.require_evidence = require_evidence
        self.max_retries = max_retries
        
        self._llm_client = None
    
    @property
    def model(self) -> str:
        """Get the model to use for evaluation."""
        if self._model:
            return self._model
        try:
            from agent_skiller.config import get_settings
            return get_settings().llm.strong_model
        except Exception:
            return "gpt-4o-mini"  # Fallback
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from agent_skiller.core.llm_client import get_client
            self._llm_client = get_client()
        return self._llm_client
    
    def evaluate(
        self,
        task: TaskDefinition,
        gold_execution: TrajectoryExecution,
        agent_trajectory: List[Dict[str, Any]],
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate agent's conversation against NL assertions.
        
        Args:
            task: Task definition with nl_assertions
            gold_execution: Golden trajectory execution result
            agent_trajectory: Agent's conversation messages
            
        Returns:
            EvaluationResult with pass/fail and LLM reasoning
        """
        assertions = task.nl_assertions
        
        if not assertions:
            return self._create_skip_result("No NL assertions defined")
        
        # Ensure agent_trajectory is a list
        if agent_trajectory is None:
            agent_trajectory = []
        
        # Format conversation for LLM
        conversation = self._format_conversation(agent_trajectory)
        
        if not conversation.strip():
            return self._create_fail_result(
                "Empty conversation - cannot evaluate assertions",
                {"assertions": assertions}
            )
        
        # Evaluate assertions
        if self.batch_assertions:
            result = self._evaluate_batch(
                conversation=conversation,
                assertions=assertions,
                instruction=task.instruction,
                reason=task.reason_for_call,
            )
        else:
            result = self._evaluate_individual(
                conversation=conversation,
                assertions=assertions,
                instruction=task.instruction,
                reason=task.reason_for_call,
            )
        
        if result["all_satisfied"]:
            return self._create_pass_result({
                "assertions_count": len(assertions),
                "evaluations": result["evaluations"],
            })
        else:
            failed = [e for e in result["evaluations"] if not e.get("satisfied", False)]
            return self._create_fail_result(
                f"{len(failed)} assertion(s) not satisfied",
                {
                    "failed_assertions": failed,
                    "evaluations": result["evaluations"],
                }
            )
    
    def _format_conversation(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """
        Format conversation into a readable transcript.
        
        Returns:
            String transcript of the conversation
        """
        if not messages:
            return ""
        
        lines = []
        
        for msg in messages:
            if not msg:
                continue
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            # Handle different content formats
            if isinstance(content, list):
                # Anthropic-style content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            text_parts.append(
                                f"[Tool: {block.get('name', 'unknown')}]"
                            )
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = " ".join(text_parts)
            
            # Handle tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_info = []
                for tc in msg["tool_calls"]:
                    if tc and "function" in tc:
                        func = tc["function"]
                        tool_info.append(f"[Called: {func['name']}]")
                if tool_info:
                    content = (content + " " if content else "") + " ".join(tool_info)
            
            if "function_call" in msg:
                fc = msg["function_call"]
                content = (content + " " if content else "") + f"[Called: {fc['name']}]"
            
            # Format line
            if content.strip():
                lines.append(f"{role}: {content.strip()}")
        
        return "\n\n".join(lines)
    
    def _evaluate_batch(
        self,
        conversation: str,
        assertions: List[str],
        instruction: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Evaluate all assertions in a single LLM call.
        """
        # Format assertions as numbered list
        assertions_text = "\n".join(
            f"{i+1}. {assertion}" 
            for i, assertion in enumerate(assertions)
        )
        
        prompt = NL_ASSERTION_PROMPT.format(
            conversation=conversation,
            assertions=assertions_text,
            instruction=instruction,
            reason=reason,
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat(query=prompt, model=self.model)
                result = self._parse_json_response(response.content)
                
                if result and "evaluations" in result:
                    return result
                    
            except Exception as e:
                logger.warning(f"Batch evaluation attempt {attempt + 1} failed: {e}")
        
        # Fallback to individual evaluation
        logger.warning("Batch evaluation failed, falling back to individual")
        return self._evaluate_individual(conversation, assertions, instruction, reason)
    
    def _evaluate_individual(
        self,
        conversation: str,
        assertions: List[str],
        instruction: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Evaluate each assertion individually.
        """
        evaluations = []
        all_satisfied = True
        
        for assertion in assertions:
            prompt = SINGLE_ASSERTION_PROMPT.format(
                conversation=conversation,
                assertion=assertion,
                instruction=instruction,
                reason=reason,
            )
            
            result = {"criterion": assertion, "satisfied": False, "reasoning": ""}
            
            for attempt in range(self.max_retries):
                try:
                    response = self.llm_client.chat(query=prompt, model=self.model)
                    parsed = self._parse_json_response(response.content)
                    
                    if parsed:
                        result["satisfied"] = parsed.get("satisfied", False)
                        result["reasoning"] = parsed.get("reasoning", "")
                        result["evidence"] = parsed.get("evidence", "")
                        break
                        
                except Exception as e:
                    logger.warning(f"Individual eval attempt {attempt + 1} failed: {e}")
            
            evaluations.append(result)
            if not result["satisfied"]:
                all_satisfied = False
        
        return {
            "evaluations": evaluations,
            "all_satisfied": all_satisfied,
        }
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response.
        """
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1] == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the content
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        return None


class ConversationalQualityEvaluator(NLAssertionsEvaluator):
    """
    Extended evaluator for assessing conversational quality.
    
    Includes predefined assertions for common quality metrics:
    - Politeness and professionalism
    - Clarity and conciseness
    - Error handling
    - Information completeness
    """
    
    # Predefined quality assertions
    QUALITY_ASSERTIONS = {
        "professional_tone": "The assistant maintained a professional and helpful tone throughout the conversation",
        "clear_instructions": "The assistant provided clear and understandable instructions or responses",
        "error_handling": "When errors occurred, the assistant explained them clearly and offered solutions",
        "information_complete": "The assistant provided all relevant information without unnecessary omissions",
        "no_hallucination": "The assistant did not make claims or provide information not supported by the available data",
        "appropriate_confirmation": "The assistant appropriately confirmed important actions before executing them",
        "context_awareness": "The assistant demonstrated awareness of previous context in the conversation",
    }
    
    def __init__(
        self,
        quality_checks: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize with quality checks to perform.
        
        Args:
            quality_checks: List of quality check names from QUALITY_ASSERTIONS
        """
        super().__init__(**kwargs)
        self.quality_checks = quality_checks or []
    
    def evaluate(
        self,
        task: TaskDefinition,
        gold_execution: TrajectoryExecution,
        agent_trajectory: List[Dict[str, Any]],
        quality_checks: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate with both task assertions and quality checks.
        """
        # Combine task assertions with quality checks
        all_assertions = list(task.nl_assertions) if task.nl_assertions else []
        
        checks = quality_checks or self.quality_checks
        for check in checks:
            if check in self.QUALITY_ASSERTIONS:
                all_assertions.append(self.QUALITY_ASSERTIONS[check])
        
        # Create modified task with combined assertions
        from copy import copy
        modified_task = copy(task)
        modified_task.nl_assertions = all_assertions
        
        return super().evaluate(modified_task, gold_execution, agent_trajectory, **kwargs)