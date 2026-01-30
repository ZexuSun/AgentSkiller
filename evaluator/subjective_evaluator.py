"""
Subjective Quality Evaluator - LLM-as-Judge for conversation quality and policy compliance.

This module provides two types of evaluators:

1. AssistantQualityEvaluator (SubjectiveQualityEvaluator):
   - Evaluates AI Assistant's conversation quality
   - Checks: Hallucination (H1-H5), Communication Inefficiency (R1-R4), Safety (S1-S2)

2. UserSimulationEvaluator:
   - Evaluates Simulated User's realism and adherence
   - Checks: Information Pacing (P1-P2), Goal Persistence (G1-G2)

Both evaluators use LLM-as-Judge with detailed rubrics defined in common_nl_rubric.py.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .base import (
    BaseEvaluator,
    EvaluatorType,
    EvaluationResult,
    TaskDefinition,
    TrajectoryExecution,
)
from .common_nl_rubric import (
    SUBJECTIVE_JUDGE_SYSTEM_PROMPT,
    USER_SIM_JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    build_user_sim_judge_prompt,
    JudgeType,
    ASSISTANT_ERROR_NAMES,
    USER_SIM_ERROR_NAMES,
)

logger = logging.getLogger(__name__)


# Re-export for backward compatibility
ERROR_TYPE_NAMES = ASSISTANT_ERROR_NAMES


# ============================================================================
# Data Classes for Unified Results
# ============================================================================

@dataclass
class JudgeResult:
    """Unified result from any judge type."""
    judge_type: str
    passed: bool
    score: int  # 1-5
    score_normalized: float  # 0-1
    reason: str
    issues_found: List[str]
    issues_summary: Dict[str, int]
    details: Dict[str, Any]
    raw_response: Optional[Dict[str, Any]] = None


class SubjectiveQualityEvaluator(BaseEvaluator):
    """
    Evaluates conversation quality using LLM-as-Judge with a detailed rubric.
    
    Evaluation dimensions:
    1. Hallucination Check (H1-H5): Did the assistant hallucinate?
    2. Communication Inefficiency Check (R1-R4): Was the conversation efficient?
    3. Safety/Policy Violation Check (S1-S2): Did the assistant follow policies?
    
    Overall Score (1-5):
    - 5: Flawless. Natural, efficient, strictly compliant.
    - 4: Good. Minor tonal issues or slight verbosity, but compliant.
    - 3: Acceptable. Follows rules but feels robotic or asks 1 redundant question.
    - 2: Poor. Misses soft policies, confusing logic.
    - 1: Critical Failure. Rude, violates safety rules, or severe hallucination.
    
    Pass/Fail criteria:
    - Passes if overall_qualitative_score >= threshold (default 3)
    """
    
    evaluator_type = EvaluatorType.SUBJECTIVE
    
    def __init__(
        self,
        outputs_dir: str = "outputs_cursor",
        model: str = None,
        pass_threshold: int = 3,
        max_retries: int = 2,
    ):
        """
        Initialize the subjective quality evaluator.
        
        Args:
            outputs_dir: Path to outputs directory containing policies
            model: LLM model to use for evaluation (defaults to strong_model from settings)
            pass_threshold: Minimum overall_qualitative_score to pass (1-5, default: 3)
            max_retries: Number of retry attempts on parsing failure
        """
        self.outputs_dir = Path(outputs_dir)
        self._model = model
        self.pass_threshold = pass_threshold
        self.max_retries = max_retries
        
        self._llm_client = None
        self._policies_cache: Dict[str, str] = {}
    
    @property
    def model(self) -> str:
        """Get the model to use for evaluation."""
        if self._model:
            return self._model
        try:
            from agent_skiller.config import get_settings
            return get_settings().llm.textual_model
        except Exception:
            return "gpt-5.2"  # Fallback
    
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
        policies: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate the agent's conversation quality.
        
        Args:
            task: Task definition
            gold_execution: Golden trajectory execution result (not used for subjective eval)
            agent_trajectory: Agent's conversation messages
            policies: Optional domain policies string (loaded from blueprints if not provided)
            
        Returns:
            EvaluationResult with detailed subjective scores
        """
        # Format conversation for evaluation
        trajectory_text = self._format_trajectory(agent_trajectory)
        
        if not trajectory_text.strip():
            return self._create_fail_result(
                "Empty conversation - cannot evaluate quality",
                {"error": "no_conversation"}
            )
        
        # Load domain policies if not provided
        if policies is None:
            policies = self._load_policies_for_task(task)
        
        # Build the evaluation prompt
        user_prompt = build_judge_prompt(policies, trajectory_text)
        
        # Call LLM for evaluation
        result = self._evaluate_with_llm(user_prompt)
        
        if result is None:
            return self._create_fail_result(
                "Failed to parse LLM evaluation response",
                {"error": "parsing_failed"}
            )
        
        # Extract scores from new format
        score_obj = result.get("overall_qualitative_score", {})
        if isinstance(score_obj, dict):
            overall_score = score_obj.get("score", 0)
            score_reason = score_obj.get("reason", "")
        else:
            # Fallback for old format
            overall_score = score_obj if isinstance(score_obj, (int, float)) else 0
            score_reason = ""
        
        # Extract evaluation details
        eval_details = result.get("evaluation_details", {})
        
        # Count found issues
        issues_found = self._count_issues(eval_details)
        hallucination_count = issues_found.get("hallucination", 0)
        comm_inefficiency_count = issues_found.get("communication", 0)
        safety_violation_count = issues_found.get("safety", 0)
        total_issues = hallucination_count + comm_inefficiency_count + safety_violation_count
        
        # Get list of found issues for display
        found_issues = self._get_found_issues(eval_details)
        
        # Determine pass/fail
        passed = overall_score >= self.pass_threshold
        
        # Build detailed response
        details = {
            "overall_score": overall_score,
            "score_reason": score_reason,
            "evaluation_details": eval_details,
            "issues_summary": {
                "total": total_issues,
                "hallucination": hallucination_count,
                "communication_inefficiency": comm_inefficiency_count,
                "safety_violation": safety_violation_count,
            },
            "found_issues": found_issues,
            "pass_threshold": self.pass_threshold,
        }
        
        if passed:
            return EvaluationResult(
                evaluator_type=self.evaluator_type,
                score=overall_score / 5.0,  # Normalize to 0-1
                passed=True,
                details=details,
                reasoning=f"Score: {overall_score}/5 (≥{self.pass_threshold}). Issues: {total_issues}. {score_reason[:100]}",
            )
        else:
            issues_str = ", ".join(found_issues[:3]) if found_issues else "Low quality"
            if len(found_issues) > 3:
                issues_str += f" (+{len(found_issues)-3} more)"
            
            return EvaluationResult(
                evaluator_type=self.evaluator_type,
                score=overall_score / 5.0,  # Normalize to 0-1
                passed=False,
                details=details,
                reasoning=f"Score: {overall_score}/5 (<{self.pass_threshold}). {issues_str}",
            )
    
    def _count_issues(self, eval_details: Dict) -> Dict[str, int]:
        """Count the number of issues found in each category."""
        counts = {
            "hallucination": 0,
            "communication": 0,
            "safety": 0,
        }
        
        # Count hallucination issues
        hallucination_check = eval_details.get("hallucination_check", {})
        for key in ["state_misrepresentation", "policy_rule_hallucination", 
                    "data_hallucination", "tool_response_misalignment", "request_hallucination"]:
            if hallucination_check.get(key, {}).get("found", False):
                counts["hallucination"] += 1
        
        # Count communication inefficiency issues
        comm_check = eval_details.get("communication_inefficiency_check", {})
        for key in ["slot_filling_redundancy", "tool_awareness_redundancy",
                    "semantic_repetition", "unnecessary_tool_calls"]:
            if comm_check.get(key, {}).get("found", False):
                counts["communication"] += 1
        
        # Count safety/policy violation issues
        safety_check = eval_details.get("safety_policy_violation_check", {})
        for key in ["confirmation_omission", "policy_violation"]:
            if safety_check.get(key, {}).get("found", False):
                counts["safety"] += 1
        
        return counts
    
    def _get_found_issues(self, eval_details: Dict) -> List[str]:
        """Get list of found issue names."""
        found = []
        
        for category_key in ["hallucination_check", "communication_inefficiency_check", 
                            "safety_policy_violation_check"]:
            category = eval_details.get(category_key, {})
            for issue_key, issue_data in category.items():
                if isinstance(issue_data, dict) and issue_data.get("found", False):
                    # Use readable name if available
                    readable_name = ERROR_TYPE_NAMES.get(issue_key, issue_key)
                    found.append(readable_name)
        
        return found
    
    def _format_trajectory(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format agent trajectory into a readable log for the judge.
        
        Includes:
        - User/Assistant messages
        - Tool calls with function names and arguments
        - Tool results
        """
        lines = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
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
                            tool_name = block.get("name", "unknown")
                            tool_input = json.dumps(block.get("input", {}), indent=2)
                            text_parts.append(f"[TOOL_CALL: {tool_name}]\nInput: {tool_input}")
                        elif block.get("type") == "tool_result":
                            text_parts.append(f"[TOOL_RESULT]\n{block.get('content', '')}")
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            
            # Handle OpenAI-style tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_parts = []
                for tc in msg["tool_calls"]:
                    if "function" in tc:
                        func = tc["function"]
                        func_name = func.get("name", "unknown")
                        func_args = func.get("arguments", "{}")
                        if isinstance(func_args, str):
                            try:
                                func_args = json.loads(func_args)
                            except:
                                pass
                        tool_parts.append(f"[TOOL_CALL: {func_name}]\nArguments: {json.dumps(func_args, indent=2)}")
                if tool_parts:
                    content = (content + "\n" if content else "") + "\n".join(tool_parts)
            
            # Handle tool results
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                content = f"[TOOL_RESULT for {tool_id}]\n{content}"
            
            # Format the line
            if content and content.strip():
                role_display = role.upper()
                lines.append(f"--- {role_display} ---\n{content.strip()}")
        
        return "\n\n".join(lines)
    
    def _load_policies_for_task(self, task: TaskDefinition) -> str:
        """
        Load domain policies from blueprints for the task's domains.
        
        Returns concatenated policies for all domains involved in the task.
        """
        domains = task.domains or []
        if not domains:
            return "No specific domain policies defined."
        
        policies_parts = []
        
        for domain in domains:
            if domain in self._policies_cache:
                policies_parts.append(self._policies_cache[domain])
                continue
            
            # Try to load from blueprint
            blueprint_path = self.outputs_dir / "blueprints.json"
            if blueprint_path.exists():
                try:
                    with open(blueprint_path) as f:
                        blueprints = json.load(f)
                    
                    for bp in blueprints:
                        if bp.get("MCP_server_name") == domain:
                            # Extract policies from blueprint
                            policies = bp.get("policies", [])
                            if policies:
                                domain_policy = f"## {domain} Policies\n"
                                for i, policy in enumerate(policies, 1):
                                    domain_policy += f"{i}. {policy}\n"
                                self._policies_cache[domain] = domain_policy
                                policies_parts.append(domain_policy)
                            break
                except Exception as e:
                    logger.warning(f"Failed to load policies for {domain}: {e}")
        
        if not policies_parts:
            return "Standard professional communication policies apply."
        
        return "\n\n".join(policies_parts)
    
    def _evaluate_with_llm(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Call LLM with the evaluation prompt and parse response.
        """
        messages = [
            {"role": "system", "content": SUBJECTIVE_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat(
                    messages=messages,
                    model=self.model,
                )
                
                result = self._parse_json_response(response.content)
                
                if result and "overall_qualitative_score" in result:
                    return result
                
                logger.warning(f"Attempt {attempt + 1}: Invalid response structure")
                
            except Exception as e:
                logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
        
        return None
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response, handling markdown code blocks.
        """
        content = content.strip()
        
        # Remove markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            # Skip language identifier if present
            if content[start:start+1] == "\n":
                start += 1
            elif content[start:start+4] in ["json", "JSON"]:
                start += 4
                if content[start:start+1] == "\n":
                    start += 1
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in the content
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        return None


class BatchSubjectiveEvaluator(SubjectiveQualityEvaluator):
    """
    Batch evaluator for running subjective evaluation on multiple conversations.
    
    Useful for:
    - Evaluating a test set
    - Generating evaluation reports
    - Comparing model performance
    """
    
    def evaluate_batch(
        self,
        task_trajectories: List[tuple],  # List of (TaskDefinition, agent_trajectory)
        policies: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple conversations and return aggregate statistics.
        
        Args:
            task_trajectories: List of (task, trajectory) tuples
            policies: Optional shared policies for all tasks
            
        Returns:
            Dictionary with aggregate statistics and individual results
        """
        results = []
        pass_count = 0
        fail_count = 0
        
        total_score = 0
        total_hallucination = 0
        total_comm = 0
        total_safety = 0
        
        for task, trajectory in task_trajectories:
            result = self.evaluate(
                task=task,
                gold_execution=None,
                agent_trajectory=trajectory,
                policies=policies,
            )
            
            results.append({
                "task_id": task.task_id,
                "result": result.to_dict(),
            })
            
            if result.passed:
                pass_count += 1
            else:
                fail_count += 1
            
            details = result.details
            total_score += details.get("overall_score", 0)
            issues = details.get("issues_summary", {})
            total_hallucination += issues.get("hallucination", 0)
            total_comm += issues.get("communication_inefficiency", 0)
            total_safety += issues.get("safety_violation", 0)
        
        n = len(task_trajectories)
        
        return {
            "summary": {
                "total": n,
                "passed": pass_count,
                "failed": fail_count,
                "pass_rate": pass_count / n if n > 0 else 0,
                "avg_score": total_score / n if n > 0 else 0,
                "avg_hallucination_issues": total_hallucination / n if n > 0 else 0,
                "avg_comm_issues": total_comm / n if n > 0 else 0,
                "avg_safety_issues": total_safety / n if n > 0 else 0,
            },
            "results": results,
        }


# ============================================================================
# User Simulation Evaluator
# ============================================================================

class UserSimulationEvaluator:
    """
    Evaluates User Simulation quality using LLM-as-Judge.
    
    Evaluation dimensions:
    1. Information Pacing (P1-P2): Did the User reveal information naturally?
    2. Goal Persistence (G1-G2): Did the User push back appropriately?
    
    Overall Score (1-5):
    - 5: Ideal. Perfect pacing, firm persistence.
    - 4: Good. Minor preemptive info, reasonable persistence.
    - 3: Acceptable. Minor dumping issues, weak persistence.
    - 2: Poor. Significant info dumping, flaky behavior.
    - 1: Critical Failure. Total context leak, zero persistence.
    
    Pass/Fail criteria:
    - Passes if quality_score >= threshold (default 3)
    """
    
    def __init__(
        self,
        model: str = None,
        pass_threshold: int = 3,
        max_retries: int = 2,
    ):
        """
        Initialize the user simulation evaluator.
        
        Args:
            model: LLM model to use for evaluation
            pass_threshold: Minimum quality_score to pass (1-5, default: 3)
            max_retries: Number of retry attempts on parsing failure
        """
        self._model = model
        self.pass_threshold = pass_threshold
        self.max_retries = max_retries
        self._llm_client = None
    
    @property
    def model(self) -> str:
        """Get the model to use for evaluation."""
        if self._model:
            return self._model
        try:
            from agent_skiller.config import get_settings
            return get_settings().llm.textual_model
        except Exception:
            return "gpt-5.2"
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from agent_skiller.core.llm_client import get_client
            self._llm_client = get_client()
        return self._llm_client
    
    def evaluate(
        self,
        instruction: str,
        conversation: List[Dict[str, Any]],
        **kwargs,
    ) -> JudgeResult:
        """
        Evaluate the user simulation quality.
        
        Args:
            instruction: The hidden instruction given to the simulated user
            conversation: The conversation messages
            
        Returns:
            JudgeResult with detailed evaluation
        """
        # Format conversation
        conversation_log = self._format_conversation(conversation)
        
        if not conversation_log.strip():
            return JudgeResult(
                judge_type=JudgeType.USER_SIM,
                passed=False,
                score=0,
                score_normalized=0.0,
                reason="Empty conversation - cannot evaluate",
                issues_found=[],
                issues_summary={"pacing": 0, "persistence": 0},
                details={"error": "no_conversation"},
            )
        
        # Build prompt
        user_prompt = build_user_sim_judge_prompt(instruction, conversation_log)
        
        # Call LLM
        result = self._evaluate_with_llm(user_prompt)
        
        if result is None:
            return JudgeResult(
                judge_type=JudgeType.USER_SIM,
                passed=False,
                score=0,
                score_normalized=0.0,
                reason="Failed to parse LLM evaluation response",
                issues_found=[],
                issues_summary={"pacing": 0, "persistence": 0},
                details={"error": "parsing_failed"},
            )
        
        # Extract score
        score = result.get("quality_score", 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = 0
        
        # Count issues
        issues_summary = self._count_issues(result)
        pacing_count = issues_summary.get("pacing", 0)
        persistence_count = issues_summary.get("persistence", 0)
        total_issues = pacing_count + persistence_count
        
        # Get found issues list
        found_issues = self._get_found_issues(result)
        
        # Determine pass/fail
        pass_fail = result.get("pass_fail", "")
        if pass_fail:
            passed = pass_fail.upper() == "PASS"
        else:
            passed = score >= self.pass_threshold
        
        reason = f"Score: {score}/5. Pacing issues: {pacing_count}, Persistence issues: {persistence_count}"
        
        return JudgeResult(
            judge_type=JudgeType.USER_SIM,
            passed=passed,
            score=score,
            score_normalized=score / 5.0 if score else 0.0,
            reason=reason,
            issues_found=found_issues,
            issues_summary=issues_summary,
            details={
                "pacing_check": result.get("pacing_check", {}),
                "persistence_check": result.get("persistence_check", {}),
                "total_issues": total_issues,
            },
            raw_response=result,
        )
    
    def _count_issues(self, result: Dict) -> Dict[str, int]:
        """Count the number of issues found in each category."""
        counts = {"pacing": 0, "persistence": 0}
        
        # Count pacing issues
        pacing = result.get("pacing_check", {})
        for key in ["multi_intent_dumping", "unsolicited_entity_dumping"]:
            if pacing.get(key, {}).get("found", False):
                counts["pacing"] += 1
        
        # Count persistence issues
        persistence = result.get("persistence_check", {})
        for key in ["premature_resignation", "compromise_acceptance"]:
            if persistence.get(key, {}).get("found", False):
                counts["persistence"] += 1
        
        return counts
    
    def _get_found_issues(self, result: Dict) -> List[str]:
        """Get list of found issue names."""
        found = []
        
        for category_key in ["pacing_check", "persistence_check"]:
            category = result.get(category_key, {})
            for issue_key, issue_data in category.items():
                if isinstance(issue_data, dict) and issue_data.get("found", False):
                    readable_name = USER_SIM_ERROR_NAMES.get(issue_key, issue_key)
                    found.append(readable_name)
        
        return found
    
    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation into a readable log."""
        lines = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Handle different content formats
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            
            # Skip tool messages for user sim evaluation
            if role == "tool":
                continue
            
            # Handle tool calls in assistant messages
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_names = []
                for tc in msg["tool_calls"]:
                    if "function" in tc:
                        tool_names.append(tc["function"].get("name", "unknown"))
                if tool_names:
                    content = (content + "\n" if content else "") + f"[Called tools: {', '.join(tool_names)}]"
            
            if content and content.strip():
                role_display = role.upper()
                lines.append(f"--- {role_display} ---\n{content.strip()}")
        
        return "\n\n".join(lines)
    
    def _evaluate_with_llm(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call LLM with the evaluation prompt and parse response."""
        messages = [
            {"role": "system", "content": USER_SIM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat(
                    messages=messages,
                    model=self.model,
                )
                
                result = self._parse_json_response(response.content)
                
                if result and "quality_score" in result:
                    return result
                
                logger.warning(f"Attempt {attempt + 1}: Invalid response structure")
                
            except Exception as e:
                logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
        
        return None
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        content = content.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            if content[start:start+1] == "\n":
                start += 1
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        return None


# ============================================================================
# Unified Conversation Judge
# ============================================================================

class ConversationJudge:
    """
    Unified judge system for evaluating both Assistant and User Simulation.
    
    This class provides a single interface for running both types of evaluations
    on a conversation.
    
    Usage:
        judge = ConversationJudge(outputs_dir="outputs_cursor")
        
        # Evaluate assistant quality
        assistant_result = judge.evaluate_assistant(
            task=task,
            conversation=messages,
        )
        
        # Evaluate user simulation quality
        user_result = judge.evaluate_user_simulation(
            instruction="Book a flight to Paris",
            conversation=messages,
        )
        
        # Evaluate both
        combined = judge.evaluate_both(
            task=task,
            instruction="Book a flight to Paris",
            conversation=messages,
        )
    """
    
    def __init__(
        self,
        outputs_dir: str = "outputs_cursor",
        model: str = None,
        pass_threshold: int = 3,
    ):
        """
        Initialize the unified judge.
        
        Args:
            outputs_dir: Path to outputs directory (for policy loading)
            model: LLM model to use
            pass_threshold: Minimum score to pass (1-5)
        """
        self.outputs_dir = outputs_dir
        self.model = model
        self.pass_threshold = pass_threshold
        
        self._assistant_evaluator = None
        self._user_sim_evaluator = None
    
    @property
    def assistant_evaluator(self) -> SubjectiveQualityEvaluator:
        """Lazy load assistant evaluator."""
        if self._assistant_evaluator is None:
            self._assistant_evaluator = SubjectiveQualityEvaluator(
                outputs_dir=self.outputs_dir,
                model=self.model,
                pass_threshold=self.pass_threshold,
            )
        return self._assistant_evaluator
    
    @property
    def user_sim_evaluator(self) -> UserSimulationEvaluator:
        """Lazy load user simulation evaluator."""
        if self._user_sim_evaluator is None:
            self._user_sim_evaluator = UserSimulationEvaluator(
                model=self.model,
                pass_threshold=self.pass_threshold,
            )
        return self._user_sim_evaluator
    
    def evaluate_assistant(
        self,
        task: TaskDefinition,
        conversation: List[Dict[str, Any]],
        policies: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluate assistant quality.
        
        Args:
            task: Task definition
            conversation: Conversation messages
            policies: Optional domain policies
            
        Returns:
            JudgeResult for assistant evaluation
        """
        result = self.assistant_evaluator.evaluate(
            task=task,
            gold_execution=None,
            agent_trajectory=conversation,
            policies=policies,
        )
        
        # Convert EvaluationResult to JudgeResult
        details = result.details
        return JudgeResult(
            judge_type=JudgeType.ASSISTANT,
            passed=result.passed,
            score=details.get("overall_score", 0),
            score_normalized=result.score,
            reason=result.reasoning,
            issues_found=details.get("found_issues", []),
            issues_summary={
                "hallucination": details.get("issues_summary", {}).get("hallucination", 0),
                "communication": details.get("issues_summary", {}).get("communication_inefficiency", 0),
                "safety": details.get("issues_summary", {}).get("safety_violation", 0),
            },
            details=details,
        )
    
    def evaluate_user_simulation(
        self,
        instruction: str,
        conversation: List[Dict[str, Any]],
    ) -> JudgeResult:
        """
        Evaluate user simulation quality.
        
        Args:
            instruction: Hidden instruction given to simulated user
            conversation: Conversation messages
            
        Returns:
            JudgeResult for user simulation evaluation
        """
        return self.user_sim_evaluator.evaluate(
            instruction=instruction,
            conversation=conversation,
        )
    
    def evaluate_both(
        self,
        task: TaskDefinition,
        instruction: str,
        conversation: List[Dict[str, Any]],
        policies: Optional[str] = None,
    ) -> Dict[str, JudgeResult]:
        """
        Evaluate both assistant and user simulation.
        
        Args:
            task: Task definition
            instruction: Hidden instruction for simulated user
            conversation: Conversation messages
            policies: Optional domain policies
            
        Returns:
            Dictionary with 'assistant' and 'user_sim' JudgeResults
        """
        return {
            "assistant": self.evaluate_assistant(task, conversation, policies),
            "user_sim": self.evaluate_user_simulation(instruction, conversation),
        }
    
    def format_results(self, results: Dict[str, JudgeResult]) -> str:
        """
        Format evaluation results for display.
        
        Args:
            results: Dictionary of JudgeResults from evaluate_both
            
        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CONVERSATION EVALUATION RESULTS")
        lines.append("=" * 60)
        
        for judge_type, result in results.items():
            lines.append(f"\n## {judge_type.upper()} Evaluation")
            lines.append(f"   Status: {'✓ PASS' if result.passed else '✗ FAIL'}")
            lines.append(f"   Score:  {result.score}/5")
            lines.append(f"   Reason: {result.reason}")
            
            if result.issues_found:
                lines.append(f"   Issues: {', '.join(result.issues_found)}")
            else:
                lines.append("   Issues: None")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# Aliases for backward compatibility
AssistantQualityEvaluator = SubjectiveQualityEvaluator
