"""
Agentic Feedback Loop for Self-Correction

Implements the closed-loop revision protocol: when a scorer agent rejects
an artifact, specific critique is routed back to the Generator agent, which
revises and re-submits.  The loop continues until the artifact passes all
gates or the maximum revision rounds are exhausted.

This transforms the pipeline from a one-pass "generate → verify → discard"
flow into an agentic "generate → verify → critique → revise → re-verify"
harness, which is characteristic of modern agent frameworks (LangGraph,
AutoGen, CrewAI).

Protocol:
  1. Generator produces artifact
  2. Scorer evaluates → if pass, keep; if fail, produce critique
  3. Critique + artifact routed back to Generator
  4. Generator revises based on critique
  5. Revised artifact re-submitted for verification
  6. Repeat until pass or max rounds reached
  7. If max rounds exhausted → artifact is DISCARDED

Key metrics tracked:
  - Revision success rate (how often revision leads to acceptance)
  - Mean rounds to acceptance
  - Score improvement trajectory across revisions
  - Critique quality (did specific critiques lead to better revisions?)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from kg_quant.agents.protocol import (
    AgentRole,
    MessageType,
    AgentMessage,
    Artifact,
    HarnessConfig,
)


# ────────────────────────────────────────────────────────────────
# Revision Prompts
# ────────────────────────────────────────────────────────────────

REVISION_PROMPT_TEMPLATE = """
# Role
你是一个金融知识图谱构建专家。你之前生成的一个 {artifact_type} 被质量评审专家拒绝了。

# 原始产物
{original_content}

# 你的原始推理过程
{original_reasoning}

# 评审专家的反馈
{critique}

# Task
请根据评审专家的反馈，修改你的 {artifact_type}。

要求：
1. 认真对待每一条批评意见
2. 如果批评合理，按建议修改
3. 如果批评不合理，你可以坚持原方案，但必须给出清晰的反驳理由
4. 修改时要保持 {artifact_type} 的金融学严谨性
5. 不要为了通过评审而牺牲质量——只在你认为确实需要修改时才修改

# 输出格式
```json
{{
  "revised_content": <修改后的完整内容>,
  "revision_reasoning": "你做了哪些修改，为什么；或者你为什么坚持原方案",
  "revision_type": "major_revision|minor_revision|no_change"
}}
```
"""


# ────────────────────────────────────────────────────────────────
# Feedback Result
# ────────────────────────────────────────────────────────────────

@dataclass
class RevisionRecord:
    """A single revision attempt within a feedback loop."""

    round_idx: int
    artifact_before: Dict[str, Any]
    artifact_after: Dict[str, Any]
    critique: str
    revision_type: str           # major_revision | minor_revision | no_change
    revision_reasoning: str
    score_before: float
    score_after: float | None = None
    passed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "critique": self.critique,
            "revision_type": self.revision_type,
            "revision_reasoning": self.revision_reasoning,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "passed": self.passed,
            "timestamp": self.timestamp,
        }


@dataclass
class FeedbackLoopResult:
    """Complete result of a feedback loop for one artifact."""

    artifact_id: str
    artifact_type: str
    initial_score: float
    final_score: float | None = None
    accepted: bool = False
    total_rounds: int = 0
    revisions: List[RevisionRecord] = field(default_factory=list)
    final_status: str = "DISCARDED"

    @property
    def score_improvement(self) -> float:
        if self.final_score is not None:
            return self.final_score - self.initial_score
        return 0.0

    @property
    def improvement_per_round(self) -> float:
        if self.total_rounds > 0:
            return self.score_improvement / self.total_rounds
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "accepted": self.accepted,
            "total_rounds": self.total_rounds,
            "score_improvement": self.score_improvement,
            "revisions": [r.to_dict() for r in self.revisions],
            "final_status": self.final_status,
        }


# ────────────────────────────────────────────────────────────────
# Feedback Loop Engine
# ────────────────────────────────────────────────────────────────

class FeedbackLoopEngine:
    """Agentic feedback loop for artifact revision and re-verification.

    Usage:
        engine = FeedbackLoopEngine(config, generator_fn, scorer_fn)
        result = engine.run(artifact, initial_critique)
    """

    def __init__(
        self,
        config: HarnessConfig | None = None,
        generator_fn: Callable[..., Any] | None = None,
        scorer_fn: Callable[..., Any] | None = None,
    ):
        self.config = config or HarnessConfig()
        self._generator_fn = generator_fn
        self._scorer_fn = scorer_fn

    # ── Public API ───────────────────────────────────────────────

    def run(
        self,
        artifact: Artifact,
        initial_score: float,
        critique: str,
    ) -> FeedbackLoopResult:
        """Run the feedback loop for a rejected artifact.

        Args:
            artifact: The rejected artifact.
            initial_score: The score that caused rejection.
            critique: Specific feedback from the scorer agent.

        Returns:
            FeedbackLoopResult with full revision history.
        """
        result = FeedbackLoopResult(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            initial_score=initial_score,
        )

        current_artifact = artifact
        current_score = initial_score

        for round_idx in range(1, self.config.max_revision_rounds + 1):
            # ── Revision Phase ───────────────────────────────────
            revision_prompt = REVISION_PROMPT_TEMPLATE.format(
                artifact_type=current_artifact.artifact_type,
                original_content=json.dumps(
                    current_artifact.content, ensure_ascii=False, indent=2
                ),
                original_reasoning=current_artifact.reasoning_trace,
                critique=critique,
            )

            if self._generator_fn is None:
                # Without a generator function, we cannot revise
                break

            revision_output = self._generator_fn(revision_prompt)
            revised_content, revision_type, revision_reasoning = (
                self._parse_revision_output(revision_output)
            )

            # Record revision
            record = RevisionRecord(
                round_idx=round_idx,
                artifact_before=current_artifact.content,
                artifact_after=revised_content,
                critique=critique,
                revision_type=revision_type,
                revision_reasoning=revision_reasoning,
                score_before=current_score,
            )
            result.revisions.append(record)

            # If no change, stop
            if revision_type == "no_change":
                result.final_score = current_score
                result.final_status = "REJECTED"
                result.accepted = False
                break

            # Update artifact
            current_artifact = Artifact(
                artifact_id=current_artifact.artifact_id,
                artifact_type=current_artifact.artifact_type,
                content=revised_content,
                reasoning_trace=(
                    current_artifact.reasoning_trace
                    + f"\n[Revision {round_idx}]: {revision_reasoning}"
                ),
                provenance=list(current_artifact.provenance),
                quality_scores=dict(current_artifact.quality_scores),
            )
            current_artifact.add_provenance(
                agent_role=AgentRole.GENERATOR,
                action=f"revision_round_{round_idx}",
                result={
                    "revision_type": revision_type,
                    "reasoning": revision_reasoning[:200],
                },
            )

            # ── Re-Verification Phase ────────────────────────────
            if self._scorer_fn is not None:
                new_score = self._scorer_fn(current_artifact)
                record.score_after = new_score
                current_score = new_score

                # Check if passed
                if new_score >= 0.6:  # Default acceptance threshold
                    record.passed = True
                    result.final_score = new_score
                    result.final_status = "VERIFIED"
                    result.accepted = True
                    break
                else:
                    # Update critique for next round
                    critique = (
                        f"Revised version scored {new_score:.2f}. "
                        f"Still below threshold (0.60). "
                        f"Previous critique: {critique}"
                    )
            else:
                # No scorer available — accept after revision
                record.passed = True
                result.final_score = None
                result.final_status = "VERIFIED"
                result.accepted = True
                break

        result.total_rounds = len(result.revisions)

        if not result.accepted and result.final_score is None:
            result.final_score = current_score

        return result

    # ── Helpers ──────────────────────────────────────────────────

    def _parse_revision_output(
        self, output: Any
    ) -> Tuple[Dict[str, Any], str, str]:
        """Parse the Generator's revision output."""
        import re

        if isinstance(output, str):
            try:
                json_match = re.search(
                    r'```json\s*(.*?)\s*```', output, re.DOTALL
                )
                if json_match:
                    output = json.loads(json_match.group(1))
                else:
                    json_match = re.search(
                        r'\{.*?"revised_content".*?\}', output, re.DOTALL
                    )
                    if json_match:
                        output = json.loads(json_match.group(0))
                    else:
                        return {}, "no_change", "Failed to parse revision output"
            except (json.JSONDecodeError, AttributeError):
                return {}, "no_change", "Failed to parse revision output"

        if isinstance(output, dict):
            return (
                output.get("revised_content", {}),
                output.get("revision_type", "minor_revision"),
                output.get("revision_reasoning", ""),
            )

        return {}, "no_change", "Unexpected output format"


# ────────────────────────────────────────────────────────────────
# Feedback Loop Analysis
# ────────────────────────────────────────────────────────────────

@dataclass
class FeedbackLoopSummary:
    """Aggregate metrics across all feedback loop sessions."""

    total_artifacts: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    mean_rounds: float = 0.0
    mean_score_improvement: float = 0.0
    revision_type_distribution: Dict[str, int] = field(default_factory=dict)
    round_acceptance_curve: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls, results: List[FeedbackLoopResult]
    ) -> "FeedbackLoopSummary":
        accepted = [r for r in results if r.accepted]
        summary = cls(
            total_artifacts=len(results),
            accepted_count=len(accepted),
            rejected_count=len([r for r in results if not r.accepted]),
            mean_rounds=(
                sum(r.total_rounds for r in results) / max(len(results), 1)
            ),
            mean_score_improvement=(
                sum(r.score_improvement for r in results)
                / max(len(results), 1)
            ),
        )

        # Revision type distribution
        for r in results:
            for rev in r.revisions:
                summary.revision_type_distribution[rev.revision_type] = (
                    summary.revision_type_distribution.get(rev.revision_type, 0) + 1
                )

        # Round acceptance curve: what fraction of remaining artifacts
        # get accepted at each round?
        remaining = len(results)
        for round_i in range(1, 6):
            if remaining <= 0:
                summary.round_acceptance_curve[round_i] = 0.0
                continue
            accepted_this_round = sum(
                1 for r in results
                if r.accepted and r.total_rounds == round_i
            )
            summary.round_acceptance_curve[round_i] = (
                accepted_this_round / max(remaining, 1)
            )
            remaining -= accepted_this_round

        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_artifacts": self.total_artifacts,
            "accepted_count": self.accepted_count,
            "acceptance_rate": (
                self.accepted_count / max(self.total_artifacts, 1)
            ),
            "rejected_count": self.rejected_count,
            "mean_rounds": self.mean_rounds,
            "mean_score_improvement": self.mean_score_improvement,
            "revision_type_distribution": self.revision_type_distribution,
            "round_acceptance_curve": self.round_acceptance_curve,
        }
