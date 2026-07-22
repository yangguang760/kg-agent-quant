"""
Deliberative Consensus Protocol for Multi-Agent Verification

Implements multi-turn agent deliberation when primary scoring reveals
disagreement among scorer agents.  Inspired by:

  - Aegean (2025): quorum-based consensus with stability windows
  - A-HMAD (2025): heterogeneous agent debate with role specialization
  - SC-MoA (2026): trace-level synthesis > voting

Protocol:
  Round 0: Primary independent scoring (existing behavior)
  Round 1+: Deliberation — each scorer sees peers' scores + reasoning,
            then independently re-evaluates
  Stop:    Consensus (std < threshold) OR max rounds reached

Key metrics tracked:
  - Round-by-round score distribution
  - Convergence speed (# rounds to consensus)
  - Score trajectory per agent (do agents converge or diverge?)
  - Final consensus quality vs single-round baseline
"""

from __future__ import annotations

import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from kg_quant.agents.protocol import (
    AgentRole,
    DeliberationConfig,
    DeliberationState,
)

# ────────────────────────────────────────────────────────────────
# Deliberation Prompts
# ────────────────────────────────────────────────────────────────

DELIBERATION_PROMPT_TEMPLATE = """
# Role
你还是金融知识图谱质量评审专家。现在进入**多轮审议 (Deliberation)** 阶段。

# Context
在上一轮独立评审中，你和其他评审专家对以下关系产生了**显著分歧**。

## 待审议的关系
**关系 ID**: {relation_id}
**头实体**: {head_entity}
**尾实体**: {tail_entity}
**关系类型**: {relation_type}
**原始证据**: {evidence}

## 上一轮各专家评审意见
{peer_reviews}

## 分歧程度
标准差 (std) = {disagreement_std:.3f}（阈值 = {threshold:.3f}）

# Task
请仔细阅读其他专家的评审意见，重新评估该关系的质量。

你可以选择：
1. **维持原评分** — 如果你坚持原来的判断，请解释为什么其他专家的顾虑不成立
2. **调整评分** — 如果其他专家提出了你忽略的视角，请说明你为什么调整
3. **部分调整** — 在某个维度上调高/调低

# 输出格式
```json
{{
  "existence_score": <0.0-1.0>,
  "logic_score": <0.0-1.0>,
  "confidence_score": <0.0-1.0>,
  "position": "maintain|revised|partial",
  "reasoning": "详细说明你为何维持/调整评分，特别是对其他专家意见的回应"
}}
```

# 要求
1. 只输出 JSON，不要其他文字
2. 你的评分必须基于金融学逻辑和证据
3. 不要为了避免分歧而修改评分——只在你真正被说服时才调整
"""


def _build_peer_review_section(
    model_scores: Dict[str, List[Dict[str, Any]]],
    relation_idx: int,
) -> str:
    """Build the peer review section showing each model's prior scores."""
    sections = []
    for model_name, scores_list in model_scores.items():
        if relation_idx < len(scores_list):
            s = scores_list[relation_idx]
            if hasattr(s, '__dict__'):
                s = s.__dict__ if hasattr(s, '__dict__') else vars(s)
            elif not isinstance(s, dict):
                s = {}
            sections.append(
                f"### 专家: {model_name}\n"
                f"- 存在性评分: {s.get('existence_score', 'N/A')}\n"
                f"- 逻辑评分: {s.get('logic_score', 'N/A')}\n"
                f"- 置信度评分: {s.get('confidence_score', 'N/A')}\n"
                f"- 评语: {s.get('comments', '无')}\n"
            )
    return "\n".join(sections)


# ────────────────────────────────────────────────────────────────
# Deliberation Engine
# ────────────────────────────────────────────────────────────────

@dataclass
class DeliberationResult:
    """Result of a deliberation process for a single artifact."""

    artifact_id: str
    primary_scores: Dict[str, float]       # model_name → confidence_score (Round 0)
    primary_std: float
    primary_status: str
    deliberation_triggered: bool
    deliberation_state: DeliberationState | None = None
    final_scores: Dict[str, float] = field(default_factory=dict)
    final_fused: float = 0.0
    final_std: float = 0.0
    final_status: str = ""
    rounds_to_converge: int = -1
    score_trajectories: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "primary_scores": self.primary_scores,
            "primary_std": self.primary_std,
            "primary_status": self.primary_status,
            "deliberation_triggered": self.deliberation_triggered,
            "deliberation_state": (
                self.deliberation_state.to_dict()
                if self.deliberation_state else None
            ),
            "final_scores": self.final_scores,
            "final_fused": self.final_fused,
            "final_std": self.final_std,
            "final_status": self.final_status,
            "rounds_to_converge": self.rounds_to_converge,
            "score_trajectories": self.score_trajectories,
        }

    @property
    def improvement(self) -> float:
        """Std reduction from primary to final (negative = got worse)."""
        return self.primary_std - self.final_std

    @property
    def status_changed(self) -> bool:
        return self.primary_status != self.final_status


class DeliberationEngine:
    """Multi-turn deliberation engine for agent consensus.

    Usage:
        engine = DeliberationEngine(config, llm_caller)
        results = engine.deliberate(
            artifacts_with_controversy,
            primary_model_scores,
        )
    """

    def __init__(
        self,
        config: DeliberationConfig | None = None,
        llm_caller: Callable[..., List[Any]] | None = None,
    ):
        self.config = config or DeliberationConfig()
        self._llm_caller = llm_caller

    # ── Public API ──────────────────────────────────────────────

    def should_deliberate(self, scores: List[float]) -> bool:
        """Check whether deliberation should be triggered."""
        return self.config.should_deliberate(scores)

    def deliberate_single(
        self,
        artifact_meta: Dict[str, Any],
        primary_scores: Dict[str, List[Any]],
        relation_idx: int,
        llm_configs: Dict[str, Dict[str, Any]],
    ) -> DeliberationResult:
        """Run deliberation for a single controversial artifact.

        Args:
            artifact_meta: Metadata for the artifact being evaluated.
            primary_scores: Primary round scores (model_name → [scores per artifact]).
            relation_idx: Index of this artifact in the scores arrays.
            llm_configs: LLM configurations keyed by model name.

        Returns:
            DeliberationResult with full trajectory.
        """
        # Extract primary scores for this artifact
        primary_confidence_scores: Dict[str, float] = {}
        for model_name, scores_list in primary_scores.items():
            if relation_idx < len(scores_list):
                s = scores_list[relation_idx]
                if hasattr(s, 'confidence_score'):
                    primary_confidence_scores[model_name] = s.confidence_score
                elif isinstance(s, dict):
                    primary_confidence_scores[model_name] = s.get('confidence_score', 0.5)
                else:
                    primary_confidence_scores[model_name] = 0.5

        score_values = list(primary_confidence_scores.values())
        primary_std = float(np.std(score_values)) if len(score_values) >= 2 else 0.0

        fused_primary = sum(score_values) / len(score_values) if score_values else 0.0
        is_controversial = primary_std > self.config.disagreement_threshold

        # Determine primary status
        if is_controversial:
            primary_status = "CONTROVERSIAL"
        elif fused_primary >= 0.8:
            primary_status = "VERIFIED_HIGH"
        elif fused_primary >= 0.6:
            primary_status = "VERIFIED_ACCEPTABLE"
        else:
            primary_status = "REJECTED"

        result = DeliberationResult(
            artifact_id=artifact_meta.get('id', 'unknown'),
            primary_scores=primary_confidence_scores,
            primary_std=primary_std,
            primary_status=primary_status,
            deliberation_triggered=is_controversial,
            final_scores=dict(primary_confidence_scores),
            final_std=primary_std,
            final_status=primary_status,
            score_trajectories={
                name: [score] for name, score in primary_confidence_scores.items()
            },
        )

        if not is_controversial:
            result.final_fused = fused_primary
            result.rounds_to_converge = 0
            return result

        # ── Deliberation rounds ──────────────────────────────────
        state = DeliberationState(artifact_id=result.artifact_id)
        state.record_round(
            round_idx=0,
            scores=primary_confidence_scores,
            statements={},
            std=primary_std,
        )

        current_scores = dict(primary_confidence_scores)
        current_std = primary_std

        for round_idx in range(1, self.config.max_rounds + 1):
            # Build peer review context
            peer_review_str = _build_peer_review_section(
                primary_scores, relation_idx
            )

            # Build deliberation prompt
            prompt = DELIBERATION_PROMPT_TEMPLATE.format(
                relation_id=artifact_meta.get('id', 'unknown'),
                head_entity=artifact_meta.get('head', 'N/A'),
                tail_entity=artifact_meta.get('tail', 'N/A'),
                relation_type=artifact_meta.get('type', 'N/A'),
                evidence=artifact_meta.get('evidence', '无'),
                peer_reviews=peer_review_str,
                disagreement_std=current_std,
                threshold=self.config.disagreement_threshold,
            )

            # Call all scorer models concurrently
            round_scores: Dict[str, float] = {}
            round_statements: Dict[str, str] = {}

            if self._llm_caller is not None:
                with ThreadPoolExecutor(max_workers=len(llm_configs)) as executor:
                    futures = {
                        executor.submit(
                            self._llm_caller, model_key, prompt
                        ): model_key
                        for model_key in llm_configs.keys()
                    }

                    for future in as_completed(futures):
                        model_key = futures[future]
                        try:
                            scores_list = future.result()
                            if isinstance(scores_list, list) and len(scores_list) > 0:
                                s = scores_list[0]
                                if hasattr(s, 'confidence_score'):
                                    round_scores[model_key] = s.confidence_score
                                elif isinstance(s, dict):
                                    round_scores[model_key] = s.get('confidence_score', 0.5)
                                else:
                                    round_scores[model_key] = 0.5

                                # Extract statement
                                if hasattr(s, 'comments'):
                                    round_statements[model_key] = s.comments
                                elif isinstance(s, dict):
                                    round_statements[model_key] = s.get('reasoning', s.get('comments', ''))
                            else:
                                round_scores[model_key] = current_scores.get(model_key, 0.5)
                                round_statements[model_key] = ""
                        except Exception as e:
                            print(f"  ⚠️  Deliberation round {round_idx}, model {model_key}: {e}")
                            round_scores[model_key] = current_scores.get(model_key, 0.5)
                            round_statements[model_key] = f"ERROR: {e}"

            # Compute new metrics
            new_values = list(round_scores.values())
            new_std = float(np.std(new_values)) if len(new_values) >= 2 else 0.0

            # Update trajectories
            for model_name in round_scores:
                if model_name not in result.score_trajectories:
                    result.score_trajectories[model_name] = []
                result.score_trajectories[model_name].append(round_scores[model_name])

            state.record_round(
                round_idx=round_idx,
                scores=round_scores,
                statements=round_statements,
                std=new_std,
            )

            current_scores = round_scores
            current_std = new_std

            # Check convergence
            if self.config.has_converged(list(round_scores.values())):
                state.mark_converged(round_idx, current_scores, current_std)
                break

        # ── Finalize ─────────────────────────────────────────────
        fused = (
            sum(current_scores.values()) / len(current_scores)
            if current_scores else 0.0
        )

        # Determine final status
        final_is_controversial = current_std > self.config.disagreement_threshold
        if final_is_controversial:
            final_status = "CONTROVERSIAL"
        elif fused >= 0.8:
            final_status = "VERIFIED_HIGH"
        elif fused >= 0.6:
            final_status = "VERIFIED_ACCEPTABLE"
        else:
            final_status = "REJECTED"

        result.deliberation_state = state
        result.final_scores = current_scores
        result.final_fused = round(fused, 3)
        result.final_std = round(current_std, 3)
        result.final_status = final_status
        result.rounds_to_converge = (
            state.converged_at_round if state.converged else -1
        )

        return result


# ────────────────────────────────────────────────────────────────
# Deliberation Analysis
# ────────────────────────────────────────────────────────────────

@dataclass
class DeliberationSummary:
    """Aggregate metrics across all deliberation sessions."""

    total_artifacts: int = 0
    triggered_count: int = 0             # How many triggered deliberation
    converged_count: int = 0             # How many reached consensus
    status_changed_count: int = 0        # How many changed status after deliberation
    mean_rounds_to_converge: float = 0.0
    mean_std_reduction: float = 0.0       # Average std improvement
    mean_score_shift: float = 0.0         # Average absolute score change
    status_transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: List[DeliberationResult]) -> "DeliberationSummary":
        triggered = [r for r in results if r.deliberation_triggered]

        summary = cls(
            total_artifacts=len(results),
            triggered_count=len(triggered),
            converged_count=sum(
                1 for r in triggered
                if r.deliberation_state and r.deliberation_state.converged
            ),
            status_changed_count=sum(1 for r in triggered if r.status_changed),
            mean_rounds_to_converge=(
                sum(r.rounds_to_converge for r in triggered
                    if r.rounds_to_converge > 0) / max(len(triggered), 1)
            ),
            mean_std_reduction=(
                sum(r.improvement for r in triggered) / max(len(triggered), 1)
            ),
            mean_score_shift=0.0,  # Computed below
        )

        # Compute score shifts
        shifts = []
        for r in triggered:
            if r.primary_scores and r.final_scores:
                for model in r.primary_scores:
                    if model in r.final_scores:
                        shifts.append(
                            abs(r.primary_scores[model] - r.final_scores[model])
                        )
        if shifts:
            summary.mean_score_shift = sum(shifts) / len(shifts)

        # Status transitions
        for r in triggered:
            fr = r.primary_status
            to = r.final_status
            if fr not in summary.status_transitions:
                summary.status_transitions[fr] = {}
            summary.status_transitions[fr][to] = (
                summary.status_transitions[fr].get(to, 0) + 1
            )

        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_artifacts": self.total_artifacts,
            "triggered_count": self.triggered_count,
            "triggered_rate": (
                self.triggered_count / max(self.total_artifacts, 1)
            ),
            "converged_count": self.converged_count,
            "convergence_rate": (
                self.converged_count / max(self.triggered_count, 1)
            ),
            "status_changed_count": self.status_changed_count,
            "status_change_rate": (
                self.status_changed_count / max(self.triggered_count, 1)
            ),
            "mean_rounds_to_converge": self.mean_rounds_to_converge,
            "mean_std_reduction": self.mean_std_reduction,
            "mean_score_shift": self.mean_score_shift,
            "status_transitions": self.status_transitions,
        }
