"""
Agent Communication Protocol for Distributed Multi-Agent Verification

Defines the formal inter-agent communication layer for the KG-AgentQuant
multi-agent verification harness.  Inspired by Google's A2A protocol and
current MAS consensus literature (Aegean, A-HMAD, SC-MoA).

Key concepts:
  - Role-specialized agents with dedicated verification criteria
  - Structured message passing with full provenance chains
  - Deliberation protocol for multi-turn consensus building
  - Agentic feedback loops for self-correction

Agent Topology:
    User Topic
        │
        ▼
    ┌──────────────────────────────────────┐
    │         Orchestrator (Harness)        │
    │   - lifecycle management              │
    │   - message routing                   │
    │   - provenance tracking               │
    │   - gate enforcement (CSC/EQ/SC)      │
    └──────────────────────────────────────┘
         │         ▲          ▲          ▲
         ▼         │ (revise)  │          │
    ┌─────────┐   │     ┌──────────┐  ┌──────────┐  ┌──────────┐
    │Generator│   │     │CSC Scorer│  │EQ Scorer │  │SC Scorer │
    │ Agent   │───┘     │ Agent    │  │ Agent    │  │ Agent    │
    │         │────────▶│          │─▶│          │─▶│          │
    │ Qwen    │         │GLM+Kimi  │  │GLM+Kimi  │  │GLM+Kimi  │
    └─────────┘         └──────────┘  └──────────┘  └──────────┘
                             │              │              │
                             ▼              ▼              ▼
                         Consensus?    Consensus?     Consensus?
                         ┌──┬──┐       ┌──┬──┐        ┌──┬──┐
                      Y→ │  │←N→   Y→  │  │←N→   Y→   │  │←N→
                      pass│  │Debate pass│  │Debate pass│  │Discard
                         └──┘          └──┘           └──┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid


# ═══════════════════════════════════════════════════════════════════════
# Agent Identity & Roles
# ═══════════════════════════════════════════════════════════════════════

class AgentRole(Enum):
    """Roles in the distributed verification agent topology.

    Each role has a distinct responsibility, evaluation rubric, and can
    be backed by a different LLM provider/model.
    """
    GENERATOR = "generator"
    CSC_SCORER = "csc_scorer"       # Consensus Score Calibration — relation quality
    EQ_SCORER = "eq_scorer"         # Explanation Quality — hypothesis coherence
    SC_SCORER = "sc_scorer"         # Semantic Consistency — factor-to-hypothesis fidelity
    ORCHESTRATOR = "orchestrator"   # Harness: lifecycle, routing, gate enforcement

    @property
    def is_scorer(self) -> bool:
        return self in (AgentRole.CSC_SCORER, AgentRole.EQ_SCORER, AgentRole.SC_SCORER)

    @property
    def verification_dimension(self) -> Optional[str]:
        _map = {
            AgentRole.CSC_SCORER: "relation_credibility",
            AgentRole.EQ_SCORER: "hypothesis_coherence",
            AgentRole.SC_SCORER: "factor_fidelity",
        }
        return _map.get(self)


# ═══════════════════════════════════════════════════════════════════════
# Message Types
# ═══════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    """Structured message types in the agent communication protocol."""

    # Generation flow
    GENERATION_REQUEST = "generation_request"       # Orchestrator → Generator: produce artifact
    GENERATION_RESULT = "generation_result"          # Generator → Orchestrator: artifact + reasoning

    # Verification flow
    VERIFICATION_REQUEST = "verification_request"    # Orchestrator → Scorer: evaluate artifact
    VERIFICATION_RESULT = "verification_result"       # Scorer → Orchestrator: score + critique

    # Deliberation flow (multi-turn consensus)
    DELIBERATION_INITIATE = "deliberation_initiate"  # Orchestrator → Scorers: begin debate
    DELIBERATION_STATEMENT = "deliberation_statement" # Scorer → Peers: reasoning exchange
    DELIBERATION_RESULT = "deliberation_result"       # Scorer → Orchestrator: revised score

    # Feedback loop (agentic self-correction)
    REVISION_REQUEST = "revision_request"            # Orchestrator → Generator: revise artifact
    REVISION_RESULT = "revision_result"               # Generator → Orchestrator: revised artifact

    # Lifecycle
    AGENT_REGISTER = "agent_register"                # Agent → Orchestrator: register capability
    AGENT_HEARTBEAT = "agent_heartbeat"              # Agent → Orchestrator: liveness check


# ═══════════════════════════════════════════════════════════════════════
# Artifact Status
# ═══════════════════════════════════════════════════════════════════════

class ArtifactStatus(Enum):
    """Lifecycle status of an artifact in the agent topology."""
    DRAFT = "draft"                       # Just generated, not yet verified
    UNDER_VERIFICATION = "under_verification"
    UNDER_DELIBERATION = "under_deliberation"
    UNDER_REVISION = "under_revision"
    VERIFIED = "verified"
    REJECTED = "rejected"
    DISCARDED = "discarded"


# ═══════════════════════════════════════════════════════════════════════
# Artifact
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Artifact:
    """A unit of work that flows through the agent topology.

    Each artifact carries its full provenance chain: every agent that
    has touched it, what they did, and what they concluded.  This is
    the "trace-level" information that SC-MoA shows is better than
    voting on final answers alone.

    Attributes:
        artifact_id: Unique identifier.
        artifact_type: "entity", "relation", "hypothesis", "factor_expression".
        content: The artifact body (structured dict).
        reasoning_trace: Generator's chain-of-thought (key for scorer evaluation).
        provenance: Ordered list of (agent_role, action, timestamp) entries.
        quality_scores: Accumulated scores from verification stages.
        status: Current lifecycle status.
    """

    artifact_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    artifact_type: str = ""               # entity | relation | hypothesis | factor_expression
    content: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: str = ""             # Generator's CoT
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    status: ArtifactStatus = ArtifactStatus.DRAFT

    def add_provenance(
        self,
        agent_role: AgentRole,
        action: str,
        result: Dict[str, Any] | None = None,
    ) -> None:
        self.provenance.append({
            "agent_role": agent_role.value,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "result": result or {},
        })

    def provenance_chain(self) -> List[str]:
        """Return compact provenance chain for display."""
        return [f"{p['agent_role']}::{p['action']}" for p in self.provenance]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "content": self.content,
            "reasoning_trace": self.reasoning_trace,
            "provenance": self.provenance,
            "quality_scores": self.quality_scores,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        status_raw = data.get("status", "draft")
        if isinstance(status_raw, str):
            status = ArtifactStatus(status_raw)
        else:
            status = ArtifactStatus.DRAFT
        return cls(
            artifact_id=data.get("artifact_id", uuid.uuid4().hex[:12]),
            artifact_type=data.get("artifact_type", ""),
            content=data.get("content", {}),
            reasoning_trace=data.get("reasoning_trace", ""),
            provenance=data.get("provenance", []),
            quality_scores=data.get("quality_scores", {}),
            status=status,
        )


# ═══════════════════════════════════════════════════════════════════════
# Agent Card (A2A-inspired capability advertisement)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentCard:
    """Describes an agent's identity, capabilities, and endpoint.

    Inspired by Google's A2A AgentCard (/.well-known/agent.json).
    Enables runtime discovery and heterogeneous agent deployment.

    Attributes:
        agent_id: Unique identifier.
        role: The agent's role in the topology.
        provider: LLM provider name (e.g., "ali", "deepseek", "31313").
        model_name: Specific model identifier (e.g., "qwen3_5_plus", "glm_5").
        capabilities: What this agent can do.
        endpoint: Logical endpoint for message routing.
        metadata: Arbitrary additional info.
    """

    agent_id: str
    role: AgentRole
    provider: str
    model_name: str
    capabilities: List[str] = field(default_factory=list)
    endpoint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "provider": self.provider,
            "model_name": self.model_name,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        role_raw = data.get("role", "generator")
        if isinstance(role_raw, str):
            role = AgentRole(role_raw)
        else:
            role = role_raw
        return cls(
            agent_id=data["agent_id"],
            role=role,
            provider=data.get("provider", ""),
            model_name=data.get("model_name", ""),
            capabilities=data.get("capabilities", []),
            endpoint=data.get("endpoint", ""),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════
# Agent Message (the core communication unit)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """A structured message in the inter-agent communication protocol.

    All agent-to-agent communication uses this envelope.  Each message
    carries full context: who sent it, who it's for, what artifact
    it concerns, the agent's reasoning, and the provenance trail.

    Attributes:
        message_id: Unique message identifier.
        from_role: Sending agent's role.
        to_role: Intended recipient's role.
        message_type: Type of message (from MessageType enum).
        artifact: The artifact this message concerns.
        scores: Quality scores assigned (for verification messages).
        critique: Specific feedback for revision (for revision messages).
        deliberation_round: Current deliberation round (0 = primary scoring).
        deliberation_history: Prior rounds' statements (for multi-turn debate).
        metadata: Arbitrary additional context.
        timestamp: When the message was created.
    """

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_role: AgentRole = AgentRole.ORCHESTRATOR
    to_role: AgentRole = AgentRole.ORCHESTRATOR
    message_type: MessageType = MessageType.GENERATION_REQUEST
    artifact: Artifact | None = None
    scores: Dict[str, float] = field(default_factory=dict)
    critique: str = ""
    deliberation_round: int = 0
    deliberation_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "from_role": self.from_role.value,
            "to_role": self.to_role.value,
            "message_type": self.message_type.value,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "scores": self.scores,
            "critique": self.critique,
            "deliberation_round": self.deliberation_round,
            "deliberation_history": self.deliberation_history,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        from_role_raw = data.get("from_role", "orchestrator")
        to_role_raw = data.get("to_role", "orchestrator")
        msg_type_raw = data.get("message_type", "generation_request")

        return cls(
            message_id=data.get("message_id", uuid.uuid4().hex[:12]),
            from_role=AgentRole(from_role_raw) if isinstance(from_role_raw, str) else from_role_raw,
            to_role=AgentRole(to_role_raw) if isinstance(to_role_raw, str) else to_role_raw,
            message_type=MessageType(msg_type_raw) if isinstance(msg_type_raw, str) else msg_type_raw,
            artifact=Artifact.from_dict(data["artifact"]) if data.get("artifact") else None,
            scores=data.get("scores", {}),
            critique=data.get("critique", ""),
            deliberation_round=data.get("deliberation_round", 0),
            deliberation_history=data.get("deliberation_history", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


# ═══════════════════════════════════════════════════════════════════════
# Deliberation Protocol
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DeliberationConfig:
    """Configuration for the multi-turn deliberation protocol.

    Inspired by Aegean's quorum and stability-window concepts.

    Attributes:
        max_rounds: Maximum deliberation rounds before forced resolution.
        disagreement_threshold: std above which deliberation triggers.
        convergence_threshold: std below which consensus is considered reached.
        stability_window: Number of consecutive rounds consensus must hold.
        quorum_ratio: Minimum fraction of scorers that must agree.
        deliberation_temperature: Temperature for deliberation prompts.
    """

    max_rounds: int = 3
    disagreement_threshold: float = 0.2      # std > this → trigger deliberation
    convergence_threshold: float = 0.1       # std < this → consensus reached
    stability_window: int = 1                 # consecutive stable rounds needed
    quorum_ratio: float = 0.6                 # fraction of scorers needed for quorum
    deliberation_temperature: float = 0.5     # slightly higher temp for debate diversity

    def should_deliberate(self, scores: List[float]) -> bool:
        """Determine if deliberation should be triggered."""
        if len(scores) < 2:
            return False
        std = _compute_std(scores)
        return std > self.disagreement_threshold

    def has_converged(self, scores: List[float]) -> bool:
        """Determine if consensus has been reached."""
        if len(scores) < 2:
            return True
        std = _compute_std(scores)
        return std < self.convergence_threshold


@dataclass
class DeliberationState:
    """Tracks the state of an ongoing deliberation.

    Records each round's scores, statements, and convergence metrics
    for both runtime decisions and post-hoc analysis.
    """

    artifact_id: str
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0
    converged: bool = False
    converged_at_round: int = -1
    final_scores: Dict[str, float] = field(default_factory=dict)
    final_consensus_std: float = 0.0

    def record_round(
        self,
        round_idx: int,
        scores: Dict[str, float],
        statements: Dict[str, str],
        std: float,
    ) -> None:
        self.rounds.append({
            "round": round_idx,
            "scores": dict(scores),
            "statements": dict(statements),
            "std": std,
            "timestamp": datetime.now().isoformat(),
        })
        self.current_round = round_idx

    def mark_converged(self, round_idx: int, scores: Dict[str, float], std: float) -> None:
        self.converged = True
        self.converged_at_round = round_idx
        self.final_scores = dict(scores)
        self.final_consensus_std = std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "rounds": self.rounds,
            "current_round": self.current_round,
            "converged": self.converged,
            "converged_at_round": self.converged_at_round,
            "final_scores": self.final_scores,
            "final_consensus_std": self.final_consensus_std,
        }


# ═══════════════════════════════════════════════════════════════════════
# Agent Harness
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HarnessConfig:
    """Configuration for the multi-agent verification harness.

    Attributes:
        generator_config: LLM config for the Generator agent.
        scorer_configs: LLM configs for scorer agents (keyed by role).
        deliberation: Deliberation protocol parameters.
        max_revision_rounds: Maximum number of revision attempts
                             before forced discard.
        quality_gates: Active verification gates (CSC/EQ/SC).
    """

    generator_config: Dict[str, Any] = field(default_factory=dict)
    scorer_configs: Dict[AgentRole, Dict[str, Any]] = field(default_factory=dict)
    deliberation: DeliberationConfig = field(default_factory=DeliberationConfig)
    max_revision_rounds: int = 3
    quality_gates: Dict[str, bool] = field(default_factory=lambda: {
        "csc": True, "eq": True, "sc": True,
    })
    gate_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "csc": 0.8, "eq": 0.75, "sc": 0.8,
    })

    def is_gate_active(self, gate: str) -> bool:
        return self.quality_gates.get(gate, False)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _compute_std(values: List[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def compute_inter_agent_agreement(
    scorer_scores: Dict[str, List[float]],
) -> Dict[str, float]:
    """Compute inter-agent agreement metrics.

    Args:
        scorer_scores: {scorer_id: [scores across artifacts]}

    Returns:
        Dict with pairwise Spearman r, Krippendorff's alpha (approximation),
        and mean pairwise correlation.
    """
    result: Dict[str, float] = {
        "mean_pairwise_spearman": 0.0,
        "pairwise_spearman_std": 0.0,
        "krippendorff_alpha_approx": 0.0,
    }

    scorer_ids = list(scorer_scores.keys())
    if len(scorer_ids) < 2:
        return result

    # Pairwise Spearman
    pairwise_rs: List[float] = []
    for i in range(len(scorer_ids)):
        for j in range(i + 1, len(scorer_ids)):
            si, sj = scorer_scores[scorer_ids[i]], scorer_scores[scorer_ids[j]]
            if len(si) < 3 or len(sj) < 3:
                continue
            # Simple correlation
            n = min(len(si), len(sj))
            try:
                r = _spearman_r(si[:n], sj[:n])
                pairwise_rs.append(r)
            except Exception:
                continue

    if pairwise_rs:
        result["mean_pairwise_spearman"] = sum(pairwise_rs) / len(pairwise_rs)
        result["pairwise_spearman_std"] = _compute_std(pairwise_rs)

    # Approximate Krippendorff's alpha for interval data
    result["krippendorff_alpha_approx"] = _krippendorff_alpha_approx(scorer_scores)

    return result


def _spearman_r(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation."""
    n = len(x)

    def _rank(vals: List[float]) -> List[float]:
        indexed = sorted(enumerate(vals), key=lambda kv: kv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    den_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    if den_x == 0 or den_y == 0:
        return 0.0
    return num / ((den_x * den_y) ** 0.5)


def _krippendorff_alpha_approx(scorer_scores: Dict[str, List[float]]) -> float:
    """Approximate Krippendorff's alpha for interval metric.

    This is a simplified version for quick inter-agent agreement estimation.
    Full implementation would use the proper Krippendorff formula.
    """
    all_values: List[float] = []
    for scores in scorer_scores.values():
        all_values.extend(scores)

    if len(all_values) < 2:
        return 0.0

    grand_mean = sum(all_values) / len(all_values)
    ss_total = sum((v - grand_mean) ** 2 for v in all_values)

    ss_within = 0.0
    for scores in scorer_scores.values():
        if len(scores) < 2:
            continue
        mean_i = sum(scores) / len(scores)
        ss_within += sum((v - mean_i) ** 2 for v in scores)

    if ss_total == 0:
        return 1.0

    # Approximate: 1 - (within / total), analogous to ICC(1)
    n_raters = len(scorer_scores)
    alpha_approx = 1.0 - (ss_within / ss_total) if n_raters > 1 else 0.0
    return max(-0.1, min(1.0, alpha_approx))  # clamp


# ═══════════════════════════════════════════════════════════════════════
# Default Agent Topology for the Verification Harness
# ═══════════════════════════════════════════════════════════════════════

def build_default_harness_config(
    generator_model: str = "qwen3_5_plus",
    generator_provider: str = "ali",
    scorer_models: Tuple[str, str, str] = ("qwen3_7_max", "kimi_k2_7", "qwen3_7_max"),
    scorer_provider: str = "ali",
    gate_thresholds: Optional[Dict[str, float]] = None,
) -> HarnessConfig:
    """Build a default harness configuration for the verification topology.

    Defaults reflect the calibration-first methodology: heterogeneous
    well-calibrated scorers (Qwen + Kimi per our heterogeneity study). SC
    uses a separate instance to avoid homogeneous echo chambers.

    Returns a HarnessConfig with:
      - One Generator agent (Qwen by default)
      - Heterogeneous scorers per gate (Qwen + Kimi by default)
      - Three verification gates (CSC, EQ, SC) with configurable thresholds
      - Default deliberation parameters
    """
    if gate_thresholds is None:
        gate_thresholds = {"csc": 0.8, "eq": 0.75, "sc": 0.8}

    scorer_config_template = {
        "provider": scorer_provider,
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    return HarnessConfig(
        generator_config={
            "provider": generator_provider,
            "model_name": generator_model,
            "temperature": 0.7,
            "max_tokens": 8000,
        },
        scorer_configs={
            AgentRole.CSC_SCORER: {
                **scorer_config_template,
                "model_name": scorer_models[0],
                "verification_dimension": "relation_credibility",
                "rubric": ["existence", "logic", "confidence"],
            },
            AgentRole.EQ_SCORER: {
                **scorer_config_template,
                "model_name": scorer_models[1],
                "verification_dimension": "hypothesis_coherence",
                "rubric": ["clarity", "economic_logic", "testability"],
            },
            AgentRole.SC_SCORER: {
                **scorer_config_template,
                "model_name": scorer_models[2],  # Separate instance → heterogeneous
                "temperature": 0.35,              # Slightly different temp for diversity
                "verification_dimension": "factor_fidelity",
                "rubric": ["semantic_alignment", "formula_correctness", "rationale_match"],
            },
        },
        deliberation=DeliberationConfig(),
        max_revision_rounds=3,
        quality_gates={"csc": True, "eq": True, "sc": True},
        gate_thresholds=gate_thresholds,
    )


@dataclass
class ScorerCalibration:
    """Result of calibrating a scorer model against human labels."""
    model_name: str
    spearman_r: float
    mae: float
    n_samples: int
    rank: int = 0


class ScorerCalibrator:
    """Calibration-first scorer selection (per paper methodology).

    Evaluates candidate scorer models against human-labeled examples,
    computes Spearman correlation and MAE, and recommends the best
    heterogeneous pair for deployment.

    Usage:
        calibrator = ScorerCalibrator(llm_caller)
        results = calibrator.calibrate(
            scorer_models=["qwen3_7_max", "kimi_k2_7", "glm_5_2", "ds_v4_pro"],
            human_examples=[...],  # list of {"artifact": ..., "human_score": 0.7}
        )
        best_pair = calibrator.recommend_pair(results)
        # → ("qwen3_7_max", "kimi_k2_7")
    """

    def __init__(self, llm_caller: Callable):
        """Args: llm_caller — function(model_name, prompt) → float (0-1 score)"""
        self._caller = llm_caller

    def calibrate(self, scorer_models: List[str],
                  human_examples: List[Dict[str, Any]],
                  prompt_template: str = "Score this artifact (0-1): {artifact}") -> List[ScorerCalibration]:
        """Score human examples with each model, compute alignment metrics."""
        import numpy as np
        from scipy.stats import spearmanr

        human_scores = [ex["human_score"] for ex in human_examples]
        artifacts = [ex["artifact"] for ex in human_examples]
        results = []

        for model in scorer_models:
            model_scores = []
            for artifact in artifacts:
                prompt = prompt_template.format(artifact=artifact)
                score = self._caller(model, prompt)
                model_scores.append(float(score))

            r_val, _ = spearmanr(model_scores, human_scores)
            r = 0.0 if np.isnan(r_val) else float(r_val)
            mae = float(np.mean(np.abs(np.array(model_scores) - np.array(human_scores))))

            results.append(ScorerCalibration(
                model_name=model, spearman_r=round(r, 3),
                mae=round(mae, 4), n_samples=len(human_examples),
            ))

        # Rank by Spearman r (descending)
        ranked = sorted(results, key=lambda x: x.spearman_r, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def recommend_pair(self, calibrations: List[ScorerCalibration],
                       min_r: float = 0.6) -> Tuple[str, str]:
        """Recommend the best heterogeneous pair: top two models with r > min_r."""
        qualified = [c for c in calibrations if c.spearman_r > min_r]
        if len(qualified) < 2:
            top_two = sorted(calibrations, key=lambda x: x.spearman_r, reverse=True)[:2]
        else:
            top_two = sorted(qualified, key=lambda x: x.spearman_r, reverse=True)[:2]
        return (top_two[0].model_name, top_two[1].model_name)
