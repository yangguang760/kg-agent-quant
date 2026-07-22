"""
Distributed Multi-Agent Verification Harness

The central harness that manages agent lifecycle, message routing, gate
enforcement, and provenance tracking for the multi-agent verification
topology.

Architecture:
    ┌──────────────────────────────────────────────┐
    │              Agent Harness                     │
    │                                                │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
    │  │Generator │  │  CSC     │  │  EQ      │... │
    │  │Registry  │  │  Registry│  │  Registry│    │
    │  └──────────┘  └──────────┘  └──────────┘    │
    │                                                │
    │  ┌──────────────────────────────────────┐     │
    │  │        Message Router                 │     │
    │  │  - Route by MessageType               │     │
    │  │  - Enforce stage ordering (CSC→EQ→SC) │     │
    │  │  - Log all messages for traceability  │     │
    │  └──────────────────────────────────────┘     │
    │                                                │
    │  ┌──────────────────────────────────────┐     │
    │  │        Quality Gate Enforcer          │     │
    │  │  - CSC gate: relation credibility     │     │
    │  │  - EQ gate:  hypothesis coherence      │     │
    │  │  - SC gate:  factor fidelity           │     │
    │  └──────────────────────────────────────┘     │
    └──────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock as ThreadLock
from typing import Any, Callable, Dict, List, Optional, Tuple

from kg_quant.agents.protocol import (
    AgentRole,
    MessageType,
    AgentMessage,
    AgentCard,
    Artifact,
    ArtifactStatus,
    DeliberationConfig,
    DeliberationState,
    HarnessConfig,
    compute_inter_agent_agreement,
)
from kg_quant.agents.deliberation import (
    DeliberationEngine,
    DeliberationResult,
    DeliberationSummary,
)


# ────────────────────────────────────────────────────────────────
# Agent Registry
# ────────────────────────────────────────────────────────────────

class AgentRegistry:
    """Registry of available agents with their capabilities.

    Each agent is registered with an AgentCard that advertises its
    role, model backend, and capabilities — A2A-inspired discovery.
    """

    def __init__(self):
        self._agents: Dict[AgentRole, List[AgentCard]] = {}
        self._lock = ThreadLock()

    def register(self, card: AgentCard) -> None:
        with self._lock:
            self._agents.setdefault(card.role, []).append(card)

    def get_agents(self, role: AgentRole) -> List[AgentCard]:
        return list(self._agents.get(role, []))

    def get_scorer_agents(self) -> List[AgentCard]:
        scorers: List[AgentCard] = []
        for role in (AgentRole.CSC_SCORER, AgentRole.EQ_SCORER, AgentRole.SC_SCORER):
            scorers.extend(self.get_agents(role))
        return scorers

    def list_roles(self) -> List[AgentRole]:
        return list(self._agents.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            str(role.value): [card.to_dict() for card in cards]
            for role, cards in self._agents.items()
        }


# ────────────────────────────────────────────────────────────────
# Message Router
# ────────────────────────────────────────────────────────────────

class MessageRouter:
    """Routes agent messages and enforces stage ordering.

    Ensures that artifacts flow through verification stages in the
    correct order: CSC → EQ → SC. Logs all messages for full
    traceability.
    """

    def __init__(self):
        self._message_log: List[AgentMessage] = []
        self._lock = ThreadLock()

    def route(self, message: AgentMessage) -> AgentMessage:
        """Log and validate a message. Returns the message unchanged."""
        with self._lock:
            self._message_log.append(message)
        return message

    def get_trace(self, artifact_id: str) -> List[AgentMessage]:
        """Retrieve the full message trace for an artifact.

        This is the trace-level information that SC-MoA shows is
        more valuable than final-answer voting.
        """
        return [
            msg for msg in self._message_log
            if msg.artifact and msg.artifact.artifact_id == artifact_id
        ]

    @property
    def message_count(self) -> int:
        return len(self._message_log)


# ────────────────────────────────────────────────────────────────
# Quality Gate Enforcer
# ────────────────────────────────────────────────────────────────

class QualityGateEnforcer:
    """Enforces stage-wise quality gates on artifacts.

    Maps to the three verification dimensions:
      - CSC: relation credibility (fused_score >= threshold)
      - EQ:  hypothesis coherence (explanation_score >= threshold)
      - SC:  factor fidelity (semantic_consistency_score >= threshold)

    Each gate can be independently activated/deactivated (for ablation).
    """

    def __init__(self, config: HarnessConfig):
        self.config = config

    def check_csc(
        self, artifact: Artifact, threshold: float = 0.6
    ) -> Tuple[bool, str]:
        if not self.config.is_gate_active("csc"):
            return True, "CSC gate inactive"
        score = artifact.quality_scores.get("csc", 0.0)
        passed = score >= threshold
        reason = f"CSC={score:.3f} {'>=' if passed else '<'} {threshold}"
        return passed, reason

    def check_eq(
        self, artifact: Artifact, threshold: float = 0.6
    ) -> Tuple[bool, str]:
        if not self.config.is_gate_active("eq"):
            return True, "EQ gate inactive"
        score = artifact.quality_scores.get("eq", 0.0)
        passed = score >= threshold
        reason = f"EQ={score:.3f} {'>=' if passed else '<'} {threshold}"
        return passed, reason

    def check_sc(
        self, artifact: Artifact, threshold: float = 0.6
    ) -> Tuple[bool, str]:
        if not self.config.is_gate_active("sc"):
            return True, "SC gate inactive"
        score = artifact.quality_scores.get("sc", 0.0)
        passed = score >= threshold
        reason = f"SC={score:.3f} {'>=' if passed else '<'} {threshold}"
        return passed, reason

    def check_all(
        self,
        artifact: Artifact,
        thresholds: Dict[str, float] | None = None,
    ) -> Tuple[bool, Dict[str, str]]:
        """Run all active quality gates on an artifact."""
        t = thresholds or {"csc": 0.6, "eq": 0.6, "sc": 0.6}
        results: Dict[str, str] = {}

        passed, reason = self.check_csc(artifact, t.get("csc", 0.6))
        results["csc"] = reason
        if not passed:
            return False, results

        passed, reason = self.check_eq(artifact, t.get("eq", 0.6))
        results["eq"] = reason
        if not passed:
            return False, results

        passed, reason = self.check_sc(artifact, t.get("sc", 0.6))
        results["sc"] = reason
        return passed, results


# ────────────────────────────────────────────────────────────────
# Agent Harness
# ────────────────────────────────────────────────────────────────

class AgentHarness:
    """Central harness for the distributed multi-agent verification system.

    Manages:
      - Agent registration and discovery
      - Message routing between agents
      - Quality gate enforcement
      - Deliberation protocol triggering
      - Provenance tracking across the full agent topology
    """

    def __init__(
        self,
        config: HarnessConfig | None = None,
        generator_fn: Callable[..., Any] | None = None,
        scorer_fn: Callable[..., Any] | None = None,
    ):
        self.config = config or HarnessConfig()
        self.registry = AgentRegistry()
        self.router = MessageRouter()
        self.gate_enforcer = QualityGateEnforcer(self.config)
        self.deliberation_engine = DeliberationEngine(
            config=self.config.deliberation,
        )

        # External agent implementations (LLM callers)
        self._generator_fn = generator_fn
        self._scorer_fn = scorer_fn

        # Session tracking
        self.session_id: str = ""
        self.started_at: str = ""
        self.finished_at: str = ""

    # ── Lifecycle ────────────────────────────────────────────────

    def start_session(self) -> None:
        self.session_id = f"harness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.started_at = datetime.now().isoformat()

    def end_session(self) -> None:
        self.finished_at = datetime.now().isoformat()

    # ── Agent Management ─────────────────────────────────────────

    def register_agent(self, card: AgentCard) -> None:
        self.registry.register(card)
        msg = AgentMessage(
            from_role=card.role,
            to_role=AgentRole.ORCHESTRATOR,
            message_type=MessageType.AGENT_REGISTER,
            metadata={"agent_card": card.to_dict()},
        )
        self.router.route(msg)

    def register_default_agents(self) -> None:
        """Register default agents from harness config."""
        # Generator
        gen_cfg = self.config.generator_config
        self.register_agent(AgentCard(
            agent_id="generator_001",
            role=AgentRole.GENERATOR,
            provider=gen_cfg.get("provider", "ali"),
            model_name=gen_cfg.get("model_name", "qwen3_5_plus"),
            capabilities=["entity_generation", "relation_proposal",
                         "hypothesis_generation", "factor_generation"],
            metadata={"temperature": gen_cfg.get("temperature", 0.7)},
        ))

        # Scorers
        for role, cfg in self.config.scorer_configs.items():
            self.register_agent(AgentCard(
                agent_id=f"{role.value}_001",
                role=role,
                provider=cfg.get("provider", "31313"),
                model_name=cfg.get("model_name", "unknown"),
                capabilities=["verification", "scoring", "deliberation"],
                metadata={
                    "temperature": cfg.get("temperature", 0.3),
                    "dimension": cfg.get("verification_dimension", ""),
                    "rubric": cfg.get("rubric", []),
                },
            ))

    # ── Message-based Agent Communication ────────────────────────

    def send_message(self, message: AgentMessage) -> AgentMessage:
        """Send a message through the router.

        The router logs the message and enforces routing rules.
        """
        if message.artifact is not None:
            message.artifact.add_provenance(
                agent_role=message.from_role,
                action=f"send_{message.message_type.value}",
                result={"to": message.to_role.value},
            )
        return self.router.route(message)

    def get_artifact_trace(self, artifact_id: str) -> List[AgentMessage]:
        """Get the full message trace for an artifact.

        This provides complete provenance: which agent did what,
        when, and with what result.
        """
        return self.router.get_trace(artifact_id)

    # ── Verification Pipeline ────────────────────────────────────

    def verify_artifact(
        self,
        artifact: Artifact,
        scorer_role: AgentRole,
        primary_scores: Dict[str, float] | None = None,
    ) -> Tuple[Artifact, DeliberationResult | None]:
        """Route an artifact through a single verification stage.

        If primary scoring shows disagreement and deliberation is
        configured, triggers the multi-turn deliberation protocol.

        Returns updated artifact and deliberation result (if any).
        """
        artifact.status = ArtifactStatus.UNDER_VERIFICATION

        # Request verification
        request_msg = AgentMessage(
            from_role=AgentRole.ORCHESTRATOR,
            to_role=scorer_role,
            message_type=MessageType.VERIFICATION_REQUEST,
            artifact=artifact,
        )
        self.send_message(request_msg)

        # Deliberation result placeholder
        deliberation_result: DeliberationResult | None = None

        if primary_scores is not None:
            score_values = list(primary_scores.values())
            if self.deliberation_engine.should_deliberate(score_values):
                artifact.status = ArtifactStatus.UNDER_DELIBERATION

                # Build artifact meta for deliberation
                meta = {
                    "id": artifact.artifact_id,
                    "head": artifact.content.get("head", ""),
                    "tail": artifact.content.get("tail", ""),
                    "type": artifact.content.get("type", ""),
                    "evidence": artifact.content.get("evidence", ""),
                }

                # Build primary scores in expected format
                scores_for_deliberation: Dict[str, List[Any]] = {}
                for model_name, score in primary_scores.items():
                    scores_for_deliberation[model_name] = [{
                        "confidence_score": score,
                        "existence_score": score,
                        "logic_score": score,
                        "comments": "",
                    }]

                llm_configs = {
                    card.model_name: {
                        "provider": card.provider,
                        "model": card.model_name,
                        "temperature": card.metadata.get("temperature", 0.3),
                        "max_tokens": card.metadata.get("max_tokens", 4000),
                        "weight": 1.0,
                    }
                    for card in self.registry.get_scorer_agents()
                }

                deliberation_result = self.deliberation_engine.deliberate_single(
                    artifact_meta=meta,
                    primary_scores=scores_for_deliberation,
                    relation_idx=0,
                    llm_configs=llm_configs,
                )

                # Update artifact with deliberation results
                artifact.quality_scores[scorer_role.value] = {
                    "primary_scores": deliberation_result.primary_scores,
                    "final_scores": deliberation_result.final_scores,
                    "primary_std": deliberation_result.primary_std,
                    "final_std": deliberation_result.final_std,
                    "deliberation_triggered": deliberation_result.deliberation_triggered,
                    "rounds": deliberation_result.rounds_to_converge,
                }

                if deliberation_result.final_status in ("VERIFIED_HIGH", "VERIFIED_ACCEPTABLE"):
                    artifact.status = ArtifactStatus.VERIFIED
                elif deliberation_result.final_status == "CONTROVERSIAL":
                    artifact.status = ArtifactStatus.REJECTED
                else:
                    artifact.status = ArtifactStatus.REJECTED

        # Send result message
        result_msg = AgentMessage(
            from_role=scorer_role,
            to_role=AgentRole.ORCHESTRATOR,
            message_type=MessageType.VERIFICATION_RESULT,
            artifact=artifact,
            scores=artifact.quality_scores.get(scorer_role.value, {}).get(
                "final_scores", primary_scores or {}
            ),
        )
        self.send_message(result_msg)

        return artifact, deliberation_result

    # ── Session Summary ──────────────────────────────────────────

    def get_session_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "agents_registered": self.registry.list_roles(),
            "messages_routed": self.router.message_count,
            "quality_gates": self.config.quality_gates,
            "deliberation_config": {
                "max_rounds": self.config.deliberation.max_rounds,
                "disagreement_threshold": (
                    self.config.deliberation.disagreement_threshold
                ),
                "convergence_threshold": (
                    self.config.deliberation.convergence_threshold
                ),
            },
        }
