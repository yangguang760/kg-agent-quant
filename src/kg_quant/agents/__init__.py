"""Agent module for KG-AgentQuant.

Provides the distributed multi-agent verification harness:
  - protocol.py   — Agent communication protocol (A2A-inspired)
  - deliberation.py — Multi-turn deliberative consensus
  - feedback_loop.py — Agentic self-correction loop
  - harness.py    — Agent lifecycle orchestration

Agent Topology:
    Generator Agent (Qwen) → CSC Scorer (GLM+Kimi) → EQ Scorer → SC Scorer
                                  ↕ deliberation       ↕             ↕
                              consensus/fail    consensus/fail  consensus/fail
"""

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
    ScorerCalibration,
    ScorerCalibrator,
    build_default_harness_config,
    compute_inter_agent_agreement,
)
from kg_quant.agents.deliberation import (
    DeliberationEngine,
    DeliberationResult,
    DeliberationSummary,
)
from kg_quant.agents.feedback_loop import (
    FeedbackLoopEngine,
    FeedbackLoopResult,
    FeedbackLoopSummary,
    RevisionRecord,
)
from kg_quant.agents.harness import (
    AgentHarness,
    AgentRegistry,
    MessageRouter,
    QualityGateEnforcer,
)

__all__ = [
    # Protocol
    "AgentRole",
    "MessageType",
    "AgentMessage",
    "AgentCard",
    "Artifact",
    "ArtifactStatus",
    "DeliberationConfig",
    "DeliberationState",
    "HarnessConfig",
    "ScorerCalibration",
    "ScorerCalibrator",
    "build_default_harness_config",
    "compute_inter_agent_agreement",
    # Deliberation
    "DeliberationEngine",
    "DeliberationResult",
    "DeliberationSummary",
    # Feedback Loop
    "FeedbackLoopEngine",
    "FeedbackLoopResult",
    "FeedbackLoopSummary",
    "RevisionRecord",
    # Harness
    "AgentHarness",
    "AgentRegistry",
    "MessageRouter",
    "QualityGateEnforcer",
]
