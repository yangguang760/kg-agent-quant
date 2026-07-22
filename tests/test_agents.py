"""Tests for the multi-agent verification module."""
import pytest
from kg_quant.agents import (
    AgentRole, MessageType, AgentMessage, AgentCard, Artifact, ArtifactStatus,
    DeliberationConfig, DeliberationState, HarnessConfig,
    DeliberationEngine,
    AgentHarness, AgentRegistry, MessageRouter, QualityGateEnforcer,
    build_default_harness_config, compute_inter_agent_agreement,
)


class TestAgentRoles:
    def test_role_enumeration(self):
        roles = list(AgentRole)
        assert AgentRole.GENERATOR in roles
        assert AgentRole.CSC_SCORER in roles
        assert AgentRole.EQ_SCORER in roles
        assert AgentRole.SC_SCORER in roles
        assert AgentRole.ORCHESTRATOR in roles

    def test_is_scorer(self):
        assert AgentRole.CSC_SCORER.is_scorer
        assert AgentRole.EQ_SCORER.is_scorer
        assert AgentRole.SC_SCORER.is_scorer
        assert not AgentRole.GENERATOR.is_scorer
        assert not AgentRole.ORCHESTRATOR.is_scorer


class TestArtifact:
    def test_create_artifact(self):
        a = Artifact(artifact_type="relation", content={"head": "ROE", "tail": "PE"})
        assert a.artifact_type == "relation"
        assert a.status == ArtifactStatus.DRAFT
        assert len(a.artifact_id) == 12

    def test_provenance_tracking(self):
        a = Artifact(artifact_type="relation", content={"head": "ROE", "tail": "PE"})
        a.add_provenance(AgentRole.GENERATOR, "generated", {"confidence": 0.9})
        a.add_provenance(AgentRole.CSC_SCORER, "scored", {"csc": 0.85})
        assert len(a.provenance) == 2
        assert a.provenance_chain() == ["generator::generated", "csc_scorer::scored"]

    def test_lifecycle(self):
        a = Artifact(artifact_type="relation")
        assert a.status == ArtifactStatus.DRAFT
        a.status = ArtifactStatus.VERIFIED
        assert a.status == ArtifactStatus.VERIFIED

    def test_serialization(self):
        a = Artifact(artifact_type="relation", content={"head": "ROE"}, reasoning_trace="test trace")
        a.add_provenance(AgentRole.GENERATOR, "generated")
        d = a.to_dict()
        a2 = Artifact.from_dict(d)
        assert a2.artifact_type == "relation"
        assert a2.reasoning_trace == "test trace"
        assert len(a2.provenance) == 1


class TestAgentMessage:
    def test_create_message(self):
        a = Artifact(artifact_type="relation")
        msg = AgentMessage(
            from_role=AgentRole.GENERATOR, to_role=AgentRole.CSC_SCORER,
            message_type=MessageType.VERIFICATION_REQUEST, artifact=a)
        assert msg.from_role == AgentRole.GENERATOR
        assert msg.to_role == AgentRole.CSC_SCORER
        assert msg.artifact is not None

    def test_message_serialization(self):
        a = Artifact(artifact_type="relation")
        msg = AgentMessage(from_role=AgentRole.GENERATOR, to_role=AgentRole.CSC_SCORER,
                           message_type=MessageType.GENERATION_RESULT, artifact=a,
                           scores={"quality": 0.85})
        d = msg.to_dict()
        msg2 = AgentMessage.from_dict(d)
        assert msg2.message_type == MessageType.GENERATION_RESULT
        assert msg2.scores == {"quality": 0.85}


class TestDeliberationConfig:
    def test_should_deliberate(self):
        dc = DeliberationConfig(disagreement_threshold=0.2)
        assert not dc.should_deliberate([0.7, 0.8])       # std=0.07
        assert dc.should_deliberate([0.3, 0.9])            # std=0.42

    def test_has_converged(self):
        dc = DeliberationConfig(convergence_threshold=0.1)
        assert dc.has_converged([0.75, 0.78])              # std=0.02
        assert not dc.has_converged([0.5, 0.8])            # std=0.21


class TestAgentRegistry:
    def test_register_and_retrieve(self):
        registry = AgentRegistry()
        card = AgentCard(agent_id="test_1", role=AgentRole.CSC_SCORER,
                         provider="openai", model_name="gpt-4")
        registry.register(card)
        agents = registry.get_agents(AgentRole.CSC_SCORER)
        assert len(agents) == 1
        assert agents[0].agent_id == "test_1"

    def test_get_scorer_agents(self):
        registry = AgentRegistry()
        for role in [AgentRole.CSC_SCORER, AgentRole.EQ_SCORER, AgentRole.GENERATOR]:
            registry.register(AgentCard(agent_id=f"{role.value}_1", role=role,
                                        provider="test", model_name="test"))
        scorers = registry.get_scorer_agents()
        assert len(scorers) == 2  # CSC + EQ, not GENERATOR


class TestQualityGateEnforcer:
    def test_gate_check(self):
        config = HarnessConfig(quality_gates={"csc": True, "eq": True, "sc": True})
        enforcer = QualityGateEnforcer(config)
        a = Artifact(artifact_type="relation")
        a.quality_scores = {"csc": 0.85, "eq": 0.80, "sc": 0.90}
        passed, reasons = enforcer.check_all(a)
        assert passed
        assert "csc" in reasons

    def test_gate_inactive(self):
        config = HarnessConfig(quality_gates={"csc": False, "eq": True, "sc": True})
        enforcer = QualityGateEnforcer(config)
        a = Artifact(artifact_type="relation")
        a.quality_scores = {"csc": 0.0, "eq": 0.80, "sc": 0.90}
        passed, _ = enforcer.check_all(a)
        assert passed  # CSC gate inactive, EQ/SC pass

    def test_gate_failure(self):
        config = HarnessConfig(quality_gates={"csc": True, "eq": True, "sc": True})
        enforcer = QualityGateEnforcer(config)
        a = Artifact(artifact_type="relation")
        a.quality_scores = {"csc": 0.5, "eq": 0.80, "sc": 0.90}
        passed, reasons = enforcer.check_all(a)
        assert not passed  # CSC fails


class TestAgentHarness:
    def test_build_default_config(self):
        config = build_default_harness_config()
        assert len(config.scorer_configs) == 3
        assert config.quality_gates == {"csc": True, "eq": True, "sc": True}

    def test_harness_lifecycle(self):
        config = build_default_harness_config()
        harness = AgentHarness(config=config)
        harness.start_session()
        harness.register_default_agents()
        assert len(harness.registry.list_roles()) == 4  # gen + 3 scorers
        summary = harness.get_session_summary()
        assert "session_id" in summary
        harness.end_session()

    def test_message_routing(self):
        config = build_default_harness_config()
        harness = AgentHarness(config=config)
        harness.start_session()
        a = Artifact(artifact_type="relation")
        msg = AgentMessage(from_role=AgentRole.GENERATOR, to_role=AgentRole.CSC_SCORER,
                           message_type=MessageType.GENERATION_RESULT, artifact=a)
        harness.send_message(msg)
        assert harness.router.message_count == 1


class TestInterAgentAgreement:
    def test_compute_agreement(self):
        result = compute_inter_agent_agreement({
            "scorer_a": [0.8, 0.7, 0.9, 0.6, 0.85],
            "scorer_b": [0.75, 0.65, 0.85, 0.55, 0.80],
        })
        assert "mean_pairwise_spearman" in result
        assert "krippendorff_alpha_approx" in result
        assert -0.2 <= result["krippendorff_alpha_approx"] <= 1.0


class TestDeliberationState:
    def test_record_rounds(self):
        state = DeliberationState(artifact_id="test_1")
        state.record_round(0, {"m1": 0.8, "m2": 0.85}, {"m1": "", "m2": ""}, 0.035)
        state.record_round(1, {"m1": 0.78, "m2": 0.82}, {"m1": "a", "m2": "b"}, 0.028)
        assert len(state.rounds) == 2
        assert state.current_round == 1
