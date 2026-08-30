"""Tests for the template module."""

import pytest
import numpy as np

from gcl.core.commitment import GroundedCommitment, VerificationResult, VerificationStatus
from gcl.core.predicates import Predicate
from gcl.templates.template import (
    ActionSchema,
    CommitmentTemplate,
    ContextRegionSpec,
    TemplateMetadata,
    TriggerSchema,
    VerificationSchema,
    create_simple_template,
)
from gcl.templates.composition import (
    CompositionType,
    TemplateComposer,
    compute_template_similarity,
)
from gcl.templates.induction import (
    CommitmentExperience,
    InductionConfig,
    TemplateCandidate,
    TemplateInducer,
    TemplateLibrary,
    induce_template_from_commitments,
)


class TestTriggerSchema:
    """Tests for TriggerSchema."""

    def test_create_trigger_schema(self):
        """Test creating a trigger schema."""
        trigger = TriggerSchema(
            name="task_available",
            expression="task_type == 'classification'",
            required_keys=["task_type"],
        )
        
        assert trigger.name == "task_available"
        assert "task_type" in trigger.required_keys

    def test_evaluate_trigger(self):
        """Test evaluating trigger condition."""
        trigger = TriggerSchema(
            name="difficulty_check",
            expression="difficulty < 0.7",
        )
        
        assert trigger.evaluate({"difficulty": 0.5}) is True
        assert trigger.evaluate({"difficulty": 0.8}) is False

    def test_trigger_with_defaults(self):
        """Test trigger with default values."""
        trigger = TriggerSchema(
            name="with_defaults",
            expression="x > threshold",
            defaults={"threshold": 0.5},
        )
        
        assert trigger.evaluate({"x": 0.6}) is True
        assert trigger.evaluate({"x": 0.4}) is False

    def test_trigger_missing_required_key(self):
        """Test trigger returns False for missing required keys."""
        trigger = TriggerSchema(
            name="requires_key",
            expression="x > 0",
            required_keys=["x", "y"],
        )
        
        assert trigger.evaluate({"x": 1}) is False  # Missing y
        assert trigger.evaluate({"x": 1, "y": 2}) is True


class TestActionSchema:
    """Tests for ActionSchema."""

    def test_create_action_schema(self):
        """Test creating an action schema."""
        action = ActionSchema(
            action_type="classify",
            parameter_templates={"input": "{task_input}"},
            static_parameters={"model": "default"},
        )
        
        assert action.action_type == "classify"

    def test_instantiate_action(self):
        """Test instantiating action with state."""
        action = ActionSchema(
            action_type="process",
            parameter_templates={"data": "{input_data}"},
            static_parameters={"format": "json"},
        )
        
        spec = action.instantiate({"input_data": "test_value"})
        
        assert spec.action_type == "process"
        assert spec.parameters["data"] == "test_value"
        assert spec.parameters["format"] == "json"


class TestVerificationSchema:
    """Tests for VerificationSchema."""

    def test_create_verification_schema(self):
        """Test creating a verification schema."""
        verification = VerificationSchema(
            name="accuracy_check",
            success_expression="accuracy > 0.8",
            failure_modes=[
                {
                    "name": "low_accuracy",
                    "condition": "accuracy <= 0.5",
                    "severity": 0.7,
                }
            ],
        )
        
        assert verification.name == "accuracy_check"
        assert len(verification.failure_modes) == 1

    def test_create_success_predicate(self):
        """Test creating success predicate."""
        verification = VerificationSchema(
            name="test",
            success_expression="result == 'success'",
        )
        
        predicate = verification.create_success_predicate()
        
        assert predicate.evaluate({"result": "success"}) is True
        assert predicate.evaluate({"result": "failure"}) is False

    def test_create_failure_modes(self):
        """Test creating failure modes."""
        verification = VerificationSchema(
            name="test",
            success_expression="True",
            failure_modes=[
                {
                    "name": "timeout",
                    "condition": "elapsed > max_time",
                    "severity": 0.5,
                    "magnitude": 0.3,
                },
                {
                    "name": "error",
                    "condition": "error_count > 0",
                    "severity": 0.8,
                    "magnitude": 0.6,
                },
            ],
        )
        
        modes = verification.create_failure_modes()
        
        assert len(modes) == 2
        assert modes[0].name == "timeout"
        assert modes[1].severity == 0.8


class TestContextRegionSpec:
    """Tests for ContextRegionSpec."""

    def test_create_context_region(self):
        """Test creating a context region."""
        region = ContextRegionSpec(
            bounds={"difficulty": (0.0, 0.7)},
            allowed_values={"task_type": {"classification", "regression"}},
            tags={"ml", "supervised"},
        )
        
        assert "difficulty" in region.bounds
        assert "classification" in region.allowed_values["task_type"]

    def test_contains_in_bounds(self):
        """Test checking if state is in bounds."""
        region = ContextRegionSpec(
            bounds={"x": (0.0, 1.0), "y": (-1.0, 1.0)},
        )
        
        assert region.contains({"x": 0.5, "y": 0.0}) is True
        assert region.contains({"x": 1.5, "y": 0.0}) is False
        assert region.contains({"x": 0.5, "y": 2.0}) is False

    def test_contains_allowed_values(self):
        """Test checking allowed values."""
        region = ContextRegionSpec(
            allowed_values={"category": {"A", "B", "C"}},
        )
        
        assert region.contains({"category": "A"}) is True
        assert region.contains({"category": "D"}) is False

    def test_similarity_in_region(self):
        """Test similarity for state in region."""
        region = ContextRegionSpec(
            bounds={"x": (0.0, 1.0)},
        )
        
        sim = region.similarity({"x": 0.5})
        assert sim == 1.0

    def test_similarity_outside_region(self):
        """Test similarity for state outside region."""
        region = ContextRegionSpec(
            bounds={"x": (0.0, 1.0)},
        )
        
        # Just outside
        sim = region.similarity({"x": 1.5})
        assert 0.0 < sim < 1.0
        
        # Far outside
        sim_far = region.similarity({"x": 5.0})
        assert sim_far < sim


class TestCommitmentTemplate:
    """Tests for CommitmentTemplate."""

    @pytest.fixture
    def simple_template(self):
        """Create a simple template for testing."""
        return create_simple_template(
            name="test_template",
            trigger_expression="task_ready == True",
            action_type="process",
            success_expression="result == 'success'",
            bounds={"difficulty": (0.0, 0.8)},
            default_confidence=0.7,
        )

    def test_create_template(self, simple_template):
        """Test creating a template."""
        assert simple_template.metadata.name == "test_template"
        assert simple_template.default_confidence == 0.7

    def test_triggers(self, simple_template):
        """Test trigger evaluation."""
        assert simple_template.triggers({"task_ready": True}) is True
        assert simple_template.triggers({"task_ready": False}) is False

    def test_is_valid_in_context(self, simple_template):
        """Test context validation."""
        assert simple_template.is_valid_in_context({"difficulty": 0.5}) is True
        assert simple_template.is_valid_in_context({"difficulty": 0.9}) is False

    def test_context_similarity(self, simple_template):
        """Test context similarity computation."""
        # In region
        sim_in = simple_template.context_similarity({"difficulty": 0.5})
        assert sim_in == 1.0
        
        # Outside region
        sim_out = simple_template.context_similarity({"difficulty": 1.0})
        assert sim_out < 1.0

    def test_compute_confidence_in_region(self, simple_template):
        """Test confidence computation in validated region."""
        conf = simple_template.compute_confidence({"difficulty": 0.5})
        assert conf == 0.7  # Default confidence

    def test_compute_confidence_outside_region(self, simple_template):
        """Test confidence is scaled outside region (Theorem 5)."""
        conf = simple_template.compute_confidence({"difficulty": 1.0})
        assert conf < 0.7  # Scaled by similarity

    def test_instantiate(self, simple_template):
        """Test template instantiation."""
        commitment = simple_template.instantiate(
            state={"task_ready": True, "difficulty": 0.5},
            issuer="agent-1",
        )
        
        assert isinstance(commitment, GroundedCommitment)
        assert commitment.issuer == "agent-1"
        assert commitment.confidence == 0.7

    def test_instantiate_with_override(self, simple_template):
        """Test instantiation with stake/confidence override."""
        commitment = simple_template.instantiate(
            state={"task_ready": True},
            issuer="agent-1",
            stake=5.0,
            confidence=0.9,
        )
        
        assert commitment.stake == 5.0
        assert commitment.confidence == 0.9

    def test_record_outcome(self, simple_template):
        """Test recording outcomes."""
        simple_template.record_outcome(True)
        simple_template.record_outcome(True)
        simple_template.record_outcome(False)
        
        assert simple_template.metadata.usage_count == 3
        assert simple_template.metadata.success_count == 2
        assert simple_template.metadata.failure_count == 1

    def test_expected_success_rate_in_region(self, simple_template):
        """Test expected success rate in validated region."""
        # Record some outcomes
        for _ in range(8):
            simple_template.record_outcome(True)
        for _ in range(2):
            simple_template.record_outcome(False)
        
        rate = simple_template.expected_success_rate({"difficulty": 0.5})
        assert rate >= 0.7  # At least confidence

    def test_expected_success_rate_outside_region(self, simple_template):
        """Test expected success rate outside region (Theorem 5)."""
        out_state = {"difficulty": 1.0}
        rate = simple_template.expected_success_rate(out_state)
        
        # Should be confidence * similarity (Theorem 5 bound)
        # The confidence is already scaled by similarity in compute_confidence
        confidence = simple_template.compute_confidence(out_state)
        similarity = simple_template.context_similarity(out_state)
        
        # Rate should equal the scaled confidence
        assert rate == pytest.approx(confidence * similarity, rel=0.01)

    def test_serialization(self, simple_template):
        """Test template serialization."""
        data = simple_template.to_dict()
        
        assert "trigger_schema" in data
        assert "action_schema" in data
        assert "metadata" in data
        
        # Deserialize
        restored = CommitmentTemplate.from_dict(data)
        assert restored.metadata.name == simple_template.metadata.name


class TestTemplateComposer:
    """Tests for TemplateComposer."""

    @pytest.fixture
    def template_a(self):
        """First template for composition."""
        return create_simple_template(
            name="fetch",
            trigger_expression="data_needed == True",
            action_type="fetch",
            success_expression="data_fetched == True",
            default_confidence=0.9,
        )

    @pytest.fixture
    def template_b(self):
        """Second template for composition."""
        return create_simple_template(
            name="process",
            trigger_expression="data_ready == True",
            action_type="process",
            success_expression="processed == True",
            default_confidence=0.8,
        )

    def test_sequential_composition(self, template_a, template_b):
        """Test sequential composition (Theorem 4)."""
        composed = TemplateComposer.sequential(template_a, template_b)
        
        assert "fetch" in composed.action_schema.action_type
        assert "process" in composed.action_schema.action_type
        assert "sequential" in composed.metadata.tags
        
        # Confidence should be product (both must succeed)
        assert composed.default_confidence == pytest.approx(0.9 * 0.8)

    def test_parallel_composition(self, template_a, template_b):
        """Test parallel composition (Theorem 4)."""
        composed = TemplateComposer.parallel(template_a, template_b)
        
        assert "parallel" in composed.action_schema.action_type
        assert "parallel" in composed.metadata.tags
        
        # Confidence should be product
        assert composed.default_confidence == pytest.approx(0.9 * 0.8)

    def test_conditional_composition(self, template_a, template_b):
        """Test conditional composition (Theorem 4)."""
        predicate = lambda s: s.get("use_fast", False)
        composed = TemplateComposer.conditional(
            template_a, template_b, predicate, "use_fast"
        )
        
        assert "conditional" in composed.action_schema.action_type
        assert "conditional" in composed.metadata.tags

    def test_repeat_composition(self, template_a):
        """Test repeat composition."""
        repeated = TemplateComposer.repeat(template_a, 3)
        
        assert "x3" in repeated.metadata.name
        # Confidence should be 0.9^3
        assert repeated.default_confidence == pytest.approx(0.9 ** 3)

    def test_fallback_composition(self, template_a, template_b):
        """Test fallback composition."""
        with_fallback = TemplateComposer.fallback(template_a, template_b)
        
        assert "fallback" in with_fallback.action_schema.action_type
        assert "fallback" in with_fallback.metadata.tags
        
        # Confidence: 1 - (1-0.9)(1-0.8) = 0.98
        expected = 1 - (1 - 0.9) * (1 - 0.8)
        assert with_fallback.default_confidence == pytest.approx(expected)

    def test_template_closure(self, template_a, template_b):
        """Test that composed templates are valid templates (Theorem 4)."""
        # Sequential
        seq = TemplateComposer.sequential(template_a, template_b)
        assert isinstance(seq, CommitmentTemplate)
        assert seq.triggers({"data_needed": True})
        
        # Parallel
        par = TemplateComposer.parallel(template_a, template_b)
        assert isinstance(par, CommitmentTemplate)
        
        # Conditional
        cond = TemplateComposer.conditional(
            template_a, template_b, lambda s: True
        )
        assert isinstance(cond, CommitmentTemplate)
        
        # Nested composition
        nested = TemplateComposer.sequential(seq, par)
        assert isinstance(nested, CommitmentTemplate)


class TestTemplateSimilarity:
    """Tests for template similarity computation."""

    def test_identical_templates(self):
        """Test similarity of identical templates."""
        t1 = create_simple_template(
            name="t1",
            trigger_expression="x > 0",
            action_type="process",
            success_expression="result == True",
        )
        t2 = create_simple_template(
            name="t2",
            trigger_expression="x > 0",
            action_type="process",
            success_expression="result == True",
        )
        
        sim = compute_template_similarity(t1, t2)
        assert sim == 1.0

    def test_different_action_types(self):
        """Test similarity with different action types."""
        t1 = create_simple_template(
            name="t1",
            trigger_expression="x > 0",
            action_type="process",
            success_expression="result == True",
        )
        t2 = create_simple_template(
            name="t2",
            trigger_expression="x > 0",
            action_type="analyze",  # Different
            success_expression="result == True",
        )
        
        sim = compute_template_similarity(t1, t2)
        assert sim < 1.0  # Should be less due to different action


class TestTemplateLibrary:
    """Tests for TemplateLibrary."""

    @pytest.fixture
    def library(self):
        """Create a template library."""
        return TemplateLibrary()

    @pytest.fixture
    def sample_template(self):
        """Create a sample template."""
        return create_simple_template(
            name="sample",
            trigger_expression="ready == True",
            action_type="execute",
            success_expression="done == True",
            default_confidence=0.8,
        )

    def test_add_template(self, library, sample_template):
        """Test adding a template."""
        library.add(sample_template)
        
        assert len(library) == 1
        assert library.get(sample_template.metadata.id) is not None

    def test_remove_template(self, library, sample_template):
        """Test removing a template."""
        library.add(sample_template)
        result = library.remove(sample_template.metadata.id)
        
        assert result is True
        assert len(library) == 0

    def test_find_matching(self, library, sample_template):
        """Test finding matching templates."""
        library.add(sample_template)
        
        matches = library.find_matching({"ready": True})
        assert len(matches) == 1
        
        matches = library.find_matching({"ready": False})
        assert len(matches) == 0

    def test_find_best_match(self, library):
        """Test finding best matching template."""
        t1 = create_simple_template(
            name="low_conf",
            trigger_expression="ready == True",
            action_type="execute",
            success_expression="done == True",
            default_confidence=0.5,
        )
        t2 = create_simple_template(
            name="high_conf",
            trigger_expression="ready == True",
            action_type="execute",
            success_expression="done == True",
            default_confidence=0.9,
        )
        
        library.add(t1)
        library.add(t2)
        
        best = library.find_best_match({"ready": True})
        assert best.metadata.name == "high_conf"

    def test_record_outcome(self, library, sample_template):
        """Test recording outcomes."""
        library.add(sample_template)
        
        library.record_outcome(sample_template.metadata.id, True)
        library.record_outcome(sample_template.metadata.id, False)
        
        template = library.get(sample_template.metadata.id)
        assert template.metadata.usage_count == 2

    def test_max_templates_pruning(self):
        """Test that library prunes when at capacity."""
        config = InductionConfig(max_templates=3)
        library = TemplateLibrary(config)
        
        for i in range(5):
            t = create_simple_template(
                name=f"template_{i}",
                trigger_expression="True",
                action_type="test",
                success_expression="True",
                default_confidence=0.1 * (i + 1),
            )
            library.add(t)
        
        assert len(library) == 3
        # Should have kept highest confidence templates
        confidences = [t.default_confidence for t in library]
        assert min(confidences) >= 0.3


class TestTemplateInducer:
    """Tests for TemplateInducer."""

    @pytest.fixture
    def inducer(self):
        """Create a template inducer."""
        config = InductionConfig(min_commitments=3, min_success_rate=0.6)
        return TemplateInducer(config=config)

    def test_observe_experience(self, inducer):
        """Test observing commitment experience."""
        commitment = _create_test_commitment("process", "result == True")
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            commitment_id=commitment.id,
        )
        
        inducer.observe(commitment, result, {"difficulty": 0.5})
        
        assert len(inducer.experiences) == 1
        assert len(inducer.candidates) == 1

    def test_induce_template(self, inducer):
        """Test inducing a template from experience."""
        # Add enough successful experiences
        for i in range(5):
            commitment = _create_test_commitment("process", "result == True")
            result = VerificationResult(
                status=VerificationStatus.SUCCESS,
                commitment_id=commitment.id,
            )
            inducer.observe(commitment, result, {"difficulty": 0.3 + i * 0.1})
        
        induced = inducer.induce()
        
        assert len(induced) == 1
        assert induced[0].metadata.source == "induced"
        assert len(inducer.library) == 1

    def test_induce_requires_min_commitments(self, inducer):
        """Test that induction requires minimum commitments."""
        # Add only 2 experiences (below threshold of 3)
        for _ in range(2):
            commitment = _create_test_commitment("process", "result == True")
            result = VerificationResult(
                status=VerificationStatus.SUCCESS,
                commitment_id=commitment.id,
            )
            inducer.observe(commitment, result, {})
        
        induced = inducer.induce()
        
        assert len(induced) == 0

    def test_induce_requires_min_success_rate(self, inducer):
        """Test that induction requires minimum success rate."""
        # Add experiences with low success rate
        for i in range(5):
            commitment = _create_test_commitment("process", "result == True")
            # Only 1 success out of 5 = 20% < 60% threshold
            status = VerificationStatus.SUCCESS if i == 0 else VerificationStatus.FAILURE
            result = VerificationResult(
                status=status,
                commitment_id=commitment.id,
            )
            inducer.observe(commitment, result, {})
        
        induced = inducer.induce()
        
        assert len(induced) == 0


class TestAnalogicalTransfer:
    """Tests for analogical transfer (Theorem 5)."""

    def test_transfer_bound(self):
        """Test that expected success follows transfer bound."""
        template = create_simple_template(
            name="test",
            trigger_expression="True",
            action_type="process",
            success_expression="True",
            bounds={"x": (0.0, 1.0)},
            default_confidence=0.8,
        )
        
        # In region: full confidence
        in_region_rate = template.expected_success_rate({"x": 0.5})
        assert in_region_rate >= 0.8
        
        # Outside region: confidence * similarity
        out_state = {"x": 2.0}
        similarity = template.context_similarity(out_state)
        expected_bound = 0.8 * similarity
        
        out_region_rate = template.expected_success_rate(out_state)
        assert out_region_rate == pytest.approx(expected_bound, rel=0.01)

    def test_confidence_scaling(self):
        """Test that confidence scales with similarity."""
        template = create_simple_template(
            name="test",
            trigger_expression="True",
            action_type="process",
            success_expression="True",
            bounds={"x": (0.0, 1.0)},
            default_confidence=0.9,
        )
        
        # Progressively further from region
        distances = [0.5, 1.5, 2.5, 5.0]
        confidences = [template.compute_confidence({"x": d}) for d in distances]
        
        # In region should have full confidence
        assert confidences[0] == 0.9
        
        # Outside region should decrease
        for i in range(1, len(confidences)):
            assert confidences[i] < confidences[0]


def _create_test_commitment(action_type: str, success_expr: str) -> GroundedCommitment:
    """Helper to create test commitments."""
    from gcl.core.commitment import ActionSpec, Consequence, ContextRegion, FailureMode
    
    return GroundedCommitment(
        issuer="test-agent",
        trigger_conditions=[Predicate(name="trigger", expression="True")],
        promised_behavior=ActionSpec(action_type=action_type),
        success_condition=Predicate(name="success", expression=success_expr),
        failure_modes=[
            FailureMode(
                name="default",
                condition=Predicate(name="fail", expression="True"),
                consequence=Consequence(
                    consequence_type="penalty",
                    magnitude=0.5,
                    description="Default failure",
                ),
                severity=0.5,
            )
        ],
        stake=1.0,
        confidence=0.7,
        valid_contexts=ContextRegion(description="Test"),
    )
