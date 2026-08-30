"""Tests for the commitment module."""

from datetime import datetime, timedelta, timezone

import pytest

from gcl.core.commitment import (
    ActionSpec,
    CommitmentPortfolio,
    Consequence,
    ContextRegion,
    FailureMode,
    GroundedCommitment,
    VerificationResult,
    VerificationStatus,
)
from gcl.core.predicates import Predicate


class TestActionSpec:
    """Tests for the ActionSpec class."""

    def test_create_action_spec(self):
        """Test creating an ActionSpec."""
        action = ActionSpec(
            action_type="respond",
            parameters={"format": "json"},
            timeout_seconds=30.0,
            description="Generate a response",
        )
        assert action.action_type == "respond"
        assert action.parameters == {"format": "json"}
        assert action.timeout_seconds == 30.0
        assert action.description == "Generate a response"

    def test_action_spec_defaults(self):
        """Test ActionSpec default values."""
        action = ActionSpec(action_type="test")
        assert action.parameters == {}
        assert action.timeout_seconds is None
        assert action.description is None

    def test_action_spec_immutable(self):
        """Test that ActionSpec is immutable."""
        action = ActionSpec(action_type="test")
        with pytest.raises(Exception):  # Pydantic ValidationError
            action.action_type = "modified"

    def test_action_spec_serialization(self):
        """Test ActionSpec JSON serialization."""
        action = ActionSpec(
            action_type="respond",
            parameters={"key": "value"},
            timeout_seconds=10.0,
        )
        json_str = action.model_dump_json()
        restored = ActionSpec.model_validate_json(json_str)
        assert restored == action

    def test_action_spec_timeout_validation(self):
        """Test that negative timeout is rejected."""
        with pytest.raises(ValueError):
            ActionSpec(action_type="test", timeout_seconds=-1.0)


class TestConsequence:
    """Tests for the Consequence class."""

    def test_create_consequence(self):
        """Test creating a Consequence."""
        consequence = Consequence(
            consequence_type="stake_slash",
            magnitude=0.5,
            description="Lose half stake",
            metadata={"reason": "inaccuracy"},
        )
        assert consequence.consequence_type == "stake_slash"
        assert consequence.magnitude == 0.5
        assert consequence.description == "Lose half stake"
        assert consequence.metadata == {"reason": "inaccuracy"}

    def test_consequence_magnitude_validation(self):
        """Test that magnitude must be between 0 and 1."""
        with pytest.raises(ValueError):
            Consequence(
                consequence_type="test",
                magnitude=1.5,
                description="Invalid",
            )
        with pytest.raises(ValueError):
            Consequence(
                consequence_type="test",
                magnitude=-0.1,
                description="Invalid",
            )

    def test_consequence_immutable(self):
        """Test that Consequence is immutable."""
        consequence = Consequence(
            consequence_type="test",
            magnitude=0.5,
            description="Test",
        )
        with pytest.raises(Exception):
            consequence.magnitude = 0.8


class TestFailureMode:
    """Tests for the FailureMode class."""

    def test_create_failure_mode(self):
        """Test creating a FailureMode."""
        failure = FailureMode(
            name="timeout",
            condition=Predicate(name="timeout", expression="elapsed > limit"),
            consequence=Consequence(
                consequence_type="stake_slash",
                magnitude=0.3,
                description="Partial loss",
            ),
            severity=0.3,
        )
        assert failure.name == "timeout"
        assert failure.severity == 0.3
        assert failure.id is not None  # Auto-generated

    def test_failure_mode_with_remediation(self):
        """Test FailureMode with remediation action."""
        failure = FailureMode(
            name="recoverable_error",
            condition=Predicate(name="error", expression="error_count > 0"),
            consequence=Consequence(
                consequence_type="warning",
                magnitude=0.1,
                description="Minor issue",
            ),
            severity=0.1,
            remediation=ActionSpec(
                action_type="retry",
                parameters={"max_retries": 3},
            ),
        )
        assert failure.remediation is not None
        assert failure.remediation.action_type == "retry"

    def test_failure_mode_is_triggered(self):
        """Test checking if failure mode is triggered."""
        failure = FailureMode(
            name="low_confidence",
            condition=Predicate(name="low_conf", expression="confidence < 0.5"),
            consequence=Consequence(
                consequence_type="escalate",
                magnitude=0.2,
                description="Escalate to human",
            ),
            severity=0.2,
        )
        assert failure.is_triggered({"confidence": 0.3}) is True
        assert failure.is_triggered({"confidence": 0.7}) is False

    def test_failure_mode_severity_validation(self):
        """Test that severity must be between 0 and 1."""
        with pytest.raises(ValueError):
            FailureMode(
                name="invalid",
                condition=Predicate(name="test", expression="True"),
                consequence=Consequence(
                    consequence_type="test",
                    magnitude=0.5,
                    description="Test",
                ),
                severity=1.5,
            )


class TestContextRegion:
    """Tests for the ContextRegion class."""

    def test_create_context_region(self):
        """Test creating a ContextRegion."""
        region = ContextRegion(
            description="QA tasks",
            tags=["qa", "text"],
            bounds={"max_tokens": (0, 4096)},
        )
        assert region.description == "QA tasks"
        assert region.tags == ["qa", "text"]

    def test_context_region_with_embedding(self):
        """Test ContextRegion with embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        region = ContextRegion(
            description="Test",
            embedding=embedding,
        )
        assert region.embedding == embedding

    def test_matches_context_tuple_bounds(self):
        """Test matching context with tuple bounds (min, max)."""
        region = ContextRegion(
            description="Test",
            bounds={"temperature": (0, 100), "tokens": (10, 1000)},
        )
        assert region.matches_context({"temperature": 50, "tokens": 500}) is True
        assert region.matches_context({"temperature": 150, "tokens": 500}) is False
        assert region.matches_context({"temperature": 50, "tokens": 5}) is False

    def test_matches_context_set_bounds(self):
        """Test matching context with set bounds (allowed values)."""
        region = ContextRegion(
            description="Test",
            bounds={"task_type": {"qa", "summarization", "translation"}},
        )
        assert region.matches_context({"task_type": "qa"}) is True
        assert region.matches_context({"task_type": "coding"}) is False

    def test_matches_context_exact_bounds(self):
        """Test matching context with exact value bounds."""
        region = ContextRegion(
            description="Test",
            bounds={"version": "1.0"},
        )
        assert region.matches_context({"version": "1.0"}) is True
        assert region.matches_context({"version": "2.0"}) is False

    def test_matches_context_missing_keys(self):
        """Test that missing keys don't violate bounds."""
        region = ContextRegion(
            description="Test",
            bounds={"required_field": (0, 100)},
        )
        # Missing key doesn't violate bounds
        assert region.matches_context({}) is True


class TestVerificationResult:
    """Tests for the VerificationResult class."""

    def test_create_verification_result(self):
        """Test creating a VerificationResult."""
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            commitment_id="abc-123",
            details={"accuracy": 0.95},
        )
        assert result.status == VerificationStatus.SUCCESS
        assert result.commitment_id == "abc-123"
        assert result.is_success is True
        assert result.is_failure is False

    def test_verification_result_failure(self):
        """Test VerificationResult for failure."""
        result = VerificationResult(
            status=VerificationStatus.FAILURE,
            commitment_id="abc-123",
            triggered_failure_mode="timeout",
        )
        assert result.is_success is False
        assert result.is_failure is True
        assert result.triggered_failure_mode == "timeout"

    def test_verification_result_timestamp(self):
        """Test that timestamp is auto-generated."""
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            commitment_id="test",
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_verification_status_enum(self):
        """Test all verification status values."""
        assert VerificationStatus.PENDING.value == "pending"
        assert VerificationStatus.SUCCESS.value == "success"
        assert VerificationStatus.FAILURE.value == "failure"
        assert VerificationStatus.TIMEOUT.value == "timeout"
        assert VerificationStatus.ERROR.value == "error"


class TestGroundedCommitment:
    """Tests for the GroundedCommitment class."""

    @pytest.fixture
    def sample_commitment(self):
        """Create a sample commitment for testing."""
        return GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[
                Predicate(name="is_question", expression="query_type == 'question'")
            ],
            promised_behavior=ActionSpec(action_type="answer"),
            success_condition=Predicate(name="accurate", expression="accuracy > 0.8"),
            failure_modes=[
                FailureMode(
                    name="inaccurate",
                    condition=Predicate(name="low_acc", expression="accuracy <= 0.8"),
                    consequence=Consequence(
                        consequence_type="stake_slash",
                        magnitude=0.5,
                        description="Lose half stake",
                    ),
                    severity=0.5,
                ),
                FailureMode(
                    name="timeout",
                    condition=Predicate(name="timeout", expression="timed_out == True"),
                    consequence=Consequence(
                        consequence_type="stake_slash",
                        magnitude=0.3,
                        description="Partial loss for timeout",
                    ),
                    severity=0.3,
                ),
            ],
            stake=1.0,
            confidence=0.85,
            valid_contexts=ContextRegion(description="General QA"),
        )

    def test_create_commitment(self, sample_commitment):
        """Test creating a GroundedCommitment."""
        assert sample_commitment.issuer == "agent-1"
        assert sample_commitment.stake == 1.0
        assert sample_commitment.confidence == 0.85
        assert sample_commitment.id is not None

    def test_commitment_auto_id(self):
        """Test that commitment ID is auto-generated."""
        c1 = GroundedCommitment(
            issuer="agent",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        c2 = GroundedCommitment(
            issuer="agent",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        assert c1.id != c2.id

    def test_failure_modes_sorted_by_severity(self, sample_commitment):
        """Test that failure modes are sorted by severity (descending)."""
        severities = [fm.severity for fm in sample_commitment.failure_modes]
        assert severities == sorted(severities, reverse=True)

    def test_triggers_match(self, sample_commitment):
        """Test checking if trigger conditions are met."""
        assert sample_commitment.triggers_match({"query_type": "question"}) is True
        assert sample_commitment.triggers_match({"query_type": "command"}) is False

    def test_is_valid_in_context(self, sample_commitment):
        """Test checking if commitment is valid in context."""
        # With no bounds, any context is valid
        assert sample_commitment.is_valid_in_context({}) is True

    def test_check_failure_modes(self, sample_commitment):
        """Test checking failure modes."""
        # Trigger inaccurate failure
        failure = sample_commitment.check_failure_modes({"accuracy": 0.5})
        assert failure is not None
        assert failure.name == "inaccurate"

        # Trigger timeout failure
        failure = sample_commitment.check_failure_modes(
            {"accuracy": 0.9, "timed_out": True}
        )
        assert failure is not None
        assert failure.name == "timeout"

        # No failure
        failure = sample_commitment.check_failure_modes(
            {"accuracy": 0.9, "timed_out": False}
        )
        assert failure is None

    def test_check_success(self, sample_commitment):
        """Test checking success condition."""
        assert sample_commitment.check_success({"accuracy": 0.9}) is True
        assert sample_commitment.check_success({"accuracy": 0.5}) is False

    def test_is_expired(self, sample_commitment):
        """Test checking if commitment is expired."""
        assert sample_commitment.is_expired() is False

        # Create expired commitment
        expired = GroundedCommitment(
            issuer="agent",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert expired.is_expired() is True

    def test_expiration_validation(self):
        """Test that expires_at must be after created_at."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="expires_at must be after created_at"):
            GroundedCommitment(
                issuer="agent",
                trigger_conditions=[Predicate(name="t", expression="True")],
                promised_behavior=ActionSpec(action_type="test"),
                success_condition=Predicate(name="s", expression="True"),
                failure_modes=[
                    FailureMode(
                        name="f",
                        condition=Predicate(name="fc", expression="False"),
                        consequence=Consequence(
                            consequence_type="test",
                            magnitude=0.1,
                            description="Test",
                        ),
                        severity=0.1,
                    )
                ],
                stake=1.0,
                confidence=0.5,
                valid_contexts=ContextRegion(description="Test"),
                created_at=now,
                expires_at=now - timedelta(hours=1),
            )

    def test_expected_value(self, sample_commitment):
        """Test expected value calculation."""
        ev = sample_commitment.expected_value()
        # confidence * stake - (1 - confidence) * expected_loss
        # 0.85 * 1.0 - 0.15 * (1.0 * 0.4)  # avg severity = 0.4
        assert ev > 0  # Should be positive for high confidence

    def test_stake_validation(self):
        """Test that stake must be non-negative."""
        with pytest.raises(ValueError):
            GroundedCommitment(
                issuer="agent",
                trigger_conditions=[Predicate(name="t", expression="True")],
                promised_behavior=ActionSpec(action_type="test"),
                success_condition=Predicate(name="s", expression="True"),
                failure_modes=[
                    FailureMode(
                        name="f",
                        condition=Predicate(name="fc", expression="False"),
                        consequence=Consequence(
                            consequence_type="test",
                            magnitude=0.1,
                            description="Test",
                        ),
                        severity=0.1,
                    )
                ],
                stake=-1.0,
                confidence=0.5,
                valid_contexts=ContextRegion(description="Test"),
            )

    def test_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            GroundedCommitment(
                issuer="agent",
                trigger_conditions=[Predicate(name="t", expression="True")],
                promised_behavior=ActionSpec(action_type="test"),
                success_condition=Predicate(name="s", expression="True"),
                failure_modes=[
                    FailureMode(
                        name="f",
                        condition=Predicate(name="fc", expression="False"),
                        consequence=Consequence(
                            consequence_type="test",
                            magnitude=0.1,
                            description="Test",
                        ),
                        severity=0.1,
                    )
                ],
                stake=1.0,
                confidence=1.5,
                valid_contexts=ContextRegion(description="Test"),
            )

    def test_trigger_conditions_required(self):
        """Test that at least one trigger condition is required."""
        with pytest.raises(ValueError):
            GroundedCommitment(
                issuer="agent",
                trigger_conditions=[],
                promised_behavior=ActionSpec(action_type="test"),
                success_condition=Predicate(name="s", expression="True"),
                failure_modes=[
                    FailureMode(
                        name="f",
                        condition=Predicate(name="fc", expression="False"),
                        consequence=Consequence(
                            consequence_type="test",
                            magnitude=0.1,
                            description="Test",
                        ),
                        severity=0.1,
                    )
                ],
                stake=1.0,
                confidence=0.5,
                valid_contexts=ContextRegion(description="Test"),
            )

    def test_failure_modes_required(self):
        """Test that at least one failure mode is required."""
        with pytest.raises(ValueError):
            GroundedCommitment(
                issuer="agent",
                trigger_conditions=[Predicate(name="t", expression="True")],
                promised_behavior=ActionSpec(action_type="test"),
                success_condition=Predicate(name="s", expression="True"),
                failure_modes=[],
                stake=1.0,
                confidence=0.5,
                valid_contexts=ContextRegion(description="Test"),
            )

    def test_commitment_serialization(self, sample_commitment):
        """Test commitment JSON serialization."""
        json_str = sample_commitment.model_dump_json()
        restored = GroundedCommitment.model_validate_json(json_str)

        assert restored.id == sample_commitment.id
        assert restored.issuer == sample_commitment.issuer
        assert restored.stake == sample_commitment.stake
        assert restored.confidence == sample_commitment.confidence
        assert len(restored.failure_modes) == len(sample_commitment.failure_modes)

    def test_get_verification_fn(self, sample_commitment):
        """Test getting verification function from registry."""
        # No verification function set
        assert sample_commitment.get_verification_fn() is None

        # Create commitment with verification function
        commitment = GroundedCommitment(
            issuer="agent",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
            verification_fn_name="always_success",
        )
        fn = commitment.get_verification_fn()
        assert fn is not None


class TestCommitmentPortfolio:
    """Tests for the CommitmentPortfolio class."""

    @pytest.fixture
    def sample_commitment(self):
        """Create a sample commitment for testing."""
        return GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.8,
            valid_contexts=ContextRegion(description="Test"),
        )

    def test_create_portfolio(self):
        """Test creating a CommitmentPortfolio."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        assert portfolio.owner == "agent-1"
        assert len(portfolio) == 0
        assert portfolio.total_stake == 0.0

    def test_add_commitment(self, sample_commitment):
        """Test adding a commitment to portfolio."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        portfolio.add_commitment(sample_commitment)

        assert len(portfolio) == 1
        assert portfolio.total_stake == 1.0
        assert sample_commitment.id in portfolio

    def test_remove_commitment(self, sample_commitment):
        """Test removing a commitment from portfolio."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        portfolio.add_commitment(sample_commitment)

        result = portfolio.remove_commitment(sample_commitment.id)
        assert result is True
        assert len(portfolio) == 0

        # Try to remove non-existent
        result = portfolio.remove_commitment("non-existent")
        assert result is False

    def test_get_commitment(self, sample_commitment):
        """Test getting a commitment by ID."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        portfolio.add_commitment(sample_commitment)

        retrieved = portfolio.get_commitment(sample_commitment.id)
        assert retrieved is not None
        assert retrieved.id == sample_commitment.id

        # Non-existent
        assert portfolio.get_commitment("non-existent") is None

    def test_max_stake_enforcement(self, sample_commitment):
        """Test that max_total_stake is enforced."""
        portfolio = CommitmentPortfolio(owner="agent-1", max_total_stake=1.5)
        portfolio.add_commitment(sample_commitment)  # stake=1.0

        # Try to add another commitment that would exceed max
        commitment2 = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )

        with pytest.raises(ValueError, match="exceed max stake"):
            portfolio.add_commitment(commitment2)

    def test_active_commitments(self):
        """Test filtering active (non-expired) commitments."""
        portfolio = CommitmentPortfolio(owner="agent-1")

        # Add active commitment
        active = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        portfolio.add_commitment(active)

        # Add expired commitment
        expired = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        portfolio.add_commitment(expired)

        assert len(portfolio) == 2
        assert len(portfolio.active_commitments) == 1
        assert len(portfolio.expired_commitments) == 1

    def test_prune_expired(self):
        """Test pruning expired commitments."""
        portfolio = CommitmentPortfolio(owner="agent-1")

        # Add active commitment
        active = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        portfolio.add_commitment(active)

        # Add expired commitment
        expired = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        portfolio.add_commitment(expired)

        removed = portfolio.prune_expired()
        assert len(removed) == 1
        assert len(portfolio) == 1

    def test_average_confidence(self, sample_commitment):
        """Test average confidence calculation."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        assert portfolio.average_confidence == 0.0

        portfolio.add_commitment(sample_commitment)  # confidence=0.8
        assert portfolio.average_confidence == 0.8

    def test_expected_portfolio_value(self, sample_commitment):
        """Test expected portfolio value calculation."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        portfolio.add_commitment(sample_commitment)

        epv = portfolio.expected_portfolio_value
        assert epv == sample_commitment.expected_value()

    def test_remaining_stake_budget(self, sample_commitment):
        """Test remaining stake budget calculation."""
        portfolio = CommitmentPortfolio(owner="agent-1", max_total_stake=5.0)
        assert portfolio.remaining_stake_budget() == 5.0

        portfolio.add_commitment(sample_commitment)  # stake=1.0
        assert portfolio.remaining_stake_budget() == 4.0

        # No max set
        portfolio2 = CommitmentPortfolio(owner="agent-1")
        assert portfolio2.remaining_stake_budget() is None

    def test_validate_stake_budget(self, sample_commitment):
        """Test stake budget validation."""
        portfolio = CommitmentPortfolio(owner="agent-1", max_total_stake=2.0)
        portfolio.add_commitment(sample_commitment)  # stake=1.0

        assert portfolio.validate_stake_budget(0.5) is True
        assert portfolio.validate_stake_budget(1.5) is False

    def test_get_commitments_by_issuer(self):
        """Test filtering commitments by issuer."""
        portfolio = CommitmentPortfolio(owner="owner")

        c1 = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        c2 = GroundedCommitment(
            issuer="agent-2",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )

        portfolio.add_commitment(c1)
        portfolio.add_commitment(c2)

        agent1_commitments = portfolio.get_commitments_by_issuer("agent-1")
        assert len(agent1_commitments) == 1
        assert agent1_commitments[0].issuer == "agent-1"

    def test_get_triggered_commitments(self):
        """Test getting commitments whose triggers are met."""
        portfolio = CommitmentPortfolio(owner="owner")

        c1 = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="x > 5")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        c2 = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="x < 5")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="True"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="False"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.1,
                        description="Test",
                    ),
                    severity=0.1,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )

        portfolio.add_commitment(c1)
        portfolio.add_commitment(c2)

        triggered = portfolio.get_triggered_commitments({"x": 10})
        assert len(triggered) == 1

    def test_portfolio_iteration(self, sample_commitment):
        """Test iterating over portfolio commitments."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        portfolio.add_commitment(sample_commitment)

        commitments = list(portfolio)
        assert len(commitments) == 1
        assert commitments[0].id == sample_commitment.id

    def test_portfolio_serialization(self, sample_commitment):
        """Test portfolio JSON serialization."""
        portfolio = CommitmentPortfolio(owner="agent-1", max_total_stake=10.0)
        portfolio.add_commitment(sample_commitment)

        json_str = portfolio.model_dump_json()
        restored = CommitmentPortfolio.model_validate_json(json_str)

        assert restored.owner == portfolio.owner
        assert restored.max_total_stake == portfolio.max_total_stake
        assert len(restored) == len(portfolio)