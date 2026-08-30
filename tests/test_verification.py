"""Tests for the verification module."""

import pytest

from gcl.core.commitment import (
    ActionSpec,
    CommitmentPortfolio,
    Consequence,
    ContextRegion,
    FailureMode,
    GroundedCommitment,
    VerificationStatus,
)
from gcl.core.predicates import Predicate
from gcl.core.verification import (
    BatchVerificationResult,
    VerificationContext,
    VerificationEngine,
    VerificationReport,
)


class TestVerificationContext:
    """Tests for VerificationContext."""

    def test_create_context(self):
        """Test creating a verification context."""
        context = VerificationContext(
            pre_state={"counter": 0},
            post_state={"counter": 5},
            action={"type": "increment"},
        )
        assert context.pre_state == {"counter": 0}
        assert context.post_state == {"counter": 5}
        assert context.action == {"type": "increment"}
        assert context.metadata == {}

    def test_context_with_metadata(self):
        """Test context with metadata."""
        context = VerificationContext(
            pre_state={},
            post_state={},
            action={},
            metadata={"request_id": "123"},
        )
        assert context.metadata == {"request_id": "123"}


class TestVerificationEngine:
    """Tests for VerificationEngine."""

    @pytest.fixture
    def engine(self):
        """Create a verification engine."""
        return VerificationEngine()

    @pytest.fixture
    def sample_commitment(self):
        """Create a sample commitment for testing."""
        return GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[
                Predicate(name="always", expression="True")
            ],
            promised_behavior=ActionSpec(action_type="test"),
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
                        description="Partial loss",
                    ),
                    severity=0.3,
                ),
            ],
            stake=1.0,
            confidence=0.85,
            valid_contexts=ContextRegion(description="Test"),
        )

    def test_verify_success(self, engine, sample_commitment):
        """Test successful verification."""
        context = VerificationContext(
            pre_state={"accuracy": 0.5},
            post_state={"accuracy": 0.95, "timed_out": False},
            action={"type": "compute"},
        )
        
        result = engine.verify(sample_commitment, context)
        
        assert result.status == VerificationStatus.SUCCESS
        assert result.commitment_id == sample_commitment.id
        assert result.triggered_failure_mode is None
        assert result.verification_duration_ms is not None

    def test_verify_failure_mode_triggered(self, engine, sample_commitment):
        """Test verification with failure mode triggered."""
        context = VerificationContext(
            pre_state={},
            post_state={"accuracy": 0.5, "timed_out": False},
            action={},
        )
        
        result = engine.verify(sample_commitment, context)
        
        assert result.status == VerificationStatus.FAILURE
        assert result.triggered_failure_mode is not None
        assert result.details.get("failure_mode_name") == "inaccurate"
        assert result.details.get("severity") == 0.5

    def test_verify_timeout_failure(self, engine, sample_commitment):
        """Test verification with timeout failure."""
        context = VerificationContext(
            pre_state={},
            post_state={"accuracy": 0.95, "timed_out": True},
            action={},
        )
        
        result = engine.verify(sample_commitment, context)
        
        assert result.status == VerificationStatus.FAILURE
        assert result.details.get("failure_mode_name") == "timeout"

    def test_verify_success_condition_not_met(self, engine):
        """Test when success condition is not met but no failure mode triggered."""
        commitment = GroundedCommitment(
            issuer="agent-1",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="high", expression="value > 100"),
            failure_modes=[
                FailureMode(
                    name="negative",
                    condition=Predicate(name="neg", expression="value < 0"),
                    consequence=Consequence(
                        consequence_type="error",
                        magnitude=1.0,
                        description="Negative value",
                    ),
                    severity=1.0,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        
        context = VerificationContext(
            pre_state={},
            post_state={"value": 50},  # Not > 100, but not < 0
            action={},
        )
        
        result = engine.verify(commitment, context)
        
        assert result.status == VerificationStatus.FAILURE
        assert "Success condition not met" in result.details.get("reason", "")

    def test_verify_with_custom_verification_fn(self, engine):
        """Test verification with custom verification function."""
        commitment = GroundedCommitment(
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
            verification_fn_name="always_success",
        )
        
        context = VerificationContext(
            pre_state={},
            post_state={},
            action={},
        )
        
        # Need to ensure the registry has the function
        import importlib
        import gcl.core.registry as reg_module
        importlib.reload(reg_module)
        
        result = engine.verify(commitment, context)
        
        assert result.status == VerificationStatus.SUCCESS

    def test_verify_batch(self, engine, sample_commitment):
        """Test batch verification."""
        commitments = [sample_commitment]
        
        # Add another commitment
        commitment2 = GroundedCommitment(
            issuer="agent-2",
            trigger_conditions=[Predicate(name="t", expression="True")],
            promised_behavior=ActionSpec(action_type="test"),
            success_condition=Predicate(name="s", expression="value > 0"),
            failure_modes=[
                FailureMode(
                    name="f",
                    condition=Predicate(name="fc", expression="value <= 0"),
                    consequence=Consequence(
                        consequence_type="test",
                        magnitude=0.5,
                        description="Test",
                    ),
                    severity=0.5,
                )
            ],
            stake=1.0,
            confidence=0.5,
            valid_contexts=ContextRegion(description="Test"),
        )
        commitments.append(commitment2)
        
        context = VerificationContext(
            pre_state={},
            post_state={"accuracy": 0.95, "timed_out": False, "value": 10},
            action={},
        )
        
        batch_result = engine.verify_batch(commitments, context)
        
        assert batch_result.total_count == 2
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 0
        assert len(batch_result.results) == 2

    def test_verify_portfolio(self, engine, sample_commitment):
        """Test portfolio verification."""
        portfolio = CommitmentPortfolio(owner="agent-1")
        portfolio.add_commitment(sample_commitment)
        
        context = VerificationContext(
            pre_state={},
            post_state={"accuracy": 0.95, "timed_out": False},
            action={},
        )
        
        batch_result = engine.verify_portfolio(portfolio, context, only_triggered=False)
        
        assert batch_result.total_count == 1
        assert batch_result.success_count == 1

    def test_strict_context_matching(self):
        """Test strict context matching mode."""
        engine = VerificationEngine(strict_context_matching=True)
        
        commitment = GroundedCommitment(
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
            valid_contexts=ContextRegion(
                description="Test",
                bounds={"temperature": (0, 100)},
            ),
        )
        
        # Context outside bounds
        context = VerificationContext(
            pre_state={},
            post_state={"temperature": 150},
            action={},
        )
        
        result = engine.verify(commitment, context)
        
        assert result.status == VerificationStatus.ERROR
        assert "Context does not match" in result.details.get("reason", "")


class TestBatchVerificationResult:
    """Tests for BatchVerificationResult."""

    def test_success_rate(self):
        """Test success rate calculation."""
        from gcl.core.commitment import VerificationResult
        
        results = [
            VerificationResult(status=VerificationStatus.SUCCESS, commitment_id="1"),
            VerificationResult(status=VerificationStatus.SUCCESS, commitment_id="2"),
            VerificationResult(status=VerificationStatus.FAILURE, commitment_id="3"),
            VerificationResult(status=VerificationStatus.ERROR, commitment_id="4"),
        ]
        
        batch = BatchVerificationResult(
            results=results,
            total_count=4,
            success_count=2,
            failure_count=1,
            error_count=1,
            total_duration_ms=100.0,
        )
        
        assert batch.success_rate == 0.5
        assert batch.all_successful is False

    def test_all_successful(self):
        """Test all_successful property."""
        from gcl.core.commitment import VerificationResult
        
        results = [
            VerificationResult(status=VerificationStatus.SUCCESS, commitment_id="1"),
            VerificationResult(status=VerificationStatus.SUCCESS, commitment_id="2"),
        ]
        
        batch = BatchVerificationResult(
            results=results,
            total_count=2,
            success_count=2,
            failure_count=0,
            error_count=0,
            total_duration_ms=50.0,
        )
        
        assert batch.all_successful is True

    def test_get_failures(self):
        """Test getting failure results."""
        from gcl.core.commitment import VerificationResult
        
        results = [
            VerificationResult(status=VerificationStatus.SUCCESS, commitment_id="1"),
            VerificationResult(status=VerificationStatus.FAILURE, commitment_id="2"),
            VerificationResult(status=VerificationStatus.FAILURE, commitment_id="3"),
        ]
        
        batch = BatchVerificationResult(
            results=results,
            total_count=3,
            success_count=1,
            failure_count=2,
            error_count=0,
            total_duration_ms=75.0,
        )
        
        failures = batch.get_failures()
        assert len(failures) == 2


class TestVerificationReport:
    """Tests for VerificationReport."""

    def test_summary(self):
        """Test report summary."""
        from gcl.core.commitment import VerificationResult
        
        results = [
            VerificationResult(status=VerificationStatus.SUCCESS, commitment_id="1"),
            VerificationResult(status=VerificationStatus.FAILURE, commitment_id="2"),
        ]
        
        batch = BatchVerificationResult(
            results=results,
            total_count=2,
            success_count=1,
            failure_count=1,
            error_count=0,
            total_duration_ms=100.0,
        )
        
        report = VerificationReport(batch)
        summary = report.summary
        
        assert summary["total"] == 2
        assert summary["success"] == 1
        assert summary["failure"] == 1
        assert summary["success_rate"] == 0.5

    def test_failure_analysis(self):
        """Test failure analysis."""
        from gcl.core.commitment import VerificationResult
        
        results = [
            VerificationResult(
                status=VerificationStatus.FAILURE,
                commitment_id="1",
                details={"failure_mode_name": "timeout", "severity": 0.3},
            ),
            VerificationResult(
                status=VerificationStatus.FAILURE,
                commitment_id="2",
                details={"failure_mode_name": "timeout", "severity": 0.3},
            ),
            VerificationResult(
                status=VerificationStatus.FAILURE,
                commitment_id="3",
                details={"failure_mode_name": "inaccurate", "severity": 0.5},
            ),
        ]
        
        batch = BatchVerificationResult(
            results=results,
            total_count=3,
            success_count=0,
            failure_count=3,
            error_count=0,
            total_duration_ms=150.0,
        )
        
        report = VerificationReport(batch)
        analysis = report.get_failure_analysis()
        
        assert analysis["total_failures"] == 3
        assert analysis["failure_modes"]["timeout"] == 2
        assert analysis["failure_modes"]["inaccurate"] == 1
        assert analysis["max_severity"] == 0.5

    def test_to_dict(self):
        """Test converting report to dictionary."""
        from gcl.core.commitment import VerificationResult
        
        results = [
            VerificationResult(status=VerificationStatus.SUCCESS, commitment_id="1"),
        ]
        
        batch = BatchVerificationResult(
            results=results,
            total_count=1,
            success_count=1,
            failure_count=0,
            error_count=0,
            total_duration_ms=50.0,
        )
        
        report = VerificationReport(batch)
        data = report.to_dict()
        
        assert "summary" in data
        assert "failure_analysis" in data
        assert "results" in data
        assert len(data["results"]) == 1
