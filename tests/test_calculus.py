"""
Tests for the Commitment Calculus (Minimal Formal Core).

These tests verify that the minimal calculus is:
1. Well-defined (all operations work)
2. Closed under composition (Theorem 4)
3. Correctly computes outcomes and settlements
"""

import pytest
from gcl.core.calculus import (
    Commitment,
    CommitmentHistory,
    CommitmentOutcome,
    FailureMode,
    Outcome,
    conditional,
    issue,
    parallel,
    sequential,
    settle,
    theorem_closure,
    verify,
)


class TestPrimitives:
    """Test the 5-tuple primitives."""
    
    def test_commitment_creation(self):
        """Test creating a basic commitment."""
        c = Commitment(
            trigger=lambda s: s.get("active", False),
            action=lambda s: "do_something",
            verification=lambda s, s_, a: s_.get("done", False),
            failures=[
                FailureMode(lambda s, s_, a: s_.get("error", False), 0.8, "error"),
            ],
            stake=1.0,
        )
        
        assert c.stake == 1.0
        assert len(c.failures) == 1
        assert c.failures[0].severity == 0.8
    
    def test_failure_mode_severity_validation(self):
        """Test that severity must be in [0, 1]."""
        with pytest.raises(ValueError):
            FailureMode(lambda s, s_, a: True, 1.5, "invalid")
        
        with pytest.raises(ValueError):
            FailureMode(lambda s, s_, a: True, -0.1, "invalid")
    
    def test_stake_validation(self):
        """Test that stake must be non-negative."""
        with pytest.raises(ValueError):
            Commitment(
                trigger=lambda s: True,
                action=lambda s: None,
                verification=lambda s, s_, a: True,
                failures=[],
                stake=-1.0,
            )
    
    def test_failure_modes_sorted_by_severity(self):
        """Test that failure modes are sorted by severity (highest first)."""
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: None,
            verification=lambda s, s_, a: True,
            failures=[
                FailureMode(lambda s, s_, a: True, 0.3, "low"),
                FailureMode(lambda s, s_, a: True, 0.9, "high"),
                FailureMode(lambda s, s_, a: True, 0.5, "medium"),
            ],
            stake=1.0,
        )
        
        assert c.failures[0].severity == 0.9
        assert c.failures[1].severity == 0.5
        assert c.failures[2].severity == 0.3


class TestOperations:
    """Test the 3 operations: issue, verify, settle."""
    
    def test_issue(self):
        """Test issuing a commitment."""
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action",
            verification=lambda s, s_, a: True,
            failures=[],
            stake=1.0,
        )
        
        issued = issue(c, "agent_1")
        
        assert issued.issuer == "agent_1"
        assert issued.id != ""
        assert issued.stake == c.stake
    
    def test_verify_success(self):
        """Test verification with success outcome."""
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action",
            verification=lambda s, s_, a: s_.get("success", False),
            failures=[
                FailureMode(lambda s, s_, a: s_.get("error", False), 0.8, "error"),
            ],
            stake=1.0,
        )
        
        outcome = verify(c, {}, {"success": True}, "action")
        
        assert outcome.is_success
        assert outcome.severity == 0.0
    
    def test_verify_failure(self):
        """Test verification with failure outcome."""
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action",
            verification=lambda s, s_, a: s_.get("success", False),
            failures=[
                FailureMode(lambda s, s_, a: s_.get("error", False), 0.8, "error"),
            ],
            stake=1.0,
        )
        
        outcome = verify(c, {}, {"success": False, "error": True}, "action")
        
        assert outcome.is_failure
        assert outcome.severity == 0.8
        assert outcome.failure_mode.name == "error"
    
    def test_verify_failure_priority(self):
        """Test that higher severity failures are checked first."""
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action",
            verification=lambda s, s_, a: False,
            failures=[
                FailureMode(lambda s, s_, a: True, 0.3, "low"),
                FailureMode(lambda s, s_, a: True, 0.9, "high"),
            ],
            stake=1.0,
        )
        
        outcome = verify(c, {}, {}, "action")
        
        # High severity should be checked first (after sorting)
        assert outcome.severity == 0.9
        assert outcome.failure_mode.name == "high"
    
    def test_settle_success(self):
        """Test settlement on success."""
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action",
            verification=lambda s, s_, a: True,
            failures=[],
            stake=10.0,
        )
        
        outcome = CommitmentOutcome(status=Outcome.SUCCESS)
        reward = settle(c, outcome, success_bonus=0.1)
        
        assert reward == 1.0  # 10.0 * 0.1
    
    def test_settle_failure(self):
        """Test settlement on failure."""
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action",
            verification=lambda s, s_, a: True,
            failures=[],
            stake=10.0,
        )
        
        failure = FailureMode(lambda s, s_, a: True, 0.5, "medium")
        outcome = CommitmentOutcome(
            status=Outcome.PENDING,
            failure_index=0,
            failure_mode=failure,
        )
        reward = settle(c, outcome)
        
        assert reward == -5.0  # -10.0 * 0.5


class TestComposition:
    """Test the 3 composition operators."""
    
    def test_sequential_composition(self):
        """Test sequential composition C₁ ; C₂."""
        c1 = Commitment(
            trigger=lambda s: s.get("start", False),
            action=lambda s: "action1",
            verification=lambda s, s_, a: s_.get("step1_done", False),
            failures=[FailureMode(lambda s, s_, a: False, 0.5, "f1")],
            stake=1.0,
        )
        
        c2 = Commitment(
            trigger=lambda s: s.get("step1_done", False),
            action=lambda s: "action2",
            verification=lambda s, s_, a: s_.get("step2_done", False),
            failures=[FailureMode(lambda s, s_, a: False, 0.3, "f2")],
            stake=2.0,
        )
        
        composed = sequential(c1, c2)
        
        # Trigger from c1
        assert composed.trigger({"start": True})
        assert not composed.trigger({"start": False})
        
        # Stake is sum
        assert composed.stake == 3.0
        
        # Failures combined
        assert len(composed.failures) == 2
    
    def test_parallel_composition(self):
        """Test parallel composition C₁ ‖ C₂."""
        c1 = Commitment(
            trigger=lambda s: s.get("ready1", False),
            action=lambda s: "action1",
            verification=lambda s, s_, a: s_.get("done1", False),
            failures=[],
            stake=1.0,
        )
        
        c2 = Commitment(
            trigger=lambda s: s.get("ready2", False),
            action=lambda s: "action2",
            verification=lambda s, s_, a: s_.get("done2", False),
            failures=[],
            stake=2.0,
        )
        
        composed = parallel(c1, c2)
        
        # Trigger requires both
        assert composed.trigger({"ready1": True, "ready2": True})
        assert not composed.trigger({"ready1": True, "ready2": False})
        
        # Stake is sum
        assert composed.stake == 3.0
        
        # Action returns tuple
        action = composed.action({})
        assert action == ("action1", "action2")
    
    def test_conditional_composition(self):
        """Test conditional composition C₁ ◁p▷ C₂."""
        c1 = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action1",
            verification=lambda s, s_, a: a == "action1",
            failures=[],
            stake=1.0,
        )
        
        c2 = Commitment(
            trigger=lambda s: True,
            action=lambda s: "action2",
            verification=lambda s, s_, a: a == "action2",
            failures=[],
            stake=2.0,
        )
        
        predicate = lambda s: s.get("use_c1", False)
        composed = conditional(c1, c2, predicate)
        
        # Action depends on predicate
        assert composed.action({"use_c1": True}) == "action1"
        assert composed.action({"use_c1": False}) == "action2"
        
        # Stake is max
        assert composed.stake == 2.0
    
    def test_theorem_closure(self):
        """Test Theorem 4: Closure under composition."""
        assert theorem_closure()
        
        # Verify by construction: composed commitments are valid
        c = Commitment(
            trigger=lambda s: True,
            action=lambda s: "a",
            verification=lambda s, s_, a: True,
            failures=[],
            stake=1.0,
        )
        
        # All compositions produce valid commitments
        seq = sequential(c, c)
        par = parallel(c, c)
        cond = conditional(c, c, lambda s: True)
        
        # All have required fields
        for composed in [seq, par, cond]:
            assert callable(composed.trigger)
            assert callable(composed.action)
            assert callable(composed.verification)
            assert isinstance(composed.failures, list)
            assert composed.stake >= 0


class TestConfidence:
    """Test derived confidence from history."""
    
    def test_initial_confidence(self):
        """Test initial confidence is 0.5 (prior)."""
        history = CommitmentHistory()
        assert history.confidence == 0.5
    
    def test_confidence_after_successes(self):
        """Test confidence increases with successes."""
        history = CommitmentHistory()
        
        for _ in range(10):
            history.update(CommitmentOutcome(status=Outcome.SUCCESS))
        
        assert history.confidence > 0.8
    
    def test_confidence_after_failures(self):
        """Test confidence decreases with failures."""
        history = CommitmentHistory()
        
        failure = FailureMode(lambda s, s_, a: True, 0.5, "f")
        for _ in range(10):
            history.update(CommitmentOutcome(
                status=Outcome.PENDING,
                failure_mode=failure,
            ))
        
        assert history.confidence < 0.2
    
    def test_laplace_smoothing(self):
        """Test Laplace smoothing prevents extreme values."""
        history = CommitmentHistory()
        
        # Even with all successes, confidence < 1
        for _ in range(100):
            history.update(CommitmentOutcome(status=Outcome.SUCCESS))
        
        assert history.confidence < 1.0
        assert history.confidence > 0.95


class TestIntegration:
    """Integration tests for the full calculus."""
    
    def test_full_commitment_lifecycle(self):
        """Test complete lifecycle: issue → verify → settle."""
        # Create commitment
        c = Commitment(
            trigger=lambda s: s.get("query", False),
            action=lambda s: f"answer_{s.get('query_id', 0)}",
            verification=lambda s, s_, a: s_.get("correct", False),
            failures=[
                FailureMode(
                    lambda s, s_, a: s_.get("wrong", False),
                    0.7,
                    "wrong_answer"
                ),
                FailureMode(
                    lambda s, s_, a: s_.get("timeout", False),
                    0.3,
                    "timeout"
                ),
            ],
            stake=5.0,
        )
        
        # Issue
        issued = issue(c, "answerer_agent")
        assert issued.issuer == "answerer_agent"
        
        # Execute (simulated)
        pre_state = {"query": True, "query_id": 42}
        action = issued.action(pre_state)
        post_state = {"correct": True}
        
        # Verify
        outcome = verify(issued, pre_state, post_state, action)
        assert outcome.is_success
        
        # Settle
        reward = settle(issued, outcome)
        assert reward == 0.5  # 5.0 * 0.1
    
    def test_composed_commitment_lifecycle(self):
        """Test lifecycle with composed commitments."""
        # Two-step task: retrieve then answer
        retrieve = Commitment(
            trigger=lambda s: s.get("need_info", False),
            action=lambda s: "retrieve",
            verification=lambda s, s_, a: s_.get("info_found", False),
            failures=[
                FailureMode(lambda s, s_, a: s_.get("not_found", False), 0.5, "not_found"),
            ],
            stake=2.0,
        )
        
        answer = Commitment(
            trigger=lambda s: s.get("info_found", False),
            action=lambda s: "answer",
            verification=lambda s, s_, a: s_.get("answered", False),
            failures=[
                FailureMode(lambda s, s_, a: s_.get("wrong", False), 0.8, "wrong"),
            ],
            stake=3.0,
        )
        
        # Compose sequentially
        task = sequential(retrieve, answer)
        
        # Issue
        issued = issue(task, "rag_agent")
        
        # Execute
        pre_state = {"need_info": True}
        action = issued.action(pre_state)
        post_state = {"info_found": True, "answered": True}
        
        # Verify
        outcome = verify(issued, pre_state, post_state, action)
        assert outcome.is_success
        
        # Settle
        reward = settle(issued, outcome)
        assert reward == 0.5  # 5.0 * 0.1
