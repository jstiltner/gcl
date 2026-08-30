"""Tests for the reputation module."""

import pytest

from gcl.core.commitment import (
    ActionSpec,
    Consequence,
    ContextRegion,
    FailureMode,
    GroundedCommitment,
    VerificationResult,
    VerificationStatus,
)
from gcl.core.predicates import Predicate
from gcl.core.reputation import (
    AgentReputation,
    ReputationConfig,
    ReputationEvent,
    ReputationEventType,
    ReputationTracker,
    StakeAccount,
    StakeConfig,
    StakeManager,
    StakeTransaction,
)


class TestReputationEvent:
    """Tests for ReputationEvent."""

    def test_create_event(self):
        """Test creating a reputation event."""
        event = ReputationEvent.create(
            agent_id="agent-1",
            event_type=ReputationEventType.COMMITMENT_FULFILLED,
            delta=0.1,
            commitment_id="commit-1",
            details={"reason": "test"},
        )
        
        assert event.agent_id == "agent-1"
        assert event.event_type == ReputationEventType.COMMITMENT_FULFILLED
        assert event.delta == 0.1
        assert event.commitment_id == "commit-1"
        assert event.id is not None
        assert event.timestamp is not None


class TestAgentReputation:
    """Tests for AgentReputation."""

    def test_create_reputation(self):
        """Test creating agent reputation."""
        rep = AgentReputation(agent_id="agent-1")
        
        assert rep.agent_id == "agent-1"
        assert rep.score == 1.0
        assert rep.total_commitments == 0
        assert rep.fulfilled_commitments == 0
        assert rep.failed_commitments == 0

    def test_fulfillment_rate(self):
        """Test fulfillment rate calculation."""
        rep = AgentReputation(
            agent_id="agent-1",
            total_commitments=10,
            fulfilled_commitments=8,
            failed_commitments=2,
        )
        
        assert rep.fulfillment_rate == 0.8

    def test_fulfillment_rate_no_commitments(self):
        """Test fulfillment rate with no commitments."""
        rep = AgentReputation(agent_id="agent-1")
        assert rep.fulfillment_rate == 1.0

    def test_stake_preservation_rate(self):
        """Test stake preservation rate."""
        rep = AgentReputation(
            agent_id="agent-1",
            total_stake_risked=100.0,
            total_stake_lost=20.0,
        )
        
        assert rep.stake_preservation_rate == 0.8

    def test_add_event(self):
        """Test adding reputation event."""
        rep = AgentReputation(agent_id="agent-1", score=1.0)
        
        event = ReputationEvent.create(
            agent_id="agent-1",
            event_type=ReputationEventType.COMMITMENT_FULFILLED,
            delta=0.5,
        )
        
        rep.add_event(event)
        
        assert rep.score == 1.5
        assert len(rep.history) == 1

    def test_score_clamping(self):
        """Test that score is clamped to bounds."""
        rep = AgentReputation(agent_id="agent-1", score=9.5)
        
        # Try to exceed max
        event = ReputationEvent.create(
            agent_id="agent-1",
            event_type=ReputationEventType.BONUS_AWARDED,
            delta=2.0,
        )
        rep.add_event(event)
        assert rep.score == 10.0  # Clamped to max
        
        # Try to go below min
        rep2 = AgentReputation(agent_id="agent-2", score=0.5)
        event2 = ReputationEvent.create(
            agent_id="agent-2",
            event_type=ReputationEventType.COMMITMENT_FAILED,
            delta=-1.0,
        )
        rep2.add_event(event2)
        assert rep2.score == 0.0  # Clamped to min


class TestReputationTracker:
    """Tests for ReputationTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a reputation tracker."""
        return ReputationTracker()

    @pytest.fixture
    def sample_commitment(self):
        """Create a sample commitment."""
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
                        magnitude=0.5,
                        description="Test",
                    ),
                    severity=0.5,
                )
            ],
            stake=1.0,
            confidence=0.8,
            valid_contexts=ContextRegion(description="Test"),
        )

    def test_register_agent(self, tracker):
        """Test registering a new agent."""
        rep = tracker.register_agent("agent-1")
        
        assert rep.agent_id == "agent-1"
        assert rep.score == 1.0

    def test_register_duplicate_raises(self, tracker):
        """Test that registering duplicate agent raises error."""
        tracker.register_agent("agent-1")
        
        with pytest.raises(ValueError, match="already registered"):
            tracker.register_agent("agent-1")

    def test_get_or_create_agent(self, tracker):
        """Test get_or_create_agent."""
        # First call creates
        rep1 = tracker.get_or_create_agent("agent-1")
        assert rep1.agent_id == "agent-1"
        
        # Second call returns existing
        rep2 = tracker.get_or_create_agent("agent-1")
        assert rep1 is rep2

    def test_get_reputation(self, tracker):
        """Test getting reputation."""
        tracker.register_agent("agent-1")
        
        rep = tracker.get_reputation("agent-1")
        assert rep is not None
        assert rep.agent_id == "agent-1"
        
        # Non-existent agent
        assert tracker.get_reputation("unknown") is None

    def test_get_score(self, tracker):
        """Test getting score."""
        tracker.register_agent("agent-1")
        
        assert tracker.get_score("agent-1") == 1.0
        assert tracker.get_score("unknown") == 1.0  # Returns initial

    def test_record_fulfillment_success(self, tracker, sample_commitment):
        """Test recording successful fulfillment."""
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            commitment_id=sample_commitment.id,
        )
        
        event = tracker.record_fulfillment("agent-1", sample_commitment, result)
        
        assert event.event_type == ReputationEventType.COMMITMENT_FULFILLED
        assert event.delta > 0
        
        rep = tracker.get_reputation("agent-1")
        assert rep.fulfilled_commitments == 1
        assert rep.total_commitments == 1
        assert rep.score > 1.0

    def test_record_fulfillment_failure(self, tracker, sample_commitment):
        """Test recording failed fulfillment."""
        result = VerificationResult(
            status=VerificationStatus.FAILURE,
            commitment_id=sample_commitment.id,
            details={"severity": 0.5},
        )
        
        event = tracker.record_fulfillment("agent-1", sample_commitment, result)
        
        assert event.event_type == ReputationEventType.COMMITMENT_FAILED
        assert event.delta < 0
        
        rep = tracker.get_reputation("agent-1")
        assert rep.failed_commitments == 1
        assert rep.score < 1.0

    def test_record_expiration(self, tracker, sample_commitment):
        """Test recording commitment expiration."""
        event = tracker.record_expiration("agent-1", sample_commitment)
        
        assert event.event_type == ReputationEventType.COMMITMENT_EXPIRED
        assert event.delta < 0
        
        rep = tracker.get_reputation("agent-1")
        assert rep.failed_commitments == 1

    def test_apply_decay(self, tracker):
        """Test reputation decay."""
        tracker.register_agent("agent-1")
        rep = tracker.get_reputation("agent-1")
        rep.score = 2.0  # Above initial
        
        new_score = tracker.apply_decay("agent-1")
        
        assert new_score < 2.0  # Decayed towards initial

    def test_get_leaderboard(self, tracker):
        """Test getting leaderboard."""
        tracker.register_agent("agent-1")
        tracker.register_agent("agent-2")
        tracker.register_agent("agent-3")
        
        # Modify scores
        tracker.get_reputation("agent-1").score = 5.0
        tracker.get_reputation("agent-2").score = 3.0
        tracker.get_reputation("agent-3").score = 7.0
        
        leaderboard = tracker.get_leaderboard(limit=2)
        
        assert len(leaderboard) == 2
        assert leaderboard[0] == ("agent-3", 7.0)
        assert leaderboard[1] == ("agent-1", 5.0)

    def test_get_statistics(self, tracker, sample_commitment):
        """Test getting statistics."""
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            commitment_id=sample_commitment.id,
        )
        tracker.record_fulfillment("agent-1", sample_commitment, result)
        
        stats = tracker.get_statistics()
        
        assert stats["total_agents"] == 1
        assert stats["total_commitments"] == 1
        assert stats["overall_fulfillment_rate"] == 1.0


class TestStakeAccount:
    """Tests for StakeAccount."""

    def test_create_account(self):
        """Test creating a stake account."""
        account = StakeAccount(agent_id="agent-1")
        
        assert account.agent_id == "agent-1"
        assert account.available_stake == 0.0
        assert account.locked_stake == 0.0
        assert account.total_stake == 0.0

    def test_total_stake(self):
        """Test total stake calculation."""
        account = StakeAccount(
            agent_id="agent-1",
            available_stake=50.0,
            locked_stake=30.0,
        )
        
        assert account.total_stake == 80.0

    def test_can_stake(self):
        """Test can_stake check."""
        account = StakeAccount(
            agent_id="agent-1",
            available_stake=50.0,
        )
        
        assert account.can_stake(30.0) is True
        assert account.can_stake(50.0) is True
        assert account.can_stake(60.0) is False


class TestStakeManager:
    """Tests for StakeManager."""

    @pytest.fixture
    def manager(self):
        """Create a stake manager."""
        return StakeManager()

    @pytest.fixture
    def sample_commitment(self):
        """Create a sample commitment."""
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
                        magnitude=0.5,
                        description="Test",
                    ),
                    severity=0.5,
                )
            ],
            stake=10.0,
            confidence=0.8,
            valid_contexts=ContextRegion(description="Test"),
        )

    def test_deposit(self, manager):
        """Test depositing stake."""
        tx = manager.deposit("agent-1", 100.0)
        
        assert tx.transaction_type == "deposit"
        assert tx.amount == 100.0
        
        account = manager.get_account("agent-1")
        assert account.available_stake == 100.0

    def test_deposit_invalid_amount(self, manager):
        """Test depositing invalid amount."""
        with pytest.raises(ValueError, match="must be positive"):
            manager.deposit("agent-1", 0)
        
        with pytest.raises(ValueError, match="must be positive"):
            manager.deposit("agent-1", -10)

    def test_deposit_exceeds_max(self, manager):
        """Test deposit exceeding max stake."""
        config = StakeConfig(max_total_stake_per_agent=100.0)
        manager = StakeManager(config=config)
        
        manager.deposit("agent-1", 80.0)
        
        with pytest.raises(ValueError, match="exceed max stake"):
            manager.deposit("agent-1", 30.0)

    def test_withdraw(self, manager):
        """Test withdrawing stake."""
        manager.deposit("agent-1", 100.0)
        tx = manager.withdraw("agent-1", 30.0)
        
        assert tx.transaction_type == "withdrawal"
        assert tx.amount == -30.0
        
        account = manager.get_account("agent-1")
        assert account.available_stake == 70.0

    def test_withdraw_insufficient_funds(self, manager):
        """Test withdrawing more than available."""
        manager.deposit("agent-1", 50.0)
        
        with pytest.raises(ValueError, match="Insufficient"):
            manager.withdraw("agent-1", 60.0)

    def test_lock_stake(self, manager, sample_commitment):
        """Test locking stake for commitment."""
        manager.deposit("agent-1", 100.0)
        tx = manager.lock_stake("agent-1", sample_commitment)
        
        assert tx.transaction_type == "lock"
        assert tx.amount == 10.0
        
        account = manager.get_account("agent-1")
        assert account.available_stake == 90.0
        assert account.locked_stake == 10.0

    def test_lock_stake_insufficient(self, manager, sample_commitment):
        """Test locking stake with insufficient funds."""
        manager.deposit("agent-1", 5.0)
        
        with pytest.raises(ValueError, match="Insufficient"):
            manager.lock_stake("agent-1", sample_commitment)

    def test_lock_stake_duplicate(self, manager, sample_commitment):
        """Test locking stake for same commitment twice."""
        manager.deposit("agent-1", 100.0)
        manager.lock_stake("agent-1", sample_commitment)
        
        with pytest.raises(ValueError, match="already has locked stake"):
            manager.lock_stake("agent-1", sample_commitment)

    def test_unlock_stake(self, manager, sample_commitment):
        """Test unlocking stake on success."""
        manager.deposit("agent-1", 100.0)
        manager.lock_stake("agent-1", sample_commitment)
        
        tx = manager.unlock_stake(sample_commitment)
        
        assert tx is not None
        assert tx.transaction_type == "unlock"
        
        account = manager.get_account("agent-1")
        assert account.available_stake == 100.0
        assert account.locked_stake == 0.0

    def test_unlock_stake_not_locked(self, manager, sample_commitment):
        """Test unlocking stake that wasn't locked."""
        tx = manager.unlock_stake(sample_commitment)
        assert tx is None

    def test_slash_stake(self, manager, sample_commitment):
        """Test slashing stake on failure."""
        manager.deposit("agent-1", 100.0)
        manager.lock_stake("agent-1", sample_commitment)
        
        result = VerificationResult(
            status=VerificationStatus.FAILURE,
            commitment_id=sample_commitment.id,
            details={"severity": 0.5},
        )
        
        tx = manager.slash_stake(sample_commitment, result)
        
        assert tx is not None
        assert tx.transaction_type == "slash"
        assert tx.amount < 0
        
        account = manager.get_account("agent-1")
        assert account.locked_stake == 0.0
        assert account.total_slashed > 0

    def test_award_bonus(self, manager):
        """Test awarding bonus."""
        manager.deposit("agent-1", 100.0)
        tx = manager.award_bonus("agent-1", 10.0, reason="good_work")
        
        assert tx.transaction_type == "bonus"
        assert tx.amount == 10.0
        
        account = manager.get_account("agent-1")
        assert account.available_stake == 110.0
        assert account.total_earned == 10.0

    def test_get_transaction_history(self, manager):
        """Test getting transaction history."""
        manager.deposit("agent-1", 100.0)
        manager.deposit("agent-2", 50.0)
        manager.withdraw("agent-1", 20.0)
        
        # All transactions
        history = manager.get_transaction_history()
        assert len(history) == 3
        
        # Filtered by agent
        history = manager.get_transaction_history(agent_id="agent-1")
        assert len(history) == 2

    def test_get_statistics(self, manager, sample_commitment):
        """Test getting statistics."""
        manager.deposit("agent-1", 100.0)
        manager.deposit("agent-2", 50.0)
        manager.lock_stake("agent-1", sample_commitment)
        
        stats = manager.get_statistics()
        
        assert stats["total_accounts"] == 2
        assert stats["total_stake"] == 150.0
        assert stats["total_locked"] == 10.0
        assert stats["active_commitments"] == 1


class TestReputationConfig:
    """Tests for ReputationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReputationConfig()
        
        assert config.fulfillment_bonus == 0.1
        assert config.failure_penalty == 0.2
        assert config.initial_reputation == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReputationConfig(
            fulfillment_bonus=0.2,
            failure_penalty=0.3,
            initial_reputation=2.0,
        )
        
        assert config.fulfillment_bonus == 0.2
        assert config.failure_penalty == 0.3
        assert config.initial_reputation == 2.0


class TestStakeConfig:
    """Tests for StakeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StakeConfig()
        
        assert config.min_stake == 0.0
        assert config.max_stake == 100.0
        assert config.base_slash_rate == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = StakeConfig(
            max_stake=1000.0,
            base_slash_rate=0.3,
        )
        
        assert config.max_stake == 1000.0
        assert config.base_slash_rate == 0.3
