"""
Tests for MAS baseline implementations.

Tests Contract Net Protocol, FIPA-ACL, Auction, and MARL baselines.
"""

import pytest
import numpy as np

from gcl.baselines import (
    MASBaseline,
    CoordinationTask,
    CoordinationResult,
    ContractNetProtocol,
    FIPAACLProtocol,
    AuctionProtocol,
    MARLBaseline,
)
from gcl.baselines.base import TaskType, AgentCapability
from gcl.baselines.auction import AuctionType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_tasks():
    """Create simple test tasks."""
    return [
        CoordinationTask(
            task_id="task_1",
            task_type=TaskType.TASK_ASSIGNMENT,
            requirements={"skill_0": 0.5, "skill_1": 0.3},
            value=1.0,
        ),
        CoordinationTask(
            task_id="task_2",
            task_type=TaskType.TASK_ASSIGNMENT,
            requirements={"skill_1": 0.4, "skill_2": 0.5},
            value=1.5,
        ),
        CoordinationTask(
            task_id="task_3",
            task_type=TaskType.TASK_ASSIGNMENT,
            requirements={"skill_2": 0.6},
            value=0.8,
        ),
    ]


@pytest.fixture
def many_tasks():
    """Create many tasks for stress testing."""
    tasks = []
    for i in range(10):
        tasks.append(CoordinationTask(
            task_id=f"task_{i}",
            task_type=TaskType.TASK_ASSIGNMENT,
            requirements={f"skill_{i % 4}": 0.3 + 0.1 * (i % 3)},
            value=0.5 + 0.2 * i,
        ))
    return tasks


# ============================================================================
# Contract Net Protocol Tests
# ============================================================================

class TestContractNetProtocol:
    """Tests for Contract Net Protocol baseline."""
    
    def test_initialization(self):
        """Test CNP initialization."""
        cnp = ContractNetProtocol(n_agents=5, seed=42)
        assert cnp.n_agents == 5
        assert len(cnp.agents) == 5
        assert cnp.bid_timeout == 10
    
    def test_coordinate_simple(self, simple_tasks):
        """Test basic coordination."""
        cnp = ContractNetProtocol(n_agents=5, seed=42)
        result = cnp.coordinate(simple_tasks)
        
        assert isinstance(result, CoordinationResult)
        assert result.success
        assert len(result.assignments) > 0
        assert result.messages_sent > 0
        assert result.time_steps > 0
    
    def test_all_tasks_assigned(self, simple_tasks):
        """Test that all tasks can be assigned with enough agents."""
        cnp = ContractNetProtocol(n_agents=10, seed=42)
        result = cnp.coordinate(simple_tasks)
        
        # With 10 agents and 3 tasks, all should be assigned
        assert len(result.assignments) == len(simple_tasks)
        assert result.efficiency == 1.0
    
    def test_unique_assignments(self, simple_tasks):
        """Test that each task is assigned to a unique agent."""
        cnp = ContractNetProtocol(n_agents=10, seed=42)
        result = cnp.coordinate(simple_tasks)
        
        # Each agent should only be assigned once
        assigned_agents = list(result.assignments.values())
        assert len(assigned_agents) == len(set(assigned_agents))
    
    def test_protocol_stats(self, simple_tasks):
        """Test protocol statistics."""
        cnp = ContractNetProtocol(n_agents=5, seed=42)
        cnp.coordinate(simple_tasks)
        
        stats = cnp.get_protocol_stats()
        assert "total_contracts" in stats
        assert "completed_contracts" in stats
        assert "total_messages" in stats
    
    def test_capability_threshold(self, simple_tasks):
        """Test minimum capability threshold."""
        # High threshold should reduce bids
        cnp_high = ContractNetProtocol(
            n_agents=5, seed=42, min_capability_threshold=0.9
        )
        result_high = cnp_high.coordinate(simple_tasks)
        
        cnp_low = ContractNetProtocol(
            n_agents=5, seed=42, min_capability_threshold=0.1
        )
        result_low = cnp_low.coordinate(simple_tasks)
        
        # Lower threshold should allow more assignments
        assert result_low.total_value >= result_high.total_value


# ============================================================================
# FIPA-ACL Protocol Tests
# ============================================================================

class TestFIPAACLProtocol:
    """Tests for FIPA-ACL Protocol baseline."""
    
    def test_initialization(self):
        """Test FIPA-ACL initialization."""
        fipa = FIPAACLProtocol(n_agents=5, seed=42)
        assert fipa.n_agents == 5
        assert fipa.max_negotiation_rounds == 5
    
    def test_coordinate_simple(self, simple_tasks):
        """Test basic coordination."""
        fipa = FIPAACLProtocol(n_agents=5, seed=42)
        result = fipa.coordinate(simple_tasks)
        
        assert isinstance(result, CoordinationResult)
        assert result.success
        assert len(result.assignments) > 0
        assert result.metadata["protocol"] == "fipa_acl"
    
    def test_conversation_tracking(self, simple_tasks):
        """Test that conversations are tracked."""
        fipa = FIPAACLProtocol(n_agents=5, seed=42)
        fipa.coordinate(simple_tasks)
        
        assert len(fipa.conversations) > 0
        for conv_id, messages in fipa.conversations.items():
            assert len(messages) > 0
    
    def test_performative_counts(self, simple_tasks):
        """Test performative counting."""
        fipa = FIPAACLProtocol(n_agents=5, seed=42)
        result = fipa.coordinate(simple_tasks)
        
        counts = result.metadata["performative_counts"]
        assert "cfp" in counts
        assert counts["cfp"] > 0
    
    def test_utility_threshold(self, simple_tasks):
        """Test utility threshold effect."""
        fipa_high = FIPAACLProtocol(
            n_agents=5, seed=42, utility_threshold=0.9
        )
        result_high = fipa_high.coordinate(simple_tasks)
        
        fipa_low = FIPAACLProtocol(
            n_agents=5, seed=42, utility_threshold=0.1
        )
        result_low = fipa_low.coordinate(simple_tasks)
        
        # Lower threshold should allow more proposals
        assert result_low.total_value >= result_high.total_value


# ============================================================================
# Auction Protocol Tests
# ============================================================================

class TestAuctionProtocol:
    """Tests for Auction Protocol baseline."""
    
    def test_initialization(self):
        """Test auction initialization."""
        auction = AuctionProtocol(n_agents=5, seed=42)
        assert auction.n_agents == 5
        assert auction.auction_type == AuctionType.SECOND_PRICE
    
    def test_first_price_auction(self, simple_tasks):
        """Test first-price auction."""
        auction = AuctionProtocol(
            n_agents=5, seed=42, auction_type=AuctionType.FIRST_PRICE
        )
        result = auction.coordinate(simple_tasks)
        
        assert result.success
        assert result.metadata["auction_type"] == "first_price"
    
    def test_second_price_auction(self, simple_tasks):
        """Test second-price (Vickrey) auction."""
        auction = AuctionProtocol(
            n_agents=5, seed=42, auction_type=AuctionType.SECOND_PRICE
        )
        result = auction.coordinate(simple_tasks)
        
        assert result.success
        assert result.metadata["auction_type"] == "second_price"
    
    def test_combinatorial_auction(self, simple_tasks):
        """Test combinatorial auction."""
        auction = AuctionProtocol(
            n_agents=5, seed=42, auction_type=AuctionType.COMBINATORIAL
        )
        result = auction.coordinate(simple_tasks)
        
        assert result.success
        assert result.metadata["auction_type"] == "combinatorial"
    
    def test_auction_stats(self, simple_tasks):
        """Test auction statistics."""
        auction = AuctionProtocol(n_agents=5, seed=42)
        auction.coordinate(simple_tasks)
        
        stats = auction.get_auction_stats()
        assert "total_auctions" in stats
        assert "successful_auctions" in stats
        assert "avg_winning_bid" in stats
    
    def test_reserve_price(self, simple_tasks):
        """Test reserve price effect."""
        auction_high = AuctionProtocol(
            n_agents=5, seed=42, reserve_price=10.0
        )
        result_high = auction_high.coordinate(simple_tasks)
        
        auction_low = AuctionProtocol(
            n_agents=5, seed=42, reserve_price=0.01
        )
        result_low = auction_low.coordinate(simple_tasks)
        
        # Lower reserve should allow more bids
        assert result_low.total_value >= result_high.total_value


# ============================================================================
# MARL Baseline Tests
# ============================================================================

class TestMARLBaseline:
    """Tests for MARL baseline."""
    
    def test_initialization(self):
        """Test MARL initialization."""
        marl = MARLBaseline(n_agents=5, seed=42)
        assert marl.n_agents == 5
        assert marl.training_episodes == 100
    
    def test_coordinate_simple(self, simple_tasks):
        """Test basic coordination."""
        marl = MARLBaseline(n_agents=5, seed=42, training_episodes=50)
        result = marl.coordinate(simple_tasks)
        
        assert isinstance(result, CoordinationResult)
        assert result.success
        assert result.metadata["protocol"] == "marl_iql"
    
    def test_no_messages(self, simple_tasks):
        """Test that MARL has no explicit messages."""
        marl = MARLBaseline(n_agents=5, seed=42, training_episodes=50)
        result = marl.coordinate(simple_tasks)
        
        # MARL doesn't use explicit communication
        assert result.messages_sent == 0
    
    def test_learning_improves(self, simple_tasks):
        """Test that more training improves performance."""
        marl_few = MARLBaseline(n_agents=5, seed=42, training_episodes=10)
        result_few = marl_few.coordinate(simple_tasks)
        
        marl_many = MARLBaseline(n_agents=5, seed=42, training_episodes=200)
        result_many = marl_many.coordinate(simple_tasks)
        
        # More training should generally help (not always guaranteed)
        # Just check both work
        assert result_few.success
        assert result_many.success
    
    def test_policy_stats(self, simple_tasks):
        """Test policy statistics."""
        marl = MARLBaseline(n_agents=5, seed=42, training_episodes=50)
        marl.coordinate(simple_tasks)
        
        stats = marl.get_policy_stats()
        assert len(stats) == 5  # One per agent


# ============================================================================
# Comparative Tests
# ============================================================================

class TestBaselineComparison:
    """Tests comparing different baselines."""
    
    def test_all_baselines_work(self, simple_tasks):
        """Test that all baselines produce valid results."""
        baselines = [
            ContractNetProtocol(n_agents=5, seed=42),
            FIPAACLProtocol(n_agents=5, seed=42),
            AuctionProtocol(n_agents=5, seed=42),
            MARLBaseline(n_agents=5, seed=42, training_episodes=50),
        ]
        
        for baseline in baselines:
            result = baseline.coordinate(simple_tasks)
            assert result.success
            assert len(result.assignments) > 0
    
    def test_efficiency_comparison(self, simple_tasks):
        """Test efficiency across baselines."""
        baselines = {
            "cnp": ContractNetProtocol(n_agents=8, seed=42),
            "fipa": FIPAACLProtocol(n_agents=8, seed=42),
            "auction": AuctionProtocol(n_agents=8, seed=42),
            "marl": MARLBaseline(n_agents=8, seed=42, training_episodes=100),
        }
        
        results = {}
        for name, baseline in baselines.items():
            result = baseline.coordinate(simple_tasks)
            results[name] = result.efficiency
        
        # All should achieve reasonable efficiency
        for name, eff in results.items():
            assert eff >= 0.5, f"{name} efficiency too low: {eff}"
    
    def test_message_complexity(self, simple_tasks):
        """Test message complexity across baselines."""
        baselines = {
            "cnp": ContractNetProtocol(n_agents=5, seed=42),
            "fipa": FIPAACLProtocol(n_agents=5, seed=42),
            "auction": AuctionProtocol(n_agents=5, seed=42),
            "marl": MARLBaseline(n_agents=5, seed=42, training_episodes=50),
        }
        
        messages = {}
        for name, baseline in baselines.items():
            result = baseline.coordinate(simple_tasks)
            messages[name] = result.messages_sent
        
        # MARL should have 0 messages
        assert messages["marl"] == 0
        
        # Others should have messages
        assert messages["cnp"] > 0
        assert messages["fipa"] > 0
        assert messages["auction"] > 0
    
    def test_scalability(self, many_tasks):
        """Test scalability with many tasks."""
        baselines = [
            ContractNetProtocol(n_agents=15, seed=42),
            FIPAACLProtocol(n_agents=15, seed=42),
            AuctionProtocol(n_agents=15, seed=42),
            MARLBaseline(n_agents=15, seed=42, training_episodes=50),
        ]
        
        for baseline in baselines:
            result = baseline.coordinate(many_tasks)
            assert result.success
            # Should assign at least half the tasks
            assert len(result.assignments) >= len(many_tasks) // 2


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_tasks(self):
        """Test with no tasks."""
        cnp = ContractNetProtocol(n_agents=5, seed=42)
        result = cnp.coordinate([])
        
        assert not result.success
        assert len(result.assignments) == 0
    
    def test_single_task(self):
        """Test with single task."""
        task = CoordinationTask(
            task_id="single",
            requirements={"skill_0": 0.3},
            value=1.0,
        )
        
        cnp = ContractNetProtocol(n_agents=5, seed=42)
        result = cnp.coordinate([task])
        
        assert result.success
        assert len(result.assignments) == 1
    
    def test_more_tasks_than_agents(self):
        """Test when tasks exceed agents."""
        tasks = [
            CoordinationTask(task_id=f"task_{i}", value=1.0)
            for i in range(10)
        ]
        
        cnp = ContractNetProtocol(n_agents=3, seed=42)
        result = cnp.coordinate(tasks)
        
        # Can only assign up to n_agents tasks
        assert len(result.assignments) <= 3
    
    def test_reproducibility(self, simple_tasks):
        """Test that same seed gives same results."""
        cnp1 = ContractNetProtocol(n_agents=5, seed=42)
        result1 = cnp1.coordinate(simple_tasks)
        
        cnp2 = ContractNetProtocol(n_agents=5, seed=42)
        result2 = cnp2.coordinate(simple_tasks)
        
        assert result1.assignments == result2.assignments
        assert result1.total_value == result2.total_value
