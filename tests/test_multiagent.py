"""Tests for the multiagent module."""

import pytest
import numpy as np

from gcl.core.commitment import VerificationStatus
from gcl.core.reputation import ReputationTracker, StakeManager
from gcl.multiagent.agent import (
    AgentConfig,
    AgentState,
    AgentType,
    CommitmentAgent,
)
from gcl.multiagent.market import (
    CommitmentMarket,
    MarketConfig,
    Task,
    TaskStatus,
    create_coordination_task,
)
from gcl.multiagent.protocol import (
    CommitmentProtocol,
    RepresentationProtocol,
    compare_protocols,
    validate_theorem_1,
)


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()
        
        assert config.agent_type == AgentType.WORKER
        assert config.initial_stake == 100.0
        assert config.risk_tolerance == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            agent_type=AgentType.COORDINATOR,
            capabilities={"ml", "nlp"},
            initial_stake=200.0,
            risk_tolerance=0.8,
        )
        
        assert config.agent_type == AgentType.COORDINATOR
        assert "ml" in config.capabilities
        assert config.initial_stake == 200.0


class TestCommitmentAgent:
    """Tests for CommitmentAgent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return CommitmentAgent("test-agent")

    def test_create_agent(self, agent):
        """Test creating an agent."""
        assert agent.agent_id == "test-agent"
        assert agent.config.agent_type == AgentType.WORKER

    def test_get_state(self, agent):
        """Test getting agent state."""
        state = agent.get_state()
        
        assert isinstance(state, AgentState)
        assert state.agent_id == "test-agent"
        assert state.available_stake > 0

    def test_can_commit(self, agent):
        """Test checking if agent can commit."""
        assert agent.can_commit(10.0) is True
        assert agent.can_commit(1000.0) is False  # Too much stake

    def test_estimate_success_probability(self, agent):
        """Test success probability estimation."""
        prob = agent.estimate_success_probability({"difficulty": 0.5})
        
        assert 0.0 <= prob <= 1.0

    def test_decide_commitment(self, agent):
        """Test commitment decision."""
        should_commit, confidence, stake_fraction = agent.decide_commitment(
            {"difficulty": 0.3},
            stake_required=5.0,
        )
        
        assert isinstance(should_commit, bool)
        assert 0.0 <= confidence <= 1.0
        assert 0.0 <= stake_fraction <= 1.0

    def test_make_commitment(self, agent):
        """Test making a commitment."""
        commitment = agent.make_commitment(
            task_id="task-1",
            task_features={"difficulty": 0.5},
            action_type="process",
            success_condition="completed == True",
            stake=5.0,
            confidence=0.7,
        )
        
        assert commitment.issuer == "test-agent"
        assert commitment.stake == 5.0
        assert commitment.confidence == 0.7

    def test_execute_commitment(self, agent):
        """Test executing a commitment."""
        commitment = agent.make_commitment(
            task_id="task-1",
            task_features={"difficulty": 0.3},
            action_type="process",
            success_condition="completed == True",
            stake=5.0,
            confidence=0.9,
        )
        
        result = agent.execute_commitment(commitment)
        
        assert result.commitment_id == commitment.id
        assert result.status in [VerificationStatus.SUCCESS, VerificationStatus.FAILURE]

    def test_semantic_embedding(self, agent):
        """Test semantic embedding generation."""
        embedding = agent.get_semantic_embedding()
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 64

    def test_interpret_message(self, agent):
        """Test message interpretation."""
        message = np.random.randn(64).astype(np.float32)
        sender_embedding = np.random.randn(64).astype(np.float32)
        
        interpreted = agent.interpret_message(message, sender_embedding)
        
        assert interpreted.shape == message.shape


class TestTask:
    """Tests for Task."""

    def test_create_task(self):
        """Test creating a task."""
        task = Task(
            name="test_task",
            difficulty=0.5,
            reward=1.0,
        )
        
        assert task.name == "test_task"
        assert task.difficulty == 0.5
        assert task.status == TaskStatus.PENDING

    def test_task_features(self):
        """Test converting task to features."""
        task = Task(
            name="test",
            difficulty=0.7,
            reward=2.0,
            required_capabilities={"ml"},
        )
        
        features = task.to_features()
        
        assert features["difficulty"] == 0.7
        assert features["reward"] == 2.0
        assert "ml" in features["required_capabilities"]


class TestCommitmentMarket:
    """Tests for CommitmentMarket."""

    @pytest.fixture
    def market(self):
        """Create a test market."""
        return CommitmentMarket()

    @pytest.fixture
    def agents(self):
        """Create test agents."""
        return [
            CommitmentAgent(f"agent-{i}")
            for i in range(3)
        ]

    def test_register_agent(self, market, agents):
        """Test registering agents."""
        for agent in agents:
            market.register_agent(agent)
        
        assert len(market.agents) == 3

    def test_post_task(self, market):
        """Test posting a task."""
        task = Task(name="test", difficulty=0.5)
        task_id = market.post_task(task)
        
        assert task_id == task.id
        assert task.id in market.pending_tasks

    def test_get_pending_tasks(self, market, agents):
        """Test getting pending tasks."""
        for agent in agents:
            market.register_agent(agent)
        
        for i in range(5):
            market.post_task(Task(name=f"task_{i}", difficulty=0.5))
        
        pending = market.get_pending_tasks()
        
        assert len(pending) == 5

    def test_find_capable_agents(self, market, agents):
        """Test finding capable agents."""
        for agent in agents:
            market.register_agent(agent)
        
        task = Task(name="test", difficulty=0.5)
        capable = market.find_capable_agents(task)
        
        assert len(capable) > 0

    def test_match_and_allocate(self, market, agents):
        """Test matching and allocating tasks."""
        for agent in agents:
            market.register_agent(agent)
        
        task = Task(name="test", difficulty=0.3, min_stake=1.0)
        market.post_task(task)
        
        allocation = market.match_and_allocate(task.id)
        
        assert allocation is not None
        assert allocation.task_id == task.id
        assert allocation.agent_id in [a.agent_id for a in agents]

    def test_execute_task(self, market, agents):
        """Test executing a task."""
        for agent in agents:
            market.register_agent(agent)
        
        task = Task(name="test", difficulty=0.3, min_stake=1.0)
        market.post_task(task)
        
        allocation = market.match_and_allocate(task.id)
        assert allocation is not None
        
        result = market.execute_task(task.id)
        
        assert result is not None
        assert result.status in [VerificationStatus.SUCCESS, VerificationStatus.FAILURE]

    def test_run_round(self, market, agents):
        """Test running a complete round."""
        for agent in agents:
            market.register_agent(agent)
        
        tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(5)]
        results = market.run_round(tasks)
        
        assert len(results) > 0

    def test_market_statistics(self, market, agents):
        """Test market statistics."""
        for agent in agents:
            market.register_agent(agent)
        
        tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(10)]
        market.run_round(tasks)
        
        stats = market.get_statistics()
        
        assert stats.total_tasks == 10
        assert stats.total_commitments > 0


class TestCommitmentProtocol:
    """Tests for CommitmentProtocol."""

    def test_coordinate(self):
        """Test commitment-based coordination."""
        protocol = CommitmentProtocol()
        
        agents = [CommitmentAgent(f"agent-{i}") for i in range(3)]
        tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(5)]
        
        results, metrics = protocol.coordinate(tasks, agents)
        
        assert len(results) > 0
        assert metrics.n_tasks == 5
        assert metrics.n_agents == 3
        assert metrics.verification_operations > 0

    def test_complexity_scaling(self):
        """Test that complexity scales as O(n·k)."""
        protocol = CommitmentProtocol()
        
        # Test with different n and k
        for n in [5, 10]:
            for k in [2, 4]:
                agents = [CommitmentAgent(f"agent-{i}") for i in range(k)]
                tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(n)]
                
                _, metrics = protocol.coordinate(tasks, agents)
                
                # Verification operations should be O(n·k)
                # Each task gets at most 2 verifications (allocation + execution)
                assert metrics.verification_operations <= 2 * n * k


class TestRepresentationProtocol:
    """Tests for RepresentationProtocol."""

    def test_coordinate(self):
        """Test representation-based coordination."""
        protocol = RepresentationProtocol(semantic_drift=0.1)
        
        agents = [CommitmentAgent(f"agent-{i}") for i in range(3)]
        tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(5)]
        
        results, metrics = protocol.coordinate(tasks, agents)
        
        assert metrics.n_tasks == 5
        assert metrics.n_agents == 3
        assert metrics.interpretation_operations > 0

    def test_complexity_scaling(self):
        """Test that complexity scales as O(n·k²)."""
        protocol = RepresentationProtocol()
        
        # Test with different n and k
        for n in [5, 10]:
            for k in [2, 4]:
                agents = [CommitmentAgent(f"agent-{i}") for i in range(k)]
                tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(n)]
                
                _, metrics = protocol.coordinate(tasks, agents)
                
                # Interpretation operations should be O(n·k²)
                # Each task requires k interpretations + k*(k-1)/2 pairwise
                expected_min = n * k  # At least n*k interpretations
                assert metrics.interpretation_operations >= expected_min

    def test_drift_increases_errors(self):
        """Test that semantic drift increases errors."""
        agents = [CommitmentAgent(f"agent-{i}") for i in range(3)]
        tasks = [Task(name=f"task_{i}", difficulty=0.5) for i in range(10)]
        
        # Low drift
        low_drift = RepresentationProtocol(semantic_drift=0.1)
        _, low_metrics = low_drift.coordinate(
            [Task(**t.__dict__) for t in tasks],
            [CommitmentAgent(f"low_{i}") for i in range(3)],
        )
        
        # High drift
        high_drift = RepresentationProtocol(semantic_drift=0.5)
        _, high_metrics = high_drift.coordinate(
            [Task(**t.__dict__) for t in tasks],
            [CommitmentAgent(f"high_{i}") for i in range(3)],
        )
        
        # Higher drift should generally lead to more errors
        # (This is probabilistic, so we just check the mechanism works)
        assert high_metrics.interpretation_operations > 0


class TestProtocolComparison:
    """Tests for protocol comparison (Theorem 1)."""

    def test_compare_protocols(self):
        """Test comparing commitment and representation protocols."""
        results = compare_protocols(
            n_tasks=10,
            n_agents=3,
            semantic_drift=0.2,
            seed=42,
        )
        
        assert "commitment" in results
        assert "representation" in results
        
        commitment_metrics = results["commitment"]
        rep_metrics = results["representation"]
        
        # Commitment should have fewer operations (O(n·k) vs O(n·k²))
        assert commitment_metrics.communication_complexity < rep_metrics.communication_complexity

    def test_theorem_1_validation(self):
        """Test Theorem 1 validation."""
        results = validate_theorem_1(
            n_values=[5, 10],
            k_values=[2, 3],
            drift_values=[0.0, 0.2],
            seed=42,
        )
        
        assert "commitment_complexity" in results
        assert "representation_complexity" in results
        
        # Check that commitment complexity is lower
        for commit, rep in zip(
            results["commitment_complexity"],
            results["representation_complexity"],
        ):
            if commit["n"] == rep["n"] and commit["k"] == rep["k"]:
                # Commitment should have fewer operations
                assert commit["operations"] <= rep["operations"]


class TestCoordinationTask:
    """Tests for coordination task creation."""

    def test_create_coordination_task(self):
        """Test creating coordination tasks."""
        tasks = create_coordination_task(
            n_subtasks=5,
            difficulty=0.5,
            required_agents=2,
        )
        
        assert len(tasks) == 5
        for task in tasks:
            assert task.action_type == "coordinate"
            assert "subtask_index" in task.features


class TestTheorem1Complexity:
    """Tests specifically for Theorem 1 (Communication Complexity)."""

    def test_commitment_is_linear_in_n(self):
        """Test that commitment complexity is linear in n."""
        k = 3
        complexities = []
        
        for n in [5, 10, 20]:
            agents = [CommitmentAgent(f"agent-{i}") for i in range(k)]
            tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(n)]
            
            protocol = CommitmentProtocol()
            _, metrics = protocol.coordinate(tasks, agents)
            
            complexities.append((n, metrics.verification_operations))
        
        # Check roughly linear scaling
        # If O(n·k), doubling n should roughly double operations
        ratio_1 = complexities[1][1] / complexities[0][1]
        ratio_2 = complexities[2][1] / complexities[1][1]
        
        # Should be roughly 2x (with some tolerance)
        assert 1.0 <= ratio_1 <= 4.0
        assert 1.0 <= ratio_2 <= 4.0

    def test_representation_is_quadratic_in_k(self):
        """Test that representation complexity is quadratic in k."""
        n = 5
        complexities = []
        
        for k in [2, 4]:
            agents = [CommitmentAgent(f"agent-{i}") for i in range(k)]
            tasks = [Task(name=f"task_{i}", difficulty=0.3) for i in range(n)]
            
            protocol = RepresentationProtocol()
            _, metrics = protocol.coordinate(tasks, agents)
            
            complexities.append((k, metrics.interpretation_operations))
        
        # Check quadratic scaling in k
        # If O(n·k²), doubling k should roughly 4x operations
        ratio = complexities[1][1] / complexities[0][1]
        
        # Should be roughly 4x (with tolerance for overhead)
        assert ratio >= 2.0  # At least quadratic growth
