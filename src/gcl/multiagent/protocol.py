"""
Communication protocols for multi-agent coordination.

This module implements two coordination protocols:
1. CommitmentProtocol: O(n·k) communication complexity (Theorem 1a)
2. RepresentationProtocol: O(n·k²) communication complexity (Theorem 1b)

The comparison validates Theorem 1 (Communication Complexity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from gcl.core.commitment import VerificationResult, VerificationStatus
from gcl.multiagent.agent import CommitmentAgent
from gcl.multiagent.market import CommitmentMarket, Task, TaskAllocation


@dataclass
class ProtocolMetrics:
    """
    Metrics for comparing communication protocols.
    
    Used to validate Theorem 1:
    - Commitment: O(n·k) verifications
    - Representation: O(n·k²) interpretations
    """
    
    # Task metrics
    n_tasks: int = 0
    n_agents: int = 0
    
    # Communication operations
    verification_operations: int = 0  # For commitment-based
    interpretation_operations: int = 0  # For representation-based
    
    # Errors
    interpretation_errors: int = 0
    verification_errors: int = 0
    
    # Outcomes
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    
    # Timing
    total_time_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Coordination success rate."""
        total = self.successful_coordinations + self.failed_coordinations
        return self.successful_coordinations / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Communication error rate."""
        total_ops = self.verification_operations + self.interpretation_operations
        total_errors = self.verification_errors + self.interpretation_errors
        return total_errors / total_ops if total_ops > 0 else 0.0
    
    @property
    def communication_complexity(self) -> int:
        """Total communication operations."""
        return self.verification_operations + self.interpretation_operations
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_tasks": self.n_tasks,
            "n_agents": self.n_agents,
            "verification_operations": self.verification_operations,
            "interpretation_operations": self.interpretation_operations,
            "interpretation_errors": self.interpretation_errors,
            "verification_errors": self.verification_errors,
            "successful_coordinations": self.successful_coordinations,
            "failed_coordinations": self.failed_coordinations,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "communication_complexity": self.communication_complexity,
            "total_time_seconds": self.total_time_seconds,
        }


class CommunicationProtocol(ABC):
    """
    Abstract base class for communication protocols.
    
    Protocols define how agents coordinate to complete tasks.
    """
    
    @abstractmethod
    def coordinate(
        self,
        tasks: list[Task],
        agents: list[CommitmentAgent],
    ) -> tuple[list[VerificationResult], ProtocolMetrics]:
        """
        Coordinate agents to complete tasks.
        
        Args:
            tasks: Tasks to complete.
            agents: Available agents.
            
        Returns:
            Tuple of (results, metrics).
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get protocol name."""
        pass


class CommitmentProtocol(CommunicationProtocol):
    """
    Commitment-based coordination protocol.
    
    Communication complexity: O(n·k) where:
    - n = number of tasks
    - k = number of agents
    
    Each task requires at most k commitments (one per agent involved).
    Each commitment requires one verification.
    
    From Theorem 1a: Total operations = O(n·k)
    """
    
    def __init__(self, market: CommitmentMarket | None = None) -> None:
        """
        Initialize the protocol.
        
        Args:
            market: Commitment market to use.
        """
        self.market = market or CommitmentMarket()
    
    def get_name(self) -> str:
        return "commitment"
    
    def coordinate(
        self,
        tasks: list[Task],
        agents: list[CommitmentAgent],
    ) -> tuple[list[VerificationResult], ProtocolMetrics]:
        """
        Coordinate using commitments.
        
        Each task is allocated to one agent who makes a commitment.
        Verification is O(1) per commitment.
        Total: O(n·k) operations.
        """
        start_time = datetime.now(timezone.utc)
        metrics = ProtocolMetrics(n_tasks=len(tasks), n_agents=len(agents))
        
        # Register agents
        for agent in agents:
            self.market.register_agent(agent)
        
        # Post tasks
        for task in tasks:
            self.market.post_task(task)
        
        # Allocate and execute
        results = []
        for task in tasks:
            # Find and allocate to best agent
            allocation = self.market.match_and_allocate(task.id)
            
            if allocation:
                # One verification operation per commitment
                metrics.verification_operations += 1
                
                # Execute task
                result = self.market.execute_task(task.id)
                
                if result:
                    results.append(result)
                    # One more verification for result
                    metrics.verification_operations += 1
                    
                    if result.status == VerificationStatus.SUCCESS:
                        metrics.successful_coordinations += 1
                    else:
                        metrics.failed_coordinations += 1
                else:
                    metrics.failed_coordinations += 1
            else:
                metrics.failed_coordinations += 1
        
        # Calculate time
        end_time = datetime.now(timezone.utc)
        metrics.total_time_seconds = (end_time - start_time).total_seconds()
        
        return results, metrics


class RepresentationProtocol(CommunicationProtocol):
    """
    Representation-based coordination protocol.
    
    Communication complexity: O(n·k²) where:
    - n = number of tasks
    - k = number of agents
    
    Each task requires communication among k agents.
    Each pair must interpret each other's messages: C(k,2) = O(k²)
    
    From Theorem 1b: Total operations = O(n·k²)
    
    Error rate scales with semantic drift (Lemma 2):
    E[error] >= ε_AB (semantic drift between agents)
    """
    
    def __init__(
        self,
        semantic_drift: float = 0.0,
        message_dim: int = 64,
    ) -> None:
        """
        Initialize the protocol.
        
        Args:
            semantic_drift: Base semantic drift between agents.
            message_dim: Dimension of message embeddings.
        """
        self.semantic_drift = semantic_drift
        self.message_dim = message_dim
        self.rng = np.random.default_rng()
    
    def get_name(self) -> str:
        return "representation"
    
    def coordinate(
        self,
        tasks: list[Task],
        agents: list[CommitmentAgent],
    ) -> tuple[list[VerificationResult], ProtocolMetrics]:
        """
        Coordinate using representation-based communication.
        
        Each task requires all agents to communicate.
        Each pair interprets each other's messages.
        Total: O(n·k²) operations.
        """
        start_time = datetime.now(timezone.utc)
        metrics = ProtocolMetrics(n_tasks=len(tasks), n_agents=len(agents))
        
        results = []
        k = len(agents)
        
        for task in tasks:
            # Phase 1: Broadcast task to all agents
            # Each agent interprets the task message
            task_message = self._encode_task(task)
            
            agent_interpretations = []
            for agent in agents:
                # Each agent interprets the task
                interpreted = self._interpret_message(
                    task_message,
                    agent,
                    metrics,
                )
                agent_interpretations.append(interpreted)
            
            # Phase 2: Agents communicate to decide who handles task
            # This requires O(k²) pairwise communications
            for i, agent_i in enumerate(agents):
                for j, agent_j in enumerate(agents):
                    if i < j:  # Each pair once
                        # Agent i sends message to agent j
                        msg_i = self._create_agent_message(agent_i, task)
                        interpreted_ij = self._interpret_message(
                            msg_i,
                            agent_j,
                            metrics,
                            sender=agent_i,
                        )
                        
                        # Agent j sends message to agent i
                        msg_j = self._create_agent_message(agent_j, task)
                        interpreted_ji = self._interpret_message(
                            msg_j,
                            agent_i,
                            metrics,
                            sender=agent_j,
                        )
            
            # Phase 3: Select agent and execute
            # Selection based on interpreted capabilities
            selected_agent = self._select_agent(agents, task, metrics)
            
            if selected_agent:
                # Execute task
                success = self._execute_task(selected_agent, task)
                
                # Create result
                result = VerificationResult(
                    status=VerificationStatus.SUCCESS if success else VerificationStatus.FAILURE,
                    commitment_id=f"rep_{task.id}",
                    details={"agent": selected_agent.agent_id},
                )
                results.append(result)
                
                if success:
                    metrics.successful_coordinations += 1
                else:
                    metrics.failed_coordinations += 1
            else:
                metrics.failed_coordinations += 1
        
        # Calculate time
        end_time = datetime.now(timezone.utc)
        metrics.total_time_seconds = (end_time - start_time).total_seconds()
        
        return results, metrics
    
    def _encode_task(self, task: Task) -> np.ndarray:
        """Encode task as message embedding."""
        # Simple encoding based on task features
        embedding = np.zeros(self.message_dim, dtype=np.float32)
        embedding[0] = task.difficulty
        embedding[1] = task.reward
        embedding[2] = task.min_stake
        
        # Add some structure
        embedding[3:10] = self.rng.standard_normal(7).astype(np.float32)
        
        return embedding
    
    def _create_agent_message(
        self,
        agent: CommitmentAgent,
        task: Task,
    ) -> np.ndarray:
        """Create a message from an agent about a task."""
        # Combine agent embedding with task info
        agent_emb = agent.get_semantic_embedding()
        
        # Create message
        message = np.zeros(self.message_dim, dtype=np.float32)
        message[:32] = agent_emb[:32]
        message[32] = agent.estimate_success_probability(task.to_features())
        message[33:] = self.rng.standard_normal(31).astype(np.float32)
        
        return message
    
    def _interpret_message(
        self,
        message: np.ndarray,
        receiver: CommitmentAgent,
        metrics: ProtocolMetrics,
        sender: CommitmentAgent | None = None,
    ) -> np.ndarray:
        """
        Interpret a message with potential errors.
        
        Error rate scales with semantic drift (Lemma 2).
        """
        metrics.interpretation_operations += 1
        
        if sender:
            # Interpret with sender's embedding
            interpreted = receiver.interpret_message(
                message,
                sender.get_semantic_embedding(),
            )
        else:
            # Interpret without sender info (more error)
            noise = self.rng.standard_normal(len(message)) * self.semantic_drift
            interpreted = message + noise.astype(np.float32)
        
        # Check for interpretation error
        error_magnitude = np.linalg.norm(interpreted - message)
        if error_magnitude > 0.5:  # Threshold for "error"
            metrics.interpretation_errors += 1
        
        return interpreted
    
    def _select_agent(
        self,
        agents: list[CommitmentAgent],
        task: Task,
        metrics: ProtocolMetrics,
    ) -> CommitmentAgent | None:
        """Select best agent based on interpreted capabilities."""
        if not agents:
            return None
        
        # Score agents based on estimated success probability
        scores = []
        for agent in agents:
            prob = agent.estimate_success_probability(task.to_features())
            
            # Add noise from interpretation errors
            noise = self.rng.normal(0, self.semantic_drift * 0.5)
            noisy_prob = np.clip(prob + noise, 0, 1)
            
            scores.append((agent, noisy_prob))
        
        # Select highest scoring agent that can commit
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for agent, score in scores:
            if agent.can_commit(task.min_stake):
                return agent
        
        return None
    
    def _execute_task(
        self,
        agent: CommitmentAgent,
        task: Task,
    ) -> bool:
        """Execute task with potential interpretation errors."""
        # Base success probability
        base_prob = agent.estimate_success_probability(task.to_features())
        
        # Reduce probability due to interpretation errors
        # Error rate scales with semantic drift (Theorem 1c)
        error_factor = 1.0 - self.semantic_drift * 0.5
        adjusted_prob = base_prob * error_factor
        
        # Simulate execution
        return self.rng.random() < adjusted_prob


def compare_protocols(
    n_tasks: int,
    n_agents: int,
    semantic_drift: float = 0.0,
    difficulty: float = 0.5,
    seed: int | None = None,
) -> dict[str, ProtocolMetrics]:
    """
    Compare commitment and representation protocols.
    
    This validates Theorem 1:
    - Commitment: O(n·k) operations
    - Representation: O(n·k²) operations
    - Error rate for representation: O(ε·n·k²)
    
    Args:
        n_tasks: Number of tasks.
        n_agents: Number of agents.
        semantic_drift: Semantic drift for representation protocol.
        difficulty: Task difficulty.
        seed: Random seed.
        
    Returns:
        Dictionary mapping protocol name to metrics.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create tasks
    tasks = [
        Task(
            name=f"task_{i}",
            difficulty=difficulty + np.random.uniform(-0.1, 0.1),
            reward=1.0,
            min_stake=0.1,
        )
        for i in range(n_tasks)
    ]
    
    # Create agents for commitment protocol
    commitment_agents = [
        CommitmentAgent(f"commit_agent_{i}")
        for i in range(n_agents)
    ]
    
    # Create agents for representation protocol
    rep_agents = [
        CommitmentAgent(f"rep_agent_{i}")
        for i in range(n_agents)
    ]
    
    # Run commitment protocol
    commitment_protocol = CommitmentProtocol()
    commitment_results, commitment_metrics = commitment_protocol.coordinate(
        [Task(**t.__dict__) for t in tasks],  # Copy tasks
        commitment_agents,
    )
    
    # Run representation protocol
    rep_protocol = RepresentationProtocol(semantic_drift=semantic_drift)
    rep_results, rep_metrics = rep_protocol.coordinate(
        [Task(**t.__dict__) for t in tasks],  # Copy tasks
        rep_agents,
    )
    
    return {
        "commitment": commitment_metrics,
        "representation": rep_metrics,
    }


def validate_theorem_1(
    n_values: list[int],
    k_values: list[int],
    drift_values: list[float],
    seed: int = 42,
) -> dict[str, Any]:
    """
    Validate Theorem 1 (Communication Complexity).
    
    Tests:
    (a) Commitment: O(n·k) operations
    (b) Representation: O(n·k²) operations
    (c) Error rate: O(ε·n·k²) for representation
    
    Args:
        n_values: Task counts to test.
        k_values: Agent counts to test.
        drift_values: Semantic drift values to test.
        seed: Random seed.
        
    Returns:
        Validation results.
    """
    results = {
        "commitment_complexity": [],
        "representation_complexity": [],
        "commitment_success": [],
        "representation_success": [],
        "parameters": {
            "n_values": n_values,
            "k_values": k_values,
            "drift_values": drift_values,
        },
    }
    
    for n in n_values:
        for k in k_values:
            for drift in drift_values:
                metrics = compare_protocols(
                    n_tasks=n,
                    n_agents=k,
                    semantic_drift=drift,
                    seed=seed,
                )
                
                results["commitment_complexity"].append({
                    "n": n,
                    "k": k,
                    "drift": drift,
                    "operations": metrics["commitment"].communication_complexity,
                    "expected_order": n * k,  # O(n·k)
                })
                
                results["representation_complexity"].append({
                    "n": n,
                    "k": k,
                    "drift": drift,
                    "operations": metrics["representation"].communication_complexity,
                    "expected_order": n * k * k,  # O(n·k²)
                })
                
                results["commitment_success"].append({
                    "n": n,
                    "k": k,
                    "drift": drift,
                    "success_rate": metrics["commitment"].success_rate,
                })
                
                results["representation_success"].append({
                    "n": n,
                    "k": k,
                    "drift": drift,
                    "success_rate": metrics["representation"].success_rate,
                    "error_rate": metrics["representation"].error_rate,
                })
    
    return results
