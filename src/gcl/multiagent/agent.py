"""
Multi-agent commitment agents.

This module provides agent abstractions for multi-agent coordination
through commitments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

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
from gcl.core.reputation import AgentReputation, ReputationTracker, StakeManager


class AgentType(str, Enum):
    """Types of agents in the market."""
    
    WORKER = "worker"  # Executes tasks
    REQUESTER = "requester"  # Requests tasks
    COORDINATOR = "coordinator"  # Coordinates between agents


class AgentConfig(BaseModel):
    """Configuration for a commitment agent."""
    
    # Identity
    agent_type: AgentType = Field(default=AgentType.WORKER)
    
    # Capabilities
    capabilities: set[str] = Field(default_factory=set)
    max_concurrent_commitments: int = Field(default=5, ge=1)
    
    # Resources
    initial_stake: float = Field(default=100.0, ge=0.0)
    
    # Behavior
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    min_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Communication
    semantic_drift: float = Field(default=0.0, ge=0.0)  # For representation-based comparison


@dataclass
class AgentState:
    """Current state of an agent."""
    
    # Identity
    agent_id: str
    agent_type: AgentType
    
    # Resources
    available_stake: float
    locked_stake: float
    
    # Reputation
    reputation_score: float
    fulfillment_rate: float
    
    # Activity
    active_commitments: int
    total_commitments: int
    total_successes: int
    total_failures: int
    
    # Timing
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def total_stake(self) -> float:
        """Total stake (available + locked)."""
        return self.available_stake + self.locked_stake
    
    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        total = self.total_successes + self.total_failures
        return self.total_successes / total if total > 0 else 0.5


class CommitmentAgent:
    """
    An agent that coordinates through commitments.
    
    Agents can:
    - Make commitments to perform tasks
    - Request commitments from other agents
    - Verify commitment fulfillment
    - Manage reputation and stake
    
    Example:
        >>> agent = CommitmentAgent("agent-1", config=AgentConfig())
        >>> commitment = agent.make_commitment(task, confidence=0.8)
        >>> result = agent.execute_commitment(commitment)
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        config: AgentConfig | None = None,
        reputation_tracker: ReputationTracker | None = None,
        stake_manager: StakeManager | None = None,
    ) -> None:
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique agent identifier.
            config: Agent configuration.
            reputation_tracker: Shared reputation tracker.
            stake_manager: Shared stake manager.
        """
        self.agent_id = agent_id or str(uuid4())
        self.config = config or AgentConfig()
        
        # Shared systems
        self.reputation_tracker = reputation_tracker or ReputationTracker()
        self.stake_manager = stake_manager or StakeManager()
        
        # Initialize in systems
        if not self.reputation_tracker.get_reputation(self.agent_id):
            self.reputation_tracker.register_agent(self.agent_id)
        
        if not self.stake_manager.get_account(self.agent_id):
            self.stake_manager.deposit(self.agent_id, self.config.initial_stake)
        
        # Local state
        self.portfolio = CommitmentPortfolio(owner=self.agent_id)
        self.commitment_history: list[tuple[GroundedCommitment, VerificationResult | None]] = []
        
        # Semantic embedding (for representation-based comparison)
        self._semantic_embedding: np.ndarray | None = None
        
        # Random state for stochastic behavior
        self.rng = np.random.default_rng()
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        account = self.stake_manager.get_account(self.agent_id)
        reputation = self.reputation_tracker.get_reputation(self.agent_id)
        
        successes = sum(
            1 for _, r in self.commitment_history
            if r and r.status == VerificationStatus.SUCCESS
        )
        failures = sum(
            1 for _, r in self.commitment_history
            if r and r.status == VerificationStatus.FAILURE
        )
        
        return AgentState(
            agent_id=self.agent_id,
            agent_type=self.config.agent_type,
            available_stake=account.available_stake if account else 0.0,
            locked_stake=account.locked_stake if account else 0.0,
            reputation_score=reputation.score if reputation else 1.0,
            fulfillment_rate=reputation.fulfillment_rate if reputation else 0.5,
            active_commitments=len(self.portfolio.active_commitments),
            total_commitments=len(self.commitment_history),
            total_successes=successes,
            total_failures=failures,
        )
    
    def can_commit(self, stake_required: float = 0.0) -> bool:
        """Check if agent can make a new commitment."""
        state = self.get_state()
        
        # Check concurrent commitment limit
        if state.active_commitments >= self.config.max_concurrent_commitments:
            return False
        
        # Check stake availability
        if stake_required > state.available_stake:
            return False
        
        return True
    
    def estimate_success_probability(
        self,
        task_features: dict[str, Any],
    ) -> float:
        """
        Estimate probability of successfully completing a task.
        
        Args:
            task_features: Features describing the task.
            
        Returns:
            Estimated success probability.
        """
        # Base probability from capabilities
        required_capabilities = task_features.get("required_capabilities", set())
        if required_capabilities:
            capability_match = len(self.config.capabilities & required_capabilities) / len(required_capabilities)
        else:
            capability_match = 1.0
        
        # Adjust for difficulty
        difficulty = task_features.get("difficulty", 0.5)
        difficulty_factor = 1.0 - difficulty * 0.5
        
        # Adjust for historical performance
        state = self.get_state()
        history_factor = 0.5 + 0.5 * state.success_rate
        
        # Combine factors
        prob = capability_match * difficulty_factor * history_factor
        
        # Add noise based on uncertainty
        noise = self.rng.normal(0, 0.1)
        prob = np.clip(prob + noise, 0.1, 0.95)
        
        return float(prob)
    
    def decide_commitment(
        self,
        task_features: dict[str, Any],
        stake_required: float,
    ) -> tuple[bool, float, float]:
        """
        Decide whether to commit to a task.
        
        Args:
            task_features: Features describing the task.
            stake_required: Stake required for the commitment.
            
        Returns:
            Tuple of (should_commit, confidence, stake_fraction).
        """
        if not self.can_commit(stake_required):
            return False, 0.0, 0.0
        
        # Estimate success probability
        success_prob = self.estimate_success_probability(task_features)
        
        # Check confidence threshold
        if success_prob < self.config.min_confidence_threshold:
            return False, success_prob, 0.0
        
        # Determine stake based on confidence and risk tolerance
        state = self.get_state()
        max_stake = state.available_stake * self.config.risk_tolerance
        stake_fraction = min(stake_required / max_stake, 1.0) if max_stake > 0 else 0.0
        
        # Adjust stake based on confidence
        stake_fraction *= success_prob
        
        return True, success_prob, stake_fraction
    
    def make_commitment(
        self,
        task_id: str,
        task_features: dict[str, Any],
        action_type: str,
        success_condition: str,
        stake: float,
        confidence: float,
    ) -> GroundedCommitment:
        """
        Create a commitment for a task.
        
        Args:
            task_id: Unique task identifier.
            task_features: Features describing the task.
            action_type: Type of action to perform.
            success_condition: Condition for success.
            stake: Stake to risk.
            confidence: Confidence level.
            
        Returns:
            The created commitment.
        """
        commitment = GroundedCommitment(
            issuer=self.agent_id,
            trigger_conditions=[
                Predicate(name="task_match", expression=f"task_id == '{task_id}'")
            ],
            promised_behavior=ActionSpec(
                action_type=action_type,
                parameters={"task_id": task_id, **task_features},
            ),
            success_condition=Predicate(name="success", expression=success_condition),
            failure_modes=[
                FailureMode(
                    name="task_failed",
                    condition=Predicate(name="failed", expression=f"not ({success_condition})"),
                    consequence=Consequence(
                        consequence_type="stake_slash",
                        magnitude=0.5,
                        description="Task completion failed",
                    ),
                    severity=0.5,
                )
            ],
            stake=stake,
            confidence=confidence,
            valid_contexts=ContextRegion(
                description=f"Task {task_id} context",
                tags=set(task_features.get("tags", [])),
            ),
            metadata={
                "task_id": task_id,
                "task_features": task_features,
            },
        )
        
        # Add to portfolio and lock stake
        self.portfolio.add_commitment(commitment)
        self.stake_manager.lock_stake(self.agent_id, commitment)
        
        return commitment
    
    def execute_commitment(
        self,
        commitment: GroundedCommitment,
        execution_fn: Callable[[GroundedCommitment], bool] | None = None,
    ) -> VerificationResult:
        """
        Execute a commitment and return the result.
        
        Args:
            commitment: The commitment to execute.
            execution_fn: Optional function to execute the task.
            
        Returns:
            Verification result.
        """
        # Execute the task
        if execution_fn:
            success = execution_fn(commitment)
        else:
            # Simulate execution based on confidence
            success = self.rng.random() < commitment.confidence
        
        # Create result
        if success:
            result = VerificationResult(
                status=VerificationStatus.SUCCESS,
                commitment_id=commitment.id,
                details={"executed_by": self.agent_id},
            )
            self.stake_manager.unlock_stake(commitment)
        else:
            result = VerificationResult(
                status=VerificationStatus.FAILURE,
                commitment_id=commitment.id,
                triggered_failure_mode="task_failed",
                details={"executed_by": self.agent_id},
            )
            self.stake_manager.slash_stake(commitment, result)
        
        # Update reputation
        self.reputation_tracker.record_fulfillment(self.agent_id, commitment, result)
        
        # Record in history
        self.commitment_history.append((commitment, result))
        
        # Remove from portfolio
        self.portfolio.remove_commitment(commitment.id)
        
        return result
    
    def get_semantic_embedding(self) -> np.ndarray:
        """
        Get semantic embedding for representation-based communication.
        
        Returns:
            Semantic embedding vector.
        """
        if self._semantic_embedding is None:
            # Initialize random embedding
            self._semantic_embedding = self.rng.standard_normal(64).astype(np.float32)
        
        # Add drift
        if self.config.semantic_drift > 0:
            drift = self.rng.standard_normal(64) * self.config.semantic_drift
            return self._semantic_embedding + drift.astype(np.float32)
        
        return self._semantic_embedding
    
    def interpret_message(
        self,
        message: np.ndarray,
        sender_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Interpret a message from another agent.
        
        For representation-based communication, interpretation
        is subject to semantic drift.
        
        Args:
            message: The message embedding.
            sender_embedding: Sender's semantic embedding.
            
        Returns:
            Interpreted message.
        """
        # Compute semantic distance
        my_embedding = self.get_semantic_embedding()
        distance = np.linalg.norm(my_embedding - sender_embedding)
        
        # Add interpretation noise proportional to distance
        noise_scale = distance * 0.1
        noise = self.rng.standard_normal(len(message)) * noise_scale
        
        return message + noise.astype(np.float32)
    
    def __repr__(self) -> str:
        state = self.get_state()
        return (
            f"CommitmentAgent(id={self.agent_id}, "
            f"type={state.agent_type.value}, "
            f"reputation={state.reputation_score:.2f}, "
            f"stake={state.total_stake:.2f})"
        )
