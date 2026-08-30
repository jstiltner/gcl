"""
Reputation and stake management for the GCL framework.

This module provides:
- ReputationTracker: Tracks agent reputation based on commitment fulfillment
- StakeManager: Manages stake allocation and slashing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from gcl.core.commitment import (
    GroundedCommitment,
    VerificationResult,
    VerificationStatus,
)


class ReputationEventType(str, Enum):
    """Types of events that affect reputation."""
    
    COMMITMENT_FULFILLED = "commitment_fulfilled"
    COMMITMENT_FAILED = "commitment_failed"
    COMMITMENT_EXPIRED = "commitment_expired"
    STAKE_SLASHED = "stake_slashed"
    BONUS_AWARDED = "bonus_awarded"
    MANUAL_ADJUSTMENT = "manual_adjustment"


@dataclass
class ReputationEvent:
    """
    A single event that affects an agent's reputation.
    
    Attributes:
        id: Unique identifier for this event.
        agent_id: The agent whose reputation is affected.
        event_type: Type of reputation event.
        delta: Change in reputation score.
        commitment_id: Related commitment ID (if applicable).
        details: Additional event details.
        timestamp: When the event occurred.
    """
    
    id: str
    agent_id: str
    event_type: ReputationEventType
    delta: float
    commitment_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create(
        cls,
        agent_id: str,
        event_type: ReputationEventType,
        delta: float,
        commitment_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ReputationEvent:
        """Create a new reputation event."""
        return cls(
            id=str(uuid4()),
            agent_id=agent_id,
            event_type=event_type,
            delta=delta,
            commitment_id=commitment_id,
            details=details or {},
        )


@dataclass
class AgentReputation:
    """
    Reputation state for a single agent.
    
    Attributes:
        agent_id: The agent's identifier.
        score: Current reputation score.
        total_commitments: Total number of commitments made.
        fulfilled_commitments: Number of fulfilled commitments.
        failed_commitments: Number of failed commitments.
        total_stake_risked: Total stake put at risk.
        total_stake_lost: Total stake lost to slashing.
        history: List of reputation events.
    """
    
    agent_id: str
    score: float = 1.0
    total_commitments: int = 0
    fulfilled_commitments: int = 0
    failed_commitments: int = 0
    total_stake_risked: float = 0.0
    total_stake_lost: float = 0.0
    history: list[ReputationEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def fulfillment_rate(self) -> float:
        """Calculate the commitment fulfillment rate."""
        if self.total_commitments == 0:
            return 1.0  # No commitments = perfect record
        return self.fulfilled_commitments / self.total_commitments
    
    @property
    def stake_preservation_rate(self) -> float:
        """Calculate the rate of stake preserved."""
        if self.total_stake_risked == 0:
            return 1.0
        return 1.0 - (self.total_stake_lost / self.total_stake_risked)
    
    def add_event(self, event: ReputationEvent) -> None:
        """Add a reputation event and update score."""
        self.history.append(event)
        self.score += event.delta
        # Clamp score to reasonable bounds
        self.score = max(0.0, min(self.score, 10.0))


class ReputationConfig(BaseModel):
    """Configuration for reputation calculations."""
    
    # Base reputation changes
    fulfillment_bonus: float = Field(default=0.1, ge=0.0)
    failure_penalty: float = Field(default=0.2, ge=0.0)
    expiration_penalty: float = Field(default=0.05, ge=0.0)
    
    # Confidence-based modifiers
    confidence_bonus_multiplier: float = Field(default=0.5, ge=0.0)
    confidence_penalty_multiplier: float = Field(default=1.5, ge=0.0)
    
    # Stake-based modifiers
    stake_weight: float = Field(default=0.1, ge=0.0)
    
    # Decay settings
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    decay_interval_hours: float = Field(default=24.0, gt=0.0)
    
    # Bounds
    min_reputation: float = Field(default=0.0)
    max_reputation: float = Field(default=10.0)
    initial_reputation: float = Field(default=1.0)


class ReputationTracker:
    """
    Tracks reputation for multiple agents.
    
    The ReputationTracker maintains reputation scores for agents based on
    their commitment fulfillment history. Reputation affects:
    - Trust from other agents
    - Ability to make high-stake commitments
    - Priority in commitment matching
    
    Example:
        >>> tracker = ReputationTracker()
        >>> tracker.register_agent("agent-1")
        >>> 
        >>> # After a successful commitment
        >>> tracker.record_fulfillment("agent-1", commitment, result)
        >>> 
        >>> # Check reputation
        >>> rep = tracker.get_reputation("agent-1")
        >>> print(f"Score: {rep.score}, Rate: {rep.fulfillment_rate}")
    """
    
    def __init__(self, config: ReputationConfig | None = None) -> None:
        """
        Initialize the reputation tracker.
        
        Args:
            config: Configuration for reputation calculations.
        """
        self.config = config or ReputationConfig()
        self._agents: dict[str, AgentReputation] = {}
    
    def register_agent(self, agent_id: str) -> AgentReputation:
        """
        Register a new agent with initial reputation.
        
        Args:
            agent_id: The agent's identifier.
            
        Returns:
            The new AgentReputation object.
            
        Raises:
            ValueError: If agent is already registered.
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent '{agent_id}' is already registered")
        
        reputation = AgentReputation(
            agent_id=agent_id,
            score=self.config.initial_reputation,
        )
        self._agents[agent_id] = reputation
        return reputation
    
    def get_or_create_agent(self, agent_id: str) -> AgentReputation:
        """
        Get an agent's reputation, creating if necessary.
        
        Args:
            agent_id: The agent's identifier.
            
        Returns:
            The AgentReputation object.
        """
        if agent_id not in self._agents:
            return self.register_agent(agent_id)
        return self._agents[agent_id]
    
    def get_reputation(self, agent_id: str) -> AgentReputation | None:
        """
        Get an agent's reputation.
        
        Args:
            agent_id: The agent's identifier.
            
        Returns:
            The AgentReputation object, or None if not found.
        """
        return self._agents.get(agent_id)
    
    def get_score(self, agent_id: str) -> float:
        """
        Get an agent's reputation score.
        
        Args:
            agent_id: The agent's identifier.
            
        Returns:
            The reputation score, or initial_reputation if not found.
        """
        rep = self._agents.get(agent_id)
        return rep.score if rep else self.config.initial_reputation
    
    def record_fulfillment(
        self,
        agent_id: str,
        commitment: GroundedCommitment,
        result: VerificationResult,
    ) -> ReputationEvent:
        """
        Record a commitment fulfillment and update reputation.
        
        Args:
            agent_id: The agent's identifier.
            commitment: The commitment that was fulfilled.
            result: The verification result.
            
        Returns:
            The reputation event that was created.
        """
        rep = self.get_or_create_agent(agent_id)
        rep.total_commitments += 1
        rep.total_stake_risked += commitment.stake
        
        if result.status == VerificationStatus.SUCCESS:
            rep.fulfilled_commitments += 1
            
            # Calculate bonus based on confidence
            # Higher confidence = higher bonus (you were right to be confident)
            base_bonus = self.config.fulfillment_bonus
            confidence_modifier = commitment.confidence * self.config.confidence_bonus_multiplier
            stake_modifier = commitment.stake * self.config.stake_weight
            
            delta = base_bonus + confidence_modifier + stake_modifier
            
            event = ReputationEvent.create(
                agent_id=agent_id,
                event_type=ReputationEventType.COMMITMENT_FULFILLED,
                delta=delta,
                commitment_id=commitment.id,
                details={
                    "confidence": commitment.confidence,
                    "stake": commitment.stake,
                    "base_bonus": base_bonus,
                },
            )
        else:
            rep.failed_commitments += 1
            
            # Calculate penalty based on confidence
            # Higher confidence = higher penalty (you were wrong to be confident)
            base_penalty = self.config.failure_penalty
            confidence_modifier = commitment.confidence * self.config.confidence_penalty_multiplier
            stake_modifier = commitment.stake * self.config.stake_weight
            
            delta = -(base_penalty + confidence_modifier + stake_modifier)
            
            # Get severity from result if available
            severity = result.details.get("severity", 0.5)
            delta *= severity  # Scale by severity
            
            event = ReputationEvent.create(
                agent_id=agent_id,
                event_type=ReputationEventType.COMMITMENT_FAILED,
                delta=delta,
                commitment_id=commitment.id,
                details={
                    "confidence": commitment.confidence,
                    "stake": commitment.stake,
                    "severity": severity,
                    "failure_mode": result.triggered_failure_mode,
                },
            )
        
        rep.add_event(event)
        return event
    
    def record_expiration(
        self,
        agent_id: str,
        commitment: GroundedCommitment,
    ) -> ReputationEvent:
        """
        Record a commitment expiration.
        
        Args:
            agent_id: The agent's identifier.
            commitment: The commitment that expired.
            
        Returns:
            The reputation event that was created.
        """
        rep = self.get_or_create_agent(agent_id)
        rep.total_commitments += 1
        rep.failed_commitments += 1
        rep.total_stake_risked += commitment.stake
        
        delta = -self.config.expiration_penalty
        
        event = ReputationEvent.create(
            agent_id=agent_id,
            event_type=ReputationEventType.COMMITMENT_EXPIRED,
            delta=delta,
            commitment_id=commitment.id,
            details={
                "stake": commitment.stake,
            },
        )
        
        rep.add_event(event)
        return event
    
    def apply_decay(self, agent_id: str) -> float:
        """
        Apply reputation decay to an agent.
        
        Reputation decays towards the initial value over time.
        
        Args:
            agent_id: The agent's identifier.
            
        Returns:
            The new reputation score.
        """
        rep = self._agents.get(agent_id)
        if rep is None:
            return self.config.initial_reputation
        
        # Decay towards initial reputation
        diff = rep.score - self.config.initial_reputation
        decay = diff * self.config.decay_rate
        rep.score -= decay
        
        return rep.score
    
    def get_leaderboard(self, limit: int = 10) -> list[tuple[str, float]]:
        """
        Get the top agents by reputation.
        
        Args:
            limit: Maximum number of agents to return.
            
        Returns:
            List of (agent_id, score) tuples, sorted by score descending.
        """
        sorted_agents = sorted(
            self._agents.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        return [(agent_id, rep.score) for agent_id, rep in sorted_agents[:limit]]
    
    def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics across all agents."""
        if not self._agents:
            return {
                "total_agents": 0,
                "average_score": self.config.initial_reputation,
                "total_commitments": 0,
                "overall_fulfillment_rate": 0.0,
            }
        
        total_commitments = sum(r.total_commitments for r in self._agents.values())
        total_fulfilled = sum(r.fulfilled_commitments for r in self._agents.values())
        
        return {
            "total_agents": len(self._agents),
            "average_score": sum(r.score for r in self._agents.values()) / len(self._agents),
            "total_commitments": total_commitments,
            "overall_fulfillment_rate": total_fulfilled / total_commitments if total_commitments > 0 else 0.0,
            "total_stake_risked": sum(r.total_stake_risked for r in self._agents.values()),
            "total_stake_lost": sum(r.total_stake_lost for r in self._agents.values()),
        }


class StakeConfig(BaseModel):
    """Configuration for stake management."""
    
    # Stake limits
    min_stake: float = Field(default=0.0, ge=0.0)
    max_stake: float = Field(default=100.0, ge=0.0)
    max_total_stake_per_agent: float = Field(default=1000.0, ge=0.0)
    
    # Slashing parameters
    base_slash_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    severity_multiplier: float = Field(default=1.0, ge=0.0)
    
    # Recovery parameters
    recovery_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    recovery_interval_hours: float = Field(default=24.0, gt=0.0)


@dataclass
class StakeAccount:
    """
    Stake account for a single agent.
    
    Attributes:
        agent_id: The agent's identifier.
        available_stake: Stake available for new commitments.
        locked_stake: Stake locked in active commitments.
        total_slashed: Total stake lost to slashing.
        total_earned: Total stake earned from bonuses.
    """
    
    agent_id: str
    available_stake: float = 0.0
    locked_stake: float = 0.0
    total_slashed: float = 0.0
    total_earned: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def total_stake(self) -> float:
        """Total stake (available + locked)."""
        return self.available_stake + self.locked_stake
    
    def can_stake(self, amount: float) -> bool:
        """Check if the agent can stake the given amount."""
        return amount <= self.available_stake


@dataclass
class StakeTransaction:
    """
    A stake transaction record.
    
    Attributes:
        id: Unique transaction identifier.
        agent_id: The agent involved.
        transaction_type: Type of transaction.
        amount: Amount of stake involved.
        commitment_id: Related commitment (if applicable).
        timestamp: When the transaction occurred.
    """
    
    id: str
    agent_id: str
    transaction_type: str
    amount: float
    commitment_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class StakeManager:
    """
    Manages stake allocation and slashing.
    
    The StakeManager handles:
    - Stake deposits and withdrawals
    - Locking stake for commitments
    - Slashing stake on commitment failure
    - Awarding bonuses on commitment success
    
    Example:
        >>> manager = StakeManager()
        >>> manager.deposit("agent-1", 100.0)
        >>> 
        >>> # Lock stake for a commitment
        >>> manager.lock_stake("agent-1", commitment)
        >>> 
        >>> # On failure, slash the stake
        >>> manager.slash_stake("agent-1", commitment, result)
    """
    
    def __init__(self, config: StakeConfig | None = None) -> None:
        """
        Initialize the stake manager.
        
        Args:
            config: Configuration for stake management.
        """
        self.config = config or StakeConfig()
        self._accounts: dict[str, StakeAccount] = {}
        self._transactions: list[StakeTransaction] = []
        self._locked_commitments: dict[str, str] = {}  # commitment_id -> agent_id
    
    def get_or_create_account(self, agent_id: str) -> StakeAccount:
        """
        Get or create a stake account for an agent.
        
        Args:
            agent_id: The agent's identifier.
            
        Returns:
            The StakeAccount object.
        """
        if agent_id not in self._accounts:
            self._accounts[agent_id] = StakeAccount(agent_id=agent_id)
        return self._accounts[agent_id]
    
    def get_account(self, agent_id: str) -> StakeAccount | None:
        """
        Get a stake account.
        
        Args:
            agent_id: The agent's identifier.
            
        Returns:
            The StakeAccount, or None if not found.
        """
        return self._accounts.get(agent_id)
    
    def deposit(self, agent_id: str, amount: float) -> StakeTransaction:
        """
        Deposit stake into an agent's account.
        
        Args:
            agent_id: The agent's identifier.
            amount: Amount to deposit.
            
        Returns:
            The transaction record.
            
        Raises:
            ValueError: If amount is invalid.
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        account = self.get_or_create_account(agent_id)
        
        # Check max stake limit
        if account.total_stake + amount > self.config.max_total_stake_per_agent:
            raise ValueError(
                f"Deposit would exceed max stake limit "
                f"({account.total_stake + amount} > {self.config.max_total_stake_per_agent})"
            )
        
        account.available_stake += amount
        
        transaction = StakeTransaction(
            id=str(uuid4()),
            agent_id=agent_id,
            transaction_type="deposit",
            amount=amount,
        )
        self._transactions.append(transaction)
        
        return transaction
    
    def withdraw(self, agent_id: str, amount: float) -> StakeTransaction:
        """
        Withdraw stake from an agent's account.
        
        Args:
            agent_id: The agent's identifier.
            amount: Amount to withdraw.
            
        Returns:
            The transaction record.
            
        Raises:
            ValueError: If amount is invalid or insufficient funds.
        """
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        account = self.get_or_create_account(agent_id)
        
        if amount > account.available_stake:
            raise ValueError(
                f"Insufficient available stake "
                f"({amount} > {account.available_stake})"
            )
        
        account.available_stake -= amount
        
        transaction = StakeTransaction(
            id=str(uuid4()),
            agent_id=agent_id,
            transaction_type="withdrawal",
            amount=-amount,
        )
        self._transactions.append(transaction)
        
        return transaction
    
    def lock_stake(
        self,
        agent_id: str,
        commitment: GroundedCommitment,
    ) -> StakeTransaction:
        """
        Lock stake for a commitment.
        
        Args:
            agent_id: The agent's identifier.
            commitment: The commitment to lock stake for.
            
        Returns:
            The transaction record.
            
        Raises:
            ValueError: If insufficient stake or commitment already locked.
        """
        if commitment.id in self._locked_commitments:
            raise ValueError(f"Commitment {commitment.id} already has locked stake")
        
        account = self.get_or_create_account(agent_id)
        
        if commitment.stake > account.available_stake:
            raise ValueError(
                f"Insufficient available stake for commitment "
                f"({commitment.stake} > {account.available_stake})"
            )
        
        account.available_stake -= commitment.stake
        account.locked_stake += commitment.stake
        self._locked_commitments[commitment.id] = agent_id
        
        transaction = StakeTransaction(
            id=str(uuid4()),
            agent_id=agent_id,
            transaction_type="lock",
            amount=commitment.stake,
            commitment_id=commitment.id,
        )
        self._transactions.append(transaction)
        
        return transaction
    
    def unlock_stake(
        self,
        commitment: GroundedCommitment,
    ) -> StakeTransaction | None:
        """
        Unlock stake for a commitment (on success).
        
        Args:
            commitment: The commitment to unlock stake for.
            
        Returns:
            The transaction record, or None if no stake was locked.
        """
        agent_id = self._locked_commitments.get(commitment.id)
        if agent_id is None:
            return None
        
        account = self._accounts.get(agent_id)
        if account is None:
            return None
        
        account.locked_stake -= commitment.stake
        account.available_stake += commitment.stake
        del self._locked_commitments[commitment.id]
        
        transaction = StakeTransaction(
            id=str(uuid4()),
            agent_id=agent_id,
            transaction_type="unlock",
            amount=commitment.stake,
            commitment_id=commitment.id,
        )
        self._transactions.append(transaction)
        
        return transaction
    
    def slash_stake(
        self,
        commitment: GroundedCommitment,
        result: VerificationResult,
    ) -> StakeTransaction | None:
        """
        Slash stake for a failed commitment.
        
        Args:
            commitment: The commitment that failed.
            result: The verification result.
            
        Returns:
            The transaction record, or None if no stake was locked.
        """
        agent_id = self._locked_commitments.get(commitment.id)
        if agent_id is None:
            return None
        
        account = self._accounts.get(agent_id)
        if account is None:
            return None
        
        # Calculate slash amount based on severity
        severity = result.details.get("severity", self.config.base_slash_rate)
        slash_rate = min(1.0, severity * self.config.severity_multiplier)
        slash_amount = commitment.stake * slash_rate
        
        # Apply slash
        account.locked_stake -= commitment.stake
        account.available_stake += (commitment.stake - slash_amount)
        account.total_slashed += slash_amount
        del self._locked_commitments[commitment.id]
        
        transaction = StakeTransaction(
            id=str(uuid4()),
            agent_id=agent_id,
            transaction_type="slash",
            amount=-slash_amount,
            commitment_id=commitment.id,
            details={
                "severity": severity,
                "slash_rate": slash_rate,
                "original_stake": commitment.stake,
            },
        )
        self._transactions.append(transaction)
        
        return transaction
    
    def award_bonus(
        self,
        agent_id: str,
        amount: float,
        commitment_id: str | None = None,
        reason: str = "fulfillment_bonus",
    ) -> StakeTransaction:
        """
        Award a stake bonus to an agent.
        
        Args:
            agent_id: The agent's identifier.
            amount: Bonus amount.
            commitment_id: Related commitment (if applicable).
            reason: Reason for the bonus.
            
        Returns:
            The transaction record.
        """
        if amount <= 0:
            raise ValueError("Bonus amount must be positive")
        
        account = self.get_or_create_account(agent_id)
        account.available_stake += amount
        account.total_earned += amount
        
        transaction = StakeTransaction(
            id=str(uuid4()),
            agent_id=agent_id,
            transaction_type="bonus",
            amount=amount,
            commitment_id=commitment_id,
            details={"reason": reason},
        )
        self._transactions.append(transaction)
        
        return transaction
    
    def get_transaction_history(
        self,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[StakeTransaction]:
        """
        Get transaction history.
        
        Args:
            agent_id: Filter by agent (optional).
            limit: Maximum number of transactions to return.
            
        Returns:
            List of transactions, most recent first.
        """
        transactions = self._transactions
        if agent_id:
            transactions = [t for t in transactions if t.agent_id == agent_id]
        
        return sorted(
            transactions,
            key=lambda t: t.timestamp,
            reverse=True,
        )[:limit]
    
    def get_statistics(self) -> dict[str, Any]:
        """Get overall stake statistics."""
        if not self._accounts:
            return {
                "total_accounts": 0,
                "total_stake": 0.0,
                "total_locked": 0.0,
                "total_slashed": 0.0,
                "total_earned": 0.0,
            }
        
        return {
            "total_accounts": len(self._accounts),
            "total_stake": sum(a.total_stake for a in self._accounts.values()),
            "total_available": sum(a.available_stake for a in self._accounts.values()),
            "total_locked": sum(a.locked_stake for a in self._accounts.values()),
            "total_slashed": sum(a.total_slashed for a in self._accounts.values()),
            "total_earned": sum(a.total_earned for a in self._accounts.values()),
            "active_commitments": len(self._locked_commitments),
        }
