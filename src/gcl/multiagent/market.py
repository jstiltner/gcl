"""
Commitment market for multi-agent coordination.

This module implements a market where agents can:
- Post tasks requiring commitments
- Make commitments to complete tasks
- Match tasks with capable agents
- Track market statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from gcl.core.commitment import GroundedCommitment, VerificationResult, VerificationStatus
from gcl.core.reputation import ReputationTracker, StakeManager
from gcl.multiagent.agent import CommitmentAgent, AgentConfig


class TaskStatus(str, Enum):
    """Status of a task in the market."""
    
    PENDING = "pending"  # Waiting for commitment
    ASSIGNED = "assigned"  # Commitment made
    IN_PROGRESS = "in_progress"  # Being executed
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed
    EXPIRED = "expired"  # No commitment made in time


@dataclass
class Task:
    """
    A task posted to the commitment market.
    
    Tasks represent work that needs to be done. Agents can
    make commitments to complete tasks.
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Task description
    name: str = ""
    description: str = ""
    action_type: str = "generic"
    
    # Requirements
    required_capabilities: set[str] = field(default_factory=set)
    difficulty: float = 0.5  # 0-1, higher = harder
    
    # Rewards
    reward: float = 1.0
    min_stake: float = 0.1
    
    # Success criteria
    success_condition: str = "completed == True"
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: datetime | None = None
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str | None = None
    commitment_id: str | None = None
    
    # Features for ML
    features: dict[str, Any] = field(default_factory=dict)
    
    def to_features(self) -> dict[str, Any]:
        """Convert task to feature dictionary."""
        return {
            "task_id": self.id,
            "action_type": self.action_type,
            "difficulty": self.difficulty,
            "reward": self.reward,
            "min_stake": self.min_stake,
            "required_capabilities": self.required_capabilities,
            "tags": list(self.required_capabilities),
            **self.features,
        }


@dataclass
class TaskAllocation:
    """Record of a task allocation."""
    
    task_id: str
    agent_id: str
    commitment: GroundedCommitment
    result: VerificationResult | None = None
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class MarketConfig(BaseModel):
    """Configuration for the commitment market."""
    
    # Market parameters
    max_pending_tasks: int = Field(default=100, ge=1)
    task_expiry_seconds: float = Field(default=3600.0, ge=0.0)
    
    # Matching parameters
    min_reputation_threshold: float = Field(default=0.0, ge=0.0)
    capability_matching: bool = Field(default=True)
    
    # Pricing
    base_reward: float = Field(default=1.0, ge=0.0)
    difficulty_multiplier: float = Field(default=0.5, ge=0.0)
    
    # Verification
    auto_verify: bool = Field(default=True)


@dataclass
class MarketStatistics:
    """Statistics about market activity."""
    
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    expired_tasks: int = 0
    
    total_commitments: int = 0
    successful_commitments: int = 0
    failed_commitments: int = 0
    
    total_stake_locked: float = 0.0
    total_stake_slashed: float = 0.0
    total_rewards_paid: float = 0.0
    
    # Communication metrics
    verification_operations: int = 0
    interpretation_operations: int = 0
    
    @property
    def task_success_rate(self) -> float:
        """Task completion success rate."""
        total = self.completed_tasks + self.failed_tasks
        return self.completed_tasks / total if total > 0 else 0.0
    
    @property
    def commitment_success_rate(self) -> float:
        """Commitment fulfillment rate."""
        total = self.successful_commitments + self.failed_commitments
        return self.successful_commitments / total if total > 0 else 0.0


class CommitmentMarket:
    """
    A market for coordinating agents through commitments.
    
    The market:
    - Accepts task postings from requesters
    - Matches tasks with capable agents
    - Tracks commitments and their outcomes
    - Manages rewards and penalties
    
    This implements the commitment-based coordination from Theorem 1,
    which has O(n·k) communication complexity vs O(n·k²) for
    representation-based coordination.
    
    Example:
        >>> market = CommitmentMarket()
        >>> market.register_agent(agent)
        >>> task = Task(name="classify", difficulty=0.5)
        >>> market.post_task(task)
        >>> allocation = market.match_and_allocate(task.id)
    """
    
    def __init__(
        self,
        config: MarketConfig | None = None,
        reputation_tracker: ReputationTracker | None = None,
        stake_manager: StakeManager | None = None,
    ) -> None:
        """
        Initialize the market.
        
        Args:
            config: Market configuration.
            reputation_tracker: Shared reputation tracker.
            stake_manager: Shared stake manager.
        """
        self.config = config or MarketConfig()
        self.reputation_tracker = reputation_tracker or ReputationTracker()
        self.stake_manager = stake_manager or StakeManager()
        
        # Registered agents
        self.agents: dict[str, CommitmentAgent] = {}
        
        # Tasks
        self.pending_tasks: dict[str, Task] = {}
        self.active_tasks: dict[str, Task] = {}
        self.completed_tasks: dict[str, Task] = {}
        
        # Allocations
        self.allocations: dict[str, TaskAllocation] = {}
        
        # Statistics
        self.stats = MarketStatistics()
        
        # Random state
        self.rng = np.random.default_rng()
    
    def register_agent(self, agent: CommitmentAgent) -> None:
        """
        Register an agent with the market.
        
        Args:
            agent: Agent to register.
        """
        self.agents[agent.agent_id] = agent
        
        # Ensure agent is in shared systems
        if not self.reputation_tracker.get_reputation(agent.agent_id):
            self.reputation_tracker.register_agent(agent.agent_id)
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the market.
        
        Args:
            agent_id: ID of agent to unregister.
            
        Returns:
            True if agent was unregistered.
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False
    
    def post_task(self, task: Task) -> str:
        """
        Post a task to the market.
        
        Args:
            task: Task to post.
            
        Returns:
            Task ID.
        """
        if len(self.pending_tasks) >= self.config.max_pending_tasks:
            # Remove oldest expired task
            self._cleanup_expired_tasks()
        
        self.pending_tasks[task.id] = task
        self.stats.total_tasks += 1
        
        return task.id
    
    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return (
            self.pending_tasks.get(task_id) or
            self.active_tasks.get(task_id) or
            self.completed_tasks.get(task_id)
        )
    
    def get_pending_tasks(
        self,
        agent_id: str | None = None,
        capability_filter: set[str] | None = None,
    ) -> list[Task]:
        """
        Get pending tasks, optionally filtered.
        
        Args:
            agent_id: Filter to tasks agent can complete.
            capability_filter: Filter by required capabilities.
            
        Returns:
            List of matching tasks.
        """
        tasks = list(self.pending_tasks.values())
        
        if capability_filter:
            tasks = [
                t for t in tasks
                if not t.required_capabilities or
                t.required_capabilities <= capability_filter
            ]
        
        if agent_id and agent_id in self.agents:
            agent = self.agents[agent_id]
            if self.config.capability_matching:
                tasks = [
                    t for t in tasks
                    if not t.required_capabilities or
                    t.required_capabilities <= agent.config.capabilities
                ]
        
        return tasks
    
    def find_capable_agents(self, task: Task) -> list[CommitmentAgent]:
        """
        Find agents capable of completing a task.
        
        Args:
            task: Task to find agents for.
            
        Returns:
            List of capable agents.
        """
        capable = []
        
        for agent in self.agents.values():
            # Check capabilities
            if self.config.capability_matching and task.required_capabilities:
                if not task.required_capabilities <= agent.config.capabilities:
                    continue
            
            # Check reputation
            rep = self.reputation_tracker.get_reputation(agent.agent_id)
            if rep and rep.score < self.config.min_reputation_threshold:
                continue
            
            # Check if agent can commit
            if agent.can_commit(task.min_stake):
                capable.append(agent)
        
        return capable
    
    def match_and_allocate(
        self,
        task_id: str,
        agent_id: str | None = None,
    ) -> TaskAllocation | None:
        """
        Match a task with an agent and create allocation.
        
        Args:
            task_id: Task to allocate.
            agent_id: Specific agent to allocate to (optional).
            
        Returns:
            Task allocation, or None if no match.
        """
        task = self.pending_tasks.get(task_id)
        if not task:
            return None
        
        # Find agent
        if agent_id:
            if agent_id not in self.agents:
                return None
            agent = self.agents[agent_id]
        else:
            # Find best capable agent
            capable = self.find_capable_agents(task)
            if not capable:
                return None
            
            # Select agent with highest reputation
            agent = max(
                capable,
                key=lambda a: self.reputation_tracker.get_score(a.agent_id),
            )
        
        # Get agent's decision
        should_commit, confidence, stake_fraction = agent.decide_commitment(
            task.to_features(),
            task.min_stake,
        )
        
        if not should_commit:
            return None
        
        # Calculate stake
        state = agent.get_state()
        stake = max(task.min_stake, state.available_stake * stake_fraction)
        
        # Create commitment
        commitment = agent.make_commitment(
            task_id=task.id,
            task_features=task.to_features(),
            action_type=task.action_type,
            success_condition=task.success_condition,
            stake=stake,
            confidence=confidence,
        )
        
        # Update task status
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent.agent_id
        task.commitment_id = commitment.id
        
        # Move to active
        del self.pending_tasks[task_id]
        self.active_tasks[task_id] = task
        
        # Create allocation
        allocation = TaskAllocation(
            task_id=task.id,
            agent_id=agent.agent_id,
            commitment=commitment,
        )
        self.allocations[task.id] = allocation
        
        # Update stats
        self.stats.total_commitments += 1
        self.stats.total_stake_locked += stake
        self.stats.verification_operations += 1  # One verification per commitment
        
        return allocation
    
    def execute_task(
        self,
        task_id: str,
        execution_fn: Any | None = None,
    ) -> VerificationResult | None:
        """
        Execute an allocated task.
        
        Args:
            task_id: Task to execute.
            execution_fn: Optional execution function.
            
        Returns:
            Verification result, or None if task not found.
        """
        task = self.active_tasks.get(task_id)
        if not task or not task.assigned_agent:
            return None
        
        allocation = self.allocations.get(task_id)
        if not allocation:
            return None
        
        agent = self.agents.get(task.assigned_agent)
        if not agent:
            return None
        
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        
        # Execute commitment
        result = agent.execute_commitment(allocation.commitment, execution_fn)
        
        # Update allocation
        allocation.result = result
        allocation.completed_at = datetime.now(timezone.utc)
        
        # Update task status
        if result.status == VerificationStatus.SUCCESS:
            task.status = TaskStatus.COMPLETED
            self.stats.completed_tasks += 1
            self.stats.successful_commitments += 1
            self.stats.total_rewards_paid += task.reward
        else:
            task.status = TaskStatus.FAILED
            self.stats.failed_tasks += 1
            self.stats.failed_commitments += 1
            self.stats.total_stake_slashed += allocation.commitment.stake * 0.5
        
        # Move to completed
        del self.active_tasks[task_id]
        self.completed_tasks[task_id] = task
        
        # Update stats
        self.stats.verification_operations += 1
        
        return result
    
    def run_round(
        self,
        tasks: list[Task] | None = None,
        execution_fn: Any | None = None,
    ) -> list[VerificationResult]:
        """
        Run a complete round of task allocation and execution.
        
        Args:
            tasks: Tasks to post (optional, uses pending if None).
            execution_fn: Optional execution function.
            
        Returns:
            List of verification results.
        """
        # Post new tasks
        if tasks:
            for task in tasks:
                self.post_task(task)
        
        # Allocate pending tasks
        for task_id in list(self.pending_tasks.keys()):
            self.match_and_allocate(task_id)
        
        # Execute active tasks
        results = []
        for task_id in list(self.active_tasks.keys()):
            result = self.execute_task(task_id, execution_fn)
            if result:
                results.append(result)
        
        return results
    
    def _cleanup_expired_tasks(self) -> int:
        """Remove expired tasks."""
        now = datetime.now(timezone.utc)
        expired = []
        
        for task_id, task in self.pending_tasks.items():
            if task.deadline and task.deadline < now:
                expired.append(task_id)
            elif (now - task.created_at).total_seconds() > self.config.task_expiry_seconds:
                expired.append(task_id)
        
        for task_id in expired:
            task = self.pending_tasks.pop(task_id)
            task.status = TaskStatus.EXPIRED
            self.completed_tasks[task_id] = task
            self.stats.expired_tasks += 1
        
        return len(expired)
    
    def get_statistics(self) -> MarketStatistics:
        """Get market statistics."""
        return self.stats
    
    def get_agent_rankings(self) -> list[tuple[str, float]]:
        """
        Get agents ranked by reputation.
        
        Returns:
            List of (agent_id, reputation_score) tuples.
        """
        rankings = []
        for agent_id in self.agents:
            score = self.reputation_tracker.get_score(agent_id)
            rankings.append((agent_id, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


def create_coordination_task(
    n_subtasks: int,
    difficulty: float = 0.5,
    required_agents: int = 1,
) -> list[Task]:
    """
    Create a coordination task with multiple subtasks.
    
    Args:
        n_subtasks: Number of subtasks.
        difficulty: Base difficulty level.
        required_agents: Agents required per subtask.
        
    Returns:
        List of subtasks.
    """
    tasks = []
    for i in range(n_subtasks):
        task = Task(
            name=f"subtask_{i}",
            description=f"Subtask {i} of coordination task",
            action_type="coordinate",
            difficulty=difficulty + np.random.uniform(-0.1, 0.1),
            reward=1.0 / n_subtasks,
            min_stake=0.1,
            features={
                "subtask_index": i,
                "total_subtasks": n_subtasks,
                "required_agents": required_agents,
            },
        )
        tasks.append(task)
    
    return tasks
