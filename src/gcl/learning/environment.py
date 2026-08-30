"""
Gymnasium-compatible environment for commitment learning.

This module provides a reinforcement learning environment where agents
learn what commitments to make based on task characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, SupportsFloat
from uuid import uuid4

import gymnasium as gym
import numpy as np
from gymnasium import spaces
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
from gcl.core.reputation import ReputationTracker, StakeManager
from gcl.core.verification import VerificationContext, VerificationEngine


class TaskDifficulty(str, Enum):
    """Difficulty levels for tasks."""
    
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    IMPOSSIBLE = "impossible"


@dataclass
class Task:
    """
    A task that the agent must decide whether to commit to.
    
    Attributes:
        id: Unique task identifier.
        difficulty: Task difficulty level.
        features: Feature vector representing task characteristics.
        true_success_probability: Actual probability of success (hidden from agent).
        context: Additional context information.
        reward_on_success: Reward if commitment is fulfilled.
        penalty_on_failure: Penalty if commitment fails.
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    features: np.ndarray = field(default_factory=lambda: np.zeros(8))
    true_success_probability: float = 0.5
    context: dict[str, Any] = field(default_factory=dict)
    reward_on_success: float = 1.0
    penalty_on_failure: float = 0.5
    
    def __post_init__(self):
        if isinstance(self.features, list):
            self.features = np.array(self.features, dtype=np.float32)


class TaskGenerator:
    """
    Generates tasks with varying difficulties and characteristics.
    
    The generator creates tasks with known success probabilities,
    allowing us to evaluate whether agents learn to calibrate
    their confidence appropriately.
    """
    
    def __init__(
        self,
        feature_dim: int = 8,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the task generator.
        
        Args:
            feature_dim: Dimension of task feature vectors.
            seed: Random seed for reproducibility.
        """
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        
        # Difficulty -> success probability ranges
        self.difficulty_ranges = {
            TaskDifficulty.EASY: (0.8, 0.95),
            TaskDifficulty.MEDIUM: (0.5, 0.8),
            TaskDifficulty.HARD: (0.2, 0.5),
            TaskDifficulty.IMPOSSIBLE: (0.0, 0.1),
        }
    
    def generate(
        self,
        difficulty: TaskDifficulty | None = None,
        batch_size: int = 1,
    ) -> list[Task]:
        """
        Generate tasks.
        
        Args:
            difficulty: Specific difficulty, or None for random.
            batch_size: Number of tasks to generate.
            
        Returns:
            List of generated tasks.
        """
        tasks = []
        difficulties = list(TaskDifficulty)
        
        for _ in range(batch_size):
            # Select difficulty
            if difficulty is None:
                idx = self.rng.integers(0, len(difficulties))
                diff = difficulties[idx]
            else:
                diff = difficulty
            
            # Generate success probability based on difficulty
            prob_range = self.difficulty_ranges[diff]
            success_prob = self.rng.uniform(prob_range[0], prob_range[1])
            
            # Generate features that correlate with difficulty
            # Features encode task characteristics that hint at difficulty
            features = self._generate_features(diff, success_prob)
            
            # Reward/penalty based on difficulty
            reward = 1.0 + (1.0 - success_prob) * 0.5  # Higher reward for harder tasks
            penalty = 0.3 + success_prob * 0.4  # Higher penalty for easier tasks (should have committed)
            
            task = Task(
                difficulty=diff,
                features=features,
                true_success_probability=success_prob,
                reward_on_success=reward,
                penalty_on_failure=penalty,
            )
            tasks.append(task)
        
        return tasks
    
    def _generate_features(
        self,
        difficulty: TaskDifficulty,
        success_prob: float,
    ) -> np.ndarray:
        """Generate feature vector for a task."""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Feature 0: Difficulty indicator (noisy)
        difficulty_map = {
            TaskDifficulty.EASY: 0.2,
            TaskDifficulty.MEDIUM: 0.5,
            TaskDifficulty.HARD: 0.8,
            TaskDifficulty.IMPOSSIBLE: 1.0,
        }
        features[0] = difficulty_map[difficulty] + self.rng.normal(0, 0.1)
        
        # Feature 1: Success probability hint (noisy)
        features[1] = 1.0 - success_prob + self.rng.normal(0, 0.15)
        
        # Features 2-4: Task type indicators
        features[2:5] = self.rng.random(3)
        
        # Features 5-7: Random noise
        features[5:8] = self.rng.normal(0, 0.5, 3)
        
        # Clip to reasonable range
        features = np.clip(features, -2.0, 2.0)
        
        return features


@dataclass
class CommitmentObservation:
    """
    Observation provided to the agent.
    
    Attributes:
        task_features: Feature vector of the current task.
        reputation_score: Agent's current reputation.
        available_stake: Stake available for commitment.
        active_commitment_count: Number of active commitments.
        recent_success_rate: Success rate over recent commitments.
    """
    
    task_features: np.ndarray
    reputation_score: float
    available_stake: float
    active_commitment_count: int
    recent_success_rate: float
    
    def to_array(self) -> np.ndarray:
        """Convert observation to flat numpy array."""
        return np.concatenate([
            self.task_features,
            np.array([
                self.reputation_score,
                self.available_stake,
                self.active_commitment_count,
                self.recent_success_rate,
            ], dtype=np.float32),
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, feature_dim: int = 8) -> CommitmentObservation:
        """Create observation from flat array."""
        return cls(
            task_features=arr[:feature_dim],
            reputation_score=float(arr[feature_dim]),
            available_stake=float(arr[feature_dim + 1]),
            active_commitment_count=int(arr[feature_dim + 2]),
            recent_success_rate=float(arr[feature_dim + 3]),
        )


@dataclass
class CommitmentAction:
    """
    Action taken by the agent.
    
    Attributes:
        commit: Whether to make a commitment.
        confidence: Confidence level (0-1) if committing.
        stake_fraction: Fraction of available stake to risk (0-1).
    """
    
    commit: bool
    confidence: float = 0.5
    stake_fraction: float = 0.1
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> CommitmentAction:
        """Create action from array [commit_prob, confidence, stake_fraction]."""
        return cls(
            commit=bool(arr[0] > 0.5),  # Convert numpy bool to Python bool
            confidence=float(np.clip(arr[1], 0.0, 1.0)),
            stake_fraction=float(np.clip(arr[2], 0.0, 1.0)),
        )
    
    def to_array(self) -> np.ndarray:
        """Convert action to array."""
        return np.array([
            1.0 if self.commit else 0.0,
            self.confidence,
            self.stake_fraction,
        ], dtype=np.float32)


class CommitmentEnvConfig(BaseModel):
    """Configuration for the commitment environment."""
    
    # Task settings
    feature_dim: int = Field(default=8, ge=1)
    max_steps: int = Field(default=100, ge=1)
    
    # Stake settings
    initial_stake: float = Field(default=100.0, ge=0.0)
    min_stake_per_commitment: float = Field(default=1.0, ge=0.0)
    max_stake_per_commitment: float = Field(default=20.0, ge=0.0)
    
    # Reputation settings
    initial_reputation: float = Field(default=1.0, ge=0.0)
    
    # Reward settings
    success_reward_multiplier: float = Field(default=1.0, ge=0.0)
    failure_penalty_multiplier: float = Field(default=1.0, ge=0.0)
    abstain_penalty: float = Field(default=0.01, ge=0.0)  # Small penalty for not committing
    
    # Calibration bonus
    calibration_bonus: float = Field(default=0.1, ge=0.0)  # Bonus for well-calibrated confidence


class CommitmentEnv(gym.Env):
    """
    Gymnasium environment for learning commitment policies.
    
    The agent receives tasks and must decide:
    1. Whether to commit to the task
    2. What confidence level to express
    3. How much stake to risk
    
    The agent learns to:
    - Commit with high confidence to easy tasks
    - Commit with low confidence to hard tasks
    - Refuse commitment to impossible tasks
    - Calibrate confidence to match actual success probability
    
    Observation Space:
        - Task features (feature_dim floats)
        - Reputation score (1 float)
        - Available stake (1 float)
        - Active commitment count (1 float)
        - Recent success rate (1 float)
    
    Action Space:
        - Commit decision (1 float, >0.5 = commit)
        - Confidence level (1 float, 0-1)
        - Stake fraction (1 float, 0-1)
    
    Example:
        >>> env = CommitmentEnv()
        >>> obs, info = env.reset()
        >>> action = np.array([0.8, 0.7, 0.1])  # Commit with 70% confidence, 10% stake
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: CommitmentEnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration.
            seed: Random seed.
        """
        super().__init__()
        
        self.config = config or CommitmentEnvConfig()
        self._seed = seed
        
        # Initialize components
        self.task_generator = TaskGenerator(
            feature_dim=self.config.feature_dim,
            seed=seed,
        )
        self.reputation_tracker = ReputationTracker()
        self.stake_manager = StakeManager()
        self.verification_engine = VerificationEngine()
        
        # Define spaces
        obs_dim = self.config.feature_dim + 4  # features + reputation + stake + count + rate
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        
        # State
        self.agent_id = "agent-0"
        self.current_task: Task | None = None
        self.step_count = 0
        self.commitment_history: list[tuple[GroundedCommitment, VerificationResult | None]] = []
        self.rng = np.random.default_rng(seed)
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Initial observation and info dict.
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.task_generator = TaskGenerator(
                feature_dim=self.config.feature_dim,
                seed=seed,
            )
        
        # Reset state
        self.step_count = 0
        self.commitment_history = []
        
        # Reset reputation and stake
        self.reputation_tracker = ReputationTracker()
        self.stake_manager = StakeManager()
        
        # Initialize agent
        self.reputation_tracker.register_agent(self.agent_id)
        self.stake_manager.deposit(self.agent_id, self.config.initial_stake)
        
        # Generate first task
        self.current_task = self.task_generator.generate()[0]
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs.to_array(), info
    
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action array [commit_prob, confidence, stake_fraction].
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.current_task is not None, "Must call reset() first"
        
        self.step_count += 1
        
        # Parse action
        commitment_action = CommitmentAction.from_array(action)
        
        # Calculate reward
        reward = 0.0
        info: dict[str, Any] = {
            "task_id": self.current_task.id,
            "task_difficulty": self.current_task.difficulty.value,
            "true_success_prob": self.current_task.true_success_probability,
            "action_commit": commitment_action.commit,
            "action_confidence": commitment_action.confidence,
            "action_stake_fraction": commitment_action.stake_fraction,
        }
        
        if commitment_action.commit:
            # Agent decided to commit
            reward, commitment_info = self._execute_commitment(commitment_action)
            info.update(commitment_info)
        else:
            # Agent abstained
            reward = -self.config.abstain_penalty
            info["outcome"] = "abstained"
            
            # Small bonus for correctly abstaining on impossible tasks
            if self.current_task.difficulty == TaskDifficulty.IMPOSSIBLE:
                reward += 0.1
                info["correct_abstain"] = True
        
        # Generate next task
        self.current_task = self.task_generator.generate()[0]
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.config.max_steps
        
        # Check if agent is bankrupt
        account = self.stake_manager.get_account(self.agent_id)
        if account and account.available_stake < self.config.min_stake_per_commitment:
            terminated = True
            info["termination_reason"] = "bankrupt"
        
        obs = self._get_observation()
        info.update(self._get_info())
        
        return obs.to_array(), reward, terminated, truncated, info
    
    def _execute_commitment(
        self,
        action: CommitmentAction,
    ) -> tuple[float, dict[str, Any]]:
        """
        Execute a commitment and return reward.
        
        Args:
            action: The commitment action.
            
        Returns:
            Tuple of (reward, info_dict).
        """
        assert self.current_task is not None
        
        info: dict[str, Any] = {}
        
        # Calculate stake amount
        account = self.stake_manager.get_account(self.agent_id)
        available = account.available_stake if account else 0.0
        stake_amount = min(
            available * action.stake_fraction,
            self.config.max_stake_per_commitment,
        )
        stake_amount = max(stake_amount, self.config.min_stake_per_commitment)
        
        if stake_amount > available:
            # Can't afford commitment
            info["outcome"] = "insufficient_stake"
            return -0.1, info
        
        # Create commitment
        commitment = self._create_commitment(action, stake_amount)
        
        # Lock stake
        try:
            self.stake_manager.lock_stake(self.agent_id, commitment)
        except ValueError:
            info["outcome"] = "stake_lock_failed"
            return -0.1, info
        
        # Simulate task execution
        success = self.rng.random() < self.current_task.true_success_probability
        
        # Create verification result
        if success:
            result = VerificationResult(
                status=VerificationStatus.SUCCESS,
                commitment_id=commitment.id,
                details={"accuracy": 0.9},
            )
            self.stake_manager.unlock_stake(commitment)
        else:
            result = VerificationResult(
                status=VerificationStatus.FAILURE,
                commitment_id=commitment.id,
                triggered_failure_mode="task_failed",
                details={"severity": 0.5, "accuracy": 0.3},
            )
            self.stake_manager.slash_stake(commitment, result)
        
        # Update reputation
        self.reputation_tracker.record_fulfillment(self.agent_id, commitment, result)
        
        # Store in history
        self.commitment_history.append((commitment, result))
        
        # Calculate reward
        reward = self._calculate_reward(commitment, result, action)
        
        info["outcome"] = "success" if success else "failure"
        info["stake_amount"] = stake_amount
        info["reward"] = reward
        
        return reward, info
    
    def _create_commitment(
        self,
        action: CommitmentAction,
        stake_amount: float,
    ) -> GroundedCommitment:
        """Create a commitment for the current task."""
        assert self.current_task is not None
        
        return GroundedCommitment(
            issuer=self.agent_id,
            trigger_conditions=[
                Predicate(name="task_match", expression=f"task_id == '{self.current_task.id}'")
            ],
            promised_behavior=ActionSpec(
                action_type="complete_task",
                parameters={"task_id": self.current_task.id},
            ),
            success_condition=Predicate(name="task_success", expression="accuracy > 0.5"),
            failure_modes=[
                FailureMode(
                    name="task_failed",
                    condition=Predicate(name="low_accuracy", expression="accuracy <= 0.5"),
                    consequence=Consequence(
                        consequence_type="stake_slash",
                        magnitude=0.5,
                        description="Task completion failed",
                    ),
                    severity=0.5,
                )
            ],
            stake=stake_amount,
            confidence=action.confidence,
            valid_contexts=ContextRegion(
                description="Task context",
                tags=[self.current_task.difficulty.value],
            ),
            metadata={
                "task_id": self.current_task.id,
                "task_difficulty": self.current_task.difficulty.value,
            },
        )
    
    def _calculate_reward(
        self,
        commitment: GroundedCommitment,
        result: VerificationResult,
        action: CommitmentAction,
    ) -> float:
        """
        Calculate reward for a commitment outcome.
        
        Reward structure:
        - Base reward/penalty for success/failure
        - Calibration bonus for well-calibrated confidence
        - Stake-weighted rewards
        """
        assert self.current_task is not None
        
        reward = 0.0
        
        if result.status == VerificationStatus.SUCCESS:
            # Success reward
            base_reward = self.current_task.reward_on_success
            reward = base_reward * self.config.success_reward_multiplier
            
            # Stake bonus (higher stake = higher reward)
            stake_bonus = commitment.stake * 0.01
            reward += stake_bonus
        else:
            # Failure penalty
            base_penalty = self.current_task.penalty_on_failure
            reward = -base_penalty * self.config.failure_penalty_multiplier
            
            # Stake penalty (higher stake = higher loss)
            stake_penalty = commitment.stake * 0.02
            reward -= stake_penalty
        
        # Calibration bonus
        # Reward confidence that matches actual success probability
        true_prob = self.current_task.true_success_probability
        confidence_error = abs(action.confidence - true_prob)
        calibration_bonus = (1.0 - confidence_error) * self.config.calibration_bonus
        reward += calibration_bonus
        
        return reward
    
    def _get_observation(self) -> CommitmentObservation:
        """Get current observation."""
        assert self.current_task is not None
        
        # Get reputation
        rep = self.reputation_tracker.get_reputation(self.agent_id)
        reputation_score = rep.score if rep else self.config.initial_reputation
        
        # Get stake
        account = self.stake_manager.get_account(self.agent_id)
        available_stake = account.available_stake if account else 0.0
        
        # Calculate recent success rate
        recent_history = self.commitment_history[-20:]  # Last 20 commitments
        if recent_history:
            successes = sum(
                1 for _, r in recent_history
                if r and r.status == VerificationStatus.SUCCESS
            )
            recent_success_rate = successes / len(recent_history)
        else:
            recent_success_rate = 0.5  # Default
        
        return CommitmentObservation(
            task_features=self.current_task.features,
            reputation_score=reputation_score,
            available_stake=available_stake / self.config.initial_stake,  # Normalize
            active_commitment_count=0,  # Simplified: no concurrent commitments
            recent_success_rate=recent_success_rate,
        )
    
    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        rep = self.reputation_tracker.get_reputation(self.agent_id)
        account = self.stake_manager.get_account(self.agent_id)
        
        return {
            "step": self.step_count,
            "reputation": rep.score if rep else 0.0,
            "available_stake": account.available_stake if account else 0.0,
            "total_commitments": len(self.commitment_history),
            "success_count": sum(
                1 for _, r in self.commitment_history
                if r and r.status == VerificationStatus.SUCCESS
            ),
        }
    
    def render(self) -> None:
        """Render the environment state."""
        if self.current_task is None:
            print("Environment not initialized. Call reset() first.")
            return
        
        info = self._get_info()
        print(f"\n=== Step {self.step_count} ===")
        print(f"Task: {self.current_task.difficulty.value} (p={self.current_task.true_success_probability:.2f})")
        print(f"Reputation: {info['reputation']:.2f}")
        print(f"Available Stake: {info['available_stake']:.2f}")
        print(f"Success Rate: {info['success_count']}/{info['total_commitments']}")
