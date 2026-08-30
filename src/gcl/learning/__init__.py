"""
GCL Learning Module

Contains reinforcement learning components for commitment-based agents.
"""

from gcl.learning.environment import (
    CommitmentEnv,
    CommitmentEnvConfig,
    CommitmentObservation,
    CommitmentAction,
    Task,
    TaskDifficulty,
    TaskGenerator,
)
from gcl.learning.reward import (
    CalibrationTracker,
    RewardComponents,
    RewardConfig,
    RewardComputer,
)
from gcl.learning.policy import (
    CommitmentPolicy,
    SimpleCommitmentPolicy,
    RandomPolicy,
    HeuristicPolicy,
)
from gcl.learning.training import (
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    TrainingMetrics,
    train_commitment_agent,
)

__all__ = [
    # Environment
    "CommitmentEnv",
    "CommitmentEnvConfig",
    "CommitmentObservation",
    "CommitmentAction",
    "Task",
    "TaskDifficulty",
    "TaskGenerator",
    # Reward
    "CalibrationTracker",
    "RewardComponents",
    "RewardConfig",
    "RewardComputer",
    # Policy
    "CommitmentPolicy",
    "SimpleCommitmentPolicy",
    "RandomPolicy",
    "HeuristicPolicy",
    # Training
    "PPOConfig",
    "PPOTrainer",
    "RolloutBuffer",
    "TrainingMetrics",
    "train_commitment_agent",
]
