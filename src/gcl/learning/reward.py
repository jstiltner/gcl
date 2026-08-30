"""
Reward computation for commitment learning.

This module provides configurable reward functions that incentivize:
- Successful commitment fulfillment
- Well-calibrated confidence estimates
- Appropriate stake management
- Strategic abstention on impossible tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from gcl.core.commitment import GroundedCommitment, VerificationResult, VerificationStatus


class RewardConfig(BaseModel):
    """Configuration for reward computation."""
    
    # Base rewards/penalties
    success_base_reward: float = Field(default=1.0, description="Base reward for successful commitment")
    failure_base_penalty: float = Field(default=0.5, description="Base penalty for failed commitment")
    abstain_penalty: float = Field(default=0.01, description="Small penalty for not committing")
    
    # Stake-based modifiers
    stake_success_multiplier: float = Field(default=0.1, description="Bonus per unit stake on success")
    stake_failure_multiplier: float = Field(default=0.2, description="Penalty per unit stake on failure")
    
    # Confidence calibration
    calibration_weight: float = Field(default=0.2, description="Weight for calibration bonus/penalty")
    overconfidence_penalty: float = Field(default=0.3, description="Extra penalty for overconfident failures")
    underconfidence_penalty: float = Field(default=0.1, description="Penalty for underconfident successes")
    
    # Reputation-based modifiers
    reputation_weight: float = Field(default=0.1, description="Weight for reputation changes in reward")
    
    # Strategic bonuses
    correct_abstain_bonus: float = Field(default=0.1, description="Bonus for correctly abstaining on impossible tasks")
    high_confidence_success_bonus: float = Field(default=0.2, description="Bonus for high-confidence successes")
    
    # Risk management
    bankruptcy_penalty: float = Field(default=5.0, description="Large penalty for going bankrupt")
    conservative_bonus: float = Field(default=0.05, description="Small bonus for conservative stake management")


@dataclass
class RewardComponents:
    """
    Breakdown of reward components.
    
    Useful for debugging and understanding agent behavior.
    """
    
    base_reward: float = 0.0
    stake_reward: float = 0.0
    calibration_reward: float = 0.0
    reputation_reward: float = 0.0
    strategic_reward: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "base_reward": self.base_reward,
            "stake_reward": self.stake_reward,
            "calibration_reward": self.calibration_reward,
            "reputation_reward": self.reputation_reward,
            "strategic_reward": self.strategic_reward,
            "total": self.total,
        }


class RewardComputer:
    """
    Computes rewards for commitment outcomes.
    
    The reward function is designed to encourage:
    1. Making commitments when likely to succeed
    2. Calibrating confidence to match actual success probability
    3. Managing stake appropriately (higher stake for confident commitments)
    4. Abstaining from impossible tasks
    
    Example:
        >>> computer = RewardComputer()
        >>> reward = computer.compute_commitment_reward(
        ...     commitment=commitment,
        ...     result=result,
        ...     true_success_prob=0.8,
        ...     confidence=0.75,
        ...     reputation_delta=0.1,
        ... )
        >>> print(f"Total reward: {reward.total}")
    """
    
    def __init__(self, config: RewardConfig | None = None) -> None:
        """
        Initialize the reward computer.
        
        Args:
            config: Reward configuration.
        """
        self.config = config or RewardConfig()
    
    def compute_commitment_reward(
        self,
        commitment: GroundedCommitment,
        result: VerificationResult,
        true_success_prob: float,
        confidence: float,
        reputation_delta: float = 0.0,
        task_difficulty: str | None = None,
    ) -> RewardComponents:
        """
        Compute reward for a commitment outcome.
        
        Args:
            commitment: The commitment that was made.
            result: The verification result.
            true_success_prob: Actual success probability of the task.
            confidence: Confidence level expressed by the agent.
            reputation_delta: Change in reputation from this commitment.
            task_difficulty: Optional task difficulty label.
            
        Returns:
            RewardComponents with breakdown of reward.
        """
        components = RewardComponents()
        
        success = result.status == VerificationStatus.SUCCESS
        
        # 1. Base reward/penalty
        if success:
            components.base_reward = self.config.success_base_reward
        else:
            components.base_reward = -self.config.failure_base_penalty
        
        # 2. Stake-based reward
        stake = commitment.stake
        if success:
            components.stake_reward = stake * self.config.stake_success_multiplier
        else:
            components.stake_reward = -stake * self.config.stake_failure_multiplier
        
        # 3. Calibration reward
        components.calibration_reward = self._compute_calibration_reward(
            success=success,
            confidence=confidence,
            true_success_prob=true_success_prob,
        )
        
        # 4. Reputation reward
        components.reputation_reward = reputation_delta * self.config.reputation_weight
        
        # 5. Strategic rewards
        components.strategic_reward = self._compute_strategic_reward(
            success=success,
            confidence=confidence,
            task_difficulty=task_difficulty,
        )
        
        # Total
        components.total = (
            components.base_reward +
            components.stake_reward +
            components.calibration_reward +
            components.reputation_reward +
            components.strategic_reward
        )
        
        return components
    
    def compute_abstain_reward(
        self,
        true_success_prob: float,
        task_difficulty: str | None = None,
    ) -> RewardComponents:
        """
        Compute reward for abstaining from a task.
        
        Args:
            true_success_prob: Actual success probability of the task.
            task_difficulty: Optional task difficulty label.
            
        Returns:
            RewardComponents with breakdown of reward.
        """
        components = RewardComponents()
        
        # Base abstain penalty
        components.base_reward = -self.config.abstain_penalty
        
        # Bonus for correctly abstaining on very hard/impossible tasks
        if true_success_prob < 0.2 or task_difficulty == "impossible":
            components.strategic_reward = self.config.correct_abstain_bonus
        elif true_success_prob > 0.8:
            # Penalty for abstaining on easy tasks
            components.strategic_reward = -self.config.abstain_penalty * 2
        
        components.total = components.base_reward + components.strategic_reward
        
        return components
    
    def compute_bankruptcy_penalty(self) -> float:
        """Compute penalty for going bankrupt."""
        return -self.config.bankruptcy_penalty
    
    def _compute_calibration_reward(
        self,
        success: bool,
        confidence: float,
        true_success_prob: float,
    ) -> float:
        """
        Compute calibration reward/penalty.
        
        Rewards confidence that matches actual success probability.
        Penalizes overconfidence on failures and underconfidence on successes.
        """
        # Calibration error
        calibration_error = abs(confidence - true_success_prob)
        
        # Base calibration reward (lower error = higher reward)
        calibration_reward = (1.0 - calibration_error) * self.config.calibration_weight
        
        # Additional penalties for miscalibration
        if not success and confidence > true_success_prob + 0.2:
            # Overconfident failure
            overconfidence = confidence - true_success_prob
            calibration_reward -= overconfidence * self.config.overconfidence_penalty
        
        if success and confidence < true_success_prob - 0.2:
            # Underconfident success
            underconfidence = true_success_prob - confidence
            calibration_reward -= underconfidence * self.config.underconfidence_penalty
        
        return calibration_reward
    
    def _compute_strategic_reward(
        self,
        success: bool,
        confidence: float,
        task_difficulty: str | None,
    ) -> float:
        """Compute strategic bonuses."""
        reward = 0.0
        
        # Bonus for high-confidence successes
        if success and confidence > 0.8:
            reward += self.config.high_confidence_success_bonus
        
        # Bonus for succeeding on hard tasks
        if success and task_difficulty == "hard":
            reward += 0.1
        
        return reward


class CalibrationTracker:
    """
    Tracks calibration metrics over time.
    
    Useful for evaluating how well an agent's confidence
    matches actual success rates.
    """
    
    def __init__(self, num_bins: int = 10) -> None:
        """
        Initialize the calibration tracker.
        
        Args:
            num_bins: Number of confidence bins for calibration analysis.
        """
        self.num_bins = num_bins
        self.bins: list[list[bool]] = [[] for _ in range(num_bins)]
    
    def record(self, confidence: float, success: bool) -> None:
        """
        Record a commitment outcome.
        
        Args:
            confidence: Confidence level (0-1).
            success: Whether the commitment succeeded.
        """
        bin_idx = min(int(confidence * self.num_bins), self.num_bins - 1)
        self.bins[bin_idx].append(success)
    
    def get_calibration_error(self) -> float:
        """
        Compute expected calibration error (ECE).
        
        Returns:
            Expected calibration error (lower is better).
        """
        total_samples = sum(len(b) for b in self.bins)
        if total_samples == 0:
            return 0.0
        
        ece = 0.0
        for i, bin_outcomes in enumerate(self.bins):
            if not bin_outcomes:
                continue
            
            # Expected confidence for this bin
            expected_conf = (i + 0.5) / self.num_bins
            
            # Actual success rate
            actual_rate = sum(bin_outcomes) / len(bin_outcomes)
            
            # Weighted absolute difference
            weight = len(bin_outcomes) / total_samples
            ece += weight * abs(expected_conf - actual_rate)
        
        return ece
    
    def get_calibration_curve(self) -> list[tuple[float, float, int]]:
        """
        Get calibration curve data.
        
        Returns:
            List of (expected_confidence, actual_success_rate, count) tuples.
        """
        curve = []
        for i, bin_outcomes in enumerate(self.bins):
            expected_conf = (i + 0.5) / self.num_bins
            if bin_outcomes:
                actual_rate = sum(bin_outcomes) / len(bin_outcomes)
                count = len(bin_outcomes)
            else:
                actual_rate = 0.0
                count = 0
            curve.append((expected_conf, actual_rate, count))
        return curve
    
    def get_statistics(self) -> dict[str, Any]:
        """Get calibration statistics."""
        total = sum(len(b) for b in self.bins)
        successes = sum(sum(b) for b in self.bins)
        
        return {
            "total_samples": total,
            "overall_success_rate": successes / total if total > 0 else 0.0,
            "expected_calibration_error": self.get_calibration_error(),
            "calibration_curve": self.get_calibration_curve(),
        }
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.bins = [[] for _ in range(self.num_bins)]
