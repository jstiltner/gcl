"""
Neural network policy for commitment learning.

This module provides policy networks that learn to make commitment decisions
based on task features and agent state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Bernoulli


class CommitmentPolicy(nn.Module):
    """
    Neural network policy for commitment decisions.
    
    The policy takes an observation and outputs:
    1. Commit probability (whether to make a commitment)
    2. Confidence distribution (Beta distribution parameters)
    3. Stake fraction distribution (Beta distribution parameters)
    
    Architecture:
        - Shared feature extractor (MLP)
        - Separate heads for commit, confidence, and stake
        - Value head for critic (actor-critic methods)
    
    Example:
        >>> policy = CommitmentPolicy(obs_dim=12, hidden_dim=64)
        >>> obs = torch.randn(1, 12)
        >>> action, log_prob, value = policy.get_action(obs)
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "tanh",
    ) -> None:
        """
        Initialize the policy network.
        
        Args:
            obs_dim: Dimension of observation space.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            activation: Activation function ("tanh", "relu", "elu").
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation)
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        # Commit head (binary decision)
        self.commit_head = nn.Linear(hidden_dim, 1)
        
        # Confidence head (Beta distribution: alpha, beta)
        self.confidence_head = nn.Linear(hidden_dim, 2)
        
        # Stake head (Beta distribution: alpha, beta)
        self.stake_head = nn.Linear(hidden_dim, 2)
        
        # Value head (for actor-critic)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller initialization for output heads
        nn.init.orthogonal_(self.commit_head.weight, gain=0.01)
        nn.init.orthogonal_(self.confidence_head.weight, gain=0.01)
        nn.init.orthogonal_(self.stake_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).
            
        Returns:
            Tuple of (commit_logits, confidence_params, stake_params, value).
        """
        features = self.feature_extractor(obs)
        
        # Commit logits
        commit_logits = self.commit_head(features)
        
        # Confidence Beta parameters (ensure positive with softplus)
        confidence_raw = self.confidence_head(features)
        confidence_params = F.softplus(confidence_raw) + 1.0  # alpha, beta >= 1
        
        # Stake Beta parameters
        stake_raw = self.stake_head(features)
        stake_params = F.softplus(stake_raw) + 1.0
        
        # Value estimate
        value = self.value_head(features)
        
        return commit_logits, confidence_params, stake_params, value
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            obs: Observation tensor.
            deterministic: If True, return mean action instead of sampling.
            
        Returns:
            Tuple of (action_array, log_prob, value).
        """
        commit_logits, confidence_params, stake_params, value = self.forward(obs)
        
        # Commit decision
        commit_prob = torch.sigmoid(commit_logits)
        if deterministic:
            commit = (commit_prob > 0.5).float()
        else:
            commit_dist = Bernoulli(probs=commit_prob)
            commit = commit_dist.sample()
        
        # Confidence (Beta distribution)
        alpha_c, beta_c = confidence_params[:, 0], confidence_params[:, 1]
        confidence_dist = Beta(alpha_c, beta_c)
        if deterministic:
            confidence = confidence_dist.mean
        else:
            confidence = confidence_dist.sample()
        
        # Stake fraction (Beta distribution)
        alpha_s, beta_s = stake_params[:, 0], stake_params[:, 1]
        stake_dist = Beta(alpha_s, beta_s)
        if deterministic:
            stake = stake_dist.mean
        else:
            stake = stake_dist.sample()
        
        # Compute log probability
        log_prob = self._compute_log_prob(
            commit, confidence, stake,
            commit_logits, confidence_params, stake_params,
        )
        
        # Create action array
        action = torch.stack([
            commit.squeeze(-1),
            confidence,
            stake,
        ], dim=-1)
        
        return action.detach().cpu().numpy(), log_prob, value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: Observation tensor.
            actions: Action tensor of shape (batch_size, 3).
            
        Returns:
            Tuple of (log_prob, entropy, value).
        """
        commit_logits, confidence_params, stake_params, value = self.forward(obs)
        
        # Extract actions
        commit = actions[:, 0:1]
        confidence = actions[:, 1]
        stake = actions[:, 2]
        
        # Compute log probability
        log_prob = self._compute_log_prob(
            commit, confidence, stake,
            commit_logits, confidence_params, stake_params,
        )
        
        # Compute entropy
        entropy = self._compute_entropy(
            commit_logits, confidence_params, stake_params,
        )
        
        return log_prob, entropy, value
    
    def _compute_log_prob(
        self,
        commit: torch.Tensor,
        confidence: torch.Tensor,
        stake: torch.Tensor,
        commit_logits: torch.Tensor,
        confidence_params: torch.Tensor,
        stake_params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of actions."""
        # Commit log prob - round to binary for Bernoulli distribution
        commit_prob = torch.sigmoid(commit_logits)
        commit_dist = Bernoulli(probs=commit_prob)
        commit_binary = torch.round(commit).clamp(0, 1)  # Ensure binary values
        commit_log_prob = commit_dist.log_prob(commit_binary).squeeze(-1)
        
        # Confidence log prob
        alpha_c, beta_c = confidence_params[:, 0], confidence_params[:, 1]
        confidence_dist = Beta(alpha_c, beta_c)
        # Clamp confidence to valid range for Beta distribution
        confidence_clamped = torch.clamp(confidence, 1e-6, 1 - 1e-6)
        confidence_log_prob = confidence_dist.log_prob(confidence_clamped)
        
        # Stake log prob
        alpha_s, beta_s = stake_params[:, 0], stake_params[:, 1]
        stake_dist = Beta(alpha_s, beta_s)
        stake_clamped = torch.clamp(stake, 1e-6, 1 - 1e-6)
        stake_log_prob = stake_dist.log_prob(stake_clamped)
        
        # Total log prob (sum of independent components)
        total_log_prob = commit_log_prob + confidence_log_prob + stake_log_prob
        
        return total_log_prob
    
    def _compute_entropy(
        self,
        commit_logits: torch.Tensor,
        confidence_params: torch.Tensor,
        stake_params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entropy of the policy."""
        # Commit entropy
        commit_prob = torch.sigmoid(commit_logits)
        commit_dist = Bernoulli(probs=commit_prob)
        commit_entropy = commit_dist.entropy().squeeze(-1)
        
        # Confidence entropy
        alpha_c, beta_c = confidence_params[:, 0], confidence_params[:, 1]
        confidence_dist = Beta(alpha_c, beta_c)
        confidence_entropy = confidence_dist.entropy()
        
        # Stake entropy
        alpha_s, beta_s = stake_params[:, 0], stake_params[:, 1]
        stake_dist = Beta(alpha_s, beta_s)
        stake_entropy = stake_dist.entropy()
        
        # Total entropy
        total_entropy = commit_entropy + confidence_entropy + stake_entropy
        
        return total_entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations."""
        features = self.feature_extractor(obs)
        return self.value_head(features)


class SimpleCommitmentPolicy(nn.Module):
    """
    Simplified policy that outputs continuous actions directly.
    
    This is a simpler alternative to CommitmentPolicy that uses
    Gaussian distributions instead of Beta distributions.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        """
        Initialize the policy.
        
        Args:
            obs_dim: Observation dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
        """
        super().__init__()
        
        # Feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        
        # Action mean
        self.action_mean = nn.Linear(hidden_dim, 3)
        
        # Action log std (learnable parameter)
        self.action_log_std = nn.Parameter(torch.zeros(3))
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.features(obs)
        action_mean = self.action_mean(features)
        
        # Apply sigmoid to constrain to [0, 1]
        action_mean = torch.sigmoid(action_mean)
        
        value = self.value_head(features)
        
        return action_mean, self.action_log_std.exp(), value
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
        else:
            # Sample from Gaussian and clip
            noise = torch.randn_like(action_mean) * action_std
            action = torch.clamp(action_mean + noise, 0.0, 1.0)
        
        # Compute log prob (approximate, since we clip)
        log_prob = -0.5 * ((action - action_mean) / action_std).pow(2).sum(dim=-1)
        
        return action.detach().cpu().numpy(), log_prob, value


class RandomPolicy:
    """
    Random baseline policy for comparison.
    
    Samples actions uniformly at random.
    """
    
    def __init__(self, seed: int | None = None) -> None:
        """Initialize random policy."""
        self.rng = np.random.default_rng(seed)
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Get random action."""
        action = self.rng.random(3).astype(np.float32)
        return action, 0.0, 0.0  # log_prob and value not meaningful


class HeuristicPolicy:
    """
    Heuristic baseline policy.
    
    Uses simple rules based on task features to make decisions.
    """
    
    def __init__(
        self,
        difficulty_threshold: float = 0.5,
        base_confidence: float = 0.6,
        base_stake: float = 0.1,
    ) -> None:
        """
        Initialize heuristic policy.
        
        Args:
            difficulty_threshold: Threshold for commit decision.
            base_confidence: Base confidence level.
            base_stake: Base stake fraction.
        """
        self.difficulty_threshold = difficulty_threshold
        self.base_confidence = base_confidence
        self.base_stake = base_stake
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """
        Get action based on heuristics.
        
        Uses the first feature (difficulty indicator) to make decisions.
        """
        # Extract difficulty indicator (first feature)
        if len(obs.shape) == 1:
            difficulty = obs[0]
        else:
            difficulty = obs[0, 0]
        
        # Commit if difficulty is below threshold
        commit = 1.0 if difficulty < self.difficulty_threshold else 0.0
        
        # Confidence inversely related to difficulty
        confidence = max(0.1, min(0.95, self.base_confidence - difficulty * 0.3))
        
        # Stake based on confidence
        stake = self.base_stake * confidence
        
        action = np.array([commit, confidence, stake], dtype=np.float32)
        return action, 0.0, 0.0
