"""
Training utilities for commitment learning.

This module provides PPO-based training for commitment policies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gcl.learning.environment import CommitmentEnv, CommitmentEnvConfig
from gcl.learning.policy import CommitmentPolicy, SimpleCommitmentPolicy
from gcl.learning.reward import CalibrationTracker


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Training
    total_timesteps: int = 100_000
    n_steps: int = 2048  # Steps per rollout
    batch_size: int = 64
    n_epochs: int = 10  # Epochs per update
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip range
    clip_range_vf: float | None = None  # Value function clip range
    
    # Loss coefficients
    vf_coef: float = 0.5  # Value function coefficient
    ent_coef: float = 0.01  # Entropy coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Policy architecture
    hidden_dim: int = 64
    num_layers: int = 2
    
    # Logging
    log_interval: int = 1  # Log every N updates
    save_interval: int = 10  # Save every N updates
    eval_interval: int = 5  # Evaluate every N updates
    eval_episodes: int = 10  # Episodes for evaluation


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    
    # Computed after rollout
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute returns and advantages using GAE."""
        n = len(self.rewards)
        
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = (
                self.rewards[t] +
                gamma * next_value * next_non_terminal -
                self.values[t]
            )
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )
        
        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)
    
    def get_tensors(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert buffer to tensors."""
        obs = torch.tensor(np.array(self.observations), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=device)
        
        return obs, actions, old_log_probs, advantages, returns
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
    
    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    
    timesteps: int = 0
    episodes: int = 0
    updates: int = 0
    
    # Episode metrics
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    
    # Training metrics
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropy_losses: list[float] = field(default_factory=list)
    
    # Commitment metrics
    commit_rates: list[float] = field(default_factory=list)
    success_rates: list[float] = field(default_factory=list)
    avg_confidences: list[float] = field(default_factory=list)
    
    def get_recent_stats(self, n: int = 100) -> dict[str, float]:
        """Get statistics over recent episodes."""
        return {
            "mean_reward": np.mean(self.episode_rewards[-n:]) if self.episode_rewards else 0.0,
            "mean_length": np.mean(self.episode_lengths[-n:]) if self.episode_lengths else 0.0,
            "mean_policy_loss": np.mean(self.policy_losses[-n:]) if self.policy_losses else 0.0,
            "mean_value_loss": np.mean(self.value_losses[-n:]) if self.value_losses else 0.0,
            "mean_commit_rate": np.mean(self.commit_rates[-n:]) if self.commit_rates else 0.0,
            "mean_success_rate": np.mean(self.success_rates[-n:]) if self.success_rates else 0.0,
        }


class PPOTrainer:
    """
    PPO trainer for commitment policies.
    
    Implements Proximal Policy Optimization for training agents
    to make well-calibrated commitment decisions.
    
    Example:
        >>> env = CommitmentEnv()
        >>> trainer = PPOTrainer(env)
        >>> metrics = trainer.train()
    """
    
    def __init__(
        self,
        env: CommitmentEnv,
        config: PPOConfig | None = None,
        device: str | torch.device = "auto",
        seed: int | None = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            env: The commitment environment.
            config: Training configuration.
            device: Device to use ("auto", "cpu", "cuda").
            seed: Random seed.
        """
        self.env = env
        self.config = config or PPOConfig()
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create policy
        obs_dim = env.observation_space.shape[0]
        self.policy = CommitmentPolicy(
            obs_dim=obs_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Buffers and metrics
        self.buffer = RolloutBuffer()
        self.metrics = TrainingMetrics()
        self.calibration_tracker = CalibrationTracker()
    
    def train(
        self,
        callback: Any | None = None,
    ) -> TrainingMetrics:
        """
        Train the policy.
        
        Args:
            callback: Optional callback function called after each update.
            
        Returns:
            Training metrics.
        """
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_commits = 0
        episode_successes = 0
        episode_confidences = []
        
        start_time = time.time()
        
        while self.metrics.timesteps < self.config.total_timesteps:
            # Collect rollout
            for _ in range(self.config.n_steps):
                # Get action
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.policy.get_action(obs_tensor)
                
                action = action[0]  # Remove batch dimension
                log_prob = log_prob.item()
                value = value.item()
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.buffer.add(obs, action, reward, value, log_prob, done)
                
                # Track episode metrics
                episode_reward += reward
                episode_length += 1
                
                if action[0] > 0.5:  # Committed
                    episode_commits += 1
                    episode_confidences.append(action[1])
                    if info.get("outcome") == "success":
                        episode_successes += 1
                        self.calibration_tracker.record(action[1], True)
                    elif info.get("outcome") == "failure":
                        self.calibration_tracker.record(action[1], False)
                
                self.metrics.timesteps += 1
                obs = next_obs
                
                if done:
                    # Record episode metrics
                    self.metrics.episodes += 1
                    self.metrics.episode_rewards.append(episode_reward)
                    self.metrics.episode_lengths.append(episode_length)
                    
                    if episode_commits > 0:
                        self.metrics.commit_rates.append(episode_commits / episode_length)
                        self.metrics.success_rates.append(episode_successes / episode_commits)
                        self.metrics.avg_confidences.append(np.mean(episode_confidences))
                    
                    # Reset
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                    episode_commits = 0
                    episode_successes = 0
                    episode_confidences = []
            
            # Compute returns and advantages
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value = self.policy.get_value(obs_tensor).item()
            
            self.buffer.compute_returns_and_advantages(
                last_value,
                self.config.gamma,
                self.config.gae_lambda,
            )
            
            # Update policy
            update_metrics = self._update()
            self.metrics.updates += 1
            self.metrics.policy_losses.append(update_metrics["policy_loss"])
            self.metrics.value_losses.append(update_metrics["value_loss"])
            self.metrics.entropy_losses.append(update_metrics["entropy_loss"])
            
            # Clear buffer
            self.buffer.clear()
            
            # Logging
            if self.metrics.updates % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.metrics.timesteps / elapsed
                stats = self.metrics.get_recent_stats()
                
                print(f"\n=== Update {self.metrics.updates} ===")
                print(f"Timesteps: {self.metrics.timesteps}/{self.config.total_timesteps}")
                print(f"Episodes: {self.metrics.episodes}")
                print(f"FPS: {fps:.0f}")
                print(f"Mean Reward: {stats['mean_reward']:.3f}")
                print(f"Mean Length: {stats['mean_length']:.1f}")
                print(f"Commit Rate: {stats['mean_commit_rate']:.2%}")
                print(f"Success Rate: {stats['mean_success_rate']:.2%}")
                print(f"Policy Loss: {stats['mean_policy_loss']:.4f}")
                print(f"Value Loss: {stats['mean_value_loss']:.4f}")
                
                # Calibration
                cal_stats = self.calibration_tracker.get_statistics()
                print(f"Calibration Error: {cal_stats['expected_calibration_error']:.3f}")
            
            # Callback
            if callback is not None:
                callback(self.metrics)
        
        return self.metrics
    
    def _update(self) -> dict[str, float]:
        """Perform PPO update."""
        obs, actions, old_log_probs, advantages, returns = self.buffer.get_tensors(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset
        dataset = TensorDataset(obs, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_batches = 0
        
        for _ in range(self.config.n_epochs):
            for batch in dataloader:
                batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch
                
                # Evaluate actions
                log_probs, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions)
                values = values.squeeze(-1)
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = batch_returns + torch.clamp(
                        values - batch_returns,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    value_loss = torch.max(
                        (values - batch_returns).pow(2),
                        (values_clipped - batch_returns).pow(2),
                    ).mean()
                else:
                    value_loss = (values - batch_returns).pow(2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.vf_coef * value_loss +
                    self.config.ent_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_batches += 1
        
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy_loss": total_entropy_loss / n_batches,
        }
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            n_episodes: Number of episodes to evaluate.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Evaluation metrics.
        """
        rewards = []
        lengths = []
        commit_rates = []
        success_rates = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            commits = 0
            successes = 0
            done = False
            
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = self.policy.get_action(obs_tensor, deterministic=deterministic)
                action = action[0]
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if action[0] > 0.5:
                    commits += 1
                    if info.get("outcome") == "success":
                        successes += 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            if commits > 0:
                commit_rates.append(commits / episode_length)
                success_rates.append(successes / commits)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "mean_commit_rate": np.mean(commit_rates) if commit_rates else 0.0,
            "mean_success_rate": np.mean(success_rates) if success_rates else 0.0,
        }
    
    def save(self, path: str | Path) -> None:
        """Save the policy."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "metrics": self.metrics,
        }, path)
    
    def load(self, path: str | Path) -> None:
        """Load a saved policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics = checkpoint.get("metrics", TrainingMetrics())


def train_commitment_agent(
    total_timesteps: int = 50_000,
    seed: int = 42,
    device: str = "auto",
) -> tuple[CommitmentPolicy, TrainingMetrics]:
    """
    Convenience function to train a commitment agent.
    
    Args:
        total_timesteps: Total training timesteps.
        seed: Random seed.
        device: Device to use.
        
    Returns:
        Tuple of (trained_policy, training_metrics).
    """
    # Create environment
    env_config = CommitmentEnvConfig(
        max_steps=100,
        initial_stake=100.0,
    )
    env = CommitmentEnv(config=env_config, seed=seed)
    
    # Create trainer
    ppo_config = PPOConfig(
        total_timesteps=total_timesteps,
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
    )
    trainer = PPOTrainer(env, config=ppo_config, device=device, seed=seed)
    
    # Train
    metrics = trainer.train()
    
    return trainer.policy, metrics
