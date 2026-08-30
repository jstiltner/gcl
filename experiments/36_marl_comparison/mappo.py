"""
MAPPO Implementation for GCL Comparison

MAPPO (Yu et al., 2021) is Multi-Agent PPO with centralized value function.
Key idea: Each agent has its own policy, but shares a centralized critic.

This implementation is simplified but captures the essential mechanisms:
1. Individual policy networks (actor) for each agent
2. Centralized value network (critic) that sees global state
3. PPO clipping for stable policy updates
4. Generalized Advantage Estimation (GAE)

For fair comparison with GCL:
- Same task environment (difficulty-based success)
- Same agent capabilities
- Measure sample efficiency and asymptotic performance
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import random


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO."""
    n_agents: int = 30
    state_dim: int = 10  # Global state dimension
    obs_dim: int = 5     # Individual observation dimension
    action_dim: int = 2  # Accept/reject task
    hidden_dim: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 32
    rollout_length: int = 128


class PolicyNetwork:
    """Policy network (actor) using numpy."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 0.0003):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning action probabilities."""
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Sample action and return log probability."""
        probs = self.forward(obs)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = np.random.choice(self.output_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob
    
    def get_log_prob(self, obs: np.ndarray, action: int) -> float:
        """Get log probability of action."""
        probs = self.forward(obs)
        return np.log(probs[action] + 1e-8)
    
    def get_entropy(self, obs: np.ndarray) -> float:
        """Get entropy of action distribution."""
        probs = self.forward(obs)
        return -np.sum(probs * np.log(probs + 1e-8))
    
    def update(self, obs: np.ndarray, action: int, advantage: float, old_log_prob: float, 
               clip_epsilon: float, entropy_coef: float):
        """PPO update with clipping."""
        # Forward pass
        h = np.tanh(obs @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        new_log_prob = np.log(probs[action] + 1e-8)
        ratio = np.exp(new_log_prob - old_log_prob)
        
        # Clipped objective
        surr1 = ratio * advantage
        surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
        policy_loss = -min(surr1, surr2)
        
        # Entropy bonus
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        # Total loss
        loss = policy_loss - entropy_coef * entropy
        
        # Compute gradients (simplified)
        # Gradient of log_prob w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[action] -= 1.0
        
        # Scale by loss gradient
        if surr1 < surr2:
            grad_scale = -advantage * ratio
        else:
            if ratio < 1 - clip_epsilon or ratio > 1 + clip_epsilon:
                grad_scale = 0.0
            else:
                grad_scale = -advantage * ratio
        
        grad_logits *= grad_scale
        
        # Add entropy gradient
        entropy_grad = probs * (np.log(probs + 1e-8) + 1)
        grad_logits -= entropy_coef * entropy_grad
        
        # Backprop
        grad_W2 = np.outer(h, grad_logits)
        grad_b2 = grad_logits
        
        grad_h = grad_logits @ self.W2.T
        grad_h = grad_h * (1 - h ** 2)  # tanh gradient
        
        grad_W1 = np.outer(obs, grad_h)
        grad_b1 = grad_h
        
        # Update
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1


class ValueNetwork:
    """Centralized value network (critic)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 0.0003):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)
    
    def forward(self, state: np.ndarray) -> float:
        """Predict value of state."""
        h = np.tanh(state @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2)[0]
    
    def update(self, state: np.ndarray, target: float):
        """Update value network."""
        # Forward
        h = np.tanh(state @ self.W1 + self.b1)
        value = (h @ self.W2 + self.b2)[0]
        
        # Loss gradient
        grad = 2 * (value - target)
        
        # Backprop
        grad_W2 = h.reshape(-1, 1) * grad
        grad_b2 = np.array([grad])
        
        grad_h = self.W2.flatten() * grad
        grad_h = grad_h * (1 - h ** 2)
        
        grad_W1 = np.outer(state, grad_h)
        grad_b1 = grad_h
        
        # Update
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1


@dataclass
class Trajectory:
    """Single step in trajectory."""
    state: np.ndarray
    obs: List[np.ndarray]
    actions: List[int]
    log_probs: List[float]
    reward: float
    value: float
    done: bool


class MAPPOAgent:
    """MAPPO agent for multi-agent coordination."""
    
    def __init__(self, config: MAPPOConfig):
        self.config = config
        
        # Individual policy networks
        self.policies = [
            PolicyNetwork(config.obs_dim, config.hidden_dim, config.action_dim, config.learning_rate)
            for _ in range(config.n_agents)
        ]
        
        # Centralized value network
        self.value_net = ValueNetwork(config.state_dim, config.hidden_dim, config.learning_rate)
        
        # Trajectory buffer
        self.trajectories: List[Trajectory] = []
        
        # Training stats
        self.episode_rewards = []
    
    def get_actions(self, state: np.ndarray, obs_list: List[np.ndarray], 
                    deterministic: bool = False) -> Tuple[List[int], List[float], float]:
        """Get actions for all agents."""
        actions = []
        log_probs = []
        
        for i, obs in enumerate(obs_list):
            action, log_prob = self.policies[i].get_action(obs, deterministic)
            actions.append(action)
            log_probs.append(log_prob)
        
        value = self.value_net.forward(state)
        
        return actions, log_probs, value
    
    def store_trajectory(self, traj: Trajectory):
        """Store trajectory step."""
        self.trajectories.append(traj)
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.config.gamma * next_val - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def train(self) -> Optional[Dict[str, float]]:
        """Train on collected trajectories."""
        if len(self.trajectories) < self.config.rollout_length:
            return None
        
        # Extract data
        states = [t.state for t in self.trajectories]
        obs_list = [t.obs for t in self.trajectories]
        actions_list = [t.actions for t in self.trajectories]
        log_probs_list = [t.log_probs for t in self.trajectories]
        rewards = [t.reward for t in self.trajectories]
        values = [t.value for t in self.trajectories]
        dones = [t.done for t in self.trajectories]
        
        # Compute advantages
        next_value = self.value_net.forward(states[-1]) if not dones[-1] else 0.0
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # PPO epochs
        policy_losses = []
        value_losses = []
        
        for _ in range(self.config.n_epochs):
            # Shuffle indices
            indices = list(range(len(self.trajectories)))
            random.shuffle(indices)
            
            for idx in indices:
                state = states[idx]
                obs = obs_list[idx]
                actions = actions_list[idx]
                old_log_probs = log_probs_list[idx]
                advantage = advantages[idx]
                ret = returns[idx]
                
                # Update each agent's policy
                for i in range(self.config.n_agents):
                    self.policies[i].update(
                        obs[i], actions[i], advantage, old_log_probs[i],
                        self.config.clip_epsilon, self.config.entropy_coef
                    )
                
                # Update value network
                self.value_net.update(state, ret)
        
        # Clear trajectories
        self.trajectories = []
        
        return {"policy_loss": np.mean(policy_losses) if policy_losses else 0.0,
                "value_loss": np.mean(value_losses) if value_losses else 0.0}
    
    def get_cooperation_rate(self) -> float:
        """Get recent cooperation rate."""
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-100:]
        return np.mean(recent)


class TaskEnvironment:
    """Task environment matching GCL experiments."""
    
    def __init__(self, n_agents: int, state_dim: int = 10, obs_dim: int = 5):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        self.capabilities = np.random.uniform(0.3, 0.9, n_agents)
        self.current_difficulty = 0.5
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Reset environment."""
        self.current_difficulty = np.random.uniform(0.3, 0.7)
        
        state = np.zeros(self.state_dim)
        state[0] = self.current_difficulty
        state[1] = np.mean(self.capabilities)
        state[2] = np.std(self.capabilities)
        state[3:min(self.state_dim, 3 + self.n_agents)] = self.capabilities[:min(self.state_dim - 3, self.n_agents)]
        
        obs_list = []
        for i in range(self.n_agents):
            obs = np.zeros(self.obs_dim)
            obs[0] = self.current_difficulty
            obs[1] = self.capabilities[i]
            obs[2] = self.capabilities[i] - np.mean(self.capabilities)
            obs[3] = self.capabilities[i] - self.current_difficulty
            obs[4] = 1.0 if self.capabilities[i] > self.current_difficulty else 0.0
            obs_list.append(obs)
        
        return state, obs_list
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[np.ndarray], float, bool]:
        """Execute actions."""
        volunteers = [i for i, a in enumerate(actions) if a == 1]
        
        if not volunteers:
            reward = 0.0
        else:
            best_idx = max(volunteers, key=lambda i: self.capabilities[i])
            capability = self.capabilities[best_idx]
            effort = 0.9
            success_prob = capability * effort * (1 - self.current_difficulty * 0.6)
            success = random.random() < success_prob
            reward = 1.0 if success else 0.0
        
        next_state, next_obs = self.reset()
        return next_state, next_obs, reward, True


def run_mappo_experiment(
    n_episodes: int = 1000,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, Any]:
    """Run MAPPO experiment."""
    random.seed(seed)
    np.random.seed(seed)
    
    config = MAPPOConfig(n_agents=n_agents)
    agent = MAPPOAgent(config)
    env = TaskEnvironment(n_agents)
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, obs = env.reset()
        actions, log_probs, value = agent.get_actions(state, obs)
        next_state, next_obs, reward, done = env.step(actions)
        
        traj = Trajectory(state, obs, actions, log_probs, reward, value, done)
        agent.store_trajectory(traj)
        
        # Train periodically
        if (episode + 1) % config.rollout_length == 0:
            agent.train()
        
        episode_rewards.append(reward)
        agent.episode_rewards.append(reward)
    
    results = {
        "final_cooperation": np.mean(episode_rewards[-100:]),
        "episode_rewards": episode_rewards,
    }
    
    cumulative = np.cumsum(episode_rewards) / (np.arange(len(episode_rewards)) + 1)
    reached_50 = np.where(cumulative >= 0.5)[0]
    results["episodes_to_50"] = int(reached_50[0]) if len(reached_50) > 0 else n_episodes
    
    return results


if __name__ == "__main__":
    print("Testing MAPPO implementation...")
    results = run_mappo_experiment(n_episodes=500, n_agents=10, seed=42)
    print(f"Final cooperation: {results['final_cooperation']:.3f}")
    print(f"Episodes to 50%: {results['episodes_to_50']}")
