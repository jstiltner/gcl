"""
QMIX Implementation for GCL Comparison

QMIX (Rashid et al., 2018) is a value decomposition method for cooperative MARL.
Key idea: Q_tot = f(Q_1, Q_2, ..., Q_n) where f is monotonic in each Q_i

This implementation is simplified but captures the essential mechanisms:
1. Individual Q-networks for each agent
2. Mixing network that combines individual Q-values
3. Centralized training with decentralized execution (CTDE)

For fair comparison with GCL:
- Same task environment (difficulty-based success)
- Same agent capabilities
- Measure sample efficiency and asymptotic performance
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import random


@dataclass
class QMIXConfig:
    """Configuration for QMIX."""
    n_agents: int = 30
    state_dim: int = 10  # Global state dimension
    obs_dim: int = 5     # Individual observation dimension
    action_dim: int = 2  # Accept/reject task
    hidden_dim: int = 64
    mixing_hidden_dim: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 32
    buffer_size: int = 10000
    target_update_freq: int = 100


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    obs: List[np.ndarray]
    actions: List[int]
    reward: float
    next_state: np.ndarray
    next_obs: List[np.ndarray]
    done: bool


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class SimpleQNetwork:
    """Simple Q-network using numpy (no PyTorch dependency)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # ReLU activation
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    
    def copy_from(self, other: 'SimpleQNetwork'):
        """Copy weights from another network."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
    
    def update(self, x: np.ndarray, target: np.ndarray, action: int):
        """Simple gradient descent update."""
        # Forward pass with stored activations
        h = np.maximum(0, x @ self.W1 + self.b1)
        q_values = h @ self.W2 + self.b2
        
        # Compute gradient for selected action only
        grad_output = np.zeros(self.output_dim)
        grad_output[action] = q_values[action] - target[action]
        
        # Backprop through output layer
        grad_W2 = np.outer(h, grad_output)
        grad_b2 = grad_output
        
        # Backprop through hidden layer
        grad_h = grad_output @ self.W2.T
        grad_h = grad_h * (h > 0)  # ReLU gradient
        
        grad_W1 = np.outer(x, grad_h)
        grad_b1 = grad_h
        
        # Update weights
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1


class MixingNetwork:
    """Mixing network that combines individual Q-values."""
    
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int, lr: float = 0.001):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Hypernetwork weights (simplified: just use state to modulate mixing)
        # W1: state -> (n_agents, hidden_dim)
        self.hyper_W1 = np.random.randn(state_dim, n_agents * hidden_dim) * 0.1
        self.hyper_b1 = np.zeros(hidden_dim)
        
        # W2: state -> (hidden_dim, 1)
        self.hyper_W2 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.hyper_b2 = np.zeros(1)
    
    def forward(self, q_values: np.ndarray, state: np.ndarray) -> float:
        """
        Combine individual Q-values into Q_tot.
        
        Args:
            q_values: (n_agents,) individual Q-values
            state: (state_dim,) global state
        
        Returns:
            Q_tot scalar
        """
        # Generate mixing weights from state (ensure monotonicity with abs)
        W1 = np.abs(state @ self.hyper_W1).reshape(self.n_agents, self.hidden_dim)
        W2 = np.abs(state @ self.hyper_W2).reshape(self.hidden_dim, 1)
        
        # Forward through mixing network
        h = np.maximum(0, q_values @ W1 + self.hyper_b1)  # (hidden_dim,)
        q_tot = (h @ W2 + self.hyper_b2)[0]  # scalar
        
        return q_tot


class QMIXAgent:
    """QMIX agent for multi-agent coordination."""
    
    def __init__(self, config: QMIXConfig):
        self.config = config
        self.epsilon = config.epsilon_start
        
        # Individual Q-networks for each agent
        self.q_networks = [
            SimpleQNetwork(config.obs_dim, config.hidden_dim, config.action_dim, config.learning_rate)
            for _ in range(config.n_agents)
        ]
        
        # Target networks
        self.target_q_networks = [
            SimpleQNetwork(config.obs_dim, config.hidden_dim, config.action_dim, config.learning_rate)
            for _ in range(config.n_agents)
        ]
        for i in range(config.n_agents):
            self.target_q_networks[i].copy_from(self.q_networks[i])
        
        # Mixing network
        self.mixer = MixingNetwork(
            config.n_agents, config.state_dim, config.mixing_hidden_dim, config.learning_rate
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size)
        
        # Training stats
        self.train_step = 0
        self.episode_rewards = []
    
    def get_actions(self, obs_list: List[np.ndarray], explore: bool = True) -> List[int]:
        """Get actions for all agents."""
        actions = []
        for i, obs in enumerate(obs_list):
            if explore and random.random() < self.epsilon:
                action = random.randint(0, self.config.action_dim - 1)
            else:
                q_values = self.q_networks[i].forward(obs)
                action = int(np.argmax(q_values))
            actions.append(action)
        return actions
    
    def store_experience(self, experience: Experience):
        """Store experience in replay buffer."""
        self.buffer.push(experience)
    
    def train(self) -> Optional[float]:
        """Train on a batch of experiences."""
        if len(self.buffer) < self.config.batch_size:
            return None
        
        batch = self.buffer.sample(self.config.batch_size)
        total_loss = 0.0
        
        for exp in batch:
            # Get current Q-values
            current_q_values = []
            for i in range(self.config.n_agents):
                q = self.q_networks[i].forward(exp.obs[i])
                current_q_values.append(q[exp.actions[i]])
            current_q_values = np.array(current_q_values)
            
            # Get target Q-values
            target_q_values = []
            for i in range(self.config.n_agents):
                q = self.target_q_networks[i].forward(exp.next_obs[i])
                target_q_values.append(np.max(q))
            target_q_values = np.array(target_q_values)
            
            # Compute Q_tot
            current_q_tot = self.mixer.forward(current_q_values, exp.state)
            target_q_tot = self.mixer.forward(target_q_values, exp.next_state)
            
            # TD target
            if exp.done:
                td_target = exp.reward
            else:
                td_target = exp.reward + self.config.gamma * target_q_tot
            
            # Loss
            loss = (current_q_tot - td_target) ** 2
            total_loss += loss
            
            # Update individual Q-networks (simplified: proportional to contribution)
            td_error = current_q_tot - td_target
            for i in range(self.config.n_agents):
                target = self.q_networks[i].forward(exp.obs[i]).copy()
                # Distribute error proportionally
                target[exp.actions[i]] -= self.config.learning_rate * td_error
                self.q_networks[i].update(exp.obs[i], target, exp.actions[i])
        
        # Update target networks periodically
        self.train_step += 1
        if self.train_step % self.config.target_update_freq == 0:
            for i in range(self.config.n_agents):
                self.target_q_networks[i].copy_from(self.q_networks[i])
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        return total_loss / len(batch)
    
    def get_cooperation_rate(self) -> float:
        """Get recent cooperation rate from episode rewards."""
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-100:]
        return np.mean(recent)


class TaskEnvironment:
    """
    Task environment matching GCL experiments.
    
    - Tasks have difficulty in [0.3, 0.7]
    - Agents have capabilities in [0.3, 0.9]
    - Success probability = capability * effort * (1 - difficulty * 0.6)
    """
    
    def __init__(self, n_agents: int, state_dim: int = 10, obs_dim: int = 5):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Agent capabilities (fixed)
        self.capabilities = np.random.uniform(0.3, 0.9, n_agents)
        
        # Current task
        self.current_difficulty = 0.5
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Reset environment and return initial state/observations."""
        self.current_difficulty = np.random.uniform(0.3, 0.7)
        
        # Global state: [difficulty, mean_capability, std_capability, ...]
        state = np.zeros(self.state_dim)
        state[0] = self.current_difficulty
        state[1] = np.mean(self.capabilities)
        state[2] = np.std(self.capabilities)
        state[3:min(self.state_dim, 3 + self.n_agents)] = self.capabilities[:min(self.state_dim - 3, self.n_agents)]
        
        # Individual observations: [difficulty, own_capability, relative_capability, ...]
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
        """
        Execute actions and return next state, observations, reward, done.
        
        Actions: 0 = reject, 1 = accept (volunteer)
        """
        volunteers = [i for i, a in enumerate(actions) if a == 1]
        
        if not volunteers:
            # No volunteers - task fails
            reward = 0.0
        else:
            # Best volunteer attempts task
            best_idx = max(volunteers, key=lambda i: self.capabilities[i])
            capability = self.capabilities[best_idx]
            
            # Effort bonus for volunteering (matching GCL)
            effort = 0.9  # Volunteers try harder
            
            # Success probability
            success_prob = capability * effort * (1 - self.current_difficulty * 0.6)
            success = random.random() < success_prob
            
            reward = 1.0 if success else 0.0
        
        # Next state (new task)
        next_state, next_obs = self.reset()
        
        return next_state, next_obs, reward, True  # Each task is one episode


def run_qmix_experiment(
    n_episodes: int = 1000,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, Any]:
    """Run QMIX experiment."""
    random.seed(seed)
    np.random.seed(seed)
    
    config = QMIXConfig(n_agents=n_agents)
    agent = QMIXAgent(config)
    env = TaskEnvironment(n_agents)
    
    # Training loop
    episode_rewards = []
    losses = []
    
    for episode in range(n_episodes):
        state, obs = env.reset()
        actions = agent.get_actions(obs, explore=True)
        next_state, next_obs, reward, done = env.step(actions)
        
        # Store experience
        exp = Experience(state, obs, actions, reward, next_state, next_obs, done)
        agent.store_experience(exp)
        
        # Train
        loss = agent.train()
        if loss is not None:
            losses.append(loss)
        
        episode_rewards.append(reward)
        agent.episode_rewards.append(reward)
    
    # Compute metrics
    results = {
        "final_cooperation": np.mean(episode_rewards[-100:]),
        "episode_rewards": episode_rewards,
        "losses": losses,
        "epsilon_final": agent.epsilon,
    }
    
    # Sample efficiency: episodes to reach 50% cooperation
    cumulative = np.cumsum(episode_rewards) / (np.arange(len(episode_rewards)) + 1)
    reached_50 = np.where(cumulative >= 0.5)[0]
    results["episodes_to_50"] = int(reached_50[0]) if len(reached_50) > 0 else n_episodes
    
    return results


if __name__ == "__main__":
    print("Testing QMIX implementation...")
    results = run_qmix_experiment(n_episodes=500, n_agents=10, seed=42)
    print(f"Final cooperation: {results['final_cooperation']:.3f}")
    print(f"Episodes to 50%: {results['episodes_to_50']}")
