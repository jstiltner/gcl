"""
Multi-Agent Reinforcement Learning (MARL) baseline implementation.

Implements Independent Q-Learning (IQL) and basic MARL approaches
for multi-agent coordination without explicit communication protocols.

This provides a learning-based baseline that doesn't use structured
commitments or protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from gcl.baselines.base import (
    MASBaseline,
    CoordinationTask,
    CoordinationResult,
    AgentCapability,
)


@dataclass
class MARLState:
    """State representation for MARL agents."""
    task_features: np.ndarray  # Task requirements
    agent_features: np.ndarray  # Agent capabilities
    availability: np.ndarray  # Which agents are available
    time_step: int = 0


@dataclass
class MARLExperience:
    """Experience tuple for replay."""
    state: MARLState
    action: int
    reward: float
    next_state: MARLState
    done: bool


class IndependentQLearner:
    """
    Independent Q-Learning agent.
    
    Each agent learns independently, treating other agents
    as part of the environment.
    """
    
    def __init__(
        self,
        agent_id: str,
        n_actions: int,
        state_dim: int,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        
        # Simple linear Q-function approximation
        self.weights = self.rng.normal(0, 0.1, (n_actions, state_dim))
        self.bias = np.zeros(n_actions)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        return self.weights @ state + self.bias
    
    def select_action(self, state: np.ndarray, available_actions: List[int]) -> int:
        """Select action using epsilon-greedy."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(available_actions)
        
        q_values = self.get_q_values(state)
        # Mask unavailable actions
        masked_q = np.full(self.n_actions, -np.inf)
        for a in available_actions:
            masked_q[a] = q_values[a]
        
        return int(np.argmax(masked_q))
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Update Q-function using TD learning."""
        current_q = self.get_q_values(state)[action]
        
        if done:
            target = reward
        else:
            next_q = np.max(self.get_q_values(next_state))
            target = reward + self.gamma * next_q
        
        td_error = target - current_q
        
        # Update weights
        self.weights[action] += self.lr * td_error * state
        self.bias[action] += self.lr * td_error
        
        return td_error


class MARLBaseline(MASBaseline):
    """
    Multi-Agent Reinforcement Learning baseline.
    
    Implements Independent Q-Learning where each agent learns
    to select tasks independently. No explicit communication
    or commitment protocols.
    
    Attributes:
        learning_rate: Learning rate for Q-learning.
        discount: Discount factor.
        epsilon: Exploration rate.
        training_episodes: Episodes for pre-training.
    """
    
    def __init__(
        self,
        n_agents: int,
        seed: int = 42,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        training_episodes: int = 100,
    ):
        super().__init__(n_agents, seed)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.training_episodes = training_episodes
        
        # Will be initialized when we know task count
        self.learners: Dict[str, IndependentQLearner] = {}
        self.state_dim = 0
        self.n_tasks = 0
        self.rng = np.random.default_rng(seed)
    
    def _initialize_learners(self, n_tasks: int) -> None:
        """Initialize Q-learners for each agent."""
        self.n_tasks = n_tasks
        # State: task features (4 per task) + agent availability (n_agents)
        self.state_dim = n_tasks * 4 + self.n_agents
        
        self.learners = {}
        for i, agent in enumerate(self.agents):
            self.learners[agent.agent_id] = IndependentQLearner(
                agent_id=agent.agent_id,
                n_actions=n_tasks + 1,  # +1 for "do nothing"
                state_dim=self.state_dim,
                learning_rate=self.learning_rate,
                discount=self.discount,
                epsilon=self.epsilon,
                seed=self.seed + i,
            )
    
    def _encode_state(
        self,
        tasks: List[CoordinationTask],
        availability: np.ndarray,
    ) -> np.ndarray:
        """Encode state as feature vector."""
        # Task features
        task_features = []
        for task in tasks:
            # Extract key features
            value = task.value
            req_sum = sum(task.requirements.values()) if task.requirements else 0
            res_sum = sum(task.resources.values()) if task.resources else 0
            deadline = task.deadline if task.deadline else 10
            task_features.extend([value, req_sum, res_sum, deadline / 10])
        
        # Pad if needed
        while len(task_features) < self.n_tasks * 4:
            task_features.extend([0, 0, 0, 0])
        
        # Combine task features with availability
        features = task_features[:self.n_tasks * 4] + list(availability)
        return np.array(features)
    
    def _train(self, tasks: List[CoordinationTask]) -> None:
        """Pre-train agents on task allocation."""
        for episode in range(self.training_episodes):
            # Reset availability
            availability = np.ones(self.n_agents)
            assigned_tasks: set = set()
            
            state = self._encode_state(tasks, availability)
            
            # Each agent takes turns
            for agent_idx, agent in enumerate(self.agents):
                if availability[agent_idx] == 0:
                    continue
                
                learner = self.learners[agent.agent_id]
                
                # Available actions: unassigned tasks + do nothing
                available_actions = [
                    i for i, t in enumerate(tasks)
                    if t.task_id not in assigned_tasks
                ]
                available_actions.append(len(tasks))  # "do nothing"
                
                if not available_actions:
                    continue
                
                # Select action
                action = learner.select_action(state, available_actions)
                
                # Compute reward
                if action < len(tasks):
                    task = tasks[action]
                    capability = self.compute_task_fitness(agent, task)
                    reward = task.value * capability - agent.cost
                    assigned_tasks.add(task.task_id)
                    availability[agent_idx] = 0
                else:
                    reward = -0.1  # Small penalty for doing nothing
                
                # Update state
                next_state = self._encode_state(tasks, availability)
                done = len(assigned_tasks) == len(tasks)
                
                # Learn
                learner.update(state, action, reward, next_state, done)
                state = next_state
    
    def coordinate(
        self,
        tasks: List[CoordinationTask],
    ) -> CoordinationResult:
        """
        Coordinate using MARL.
        
        Args:
            tasks: Tasks to coordinate.
            
        Returns:
            CoordinationResult with assignments.
        """
        # Initialize and train if needed
        if not self.learners or self.n_tasks != len(tasks):
            self._initialize_learners(len(tasks))
            self._train(tasks)
        
        # Execute learned policy (with reduced exploration)
        old_epsilons = {}
        for agent_id, learner in self.learners.items():
            old_epsilons[agent_id] = learner.epsilon
            learner.epsilon = 0.05  # Reduced exploration for execution
        
        assignments: Dict[str, str] = {}
        availability = np.ones(self.n_agents)
        assigned_tasks: set = set()
        total_value = 0.0
        messages_sent = 0  # MARL has no explicit messages
        time_steps = 0
        
        state = self._encode_state(tasks, availability)
        
        # Agents act in sequence
        for agent_idx, agent in enumerate(self.agents):
            if availability[agent_idx] == 0:
                continue
            
            learner = self.learners[agent.agent_id]
            
            # Available actions
            available_actions = [
                i for i, t in enumerate(tasks)
                if t.task_id not in assigned_tasks
            ]
            available_actions.append(len(tasks))
            
            if len(available_actions) == 1:  # Only "do nothing"
                continue
            
            # Select action
            action = learner.select_action(state, available_actions)
            time_steps += 1
            
            if action < len(tasks):
                task = tasks[action]
                assignments[task.task_id] = agent.agent_id
                assigned_tasks.add(task.task_id)
                availability[agent_idx] = 0
                total_value += task.value
            
            state = self._encode_state(tasks, availability)
        
        # Restore exploration rates
        for agent_id, eps in old_epsilons.items():
            self.learners[agent_id].epsilon = eps
        
        # Calculate efficiency
        max_value = sum(t.value for t in tasks)
        efficiency = total_value / max_value if max_value > 0 else 0.0
        
        return CoordinationResult(
            success=len(assignments) > 0,
            assignments=assignments,
            messages_sent=messages_sent,
            time_steps=time_steps,
            total_value=total_value,
            efficiency=efficiency,
            metadata={
                "protocol": "marl_iql",
                "training_episodes": self.training_episodes,
                "agents_acted": sum(1 for a in availability if a == 0),
            },
        )
    
    def get_q_values(self, agent_id: str, state: np.ndarray) -> np.ndarray:
        """Get Q-values for an agent."""
        if agent_id in self.learners:
            return self.learners[agent_id].get_q_values(state)
        return np.zeros(self.n_tasks + 1)
    
    def get_policy_stats(self) -> Dict:
        """Get statistics about learned policies."""
        stats = {}
        for agent_id, learner in self.learners.items():
            # Compute average Q-value magnitude
            avg_weight = np.mean(np.abs(learner.weights))
            max_weight = np.max(np.abs(learner.weights))
            stats[agent_id] = {
                "avg_weight_magnitude": float(avg_weight),
                "max_weight_magnitude": float(max_weight),
            }
        return stats
