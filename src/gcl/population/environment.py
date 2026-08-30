"""
Population environment for commitment experiments.

This module provides the environment for running population-scale
experiments with 50-500 lightweight agents. Agents interact pairwise,
exchanging commitments to complete tasks.

No central coordination — all structure emerges from agent interactions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from gcl.population.lightweight_agent import (
    AgentConfig,
    CommitmentOutcome,
    LightweightCommitment,
    LightweightCommitmentAgent,
)


@dataclass
class PopulationConfig:
    """Configuration for population experiment."""
    
    n_agents: int = 100
    n_timesteps: int = 10000
    task_complexity: int = 4        # Subtasks per interaction
    interaction_type: str = "random"  # "random", "network", "spatial"
    commitment_vocab: int = 16
    state_dim: int = 32
    
    # Agent configuration
    agent_hidden_dim: int = 64
    agent_learning_rate: float = 1e-3
    agent_template_capacity: int = 10
    
    # Environment parameters
    policy_update_frequency: int = 10  # Update policy every N timesteps
    trust_learning_rate: float = 0.1
    trust_baseline: float = 1.0
    trust_min: float = 0.1
    trust_max: float = 3.0
    
    # Emergence parameters - create selective pressure
    type_distribution: str = "skewed"  # "uniform", "skewed", "clustered"
    high_value_types: int = 4  # Number of high-value commitment types
    type_value_ratio: float = 3.0  # How much more valuable high-value types are
    
    # Template sharing (from Experiment 24 - validated +5.2% cooperation, +29% bottom quartile)
    # "none" = no sharing, "random" = random sharing, "directed" = high->low capability
    template_sharing: str = "directed"  # Default to best-performing mode
    template_share_frequency: int = 10  # Share templates every N timesteps
    template_share_probability: float = 0.3  # Probability of sharing per eligible pair
    
    # Difficulty-weighted reputation (from Experiment 27 - best GCL-compatible anti-gaming)
    # Harder tasks give more reputation, reducing easy-task gaming while preserving agent choice
    # Score: 0.838 (best), Gaming reduction: -59.8%, Difficulty increase: +20.4%
    difficulty_weighted_reputation: bool = True  # Enable by default
    difficulty_reputation_multiplier: float = 2.0  # Max multiplier for hard tasks (1x to 3x)


@dataclass
class Task:
    """A coordination task requiring multiple commitment types."""
    
    subtasks: int
    state: torch.Tensor
    required_types: List[int]
    difficulty: float = 0.5


class PopulationEnvironment:
    """
    Environment for population-scale commitment experiments.
    
    Agents interact pairwise, exchanging commitments to complete tasks.
    No central coordination — all structure emerges from agent interactions.
    
    Example:
        >>> config = PopulationConfig(n_agents=100, n_timesteps=5000)
        >>> env = PopulationEnvironment(config)
        >>> metrics = env.run()
        >>> print(metrics.get_summary())
    """
    
    def __init__(self, config: PopulationConfig):
        """
        Initialize the population environment.
        
        Args:
            config: Population configuration.
        """
        self.config = config
        self._seed: Optional[int] = None
        
        # Create agents
        agent_config = AgentConfig(
            state_dim=config.state_dim,
            hidden_dim=config.agent_hidden_dim,
            commitment_vocab=config.commitment_vocab,
            template_capacity=config.agent_template_capacity,
            learning_rate=config.agent_learning_rate,
        )
        self.agents = [
            LightweightCommitmentAgent(f"agent_{i}", agent_config)
            for i in range(config.n_agents)
        ]
        
        # Trust network (who has interacted with whom, outcomes)
        self.trust_matrix = np.ones((config.n_agents, config.n_agents)) * config.trust_baseline
        
        # Metrics tracking (imported lazily to avoid circular import)
        self.metrics: Optional[Any] = None
        
        # Timestep counter
        self.timestep = 0
        
        # Create type value distribution for selective pressure
        self._setup_type_values()
    
    def _setup_type_values(self) -> None:
        """Setup commitment type values to create selective pressure."""
        n_types = self.config.commitment_vocab
        
        if self.config.type_distribution == "uniform":
            # All types equally valuable
            self.type_values = np.ones(n_types)
            self.type_probs = np.ones(n_types) / n_types
            
        elif self.config.type_distribution == "skewed":
            # Some types are more valuable (appear more often in tasks)
            self.type_values = np.ones(n_types)
            high_value = list(range(self.config.high_value_types))
            for t in high_value:
                self.type_values[t] = self.config.type_value_ratio
            
            # High-value types appear more often in tasks
            self.type_probs = self.type_values / self.type_values.sum()
            
        elif self.config.type_distribution == "clustered":
            # Types form clusters - agents should specialize in clusters
            n_clusters = 4
            cluster_size = n_types // n_clusters
            self.type_values = np.ones(n_types)
            self.type_clusters = {}
            for c in range(n_clusters):
                cluster_types = list(range(c * cluster_size, (c + 1) * cluster_size))
                self.type_clusters[c] = cluster_types
                # First type in each cluster is most valuable
                self.type_values[cluster_types[0]] = 2.0
            self.type_probs = self.type_values / self.type_values.sum()
        else:
            self.type_values = np.ones(n_types)
            self.type_probs = np.ones(n_types) / n_types
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def step(self) -> Dict[str, Any]:
        """
        Run one timestep of population dynamics.
        
        1. Select pairs of agents to interact
        2. Each pair attempts to coordinate via commitments
        3. Resolve outcomes, update reputations
        4. Agents learn from experience
        5. Record metrics
        
        Returns:
            Dictionary with timestep info and outcomes.
        """
        self.timestep += 1
        
        # Select interacting pairs
        pairs = self._select_pairs()
        
        timestep_outcomes = []
        
        for i, j in pairs:
            agent_i = self.agents[i]
            agent_j = self.agents[j]
            
            # Generate task
            task = self._generate_task()
            
            # Agents propose commitments
            state_i = self._get_agent_state(i, task)
            state_j = self._get_agent_state(j, task)
            
            commit_i = agent_i.propose_commitment(
                state_i,
                partner_reputation=agent_j.reputation,
            )
            commit_j = agent_j.propose_commitment(
                state_j,
                partner_reputation=agent_i.reputation,
            )
            
            # Resolve task based on commitments
            outcome_i, outcome_j, task_success = self._resolve_task(
                task, commit_i, commit_j, agent_i, agent_j
            )
            
            # Compute rewards
            reward_i = self._compute_reward(commit_i, outcome_i, task_success)
            reward_j = self._compute_reward(commit_j, outcome_j, task_success)
            
            # Agents receive outcomes with difficulty-weighted reputation
            self._receive_outcome_with_difficulty(
                agent_i, commit_i, outcome_i, reward_i, task.difficulty
            )
            self._receive_outcome_with_difficulty(
                agent_j, commit_j, outcome_j, reward_j, task.difficulty
            )
            
            # Update trust matrix
            self._update_trust(i, j, outcome_i.success, outcome_j.success)
            
            timestep_outcomes.append({
                "pair": (i, j),
                "task_success": task_success,
                "commitments": (commit_i.commitment_type, commit_j.commitment_type),
                "outcomes": (outcome_i.success, outcome_j.success),
                "rewards": (reward_i, reward_j),
            })
        
        # Periodic policy updates
        if self.timestep % self.config.policy_update_frequency == 0:
            for agent in self.agents:
                if len(agent.commitment_history) >= 20:
                    agent.update_policy()
        
        # Template sharing (from Experiment 24 - validated +5.2% cooperation)
        if (self.config.template_sharing != "none" and
            self.timestep % self.config.template_share_frequency == 0):
            self._share_templates()
        
        # Record metrics
        if self.metrics is not None:
            self.metrics.record_timestep(self, timestep_outcomes)
        
        return {
            "timestep": self.timestep,
            "outcomes": timestep_outcomes,
            "summary": self.metrics.get_summary() if self.metrics else {},
        }
    
    def _select_pairs(self) -> List[Tuple[int, int]]:
        """Select agent pairs for this timestep."""
        n = self.config.n_agents
        
        if self.config.interaction_type == "random":
            # Random pairing
            indices = list(range(n))
            random.shuffle(indices)
            return [(indices[i], indices[i + 1]) for i in range(0, n - 1, 2)]
        
        elif self.config.interaction_type == "network":
            # Preferential interaction based on trust
            pairs = []
            available = set(range(n))
            while len(available) >= 2:
                i = random.choice(list(available))
                available.remove(i)
                # Select partner weighted by trust
                candidates = list(available)
                weights = np.array([self.trust_matrix[i, j] for j in candidates])
                weights = weights / weights.sum()
                j = int(np.random.choice(candidates, p=weights))
                available.remove(j)
                pairs.append((i, j))
            return pairs
        
        elif self.config.interaction_type == "spatial":
            # Spatial interaction (neighbors only)
            # Arrange agents in a ring, interact with neighbors
            pairs = []
            for i in range(0, n - 1, 2):
                j = (i + 1) % n
                pairs.append((i, j))
            return pairs
        
        else:
            raise ValueError(f"Unknown interaction type: {self.config.interaction_type}")
    
    def _generate_task(self) -> Task:
        """Generate a task with skewed type distribution for selective pressure."""
        # Sample required types according to value distribution
        # High-value types appear more often, creating selective pressure
        required_types = list(np.random.choice(
            range(self.config.commitment_vocab),
            size=min(self.config.task_complexity, self.config.commitment_vocab),
            replace=False,
            p=self.type_probs,
        ))
        
        # Create state that encodes which types are valuable
        # This helps agents learn to recognize high-value situations
        state = torch.randn(self.config.state_dim)
        # Embed type information in first few dimensions
        for i, t in enumerate(required_types[:4]):
            if i < self.config.state_dim:
                state[i] = float(t) / self.config.commitment_vocab
        
        return Task(
            subtasks=self.config.task_complexity,
            state=state,
            required_types=required_types,
            difficulty=random.random() * 0.5,  # Reduce difficulty for more success
        )
    
    def _get_agent_state(self, agent_idx: int, task: Task) -> torch.Tensor:
        """Get state observation for an agent."""
        # Combine task state with agent's private info
        agent = self.agents[agent_idx]
        private_state = torch.tensor([
            agent.reputation,
            agent.total_commitments / 100.0,
            agent.fulfilled_commitments / max(1, agent.total_commitments),
            len(agent.templates) / agent.config.template_capacity,
        ])
        return torch.cat([task.state, private_state.float()])
    
    def _resolve_task(
        self,
        task: Task,
        commit_i: LightweightCommitment,
        commit_j: LightweightCommitment,
        agent_i: LightweightCommitmentAgent,
        agent_j: LightweightCommitmentAgent,
    ) -> Tuple[CommitmentOutcome, CommitmentOutcome, bool]:
        """
        Resolve task based on commitments.
        
        Creates selective pressure for emergence:
        - High-value commitment types have higher success rates
        - Matching task requirements gives bonuses
        - Complementary types are rewarded
        - Templates boost success significantly
        """
        # Check if commitment types are useful for task
        useful_i = commit_i.commitment_type in task.required_types
        useful_j = commit_j.commitment_type in task.required_types
        
        # Get type values (high-value types succeed more often)
        value_i = self.type_values[commit_i.commitment_type]
        value_j = self.type_values[commit_j.commitment_type]
        
        # Complementary bonus: different types that both contribute
        complementary = (
            commit_i.commitment_type != commit_j.commitment_type and
            (useful_i or useful_j)
        )
        
        # Coverage bonus
        covered_types = set()
        if useful_i:
            covered_types.add(commit_i.commitment_type)
        if useful_j:
            covered_types.add(commit_j.commitment_type)
        coverage = len(covered_types) / max(1, len(task.required_types))
        
        def agent_succeeds(
            commit: LightweightCommitment,
            agent: LightweightCommitmentAgent,
            useful: bool,
            partner_useful: bool,
            type_value: float,
        ) -> Tuple[bool, float]:
            """Determine if agent succeeds and failure severity."""
            # Base success probability - higher for high-value types
            # This creates selective pressure for convergence
            base_prob = 0.4 + 0.2 * (type_value / self.config.type_value_ratio)
            
            # Confidence modulates (but doesn't dominate)
            base_prob = 0.7 * base_prob + 0.3 * commit.confidence
            
            # Usefulness bonus (significant)
            if useful:
                base_prob *= 1.4  # Strong bonus for matching task
            elif partner_useful:
                base_prob *= 1.0  # Neutral if partner covers
            else:
                base_prob *= 0.6  # Penalty for mismatch
            
            # Complementary bonus
            if complementary:
                base_prob *= 1.2
            
            # Template bonus (significant to encourage template formation)
            if commit.template_id:
                template = next(
                    (t for t in agent.templates if t.id == commit.template_id),
                    None,
                )
                if template:
                    base_prob = 0.5 * base_prob + 0.5 * template.success_rate
            
            # Difficulty modifier (mild)
            base_prob *= (1.0 - 0.15 * task.difficulty)
            
            # Clamp probability
            base_prob = min(0.95, max(0.15, base_prob))
            
            success = random.random() < base_prob
            severity = 0.2 if not success else 0.0
            return success, severity
        
        success_i, severity_i = agent_succeeds(commit_i, agent_i, useful_i, useful_j, value_i)
        success_j, severity_j = agent_succeeds(commit_j, agent_j, useful_j, useful_i, value_j)
        
        # Task success: either both succeed, or good coverage with one success
        task_success = (success_i and success_j) or (coverage >= 0.5 and (success_i or success_j))
        
        return (
            CommitmentOutcome(success=success_i, severity=severity_i),
            CommitmentOutcome(success=success_j, severity=severity_j),
            task_success,
        )
    
    def _compute_reward(
        self,
        commitment: LightweightCommitment,
        outcome: CommitmentOutcome,
        task_success: bool,
    ) -> float:
        """Compute reward for an agent."""
        if outcome.success:
            reward = commitment.stake * 0.1  # Keep stake
            if task_success:
                reward += 1.0  # Task bonus
            return reward
        else:
            return -commitment.stake * outcome.severity
    
    def _receive_outcome_with_difficulty(
        self,
        agent: LightweightCommitmentAgent,
        commitment: LightweightCommitment,
        outcome: CommitmentOutcome,
        reward: float,
        difficulty: float,
    ) -> None:
        """
        Update agent with difficulty-weighted reputation.
        
        From Experiment 27: Difficulty-weighted reputation is the best
        GCL-compatible anti-gaming mechanism (score 0.838).
        
        - Harder tasks give MORE reputation on success
        - Easier tasks give LESS reputation on success
        - Failure penalty is inversely weighted (hard task failures hurt less)
        
        This preserves agent choice while incentivizing harder tasks.
        """
        # Calculate difficulty multiplier (1x to 3x based on config)
        if self.config.difficulty_weighted_reputation:
            # Linear scaling: difficulty 0 -> 1x, difficulty 1 -> (1 + multiplier)x
            difficulty_mult = 1.0 + difficulty * self.config.difficulty_reputation_multiplier
        else:
            difficulty_mult = 1.0
        
        # Update reputation with difficulty weighting
        if outcome.success:
            # Harder tasks give more reputation
            rep_gain = 0.1 * commitment.stake * difficulty_mult
            agent.reputation = min(2.0, agent.reputation + rep_gain)
            agent.fulfilled_commitments += 1
        else:
            # Harder task failures are less punishing (inverse weighting)
            rep_loss = outcome.severity * commitment.stake / difficulty_mult
            agent.reputation = max(0.1, agent.reputation - rep_loss)
        
        agent.reputation *= agent.config.reputation_decay  # Natural decay
        agent.total_commitments += 1
        
        # Record for template learning
        from gcl.population.lightweight_agent import CommitmentRecord
        agent.commitment_history.append(CommitmentRecord(
            commitment=commitment,
            outcome=outcome,
            reward=reward,
        ))
        
        # Update stats
        agent.stats.update(commitment, outcome)
        agent.stats.reputation_history.append(agent.reputation)
        
        # Update template if used
        if commitment.template_id:
            template = next(
                (t for t in agent.templates if t.id == commitment.template_id),
                None
            )
            if template:
                template.record_usage(outcome.success)
        
        # Induce templates periodically
        if len(agent.commitment_history) % 10 == 0:
            agent._maybe_induce_template()
    
    def _update_trust(
        self,
        i: int,
        j: int,
        success_i: bool,
        success_j: bool,
    ) -> None:
        """Update trust matrix based on interaction outcome."""
        alpha = self.config.trust_learning_rate
        
        # i's trust in j
        if success_j:
            self.trust_matrix[i, j] = (1 - alpha) * self.trust_matrix[i, j] + alpha * 1.5
        else:
            self.trust_matrix[i, j] = (1 - alpha) * self.trust_matrix[i, j] + alpha * 0.5
        
        # j's trust in i
        if success_i:
            self.trust_matrix[j, i] = (1 - alpha) * self.trust_matrix[j, i] + alpha * 1.5
        else:
            self.trust_matrix[j, i] = (1 - alpha) * self.trust_matrix[j, i] + alpha * 0.5
        
        # Clip trust values
        self.trust_matrix = np.clip(
            self.trust_matrix,
            self.config.trust_min,
            self.config.trust_max,
        )
    
    def run(self, n_steps: int | None = None) -> Any:
        """
        Run simulation for n_steps.
        
        Args:
            n_steps: Number of steps to run. If None, uses config.n_timesteps.
            
        Returns:
            PopulationMetrics object with recorded data.
        """
        # Import here to avoid circular import
        from gcl.population.metrics import PopulationMetrics
        
        if self.metrics is None:
            self.metrics = PopulationMetrics()
        
        n_steps = n_steps or self.config.n_timesteps
        
        for _ in range(n_steps):
            self.step()
        
        return self.metrics
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about agents."""
        reputations = [a.reputation for a in self.agents]
        success_rates = [a.success_rate for a in self.agents]
        template_counts = [len(a.templates) for a in self.agents]
        
        return {
            "n_agents": len(self.agents),
            "avg_reputation": float(np.mean(reputations)),
            "std_reputation": float(np.std(reputations)),
            "avg_success_rate": float(np.mean(success_rates)),
            "std_success_rate": float(np.std(success_rates)),
            "avg_templates": float(np.mean(template_counts)),
            "total_templates": sum(template_counts),
        }
    
    def get_trust_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the trust network."""
        # Threshold for "high trust" edges
        threshold = self.config.trust_baseline * 1.2
        high_trust = self.trust_matrix > threshold
        
        return {
            "mean_trust": float(np.mean(self.trust_matrix)),
            "std_trust": float(np.std(self.trust_matrix)),
            "high_trust_edges": int(np.sum(high_trust) - len(self.agents)),  # Exclude diagonal
            "trust_density": float(np.sum(high_trust) / (len(self.agents) ** 2)),
        }
    
    def _share_templates(self) -> None:
        """
        Share templates between agents according to configured mode.
        
        From Experiment 24 (validated +5.2% cooperation, +29% bottom quartile):
        - "none": No sharing
        - "random": Random pairs share templates
        - "directed": High-capability agents share with low-capability agents
        
        This is GCL-compatible because:
        - Agents still CHOOSE what commitments to make
        - Template sharing is knowledge transfer, not task assignment
        - Preserves agent autonomy while improving collective capability
        """
        if self.config.template_sharing == "none":
            return
        
        n = self.config.n_agents
        
        if self.config.template_sharing == "random":
            # Random pairs share templates
            indices = list(range(n))
            random.shuffle(indices)
            for i in range(0, n - 1, 2):
                if random.random() < self.config.template_share_probability:
                    self._transfer_template(indices[i], indices[i + 1])
        
        elif self.config.template_sharing == "directed":
            # High-capability agents share with low-capability agents
            # Sort by success rate (proxy for capability)
            sorted_agents = sorted(
                enumerate(self.agents),
                key=lambda x: x[1].success_rate,
                reverse=True
            )
            
            # Top half shares with bottom half
            n_pairs = n // 4  # Share with 25% of population
            for i in range(n_pairs):
                if random.random() < self.config.template_share_probability:
                    high_idx = sorted_agents[i][0]
                    low_idx = sorted_agents[-(i + 1)][0]
                    self._transfer_template(high_idx, low_idx)
        
        elif self.config.template_sharing == "mutual":
            # Mutual sharing between interacting pairs (based on trust)
            for i in range(n):
                # Find highest trust partner
                trust_row = self.trust_matrix[i].copy()
                trust_row[i] = 0  # Exclude self
                j = int(np.argmax(trust_row))
                
                if random.random() < self.config.template_share_probability:
                    self._transfer_template(i, j)
    
    def _transfer_template(self, from_idx: int, to_idx: int) -> bool:
        """
        Transfer a template from one agent to another.
        
        Returns True if transfer was successful.
        """
        from_agent = self.agents[from_idx]
        to_agent = self.agents[to_idx]
        
        # Check if from_agent has templates to share
        if not from_agent.templates:
            return False
        
        # Check if to_agent has capacity
        if len(to_agent.templates) >= to_agent.config.template_capacity:
            return False
        
        # Select best template to share (highest success rate)
        best_template = max(from_agent.templates, key=lambda t: t.success_rate)
        
        # Check if to_agent already has this template type
        existing_types = {t.commitment_type for t in to_agent.templates}
        if best_template.commitment_type in existing_types:
            # Try to find a different template
            for template in sorted(from_agent.templates, key=lambda t: t.success_rate, reverse=True):
                if template.commitment_type not in existing_types:
                    best_template = template
                    break
            else:
                return False  # No new templates to share
        
        # Create a copy of the template for the receiving agent
        from gcl.population.lightweight_agent import CommitmentTemplate
        new_template = CommitmentTemplate(
            id=f"{best_template.id}_shared_{to_agent.id}",
            commitment_type=best_template.commitment_type,
            success_rate=best_template.success_rate * 0.9,  # Slight degradation on transfer
            usage_count=0,
        )
        
        to_agent.templates.append(new_template)
        return True
