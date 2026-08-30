"""
Infrastructure Ablation Framework for Commitment-Based Coordination.

This module enables systematic study of what social infrastructure is
necessary for commitment-based coordination to emerge and function.

Key questions:
- What minimal infrastructure enables commitment?
- Which components are necessary vs. sufficient?
- Are there phase transitions as infrastructure is added/removed?

Infrastructure components:
1. Identity - Persistent agent identifiers
2. Memory - History of past interactions
3. Reputation - Aggregated trust scores
4. Consequences - Penalties for violations
5. Communication - Ability to share information
6. Observation - Ability to see others' behavior
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from collections import defaultdict


class InfrastructureComponent(Enum):
    """Components of social infrastructure for commitment."""
    
    IDENTITY = auto()       # Persistent agent IDs across interactions
    MEMORY = auto()         # Agents remember past interactions
    REPUTATION = auto()     # Aggregated trust/reliability scores
    CONSEQUENCES = auto()   # Penalties for commitment violations
    COMMUNICATION = auto()  # Agents can share information
    OBSERVATION = auto()    # Agents can observe others' behavior
    INSTITUTIONS = auto()   # Formal rules and enforcement


@dataclass
class InfrastructureConfig:
    """
    Configuration for infrastructure ablation experiments.
    
    Each component can be enabled/disabled to study its effect
    on commitment-based coordination.
    """
    
    # Core identity
    identity_enabled: bool = True
    identity_persistence: float = 1.0  # Probability ID persists across rounds
    
    # Memory
    memory_enabled: bool = True
    memory_length: int = 100  # How many interactions to remember
    memory_decay: float = 0.99  # Decay factor for old memories
    
    # Reputation
    reputation_enabled: bool = True
    reputation_visibility: float = 1.0  # How visible reputation is (0-1)
    reputation_update_rate: float = 0.1  # Learning rate for reputation updates
    
    # Consequences
    consequences_enabled: bool = True
    violation_penalty: float = 0.5  # Penalty for breaking commitments
    penalty_duration: int = 10  # How long penalties last
    
    # Communication
    communication_enabled: bool = True
    communication_range: float = 1.0  # Fraction of population reachable
    communication_noise: float = 0.0  # Noise in communication
    
    # Observation
    observation_enabled: bool = True
    observation_range: float = 0.5  # Fraction of interactions observable
    observation_accuracy: float = 1.0  # Accuracy of observations
    
    # Institutions
    institutions_enabled: bool = False
    institution_strength: float = 0.0  # Enforcement strength
    
    def get_enabled_components(self) -> Set[InfrastructureComponent]:
        """Get set of enabled infrastructure components."""
        enabled = set()
        if self.identity_enabled:
            enabled.add(InfrastructureComponent.IDENTITY)
        if self.memory_enabled:
            enabled.add(InfrastructureComponent.MEMORY)
        if self.reputation_enabled:
            enabled.add(InfrastructureComponent.REPUTATION)
        if self.consequences_enabled:
            enabled.add(InfrastructureComponent.CONSEQUENCES)
        if self.communication_enabled:
            enabled.add(InfrastructureComponent.COMMUNICATION)
        if self.observation_enabled:
            enabled.add(InfrastructureComponent.OBSERVATION)
        if self.institutions_enabled:
            enabled.add(InfrastructureComponent.INSTITUTIONS)
        return enabled
    
    def disable_component(self, component: InfrastructureComponent) -> "InfrastructureConfig":
        """Return new config with specified component disabled."""
        new_config = InfrastructureConfig(
            identity_enabled=self.identity_enabled,
            identity_persistence=self.identity_persistence,
            memory_enabled=self.memory_enabled,
            memory_length=self.memory_length,
            memory_decay=self.memory_decay,
            reputation_enabled=self.reputation_enabled,
            reputation_visibility=self.reputation_visibility,
            reputation_update_rate=self.reputation_update_rate,
            consequences_enabled=self.consequences_enabled,
            violation_penalty=self.violation_penalty,
            penalty_duration=self.penalty_duration,
            communication_enabled=self.communication_enabled,
            communication_range=self.communication_range,
            communication_noise=self.communication_noise,
            observation_enabled=self.observation_enabled,
            observation_range=self.observation_range,
            observation_accuracy=self.observation_accuracy,
            institutions_enabled=self.institutions_enabled,
            institution_strength=self.institution_strength,
        )
        
        if component == InfrastructureComponent.IDENTITY:
            new_config.identity_enabled = False
        elif component == InfrastructureComponent.MEMORY:
            new_config.memory_enabled = False
        elif component == InfrastructureComponent.REPUTATION:
            new_config.reputation_enabled = False
        elif component == InfrastructureComponent.CONSEQUENCES:
            new_config.consequences_enabled = False
        elif component == InfrastructureComponent.COMMUNICATION:
            new_config.communication_enabled = False
        elif component == InfrastructureComponent.OBSERVATION:
            new_config.observation_enabled = False
        elif component == InfrastructureComponent.INSTITUTIONS:
            new_config.institutions_enabled = False
        
        return new_config
    
    @classmethod
    def minimal(cls) -> "InfrastructureConfig":
        """Create minimal infrastructure config (everything disabled)."""
        return cls(
            identity_enabled=False,
            memory_enabled=False,
            reputation_enabled=False,
            consequences_enabled=False,
            communication_enabled=False,
            observation_enabled=False,
            institutions_enabled=False,
        )
    
    @classmethod
    def full(cls) -> "InfrastructureConfig":
        """Create full infrastructure config (everything enabled)."""
        return cls(
            identity_enabled=True,
            memory_enabled=True,
            reputation_enabled=True,
            consequences_enabled=True,
            communication_enabled=True,
            observation_enabled=True,
            institutions_enabled=True,
            institution_strength=0.5,
        )


@dataclass
class AblationAgent:
    """
    Agent with configurable infrastructure support.
    
    This agent's capabilities depend on the infrastructure config,
    allowing us to study how different infrastructure affects behavior.
    """
    
    agent_id: int
    config: InfrastructureConfig
    
    # Internal state
    memory: List[Dict[str, Any]] = field(default_factory=list)
    reputation_beliefs: Dict[int, float] = field(default_factory=dict)
    own_reputation: float = 0.5
    active_penalties: Dict[int, int] = field(default_factory=dict)  # agent_id -> rounds remaining
    
    # Behavioral parameters
    base_cooperation_rate: float = 0.5
    learning_rate: float = 0.1
    
    # Statistics
    commitments_made: int = 0
    commitments_kept: int = 0
    commitments_broken: int = 0
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.agent_id)
    
    def get_effective_id(self) -> int:
        """Get agent's effective ID (may change if identity not persistent)."""
        if not self.config.identity_enabled:
            # Without identity, appear as random agent each time
            return self.rng.integers(0, 1000000)
        
        if self.config.identity_persistence < 1.0:
            # Identity may change with some probability
            if self.rng.random() > self.config.identity_persistence:
                return self.rng.integers(0, 1000000)
        
        return self.agent_id
    
    def remember_interaction(
        self,
        partner_id: int,
        action: str,
        outcome: str,
        reward: float
    ) -> None:
        """Record an interaction in memory."""
        if not self.config.memory_enabled:
            return
        
        self.memory.append({
            "partner_id": partner_id,
            "action": action,
            "outcome": outcome,
            "reward": reward,
            "round": len(self.memory)
        })
        
        # Trim memory if too long
        if len(self.memory) > self.config.memory_length:
            self.memory = self.memory[-self.config.memory_length:]
    
    def get_partner_history(self, partner_id: int) -> List[Dict[str, Any]]:
        """Get history of interactions with a specific partner."""
        if not self.config.memory_enabled:
            return []
        
        return [m for m in self.memory if m["partner_id"] == partner_id]
    
    def update_reputation_belief(self, agent_id: int, kept_commitment: bool) -> None:
        """Update belief about another agent's reputation."""
        if not self.config.reputation_enabled:
            return
        
        current = self.reputation_beliefs.get(agent_id, 0.5)
        target = 1.0 if kept_commitment else 0.0
        
        # Exponential moving average update
        new_value = current + self.config.reputation_update_rate * (target - current)
        self.reputation_beliefs[agent_id] = np.clip(new_value, 0.0, 1.0)
    
    def get_reputation_belief(self, agent_id: int) -> float:
        """Get belief about another agent's reputation."""
        if not self.config.reputation_enabled:
            return 0.5  # No information, assume neutral
        
        # Apply visibility - may not see full reputation
        true_rep = self.reputation_beliefs.get(agent_id, 0.5)
        if self.config.reputation_visibility < 1.0:
            # Partial visibility - blend with prior
            return self.config.reputation_visibility * true_rep + \
                   (1 - self.config.reputation_visibility) * 0.5
        return true_rep
    
    def apply_penalty(self, violator_id: int) -> None:
        """Apply penalty to an agent who violated commitment."""
        if not self.config.consequences_enabled:
            return
        
        self.active_penalties[violator_id] = self.config.penalty_duration
    
    def has_penalty(self, agent_id: int) -> bool:
        """Check if an agent is currently penalized."""
        if not self.config.consequences_enabled:
            return False
        return self.active_penalties.get(agent_id, 0) > 0
    
    def decay_penalties(self) -> None:
        """Decay active penalties by one round."""
        for agent_id in list(self.active_penalties.keys()):
            self.active_penalties[agent_id] -= 1
            if self.active_penalties[agent_id] <= 0:
                del self.active_penalties[agent_id]
    
    def decide_cooperation(self, partner_id: int, context: Dict[str, Any] = None) -> bool:
        """
        Decide whether to cooperate (keep commitment) with a partner.
        
        Decision depends on available infrastructure:
        - Without memory: random based on base rate
        - With memory: consider past interactions
        - With reputation: consider partner's reputation
        - With consequences: consider penalties
        """
        # Base cooperation probability
        coop_prob = self.base_cooperation_rate
        
        # Adjust based on memory
        if self.config.memory_enabled:
            history = self.get_partner_history(partner_id)
            if history:
                # Tit-for-tat-like: cooperate if partner cooperated last time
                recent = history[-1]
                if recent["outcome"] == "cooperated":
                    coop_prob += 0.2
                elif recent["outcome"] == "defected":
                    coop_prob -= 0.2
        
        # Adjust based on reputation
        if self.config.reputation_enabled:
            partner_rep = self.get_reputation_belief(partner_id)
            # More likely to cooperate with reputable partners
            coop_prob += 0.3 * (partner_rep - 0.5)
        
        # Adjust based on consequences
        if self.config.consequences_enabled:
            if self.has_penalty(partner_id):
                # Less likely to cooperate with penalized agents
                coop_prob -= self.config.violation_penalty
        
        # Clip to valid probability
        coop_prob = np.clip(coop_prob, 0.0, 1.0)
        
        return self.rng.random() < coop_prob
    
    def make_commitment(self, partner_id: int, commitment_type: str) -> Dict[str, Any]:
        """Make a commitment to a partner."""
        self.commitments_made += 1
        
        return {
            "issuer": self.get_effective_id(),
            "target": partner_id,
            "type": commitment_type,
            "round": self.commitments_made
        }
    
    def fulfill_commitment(self, commitment: Dict[str, Any], success: bool) -> None:
        """Record commitment fulfillment."""
        if success:
            self.commitments_kept += 1
        else:
            self.commitments_broken += 1
    
    @property
    def reliability(self) -> float:
        """Calculate agent's reliability (commitments kept / made)."""
        if self.commitments_made == 0:
            return 0.5
        return self.commitments_kept / self.commitments_made


@dataclass
class AblationExperimentResult:
    """Results from an infrastructure ablation experiment."""
    
    config: InfrastructureConfig
    enabled_components: Set[InfrastructureComponent]
    
    # Coordination metrics
    cooperation_rate: float
    commitment_fulfillment_rate: float
    average_reward: float
    
    # Network metrics
    trust_network_density: float
    trust_network_clustering: float
    
    # Emergence metrics
    reputation_variance: float
    specialization_index: float
    
    # Time series
    cooperation_over_time: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled_components": [c.name for c in self.enabled_components],
            "cooperation_rate": self.cooperation_rate,
            "commitment_fulfillment_rate": self.commitment_fulfillment_rate,
            "average_reward": self.average_reward,
            "trust_network_density": self.trust_network_density,
            "trust_network_clustering": self.trust_network_clustering,
            "reputation_variance": self.reputation_variance,
            "specialization_index": self.specialization_index,
            "cooperation_over_time": self.cooperation_over_time,
        }


class AblationExperiment:
    """
    Run infrastructure ablation experiments.
    
    Systematically enables/disables infrastructure components
    to study their effect on commitment-based coordination.
    """
    
    def __init__(
        self,
        n_agents: int = 50,
        n_rounds: int = 100,
        interactions_per_round: int = 100,
        seed: int = 42
    ):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.interactions_per_round = interactions_per_round
        self.seed = seed  # Store seed for resetting
        self.rng = np.random.default_rng(seed)
    
    def create_agents(self, config: InfrastructureConfig) -> List[AblationAgent]:
        """Create agents with given infrastructure config."""
        agents = []
        for i in range(self.n_agents):
            agent = AblationAgent(
                agent_id=i,
                config=config,
                base_cooperation_rate=self.rng.uniform(0.3, 0.7)
            )
            agents.append(agent)
        return agents
    
    def run_interaction(
        self,
        agent_a: AblationAgent,
        agent_b: AblationAgent
    ) -> Tuple[bool, bool, float, float]:
        """
        Run a single interaction between two agents.
        
        Returns:
            (a_cooperated, b_cooperated, a_reward, b_reward)
        """
        # Each agent decides whether to cooperate
        a_cooperates = agent_a.decide_cooperation(agent_b.agent_id)
        b_cooperates = agent_b.decide_cooperation(agent_a.agent_id)
        
        # Prisoner's dilemma payoffs
        if a_cooperates and b_cooperates:
            a_reward, b_reward = 3.0, 3.0
        elif a_cooperates and not b_cooperates:
            a_reward, b_reward = 0.0, 5.0
        elif not a_cooperates and b_cooperates:
            a_reward, b_reward = 5.0, 0.0
        else:
            a_reward, b_reward = 1.0, 1.0
        
        # Update memories
        agent_a.remember_interaction(
            agent_b.agent_id,
            "cooperate" if a_cooperates else "defect",
            "cooperated" if b_cooperates else "defected",
            a_reward
        )
        agent_b.remember_interaction(
            agent_a.agent_id,
            "cooperate" if b_cooperates else "defect",
            "cooperated" if a_cooperates else "defected",
            b_reward
        )
        
        # Update reputations
        agent_a.update_reputation_belief(agent_b.agent_id, b_cooperates)
        agent_b.update_reputation_belief(agent_a.agent_id, a_cooperates)
        
        # Apply penalties for defection
        if not a_cooperates:
            agent_b.apply_penalty(agent_a.agent_id)
        if not b_cooperates:
            agent_a.apply_penalty(agent_b.agent_id)
        
        return a_cooperates, b_cooperates, a_reward, b_reward
    
    def run_round(self, agents: List[AblationAgent]) -> Dict[str, float]:
        """Run one round of interactions."""
        cooperations = 0
        total_interactions = 0
        total_reward = 0.0
        
        for _ in range(self.interactions_per_round):
            # Random pairing
            i, j = self.rng.choice(len(agents), size=2, replace=False)
            agent_a, agent_b = agents[i], agents[j]
            
            a_coop, b_coop, a_reward, b_reward = self.run_interaction(agent_a, agent_b)
            
            cooperations += int(a_coop) + int(b_coop)
            total_interactions += 2
            total_reward += a_reward + b_reward
        
        # Decay penalties
        for agent in agents:
            agent.decay_penalties()
        
        return {
            "cooperation_rate": cooperations / total_interactions,
            "average_reward": total_reward / total_interactions
        }
    
    def compute_network_metrics(self, agents: List[AblationAgent]) -> Dict[str, float]:
        """Compute trust network metrics."""
        # Build adjacency matrix from reputation beliefs
        n = len(agents)
        adj_matrix = np.zeros((n, n))
        
        for i, agent in enumerate(agents):
            for j, rep in agent.reputation_beliefs.items():
                if j < n:  # Valid agent ID
                    adj_matrix[i, j] = rep
        
        # Density: fraction of possible edges with trust > 0.5
        trust_edges = np.sum(adj_matrix > 0.5)
        max_edges = n * (n - 1)
        density = trust_edges / max_edges if max_edges > 0 else 0
        
        # Clustering: average local clustering coefficient
        clustering_sum = 0.0
        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0.5)[0]
            if len(neighbors) >= 2:
                # Count edges between neighbors
                neighbor_edges = 0
                for ni in neighbors:
                    for nj in neighbors:
                        if ni != nj and adj_matrix[ni, nj] > 0.5:
                            neighbor_edges += 1
                possible = len(neighbors) * (len(neighbors) - 1)
                clustering_sum += neighbor_edges / possible if possible > 0 else 0
        
        clustering = clustering_sum / n if n > 0 else 0
        
        return {
            "density": density,
            "clustering": clustering
        }
    
    def compute_emergence_metrics(self, agents: List[AblationAgent]) -> Dict[str, float]:
        """Compute metrics related to emergent phenomena."""
        # Reputation variance - high variance indicates differentiation
        reliabilities = [agent.reliability for agent in agents]
        rep_variance = np.var(reliabilities)
        
        # Specialization index - based on cooperation patterns
        # Higher if agents have distinct cooperation patterns
        coop_rates = [agent.base_cooperation_rate for agent in agents]
        specialization = np.std(coop_rates)
        
        return {
            "reputation_variance": rep_variance,
            "specialization_index": specialization
        }
    
    def run_experiment(self, config: InfrastructureConfig) -> AblationExperimentResult:
        """Run a complete ablation experiment with given config."""
        # CRITICAL: Reset RNG to ensure identical random pairings across configs
        # This makes experiments comparable - same agents meet in same order
        self.rng = np.random.default_rng(self.seed)
        
        agents = self.create_agents(config)
        
        cooperation_over_time = []
        total_cooperation = 0.0
        total_reward = 0.0
        
        for round_num in range(self.n_rounds):
            round_results = self.run_round(agents)
            cooperation_over_time.append(round_results["cooperation_rate"])
            total_cooperation += round_results["cooperation_rate"]
            total_reward += round_results["average_reward"]
        
        # Compute final metrics
        network_metrics = self.compute_network_metrics(agents)
        emergence_metrics = self.compute_emergence_metrics(agents)
        
        # Commitment fulfillment rate
        total_kept = sum(a.commitments_kept for a in agents)
        total_made = sum(a.commitments_made for a in agents)
        fulfillment_rate = total_kept / total_made if total_made > 0 else 0
        
        return AblationExperimentResult(
            config=config,
            enabled_components=config.get_enabled_components(),
            cooperation_rate=total_cooperation / self.n_rounds,
            commitment_fulfillment_rate=fulfillment_rate,
            average_reward=total_reward / self.n_rounds,
            trust_network_density=network_metrics["density"],
            trust_network_clustering=network_metrics["clustering"],
            reputation_variance=emergence_metrics["reputation_variance"],
            specialization_index=emergence_metrics["specialization_index"],
            cooperation_over_time=cooperation_over_time
        )
    
    def run_full_ablation(self) -> Dict[str, AblationExperimentResult]:
        """
        Run full ablation study.
        
        Tests:
        1. Full infrastructure
        2. Minimal infrastructure
        3. Each component removed individually
        4. Key combinations
        """
        results = {}
        
        # Full infrastructure
        print("Running: Full infrastructure")
        results["full"] = self.run_experiment(InfrastructureConfig.full())
        
        # Minimal infrastructure
        print("Running: Minimal infrastructure")
        results["minimal"] = self.run_experiment(InfrastructureConfig.minimal())
        
        # Remove each component individually
        full_config = InfrastructureConfig.full()
        for component in InfrastructureComponent:
            name = f"no_{component.name.lower()}"
            print(f"Running: {name}")
            config = full_config.disable_component(component)
            results[name] = self.run_experiment(config)
        
        # Key combinations
        # Identity + Memory only
        print("Running: identity_memory_only")
        config = InfrastructureConfig.minimal()
        config.identity_enabled = True
        config.memory_enabled = True
        results["identity_memory_only"] = self.run_experiment(config)
        
        # Identity + Reputation only
        print("Running: identity_reputation_only")
        config = InfrastructureConfig.minimal()
        config.identity_enabled = True
        config.reputation_enabled = True
        results["identity_reputation_only"] = self.run_experiment(config)
        
        # Identity + Consequences only
        print("Running: identity_consequences_only")
        config = InfrastructureConfig.minimal()
        config.identity_enabled = True
        config.consequences_enabled = True
        results["identity_consequences_only"] = self.run_experiment(config)
        
        return results
    
    def analyze_results(
        self,
        results: Dict[str, AblationExperimentResult]
    ) -> Dict[str, Any]:
        """Analyze ablation results to identify necessary components."""
        analysis = {
            "component_importance": {},
            "minimal_viable_infrastructure": [],
            "synergies": [],
            "phase_transitions": []
        }
        
        # Component importance: how much does removing each component hurt?
        full_coop = results["full"].cooperation_rate
        for component in InfrastructureComponent:
            name = f"no_{component.name.lower()}"
            if name in results:
                removed_coop = results[name].cooperation_rate
                importance = full_coop - removed_coop
                analysis["component_importance"][component.name] = importance
        
        # Find minimal viable infrastructure
        # Components that, when removed, cause > 10% drop in cooperation
        threshold = 0.1
        for component, importance in analysis["component_importance"].items():
            if importance > threshold:
                analysis["minimal_viable_infrastructure"].append(component)
        
        # Identify synergies
        # Check if combinations perform better than sum of parts
        if "identity_memory_only" in results and "identity_reputation_only" in results:
            combined = results.get("identity_memory_reputation", None)
            if combined:
                expected = (results["identity_memory_only"].cooperation_rate + 
                           results["identity_reputation_only"].cooperation_rate) / 2
                actual = combined.cooperation_rate
                if actual > expected * 1.1:  # 10% synergy threshold
                    analysis["synergies"].append(("MEMORY", "REPUTATION"))
        
        return analysis


def run_ablation_study(
    n_agents: int = 50,
    n_rounds: int = 100,
    seed: int = 42
) -> Tuple[Dict[str, AblationExperimentResult], Dict[str, Any]]:
    """
    Convenience function to run a complete ablation study.
    
    Returns:
        (results, analysis)
    """
    experiment = AblationExperiment(
        n_agents=n_agents,
        n_rounds=n_rounds,
        seed=seed
    )
    
    results = experiment.run_full_ablation()
    analysis = experiment.analyze_results(results)
    
    return results, analysis
