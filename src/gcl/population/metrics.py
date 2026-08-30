"""
Population metrics for tracking emergent phenomena.

This module tracks population-level metrics over time:
- Task success rates
- Reputation distribution
- Commitment type entropy (protocol convergence)
- Template propagation
- Specialization emergence
- Trust network structure
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gcl.population.environment import PopulationEnvironment


@dataclass
class PopulationMetrics:
    """
    Track population-level metrics over time.
    
    Records time series data for analyzing emergent phenomena:
    - Protocol convergence (entropy decrease)
    - Trust network formation
    - Template propagation
    - Agent specialization
    """
    
    # Time series
    timesteps: List[int] = field(default_factory=list)
    task_success_rate: List[float] = field(default_factory=list)
    avg_reputation: List[float] = field(default_factory=list)
    reputation_gini: List[float] = field(default_factory=list)
    commitment_entropy: List[float] = field(default_factory=list)
    template_count: List[int] = field(default_factory=list)
    specialization_index: List[float] = field(default_factory=list)
    avg_reward: List[float] = field(default_factory=list)
    
    # Network metrics (computed periodically)
    trust_network_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_timestep(
        self,
        env: "PopulationEnvironment",
        outcomes: List[Dict[str, Any]],
    ) -> None:
        """
        Record metrics for one timestep.
        
        Args:
            env: The population environment.
            outcomes: List of outcome dictionaries from this timestep.
        """
        self.timesteps.append(env.timestep)
        
        # Task success rate
        if outcomes:
            success_rate = np.mean([o["task_success"] for o in outcomes])
            self.task_success_rate.append(float(success_rate))
            
            # Average reward
            rewards = []
            for o in outcomes:
                rewards.extend(o.get("rewards", [0, 0]))
            self.avg_reward.append(float(np.mean(rewards)) if rewards else 0.0)
        else:
            self.task_success_rate.append(0.0)
            self.avg_reward.append(0.0)
        
        # Reputation statistics
        reps = [a.reputation for a in env.agents]
        self.avg_reputation.append(float(np.mean(reps)))
        self.reputation_gini.append(self._gini_coefficient(reps))
        
        # Commitment type entropy (protocol convergence)
        all_types: List[int] = []
        for o in outcomes:
            all_types.extend(o["commitments"])
        if all_types:
            self.commitment_entropy.append(self._entropy(all_types))
        else:
            # Use previous value or max entropy
            if self.commitment_entropy:
                self.commitment_entropy.append(self.commitment_entropy[-1])
            else:
                self.commitment_entropy.append(np.log(env.config.commitment_vocab))
        
        # Template statistics
        total_templates = sum(len(a.templates) for a in env.agents)
        self.template_count.append(total_templates)
        
        # Specialization index
        self.specialization_index.append(self._compute_specialization(env))
        
        # Network metrics (every 100 steps)
        if env.timestep % 100 == 0:
            self.trust_network_metrics.append(
                self._compute_network_metrics(env)
            )
    
    def _gini_coefficient(self, values: List[float]) -> float:
        """
        Compute Gini coefficient (0 = equal, 1 = unequal).
        
        Args:
            values: List of values to compute Gini for.
            
        Returns:
            Gini coefficient in [0, 1].
        """
        values = np.array(sorted(values))
        n = len(values)
        if n == 0 or np.sum(values) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))
    
    def _entropy(self, types: List[int]) -> float:
        """
        Compute entropy of commitment type distribution.
        
        Args:
            types: List of commitment types used.
            
        Returns:
            Entropy value.
        """
        counts = Counter(types)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return float(-sum(p * np.log(p + 1e-10) for p in probs))
    
    def _compute_specialization(self, env: "PopulationEnvironment") -> float:
        """
        Compute specialization index.
        
        High = agents specialize in few commitment types
        Low = agents use many commitment types equally
        
        Args:
            env: The population environment.
            
        Returns:
            Average Gini coefficient of agent commitment portfolios.
        """
        agent_ginis = []
        for agent in env.agents:
            if agent.stats.commitment_types_used:
                counts = list(agent.stats.commitment_types_used.values())
                agent_ginis.append(self._gini_coefficient(counts))
        return float(np.mean(agent_ginis)) if agent_ginis else 0.0
    
    def _compute_network_metrics(self, env: "PopulationEnvironment") -> Dict[str, Any]:
        """
        Compute trust network topology metrics.
        
        Args:
            env: The population environment.
            
        Returns:
            Dictionary of network metrics.
        """
        n = len(env.agents)
        
        # Build adjacency from trust matrix
        threshold = env.config.trust_baseline * 1.2  # Above baseline trust
        adjacency = env.trust_matrix > threshold
        np.fill_diagonal(adjacency, False)  # No self-loops
        
        edges = int(np.sum(adjacency))
        
        if edges == 0:
            return {"edges": 0, "density": 0.0}
        
        # Basic metrics
        density = edges / (n * (n - 1))  # Directed graph
        
        # Degree statistics
        out_degrees = np.sum(adjacency, axis=1)
        in_degrees = np.sum(adjacency, axis=0)
        
        metrics = {
            "edges": edges,
            "density": float(density),
            "avg_out_degree": float(np.mean(out_degrees)),
            "avg_in_degree": float(np.mean(in_degrees)),
            "degree_std": float(np.std(out_degrees)),
        }
        
        # Try to compute clustering (requires networkx)
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            
            for i in range(n):
                for j in range(n):
                    if adjacency[i, j]:
                        G.add_edge(i, j, weight=env.trust_matrix[i, j])
            
            # Clustering (on undirected version)
            G_undirected = G.to_undirected()
            if G_undirected.number_of_edges() > 0:
                metrics["clustering"] = float(nx.average_clustering(G_undirected))
                
                # Average path length (if connected)
                if nx.is_connected(G_undirected):
                    metrics["avg_path_length"] = float(nx.average_shortest_path_length(G_undirected))
                else:
                    # Use largest connected component
                    largest_cc = max(nx.connected_components(G_undirected), key=len)
                    if len(largest_cc) > 1:
                        subgraph = G_undirected.subgraph(largest_cc)
                        metrics["avg_path_length"] = float(nx.average_shortest_path_length(subgraph))
                    metrics["largest_cc_fraction"] = len(largest_cc) / n
            
            # PageRank (identifies influential agents)
            if G.number_of_edges() > 0:
                pagerank = nx.pagerank(G)
                metrics["pagerank_gini"] = self._gini_coefficient(list(pagerank.values()))
        
        except ImportError:
            # networkx not available, skip advanced metrics
            pass
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        return {
            "timestep": self.timesteps[-1] if self.timesteps else 0,
            "task_success_rate": self.task_success_rate[-1] if self.task_success_rate else 0,
            "avg_reputation": self.avg_reputation[-1] if self.avg_reputation else 1.0,
            "commitment_entropy": self.commitment_entropy[-1] if self.commitment_entropy else 0,
            "template_count": self.template_count[-1] if self.template_count else 0,
            "specialization": self.specialization_index[-1] if self.specialization_index else 0,
            "avg_reward": self.avg_reward[-1] if self.avg_reward else 0,
        }
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        Analyze whether protocol has converged.
        
        Returns:
            Dictionary with convergence analysis results.
        """
        if len(self.commitment_entropy) < 100:
            return {"converged": False, "reason": "insufficient_data"}
        
        # Check if entropy is decreasing
        early_entropy = float(np.mean(self.commitment_entropy[:50]))
        late_entropy = float(np.mean(self.commitment_entropy[-50:]))
        
        entropy_decrease = (early_entropy - late_entropy) / (early_entropy + 1e-10)
        
        # Check if success rate is increasing
        early_success = float(np.mean(self.task_success_rate[:50]))
        late_success = float(np.mean(self.task_success_rate[-50:]))
        
        success_increase = late_success - early_success
        
        return {
            "converged": entropy_decrease > 0.2 and success_increase > 0.1,
            "entropy_decrease": entropy_decrease,
            "success_increase": success_increase,
            "early_entropy": early_entropy,
            "late_entropy": late_entropy,
            "early_success": early_success,
            "late_success": late_success,
        }
    
    def get_time_series(self) -> Dict[str, List[float]]:
        """Get all time series data."""
        return {
            "timesteps": self.timesteps,
            "task_success_rate": self.task_success_rate,
            "avg_reputation": self.avg_reputation,
            "reputation_gini": self.reputation_gini,
            "commitment_entropy": self.commitment_entropy,
            "template_count": [float(x) for x in self.template_count],
            "specialization_index": self.specialization_index,
            "avg_reward": self.avg_reward,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "time_series": self.get_time_series(),
            "network_metrics": self.trust_network_metrics,
            "summary": self.get_summary(),
            "convergence": self.get_convergence_analysis(),
        }
