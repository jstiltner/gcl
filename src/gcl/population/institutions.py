"""
Institutional Emergence Module for Commitment-Based Coordination.

This module studies how institutions emerge from commitment-based
interactions at the population level.

Key concepts:
- Institutions: Persistent patterns of rules and enforcement
- Norms: Shared expectations about behavior
- Roles: Specialized positions within coordination structures
- Governance: Mechanisms for collective decision-making

Research questions:
- Do institutions emerge spontaneously from commitment interactions?
- What triggers institutional formation?
- How do institutions affect coordination efficiency?
- Are there phase transitions in institutional complexity?
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from collections import defaultdict
import networkx as nx


class InstitutionType(Enum):
    """Types of emergent institutions."""
    
    NORM = auto()           # Shared behavioral expectation
    ROLE = auto()           # Specialized position
    RULE = auto()           # Explicit constraint
    HIERARCHY = auto()      # Authority structure
    MARKET = auto()         # Exchange mechanism
    COALITION = auto()      # Cooperative group
    ENFORCEMENT = auto()    # Punishment mechanism


@dataclass
class Norm:
    """
    An emergent behavioral norm.
    
    Norms are patterns of behavior that become expected
    and are enforced through social pressure.
    """
    
    norm_id: int
    description: str
    behavior_pattern: str  # e.g., "cooperate_with_cooperators"
    
    # Adoption metrics
    adopters: Set[int] = field(default_factory=set)
    adoption_rate: float = 0.0
    
    # Enforcement
    violations: int = 0
    enforcements: int = 0
    
    # Stability
    emergence_round: int = 0
    stability_score: float = 0.0
    
    def is_established(self, threshold: float = 0.5) -> bool:
        """Check if norm is established (adopted by majority)."""
        return self.adoption_rate >= threshold


@dataclass
class Role:
    """
    An emergent role within the population.
    
    Roles are specialized positions that agents can occupy,
    often with associated expectations and privileges.
    """
    
    role_id: int
    name: str
    
    # Role characteristics
    responsibilities: List[str] = field(default_factory=list)
    privileges: List[str] = field(default_factory=list)
    
    # Occupants
    current_occupants: Set[int] = field(default_factory=set)
    max_occupants: int = 1
    
    # Performance
    performance_history: List[float] = field(default_factory=list)
    
    def is_vacant(self) -> bool:
        """Check if role has vacancies."""
        return len(self.current_occupants) < self.max_occupants


@dataclass
class Coalition:
    """
    An emergent coalition of cooperating agents.
    
    Coalitions form when agents find mutual benefit in
    sustained cooperation.
    """
    
    coalition_id: int
    members: Set[int] = field(default_factory=set)
    
    # Coalition properties
    formation_round: int = 0
    internal_cooperation_rate: float = 0.0
    external_cooperation_rate: float = 0.0
    
    # Stability
    membership_changes: int = 0
    stability_score: float = 0.0
    
    def size(self) -> int:
        """Get coalition size."""
        return len(self.members)
    
    def is_stable(self, threshold: float = 0.8) -> bool:
        """Check if coalition is stable."""
        return self.stability_score >= threshold


@dataclass
class Institution:
    """
    A formal institution with rules and enforcement.
    
    Institutions are more structured than norms, with
    explicit rules and enforcement mechanisms.
    """
    
    institution_id: int
    name: str
    institution_type: InstitutionType
    
    # Rules
    rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Membership
    members: Set[int] = field(default_factory=set)
    
    # Governance
    decision_mechanism: str = "majority"  # majority, consensus, hierarchy
    leaders: Set[int] = field(default_factory=set)
    
    # Performance
    coordination_efficiency: float = 0.0
    member_satisfaction: float = 0.0
    
    # History
    formation_round: int = 0
    rule_changes: int = 0
    
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add a rule to the institution."""
        self.rules.append(rule)
        self.rule_changes += 1
    
    def is_member(self, agent_id: int) -> bool:
        """Check if agent is a member."""
        return agent_id in self.members


class InstitutionalAgent:
    """
    Agent capable of participating in institutional structures.
    
    Extends basic agent with institutional awareness and
    role-playing capabilities.
    """
    
    def __init__(
        self,
        agent_id: int,
        seed: int = None
    ):
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed or agent_id)
        
        # Institutional memberships
        self.institution_memberships: Set[int] = set()
        self.coalition_memberships: Set[int] = set()
        self.current_roles: Set[int] = set()
        
        # Norm adoption
        self.adopted_norms: Set[int] = set()
        self.norm_compliance_history: Dict[int, List[bool]] = defaultdict(list)
        
        # Behavioral parameters
        self.conformity: float = self.rng.uniform(0.3, 0.7)  # Tendency to follow norms
        self.ambition: float = self.rng.uniform(0.2, 0.8)    # Desire for roles/status
        self.loyalty: float = self.rng.uniform(0.4, 0.9)     # Coalition loyalty
        
        # Interaction history
        self.cooperation_history: Dict[int, List[bool]] = defaultdict(list)
        self.reputation: float = 0.5
        
        # Statistics
        self.total_interactions: int = 0
        self.total_cooperations: int = 0
    
    def decide_norm_adoption(self, norm: Norm, social_pressure: float) -> bool:
        """Decide whether to adopt a norm."""
        # Higher conformity and social pressure increase adoption
        adoption_prob = self.conformity * social_pressure + (1 - self.conformity) * norm.adoption_rate
        return self.rng.random() < adoption_prob
    
    def decide_coalition_join(self, coalition: Coalition, benefit: float) -> bool:
        """Decide whether to join a coalition."""
        # Consider benefit and loyalty to existing coalitions
        if self.coalition_memberships:
            # Less likely to join new coalition if already in one
            join_prob = benefit * (1 - self.loyalty)
        else:
            join_prob = benefit
        return self.rng.random() < join_prob
    
    def decide_role_pursuit(self, role: Role, competition: float) -> bool:
        """Decide whether to pursue a role."""
        # Higher ambition increases pursuit, competition decreases it
        pursuit_prob = self.ambition * (1 - competition * 0.5)
        return self.rng.random() < pursuit_prob
    
    def comply_with_norm(self, norm: Norm) -> bool:
        """Decide whether to comply with a norm."""
        if norm.norm_id in self.adopted_norms:
            # High compliance if adopted
            comply_prob = 0.9
        else:
            # Lower compliance if not adopted
            comply_prob = self.conformity * norm.adoption_rate
        
        complied = self.rng.random() < comply_prob
        self.norm_compliance_history[norm.norm_id].append(complied)
        return complied
    
    def update_reputation(self, cooperated: bool) -> None:
        """Update reputation based on behavior."""
        self.total_interactions += 1
        if cooperated:
            self.total_cooperations += 1
        self.reputation = self.total_cooperations / self.total_interactions


class InstitutionalEmergenceTracker:
    """
    Tracks and analyzes institutional emergence in a population.
    
    Monitors:
    - Norm formation and spread
    - Coalition formation and stability
    - Role differentiation
    - Institutional complexity
    """
    
    def __init__(self, n_agents: int, seed: int = 42):
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        
        # Agents
        self.agents: Dict[int, InstitutionalAgent] = {
            i: InstitutionalAgent(i, seed + i) for i in range(n_agents)
        }
        
        # Emergent structures
        self.norms: Dict[int, Norm] = {}
        self.roles: Dict[int, Role] = {}
        self.coalitions: Dict[int, Coalition] = {}
        self.institutions: Dict[int, Institution] = {}
        
        # Counters
        self.next_norm_id = 0
        self.next_role_id = 0
        self.next_coalition_id = 0
        self.next_institution_id = 0
        
        # Interaction network
        self.interaction_graph = nx.Graph()
        self.interaction_graph.add_nodes_from(range(n_agents))
        
        # History
        self.round_history: List[Dict[str, Any]] = []
        self.current_round = 0
    
    def detect_norm_emergence(self, behavior_counts: Dict[str, int], threshold: float = 0.6) -> Optional[Norm]:
        """
        Detect if a new norm has emerged based on behavior patterns.
        
        A norm emerges when a behavior pattern is adopted by > threshold of population.
        """
        total = sum(behavior_counts.values())
        if total == 0:
            return None
        
        for behavior, count in behavior_counts.items():
            rate = count / total
            if rate >= threshold:
                # Check if this norm already exists
                existing = [n for n in self.norms.values() if n.behavior_pattern == behavior]
                if not existing:
                    # New norm emerged
                    norm = Norm(
                        norm_id=self.next_norm_id,
                        description=f"Norm: {behavior}",
                        behavior_pattern=behavior,
                        adoption_rate=rate,
                        emergence_round=self.current_round
                    )
                    self.norms[norm.norm_id] = norm
                    self.next_norm_id += 1
                    return norm
        
        return None
    
    def detect_coalition_formation(self, cooperation_matrix: np.ndarray, threshold: float = 0.7) -> List[Coalition]:
        """
        Detect coalition formation from cooperation patterns.
        
        Uses community detection on the cooperation network.
        """
        # Build cooperation graph
        G = nx.Graph()
        G.add_nodes_from(range(self.n_agents))
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if cooperation_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=cooperation_matrix[i, j])
        
        # Detect communities
        if G.number_of_edges() == 0:
            return []
        
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except Exception:
            return []
        
        new_coalitions = []
        for community in communities:
            if len(community) >= 3:  # Minimum coalition size
                # Check if this coalition already exists
                existing = False
                for coalition in self.coalitions.values():
                    overlap = len(coalition.members & community) / len(coalition.members | community)
                    if overlap > 0.8:
                        existing = True
                        break
                
                if not existing:
                    coalition = Coalition(
                        coalition_id=self.next_coalition_id,
                        members=set(community),
                        formation_round=self.current_round
                    )
                    self.coalitions[coalition.coalition_id] = coalition
                    self.next_coalition_id += 1
                    new_coalitions.append(coalition)
                    
                    # Update agent memberships
                    for agent_id in community:
                        self.agents[agent_id].coalition_memberships.add(coalition.coalition_id)
        
        return new_coalitions
    
    def detect_role_differentiation(self, agent_behaviors: Dict[int, Dict[str, float]]) -> List[Role]:
        """
        Detect role differentiation based on behavioral specialization.
        
        Roles emerge when agents consistently exhibit distinct behavior patterns.
        """
        # Cluster agents by behavior
        behavior_vectors = []
        agent_ids = []
        
        for agent_id, behaviors in agent_behaviors.items():
            vector = [
                behaviors.get("cooperation_rate", 0.5),
                behaviors.get("initiation_rate", 0.5),
                behaviors.get("enforcement_rate", 0.5),
            ]
            behavior_vectors.append(vector)
            agent_ids.append(agent_id)
        
        if len(behavior_vectors) < 3:
            return []
        
        behavior_array = np.array(behavior_vectors)
        
        # Simple clustering: identify extremes
        new_roles = []
        
        # High cooperators -> "Cooperator" role
        high_coop = np.where(behavior_array[:, 0] > 0.8)[0]
        if len(high_coop) >= 2 and "cooperator" not in [r.name for r in self.roles.values()]:
            role = Role(
                role_id=self.next_role_id,
                name="cooperator",
                responsibilities=["maintain_cooperation"],
                current_occupants={agent_ids[i] for i in high_coop}
            )
            self.roles[role.role_id] = role
            self.next_role_id += 1
            new_roles.append(role)
        
        # High enforcers -> "Enforcer" role
        high_enforce = np.where(behavior_array[:, 2] > 0.7)[0]
        if len(high_enforce) >= 1 and "enforcer" not in [r.name for r in self.roles.values()]:
            role = Role(
                role_id=self.next_role_id,
                name="enforcer",
                responsibilities=["punish_defectors"],
                privileges=["enforcement_authority"],
                current_occupants={agent_ids[i] for i in high_enforce}
            )
            self.roles[role.role_id] = role
            self.next_role_id += 1
            new_roles.append(role)
        
        return new_roles
    
    def detect_institution_formation(
        self,
        coalitions: List[Coalition],
        norms: List[Norm],
        roles: List[Role]
    ) -> Optional[Institution]:
        """
        Detect institution formation from combination of structures.
        
        An institution forms when:
        - A stable coalition exists
        - With established norms
        - And differentiated roles
        """
        for coalition in coalitions:
            if coalition.size() < 5:
                continue
            if coalition.stability_score < 0.7:
                continue
            
            # Check for associated norms
            coalition_norms = [n for n in norms if n.adoption_rate > 0.6]
            if not coalition_norms:
                continue
            
            # Check for roles
            coalition_roles = [r for r in roles if r.current_occupants & coalition.members]
            if not coalition_roles:
                continue
            
            # Institution can form
            institution = Institution(
                institution_id=self.next_institution_id,
                name=f"Institution_{self.next_institution_id}",
                institution_type=InstitutionType.COALITION,
                members=coalition.members.copy(),
                formation_round=self.current_round
            )
            
            # Add rules from norms
            for norm in coalition_norms:
                institution.add_rule({
                    "type": "norm",
                    "norm_id": norm.norm_id,
                    "description": norm.description
                })
            
            self.institutions[institution.institution_id] = institution
            self.next_institution_id += 1
            
            return institution
        
        return None
    
    def run_round(self, interactions: List[Tuple[int, int, bool, bool]]) -> Dict[str, Any]:
        """
        Process one round of interactions and detect emergence.
        
        Args:
            interactions: List of (agent_a, agent_b, a_cooperated, b_cooperated)
        
        Returns:
            Round summary with detected emergent structures
        """
        self.current_round += 1
        
        # Update interaction network
        cooperation_matrix = np.zeros((self.n_agents, self.n_agents))
        behavior_counts = defaultdict(int)
        agent_behaviors = defaultdict(lambda: defaultdict(float))
        
        for a, b, a_coop, b_coop in interactions:
            # Update cooperation matrix
            if a_coop:
                cooperation_matrix[a, b] += 1
            if b_coop:
                cooperation_matrix[b, a] += 1
            
            # Track behaviors
            if a_coop and b_coop:
                behavior_counts["mutual_cooperation"] += 1
            elif not a_coop and not b_coop:
                behavior_counts["mutual_defection"] += 1
            else:
                behavior_counts["mixed"] += 1
            
            # Update agent behaviors
            agent_behaviors[a]["cooperation_rate"] = (
                agent_behaviors[a].get("cooperation_rate", 0) * 0.9 + 
                (1.0 if a_coop else 0.0) * 0.1
            )
            agent_behaviors[b]["cooperation_rate"] = (
                agent_behaviors[b].get("cooperation_rate", 0) * 0.9 + 
                (1.0 if b_coop else 0.0) * 0.1
            )
            
            # Update agent reputations
            self.agents[a].update_reputation(a_coop)
            self.agents[b].update_reputation(b_coop)
        
        # Normalize cooperation matrix
        max_val = cooperation_matrix.max()
        if max_val > 0:
            cooperation_matrix /= max_val
        
        # Detect emergent structures
        new_norm = self.detect_norm_emergence(behavior_counts)
        new_coalitions = self.detect_coalition_formation(cooperation_matrix)
        new_roles = self.detect_role_differentiation(dict(agent_behaviors))
        
        # Update coalition stability
        for coalition in self.coalitions.values():
            internal_coop = 0
            internal_total = 0
            for a in coalition.members:
                for b in coalition.members:
                    if a != b:
                        internal_coop += cooperation_matrix[a, b]
                        internal_total += 1
            
            if internal_total > 0:
                coalition.internal_cooperation_rate = internal_coop / internal_total
                coalition.stability_score = coalition.internal_cooperation_rate
        
        # Check for institution formation
        new_institution = self.detect_institution_formation(
            list(self.coalitions.values()),
            list(self.norms.values()),
            list(self.roles.values())
        )
        
        # Record round summary
        summary = {
            "round": self.current_round,
            "n_norms": len(self.norms),
            "n_coalitions": len(self.coalitions),
            "n_roles": len(self.roles),
            "n_institutions": len(self.institutions),
            "new_norm": new_norm.norm_id if new_norm else None,
            "new_coalitions": [c.coalition_id for c in new_coalitions],
            "new_roles": [r.role_id for r in new_roles],
            "new_institution": new_institution.institution_id if new_institution else None,
            "avg_cooperation": np.mean(cooperation_matrix),
        }
        
        self.round_history.append(summary)
        return summary
    
    def get_institutional_complexity(self) -> float:
        """
        Calculate overall institutional complexity score.
        
        Higher score indicates more developed institutional structure.
        """
        # Weight different components
        norm_weight = 0.2
        coalition_weight = 0.3
        role_weight = 0.2
        institution_weight = 0.3
        
        # Normalize counts
        norm_score = min(len(self.norms) / 5, 1.0)
        coalition_score = min(len(self.coalitions) / 3, 1.0)
        role_score = min(len(self.roles) / 4, 1.0)
        institution_score = min(len(self.institutions) / 2, 1.0)
        
        complexity = (
            norm_weight * norm_score +
            coalition_weight * coalition_score +
            role_weight * role_score +
            institution_weight * institution_score
        )
        
        return complexity
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of institutional emergence."""
        return {
            "total_rounds": self.current_round,
            "n_agents": self.n_agents,
            "norms": {
                "count": len(self.norms),
                "established": sum(1 for n in self.norms.values() if n.is_established()),
                "details": [
                    {"id": n.norm_id, "pattern": n.behavior_pattern, "adoption": n.adoption_rate}
                    for n in self.norms.values()
                ]
            },
            "coalitions": {
                "count": len(self.coalitions),
                "stable": sum(1 for c in self.coalitions.values() if c.is_stable()),
                "avg_size": np.mean([c.size() for c in self.coalitions.values()]) if self.coalitions else 0,
            },
            "roles": {
                "count": len(self.roles),
                "names": [r.name for r in self.roles.values()],
            },
            "institutions": {
                "count": len(self.institutions),
                "types": [i.institution_type.name for i in self.institutions.values()],
            },
            "complexity_score": self.get_institutional_complexity(),
        }


def run_institutional_emergence_experiment(
    n_agents: int = 50,
    n_rounds: int = 200,
    cooperation_bias: float = 0.5,
    seed: int = 42
) -> Tuple[InstitutionalEmergenceTracker, Dict[str, Any]]:
    """
    Run an institutional emergence experiment.
    
    Args:
        n_agents: Number of agents
        n_rounds: Number of rounds to simulate
        cooperation_bias: Base probability of cooperation
        seed: Random seed
    
    Returns:
        (tracker, summary)
    """
    rng = np.random.default_rng(seed)
    tracker = InstitutionalEmergenceTracker(n_agents, seed)
    
    for round_num in range(n_rounds):
        # Generate interactions
        interactions = []
        n_interactions = n_agents * 2  # Average 4 interactions per agent
        
        for _ in range(n_interactions):
            a, b = rng.choice(n_agents, size=2, replace=False)
            
            # Cooperation decision based on reputation and bias
            a_coop_prob = cooperation_bias + 0.3 * (tracker.agents[b].reputation - 0.5)
            b_coop_prob = cooperation_bias + 0.3 * (tracker.agents[a].reputation - 0.5)
            
            a_coop = rng.random() < a_coop_prob
            b_coop = rng.random() < b_coop_prob
            
            interactions.append((a, b, a_coop, b_coop))
        
        tracker.run_round(interactions)
        
        if (round_num + 1) % 50 == 0:
            print(f"Round {round_num + 1}: complexity = {tracker.get_institutional_complexity():.3f}")
    
    return tracker, tracker.get_summary()
