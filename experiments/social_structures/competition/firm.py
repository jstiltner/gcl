"""
Firm class - a group of agents with an internal social structure
competing in the capitalist arena.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..structures.base import SocialStructure
    from ..agents.agent import Agent, Task, TaskOutcome


@dataclass
class FirmTaskOutcome:
    """Result of a firm attempting a task."""
    firm: 'Firm'
    task: 'Task'
    output: float
    agent_outcomes: List['TaskOutcome']
    success: bool = False


@dataclass
class Firm:
    """
    A firm with internal structure competing in capitalism.
    
    The firm has:
    - A social structure that governs internal coordination
    - A population of agents
    - Resources that can grow or shrink based on competition
    """
    
    id: str
    structure: 'SocialStructure'
    agents: List['Agent']
    resources: float = 10.0  # Starting capital
    market_share: float = 0.0
    
    # History
    wins: int = 0
    losses: int = 0
    tasks_attempted: int = 0
    bankruptcies_survived: int = 0
    
    def attempt_task(self, task: 'Task') -> FirmTaskOutcome:
        """
        Firm attempts task using internal coordination.
        
        GCL-COMPATIBLE DESIGN (agents CHOOSE, not assigned):
        1. Structure provides INFORMATION about task (difficulty, requirements)
        2. Agents CHOOSE whether to volunteer based on their assessment
        3. Structure provides INCENTIVES (difficulty-weighted reputation)
        4. Agents attempt the task with social awareness context
        5. Structure processes outcomes
        6. Aggregate outputs for competition
        
        This preserves GCL's core tenet: π(s) → C (agents learn what commitments to make)
        Anti-gaming is achieved through difficulty-weighted reputation, not forced assignment.
        """
        from ..agents.agent import TaskOutcome
        
        # Agents CHOOSE to volunteer for task (GCL-compatible)
        # Structure provides information, agents decide
        volunteers = self.structure.get_volunteers(task, self.agents)
        
        if not volunteers:
            return FirmTaskOutcome(
                firm=self,
                task=task,
                output=0.0,
                agent_outcomes=[],
                success=False
            )
        
        # Calculate average reputation for social awareness context
        avg_reputation = self.get_average_reputation()
        
        # Agents attempt with social awareness context
        # Partner reputation enables trust-based effort adjustment
        outcomes = []
        for i, agent in enumerate(volunteers):
            # Get partner reputation (use average of other volunteers, or firm average)
            if len(volunteers) > 1:
                other_reps = [a.reputation for j, a in enumerate(volunteers) if j != i]
                partner_rep = sum(other_reps) / len(other_reps)
            else:
                partner_rep = avg_reputation
            
            outcome = agent.attempt_task(task, partner_reputation=partner_rep)
            outcomes.append(outcome)
        
        # Update internal state with difficulty-weighted reputation (GCL-compatible anti-gaming)
        for outcome in outcomes:
            self.structure.process_outcome_with_difficulty(
                outcome.agent, outcome, self.agents, task.difficulty
            )
        
        # Aggregate (structure-dependent)
        firm_output = self.structure.aggregate_outputs(outcomes)
        
        # Track
        self.tasks_attempted += 1
        any_success = any(o.success for o in outcomes)
        
        return FirmTaskOutcome(
            firm=self,
            task=task,
            output=firm_output,
            agent_outcomes=outcomes,
            success=any_success
        )
    
    def run_internal_round(self):
        """Run internal structure updates (start of round)."""
        self.structure.run_round_start(self.agents)
    
    def end_internal_round(self):
        """Run internal structure updates (end of round)."""
        self.structure.run_round_end(self.agents)
        self.structure.maybe_share_knowledge(self.agents)
    
    def is_bankrupt(self) -> bool:
        """Check if firm is bankrupt."""
        return self.resources <= 0
    
    def add_resources(self, amount: float):
        """Add resources (from winning)."""
        self.resources += amount
        if amount > 0:
            self.wins += 1
    
    def remove_resources(self, amount: float):
        """Remove resources (from losing)."""
        self.resources -= amount
        if amount > 0:
            self.losses += 1
    
    def get_average_reputation(self) -> float:
        """Get average agent reputation."""
        if not self.agents:
            return 0.0
        return sum(a.reputation for a in self.agents) / len(self.agents)
    
    def get_average_capability(self) -> float:
        """Get average agent effective capability."""
        if not self.agents:
            return 0.0
        return sum(a.effective_capability for a in self.agents) / len(self.agents)
    
    def get_total_templates(self) -> int:
        """Get total templates across all agents."""
        return sum(len(a.template_library) for a in self.agents)
    
    def get_unique_templates(self) -> int:
        """Get unique templates (knowledge diffusion measure)."""
        all_templates = set()
        for agent in self.agents:
            all_templates.update(t.id for t in agent.template_library)
        return len(all_templates)
    
    def get_gini_coefficient(self) -> float:
        """Calculate Gini coefficient of reputation distribution."""
        reps = sorted([a.reputation for a in self.agents])
        n = len(reps)
        if n == 0 or sum(reps) == 0:
            return 0.0
        
        cumsum = sum((i + 1) * r for i, r in enumerate(reps))
        return (2 * cumsum) / (n * sum(reps)) - (n + 1) / n
    
    def get_status_distribution(self) -> dict:
        """Get distribution of agent statuses."""
        dist = {"elite": 0, "middle": 0, "struggling": 0}
        for agent in self.agents:
            dist[agent.status_level] = dist.get(agent.status_level, 0) + 1
        return dist
    
    def __repr__(self):
        return f"Firm({self.id}, {self.structure.name}, resources={self.resources:.2f})"


def create_firm(
    firm_id: str,
    structure_type: str,
    n_agents: int = 30,
    starting_resources: float = 10.0,
    capability_mean: float = 0.5,
    capability_std: float = 0.15
) -> Firm:
    """
    Create a firm with the specified structure and agents.
    """
    from ..structures.implementations import create_structure
    from ..agents.agent import create_population
    
    structure = create_structure(structure_type)
    agents = create_population(n_agents, prefix=f"{firm_id}_agent", 
                               capability_mean=capability_mean,
                               capability_std=capability_std)
    
    return Firm(
        id=firm_id,
        structure=structure,
        agents=agents,
        resources=starting_resources
    )
