"""
Decomposition Structure for Experiment 30.

This structure allows independent control of:
1. Knowledge pooling (individual vs shared)
2. Reputation pooling (individual vs collective)

Used to isolate which factor drives Ubuntu's dominance.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

from .base import BaseStructure
from ..agents.agent import Agent, Task, TaskOutcome, Template


@dataclass
class DecompositionConfig:
    """Configuration for decomposition experiment."""
    
    # Pooling settings
    knowledge_pooling: bool = False  # True = share all templates
    reputation_pooling: bool = False  # True = collective reputation
    
    # Standard parameters (from MeritocracyConfig baseline)
    success_reward: float = 0.1
    failure_penalty: float = 0.3
    redemption_bonus: float = 0.2
    
    @property
    def name(self) -> str:
        """Generate descriptive name based on settings."""
        k = "K-Pool" if self.knowledge_pooling else "K-Indiv"
        r = "R-Pool" if self.reputation_pooling else "R-Indiv"
        return f"{k}_{r}"


class DecompositionStructure(BaseStructure):
    """
    Parameterized structure for decomposition study.
    
    Allows independent control of:
    - Knowledge pooling: Whether templates are shared with all agents
    - Reputation pooling: Whether all agents share one collective reputation
    
    This isolates the effects of each factor to understand what drives
    Ubuntu's dominance in Experiments 27-29.
    """
    
    def __init__(self, config: DecompositionConfig = None):
        self.config = config or DecompositionConfig()
        self.collective_reputation = 0.5  # Used if reputation_pooling=True
        self._name = self.config.name
    
    @property
    def name(self) -> str:
        return self._name
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        GCL-COMPATIBLE: Agents CHOOSE based on capability.
        
        Volunteering logic is the same regardless of pooling settings.
        Most capable agents volunteer for harder tasks.
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        for agent in available:
            # Agent assesses capability vs task difficulty
            if agent.effective_capability > task.difficulty * 0.5:
                volunteers.append(agent)
        
        if not volunteers:
            # Fallback: most capable available
            return [max(available, key=lambda a: a.effective_capability)]
        
        # Most capable volunteers (serves the collective or self)
        return [max(volunteers, key=lambda a: a.effective_capability)]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """
        Process outcome based on reputation pooling setting.
        
        If reputation_pooling=True: All agents share collective reputation
        If reputation_pooling=False: Individual reputation updates
        """
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        n_agents = len(all_agents)
        
        if self.config.reputation_pooling:
            # COLLECTIVE REPUTATION (Ubuntu-style)
            if outcome.success:
                # Collective gains (difficulty-weighted, divided by N)
                delta = (self.config.success_reward * difficulty_mult) / n_agents
                self.collective_reputation = min(1.0, self.collective_reputation + delta)
                
                # Template learning (always individual first)
                agent.learn_template(outcome.task, True)
            else:
                # Collective absorbs failure (difficulty-weighted, divided by N)
                delta = (self.config.failure_penalty / difficulty_mult) / n_agents
                self.collective_reputation = max(0.0, self.collective_reputation - delta)
                
                # Community restoration bonus
                agent.reputation += self.config.redemption_bonus
                agent.recovered_recently = True
            
            # Sync all agents to collective reputation
            for a in all_agents:
                a.reputation = self.collective_reputation
        else:
            # INDIVIDUAL REPUTATION (Meritocracy-style)
            if outcome.success:
                # Difficulty-weighted reward
                delta = self.config.success_reward * difficulty_mult
                agent.reputation = min(1.0, agent.reputation + delta)
                
                # Template learning
                agent.learn_template(outcome.task, True)
                
                # Check for redemption bonus
                if agent.is_struggling():
                    agent.reputation += self.config.redemption_bonus
                    agent.recovered_recently = True
            else:
                # Difficulty-weighted penalty (harder = less penalty)
                delta = self.config.failure_penalty / difficulty_mult
                agent.reputation = max(0.0, agent.reputation - delta)
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """
        Share knowledge based on pooling setting.
        
        If knowledge_pooling=True: Share ALL templates with ALL agents
        If knowledge_pooling=False: No automatic sharing
        """
        if self.config.knowledge_pooling:
            # FULL POOLING: Share all templates with all agents (Ubuntu-style)
            all_template_ids = set()
            template_map = {}
            
            # Collect all unique templates
            for agent in agents:
                for t in agent.template_library:
                    if t.id not in all_template_ids:
                        all_template_ids.add(t.id)
                        template_map[t.id] = t
            
            # Share all with all (only new ones)
            for agent in agents:
                agent_template_ids = {t.id for t in agent.template_library}
                for tid, template in template_map.items():
                    if tid not in agent_template_ids:
                        agent.template_library.append(template)
        # else: No automatic sharing (individual knowledge)
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        """Standard output aggregation."""
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        """
        Explicit knowledge transfer (used by check_obligations).
        
        In pooled mode, this is redundant (already shared).
        In individual mode, this allows targeted sharing.
        """
        if from_agent.template_library:
            template = random.choice(from_agent.template_library)
            return to_agent.receive_template(template)
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        """
        Calculate status based on reputation.
        
        In pooled mode, all agents have same reputation (same status).
        In individual mode, status varies by percentile.
        """
        if self.config.reputation_pooling:
            # All agents have same reputation, so status based on contribution
            recent_successes = len(agent.success_history[-10:]) if agent.success_history else 0
            if recent_successes >= 7:
                return "elite"
            elif recent_successes >= 3:
                return "middle"
            else:
                return "struggling"
        else:
            # Individual reputation percentile
            return self.status_from_percentile(agent, all_agents)
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        """
        Check obligations based on status.
        
        Elite agents help struggling agents (in both modes).
        """
        if agent.is_elite():
            struggling = [a for a in all_agents if a.is_struggling() and a.id != agent.id]
            for s in struggling[:2]:  # Help up to 2
                self.transfer_knowledge(agent, s)


def create_decomposition_conditions():
    """
    Create the 4 conditions for the 2x2 factorial design.
    
    Returns dict mapping condition name to DecompositionStructure.
    """
    conditions = {
        "baseline": DecompositionConfig(
            knowledge_pooling=False,
            reputation_pooling=False
        ),
        "k_pool": DecompositionConfig(
            knowledge_pooling=True,
            reputation_pooling=False
        ),
        "r_pool": DecompositionConfig(
            knowledge_pooling=False,
            reputation_pooling=True
        ),
        "full_pool": DecompositionConfig(
            knowledge_pooling=True,
            reputation_pooling=True
        ),
    }
    
    return {name: DecompositionStructure(config) for name, config in conditions.items()}
