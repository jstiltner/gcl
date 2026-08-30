"""
Abstract base class for social structures.

GCL-COMPATIBLE DESIGN:
Each structure implements different approaches to:
- Agent volunteering (agents CHOOSE tasks, not assigned)
- Outcome processing with difficulty-weighted reputation
- Output aggregation
- Knowledge transfer
- Status calculation
- Obligation checking

Key GCL principle: Agents learn what commitments to make via π(s) → C
Structures provide INFORMATION and INCENTIVES, not forced assignment.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from ..agents.agent import Agent, Task, TaskOutcome, Template


class SocialStructure(ABC):
    """
    Abstract base for all social structures.
    
    GCL-COMPATIBLE: Structures influence agent behavior through:
    - Information (what tasks are available, partner reputations)
    - Incentives (difficulty-weighted reputation, status rewards)
    
    NOT through forced assignment (which would violate GCL's core tenet).
    """
    
    name: str = "base"
    
    @abstractmethod
    def get_volunteers(self, task: 'Task', agents: List['Agent']) -> List['Agent']:
        """
        Get agents who CHOOSE to volunteer for this task.
        
        GCL-COMPATIBLE: Agents decide based on:
        - Task difficulty vs their capability
        - Expected reputation gain (difficulty-weighted)
        - Social context (partner reputations)
        
        Returns list of agents who volunteered.
        """
        pass
    
    @abstractmethod
    def process_outcome_with_difficulty(
        self,
        agent: 'Agent',
        outcome: 'TaskOutcome',
        all_agents: List['Agent'],
        difficulty: float
    ):
        """
        Update agent state with difficulty-weighted reputation.
        
        GCL-COMPATIBLE anti-gaming mechanism:
        - Harder tasks give MORE reputation on success
        - Easier tasks give LESS reputation on success
        - This incentivizes harder tasks without forcing assignment
        """
        pass
    
    @abstractmethod
    def aggregate_outputs(self, outcomes: List['TaskOutcome']) -> float:
        """
        How does this structure combine agent outputs into firm output?
        """
        pass
    
    @abstractmethod
    def transfer_knowledge(self, from_agent: 'Agent', to_agent: 'Agent') -> bool:
        """
        How does knowledge transfer work in this structure?
        Returns True if transfer occurred.
        """
        pass
    
    @abstractmethod
    def calculate_status(self, agent: 'Agent', all_agents: List['Agent']) -> str:
        """
        How does this structure assign status?
        Returns "elite", "middle", or "struggling".
        """
        pass
    
    @abstractmethod
    def check_obligations(self, agent: 'Agent', all_agents: List['Agent']):
        """
        Does this agent have obligations to fulfill?
        May trigger helping behavior.
        """
        pass
    
    def run_round_start(self, agents: List['Agent']):
        """Called at the start of each round. Reset tracking, update tenure."""
        for agent in agents:
            agent.reset_tracking()
            agent.increment_tenure()
    
    def run_round_end(self, agents: List['Agent']):
        """Called at the end of each round. Update status, check obligations."""
        for agent in agents:
            new_status = self.calculate_status(agent, agents)
            agent.update_status(new_status)
        
        for agent in agents:
            self.check_obligations(agent, agents)
    
    def maybe_share_knowledge(self, agents: List['Agent']):
        """
        Structure-specific knowledge sharing at end of round.
        Override in subclasses.
        """
        pass


class BaseStructure(SocialStructure):
    """
    Base implementation with common utilities.
    
    GCL-COMPATIBLE: Uses agent choice + difficulty-weighted reputation.
    Subclasses override specific methods.
    """
    
    # Difficulty-weighted reputation parameters (from Experiment 27)
    difficulty_reputation_multiplier: float = 2.0  # 1x to 3x scaling
    
    def __init__(self, config):
        self.config = config
    
    def get_available_agents(self, agents: List['Agent']) -> List['Agent']:
        """Get agents that can participate (have resources)."""
        return [a for a in agents if a.resources > 0]
    
    def get_volunteers(self, task: 'Task', agents: List['Agent']) -> List['Agent']:
        """
        GCL-COMPATIBLE: Agents CHOOSE to volunteer based on task assessment.
        
        Default implementation: Agents volunteer if they assess they can handle
        the task (capability > difficulty * threshold).
        
        Subclasses can override to add structure-specific volunteering logic.
        """
        available = self.get_available_agents(agents)
        volunteers = []
        
        for agent in available:
            # Agent assesses task difficulty vs their capability
            # Higher capability agents more likely to volunteer for harder tasks
            # This is GCL-compatible: agent CHOOSES based on their assessment
            capability_threshold = task.difficulty * 0.8  # 80% of difficulty
            
            if agent.effective_capability >= capability_threshold:
                # Agent believes they can handle it
                # Social awareness affects willingness (from Exp 25-26b)
                if agent.awareness_level == "social":
                    # Social agents consider expected reputation gain
                    # Harder tasks = more reputation (difficulty-weighted)
                    expected_gain = task.difficulty * self.difficulty_reputation_multiplier
                    if expected_gain > 0.1:  # Worth the effort
                        volunteers.append(agent)
                else:
                    # Non-social agents just check capability
                    volunteers.append(agent)
            elif random.random() < 0.1:
                # Small chance to volunteer anyway (exploration)
                volunteers.append(agent)
        
        # Limit to reasonable team size
        max_team = min(3, len(volunteers))
        if len(volunteers) > max_team:
            # Prefer higher capability agents
            volunteers = sorted(volunteers, key=lambda a: a.effective_capability, reverse=True)[:max_team]
        
        return volunteers
    
    def process_outcome_with_difficulty(
        self,
        agent: 'Agent',
        outcome: 'TaskOutcome',
        all_agents: List['Agent'],
        difficulty: float
    ):
        """
        GCL-COMPATIBLE: Update reputation with difficulty weighting.
        
        From Experiment 27 (best GCL-compatible anti-gaming mechanism):
        - Harder tasks give MORE reputation on success (1x to 3x)
        - Easier tasks give LESS reputation on success
        - Failure penalty is inversely weighted (hard failures hurt less)
        
        This incentivizes harder tasks without forcing assignment.
        """
        # Calculate difficulty multiplier
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            # Harder tasks give more reputation
            base_reward = self.config.success_reward
            rep_gain = base_reward * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + rep_gain)
            
            # Check for redemption bonus
            self.check_redemption(agent, outcome)
        else:
            # Harder task failures are less punishing
            base_penalty = self.config.failure_penalty
            rep_loss = base_penalty / difficulty_mult
            agent.reputation = max(0.0, agent.reputation - rep_loss)
    
    def apply_success_reward(self, agent: 'Agent'):
        """Apply standard success reward (use process_outcome_with_difficulty instead)."""
        agent.reputation += self.config.success_reward
        agent.reputation = min(1.0, agent.reputation)
    
    def apply_failure_penalty(self, agent: 'Agent'):
        """Apply standard failure penalty (use process_outcome_with_difficulty instead)."""
        agent.reputation -= self.config.failure_penalty
        agent.reputation = max(0.0, agent.reputation)
    
    def check_redemption(self, agent: 'Agent', outcome: 'TaskOutcome'):
        """Check if agent qualifies for redemption bonus."""
        # Agent was struggling and succeeded
        if agent.is_struggling() and outcome.success:
            agent.reputation += self.config.redemption_bonus
            agent.recovered_recently = True
    
    def status_from_percentile(self, agent: 'Agent', all_agents: List['Agent']) -> str:
        """Calculate status based on reputation percentile."""
        reps = [a.reputation for a in all_agents]
        percentile = sum(r < agent.reputation for r in reps) / len(reps) if reps else 0.5
        
        if percentile > 0.8:
            return "elite"
        elif percentile > 0.3:
            return "middle"
        else:
            return "struggling"
