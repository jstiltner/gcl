"""
Agent class for social structures experiment.

Agents have base capabilities, learnable templates, and social state
that is interpreted differently by different social structures.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..structures.base import SocialStructure


@dataclass
class Template:
    """
    Learnable strategy/knowledge that improves task performance.
    Templates can be shared between agents according to structure rules.
    """
    
    id: str
    capability_boost: float  # How much it improves performance (0.05-0.2)
    task_type_affinity: Optional[str] = None  # Specialization
    source_agent: Optional[str] = None  # Who it came from
    
    def __eq__(self, other):
        if not isinstance(other, Template):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Task:
    """A task that agents/firms compete on."""
    
    id: str
    difficulty: float  # 0-1
    reward: float  # Resources gained if won
    task_type: str = "general"


@dataclass
class TaskOutcome:
    """Result of a task attempt."""
    
    success: bool
    agent: 'Agent'
    task: Task
    output_quality: float = 0.0  # For competition comparison


@dataclass
class Agent:
    """
    Base agent that operates within a social structure.
    
    The agent has fixed base capability and can acquire templates
    that improve performance. Social state (reputation, status, obligations)
    is interpreted differently by different structures.
    
    GCL-COMPATIBLE DESIGN (from Experiments 25-27):
    - Agents have SOCIAL awareness (can see own + others' reputations)
    - Agents CHOOSE tasks (not assigned) - preserves GCL's core tenet: π(s) → C
    - Anti-gaming via difficulty-weighted reputation (not peer assignment)
    - This enables trust-based cooperation while preserving agent autonomy
    """
    
    id: str
    
    # Capabilities
    base_capability: float  # 0-1, innate ability (fixed at birth)
    learned_capability: float = 0.0  # From template acquisition
    
    # Templates (learnable strategies)
    template_library: List[Template] = field(default_factory=list)
    
    # State
    reputation: float = 0.5  # Current reputation (interpretation varies by structure)
    resources: float = 1.0  # Current resources
    
    # History
    success_history: List[bool] = field(default_factory=list)
    failure_history: List[bool] = field(default_factory=list)
    
    # Social
    obligations_owed_to: Dict[str, float] = field(default_factory=dict)  # agent_id -> amount
    obligations_owed_by: Dict[str, float] = field(default_factory=dict)  # agent_id -> amount
    guild_id: Optional[str] = None
    
    # Status (assigned by structure)
    status_level: str = "middle"  # "elite", "middle", "struggling"
    previous_status: str = "middle"  # For tracking mobility
    
    # Tenure (for guild progression)
    tenure: int = 0  # Rounds in the system
    guild_rank: str = "apprentice"  # "apprentice", "journeyman", "master"
    
    # Redemption state (for monastic)
    in_redemption: bool = False
    redemption_progress: int = 0
    
    # Rotation state
    current_role: str = "worker"  # "leader", "worker"
    rounds_in_role: int = 0
    
    # Tracking
    status_changed_recently: bool = False
    attempted_hard_task_recently: bool = False
    recovered_recently: bool = False
    helped_recently: bool = False
    was_helped_recently: bool = False
    
    # Social Awareness (from Exp 25-26b)
    awareness_level: str = "social"  # "blind", "self", "social", "strategic"
    effort_level: float = 0.8  # Base effort level
    
    @property
    def effective_capability(self) -> float:
        """Total capability including template boosts."""
        template_boost = sum(t.capability_boost for t in self.template_library)
        return min(1.0, self.base_capability + self.learned_capability + template_boost)
    
    def get_own_reputation(self) -> float:
        """Get own reputation based on awareness level."""
        if self.awareness_level == "blind":
            return 0.5  # No awareness
        return self.reputation
    
    def get_partner_reputation(self, partner_rep: float) -> float:
        """Get partner's reputation based on awareness level."""
        if self.awareness_level in ["blind", "self"]:
            return 0.5  # No awareness of others
        return partner_rep
    
    def calculate_effort(self, task: Task, partner_reputation: float = 0.5) -> float:
        """
        Calculate effort level based on social awareness.
        From Experiment 26b: Social awareness enables trust-based effort adjustment.
        """
        base_effort = 0.8
        my_rep = self.get_own_reputation()
        partner_rep = self.get_partner_reputation(partner_reputation)
        
        # Blind: Always full effort (no strategic adjustment)
        if self.awareness_level == "blind":
            return base_effort
        
        # Self-aware: Adjust effort based on own reputation
        if self.awareness_level == "self":
            if my_rep < 0.3:
                # Low rep: try harder to recover
                return min(1.0, base_effort + 0.15)
            elif my_rep > 0.7:
                # High rep: can coast a bit
                return base_effort - 0.1
            return base_effort
        
        # Social-aware: Consider partner reputation (OPTIMAL from Exp 26b)
        if self.awareness_level == "social":
            # Trust high-rep partners more, invest more effort
            if partner_rep > 0.6:
                return min(1.0, base_effort + 0.1)
            elif partner_rep < 0.3:
                # Low-rep partner: still try but less invested
                return base_effort - 0.1
            return base_effort
        
        # Strategic: Full optimization
        if self.awareness_level == "strategic":
            if my_rep < 0.3:
                # Must recover reputation
                return min(1.0, base_effort + 0.2)
            elif my_rep > 0.7 and partner_rep < 0.4:
                # High rep, low-value partner: minimal effort
                return base_effort - 0.15
            elif partner_rep > 0.6:
                # Good partner: invest more
                return min(1.0, base_effort + 0.1)
            return base_effort
        
        return base_effort
    
    def attempt_task(self, task: Task, partner_reputation: float = 0.5) -> TaskOutcome:
        """
        Attempt a task. Success probability based on capability, difficulty, and effort.
        
        Updated from Exp 26b: Effort is calculated based on social awareness.
        Social awareness + peer assignment enables trust-based cooperation.
        """
        # Track hard task attempts
        self.attempted_hard_task_recently = task.difficulty > 0.6
        
        # Calculate effort based on social awareness (from Exp 26b)
        effort = self.calculate_effort(task, partner_reputation)
        self.effort_level = effort  # Store for metrics
        
        # Calculate success probability (effort affects success)
        success_prob = self.effective_capability * effort * (1 - task.difficulty * 0.6)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        
        # Calculate output quality (for competition) - effort affects quality
        if success:
            output_quality = self.effective_capability * effort * (1 + random.uniform(-0.1, 0.1))
        else:
            output_quality = 0.0
        
        # Update history
        if success:
            self.success_history.append(True)
        else:
            self.failure_history.append(True)
        
        return TaskOutcome(
            success=success,
            agent=self,
            task=task,
            output_quality=output_quality
        )
    
    def learn_template(self, task: Task, success: bool) -> Optional[Template]:
        """
        Generate template from successful experience.
        10% chance to crystallize learning into a template.
        """
        if success and random.random() < 0.1:
            template = Template(
                id=f"template_{self.id}_{len(self.template_library)}_{random.randint(0, 10000)}",
                capability_boost=random.uniform(0.05, 0.15),
                source_agent=self.id,
                task_type_affinity=task.task_type
            )
            self.template_library.append(template)
            return template
        return None
    
    def receive_template(self, template: Template) -> bool:
        """
        Receive a template from another agent.
        Returns True if template was new and added.
        """
        if template not in self.template_library:
            self.template_library.append(template)
            return True
        return False
    
    def update_status(self, new_status: str):
        """Update status and track changes."""
        self.previous_status = self.status_level
        self.status_changed_recently = (self.status_level != new_status)
        self.status_level = new_status
    
    def increment_tenure(self):
        """Increment tenure counter."""
        self.tenure += 1
        self.rounds_in_role += 1
    
    def reset_tracking(self):
        """Reset per-round tracking flags."""
        self.status_changed_recently = False
        self.attempted_hard_task_recently = False
        self.recovered_recently = False
        self.helped_recently = False
        self.was_helped_recently = False
    
    def total_obligations_owed_to_me(self) -> float:
        """Total obligations others owe to this agent."""
        return sum(self.obligations_owed_by.values())
    
    def total_obligations_i_owe(self) -> float:
        """Total obligations this agent owes to others."""
        return sum(self.obligations_owed_to.values())
    
    def is_struggling(self) -> bool:
        """Check if agent is in struggling state."""
        return self.status_level == "struggling" or self.reputation < 0.3
    
    def is_elite(self) -> bool:
        """Check if agent is in elite state."""
        return self.status_level == "elite" or self.reputation > 0.7


def create_agent(agent_id: str, capability_mean: float = 0.5, capability_std: float = 0.15) -> Agent:
    """Create an agent with random capability from Gaussian distribution."""
    base_cap = random.gauss(capability_mean, capability_std)
    base_cap = max(0.1, min(0.9, base_cap))  # Clamp to [0.1, 0.9]
    
    return Agent(
        id=agent_id,
        base_capability=base_cap
    )


def create_population(n_agents: int, prefix: str = "agent", 
                      capability_mean: float = 0.5, 
                      capability_std: float = 0.15) -> List[Agent]:
    """Create a population of agents."""
    return [
        create_agent(f"{prefix}_{i}", capability_mean, capability_std)
        for i in range(n_agents)
    ]
