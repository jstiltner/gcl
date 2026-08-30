"""
Implementations of the six social structures.

GCL-COMPATIBLE DESIGN:
All structures use agent CHOICE (get_volunteers) instead of forced assignment.
Anti-gaming is achieved through difficulty-weighted reputation, not peer assignment.

1. Meritocracy - Individual performance determines status
2. Guild - Collective reputation, apprenticeship progression
3. ObligationNetwork - Status from giving, not accumulating
4. Ubuntu - Collective identity, shared reputation
5. RotatingLeadership - No permanent hierarchy
6. Monastic - Formalized failure and redemption
"""

import random
from typing import List, Optional

from .base import BaseStructure
from ..agents.agent import Agent, Task, TaskOutcome, Template
from ..config.structures import (
    MeritocracyConfig, GuildConfig, ObligationNetworkConfig,
    UbuntuConfig, RotatingLeadershipConfig, MonasticConfig
)


class Meritocracy(BaseStructure):
    """
    Standard Western corporate model.
    Individual performance determines status. Work hard, get ahead.
    
    GCL-COMPATIBLE: Agents CHOOSE tasks based on expected reputation gain.
    High-capability agents volunteer for harder tasks (more reputation).
    
    Dynamics:
    - Winners accumulate reputation → get more opportunities → accumulate more
    - Losers lose reputation → get fewer opportunities → fall further
    - No structural mechanism to prevent stratification
    - No obligation for successful to help struggling
    """
    
    name = "meritocracy"
    
    def __init__(self, config: MeritocracyConfig = None):
        self.config = config or MeritocracyConfig()
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        GCL-COMPATIBLE: Agents CHOOSE to volunteer based on capability assessment.
        In meritocracy, high-reputation agents are more confident and volunteer more.
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        for agent in available:
            # Meritocracy: confidence based on reputation
            # High-rep agents volunteer for harder tasks
            confidence = agent.reputation * agent.effective_capability
            threshold = task.difficulty * 0.7
            
            if confidence >= threshold:
                volunteers.append(agent)
            elif random.random() < 0.05:  # Small exploration chance
                volunteers.append(agent)
        
        # In meritocracy, best performer typically takes the task
        if volunteers:
            return [max(volunteers, key=lambda a: a.reputation)]
        
        # Fallback: best available agent volunteers
        return [max(available, key=lambda a: a.reputation)]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Individual bears full cost/reward with difficulty weighting."""
        # Use base class difficulty-weighted reputation
        super().process_outcome_with_difficulty(agent, outcome, all_agents, difficulty)
        
        if outcome.success:
            # Maybe learn a template
            agent.learn_template(outcome.task, True)
        
        # No redemption bonus in meritocracy
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        """Individual performance."""
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        """No transfer in pure meritocracy."""
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        """Status based on accumulated reputation."""
        return self.status_from_percentile(agent, all_agents)
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        """No obligations in meritocracy."""
        pass


class Guild(BaseStructure):
    """
    Medieval guild / professional association model.
    Collective reputation, apprenticeship progression, mutual protection.
    
    GCL-COMPATIBLE: Agents CHOOSE tasks based on their rank and capability.
    Masters volunteer for hard tasks, apprentices for easy tasks.
    
    Dynamics:
    - Individual failure doesn't destroy you (guild absorbs cost)
    - Encourages risk-taking (buffered downside)
    - Guaranteed progression creates loyalty
    - Knowledge transfers through mentorship
    - Guild reputation creates collective accountability
    """
    
    name = "guild"
    
    def __init__(self, config: GuildConfig = None):
        self.config = config or GuildConfig()
        self.guild_reputation = 0.5  # Shared guild reputation
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        GCL-COMPATIBLE: Agents CHOOSE based on rank and task difficulty.
        Masters volunteer for hard tasks, journeymen for medium, apprentices for easy.
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        
        # Match task difficulty to agent rank (agents self-select)
        if task.difficulty > 0.7:
            # Hard task - masters volunteer
            masters = [a for a in available if a.guild_rank == "master"]
            for m in masters:
                if m.effective_capability > task.difficulty * 0.6:
                    volunteers.append(m)
        elif task.difficulty > 0.4:
            # Medium task - journeymen and masters volunteer
            qualified = [a for a in available if a.guild_rank in ["journeyman", "master"]]
            for q in qualified:
                if q.effective_capability > task.difficulty * 0.5:
                    volunteers.append(q)
        else:
            # Easy task - anyone can volunteer (good for apprentices)
            for a in available:
                if a.effective_capability > task.difficulty * 0.4:
                    volunteers.append(a)
        
        # If no volunteers, fallback to random available
        if not volunteers:
            return [random.choice(available)]
        
        return [random.choice(volunteers)]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Guild absorbs failure cost with difficulty weighting."""
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            # Difficulty-weighted reward
            rep_gain = self.config.success_reward * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + rep_gain)
            
            # Update guild reputation
            self.guild_reputation = min(1.0, self.guild_reputation + 0.02)
            
            # Maybe learn a template
            template = agent.learn_template(outcome.task, True)
            if template:
                # Share with apprentices
                self._share_with_apprentices(agent, template, all_agents)
        else:
            # Guild absorbs half the penalty (with difficulty weighting)
            base_penalty = self.config.failure_penalty / difficulty_mult
            individual_penalty = base_penalty * (1 - self.config.failure_absorption_rate)
            agent.reputation -= individual_penalty
            agent.reputation = max(0.0, agent.reputation)
            
            # Guild reputation takes a small hit
            self.guild_reputation = max(0.0, self.guild_reputation - 0.01)
            
            # Check for guild-supported recovery
            if agent.is_struggling():
                agent.reputation += self.config.redemption_bonus
                agent.recovered_recently = True
                agent.was_helped_recently = True
        
        # Update guild rank based on tenure
        self._update_rank(agent)
    
    def _update_rank(self, agent: Agent):
        """Update guild rank based on tenure."""
        if agent.tenure >= self.config.progression_time * 2 and agent.guild_rank != "master":
            agent.guild_rank = "master"
        elif agent.tenure >= self.config.progression_time and agent.guild_rank == "apprentice":
            agent.guild_rank = "journeyman"
    
    def _share_with_apprentices(self, agent: Agent, template: Template, all_agents: List[Agent]):
        """Masters share templates with apprentices."""
        if agent.guild_rank == "master":
            apprentices = [a for a in all_agents if a.guild_rank == "apprentice"]
            for apprentice in apprentices[:self.config.mentorship_requirement]:
                self.transfer_knowledge(agent, apprentice)
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        """Collective output weighted by guild reputation."""
        base_output = sum(o.output_quality for o in outcomes if o.success)
        return base_output * (0.8 + 0.4 * self.guild_reputation)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        """Mandatory within-guild sharing."""
        if from_agent.template_library:
            template = random.choice(from_agent.template_library)
            if to_agent.receive_template(template):
                from_agent.helped_recently = True
                to_agent.was_helped_recently = True
                return True
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        """Status based on tenure + skill (guild rank)."""
        if agent.guild_rank == "master":
            return "elite"
        elif agent.guild_rank == "journeyman":
            return "middle"
        else:
            return "struggling" if agent.reputation < 0.3 else "middle"
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        """Masters must mentor apprentices."""
        if agent.guild_rank == "master":
            apprentices = [a for a in all_agents if a.guild_rank == "apprentice" and a.is_struggling()]
            for apprentice in apprentices[:self.config.mentorship_requirement]:
                self.transfer_knowledge(agent, apprentice)


class ObligationNetwork(BaseStructure):
    """
    Potlatch / Kula Ring / Guanxi model.
    Status comes from giving, not accumulating. You rise by helping others.
    
    GCL-COMPATIBLE: Agents CHOOSE tasks based on obligation status.
    High-status agents (many obligations owed to them) volunteer for harder tasks.
    
    Dynamics:
    - Can't rise by hoarding—must give to gain status
    - High-status agents MUST help (or lose status)
    - Creates circulation, not accumulation
    - Struggling agents are helped (by obligation, not charity)
    - Dense reciprocity network emerges
    """
    
    name = "obligation"
    
    def __init__(self, config: ObligationNetworkConfig = None):
        self.config = config or ObligationNetworkConfig()
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        GCL-COMPATIBLE: Agents CHOOSE based on obligation status.
        High-status agents (many obligations owed to them) volunteer for harder tasks.
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        for agent in available:
            # Status based on obligations owed to them
            status = agent.total_obligations_owed_to_me()
            capability = agent.effective_capability
            
            # High-status agents volunteer for harder tasks (noblesse oblige)
            if status > 0.3 and capability > task.difficulty * 0.6:
                volunteers.append(agent)
            elif capability > task.difficulty * 0.5:
                # Others volunteer if capable
                volunteers.append(agent)
        
        if not volunteers:
            return [random.choice(available)]
        
        # Prefer highest-status volunteer
        return [max(volunteers, key=lambda a: a.total_obligations_owed_to_me())]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Creditors help on failure; success creates opportunity to help."""
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            # Difficulty-weighted reward
            rep_gain = self.config.success_reward * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + rep_gain)
            # Maybe learn a template
            agent.learn_template(outcome.task, True)
        else:
            # Low individual penalty (difficulty-weighted)
            rep_loss = self.config.failure_penalty / difficulty_mult
            agent.reputation -= rep_loss
            agent.reputation = max(0.0, agent.reputation)
            
            # Creditors (those who owe agent) must help
            creditors = [a for a in all_agents
                        if agent.id in a.obligations_owed_to and a.obligations_owed_to[agent.id] > 0]
            
            for creditor in creditors:
                # Creditor helps, reducing their debt
                agent.reputation += 0.1
                creditor.obligations_owed_to[agent.id] -= 0.1
                if creditor.obligations_owed_to[agent.id] <= 0:
                    del creditor.obligations_owed_to[agent.id]
                    if agent.id in creditor.obligations_owed_by:
                        del creditor.obligations_owed_by[agent.id]
                
                creditor.helped_recently = True
                agent.was_helped_recently = True
                agent.recovered_recently = True
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        """Output weighted by network density."""
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        """Sharing creates obligation - recipient owes giver."""
        if from_agent.template_library:
            template = random.choice(from_agent.template_library)
            if to_agent.receive_template(template):
                # Create obligation
                obligation_amount = 0.2
                if from_agent.id not in to_agent.obligations_owed_to:
                    to_agent.obligations_owed_to[from_agent.id] = 0
                to_agent.obligations_owed_to[from_agent.id] += obligation_amount
                
                if to_agent.id not in from_agent.obligations_owed_by:
                    from_agent.obligations_owed_by[to_agent.id] = 0
                from_agent.obligations_owed_by[to_agent.id] += obligation_amount
                
                from_agent.helped_recently = True
                to_agent.was_helped_recently = True
                return True
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        """Status based on obligations owed TO you (you've helped others)."""
        total_owed = agent.total_obligations_owed_to_me()
        
        all_totals = [a.total_obligations_owed_to_me() for a in all_agents]
        if not all_totals:
            return "middle"
        
        percentile = sum(t < total_owed for t in all_totals) / len(all_totals)
        
        if percentile > 0.8:
            return "elite"
        elif percentile > 0.3:
            return "middle"
        else:
            return "struggling"
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        """High-status agents must help or lose status."""
        if agent.is_elite():
            # Must help struggling agents
            struggling = [a for a in all_agents if a.is_struggling() and a.id != agent.id]
            if struggling and not agent.helped_recently:
                # Penalize hoarding
                agent.reputation -= self.config.hoarding_penalty
            elif agent.helped_recently:
                # Reward helping
                agent.reputation += self.config.helping_reward
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """Encourage knowledge sharing to build obligations."""
        # Elite agents share with struggling
        elite = [a for a in agents if a.is_elite() and a.template_library]
        struggling = [a for a in agents if a.is_struggling()]
        
        for e in elite:
            for s in struggling[:2]:  # Help up to 2 struggling agents
                self.transfer_knowledge(e, s)


class Ubuntu(BaseStructure):
    """
    Southern African communal philosophy model.
    "I am because we are." Individual identity is collective identity.
    
    GCL-COMPATIBLE: Agents CHOOSE based on capability (all share reputation).
    Most capable agents volunteer for harder tasks to benefit the collective.
    
    Dynamics:
    - No individual failure (collective absorbs all)
    - No individual success (collective benefits)
    - Removes competition entirely within group
    - Maximum knowledge sharing
    - Risk: free-rider problem? (test this)
    """
    
    name = "ubuntu"
    
    def __init__(self, config: UbuntuConfig = None):
        self.config = config or UbuntuConfig()
        self.collective_reputation = 0.5  # One shared reputation
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        GCL-COMPATIBLE: Agents CHOOSE based on capability.
        In Ubuntu, most capable volunteer for harder tasks (for collective benefit).
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        for agent in available:
            # Ubuntu: volunteer if capable (for collective benefit)
            if agent.effective_capability > task.difficulty * 0.5:
                volunteers.append(agent)
        
        if not volunteers:
            return [max(available, key=lambda a: a.effective_capability)]
        
        # Most capable volunteers (serves the collective)
        return [max(volunteers, key=lambda a: a.effective_capability)]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """All outcomes affect collective reputation with difficulty weighting."""
        n_agents = len(all_agents)
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            # Collective gains (difficulty-weighted)
            rep_gain = (self.config.success_reward * difficulty_mult) / n_agents
            self.collective_reputation += rep_gain
            self.collective_reputation = min(1.0, self.collective_reputation)
            
            # Template goes to collective (shared with all)
            template = agent.learn_template(outcome.task, True)
            if template:
                for other in all_agents:
                    if other.id != agent.id:
                        other.receive_template(template)
        else:
            # Collective absorbs failure (difficulty-weighted - less penalty for hard tasks)
            penalty = (self.config.failure_penalty / difficulty_mult) / n_agents
            self.collective_reputation -= penalty
            self.collective_reputation = max(0.0, self.collective_reputation)
            
            # Community restoration
            agent.reputation += self.config.redemption_bonus
            agent.recovered_recently = True
        
        # Sync individual reputation to collective
        for a in all_agents:
            a.reputation = self.collective_reputation
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        """Collective output."""
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        """All knowledge is shared freely."""
        if from_agent.template_library:
            template = random.choice(from_agent.template_library)
            return to_agent.receive_template(template)
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        """Status based on contribution to collective."""
        # In Ubuntu, status is more about contribution than hierarchy
        recent_successes = len(agent.success_history[-10:]) if agent.success_history else 0
        
        if recent_successes >= 7:
            return "elite"  # High contributor
        elif recent_successes >= 3:
            return "middle"
        else:
            return "struggling"  # Needs community support
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        """High contributors serve the collective."""
        if agent.is_elite():
            # Share knowledge with struggling
            struggling = [a for a in all_agents if a.is_struggling() and a.id != agent.id]
            for s in struggling:
                self.transfer_knowledge(agent, s)
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """Full knowledge sharing every round - optimized."""
        # Collect all unique template IDs
        all_template_ids = set()
        template_map = {}
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


class RotatingLeadership(BaseStructure):
    """
    Athenian democracy / some indigenous council models.
    No permanent hierarchy. Status is assigned, not earned.
    
    GCL-COMPATIBLE: Agents CHOOSE based on role and capability.
    Leaders volunteer for harder tasks during their term.
    
    Dynamics:
    - No status anxiety (your turn will come)
    - No gaming (nothing to game)
    - No permanent underclass
    - Forces knowledge transfer at rotation
    - Risk: no excellence incentive? (test this)
    """
    
    name = "rotating"
    
    def __init__(self, config: RotatingLeadershipConfig = None):
        self.config = config or RotatingLeadershipConfig()
        self.current_leader_idx = 0
        self.rounds_since_rotation = 0
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        GCL-COMPATIBLE: Agents CHOOSE based on role.
        Leaders volunteer for harder tasks during their term.
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        
        # Leaders volunteer for hard tasks (service during term)
        leaders = [a for a in available if a.current_role == "leader"]
        if task.difficulty > 0.5 and leaders:
            for leader in leaders:
                if leader.effective_capability > task.difficulty * 0.5:
                    volunteers.append(leader)
        
        # Workers volunteer for easier tasks
        if task.difficulty <= 0.5:
            workers = [a for a in available if a.current_role == "worker"]
            for worker in workers:
                if worker.effective_capability > task.difficulty * 0.4:
                    volunteers.append(worker)
        
        if not volunteers:
            return [random.choice(available)]
        
        return [random.choice(volunteers)]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Minimal personal consequence with difficulty weighting."""
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            # Slight competence tracking (difficulty-weighted)
            rep_gain = self.config.success_reward * 0.5 * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + rep_gain)
            agent.learn_template(outcome.task, True)
        else:
            # Minimal penalty (difficulty-weighted)
            rep_loss = self.config.failure_penalty / difficulty_mult
            agent.reputation = max(0.0, agent.reputation - rep_loss)
        
        # Check for rotation
        self.rounds_since_rotation += 1
        if self.rounds_since_rotation >= self.config.rotation_period:
            self._rotate_leadership(all_agents)
    
    def _rotate_leadership(self, agents: List[Agent]):
        """Rotate leadership to next agent."""
        # Current leader becomes worker
        for agent in agents:
            if agent.current_role == "leader":
                agent.current_role = "worker"
                agent.rounds_in_role = 0
        
        # Select new leader (round-robin or random)
        self.current_leader_idx = (self.current_leader_idx + 1) % len(agents)
        new_leader = agents[self.current_leader_idx]
        new_leader.current_role = "leader"
        new_leader.rounds_in_role = 0
        
        # Mandatory knowledge handoff
        old_leader_idx = (self.current_leader_idx - 1) % len(agents)
        old_leader = agents[old_leader_idx]
        self.transfer_knowledge(old_leader, new_leader)
        
        self.rounds_since_rotation = 0
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        """Standard aggregation."""
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        """Mandatory handoff at rotation."""
        transferred = False
        for template in from_agent.template_library:
            if to_agent.receive_template(template):
                transferred = True
        return transferred
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        """Status based on current role, not performance."""
        if agent.current_role == "leader":
            return "elite"
        else:
            return "middle"  # No struggling in rotating - everyone gets a turn
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        """Leaders serve during their term."""
        if agent.current_role == "leader":
            # Leader should help struggling workers
            struggling = [a for a in all_agents if a.reputation < 0.3 and a.id != agent.id]
            for s in struggling[:2]:
                self.transfer_knowledge(agent, s)
    
    def run_round_start(self, agents: List[Agent]):
        """Initialize leadership if needed."""
        super().run_round_start(agents)
        
        # Ensure someone is leader
        leaders = [a for a in agents if a.current_role == "leader"]
        if not leaders and agents:
            agents[0].current_role = "leader"


class Monastic(BaseStructure):
    """
    Benedictine / Buddhist Sangha model.
    Formalized failure and redemption. Humility as virtue.
    
    GCL-COMPATIBLE: Agents CHOOSE based on service and redemption state.
    Those in redemption volunteer for easier tasks to rebuild.
    
    Dynamics:
    - Failure is normalized (everyone fails, there's a ritual)
    - Redemption is formalized (clear path back)
    - Accumulation forbidden (prevents stratification)
    - Teaching is virtue (knowledge flows freely)
    - Risk: low performance ceiling? (test this)
    """
    
    name = "monastic"
    
    def __init__(self, config: MonasticConfig = None):
        self.config = config or MonasticConfig()
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        GCL-COMPATIBLE: Agents CHOOSE based on service and redemption state.
        Those in redemption volunteer for easier tasks to rebuild.
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        
        # Those in redemption volunteer for easier tasks
        in_redemption = [a for a in available if a.in_redemption]
        not_in_redemption = [a for a in available if not a.in_redemption]
        
        if task.difficulty < 0.4:
            # Easy task - those in redemption volunteer (path to recovery)
            for agent in in_redemption:
                if agent.effective_capability > task.difficulty * 0.4:
                    volunteers.append(agent)
        
        if task.difficulty >= 0.4 or not volunteers:
            # Harder tasks - those with service record volunteer
            for agent in not_in_redemption:
                if agent.effective_capability > task.difficulty * 0.5:
                    volunteers.append(agent)
        
        if not volunteers:
            return [random.choice(available)]
        
        # Prefer those with highest service (helped others most)
        return [max(volunteers, key=lambda a: len([h for h in a.success_history if h]))]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Failure is expected and ritualized with difficulty weighting."""
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            # Modest reward (humility) with difficulty weighting
            rep_gain = self.config.success_reward * 0.5 * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + rep_gain)
            
            # Teaching is practice - share immediately
            template = agent.learn_template(outcome.task, True)
            if template:
                # Share with all as teaching
                for other in all_agents:
                    if other.id != agent.id:
                        self.transfer_knowledge(agent, other)
            
            # Progress redemption if in it
            if agent.in_redemption:
                agent.redemption_progress += 1
                if agent.redemption_progress >= 3:  # 3 successes to complete
                    agent.in_redemption = False
                    agent.redemption_progress = 0
                    agent.reputation += self.config.redemption_bonus
                    agent.recovered_recently = True
        else:
            # Very low penalty - failure is expected (difficulty-weighted)
            rep_loss = self.config.failure_penalty / difficulty_mult
            agent.reputation = max(0.0, agent.reputation - rep_loss)
            
            # Enter redemption ritual if not already
            if not agent.in_redemption:
                agent.in_redemption = True
                agent.redemption_progress = 0
        
        # Prevent accumulation
        if self.config.accumulation_forbidden:
            agent.reputation = min(0.7, agent.reputation)  # Cap reputation
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        """Collective output, not individual glory."""
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        """Teaching IS the spiritual practice."""
        if from_agent.template_library:
            template = random.choice(from_agent.template_library)
            if to_agent.receive_template(template):
                # Teaching increases spiritual standing
                from_agent.reputation += 0.05
                from_agent.helped_recently = True
                to_agent.was_helped_recently = True
                return True
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        """Status based on service and humility, not achievement."""
        # In redemption = struggling (but supported)
        if agent.in_redemption:
            return "struggling"
        
        # Count teaching/helping acts
        if agent.helped_recently:
            return "elite"  # Service = status
        
        return "middle"
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        """Greater service for higher status."""
        if agent.is_elite():
            # Must teach/help more
            struggling = [a for a in all_agents if a.in_redemption and a.id != agent.id]
            for s in struggling:
                self.transfer_knowledge(agent, s)
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """Teaching as regular practice."""
        # Everyone teaches everyone
        for agent in agents:
            if agent.template_library and not agent.in_redemption:
                others = [a for a in agents if a.id != agent.id]
                for other in random.sample(others, min(2, len(others))):
                    self.transfer_knowledge(agent, other)


# Factory function
def create_structure(structure_type: str):
    """Create a social structure by name."""
    structures = {
        "meritocracy": Meritocracy,
        "guild": Guild,
        "obligation": ObligationNetwork,
        "ubuntu": Ubuntu,
        "rotating": RotatingLeadership,
        "monastic": Monastic,
    }
    
    structure_class = structures.get(structure_type.lower())
    if structure_class:
        return structure_class()
    raise ValueError(f"Unknown structure type: {structure_type}")
