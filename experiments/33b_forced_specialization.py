"""
Experiment 33b: Forced Specialization Analysis

Research Question: Why doesn't specialization emerge naturally?

Findings from 33: Specialization HHI = 0.011 (essentially zero)
This means agents are NOT specializing - they're taking all task types equally.

Hypotheses for why:
1. Full template sharing eliminates specialization pressure
2. Self-selection doesn't favor specialization
3. Task types don't differ enough to reward specialization
4. Agents don't have enough information to specialize

This experiment tests:
1. No sharing vs full sharing (does sharing prevent specialization?)
2. Specialization rewards (does explicit reward induce specialization?)
3. Task type difficulty variation (do different types have different difficulties?)
4. Agent capability variation by type (do agents have type-specific capabilities?)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, Template, create_population
from experiments.social_structures.structures.base import BaseStructure


class TaskType(Enum):
    TECHNICAL = "technical"
    SOCIAL = "social"
    CREATIVE = "creative"


@dataclass
class TypedTemplate:
    """Template with task type affinity."""
    id: str
    capability_boost: float
    task_type: TaskType
    source_agent: Optional[str] = None
    
    def __eq__(self, other):
        if not isinstance(other, TypedTemplate):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class TypedTask:
    """Task with type."""
    id: str
    difficulty: float
    reward: float
    task_type: TaskType


@dataclass
class SpecializationPressureConfig:
    """Configuration for specialization pressure experiment."""
    sharing_enabled: bool = True
    specialization_bonus: float = 0.0  # Extra reward for matching type
    type_difficulty_variation: bool = False  # Different types have different difficulties
    agent_type_affinity: bool = False  # Agents have innate type preferences
    success_reward: float = 0.1
    failure_penalty: float = 0.3
    
    @property
    def name(self) -> str:
        parts = []
        parts.append("share" if self.sharing_enabled else "noshare")
        if self.specialization_bonus > 0:
            parts.append(f"bonus{int(self.specialization_bonus*100)}")
        if self.type_difficulty_variation:
            parts.append("typediff")
        if self.agent_type_affinity:
            parts.append("affinity")
        return "_".join(parts)


@dataclass
class AgentTypeState:
    """Track agent's type-specific performance."""
    agent_id: str
    innate_affinity: Optional[TaskType] = None  # If agent_type_affinity is enabled
    attempts_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successes_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    templates_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_attempt(self, task_type: TaskType, success: bool):
        self.attempts_by_type[task_type.value] += 1
        if success:
            self.successes_by_type[task_type.value] += 1
    
    def get_specialization_index(self) -> float:
        """HHI for task type distribution."""
        total = sum(self.attempts_by_type.values())
        if total == 0:
            return 0.0
        shares = [count / total for count in self.attempts_by_type.values()]
        return sum(s ** 2 for s in shares)
    
    def get_dominant_type(self) -> Optional[TaskType]:
        if not self.attempts_by_type:
            return None
        dominant = max(self.attempts_by_type.items(), key=lambda x: x[1])
        return TaskType(dominant[0])
    
    def get_success_rate(self, task_type: TaskType) -> float:
        attempts = self.attempts_by_type[task_type.value]
        if attempts == 0:
            return 0.5
        return self.successes_by_type[task_type.value] / attempts


class SpecializationPressureStructure(BaseStructure):
    """Structure with configurable specialization pressure."""
    
    def __init__(self, config: SpecializationPressureConfig):
        self.config = config
        self._name = config.name
        self.agent_states: Dict[str, AgentTypeState] = {}
        self.current_round = 0
        
        # Type-specific difficulties (if enabled)
        self.type_difficulties = {
            TaskType.TECHNICAL: 0.5,
            TaskType.SOCIAL: 0.5,
            TaskType.CREATIVE: 0.5,
        }
        if config.type_difficulty_variation:
            self.type_difficulties = {
                TaskType.TECHNICAL: 0.6,  # Harder
                TaskType.SOCIAL: 0.4,     # Easier
                TaskType.CREATIVE: 0.5,   # Medium
            }
    
    @property
    def name(self) -> str:
        return self._name
    
    def initialize_agent(self, agent: Agent):
        """Initialize agent state with optional type affinity."""
        state = AgentTypeState(agent_id=agent.id)
        
        if self.config.agent_type_affinity:
            # Assign random innate affinity
            state.innate_affinity = random.choice(list(TaskType))
        
        self.agent_states[agent.id] = state
        return state
    
    def get_agent_state(self, agent: Agent) -> AgentTypeState:
        if agent.id not in self.agent_states:
            return self.initialize_agent(agent)
        return self.agent_states[agent.id]
    
    def generate_task(self, round_num: int) -> TypedTask:
        self.current_round = round_num
        task_type = random.choice(list(TaskType))
        
        base_difficulty = self.type_difficulties[task_type]
        difficulty = base_difficulty + random.uniform(-0.1, 0.1)
        difficulty = max(0.2, min(0.8, difficulty))
        
        return TypedTask(
            id=f"task_{round_num}",
            difficulty=difficulty,
            reward=1.0,
            task_type=task_type
        )
    
    def calculate_effective_capability(self, agent: Agent, task: TypedTask) -> float:
        """Calculate capability with type-specific bonuses."""
        base = agent.base_capability + agent.learned_capability
        
        # Template boost (type-matched templates give extra)
        template_boost = 0.0
        for t in agent.template_library:
            if isinstance(t, TypedTemplate):
                boost = t.capability_boost
                if t.task_type == task.task_type:
                    boost += self.config.specialization_bonus
                template_boost += boost
            else:
                template_boost += t.capability_boost
        
        # Innate affinity bonus
        state = self.get_agent_state(agent)
        if state.innate_affinity == task.task_type:
            template_boost += 0.1
        
        return min(1.0, base + template_boost)
    
    def get_volunteers(self, task: TypedTask, agents: List[Agent]) -> List[Agent]:
        """Agents volunteer based on type match and past success."""
        available = [a for a in agents if a.reputation > 0.1]
        if not available:
            available = agents
        
        volunteers = []
        for agent in available:
            state = self.get_agent_state(agent)
            
            # Calculate volunteer score
            cap = self.calculate_effective_capability(agent, task)
            
            # Bonus for past success with this type
            past_success = state.get_success_rate(task.task_type)
            
            # Bonus for type-matched templates
            type_templates = sum(1 for t in agent.template_library 
                               if isinstance(t, TypedTemplate) and t.task_type == task.task_type)
            template_match = type_templates * 0.05
            
            score = cap + past_success * 0.2 + template_match
            
            if score > task.difficulty * 0.5:
                volunteers.append((agent, score))
        
        if not volunteers:
            return [max(available, key=lambda a: self.calculate_effective_capability(a, task))]
        
        # Pick best volunteer
        best = max(volunteers, key=lambda x: x[1])
        return [best[0]]
    
    def attempt_task(self, agent: Agent, task: TypedTask) -> TaskOutcome:
        """Attempt a typed task."""
        state = self.get_agent_state(agent)
        
        cap = self.calculate_effective_capability(agent, task)
        success_prob = cap * 0.8 * (1 - task.difficulty * 0.6)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        state.record_attempt(task.task_type, success)
        
        if success:
            agent.success_history.append(True)
            output_quality = cap * (1 + random.uniform(-0.1, 0.1))
        else:
            agent.failure_history.append(True)
            output_quality = 0.0
        
        return TaskOutcome(
            success=success,
            agent=agent,
            task=Task(id=task.id, difficulty=task.difficulty, reward=task.reward, task_type=task.task_type.value),
            output_quality=output_quality
        )
    
    def learn_template(self, agent: Agent, task: TypedTask, success: bool) -> Optional[TypedTemplate]:
        """Learn a typed template."""
        if not success or random.random() > 0.1:
            return None
        
        template = TypedTemplate(
            id=f"typed_{agent.id}_{len(agent.template_library)}_{random.randint(0, 10000)}",
            capability_boost=random.uniform(0.05, 0.15),
            task_type=task.task_type,
            source_agent=agent.id
        )
        
        agent.template_library.append(template)
        state = self.get_agent_state(agent)
        state.templates_by_type[task.task_type.value] += 1
        
        return template
    
    def update_reputation(self, agent: Agent, success: bool, difficulty: float):
        """Update reputation."""
        difficulty_mult = 1.0 + difficulty * 0.5
        
        if success:
            delta = self.config.success_reward * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + delta)
        else:
            delta = self.config.failure_penalty / difficulty_mult
            agent.reputation = max(0.0, agent.reputation - delta)
    
    def share_templates(self, agents: List[Agent]):
        """Share templates if enabled."""
        if not self.config.sharing_enabled:
            return
        
        all_templates = {}
        for agent in agents:
            for t in agent.template_library:
                if t.id not in all_templates:
                    all_templates[t.id] = t
        
        for agent in agents:
            agent_ids = {t.id for t in agent.template_library}
            for tid, template in all_templates.items():
                if tid not in agent_ids:
                    agent.template_library.append(template)
    
    # Required abstract methods
    def get_available_agents(self, agents: List[Agent]) -> List[Agent]:
        return [a for a in agents if a.reputation > 0.1]
    
    def process_outcome_with_difficulty(self, agent, outcome, all_agents, difficulty):
        self.update_reputation(agent, outcome.success, difficulty)
    
    def maybe_share_knowledge(self, agents):
        self.share_templates(agents)
    
    def aggregate_outputs(self, outcomes):
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent, to_agent):
        return False
    
    def calculate_status(self, agent, all_agents):
        return "middle"
    
    def check_obligations(self, agent, all_agents):
        pass
    
    def get_population_specialization(self) -> float:
        """Average HHI across agents."""
        if not self.agent_states:
            return 0.0
        indices = [s.get_specialization_index() for s in self.agent_states.values()]
        return sum(indices) / len(indices)


def run_condition(
    config: SpecializationPressureConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, Any]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    structure = SpecializationPressureStructure(config)
    agents = create_population(n_agents, prefix=f"{config.name}_{seed}")
    
    # Initialize agent states
    for agent in agents:
        structure.initialize_agent(agent)
    
    specialization_over_time = []
    
    for round_num in range(n_rounds):
        task = structure.generate_task(round_num)
        volunteers = structure.get_volunteers(task, agents)
        
        if volunteers:
            agent = volunteers[0]
            outcome = structure.attempt_task(agent, task)
            structure.update_reputation(agent, outcome.success, task.difficulty)
            structure.learn_template(agent, task, outcome.success)
        
        structure.share_templates(agents)
        
        if round_num % 10 == 0:
            spec = structure.get_population_specialization()
            specialization_over_time.append(spec)
    
    # Final metrics
    total = sum(len(a.success_history) + len(a.failure_history) for a in agents)
    successes = sum(len(a.success_history) for a in agents)
    cooperation_rate = successes / total if total > 0 else 0
    
    final_spec = structure.get_population_specialization()
    
    # Count agents who specialized (HHI > 0.5)
    specialized_agents = sum(1 for s in structure.agent_states.values() 
                           if s.get_specialization_index() > 0.5)
    
    return {
        "cooperation_rate": cooperation_rate,
        "final_specialization": final_spec,
        "specialized_agents": specialized_agents / n_agents,
        "specialization_over_time": specialization_over_time,
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 33b: FORCED SPECIALIZATION ANALYSIS")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test conditions
    conditions = [
        # Baseline: sharing, no bonuses
        SpecializationPressureConfig(sharing_enabled=True, specialization_bonus=0.0),
        # No sharing
        SpecializationPressureConfig(sharing_enabled=False, specialization_bonus=0.0),
        # Specialization bonus
        SpecializationPressureConfig(sharing_enabled=True, specialization_bonus=0.2),
        # No sharing + bonus
        SpecializationPressureConfig(sharing_enabled=False, specialization_bonus=0.2),
        # Type difficulty variation
        SpecializationPressureConfig(sharing_enabled=True, type_difficulty_variation=True),
        # Agent type affinity
        SpecializationPressureConfig(sharing_enabled=True, agent_type_affinity=True),
        # Full pressure: no sharing + bonus + affinity
        SpecializationPressureConfig(sharing_enabled=False, specialization_bonus=0.2, agent_type_affinity=True),
    ]
    
    for config in conditions:
        print(f"Running {config.name}...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(config, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        aggregated = {}
        for key in ["cooperation_rate", "final_specialization", "specialized_agents"]:
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        results[config.name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"spec={aggregated['final_specialization']['mean']:.3f}, "
              f"specialized={aggregated['specialized_agents']['mean']:.1%}")
    
    return results


def analyze_and_print(results: Dict[str, Any]):
    """Analyze and print results."""
    print("\n" + "=" * 70)
    print("RESULTS: Specialization Pressure Analysis")
    print("=" * 70)
    
    print("\n--- Results by Condition ---")
    print(f"{'Condition':>35} {'Cooperation':>12} {'Spec HHI':>10} {'% Specialized':>14}")
    print("-" * 75)
    
    for name, metrics in results.items():
        coop = metrics["cooperation_rate"]["mean"]
        spec = metrics["final_specialization"]["mean"]
        pct = metrics["specialized_agents"]["mean"]
        print(f"{name:>35} {coop:>12.3f} {spec:>10.3f} {pct:>13.1%}")
    
    # Key comparisons
    print("\n--- Key Comparisons ---")
    
    # Sharing effect
    if "share" in results and "noshare" in results:
        share_spec = results["share"]["final_specialization"]["mean"]
        noshare_spec = results["noshare"]["final_specialization"]["mean"]
        print(f"Sharing effect on specialization: {share_spec:.3f} → {noshare_spec:.3f} "
              f"({'+' if noshare_spec > share_spec else ''}{noshare_spec - share_spec:.3f})")
    
    # Bonus effect
    if "share" in results and "share_bonus20" in results:
        base_spec = results["share"]["final_specialization"]["mean"]
        bonus_spec = results["share_bonus20"]["final_specialization"]["mean"]
        print(f"Bonus effect on specialization: {base_spec:.3f} → {bonus_spec:.3f} "
              f"({'+' if bonus_spec > base_spec else ''}{bonus_spec - base_spec:.3f})")
    
    # Full pressure
    full_pressure_key = "noshare_bonus20_affinity"
    if full_pressure_key in results:
        full_spec = results[full_pressure_key]["final_specialization"]["mean"]
        full_coop = results[full_pressure_key]["cooperation_rate"]["mean"]
        print(f"\nFull pressure (no share + bonus + affinity):")
        print(f"  Specialization: {full_spec:.3f}")
        print(f"  Cooperation: {full_coop:.3f}")
    
    # Key finding
    print("\n--- KEY FINDING ---")
    max_spec = max(r["final_specialization"]["mean"] for r in results.values())
    if max_spec < 0.4:
        print(f"✗ Specialization does NOT emerge even with pressure (max HHI = {max_spec:.3f})")
        print("  This suggests the task structure doesn't reward specialization.")
    elif max_spec < 0.6:
        print(f"○ Weak specialization emerges with pressure (max HHI = {max_spec:.3f})")
    else:
        print(f"✓ Strong specialization emerges with pressure (max HHI = {max_spec:.3f})")


def main():
    results = run_experiment(n_rounds=100, n_seeds=10, n_agents=30)
    analyze_and_print(results)
    
    # Save
    os.makedirs("results", exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open("results/experiment_33b_specialization_pressure.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_33b_specialization_pressure.json")


if __name__ == "__main__":
    main()
