"""
Experiment 34E: Task Scarcity and Specialization

Research Question: Does task type scarcity induce specialization?

Current experiments have equal task type distribution (33%/33%/33%).
What if some types are rare?

Design:
- Vary task type distribution: equal vs skewed vs extreme
- Test if agents specialize in rare or common tasks
- Measure specialization emergence

Hypothesis: Scarcity creates specialization pressure - agents may specialize
in rare tasks (high value) or common tasks (more practice).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, Template, create_population


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
class ScarcityConfig:
    """Configuration for task scarcity experiment."""
    distribution: str  # "equal", "skewed", "extreme"
    rare_type: TaskType = TaskType.CREATIVE
    rare_bonus: float = 0.0  # Extra reward for rare tasks
    sharing_rate: float = 0.5
    
    @property
    def name(self) -> str:
        bonus_str = f"_bonus{int(self.rare_bonus*100)}" if self.rare_bonus > 0 else ""
        return f"{self.distribution}{bonus_str}"
    
    def get_type_probabilities(self) -> Dict[TaskType, float]:
        """Get probability of each task type."""
        if self.distribution == "equal":
            return {t: 1/3 for t in TaskType}
        elif self.distribution == "skewed":
            # 50% / 35% / 15%
            probs = {t: 0.35 for t in TaskType}
            probs[self.rare_type] = 0.15
            probs[TaskType.TECHNICAL] = 0.50
            return probs
        elif self.distribution == "extreme":
            # 70% / 20% / 10%
            probs = {t: 0.20 for t in TaskType}
            probs[self.rare_type] = 0.10
            probs[TaskType.TECHNICAL] = 0.70
            return probs
        else:
            return {t: 1/3 for t in TaskType}


@dataclass
class AgentState:
    """Track agent's task type distribution."""
    agent_id: str
    attempts_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successes_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_attempt(self, task_type: TaskType, success: bool):
        self.attempts_by_type[task_type.value] += 1
        if success:
            self.successes_by_type[task_type.value] += 1
    
    def get_specialization_index(self) -> float:
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
    
    def get_rare_type_share(self, rare_type: TaskType) -> float:
        """Get share of attempts on rare type."""
        total = sum(self.attempts_by_type.values())
        if total == 0:
            return 0.0
        return self.attempts_by_type[rare_type.value] / total


class ScarcityStructure:
    """Structure with configurable task scarcity."""
    
    def __init__(self, config: ScarcityConfig):
        self.config = config
        self.agent_states: Dict[str, AgentState] = {}
        self.type_probs = config.get_type_probabilities()
        
        # Track task distribution
        self.tasks_by_type: Dict[str, int] = defaultdict(int)
    
    def get_agent_state(self, agent: Agent) -> AgentState:
        if agent.id not in self.agent_states:
            self.agent_states[agent.id] = AgentState(agent_id=agent.id)
        return self.agent_states[agent.id]
    
    def generate_task(self, round_num: int) -> Task:
        """Generate task with configured type distribution."""
        # Sample task type according to distribution
        r = random.random()
        cumulative = 0.0
        task_type = TaskType.TECHNICAL  # Default
        for t, prob in self.type_probs.items():
            cumulative += prob
            if r < cumulative:
                task_type = t
                break
        
        self.tasks_by_type[task_type.value] += 1
        
        difficulty = random.uniform(0.3, 0.7)
        
        # Rare tasks may have bonus reward
        reward = 1.0
        if task_type == self.config.rare_type and self.config.rare_bonus > 0:
            reward = 1.0 + self.config.rare_bonus
        
        return Task(
            id=f"task_{round_num}",
            difficulty=difficulty,
            reward=reward,
            task_type=task_type.value
        )
    
    def get_template_boost(self, agent: Agent, task_type: str) -> float:
        total_boost = 0.0
        for t in agent.template_library:
            if isinstance(t, TypedTemplate):
                boost = t.capability_boost
                if t.task_type.value == task_type:
                    boost += 0.1
                total_boost += boost
            else:
                total_boost += t.capability_boost
        return total_boost
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        available = [a for a in agents if a.reputation > 0.1]
        if not available:
            available = agents
        
        volunteers = []
        for agent in available:
            state = self.get_agent_state(agent)
            
            base_cap = agent.base_capability + agent.learned_capability
            template_boost = self.get_template_boost(agent, task.task_type)
            effective_cap = min(1.0, base_cap + template_boost)
            
            # Experience bonus
            type_experience = state.attempts_by_type.get(task.task_type, 0)
            experience_bonus = min(0.1, type_experience * 0.01)
            
            # Reward consideration
            reward_bonus = (task.reward - 1.0) * 0.2  # Higher reward = more attractive
            
            score = effective_cap + experience_bonus + reward_bonus - task.difficulty * 0.5
            if score > 0:
                volunteers.append((agent, score))
        
        if not volunteers:
            return [max(available, key=lambda a: a.effective_capability)]
        
        best = max(volunteers, key=lambda x: x[1])
        return [best[0]]
    
    def attempt_task(self, agent: Agent, task: Task) -> TaskOutcome:
        state = self.get_agent_state(agent)
        
        base_cap = agent.base_capability + agent.learned_capability
        template_boost = self.get_template_boost(agent, task.task_type)
        effective_cap = min(1.0, base_cap + template_boost)
        
        success_prob = effective_cap * 0.8 * (1 - task.difficulty * 0.6)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        state.record_attempt(TaskType(task.task_type), success)
        
        if success:
            agent.success_history.append(True)
            output_quality = effective_cap * task.reward
        else:
            agent.failure_history.append(True)
            output_quality = 0.0
        
        return TaskOutcome(
            success=success,
            agent=agent,
            task=task,
            output_quality=output_quality
        )
    
    def learn_template(self, agent: Agent, task: Task, success: bool) -> Optional[TypedTemplate]:
        if not success or random.random() > 0.1:
            return None
        
        template = TypedTemplate(
            id=f"typed_{agent.id}_{len(agent.template_library)}_{random.randint(0, 10000)}",
            capability_boost=random.uniform(0.05, 0.15),
            task_type=TaskType(task.task_type),
            source_agent=agent.id
        )
        
        agent.template_library.append(template)
        return template
    
    def share_templates(self, agents: List[Agent]):
        if self.config.sharing_rate <= 0:
            return
        
        if self.config.sharing_rate >= 1.0:
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
        else:
            for agent in agents:
                if not agent.template_library:
                    continue
                
                n_to_share = max(1, int(len(agent.template_library) * self.config.sharing_rate))
                templates_to_share = random.sample(
                    agent.template_library,
                    min(n_to_share, len(agent.template_library))
                )
                
                other_agents = [a for a in agents if a.id != agent.id]
                if not other_agents:
                    continue
                n_recipients = max(1, int(len(other_agents) * self.config.sharing_rate))
                recipients = random.sample(
                    other_agents,
                    min(n_recipients, len(other_agents))
                )
                
                for template in templates_to_share:
                    for recipient in recipients:
                        if template not in recipient.template_library:
                            recipient.template_library.append(template)
    
    def get_population_specialization(self) -> float:
        if not self.agent_states:
            return 0.0
        indices = [s.get_specialization_index() for s in self.agent_states.values()]
        return sum(indices) / len(indices)
    
    def get_rare_type_specialists(self) -> float:
        """Get fraction of agents who specialize in rare type."""
        if not self.agent_states:
            return 0.0
        specialists = sum(
            1 for s in self.agent_states.values()
            if s.get_dominant_type() == self.config.rare_type
        )
        return specialists / len(self.agent_states)


def run_condition(
    config: ScarcityConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    structure = ScarcityStructure(config)
    agents = create_population(n_agents, prefix=f"{config.name}_{seed}")
    
    for round_num in range(n_rounds):
        task = structure.generate_task(round_num)
        volunteers = structure.get_volunteers(task, agents)
        
        if volunteers:
            agent = volunteers[0]
            outcome = structure.attempt_task(agent, task)
            structure.learn_template(agent, task, outcome.success)
        
        structure.share_templates(agents)
    
    # Calculate metrics
    total = sum(len(a.success_history) + len(a.failure_history) for a in agents)
    successes = sum(len(a.success_history) for a in agents)
    cooperation_rate = successes / total if total > 0 else 0
    
    specialization = structure.get_population_specialization()
    rare_specialists = structure.get_rare_type_specialists()
    
    # Task distribution achieved
    total_tasks = sum(structure.tasks_by_type.values())
    rare_task_share = structure.tasks_by_type[config.rare_type.value] / total_tasks if total_tasks > 0 else 0
    
    # Agent rare type participation
    rare_participation = np.mean([
        structure.get_agent_state(a).get_rare_type_share(config.rare_type)
        for a in agents
    ])
    
    return {
        "cooperation_rate": cooperation_rate,
        "specialization": specialization,
        "rare_specialists": rare_specialists,
        "rare_task_share": rare_task_share,
        "rare_participation": rare_participation,
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 34E: TASK SCARCITY AND SPECIALIZATION")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test conditions
    conditions = [
        # Baseline: equal distribution
        ScarcityConfig(distribution="equal"),
        # Skewed: 50/35/15
        ScarcityConfig(distribution="skewed"),
        # Extreme: 70/20/10
        ScarcityConfig(distribution="extreme"),
        # Skewed with bonus for rare
        ScarcityConfig(distribution="skewed", rare_bonus=0.5),
        # Extreme with bonus for rare
        ScarcityConfig(distribution="extreme", rare_bonus=1.0),
    ]
    
    for config in conditions:
        print(f"Running {config.name}...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(config, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        results[config.name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"spec={aggregated['specialization']['mean']:.3f}, "
              f"rare_spec={aggregated['rare_specialists']['mean']:.1%}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze scarcity effects."""
    analysis = {}
    
    # Compare specialization across distributions
    equal_spec = results["equal"]["specialization"]["mean"]
    skewed_spec = results["skewed"]["specialization"]["mean"]
    extreme_spec = results["extreme"]["specialization"]["mean"]
    
    analysis["equal_specialization"] = equal_spec
    analysis["skewed_specialization"] = skewed_spec
    analysis["extreme_specialization"] = extreme_spec
    analysis["scarcity_effect"] = extreme_spec - equal_spec
    
    # Effect of bonus on rare type specialists
    skewed_rare = results["skewed"]["rare_specialists"]["mean"]
    skewed_bonus_rare = results["skewed_bonus50"]["rare_specialists"]["mean"]
    analysis["bonus_effect_skewed"] = skewed_bonus_rare - skewed_rare
    
    extreme_rare = results["extreme"]["rare_specialists"]["mean"]
    extreme_bonus_rare = results["extreme_bonus100"]["rare_specialists"]["mean"]
    analysis["bonus_effect_extreme"] = extreme_bonus_rare - extreme_rare
    
    # Does scarcity induce specialization?
    analysis["scarcity_induces_specialization"] = extreme_spec > equal_spec + 0.05
    
    # Does bonus attract specialists to rare type?
    analysis["bonus_attracts_specialists"] = (
        analysis["bonus_effect_skewed"] > 0.05 or 
        analysis["bonus_effect_extreme"] > 0.05
    )
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Task Scarcity and Specialization")
    print("=" * 70)
    
    # Results table
    print("\n--- Results by Condition ---")
    print(f"{'Condition':>20} {'Cooperation':>12} {'Specialization':>14} {'Rare Spec':>12}")
    print("-" * 62)
    
    for name in ["equal", "skewed", "extreme", "skewed_bonus50", "extreme_bonus100"]:
        coop = results[name]["cooperation_rate"]["mean"]
        spec = results[name]["specialization"]["mean"]
        rare = results[name]["rare_specialists"]["mean"]
        print(f"{name:>20} {coop:>12.3f} {spec:>14.3f} {rare:>11.1%}")
    
    # Analysis
    print("\n--- Analysis ---")
    print(f"Scarcity effect on specialization: {analysis['scarcity_effect']:+.3f}")
    print(f"  Equal: {analysis['equal_specialization']:.3f}")
    print(f"  Extreme: {analysis['extreme_specialization']:.3f}")
    
    print(f"\nBonus effect on rare specialists:")
    print(f"  Skewed: {analysis['bonus_effect_skewed']:+.3f}")
    print(f"  Extreme: {analysis['bonus_effect_extreme']:+.3f}")
    
    # Key findings
    print("\n--- KEY FINDINGS ---")
    
    if analysis["scarcity_induces_specialization"]:
        print("✓ Scarcity INDUCES specialization")
    else:
        print("✗ Scarcity does NOT induce specialization")
    
    if analysis["bonus_attracts_specialists"]:
        print("✓ Bonus ATTRACTS specialists to rare tasks")
    else:
        print("✗ Bonus does NOT attract specialists to rare tasks")


def main():
    results = run_experiment(n_rounds=100, n_seeds=10, n_agents=30)
    analysis = analyze_results(results)
    print_results(results, analysis)
    
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
    
    output = {"conditions": results, "analysis": analysis}
    with open("results/experiment_34e_scarcity.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_34e_scarcity.json")


if __name__ == "__main__":
    main()
