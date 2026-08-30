"""
Experiment 34C: Sharing-Specialization Frontier

Research Question: What's the optimal sharing rate that balances cooperation and specialization?

Design: Test sharing rates from 0% to 100% in 10% increments
- Measure both cooperation AND specialization
- Find Pareto frontier

From Exp 33b:
- 0% sharing: spec=0.457, coop=0.312
- 100% sharing: spec=0.375, coop=0.420

Hypothesis: There's an optimal sharing rate (around 50%) that balances both.
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
        """HHI for task type distribution."""
        total = sum(self.attempts_by_type.values())
        if total == 0:
            return 0.0
        shares = [count / total for count in self.attempts_by_type.values()]
        return sum(s ** 2 for s in shares)


class SharingFrontierStructure:
    """Structure with configurable sharing rate."""
    
    def __init__(self, sharing_rate: float):
        self.sharing_rate = sharing_rate
        self.agent_states: Dict[str, AgentState] = {}
        self.templates_transferred = 0
    
    def get_agent_state(self, agent: Agent) -> AgentState:
        if agent.id not in self.agent_states:
            self.agent_states[agent.id] = AgentState(agent_id=agent.id)
        return self.agent_states[agent.id]
    
    def generate_task(self, round_num: int) -> Task:
        """Generate a typed task."""
        task_type = random.choice(list(TaskType))
        difficulty = random.uniform(0.3, 0.7)
        return Task(
            id=f"task_{round_num}",
            difficulty=difficulty,
            reward=1.0,
            task_type=task_type.value
        )
    
    def get_template_boost(self, agent: Agent, task_type: str) -> float:
        """Calculate template boost with type matching."""
        total_boost = 0.0
        for t in agent.template_library:
            if isinstance(t, TypedTemplate):
                boost = t.capability_boost
                if t.task_type.value == task_type:
                    boost += 0.1  # Type match bonus
                total_boost += boost
            else:
                total_boost += t.capability_boost
        return total_boost
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """Self-selection based on capability and type match."""
        available = [a for a in agents if a.reputation > 0.1]
        if not available:
            available = agents
        
        volunteers = []
        for agent in available:
            base_cap = agent.base_capability + agent.learned_capability
            template_boost = self.get_template_boost(agent, task.task_type)
            effective_cap = min(1.0, base_cap + template_boost)
            
            # Type match bonus for volunteering
            state = self.get_agent_state(agent)
            type_experience = state.attempts_by_type.get(task.task_type, 0)
            experience_bonus = min(0.1, type_experience * 0.01)
            
            score = effective_cap + experience_bonus - task.difficulty * 0.5
            if score > 0:
                volunteers.append((agent, score))
        
        if not volunteers:
            return [max(available, key=lambda a: a.effective_capability)]
        
        best = max(volunteers, key=lambda x: x[1])
        return [best[0]]
    
    def attempt_task(self, agent: Agent, task: Task) -> TaskOutcome:
        """Attempt a task."""
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
            output_quality = effective_cap
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
        """Learn a typed template."""
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
        """Share templates at configured rate."""
        if self.sharing_rate <= 0:
            return
        
        if self.sharing_rate >= 1.0:
            # Full sharing
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
                        self.templates_transferred += 1
        else:
            # Partial sharing
            for agent in agents:
                if not agent.template_library:
                    continue
                
                # Share fraction of templates
                n_to_share = max(1, int(len(agent.template_library) * self.sharing_rate))
                templates_to_share = random.sample(
                    agent.template_library, 
                    min(n_to_share, len(agent.template_library))
                )
                
                # Share with fraction of agents
                other_agents = [a for a in agents if a.id != agent.id]
                if not other_agents:
                    continue
                n_recipients = max(1, int(len(other_agents) * self.sharing_rate))
                recipients = random.sample(
                    other_agents,
                    min(n_recipients, len(other_agents))
                )
                
                for template in templates_to_share:
                    for recipient in recipients:
                        if template not in recipient.template_library:
                            recipient.template_library.append(template)
                            self.templates_transferred += 1
    
    def get_population_specialization(self) -> float:
        """Average HHI across agents."""
        if not self.agent_states:
            return 0.0
        indices = [s.get_specialization_index() for s in self.agent_states.values()]
        return sum(indices) / len(indices)


def run_condition(
    sharing_rate: float,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    structure = SharingFrontierStructure(sharing_rate)
    agents = create_population(n_agents, prefix=f"share{int(sharing_rate*100)}_{seed}")
    
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
    
    mean_capability = sum(a.effective_capability for a in agents) / len(agents)
    
    return {
        "cooperation_rate": cooperation_rate,
        "specialization": specialization,
        "mean_capability": mean_capability,
        "templates_transferred": structure.templates_transferred,
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 34C: SHARING-SPECIALIZATION FRONTIER")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test sharing rates from 0% to 100%
    sharing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for rate in sharing_rates:
        print(f"Running {int(rate*100)}% sharing...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(rate, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        results[f"share_{int(rate*100)}"] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"spec={aggregated['specialization']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the sharing-specialization frontier."""
    analysis = {}
    
    rates = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cooperation = [results[f"share_{r}"]["cooperation_rate"]["mean"] for r in rates]
    specialization = [results[f"share_{r}"]["specialization"]["mean"] for r in rates]
    
    analysis["rates"] = rates
    analysis["cooperation"] = cooperation
    analysis["specialization"] = specialization
    
    # Find optimal rate for cooperation
    best_coop_idx = cooperation.index(max(cooperation))
    analysis["best_coop_rate"] = rates[best_coop_idx]
    analysis["best_coop_value"] = cooperation[best_coop_idx]
    
    # Find optimal rate for specialization
    best_spec_idx = specialization.index(max(specialization))
    analysis["best_spec_rate"] = rates[best_spec_idx]
    analysis["best_spec_value"] = specialization[best_spec_idx]
    
    # Find Pareto optimal points
    # A point is Pareto optimal if no other point dominates it
    pareto_points = []
    for i, (c, s) in enumerate(zip(cooperation, specialization)):
        dominated = False
        for j, (c2, s2) in enumerate(zip(cooperation, specialization)):
            if i != j and c2 >= c and s2 >= s and (c2 > c or s2 > s):
                dominated = True
                break
        if not dominated:
            pareto_points.append(rates[i])
    
    analysis["pareto_optimal_rates"] = pareto_points
    
    # Calculate trade-off slope
    # How much specialization do we lose per unit of cooperation gained?
    if len(cooperation) > 1:
        coop_range = max(cooperation) - min(cooperation)
        spec_range = max(specialization) - min(specialization)
        if coop_range > 0:
            analysis["tradeoff_slope"] = spec_range / coop_range
        else:
            analysis["tradeoff_slope"] = 0
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Sharing-Specialization Frontier")
    print("=" * 70)
    
    # Results table
    print("\n--- Results by Sharing Rate ---")
    print(f"{'Rate':>8} {'Cooperation':>12} {'Specialization':>14} {'Capability':>12}")
    print("-" * 50)
    
    for rate in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        coop = results[f"share_{rate}"]["cooperation_rate"]["mean"]
        spec = results[f"share_{rate}"]["specialization"]["mean"]
        cap = results[f"share_{rate}"]["mean_capability"]["mean"]
        print(f"{rate:>7}% {coop:>12.3f} {spec:>14.3f} {cap:>12.3f}")
    
    # Analysis
    print("\n--- Analysis ---")
    print(f"Best rate for cooperation: {analysis['best_coop_rate']}% (coop={analysis['best_coop_value']:.3f})")
    print(f"Best rate for specialization: {analysis['best_spec_rate']}% (spec={analysis['best_spec_value']:.3f})")
    print(f"Pareto optimal rates: {analysis['pareto_optimal_rates']}")
    print(f"Trade-off slope: {analysis['tradeoff_slope']:.3f} (spec lost per coop gained)")
    
    # Key finding
    print("\n--- KEY FINDING ---")
    if analysis['best_coop_rate'] == 100 and analysis['best_spec_rate'] == 0:
        print("Full trade-off: 100% sharing maximizes cooperation, 0% maximizes specialization")
    elif len(analysis['pareto_optimal_rates']) == 1:
        optimal = analysis['pareto_optimal_rates'][0]
        print(f"Single optimal rate: {optimal}% balances both cooperation and specialization")
    else:
        print(f"Multiple Pareto optimal rates: {analysis['pareto_optimal_rates']}")
        print("Choose based on whether cooperation or specialization is more important")


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
    with open("results/experiment_34c_frontier.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_34c_frontier.json")


if __name__ == "__main__":
    main()
