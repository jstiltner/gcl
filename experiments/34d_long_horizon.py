"""
Experiment 34D: Long-Horizon Dynamics

Research Question: Does specialization emerge over longer time horizons?

Current experiments run 100 rounds. Specialization may emerge slowly.

Design:
- Run 500-1000 rounds
- Track specialization emergence over time
- Compare early vs late dynamics

Hypothesis: Specialization may emerge after ~300 rounds as agents accumulate
type-specific experience and templates.
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
    generation: int = 0
    
    def __eq__(self, other):
        if not isinstance(other, TypedTemplate):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class AgentState:
    """Track agent's task type distribution over time."""
    agent_id: str
    attempts_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successes_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    templates_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    specialization_history: List[float] = field(default_factory=list)
    
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


class LongHorizonStructure:
    """Structure for long-horizon experiments."""
    
    def __init__(self, sharing_rate: float = 0.5):
        self.sharing_rate = sharing_rate
        self.agent_states: Dict[str, AgentState] = {}
        self.current_round = 0
        
        # Track metrics over time
        self.cooperation_history: List[float] = []
        self.specialization_history: List[float] = []
        self.capability_history: List[float] = []
    
    def get_agent_state(self, agent: Agent) -> AgentState:
        if agent.id not in self.agent_states:
            self.agent_states[agent.id] = AgentState(agent_id=agent.id)
        return self.agent_states[agent.id]
    
    def generate_task(self, round_num: int) -> Task:
        self.current_round = round_num
        task_type = random.choice(list(TaskType))
        difficulty = random.uniform(0.3, 0.7)
        return Task(
            id=f"task_{round_num}",
            difficulty=difficulty,
            reward=1.0,
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
            experience_bonus = min(0.15, type_experience * 0.005)  # Grows with experience
            
            score = effective_cap + experience_bonus - task.difficulty * 0.5
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
        if not success or random.random() > 0.1:
            return None
        
        template = TypedTemplate(
            id=f"typed_{agent.id}_{len(agent.template_library)}_{random.randint(0, 10000)}",
            capability_boost=random.uniform(0.05, 0.15),
            task_type=TaskType(task.task_type),
            source_agent=agent.id,
            generation=self.current_round
        )
        
        agent.template_library.append(template)
        state = self.get_agent_state(agent)
        state.templates_by_type[task.task_type] += 1
        
        return template
    
    def share_templates(self, agents: List[Agent]):
        if self.sharing_rate <= 0:
            return
        
        if self.sharing_rate >= 1.0:
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
                
                n_to_share = max(1, int(len(agent.template_library) * self.sharing_rate))
                templates_to_share = random.sample(
                    agent.template_library,
                    min(n_to_share, len(agent.template_library))
                )
                
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
    
    def record_metrics(self, agents: List[Agent]):
        """Record metrics for this round."""
        # Cooperation
        total = sum(len(a.success_history) + len(a.failure_history) for a in agents)
        successes = sum(len(a.success_history) for a in agents)
        coop = successes / total if total > 0 else 0
        self.cooperation_history.append(coop)
        
        # Specialization
        specs = [self.get_agent_state(a).get_specialization_index() for a in agents]
        avg_spec = sum(specs) / len(specs) if specs else 0
        self.specialization_history.append(avg_spec)
        
        # Capability
        caps = [a.effective_capability for a in agents]
        avg_cap = sum(caps) / len(caps) if caps else 0
        self.capability_history.append(avg_cap)
        
        # Update agent specialization histories
        for agent in agents:
            state = self.get_agent_state(agent)
            state.specialization_history.append(state.get_specialization_index())
    
    def get_population_specialization(self) -> float:
        if not self.agent_states:
            return 0.0
        indices = [s.get_specialization_index() for s in self.agent_states.values()]
        return sum(indices) / len(indices)


def run_condition(
    n_rounds: int,
    sharing_rate: float = 0.5,
    n_agents: int = 30,
    seed: int = 0,
    record_interval: int = 10
) -> Dict[str, Any]:
    """Run a single long-horizon condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    structure = LongHorizonStructure(sharing_rate)
    agents = create_population(n_agents, prefix=f"long_{n_rounds}_{seed}")
    
    for round_num in range(n_rounds):
        task = structure.generate_task(round_num)
        volunteers = structure.get_volunteers(task, agents)
        
        if volunteers:
            agent = volunteers[0]
            outcome = structure.attempt_task(agent, task)
            structure.learn_template(agent, task, outcome.success)
        
        structure.share_templates(agents)
        
        if round_num % record_interval == 0:
            structure.record_metrics(agents)
    
    # Final metrics
    total = sum(len(a.success_history) + len(a.failure_history) for a in agents)
    successes = sum(len(a.success_history) for a in agents)
    final_cooperation = successes / total if total > 0 else 0
    final_specialization = structure.get_population_specialization()
    final_capability = sum(a.effective_capability for a in agents) / len(agents)
    
    # Analyze emergence timing
    spec_history = structure.specialization_history
    if len(spec_history) > 10:
        early_spec = np.mean(spec_history[:10])
        late_spec = np.mean(spec_history[-10:])
        spec_growth = late_spec - early_spec
    else:
        early_spec = late_spec = spec_growth = 0
    
    return {
        "final_cooperation": final_cooperation,
        "final_specialization": final_specialization,
        "final_capability": final_capability,
        "early_specialization": early_spec,
        "late_specialization": late_spec,
        "specialization_growth": spec_growth,
        "cooperation_history": structure.cooperation_history,
        "specialization_history": structure.specialization_history,
        "capability_history": structure.capability_history,
    }


def run_experiment(n_seeds: int = 5, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full long-horizon experiment."""
    print("=" * 70)
    print("EXPERIMENT 34D: LONG-HORIZON DYNAMICS")
    print("=" * 70)
    print(f"Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test different time horizons
    horizons = [100, 250, 500, 750, 1000]
    
    for n_rounds in horizons:
        print(f"Running {n_rounds} rounds...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(n_rounds, sharing_rate=0.5, n_agents=n_agents, seed=seed)
            seed_results.append(metrics)
        
        # Aggregate scalar metrics
        aggregated = {}
        scalar_keys = ["final_cooperation", "final_specialization", "final_capability",
                       "early_specialization", "late_specialization", "specialization_growth"]
        for key in scalar_keys:
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        # Aggregate time series
        for key in ["cooperation_history", "specialization_history", "capability_history"]:
            arrays = [np.array(r[key]) for r in seed_results]
            min_len = min(len(a) for a in arrays)
            trimmed = [a[:min_len] for a in arrays]
            aggregated[key] = {
                "mean": np.mean(trimmed, axis=0).tolist(),
                "std": np.std(trimmed, axis=0).tolist(),
            }
        
        results[f"rounds_{n_rounds}"] = aggregated
        print(f"coop={aggregated['final_cooperation']['mean']:.3f}, "
              f"spec={aggregated['final_specialization']['mean']:.3f}, "
              f"growth={aggregated['specialization_growth']['mean']:+.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze long-horizon dynamics."""
    analysis = {}
    
    horizons = [100, 250, 500, 750, 1000]
    
    # Extract final values
    final_specs = [results[f"rounds_{h}"]["final_specialization"]["mean"] for h in horizons]
    final_coops = [results[f"rounds_{h}"]["final_cooperation"]["mean"] for h in horizons]
    spec_growths = [results[f"rounds_{h}"]["specialization_growth"]["mean"] for h in horizons]
    
    analysis["horizons"] = horizons
    analysis["final_specializations"] = final_specs
    analysis["final_cooperations"] = final_coops
    analysis["specialization_growths"] = spec_growths
    
    # Does specialization increase with time?
    spec_trend = np.corrcoef(horizons, final_specs)[0, 1]
    analysis["specialization_time_correlation"] = float(spec_trend) if not np.isnan(spec_trend) else 0
    
    # When does specialization stabilize?
    # Find first horizon where growth < 0.01
    stabilization_horizon = None
    for i, growth in enumerate(spec_growths):
        if abs(growth) < 0.01:
            stabilization_horizon = horizons[i]
            break
    analysis["stabilization_horizon"] = stabilization_horizon
    
    # Maximum specialization achieved
    analysis["max_specialization"] = max(final_specs)
    analysis["max_spec_horizon"] = horizons[final_specs.index(max(final_specs))]
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Long-Horizon Dynamics")
    print("=" * 70)
    
    # Results table
    print("\n--- Final Metrics by Time Horizon ---")
    print(f"{'Rounds':>8} {'Cooperation':>12} {'Specialization':>14} {'Growth':>10}")
    print("-" * 50)
    
    for h in [100, 250, 500, 750, 1000]:
        coop = results[f"rounds_{h}"]["final_cooperation"]["mean"]
        spec = results[f"rounds_{h}"]["final_specialization"]["mean"]
        growth = results[f"rounds_{h}"]["specialization_growth"]["mean"]
        print(f"{h:>8} {coop:>12.3f} {spec:>14.3f} {growth:>+10.3f}")
    
    # Analysis
    print("\n--- Analysis ---")
    print(f"Specialization-time correlation: {analysis['specialization_time_correlation']:.3f}")
    print(f"Maximum specialization: {analysis['max_specialization']:.3f} at {analysis['max_spec_horizon']} rounds")
    if analysis['stabilization_horizon']:
        print(f"Stabilization horizon: {analysis['stabilization_horizon']} rounds")
    else:
        print("Stabilization: Not reached within tested horizons")
    
    # Key finding
    print("\n--- KEY FINDING ---")
    if analysis['specialization_time_correlation'] > 0.5:
        print(f"Specialization INCREASES with time (r={analysis['specialization_time_correlation']:.3f})")
        print(f"Longer horizons allow specialization to emerge")
    elif analysis['specialization_time_correlation'] < -0.5:
        print(f"Specialization DECREASES with time (r={analysis['specialization_time_correlation']:.3f})")
        print(f"Longer horizons lead to homogenization")
    else:
        print(f"Specialization is STABLE across time horizons (r={analysis['specialization_time_correlation']:.3f})")
        print(f"Time horizon doesn't significantly affect specialization")


def main():
    results = run_experiment(n_seeds=5, n_agents=30)
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
    with open("results/experiment_34d_long_horizon.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_34d_long_horizon.json")


if __name__ == "__main__":
    main()
