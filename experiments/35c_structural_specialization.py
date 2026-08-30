"""
Experiment 35C: Structural Specialization Pressure

Addresses criticism: "Your task structure doesn't require specialization"

Enhancements:
1. Tasks that REQUIRE type-specific capability (not just bonus)
2. Penalty for attempting wrong-type tasks
3. Task chains where output feeds into next task
4. Prerequisite templates for certain tasks

Tests if specialization CAN emerge under structural pressure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class TaskType(Enum):
    TECHNICAL = "technical"
    SOCIAL = "social"
    CREATIVE = "creative"


@dataclass
class SpecializedTemplate:
    """Template that only works for specific task types."""
    id: str
    capability_boost: float
    task_type: TaskType
    is_prerequisite: bool = False  # Required for certain tasks
    
    def __eq__(self, other):
        if not isinstance(other, SpecializedTemplate):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class StructuredTask:
    """Task with structural requirements."""
    id: str
    difficulty: float
    task_type: TaskType
    requires_prerequisite: bool = False  # Needs specific template
    mismatch_penalty: float = 0.0  # Penalty for wrong-type agent
    chain_position: int = 0  # Position in task chain (0 = standalone)
    
    def get_effective_difficulty(self, agent_type_match: bool) -> float:
        """Get difficulty adjusted for type match."""
        if agent_type_match:
            return self.difficulty
        else:
            return min(0.95, self.difficulty + self.mismatch_penalty)


@dataclass
class StructuredAgent:
    """Agent with type-specific capabilities."""
    id: str
    base_capability: float
    primary_type: Optional[TaskType] = None  # Innate specialization
    
    # Templates
    templates: List[SpecializedTemplate] = field(default_factory=list)
    
    # History
    attempts_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successes_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # State
    reputation: float = 0.5
    
    def get_type_capability(self, task_type: TaskType) -> float:
        """Get capability for a specific task type."""
        base = self.base_capability
        
        # Innate type bonus
        if self.primary_type == task_type:
            base += 0.15
        
        # Template bonus (only matching templates count)
        template_boost = sum(
            t.capability_boost for t in self.templates 
            if t.task_type == task_type
        )
        
        return min(1.0, base + template_boost)
    
    def has_prerequisite(self, task_type: TaskType) -> bool:
        """Check if agent has prerequisite template for task type."""
        return any(
            t.task_type == task_type and t.is_prerequisite 
            for t in self.templates
        )
    
    def get_specialization_index(self) -> float:
        """HHI for task type distribution."""
        total = sum(self.attempts_by_type.values())
        if total == 0:
            return 0.0
        shares = [count / total for count in self.attempts_by_type.values()]
        return sum(s ** 2 for s in shares)
    
    def get_dominant_type(self) -> Optional[TaskType]:
        """Get most attempted task type."""
        if not self.attempts_by_type:
            return None
        dominant = max(self.attempts_by_type.items(), key=lambda x: x[1])
        return TaskType(dominant[0])


@dataclass
class StructuralConfig:
    """Configuration for structural specialization."""
    mismatch_penalty: float = 0.0  # 0.0 = no penalty, 0.3 = strong penalty
    require_prerequisites: bool = False
    use_task_chains: bool = False
    innate_specialization: bool = False
    sharing_rate: float = 0.5
    
    @property
    def name(self) -> str:
        parts = []
        if self.mismatch_penalty > 0:
            parts.append(f"penalty{int(self.mismatch_penalty*100)}")
        if self.require_prerequisites:
            parts.append("prereq")
        if self.use_task_chains:
            parts.append("chains")
        if self.innate_specialization:
            parts.append("innate")
        return "_".join(parts) if parts else "baseline"


class StructuralEnvironment:
    """Environment with structural specialization pressure."""
    
    def __init__(self, config: StructuralConfig):
        self.config = config
        self.task_chain: List[StructuredTask] = []
        self.chain_position = 0
    
    def create_agent(self, agent_id: str) -> StructuredAgent:
        """Create an agent with optional innate specialization."""
        base_cap = random.gauss(0.5, 0.15)
        base_cap = max(0.1, min(0.9, base_cap))
        
        primary_type = None
        if self.config.innate_specialization:
            primary_type = random.choice(list(TaskType))
        
        return StructuredAgent(
            id=agent_id,
            base_capability=base_cap,
            primary_type=primary_type
        )
    
    def generate_task(self, round_num: int) -> StructuredTask:
        """Generate a task with structural requirements."""
        task_type = random.choice(list(TaskType))
        difficulty = random.uniform(0.3, 0.7)
        
        # Task chains
        chain_position = 0
        if self.config.use_task_chains:
            if random.random() < 0.3:  # 30% chance of chain task
                chain_position = random.randint(1, 3)
        
        return StructuredTask(
            id=f"task_{round_num}",
            difficulty=difficulty,
            task_type=task_type,
            requires_prerequisite=self.config.require_prerequisites and random.random() < 0.3,
            mismatch_penalty=self.config.mismatch_penalty,
            chain_position=chain_position
        )
    
    def get_volunteers(self, task: StructuredTask, agents: List[StructuredAgent]) -> List[StructuredAgent]:
        """Get volunteers considering structural requirements."""
        volunteers = []
        
        for agent in agents:
            # Check prerequisite
            if task.requires_prerequisite and not agent.has_prerequisite(task.task_type):
                continue  # Can't attempt without prerequisite
            
            # Calculate volunteer score
            type_cap = agent.get_type_capability(task.task_type)
            effective_diff = task.get_effective_difficulty(agent.primary_type == task.task_type)
            
            score = type_cap - effective_diff * 0.5
            
            if score > 0:
                volunteers.append((agent, score))
        
        if not volunteers:
            # Fallback: pick agent with highest type capability
            available = [a for a in agents if not task.requires_prerequisite or a.has_prerequisite(task.task_type)]
            if available:
                return [max(available, key=lambda a: a.get_type_capability(task.task_type))]
            return []
        
        # Return best volunteer
        best = max(volunteers, key=lambda x: x[1])
        return [best[0]]
    
    def attempt_task(self, agent: StructuredAgent, task: StructuredTask) -> bool:
        """Attempt a task with structural effects."""
        type_cap = agent.get_type_capability(task.task_type)
        type_match = agent.primary_type == task.task_type
        effective_diff = task.get_effective_difficulty(type_match)
        
        success_prob = type_cap * 0.8 * (1 - effective_diff * 0.6)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        
        # Record
        agent.attempts_by_type[task.task_type.value] += 1
        if success:
            agent.successes_by_type[task.task_type.value] += 1
        
        return success
    
    def learn_template(self, agent: StructuredAgent, task: StructuredTask, success: bool) -> Optional[SpecializedTemplate]:
        """Learn a specialized template."""
        if not success or random.random() > 0.1:
            return None
        
        # Templates are type-specific
        is_prereq = self.config.require_prerequisites and random.random() < 0.2
        
        template = SpecializedTemplate(
            id=f"template_{agent.id}_{len(agent.templates)}_{random.randint(0, 10000)}",
            capability_boost=random.uniform(0.05, 0.15),
            task_type=task.task_type,
            is_prerequisite=is_prereq
        )
        
        agent.templates.append(template)
        return template
    
    def share_templates(self, agents: List[StructuredAgent]):
        """Share templates at configured rate."""
        if self.config.sharing_rate <= 0:
            return
        
        all_templates = {}
        for agent in agents:
            for t in agent.templates:
                if t.id not in all_templates:
                    all_templates[t.id] = t
        
        for agent in agents:
            agent_ids = {t.id for t in agent.templates}
            templates_to_add = []
            
            for tid, template in all_templates.items():
                if tid not in agent_ids:
                    if random.random() < self.config.sharing_rate:
                        templates_to_add.append(template)
            
            agent.templates.extend(templates_to_add)


def run_condition(
    config: StructuralConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    env = StructuralEnvironment(config)
    agents = [env.create_agent(f"agent_{i}") for i in range(n_agents)]
    
    successes = 0
    total = 0
    
    for round_num in range(n_rounds):
        task = env.generate_task(round_num)
        volunteers = env.get_volunteers(task, agents)
        
        if volunteers:
            agent = volunteers[0]
            success = env.attempt_task(agent, task)
            env.learn_template(agent, task, success)
            
            total += 1
            if success:
                successes += 1
        
        env.share_templates(agents)
    
    # Calculate metrics
    cooperation_rate = successes / total if total > 0 else 0
    
    # Specialization
    specs = [a.get_specialization_index() for a in agents]
    mean_specialization = np.mean(specs)
    
    # Type alignment (do agents specialize in their innate type?)
    if config.innate_specialization:
        aligned = sum(1 for a in agents if a.get_dominant_type() == a.primary_type)
        type_alignment = aligned / len(agents)
    else:
        type_alignment = 0.0
    
    return {
        "cooperation_rate": cooperation_rate,
        "specialization": mean_specialization,
        "type_alignment": type_alignment,
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 35C: STRUCTURAL SPECIALIZATION PRESSURE")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test conditions with increasing structural pressure
    conditions = [
        StructuralConfig(),  # Baseline
        StructuralConfig(mismatch_penalty=0.2),  # Mild penalty
        StructuralConfig(mismatch_penalty=0.4),  # Strong penalty
        StructuralConfig(require_prerequisites=True),  # Prerequisites
        StructuralConfig(innate_specialization=True),  # Innate types
        StructuralConfig(mismatch_penalty=0.3, innate_specialization=True),  # Combined
        StructuralConfig(mismatch_penalty=0.4, require_prerequisites=True, innate_specialization=True),  # Full pressure
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
              f"spec={aggregated['specialization']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results."""
    analysis = {}
    
    # Baseline vs full pressure
    baseline_spec = results["baseline"]["specialization"]["mean"]
    
    # Find condition with highest specialization
    max_spec = 0
    max_spec_condition = "baseline"
    for name, metrics in results.items():
        spec = metrics["specialization"]["mean"]
        if spec > max_spec:
            max_spec = spec
            max_spec_condition = name
    
    analysis["baseline_specialization"] = baseline_spec
    analysis["max_specialization"] = max_spec
    analysis["max_spec_condition"] = max_spec_condition
    analysis["specialization_increase"] = max_spec - baseline_spec
    
    # Does structural pressure induce specialization?
    analysis["pressure_induces_specialization"] = max_spec > baseline_spec + 0.1
    
    # Effect of each factor
    if "penalty20" in results:
        analysis["penalty_effect"] = results["penalty20"]["specialization"]["mean"] - baseline_spec
    if "innate" in results:
        analysis["innate_effect"] = results["innate"]["specialization"]["mean"] - baseline_spec
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Structural Specialization Pressure")
    print("=" * 70)
    
    print("\n--- Results by Condition ---")
    print(f"{'Condition':>40} {'Cooperation':>12} {'Specialization':>14}")
    print("-" * 70)
    
    for name, metrics in results.items():
        coop = metrics["cooperation_rate"]["mean"]
        spec = metrics["specialization"]["mean"]
        print(f"{name:>40} {coop:>12.3f} {spec:>14.3f}")
    
    print("\n--- Analysis ---")
    print(f"Baseline specialization: {analysis['baseline_specialization']:.3f}")
    print(f"Maximum specialization: {analysis['max_specialization']:.3f} ({analysis['max_spec_condition']})")
    print(f"Specialization increase: {analysis['specialization_increase']:+.3f}")
    
    if "penalty_effect" in analysis:
        print(f"Penalty effect: {analysis['penalty_effect']:+.3f}")
    if "innate_effect" in analysis:
        print(f"Innate type effect: {analysis['innate_effect']:+.3f}")
    
    print("\n--- KEY FINDING ---")
    if analysis["pressure_induces_specialization"]:
        print("✓ Structural pressure CAN induce specialization")
        print(f"  Baseline: {analysis['baseline_specialization']:.3f}")
        print(f"  With pressure: {analysis['max_specialization']:.3f}")
        print(f"  Increase: {analysis['specialization_increase']:+.3f}")
    else:
        print("✗ Even structural pressure doesn't induce strong specialization")
        print(f"  Maximum achieved: {analysis['max_specialization']:.3f}")


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
    with open("results/experiment_35c_structural.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_35c_structural.json")


if __name__ == "__main__":
    main()
