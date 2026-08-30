"""
Experiment 35A: Rich Agent Representations

Addresses criticism: "Your model is too simple"

Enhancements:
1. Agent memory (past interactions, task history)
2. Strategic reasoning (anticipate others' choices)
3. Learning from experience (Bayesian updating)
4. Social relationships (trust networks)

Tests if findings hold with more complex agents.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class Memory:
    """Agent's memory of past interactions."""
    task_history: List[Tuple[str, bool]] = field(default_factory=list)  # (task_type, success)
    partner_history: Dict[str, List[bool]] = field(default_factory=lambda: defaultdict(list))  # agent_id -> [outcomes]
    capability_estimates: Dict[str, float] = field(default_factory=dict)  # agent_id -> estimated capability
    
    def record_task(self, task_type: str, success: bool):
        self.task_history.append((task_type, success))
    
    def record_partner(self, partner_id: str, success: bool):
        self.partner_history[partner_id].append(success)
    
    def update_capability_estimate(self, agent_id: str, observed_capability: float, learning_rate: float = 0.1):
        """Bayesian-like update of capability estimate."""
        if agent_id not in self.capability_estimates:
            self.capability_estimates[agent_id] = 0.5  # Prior
        current = self.capability_estimates[agent_id]
        self.capability_estimates[agent_id] = current + learning_rate * (observed_capability - current)
    
    def get_success_rate_by_type(self, task_type: str) -> float:
        """Get historical success rate for a task type."""
        relevant = [s for t, s in self.task_history if t == task_type]
        if not relevant:
            return 0.5  # Prior
        return sum(relevant) / len(relevant)
    
    def get_partner_trust(self, partner_id: str) -> float:
        """Get trust level for a partner based on history."""
        history = self.partner_history[partner_id]
        if not history:
            return 0.5  # Prior
        return sum(history) / len(history)


@dataclass
class RichAgent:
    """Agent with memory, strategic reasoning, and social relationships."""
    id: str
    base_capability: float
    
    # Memory
    memory: Memory = field(default_factory=Memory)
    
    # Learning
    learned_capability: float = 0.0
    learning_rate: float = 0.1
    
    # Social
    reputation: float = 0.5
    trust_network: Dict[str, float] = field(default_factory=dict)
    
    # Strategic
    strategic_level: int = 0  # 0=reactive, 1=anticipatory, 2=recursive
    risk_tolerance: float = 0.5
    
    # State
    success_history: List[bool] = field(default_factory=list)
    failure_history: List[bool] = field(default_factory=list)
    
    @property
    def effective_capability(self) -> float:
        return min(1.0, self.base_capability + self.learned_capability)
    
    def estimate_success_probability(self, task_difficulty: float, task_type: str) -> float:
        """Estimate own success probability using memory."""
        base_prob = self.effective_capability * 0.8 * (1 - task_difficulty * 0.6)
        
        # Adjust based on historical performance on this type
        historical_rate = self.memory.get_success_rate_by_type(task_type)
        adjusted_prob = 0.7 * base_prob + 0.3 * historical_rate
        
        return max(0.0, min(1.0, adjusted_prob))
    
    def decide_to_volunteer(
        self, 
        task_difficulty: float, 
        task_type: str,
        other_agents: List['RichAgent'] = None
    ) -> Tuple[bool, float]:
        """Decide whether to volunteer, with strategic reasoning."""
        
        # Estimate own success probability
        own_prob = self.estimate_success_probability(task_difficulty, task_type)
        
        # Level 0: Reactive - just compare to threshold
        if self.strategic_level == 0:
            threshold = 0.5 - self.risk_tolerance * 0.2
            return own_prob > threshold, own_prob
        
        # Level 1: Anticipatory - consider if others might volunteer
        if self.strategic_level >= 1 and other_agents:
            # Estimate how many others might volunteer
            expected_volunteers = 0
            for other in other_agents:
                if other.id != self.id:
                    other_cap = self.memory.capability_estimates.get(other.id, 0.5)
                    if other_cap > task_difficulty * 0.5:
                        expected_volunteers += 0.5  # Assume 50% chance they volunteer
            
            # If many others likely to volunteer, be more selective
            if expected_volunteers > 2:
                threshold = 0.6  # Higher threshold
            else:
                threshold = 0.4  # Lower threshold
            
            return own_prob > threshold, own_prob
        
        # Level 2: Recursive - consider what others think about me
        if self.strategic_level >= 2 and other_agents:
            # Consider reputation effects
            if self.reputation < 0.3:
                # Need to rebuild reputation - volunteer more
                threshold = 0.3
            elif self.reputation > 0.7:
                # Can be selective
                threshold = 0.6
            else:
                threshold = 0.5
            
            return own_prob > threshold, own_prob
        
        return own_prob > 0.5, own_prob
    
    def calculate_effort(self, volunteered: bool, partner: Optional['RichAgent'] = None) -> float:
        """Calculate effort based on volunteering and partner trust."""
        base_effort = 0.8
        
        # Volunteers try harder
        if volunteered:
            base_effort += 0.1
        else:
            base_effort -= 0.1
        
        # Adjust based on partner trust
        if partner:
            trust = self.trust_network.get(partner.id, 0.5)
            if trust > 0.7:
                base_effort += 0.05  # Trust partner, invest more
            elif trust < 0.3:
                base_effort -= 0.05  # Don't trust, invest less
        
        return max(0.5, min(1.0, base_effort))
    
    def learn_from_outcome(self, task_type: str, success: bool, partner: Optional['RichAgent'] = None):
        """Update memory and learning from task outcome."""
        # Record in memory
        self.memory.record_task(task_type, success)
        
        if partner:
            self.memory.record_partner(partner.id, success)
            # Update trust
            current_trust = self.trust_network.get(partner.id, 0.5)
            if success:
                self.trust_network[partner.id] = min(1.0, current_trust + 0.1)
            else:
                self.trust_network[partner.id] = max(0.0, current_trust - 0.1)
        
        # Update learned capability
        if success:
            self.learned_capability = min(0.5, self.learned_capability + 0.01)
            self.success_history.append(True)
        else:
            self.failure_history.append(True)
    
    def observe_other(self, other: 'RichAgent', outcome: bool, task_difficulty: float):
        """Update estimates of other agent's capability."""
        # Infer capability from outcome
        if outcome:
            inferred_cap = task_difficulty + 0.2  # Succeeded, so capable
        else:
            inferred_cap = task_difficulty - 0.2  # Failed, so less capable
        inferred_cap = max(0.1, min(0.9, inferred_cap))
        
        self.memory.update_capability_estimate(other.id, inferred_cap)


@dataclass
class Task:
    id: str
    difficulty: float
    task_type: str
    reward: float = 1.0


def run_condition(
    strategic_level: int,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition with rich agents."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create agents with varying strategic levels
    agents = []
    for i in range(n_agents):
        base_cap = random.gauss(0.5, 0.15)
        base_cap = max(0.1, min(0.9, base_cap))
        
        agent = RichAgent(
            id=f"agent_{i}",
            base_capability=base_cap,
            strategic_level=strategic_level,
            risk_tolerance=random.uniform(0.3, 0.7)
        )
        agents.append(agent)
    
    task_types = ["technical", "social", "creative"]
    
    volunteer_successes = 0
    volunteer_attempts = 0
    assigned_successes = 0
    assigned_attempts = 0
    
    for round_num in range(n_rounds):
        # Generate task
        task_type = random.choice(task_types)
        difficulty = random.uniform(0.3, 0.7)
        task = Task(id=f"task_{round_num}", difficulty=difficulty, task_type=task_type)
        
        # Collect volunteers
        volunteers = []
        for agent in agents:
            should_volunteer, score = agent.decide_to_volunteer(
                task.difficulty, task.task_type, agents
            )
            if should_volunteer:
                volunteers.append((agent, score))
        
        # Select agent
        if volunteers:
            # Pick best volunteer
            best = max(volunteers, key=lambda x: x[1])
            selected_agent = best[0]
            volunteered = True
            volunteer_attempts += 1
        else:
            # Random assignment
            selected_agent = random.choice(agents)
            volunteered = False
            assigned_attempts += 1
        
        # Execute task
        effort = selected_agent.calculate_effort(volunteered)
        success_prob = selected_agent.effective_capability * effort * (1 - task.difficulty * 0.6)
        success = random.random() < success_prob
        
        # Update
        selected_agent.learn_from_outcome(task.task_type, success)
        
        if volunteered:
            if success:
                volunteer_successes += 1
        else:
            if success:
                assigned_successes += 1
        
        # Others observe
        for agent in agents:
            if agent.id != selected_agent.id:
                agent.observe_other(selected_agent, success, task.difficulty)
    
    # Calculate metrics
    total = volunteer_attempts + assigned_attempts
    total_successes = volunteer_successes + assigned_successes
    cooperation_rate = total_successes / total if total > 0 else 0
    
    volunteer_rate = volunteer_attempts / total if total > 0 else 0
    volunteer_success_rate = volunteer_successes / volunteer_attempts if volunteer_attempts > 0 else 0
    assigned_success_rate = assigned_successes / assigned_attempts if assigned_attempts > 0 else 0
    
    # Trust network density
    trust_values = []
    for agent in agents:
        trust_values.extend(agent.trust_network.values())
    mean_trust = np.mean(trust_values) if trust_values else 0.5
    
    return {
        "cooperation_rate": cooperation_rate,
        "volunteer_rate": volunteer_rate,
        "volunteer_success_rate": volunteer_success_rate,
        "assigned_success_rate": assigned_success_rate,
        "mean_trust": mean_trust,
        "mean_capability": np.mean([a.effective_capability for a in agents]),
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 35A: RICH AGENT REPRESENTATIONS")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test different strategic levels
    for level in [0, 1, 2]:
        level_name = ["reactive", "anticipatory", "recursive"][level]
        print(f"Running {level_name} (level {level})...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(level, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        results[level_name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"vol_rate={aggregated['volunteer_rate']['mean']:.3f}, "
              f"trust={aggregated['mean_trust']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results."""
    analysis = {}
    
    # Compare strategic levels
    reactive_coop = results["reactive"]["cooperation_rate"]["mean"]
    anticipatory_coop = results["anticipatory"]["cooperation_rate"]["mean"]
    recursive_coop = results["recursive"]["cooperation_rate"]["mean"]
    
    analysis["reactive_cooperation"] = reactive_coop
    analysis["anticipatory_cooperation"] = anticipatory_coop
    analysis["recursive_cooperation"] = recursive_coop
    
    analysis["anticipatory_effect"] = anticipatory_coop - reactive_coop
    analysis["recursive_effect"] = recursive_coop - reactive_coop
    
    # Does strategic reasoning help?
    analysis["strategic_helps"] = recursive_coop > reactive_coop + 0.02
    
    # Compare volunteer rates
    analysis["reactive_volunteer_rate"] = results["reactive"]["volunteer_rate"]["mean"]
    analysis["recursive_volunteer_rate"] = results["recursive"]["volunteer_rate"]["mean"]
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Rich Agent Representations")
    print("=" * 70)
    
    print("\n--- Results by Strategic Level ---")
    print(f"{'Level':>15} {'Cooperation':>12} {'Vol Rate':>10} {'Vol Success':>12} {'Trust':>8}")
    print("-" * 60)
    
    for level in ["reactive", "anticipatory", "recursive"]:
        coop = results[level]["cooperation_rate"]["mean"]
        vol_rate = results[level]["volunteer_rate"]["mean"]
        vol_success = results[level]["volunteer_success_rate"]["mean"]
        trust = results[level]["mean_trust"]["mean"]
        print(f"{level:>15} {coop:>12.3f} {vol_rate:>10.3f} {vol_success:>12.3f} {trust:>8.3f}")
    
    print("\n--- Analysis ---")
    print(f"Anticipatory effect: {analysis['anticipatory_effect']:+.4f}")
    print(f"Recursive effect: {analysis['recursive_effect']:+.4f}")
    
    print("\n--- KEY FINDING ---")
    if analysis["strategic_helps"]:
        print("✓ Strategic reasoning IMPROVES cooperation")
        print(f"  Reactive: {analysis['reactive_cooperation']:.3f}")
        print(f"  Recursive: {analysis['recursive_cooperation']:.3f}")
    else:
        print("✗ Strategic reasoning does NOT significantly improve cooperation")
        print(f"  Reactive: {analysis['reactive_cooperation']:.3f}")
        print(f"  Recursive: {analysis['recursive_cooperation']:.3f}")
    
    print("\n--- VALIDATION ---")
    # Compare to simple agents (from Exp 34A)
    simple_baseline = 0.45  # Approximate from Exp 34A
    rich_best = max(analysis['reactive_cooperation'], analysis['recursive_cooperation'])
    
    if abs(rich_best - simple_baseline) < 0.05:
        print("✓ Rich agents produce SIMILAR results to simple agents")
        print("  This validates that findings are not model-specific")
    else:
        print(f"○ Rich agents differ from simple agents by {rich_best - simple_baseline:+.3f}")


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
    with open("results/experiment_35a_rich_agents.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_35a_rich_agents.json")


if __name__ == "__main__":
    main()
