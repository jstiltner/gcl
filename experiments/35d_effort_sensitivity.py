"""
Experiment 35D: Effort Mechanism Sensitivity

Addresses criticism: "Your effort mechanism is ad-hoc"

Tests:
1. Vary effort adjustment magnitude (0.05, 0.1, 0.15, 0.2)
2. Add effort cost (higher effort = resource drain)
3. Test robustness of the finding that effort is key

Validates that effort is truly the key mechanism.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import json

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, create_population


@dataclass
class EffortConfig:
    """Configuration for effort sensitivity experiment."""
    effort_bonus: float = 0.1  # Bonus for volunteers
    effort_cost: float = 0.0  # Cost per unit of effort
    base_effort: float = 0.8
    
    @property
    def name(self) -> str:
        parts = [f"bonus{int(self.effort_bonus*100)}"]
        if self.effort_cost > 0:
            parts.append(f"cost{int(self.effort_cost*100)}")
        return "_".join(parts)


def run_condition(
    config: EffortConfig,
    selection_method: str,  # "self_selection" or "random"
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    agents = create_population(n_agents, prefix=f"{config.name}_{selection_method}_{seed}")
    
    # Track resources for effort cost
    agent_resources = {a.id: 1.0 for a in agents}
    
    successes = 0
    total = 0
    total_effort = 0
    total_cost = 0
    
    for round_num in range(n_rounds):
        difficulty = random.uniform(0.3, 0.7)
        task = Task(id=f"task_{round_num}", difficulty=difficulty, reward=1.0)
        
        # Filter agents with resources
        available = [a for a in agents if agent_resources[a.id] > 0.1]
        if not available:
            available = agents  # Reset if all depleted
            for a in agents:
                agent_resources[a.id] = 1.0
        
        if selection_method == "self_selection":
            # Volunteers
            volunteers = []
            for agent in available:
                if agent.effective_capability > difficulty * 0.5:
                    score = agent.effective_capability - difficulty * 0.3
                    volunteers.append((agent, score))
            
            if volunteers:
                selected = max(volunteers, key=lambda x: x[1])[0]
                effort = config.base_effort + config.effort_bonus  # Volunteers try harder
            else:
                selected = max(available, key=lambda a: a.effective_capability)
                effort = config.base_effort
        else:
            # Random
            selected = random.choice(available)
            effort = config.base_effort
        
        effort = min(1.0, effort)
        total_effort += effort
        
        # Apply effort cost
        if config.effort_cost > 0:
            cost = effort * config.effort_cost
            agent_resources[selected.id] -= cost
            total_cost += cost
        
        # Execute
        success_prob = selected.effective_capability * effort * (1 - difficulty * 0.6)
        success = random.random() < success_prob
        
        total += 1
        if success:
            successes += 1
            selected.success_history.append(True)
            # Replenish resources on success
            if config.effort_cost > 0:
                agent_resources[selected.id] = min(1.0, agent_resources[selected.id] + 0.1)
        else:
            selected.failure_history.append(True)
    
    return {
        "cooperation_rate": successes / total if total > 0 else 0,
        "mean_effort": total_effort / total if total > 0 else 0,
        "total_cost": total_cost,
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 35D: EFFORT MECHANISM SENSITIVITY")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test different effort bonuses
    effort_bonuses = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    for bonus in effort_bonuses:
        config = EffortConfig(effort_bonus=bonus)
        
        for method in ["self_selection", "random"]:
            key = f"{config.name}_{method}"
            print(f"Running {key}...", end=" ", flush=True)
            
            seed_results = []
            for seed in range(n_seeds):
                metrics = run_condition(config, method, n_rounds, n_agents, seed)
                seed_results.append(metrics)
            
            aggregated = {}
            for k in seed_results[0].keys():
                values = [r[k] for r in seed_results]
                aggregated[k] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            
            results[key] = aggregated
            print(f"coop={aggregated['cooperation_rate']['mean']:.3f}")
    
    # Test effort cost
    print("\nTesting effort cost...")
    for cost in [0.0, 0.1, 0.2]:
        config = EffortConfig(effort_bonus=0.1, effort_cost=cost)
        
        for method in ["self_selection", "random"]:
            key = f"cost{int(cost*100)}_{method}"
            print(f"Running {key}...", end=" ", flush=True)
            
            seed_results = []
            for seed in range(n_seeds):
                metrics = run_condition(config, method, n_rounds, n_agents, seed)
                seed_results.append(metrics)
            
            aggregated = {}
            for k in seed_results[0].keys():
                values = [r[k] for r in seed_results]
                aggregated[k] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            
            results[key] = aggregated
            print(f"coop={aggregated['cooperation_rate']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results."""
    analysis = {}
    
    # Effect of effort bonus on self-selection advantage
    bonuses = [0, 5, 10, 15, 20]
    advantages = []
    
    for bonus in bonuses:
        ss_key = f"bonus{bonus}_self_selection"
        rand_key = f"bonus{bonus}_random"
        
        if ss_key in results and rand_key in results:
            ss_coop = results[ss_key]["cooperation_rate"]["mean"]
            rand_coop = results[rand_key]["cooperation_rate"]["mean"]
            advantage = ss_coop - rand_coop
            advantages.append(advantage)
            analysis[f"advantage_bonus{bonus}"] = advantage
    
    # Does advantage scale with effort bonus?
    if len(advantages) > 1:
        correlation = np.corrcoef(bonuses[:len(advantages)], advantages)[0, 1]
        analysis["bonus_advantage_correlation"] = float(correlation) if not np.isnan(correlation) else 0
    
    # Effect of effort cost
    for cost in [0, 10, 20]:
        ss_key = f"cost{cost}_self_selection"
        rand_key = f"cost{cost}_random"
        
        if ss_key in results and rand_key in results:
            ss_coop = results[ss_key]["cooperation_rate"]["mean"]
            rand_coop = results[rand_key]["cooperation_rate"]["mean"]
            analysis[f"advantage_cost{cost}"] = ss_coop - rand_coop
    
    # Is effort mechanism robust?
    if advantages:
        analysis["min_advantage"] = min(advantages)
        analysis["max_advantage"] = max(advantages)
        analysis["effort_robust"] = min(advantages) > 0.05
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Effort Mechanism Sensitivity")
    print("=" * 70)
    
    # Effort bonus results
    print("\n--- Effect of Effort Bonus ---")
    print(f"{'Bonus':>8} {'Self-Select':>12} {'Random':>12} {'Advantage':>12}")
    print("-" * 50)
    
    for bonus in [0, 5, 10, 15, 20]:
        ss_key = f"bonus{bonus}_self_selection"
        rand_key = f"bonus{bonus}_random"
        
        if ss_key in results and rand_key in results:
            ss = results[ss_key]["cooperation_rate"]["mean"]
            rand = results[rand_key]["cooperation_rate"]["mean"]
            adv = ss - rand
            print(f"{bonus:>7}% {ss:>12.3f} {rand:>12.3f} {adv:>+12.3f}")
    
    # Effort cost results
    print("\n--- Effect of Effort Cost ---")
    print(f"{'Cost':>8} {'Self-Select':>12} {'Random':>12} {'Advantage':>12}")
    print("-" * 50)
    
    for cost in [0, 10, 20]:
        ss_key = f"cost{cost}_self_selection"
        rand_key = f"cost{cost}_random"
        
        if ss_key in results and rand_key in results:
            ss = results[ss_key]["cooperation_rate"]["mean"]
            rand = results[rand_key]["cooperation_rate"]["mean"]
            adv = ss - rand
            print(f"{cost:>7}% {ss:>12.3f} {rand:>12.3f} {adv:>+12.3f}")
    
    # Analysis
    print("\n--- Analysis ---")
    if "bonus_advantage_correlation" in analysis:
        print(f"Bonus-advantage correlation: {analysis['bonus_advantage_correlation']:.3f}")
    
    if "min_advantage" in analysis:
        print(f"Advantage range: [{analysis['min_advantage']:.3f}, {analysis['max_advantage']:.3f}]")
    
    print("\n--- KEY FINDING ---")
    if analysis.get("effort_robust", False):
        print("✓ Effort mechanism is ROBUST")
        print(f"  Self-selection advantage persists across all effort levels")
        print(f"  Minimum advantage: {analysis['min_advantage']:.3f}")
    else:
        print("○ Effort mechanism shows sensitivity")
        if "min_advantage" in analysis:
            print(f"  Minimum advantage: {analysis['min_advantage']:.3f}")


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
    with open("results/experiment_35d_effort.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_35d_effort.json")


if __name__ == "__main__":
    main()
