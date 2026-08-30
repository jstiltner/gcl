"""
Experiment 35B: Large-Scale Statistical Validation

Addresses criticism: "Your sample sizes are too small"

Enhancements:
1. 100 seeds (vs 10)
2. Bootstrap confidence intervals
3. Effect sizes with uncertainty
4. Power analysis

Validates key findings with rigorous statistics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json
from scipy import stats

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, create_population


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval."""
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data)
    
    data = np.array(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return np.mean(data), lower, upper


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Approximate power for two-sample t-test."""
    # Using normal approximation
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_power = effect_size * np.sqrt(n/2) - z_alpha
    power = stats.norm.cdf(z_power)
    return power


def run_self_selection_condition(
    selection_method: str,  # "self_selection" or "random"
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition for self-selection validation."""
    random.seed(seed)
    np.random.seed(seed)
    
    agents = create_population(n_agents, prefix=f"{selection_method}_{seed}")
    
    successes = 0
    total = 0
    
    for round_num in range(n_rounds):
        difficulty = random.uniform(0.3, 0.7)
        task = Task(id=f"task_{round_num}", difficulty=difficulty, reward=1.0)
        
        if selection_method == "self_selection":
            # Agents volunteer based on capability
            volunteers = []
            for agent in agents:
                if agent.effective_capability > difficulty * 0.5:
                    score = agent.effective_capability - difficulty * 0.3
                    volunteers.append((agent, score))
            
            if volunteers:
                selected = max(volunteers, key=lambda x: x[1])[0]
                effort = 0.9  # Volunteers try harder
            else:
                selected = max(agents, key=lambda a: a.effective_capability)
                effort = 0.8
        else:
            # Random selection
            selected = random.choice(agents)
            effort = 0.8
        
        # Execute
        success_prob = selected.effective_capability * effort * (1 - difficulty * 0.6)
        success = random.random() < success_prob
        
        total += 1
        if success:
            successes += 1
            selected.success_history.append(True)
        else:
            selected.failure_history.append(True)
    
    return {
        "cooperation_rate": successes / total if total > 0 else 0,
    }


def run_experiment(n_seeds: int = 100, n_rounds: int = 100, n_agents: int = 30) -> Dict[str, Any]:
    """Run large-scale validation."""
    print("=" * 70)
    print("EXPERIMENT 35B: LARGE-SCALE STATISTICAL VALIDATION")
    print("=" * 70)
    print(f"Seeds: {n_seeds}, Rounds: {n_rounds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Run self-selection
    print("Running self-selection...", end=" ", flush=True)
    self_selection_results = []
    for seed in range(n_seeds):
        if seed % 20 == 0:
            print(f"{seed}", end=" ", flush=True)
        metrics = run_self_selection_condition("self_selection", n_rounds, n_agents, seed)
        self_selection_results.append(metrics["cooperation_rate"])
    print("done")
    
    # Run random
    print("Running random...", end=" ", flush=True)
    random_results = []
    for seed in range(n_seeds):
        if seed % 20 == 0:
            print(f"{seed}", end=" ", flush=True)
        metrics = run_self_selection_condition("random", n_rounds, n_agents, seed)
        random_results.append(metrics["cooperation_rate"])
    print("done")
    
    # Calculate statistics
    ss_mean, ss_lower, ss_upper = bootstrap_ci(self_selection_results)
    rand_mean, rand_lower, rand_upper = bootstrap_ci(random_results)
    
    results["self_selection"] = {
        "mean": ss_mean,
        "std": np.std(self_selection_results),
        "ci_lower": ss_lower,
        "ci_upper": ss_upper,
        "n": n_seeds,
        "raw": self_selection_results,
    }
    
    results["random"] = {
        "mean": rand_mean,
        "std": np.std(random_results),
        "ci_lower": rand_lower,
        "ci_upper": rand_upper,
        "n": n_seeds,
        "raw": random_results,
    }
    
    # Effect size
    effect_size = cohens_d(self_selection_results, random_results)
    results["effect_size"] = effect_size
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(self_selection_results, random_results)
    results["t_statistic"] = t_stat
    results["p_value"] = p_value
    
    # Power analysis
    power = power_analysis(effect_size, n_seeds)
    results["power"] = power
    
    return results


def print_results(results: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Large-Scale Statistical Validation")
    print("=" * 70)
    
    ss = results["self_selection"]
    rand = results["random"]
    
    print("\n--- Cooperation Rates with 95% Confidence Intervals ---")
    print(f"Self-selection: {ss['mean']:.4f} [{ss['ci_lower']:.4f}, {ss['ci_upper']:.4f}]")
    print(f"Random:         {rand['mean']:.4f} [{rand['ci_lower']:.4f}, {rand['ci_upper']:.4f}]")
    
    print("\n--- Effect Size ---")
    d = results["effect_size"]
    if abs(d) < 0.2:
        effect_label = "negligible"
    elif abs(d) < 0.5:
        effect_label = "small"
    elif abs(d) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"
    print(f"Cohen's d: {d:.4f} ({effect_label})")
    
    print("\n--- Statistical Test ---")
    print(f"t-statistic: {results['t_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.2e}")
    
    if results['p_value'] < 0.001:
        sig_label = "highly significant (p < 0.001)"
    elif results['p_value'] < 0.01:
        sig_label = "very significant (p < 0.01)"
    elif results['p_value'] < 0.05:
        sig_label = "significant (p < 0.05)"
    else:
        sig_label = "not significant (p >= 0.05)"
    print(f"Result: {sig_label}")
    
    print("\n--- Power Analysis ---")
    print(f"Statistical power: {results['power']:.4f}")
    if results['power'] > 0.8:
        print("✓ Adequate power (>0.8) to detect this effect")
    else:
        print(f"○ Power below 0.8 - would need more samples")
    
    print("\n--- KEY FINDING ---")
    diff = ss['mean'] - rand['mean']
    print(f"Self-selection advantage: {diff:+.4f}")
    print(f"95% CI for difference: [{ss['ci_lower'] - rand['ci_upper']:.4f}, {ss['ci_upper'] - rand['ci_lower']:.4f}]")
    
    if ss['ci_lower'] > rand['ci_upper']:
        print("✓ Self-selection is SIGNIFICANTLY better than random")
        print("  (confidence intervals do not overlap)")
    elif ss['ci_upper'] < rand['ci_lower']:
        print("✗ Random is SIGNIFICANTLY better than self-selection")
    else:
        print("○ Confidence intervals overlap - difference may not be robust")


def main():
    results = run_experiment(n_seeds=100, n_rounds=100, n_agents=30)
    print_results(results)
    
    # Save (without raw data to keep file small)
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
    
    # Remove raw data for saving
    save_results = {
        "self_selection": {k: v for k, v in results["self_selection"].items() if k != "raw"},
        "random": {k: v for k, v in results["random"].items() if k != "raw"},
        "effect_size": results["effect_size"],
        "t_statistic": results["t_statistic"],
        "p_value": results["p_value"],
        "power": results["power"],
    }
    
    with open("results/experiment_35b_statistical.json", "w") as f:
        json.dump(save_results, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_35b_statistical.json")


if __name__ == "__main__":
    main()
