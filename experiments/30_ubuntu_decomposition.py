"""
Experiment 30: Ubuntu Decomposition Study

Research Question: What drives Ubuntu's dominance in Experiments 27-29?
Is it knowledge pooling, reputation pooling, or their interaction?

Design: 2x2 factorial
- Knowledge: Individual vs Pooled
- Reputation: Individual vs Pooled

Conditions:
1. Baseline (K-Indiv, R-Indiv) - Standard meritocracy
2. K-Pool (K-Pool, R-Indiv) - Full knowledge sharing, individual reputation
3. R-Pool (K-Indiv, R-Pool) - Individual knowledge, collective reputation
4. Full-Pool (K-Pool, R-Pool) - Ubuntu-style (both pooled)

Hypotheses:
H1: Knowledge pooling is the primary driver (~80% of effect)
H2: Reputation pooling reduces variance (Gini → 0)
H3: Interaction effect exists (Full > K + R)
H4: "Ubuntu philosophy" residual is small (<10%)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import json
from scipy import stats

from experiments.social_structures.agents.agent import Agent, Task, create_population
from experiments.social_structures.structures.decomposition import (
    DecompositionStructure, DecompositionConfig, create_decomposition_conditions
)
from experiments.social_structures.competition.firm import Firm


def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient."""
    if not values or len(values) <= 1:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
    return (2 * cumsum) / (n * total) - (n + 1) / n


def create_decomposition_firm(
    firm_id: str,
    config: DecompositionConfig,
    n_agents: int = 30,
    starting_resources: float = 10.0
) -> Firm:
    """Create a firm with decomposition structure."""
    structure = DecompositionStructure(config)
    agents = create_population(n_agents, prefix=f"{firm_id}_agent")
    
    firm = Firm(
        id=firm_id,
        structure=structure,
        agents=agents,
        resources=starting_resources
    )
    return firm


def collect_metrics(firm: Firm) -> Dict[str, float]:
    """Collect metrics for a firm."""
    agents = firm.agents
    n = len(agents)
    
    # Cooperation rate (success rate)
    total_tasks = sum(len(a.success_history) + len(a.failure_history) for a in agents)
    successes = sum(len(a.success_history) for a in agents)
    cooperation_rate = successes / total_tasks if total_tasks > 0 else 0
    
    # Gini coefficient
    reps = [a.reputation for a in agents]
    gini = calculate_gini(reps)
    
    # Mean capability
    mean_capability = sum(a.effective_capability for a in agents) / n if n > 0 else 0
    
    # Knowledge diffusion
    total_templates = sum(len(a.template_library) for a in agents)
    unique_templates = len(set(t.id for a in agents for t in a.template_library))
    diffusion = total_templates / (unique_templates * n) if unique_templates > 0 and n > 0 else 0
    
    # Underclass
    struggling = [a for a in agents if a.reputation < 0.3]
    underclass_size = len(struggling) / n if n > 0 else 0
    
    # Recovery rate
    failed_agents = [a for a in agents if len(a.failure_history) > 0]
    recovered = [a for a in failed_agents if a.recovered_recently]
    recovery_rate = len(recovered) / len(failed_agents) if failed_agents else 1.0
    
    # Total output
    total_output = firm.resources
    
    return {
        "cooperation_rate": cooperation_rate,
        "gini": gini,
        "mean_capability": mean_capability,
        "knowledge_diffusion": diffusion,
        "underclass_size": underclass_size,
        "recovery_rate": recovery_rate,
        "total_output": total_output,
        "total_templates": total_templates,
        "unique_templates": unique_templates,
    }


def run_condition(
    config: DecompositionConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition with given seed."""
    random.seed(seed)
    np.random.seed(seed)
    
    firm = create_decomposition_firm(
        firm_id=f"{config.name}_{seed}",
        config=config,
        n_agents=n_agents
    )
    
    # Run simulation
    for round_num in range(n_rounds):
        firm.run_internal_round()
        
        # Generate and attempt task
        difficulty = random.uniform(0.3, 0.7)
        task = Task(id=f"task_{round_num}", difficulty=difficulty, reward=1.0)
        firm.attempt_task(task)
        
        firm.end_internal_round()
    
    return collect_metrics(firm)


def run_decomposition_experiment(
    n_rounds: int = 100,
    n_seeds: int = 10,
    n_agents: int = 30
) -> Dict[str, Any]:
    """
    Run the full 2x2 factorial decomposition experiment.
    """
    print("=" * 70)
    print("EXPERIMENT 30: UBUNTU DECOMPOSITION STUDY")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    # Define conditions
    conditions = {
        "baseline": DecompositionConfig(
            knowledge_pooling=False,
            reputation_pooling=False
        ),
        "k_pool": DecompositionConfig(
            knowledge_pooling=True,
            reputation_pooling=False
        ),
        "r_pool": DecompositionConfig(
            knowledge_pooling=False,
            reputation_pooling=True
        ),
        "full_pool": DecompositionConfig(
            knowledge_pooling=True,
            reputation_pooling=True
        ),
    }
    
    results = {}
    
    for condition_name, config in conditions.items():
        print(f"Running {condition_name}...", end=" ")
        seed_results = []
        
        for seed in range(n_seeds):
            metrics = run_condition(config, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        # Aggregate across seeds
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,  # Keep raw values for statistical tests
            }
        
        results[condition_name] = aggregated
        print(f"done (coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"cap={aggregated['mean_capability']['mean']:.3f})")
    
    return results


def compute_effects(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute main effects, interaction, and decomposition.
    """
    # Extract mean cooperation rates
    baseline = results["baseline"]["cooperation_rate"]["mean"]
    k_pool = results["k_pool"]["cooperation_rate"]["mean"]
    r_pool = results["r_pool"]["cooperation_rate"]["mean"]
    full_pool = results["full_pool"]["cooperation_rate"]["mean"]
    
    # Main effects
    effect_knowledge = ((k_pool + full_pool) / 2) - ((baseline + r_pool) / 2)
    effect_reputation = ((r_pool + full_pool) / 2) - ((baseline + k_pool) / 2)
    
    # Interaction effect
    interaction = full_pool - k_pool - r_pool + baseline
    
    # Decomposition
    ubuntu_advantage = full_pool - baseline
    knowledge_contribution = k_pool - baseline
    reputation_contribution = r_pool - baseline
    
    # Residual (what's left after accounting for main effects and interaction)
    residual = ubuntu_advantage - knowledge_contribution - reputation_contribution - interaction
    
    # Percentages
    if ubuntu_advantage > 0:
        pct_knowledge = (knowledge_contribution / ubuntu_advantage) * 100
        pct_reputation = (reputation_contribution / ubuntu_advantage) * 100
        pct_interaction = (interaction / ubuntu_advantage) * 100
        pct_residual = (residual / ubuntu_advantage) * 100
    else:
        pct_knowledge = pct_reputation = pct_interaction = pct_residual = 0
    
    return {
        "main_effect_knowledge": effect_knowledge,
        "main_effect_reputation": effect_reputation,
        "interaction_effect": interaction,
        "ubuntu_advantage": ubuntu_advantage,
        "knowledge_contribution": knowledge_contribution,
        "reputation_contribution": reputation_contribution,
        "residual": residual,
        "pct_knowledge": pct_knowledge,
        "pct_reputation": pct_reputation,
        "pct_interaction": pct_interaction,
        "pct_residual": pct_residual,
    }


def compute_statistical_tests(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute statistical significance of effects.
    """
    tests = {}
    
    # Get raw values
    baseline_vals = results["baseline"]["cooperation_rate"]["values"]
    k_pool_vals = results["k_pool"]["cooperation_rate"]["values"]
    r_pool_vals = results["r_pool"]["cooperation_rate"]["values"]
    full_pool_vals = results["full_pool"]["cooperation_rate"]["values"]
    
    # T-tests for each condition vs baseline
    for name, vals in [("k_pool", k_pool_vals), ("r_pool", r_pool_vals), ("full_pool", full_pool_vals)]:
        t_stat, p_val = stats.ttest_ind(vals, baseline_vals)
        effect_size = (np.mean(vals) - np.mean(baseline_vals)) / np.std(baseline_vals + vals)
        tests[f"{name}_vs_baseline"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "effect_size_d": float(effect_size),
            "significant": p_val < 0.05,
        }
    
    # Test for interaction: Is full_pool > k_pool + r_pool - baseline?
    # This is equivalent to testing if interaction > 0
    expected_additive = [k + r - b for k, r, b in zip(k_pool_vals, r_pool_vals, baseline_vals)]
    t_stat, p_val = stats.ttest_ind(full_pool_vals, expected_additive)
    tests["interaction_test"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
        "interpretation": "superadditive" if np.mean(full_pool_vals) > np.mean(expected_additive) else "subadditive"
    }
    
    return tests


def print_results(results: Dict[str, Any], effects: Dict[str, Any], tests: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: 2x2 Factorial Decomposition")
    print("=" * 70)
    
    # Condition means
    print("\n--- Condition Means ---")
    print(f"{'Condition':<15} {'Cooperation':>12} {'Capability':>12} {'Gini':>12} {'Diffusion':>12}")
    print("-" * 65)
    
    for cond in ["baseline", "k_pool", "r_pool", "full_pool"]:
        coop = results[cond]["cooperation_rate"]["mean"]
        cap = results[cond]["mean_capability"]["mean"]
        gini = results[cond]["gini"]["mean"]
        diff = results[cond]["knowledge_diffusion"]["mean"]
        print(f"{cond:<15} {coop:>12.3f} {cap:>12.3f} {gini:>12.3f} {diff:>12.3f}")
    
    # Effects
    print("\n--- Main Effects and Interaction ---")
    print(f"Main Effect (Knowledge):  {effects['main_effect_knowledge']:+.3f}")
    print(f"Main Effect (Reputation): {effects['main_effect_reputation']:+.3f}")
    print(f"Interaction Effect:       {effects['interaction_effect']:+.3f}")
    
    # Decomposition
    print("\n--- Decomposition of Ubuntu Advantage ---")
    print(f"Total Ubuntu Advantage:   {effects['ubuntu_advantage']:+.3f}")
    print(f"  Knowledge Contribution: {effects['knowledge_contribution']:+.3f} ({effects['pct_knowledge']:.1f}%)")
    print(f"  Reputation Contribution:{effects['reputation_contribution']:+.3f} ({effects['pct_reputation']:.1f}%)")
    print(f"  Interaction:            {effects['interaction_effect']:+.3f} ({effects['pct_interaction']:.1f}%)")
    print(f"  Residual:               {effects['residual']:+.3f} ({effects['pct_residual']:.1f}%)")
    
    # Statistical tests
    print("\n--- Statistical Tests ---")
    for test_name, test_result in tests.items():
        sig = "***" if test_result.get("significant", False) else ""
        p = test_result.get("p_value", 0)
        print(f"{test_name}: p={p:.4f} {sig}")
    
    # Hypothesis evaluation
    print("\n--- Hypothesis Evaluation ---")
    
    # H1: Knowledge is primary driver (>50% of effect)
    h1_result = effects['pct_knowledge'] > 50
    print(f"H1 (Knowledge is primary, >50%): {'SUPPORTED' if h1_result else 'NOT SUPPORTED'} "
          f"({effects['pct_knowledge']:.1f}%)")
    
    # H2: Reputation pooling reduces variance
    baseline_gini = results["baseline"]["gini"]["mean"]
    r_pool_gini = results["r_pool"]["gini"]["mean"]
    h2_result = r_pool_gini < baseline_gini * 0.5
    print(f"H2 (Reputation reduces Gini by >50%): {'SUPPORTED' if h2_result else 'NOT SUPPORTED'} "
          f"(baseline={baseline_gini:.3f}, r_pool={r_pool_gini:.3f})")
    
    # H3: Interaction exists
    h3_result = tests["interaction_test"]["significant"] and tests["interaction_test"]["interpretation"] == "superadditive"
    print(f"H3 (Superadditive interaction): {'SUPPORTED' if h3_result else 'NOT SUPPORTED'} "
          f"(p={tests['interaction_test']['p_value']:.4f})")
    
    # H4: Residual is small (<10%)
    h4_result = abs(effects['pct_residual']) < 10
    print(f"H4 (Residual <10%): {'SUPPORTED' if h4_result else 'NOT SUPPORTED'} "
          f"({effects['pct_residual']:.1f}%)")
    
    # Overall interpretation
    print("\n--- INTERPRETATION ---")
    if effects['pct_knowledge'] > 60:
        print("FINDING: Knowledge pooling is the PRIMARY driver of Ubuntu's advantage.")
        print("The 'Ubuntu philosophy' effect is largely explained by information sharing.")
    elif effects['pct_knowledge'] > effects['pct_reputation']:
        print("FINDING: Knowledge pooling contributes more than reputation pooling,")
        print("but both factors are important.")
    else:
        print("FINDING: Reputation pooling contributes more than knowledge pooling.")
        print("Collective risk-sharing may be the key mechanism.")
    
    if h3_result:
        print("\nIMPORTANT: Significant interaction effect detected.")
        print("Knowledge and reputation pooling are synergistic - both are needed for full benefit.")


def main():
    """Run the full experiment."""
    # Run experiment
    results = run_decomposition_experiment(
        n_rounds=100,
        n_seeds=10,
        n_agents=30
    )
    
    # Compute effects
    effects = compute_effects(results)
    
    # Statistical tests
    tests = compute_statistical_tests(results)
    
    # Print results
    print_results(results, effects, tests)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    # Remove raw values for JSON serialization
    results_clean = {}
    for cond, metrics in results.items():
        results_clean[cond] = {}
        for metric, data in metrics.items():
            results_clean[cond][metric] = {k: v for k, v in data.items() if k != "values"}
    
    output = {
        "conditions": results_clean,
        "effects": effects,
        "statistical_tests": tests,
        "hypotheses": {
            "H1_knowledge_primary": bool(effects['pct_knowledge'] > 50),
            "H2_reputation_reduces_gini": bool(results["r_pool"]["gini"]["mean"] < results["baseline"]["gini"]["mean"] * 0.5),
            "H3_superadditive_interaction": bool(tests["interaction_test"]["significant"] and tests["interaction_test"]["interpretation"] == "superadditive"),
            "H4_residual_small": bool(abs(effects['pct_residual']) < 10),
        }
    }
    
    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open("results/experiment_30_decomposition.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_30_decomposition.json")
    
    return output


if __name__ == "__main__":
    main()
