"""
Experiment 24: Template Sharing Validation

Purpose: Validate that knowledge transfer (template sharing) improves coordination
before building complex social structures around this mechanism.

Templates are successful strategies/patterns that improve task performance.
This experiment tests different sharing policies to determine which best
improves population-level coordination.

Success Criteria:
1. Directed sharing > No sharing on cooperation rate (p < 0.05)
2. Directed sharing < No sharing on capability Gini (inequality reduced)
3. Bottom quartile improves faster with directed sharing
4. Effect size meaningful: >5% improvement in cooperation
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from scipy import stats
import json


@dataclass
class Template:
    """A learned strategy/pattern that improves task performance."""
    id: str
    capability_boost: float  # 0.05 - 0.2 improvement
    source_agent: str
    task_type: Optional[str] = None  # Specialization
    
    def __eq__(self, other):
        if not isinstance(other, Template):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Task:
    """A task with a difficulty level."""
    difficulty: float  # 0.0 - 1.0
    task_type: str = "general"


@dataclass
class Agent:
    """An agent with base capability and learnable templates."""
    id: str
    base_capability: float  # Fixed at birth
    templates: List[Template] = field(default_factory=list)
    
    @property
    def effective_capability(self) -> float:
        """Total capability including template boosts."""
        return min(1.0, self.base_capability + sum(t.capability_boost for t in self.templates))
    
    def attempt_task(self, task: Task) -> bool:
        """Attempt a task with probability based on capability and difficulty."""
        success_prob = self.effective_capability * (1 - task.difficulty * 0.8)
        success_prob = max(0.0, min(1.0, success_prob))  # Clamp to [0, 1]
        return random.random() < success_prob
    
    def learn_template(self, task: Task, success: bool) -> Optional[Template]:
        """Generate template from successful experience."""
        if success and random.random() < 0.1:  # 10% chance to crystallize learning
            template = Template(
                id=f"template_{self.id}_{len(self.templates)}_{random.randint(0, 10000)}",
                capability_boost=random.uniform(0.05, 0.15),
                source_agent=self.id,
                task_type=task.task_type
            )
            self.templates.append(template)
            return template
        return None


def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    if len(values) == 0:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]) if cumsum[-1] > 0 else 0.0


# ============================================================================
# Sharing Policies
# ============================================================================

def no_sharing(population: List[Agent]) -> int:
    """No template sharing - baseline condition."""
    return 0


def random_sharing(population: List[Agent], rate: float = 0.1) -> int:
    """Random pairs exchange templates."""
    transfers = 0
    for _ in range(int(len(population) * rate)):
        a, b = random.sample(population, 2)
        if a.templates:
            template = random.choice(a.templates)
            if template not in b.templates:
                b.templates.append(template)
                transfers += 1
    return transfers


def directed_sharing(population: List[Agent], rate: float = 0.1) -> int:
    """High capability agents share with low capability agents."""
    transfers = 0
    # Sort by capability
    sorted_pop = sorted(population, key=lambda a: a.effective_capability, reverse=True)
    quartile_size = max(1, len(sorted_pop) // 4)
    top_quartile = sorted_pop[:quartile_size]
    bottom_quartile = sorted_pop[-quartile_size:]
    
    # Transfer from top to bottom
    for _ in range(int(len(population) * rate)):
        teacher = random.choice(top_quartile)
        student = random.choice(bottom_quartile)
        if teacher.templates:
            template = random.choice(teacher.templates)
            if template not in student.templates:
                student.templates.append(template)
                transfers += 1
    return transfers


def mutual_sharing(population: List[Agent], rate: float = 0.1) -> int:
    """Bidirectional exchange between pairs."""
    transfers = 0
    for _ in range(int(len(population) * rate)):
        a, b = random.sample(population, 2)
        if a.templates and b.templates:
            # Exchange
            t_a = random.choice(a.templates)
            t_b = random.choice(b.templates)
            if t_a not in b.templates:
                b.templates.append(t_a)
                transfers += 1
            if t_b not in a.templates:
                a.templates.append(t_b)
                transfers += 1
        elif a.templates:
            t_a = random.choice(a.templates)
            if t_a not in b.templates:
                b.templates.append(t_a)
                transfers += 1
        elif b.templates:
            t_b = random.choice(b.templates)
            if t_b not in a.templates:
                a.templates.append(t_b)
                transfers += 1
    return transfers


# ============================================================================
# Simulation
# ============================================================================

def run_simulation(
    n_agents: int = 50,
    n_rounds: int = 200,
    sharing_policy: str = "none",
    sharing_rate: float = 0.1,
    base_capability_mean: float = 0.5,
    base_capability_std: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Agent]]:
    """
    Run a simulation with the specified sharing policy.
    
    Args:
        n_agents: Number of agents in population
        n_rounds: Number of simulation rounds
        sharing_policy: One of "none", "random", "directed", "mutual"
        sharing_rate: Rate of template sharing per round
        base_capability_mean: Mean of initial capability distribution
        base_capability_std: Std dev of initial capability distribution
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (history, final_population)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize population
    population = [
        Agent(
            id=f"agent_{i}",
            base_capability=max(0.1, min(0.9, random.gauss(base_capability_mean, base_capability_std))),
            templates=[]
        )
        for i in range(n_agents)
    ]
    
    # Select sharing function
    sharing_functions = {
        "none": lambda p: no_sharing(p),
        "random": lambda p: random_sharing(p, sharing_rate),
        "directed": lambda p: directed_sharing(p, sharing_rate),
        "mutual": lambda p: mutual_sharing(p, sharing_rate)
    }
    share_fn = sharing_functions.get(sharing_policy, no_sharing)
    
    history = []
    
    for round_num in range(n_rounds):
        round_successes = 0
        templates_created = 0
        
        # Each agent attempts a task
        for agent in population:
            task = Task(difficulty=random.uniform(0.3, 0.7))
            success = agent.attempt_task(task)
            
            if success:
                round_successes += 1
                # Maybe learn a template
                new_template = agent.learn_template(task, success)
                if new_template:
                    templates_created += 1
        
        # Apply sharing policy
        transfers = share_fn(population)
        
        # Collect metrics
        capabilities = [a.effective_capability for a in population]
        sorted_caps = sorted(capabilities)
        quartile_size = max(1, n_agents // 4)
        
        history.append({
            "round": round_num,
            "cooperation_rate": round_successes / n_agents,
            "mean_capability": np.mean(capabilities),
            "std_capability": np.std(capabilities),
            "capability_gini": calculate_gini(capabilities),
            "total_templates": sum(len(a.templates) for a in population),
            "templates_created": templates_created,
            "templates_transferred": transfers,
            "bottom_quartile_capability": np.mean(sorted_caps[:quartile_size]),
            "top_quartile_capability": np.mean(sorted_caps[-quartile_size:]),
            "min_capability": min(capabilities),
            "max_capability": max(capabilities),
        })
    
    return history, population


def analyze_results(histories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze results across conditions.
    
    Args:
        histories: Dict mapping condition name to history list
        
    Returns:
        Analysis results including statistical tests
    """
    results = {
        "conditions": {},
        "statistical_tests": {},
        "predictions": {}
    }
    
    # Extract final metrics for each condition
    for condition, history in histories.items():
        final_rounds = history[-20:]  # Last 20 rounds for stability
        
        results["conditions"][condition] = {
            "final_cooperation": np.mean([r["cooperation_rate"] for r in final_rounds]),
            "final_gini": np.mean([r["capability_gini"] for r in final_rounds]),
            "final_mean_capability": np.mean([r["mean_capability"] for r in final_rounds]),
            "bottom_quartile_final": np.mean([r["bottom_quartile_capability"] for r in final_rounds]),
            "top_quartile_final": np.mean([r["top_quartile_capability"] for r in final_rounds]),
            "total_templates": history[-1]["total_templates"],
            "cooperation_trajectory": [r["cooperation_rate"] for r in history],
            "gini_trajectory": [r["capability_gini"] for r in history],
            "bottom_quartile_trajectory": [r["bottom_quartile_capability"] for r in history],
        }
    
    # Statistical tests: directed vs none
    if "directed" in histories and "none" in histories:
        directed_coop = [r["cooperation_rate"] for r in histories["directed"][-50:]]
        none_coop = [r["cooperation_rate"] for r in histories["none"][-50:]]
        
        t_stat, p_value = stats.ttest_ind(directed_coop, none_coop)
        effect_size = (np.mean(directed_coop) - np.mean(none_coop)) / np.std(none_coop) if np.std(none_coop) > 0 else 0
        
        results["statistical_tests"]["directed_vs_none_cooperation"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "directed_mean": float(np.mean(directed_coop)),
            "none_mean": float(np.mean(none_coop)),
            "improvement": float(np.mean(directed_coop) - np.mean(none_coop))
        }
        
        # Gini comparison
        directed_gini = [r["capability_gini"] for r in histories["directed"][-50:]]
        none_gini = [r["capability_gini"] for r in histories["none"][-50:]]
        
        t_stat_gini, p_value_gini = stats.ttest_ind(directed_gini, none_gini)
        
        results["statistical_tests"]["directed_vs_none_gini"] = {
            "t_stat": float(t_stat_gini),
            "p_value": float(p_value_gini),
            "directed_mean": float(np.mean(directed_gini)),
            "none_mean": float(np.mean(none_gini)),
            "reduction": float(np.mean(none_gini) - np.mean(directed_gini))
        }
        
        # Bottom quartile comparison
        directed_bottom = [r["bottom_quartile_capability"] for r in histories["directed"][-50:]]
        none_bottom = [r["bottom_quartile_capability"] for r in histories["none"][-50:]]
        
        t_stat_bottom, p_value_bottom = stats.ttest_ind(directed_bottom, none_bottom)
        
        results["statistical_tests"]["directed_vs_none_bottom_quartile"] = {
            "t_stat": float(t_stat_bottom),
            "p_value": float(p_value_bottom),
            "directed_mean": float(np.mean(directed_bottom)),
            "none_mean": float(np.mean(none_bottom)),
            "improvement": float(np.mean(directed_bottom) - np.mean(none_bottom))
        }
    
    # Evaluate predictions
    if "directed" in results["conditions"] and "none" in results["conditions"]:
        directed = results["conditions"]["directed"]
        none = results["conditions"]["none"]
        
        # Prediction 1: Directed sharing > No sharing on cooperation rate
        pred1 = directed["final_cooperation"] > none["final_cooperation"]
        pred1_significant = results["statistical_tests"].get("directed_vs_none_cooperation", {}).get("p_value", 1.0) < 0.05
        
        # Prediction 2: Directed sharing < No sharing on Gini (inequality reduced)
        pred2 = directed["final_gini"] < none["final_gini"]
        pred2_significant = results["statistical_tests"].get("directed_vs_none_gini", {}).get("p_value", 1.0) < 0.05
        
        # Prediction 3: Bottom quartile improves more with directed sharing
        pred3 = directed["bottom_quartile_final"] > none["bottom_quartile_final"]
        pred3_significant = results["statistical_tests"].get("directed_vs_none_bottom_quartile", {}).get("p_value", 1.0) < 0.05
        
        # Prediction 4: Effect size > 5%
        improvement = directed["final_cooperation"] - none["final_cooperation"]
        pred4 = improvement > 0.05
        
        results["predictions"] = {
            "pred1_cooperation_improved": {
                "passed": pred1 and pred1_significant,
                "direction_correct": pred1,
                "statistically_significant": pred1_significant,
                "description": "Directed sharing > No sharing on cooperation rate (p < 0.05)"
            },
            "pred2_inequality_reduced": {
                "passed": pred2,  # Direction matters more than significance for Gini
                "direction_correct": pred2,
                "statistically_significant": pred2_significant,
                "description": "Directed sharing < No sharing on capability Gini"
            },
            "pred3_bottom_quartile_improved": {
                "passed": pred3 and pred3_significant,
                "direction_correct": pred3,
                "statistically_significant": pred3_significant,
                "description": "Bottom quartile improves faster with directed sharing"
            },
            "pred4_meaningful_effect": {
                "passed": pred4,
                "improvement": float(improvement),
                "threshold": 0.05,
                "description": "Effect size > 5% improvement in cooperation"
            }
        }
        
        results["predictions"]["summary"] = {
            "passed": sum(1 for p in ["pred1_cooperation_improved", "pred2_inequality_reduced", 
                                       "pred3_bottom_quartile_improved", "pred4_meaningful_effect"]
                         if results["predictions"][p]["passed"]),
            "total": 4
        }
    
    return results


def time_to_competence(history: List[Dict[str, Any]], threshold: float = 0.5) -> Optional[int]:
    """Find the round when bottom quartile reaches threshold capability."""
    for entry in history:
        if entry["bottom_quartile_capability"] >= threshold:
            return entry["round"]
    return None


def run_full_experiment(
    n_runs: int = 10,
    n_agents: int = 50,
    n_rounds: int = 200,
    sharing_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Run the full experiment with multiple runs for statistical power.
    
    Args:
        n_runs: Number of independent runs per condition
        n_agents: Number of agents per run
        n_rounds: Number of rounds per run
        sharing_rate: Template sharing rate
        
    Returns:
        Comprehensive results
    """
    conditions = ["none", "random", "directed", "mutual"]
    all_histories = {c: [] for c in conditions}
    all_final_metrics = {c: {"cooperation": [], "gini": [], "bottom_quartile": []} for c in conditions}
    
    print(f"Running Template Sharing Validation Experiment")
    print(f"  Conditions: {conditions}")
    print(f"  Runs per condition: {n_runs}")
    print(f"  Agents: {n_agents}, Rounds: {n_rounds}")
    print(f"  Sharing rate: {sharing_rate}")
    print()
    
    for condition in conditions:
        print(f"Running condition: {condition}...", end=" ")
        for run in range(n_runs):
            history, _ = run_simulation(
                n_agents=n_agents,
                n_rounds=n_rounds,
                sharing_policy=condition,
                sharing_rate=sharing_rate,
                seed=run * 1000 + hash(condition) % 1000
            )
            all_histories[condition].append(history)
            
            # Extract final metrics
            final_rounds = history[-20:]
            all_final_metrics[condition]["cooperation"].append(
                np.mean([r["cooperation_rate"] for r in final_rounds])
            )
            all_final_metrics[condition]["gini"].append(
                np.mean([r["capability_gini"] for r in final_rounds])
            )
            all_final_metrics[condition]["bottom_quartile"].append(
                np.mean([r["bottom_quartile_capability"] for r in final_rounds])
            )
        print("done")
    
    # Aggregate histories (average across runs)
    aggregated_histories = {}
    for condition in conditions:
        n_rounds_actual = len(all_histories[condition][0])
        aggregated = []
        for round_idx in range(n_rounds_actual):
            round_data = {
                "round": round_idx,
                "cooperation_rate": np.mean([h[round_idx]["cooperation_rate"] for h in all_histories[condition]]),
                "capability_gini": np.mean([h[round_idx]["capability_gini"] for h in all_histories[condition]]),
                "mean_capability": np.mean([h[round_idx]["mean_capability"] for h in all_histories[condition]]),
                "bottom_quartile_capability": np.mean([h[round_idx]["bottom_quartile_capability"] for h in all_histories[condition]]),
                "top_quartile_capability": np.mean([h[round_idx]["top_quartile_capability"] for h in all_histories[condition]]),
                "total_templates": np.mean([h[round_idx]["total_templates"] for h in all_histories[condition]]),
            }
            aggregated.append(round_data)
        aggregated_histories[condition] = aggregated
    
    # Analyze aggregated results
    results = analyze_results(aggregated_histories)
    
    # Add cross-run statistics
    results["cross_run_statistics"] = {}
    for condition in conditions:
        results["cross_run_statistics"][condition] = {
            "cooperation_mean": float(np.mean(all_final_metrics[condition]["cooperation"])),
            "cooperation_std": float(np.std(all_final_metrics[condition]["cooperation"])),
            "gini_mean": float(np.mean(all_final_metrics[condition]["gini"])),
            "gini_std": float(np.std(all_final_metrics[condition]["gini"])),
            "bottom_quartile_mean": float(np.mean(all_final_metrics[condition]["bottom_quartile"])),
            "bottom_quartile_std": float(np.std(all_final_metrics[condition]["bottom_quartile"])),
        }
    
    # Cross-run statistical tests
    if "directed" in all_final_metrics and "none" in all_final_metrics:
        # Cooperation
        t_stat, p_value = stats.ttest_ind(
            all_final_metrics["directed"]["cooperation"],
            all_final_metrics["none"]["cooperation"]
        )
        results["cross_run_statistics"]["directed_vs_none_cooperation"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
        
        # Gini
        t_stat, p_value = stats.ttest_ind(
            all_final_metrics["directed"]["gini"],
            all_final_metrics["none"]["gini"]
        )
        results["cross_run_statistics"]["directed_vs_none_gini"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
        
        # Bottom quartile
        t_stat, p_value = stats.ttest_ind(
            all_final_metrics["directed"]["bottom_quartile"],
            all_final_metrics["none"]["bottom_quartile"]
        )
        results["cross_run_statistics"]["directed_vs_none_bottom_quartile"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
    
    # Time to competence analysis
    results["time_to_competence"] = {}
    for condition in conditions:
        times = []
        for history in all_histories[condition]:
            ttc = time_to_competence(history, threshold=0.5)
            if ttc is not None:
                times.append(ttc)
        if times:
            results["time_to_competence"][condition] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": int(min(times)),
                "max": int(max(times)),
                "reached_count": len(times),
                "total_runs": n_runs
            }
        else:
            results["time_to_competence"][condition] = {
                "mean": None,
                "reached_count": 0,
                "total_runs": n_runs
            }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("TEMPLATE SHARING VALIDATION RESULTS")
    print("=" * 70)
    
    print("\n--- Final Metrics by Condition ---")
    print(f"{'Condition':<12} {'Cooperation':>12} {'Gini':>10} {'Bottom Q':>12} {'Templates':>10}")
    print("-" * 58)
    for condition, metrics in results["conditions"].items():
        print(f"{condition:<12} {metrics['final_cooperation']:>12.3f} {metrics['final_gini']:>10.3f} "
              f"{metrics['bottom_quartile_final']:>12.3f} {metrics['total_templates']:>10.0f}")
    
    if "cross_run_statistics" in results:
        print("\n--- Cross-Run Statistics ---")
        print(f"{'Condition':<12} {'Coop Mean':>10} {'Coop Std':>10} {'Gini Mean':>10} {'BQ Mean':>10}")
        print("-" * 54)
        for condition, stats_data in results["cross_run_statistics"].items():
            if isinstance(stats_data, dict) and "cooperation_mean" in stats_data:
                print(f"{condition:<12} {stats_data['cooperation_mean']:>10.3f} {stats_data['cooperation_std']:>10.3f} "
                      f"{stats_data['gini_mean']:>10.3f} {stats_data['bottom_quartile_mean']:>10.3f}")
    
    print("\n--- Statistical Tests (Directed vs None) ---")
    if "statistical_tests" in results:
        for test_name, test_results in results["statistical_tests"].items():
            print(f"\n{test_name}:")
            print(f"  t-statistic: {test_results['t_stat']:.3f}")
            print(f"  p-value: {test_results['p_value']:.4f}")
            if "improvement" in test_results:
                print(f"  improvement: {test_results['improvement']:.4f}")
            if "reduction" in test_results:
                print(f"  reduction: {test_results['reduction']:.4f}")
    
    if "cross_run_statistics" in results:
        print("\n--- Cross-Run Statistical Tests ---")
        for key in ["directed_vs_none_cooperation", "directed_vs_none_gini", "directed_vs_none_bottom_quartile"]:
            if key in results["cross_run_statistics"]:
                test = results["cross_run_statistics"][key]
                sig = "✓" if test["significant"] else "✗"
                print(f"  {key}: t={test['t_stat']:.3f}, p={test['p_value']:.4f} {sig}")
    
    print("\n--- Predictions ---")
    if "predictions" in results:
        for pred_name, pred_data in results["predictions"].items():
            if pred_name == "summary":
                continue
            status = "✓ PASSED" if pred_data["passed"] else "✗ FAILED"
            print(f"\n{pred_name}: {status}")
            print(f"  {pred_data['description']}")
            if "direction_correct" in pred_data:
                print(f"  Direction correct: {pred_data['direction_correct']}")
            if "statistically_significant" in pred_data:
                print(f"  Statistically significant: {pred_data['statistically_significant']}")
            if "improvement" in pred_data:
                print(f"  Improvement: {pred_data['improvement']:.4f} (threshold: {pred_data['threshold']})")
        
        if "summary" in results["predictions"]:
            summary = results["predictions"]["summary"]
            print(f"\n--- SUMMARY: {summary['passed']}/{summary['total']} predictions passed ---")
    
    if "time_to_competence" in results:
        print("\n--- Time to Competence (rounds to reach 0.5 capability in bottom quartile) ---")
        for condition, ttc in results["time_to_competence"].items():
            if ttc["mean"] is not None:
                print(f"  {condition}: {ttc['mean']:.1f} rounds (std={ttc['std']:.1f}, reached {ttc['reached_count']}/{ttc['total_runs']})")
            else:
                print(f"  {condition}: Never reached (0/{ttc['total_runs']})")
    
    # Overall validation
    print("\n" + "=" * 70)
    if "predictions" in results and "summary" in results["predictions"]:
        summary = results["predictions"]["summary"]
        if summary["passed"] >= 3:
            print("TEMPLATE SHARING VALIDATED: Knowledge transfer improves coordination")
        elif summary["passed"] >= 2:
            print("TEMPLATE SHARING PARTIALLY VALIDATED: Some benefits observed")
        else:
            print("TEMPLATE SHARING NOT VALIDATED: Insufficient evidence of benefit")
    print("=" * 70)


def run_parameter_sweep() -> Dict[str, Any]:
    """Run parameter sweep to test robustness."""
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP")
    print("=" * 70)
    
    sweep_results = {}
    
    # Test different sharing rates
    print("\n--- Sharing Rate Sweep ---")
    for rate in [0.05, 0.1, 0.2]:
        print(f"\nSharing rate: {rate}")
        results = run_full_experiment(n_runs=5, sharing_rate=rate)
        sweep_results[f"rate_{rate}"] = {
            "directed_cooperation": results["cross_run_statistics"]["directed"]["cooperation_mean"],
            "none_cooperation": results["cross_run_statistics"]["none"]["cooperation_mean"],
            "improvement": results["cross_run_statistics"]["directed"]["cooperation_mean"] - 
                          results["cross_run_statistics"]["none"]["cooperation_mean"],
            "predictions_passed": results["predictions"]["summary"]["passed"]
        }
        print(f"  Improvement: {sweep_results[f'rate_{rate}']['improvement']:.4f}")
        print(f"  Predictions passed: {sweep_results[f'rate_{rate}']['predictions_passed']}/4")
    
    # Test different population sizes
    print("\n--- Population Size Sweep ---")
    for n_agents in [30, 50, 100]:
        print(f"\nPopulation size: {n_agents}")
        results = run_full_experiment(n_runs=5, n_agents=n_agents)
        sweep_results[f"pop_{n_agents}"] = {
            "directed_cooperation": results["cross_run_statistics"]["directed"]["cooperation_mean"],
            "none_cooperation": results["cross_run_statistics"]["none"]["cooperation_mean"],
            "improvement": results["cross_run_statistics"]["directed"]["cooperation_mean"] - 
                          results["cross_run_statistics"]["none"]["cooperation_mean"],
            "predictions_passed": results["predictions"]["summary"]["passed"]
        }
        print(f"  Improvement: {sweep_results[f'pop_{n_agents}']['improvement']:.4f}")
        print(f"  Predictions passed: {sweep_results[f'pop_{n_agents}']['predictions_passed']}/4")
    
    return sweep_results


if __name__ == "__main__":
    # Run main experiment
    print("=" * 70)
    print("EXPERIMENT 24: TEMPLATE SHARING VALIDATION")
    print("=" * 70)
    print("\nPurpose: Validate that knowledge transfer (template sharing)")
    print("improves coordination before building complex social structures.")
    print()
    
    # Main experiment with 10 runs per condition
    results = run_full_experiment(n_runs=10, n_agents=50, n_rounds=200, sharing_rate=0.1)
    
    # Print results
    print_results(results)
    
    # Save results to JSON
    output_file = "results/experiment_24_template_sharing.json"
    import os
    os.makedirs("results", exist_ok=True)
    
    # Create serializable version (remove trajectories and convert numpy types)
    def make_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif obj is None:
            return None
        else:
            return obj
    
    results_serializable = {}
    for key, value in results.items():
        if key == "conditions":
            results_serializable[key] = {}
            for cond, metrics in value.items():
                results_serializable[key][cond] = {
                    k: make_serializable(v) for k, v in metrics.items()
                    if not k.endswith("_trajectory")
                }
        else:
            results_serializable[key] = make_serializable(value)
    
    with open(output_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Optional: Run parameter sweep (uncomment to run)
    # sweep_results = run_parameter_sweep()