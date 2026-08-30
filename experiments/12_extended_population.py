"""
Experiment 12: Extended Population Experiments

This experiment studies:
1. Infrastructure ablation - What minimal infrastructure enables commitment?
2. Institutional emergence - Do institutions emerge spontaneously?
3. Phase transitions - Are there Dunbar-like thresholds?

Key questions:
- What is the minimal social infrastructure for commitment-based coordination?
- Do institutions emerge from commitment interactions?
- Are there phase transitions as population scales?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gcl.population.ablation import (
    AblationExperiment,
    InfrastructureConfig,
    InfrastructureComponent,
    AblationExperimentResult,
    run_ablation_study
)
from gcl.population.institutions import (
    InstitutionalEmergenceTracker,
    run_institutional_emergence_experiment
)

# Results directory
RESULTS_DIR = Path("results/12_extended_population")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Part 1: Infrastructure Ablation
# =============================================================================

def run_ablation_experiments() -> Dict[str, Any]:
    """
    Run infrastructure ablation experiments.
    
    Tests which components of social infrastructure are necessary
    for commitment-based coordination to function.
    """
    print("=" * 60)
    print("PART 1: Infrastructure Ablation")
    print("=" * 60)
    
    results, analysis = run_ablation_study(
        n_agents=50,
        n_rounds=100,
        seed=42
    )
    
    # Convert results to serializable format
    results_dict = {}
    for name, result in results.items():
        results_dict[name] = {
            "enabled_components": [c.name for c in result.enabled_components],
            "cooperation_rate": result.cooperation_rate,
            "commitment_fulfillment_rate": result.commitment_fulfillment_rate,
            "average_reward": result.average_reward,
            "trust_network_density": result.trust_network_density,
            "trust_network_clustering": result.trust_network_clustering,
            "reputation_variance": result.reputation_variance,
            "specialization_index": result.specialization_index,
        }
    
    # Print summary
    print("\nAblation Results:")
    print("-" * 40)
    print(f"{'Configuration':<30} {'Cooperation':>12}")
    print("-" * 40)
    
    for name, result in sorted(results_dict.items(), key=lambda x: -x[1]["cooperation_rate"]):
        print(f"{name:<30} {result['cooperation_rate']:>12.1%}")
    
    print("\nComponent Importance (drop in cooperation when removed):")
    print("-" * 40)
    for component, importance in sorted(
        analysis["component_importance"].items(),
        key=lambda x: -x[1]
    ):
        print(f"  {component:<20} {importance:>+.1%}")
    
    print(f"\nMinimal Viable Infrastructure: {analysis['minimal_viable_infrastructure']}")
    
    return {
        "results": results_dict,
        "analysis": analysis
    }


# =============================================================================
# Part 2: Institutional Emergence
# =============================================================================

def run_institutional_experiments() -> Dict[str, Any]:
    """
    Run institutional emergence experiments.
    
    Studies whether institutions emerge spontaneously from
    commitment-based interactions.
    """
    print("\n" + "=" * 60)
    print("PART 2: Institutional Emergence")
    print("=" * 60)
    
    # Run with different cooperation biases
    results = {}
    
    for bias in [0.3, 0.5, 0.7]:
        print(f"\nRunning with cooperation_bias = {bias}")
        tracker, summary = run_institutional_emergence_experiment(
            n_agents=50,
            n_rounds=200,
            cooperation_bias=bias,
            seed=42
        )
        results[f"bias_{bias}"] = summary
        
        print(f"  Norms emerged: {summary['norms']['count']}")
        print(f"  Coalitions formed: {summary['coalitions']['count']}")
        print(f"  Roles differentiated: {summary['roles']['count']}")
        print(f"  Institutions formed: {summary['institutions']['count']}")
        print(f"  Complexity score: {summary['complexity_score']:.3f}")
    
    return results


# =============================================================================
# Part 3: Phase Transitions (Dunbar Number)
# =============================================================================

def run_phase_transition_experiments() -> Dict[str, Any]:
    """
    Study phase transitions as population scales.
    
    Tests whether there are Dunbar-like thresholds where
    coordination dynamics change qualitatively.
    """
    print("\n" + "=" * 60)
    print("PART 3: Phase Transitions (Dunbar Number)")
    print("=" * 60)
    
    # Test different population sizes
    population_sizes = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    
    results = {
        "population_sizes": population_sizes,
        "cooperation_rates": [],
        "institutional_complexity": [],
        "trust_network_density": [],
        "coalition_count": [],
        "avg_coalition_size": [],
    }
    
    for n_agents in population_sizes:
        print(f"\nTesting population size: {n_agents}")
        
        # Run ablation experiment for cooperation metrics
        experiment = AblationExperiment(
            n_agents=n_agents,
            n_rounds=100,
            interactions_per_round=n_agents * 2,
            seed=42
        )
        ablation_result = experiment.run_experiment(InfrastructureConfig.full())
        
        # Run institutional emergence
        tracker, inst_summary = run_institutional_emergence_experiment(
            n_agents=n_agents,
            n_rounds=100,
            cooperation_bias=0.5,
            seed=42
        )
        
        results["cooperation_rates"].append(ablation_result.cooperation_rate)
        results["institutional_complexity"].append(inst_summary["complexity_score"])
        results["trust_network_density"].append(ablation_result.trust_network_density)
        results["coalition_count"].append(inst_summary["coalitions"]["count"])
        results["avg_coalition_size"].append(inst_summary["coalitions"]["avg_size"])
        
        print(f"  Cooperation: {ablation_result.cooperation_rate:.1%}")
        print(f"  Complexity: {inst_summary['complexity_score']:.3f}")
        print(f"  Coalitions: {inst_summary['coalitions']['count']}")
    
    # Analyze phase transitions
    transitions = analyze_phase_transitions(results)
    results["transitions"] = transitions
    
    return results


def analyze_phase_transitions(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze results for phase transitions.
    
    Looks for:
    - Sudden changes in cooperation rate
    - Threshold where institutions emerge
    - Dunbar-like limits on coalition size
    """
    sizes = np.array(results["population_sizes"])
    coop = np.array(results["cooperation_rates"])
    complexity = np.array(results["institutional_complexity"])
    density = np.array(results["trust_network_density"])
    
    transitions = {
        "cooperation_threshold": None,
        "institutional_threshold": None,
        "dunbar_estimate": None,
        "observations": []
    }
    
    # Look for cooperation drop
    coop_diff = np.diff(coop)
    if np.any(coop_diff < -0.1):
        idx = np.argmin(coop_diff)
        transitions["cooperation_threshold"] = int(sizes[idx + 1])
        transitions["observations"].append(
            f"Cooperation drops significantly at n={sizes[idx + 1]}"
        )
    
    # Look for institutional emergence
    for i, c in enumerate(complexity):
        if c > 0.3 and (i == 0 or complexity[i-1] <= 0.3):
            transitions["institutional_threshold"] = int(sizes[i])
            transitions["observations"].append(
                f"Institutions emerge at n={sizes[i]}"
            )
            break
    
    # Estimate Dunbar number from trust network density
    # Dunbar's number is where personal relationships become unsustainable
    for i, d in enumerate(density):
        if d < 0.1 and (i == 0 or density[i-1] >= 0.1):
            transitions["dunbar_estimate"] = int(sizes[i])
            transitions["observations"].append(
                f"Trust network becomes sparse at n={sizes[i]} (Dunbar-like threshold)"
            )
            break
    
    # Check for coalition size limits
    avg_sizes = results["avg_coalition_size"]
    if avg_sizes:
        max_coalition = max(avg_sizes)
        if max_coalition > 0:
            transitions["max_coalition_size"] = max_coalition
            transitions["observations"].append(
                f"Maximum average coalition size: {max_coalition:.1f}"
            )
    
    return transitions


# =============================================================================
# Part 4: Visualization
# =============================================================================

def create_visualizations(
    ablation_results: Dict[str, Any],
    institutional_results: Dict[str, Any],
    phase_results: Dict[str, Any]
) -> None:
    """Create visualizations for all experiments."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Ablation results
    ax1 = axes[0, 0]
    configs = list(ablation_results["results"].keys())
    coop_rates = [ablation_results["results"][c]["cooperation_rate"] for c in configs]
    
    # Sort by cooperation rate
    sorted_pairs = sorted(zip(configs, coop_rates), key=lambda x: -x[1])
    configs, coop_rates = zip(*sorted_pairs)
    
    colors = ['green' if c == 'full' else 'red' if c == 'minimal' else 'blue' for c in configs]
    ax1.barh(range(len(configs)), coop_rates, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels(configs, fontsize=8)
    ax1.set_xlabel("Cooperation Rate")
    ax1.set_title("Infrastructure Ablation Results")
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Institutional emergence over cooperation bias
    ax2 = axes[0, 1]
    biases = [0.3, 0.5, 0.7]
    metrics = ["norms", "coalitions", "roles", "institutions"]
    
    x = np.arange(len(biases))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [institutional_results[f"bias_{b}"][metric]["count"] for b in biases]
        ax2.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax2.set_xlabel("Cooperation Bias")
    ax2.set_ylabel("Count")
    ax2.set_title("Institutional Emergence by Cooperation Bias")
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([str(b) for b in biases])
    ax2.legend()
    
    # Plot 3: Phase transitions - cooperation and complexity
    ax3 = axes[1, 0]
    sizes = phase_results["population_sizes"]
    
    ax3.plot(sizes, phase_results["cooperation_rates"], 'b-o', label="Cooperation Rate")
    ax3.set_xlabel("Population Size")
    ax3.set_ylabel("Cooperation Rate", color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(sizes, phase_results["institutional_complexity"], 'r-s', label="Complexity")
    ax3_twin.set_ylabel("Institutional Complexity", color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    ax3.set_title("Phase Transitions: Cooperation & Complexity vs Population")
    
    # Add transition markers
    if phase_results["transitions"].get("cooperation_threshold"):
        ax3.axvline(x=phase_results["transitions"]["cooperation_threshold"], 
                   color='blue', linestyle='--', alpha=0.5)
    if phase_results["transitions"].get("dunbar_estimate"):
        ax3.axvline(x=phase_results["transitions"]["dunbar_estimate"],
                   color='green', linestyle='--', alpha=0.5, label="Dunbar threshold")
    
    # Plot 4: Trust network density and coalition dynamics
    ax4 = axes[1, 1]
    ax4.plot(sizes, phase_results["trust_network_density"], 'g-o', label="Trust Density")
    ax4.set_xlabel("Population Size")
    ax4.set_ylabel("Trust Network Density", color='green')
    ax4.tick_params(axis='y', labelcolor='green')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(sizes, phase_results["coalition_count"], 'm-s', label="Coalition Count")
    ax4_twin.set_ylabel("Number of Coalitions", color='magenta')
    ax4_twin.tick_params(axis='y', labelcolor='magenta')
    
    ax4.set_title("Trust Networks & Coalitions vs Population")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "extended_population_results.png", dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {RESULTS_DIR / 'extended_population_results.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all extended population experiments."""
    print("=" * 60)
    print("EXPERIMENT 12: Extended Population Experiments")
    print("=" * 60)
    
    all_results = {}
    
    # Part 1: Infrastructure Ablation
    ablation_results = run_ablation_experiments()
    all_results["ablation"] = ablation_results
    
    # Part 2: Institutional Emergence
    institutional_results = run_institutional_experiments()
    all_results["institutional"] = institutional_results
    
    # Part 3: Phase Transitions
    phase_results = run_phase_transition_experiments()
    all_results["phase_transitions"] = phase_results
    
    # Create visualizations
    create_visualizations(ablation_results, institutional_results, phase_results)
    
    # Save results
    (RESULTS_DIR / "full_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n1. INFRASTRUCTURE ABLATION")
    print("-" * 40)
    print(f"Minimal viable infrastructure: {ablation_results['analysis']['minimal_viable_infrastructure']}")
    print("Most important components:")
    for comp, imp in sorted(
        ablation_results['analysis']['component_importance'].items(),
        key=lambda x: -x[1]
    )[:3]:
        print(f"  - {comp}: {imp:+.1%} impact")
    
    print("\n2. INSTITUTIONAL EMERGENCE")
    print("-" * 40)
    best_bias = max(institutional_results.keys(), 
                   key=lambda k: institutional_results[k]["complexity_score"])
    print(f"Highest complexity at: {best_bias}")
    print(f"  Complexity score: {institutional_results[best_bias]['complexity_score']:.3f}")
    print(f"  Norms: {institutional_results[best_bias]['norms']['count']}")
    print(f"  Coalitions: {institutional_results[best_bias]['coalitions']['count']}")
    print(f"  Institutions: {institutional_results[best_bias]['institutions']['count']}")
    
    print("\n3. PHASE TRANSITIONS")
    print("-" * 40)
    transitions = phase_results["transitions"]
    for obs in transitions.get("observations", []):
        print(f"  - {obs}")
    
    if transitions.get("dunbar_estimate"):
        print(f"\n  DUNBAR-LIKE THRESHOLD: ~{transitions['dunbar_estimate']} agents")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    main()
