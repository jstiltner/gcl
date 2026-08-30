#!/usr/bin/env python3
"""
Experiment 07: Population Dynamics and Emergent Phenomena

This experiment validates the five falsifiable predictions from SECOND_ADDENDUM:
1. Protocol Convergence: Entropy decreases logarithmically
2. Trust Network Structure: Small-world properties emerge
3. Template Fitness Dynamics: Replicator dynamics
4. Specialization Emergence: Gini coefficient increases
5. Critical Population Size: N* ≈ 20-50

Track 2 experiments (local compute, 50-500 agents):
- H.1: Protocol convergence
- H.2: Trust network formation
- H.3: Template propagation
- H.4: Specialization emergence
- H.5: Critical population thresholds

Usage:
    python experiments/07_population_dynamics.py [--n-agents N] [--n-timesteps T] [--seed S]

Example:
    python experiments/07_population_dynamics.py --n-agents 100 --n-timesteps 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from gcl.population.environment import PopulationConfig, PopulationEnvironment
from gcl.population.predictions import (
    test_prediction_1_protocol_convergence,
    test_prediction_2_trust_network,
    test_prediction_3_template_dynamics,
    test_prediction_4_specialization,
    test_all_predictions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run population dynamics experiments (Track 2)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=100,
        help="Number of agents (default: 100)",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=5000,
        help="Number of timesteps (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/07_population",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with fewer timesteps",
    )
    return parser.parse_args()


def run_population_experiment(
    n_agents: int = 100,
    n_timesteps: int = 5000,
    seed: int = 42,
) -> tuple[PopulationEnvironment, Dict[str, Any]]:
    """
    Run a population dynamics experiment.
    
    Args:
        n_agents: Number of agents.
        n_timesteps: Number of timesteps.
        seed: Random seed.
        
    Returns:
        Tuple of (environment, metrics dictionary).
    """
    logger.info(f"Running population experiment: N={n_agents}, T={n_timesteps}, seed={seed}")
    
    config = PopulationConfig(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        task_complexity=4,
        commitment_vocab=16,
        state_dim=32,
        interaction_type="random",
    )
    
    env = PopulationEnvironment(config)
    env.set_seed(seed)
    
    # Run simulation using the run() method which initializes metrics
    metrics = env.run(n_timesteps)
    
    # Log final summary
    summary = metrics.get_summary()
    logger.info(
        f"  Final: success={summary.get('task_success_rate', 0):.2%}, "
        f"entropy={summary.get('commitment_entropy', 0):.2f}, "
        f"templates={summary.get('template_count', 0)}"
    )
    
    return env, metrics.to_dict()


def plot_time_series(metrics: Dict[str, Any], output_path: Path) -> None:
    """Plot time series metrics."""
    ts = metrics.get("time_series", {})
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    timesteps = ts.get("timesteps", [])
    if not timesteps:
        return
    
    # Plot 1: Task Success Rate
    ax = axes[0, 0]
    ax.plot(timesteps, ts.get("task_success_rate", []), color='#2ecc71', alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Success Rate")
    ax.set_title("Task Success Rate Over Time")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Commitment Entropy
    ax = axes[0, 1]
    ax.plot(timesteps, ts.get("commitment_entropy", []), color='#3498db', alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Entropy")
    ax.set_title("Commitment Type Entropy (Protocol Convergence)")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Template Count
    ax = axes[0, 2]
    ax.plot(timesteps, ts.get("template_count", []), color='#9b59b6', alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total Templates")
    ax.set_title("Template Count Over Time")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Average Reputation
    ax = axes[1, 0]
    ax.plot(timesteps, ts.get("avg_reputation", []), color='#e74c3c', alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Avg Reputation")
    ax.set_title("Average Agent Reputation")
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Specialization Index
    ax = axes[1, 1]
    ax.plot(timesteps, ts.get("specialization_index", []), color='#f39c12', alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Specialization (Gini)")
    ax.set_title("Agent Specialization Over Time")
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Reputation Gini
    ax = axes[1, 2]
    ax.plot(timesteps, ts.get("reputation_gini", []), color='#1abc9c', alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Reputation Inequality")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "time_series.png", dpi=150)
    plt.close()


def plot_network_evolution(metrics: Dict[str, Any], output_path: Path) -> None:
    """Plot trust network evolution."""
    network_metrics = metrics.get("network_metrics", [])
    
    if not network_metrics:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    timesteps = [i * 100 for i in range(len(network_metrics))]
    
    # Plot 1: Network Density
    ax = axes[0]
    densities = [m.get("density", 0) for m in network_metrics]
    ax.plot(timesteps, densities, 'o-', color='#3498db')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Density")
    ax.set_title("Trust Network Density")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Clustering Coefficient
    ax = axes[1]
    clustering = [m.get("clustering", 0) for m in network_metrics]
    ax.plot(timesteps, clustering, 'o-', color='#2ecc71')
    ax.axhline(y=0.3, color='r', linestyle='--', label='Threshold (0.3)')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Clustering")
    ax.set_title("Trust Network Clustering")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average Degree
    ax = axes[2]
    degrees = [m.get("avg_out_degree", 0) for m in network_metrics]
    ax.plot(timesteps, degrees, 'o-', color='#9b59b6')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Avg Degree")
    ax.set_title("Average Node Degree")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "network_evolution.png", dpi=150)
    plt.close()


def plot_prediction_results(results: Dict[str, Any], output_path: Path) -> None:
    """Plot prediction test results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    predictions = []
    passed = []
    
    for key in ["prediction_1", "prediction_2", "prediction_3", "prediction_4"]:
        if key in results:
            pred = results[key]
            name = pred.get("prediction", key)
            predictions.append(name.replace(" ", "\n"))
            passed.append(1 if pred.get("passed", False) else 0)
    
    colors = ['#2ecc71' if p else '#e74c3c' for p in passed]
    bars = ax.bar(predictions, passed, color=colors)
    
    ax.set_ylabel("Passed (1=Yes, 0=No)")
    ax.set_title("Falsifiable Prediction Test Results")
    ax.set_ylim(0, 1.2)
    
    # Add pass/fail labels
    for bar, p in zip(bars, passed):
        label = "✓" if p else "✗"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            label,
            ha='center',
            va='bottom',
            fontsize=20,
        )
    
    plt.tight_layout()
    plt.savefig(output_path / "prediction_results.png", dpi=150)
    plt.close()


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Adjust for quick mode
    n_timesteps = 1000 if args.quick else args.n_timesteps
    
    logger.info(f"Starting Population Dynamics Experiment")
    logger.info(f"  Agents: {args.n_agents}")
    logger.info(f"  Timesteps: {n_timesteps}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {output_path}")
    
    # Run main experiment
    env, metrics = run_population_experiment(
        n_agents=args.n_agents,
        n_timesteps=n_timesteps,
        seed=args.seed,
    )
    
    # Run prediction tests
    logger.info("Running prediction tests...")
    prediction_results = test_all_predictions(env, run_threshold_test=False)
    
    # Print results
    print("\n" + "=" * 60)
    print("POPULATION DYNAMICS EXPERIMENT RESULTS")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Agents: {args.n_agents}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Seed: {args.seed}")
    
    print("\n--- Convergence Analysis ---")
    convergence = metrics.get("convergence", {})
    print(f"Converged: {'✓' if convergence.get('converged') else '✗'}")
    print(f"Entropy decrease: {convergence.get('entropy_decrease', 0):.1%}")
    print(f"Success increase: {convergence.get('success_increase', 0):.1%}")
    print(f"Early entropy: {convergence.get('early_entropy', 0):.2f}")
    print(f"Late entropy: {convergence.get('late_entropy', 0):.2f}")
    
    print("\n--- Prediction Test Results ---")
    for key in ["prediction_1", "prediction_2", "prediction_3", "prediction_4"]:
        if key in prediction_results:
            pred = prediction_results[key]
            status = "✓" if pred.get("passed") else "✗"
            name = pred.get("prediction", key)
            print(f"{status} {name}")
            
            # Print key metrics
            if "alpha" in pred:
                print(f"    α = {pred['alpha']:.4f}, R² = {pred['r_squared']:.3f}")
            if "clustering" in pred:
                print(f"    Clustering = {pred['clustering']:.3f}")
            if "correlation" in pred:
                print(f"    Correlation = {pred['correlation']:.3f}")
            if "late_specialization" in pred:
                print(f"    Specialization = {pred['late_specialization']:.3f}")
    
    summary = prediction_results.get("summary", {})
    print(f"\nOverall: {summary.get('passed', 0)}/{summary.get('total', 0)} predictions passed")
    
    print("\n--- Agent Statistics ---")
    agent_stats = env.get_agent_statistics()
    print(f"Avg reputation: {agent_stats['avg_reputation']:.2f}")
    print(f"Avg success rate: {agent_stats['avg_success_rate']:.1%}")
    print(f"Total templates: {agent_stats['total_templates']}")
    
    print("\n--- Trust Network ---")
    network_stats = env.get_trust_network_stats()
    print(f"Mean trust: {network_stats['mean_trust']:.2f}")
    print(f"High trust edges: {network_stats['high_trust_edges']}")
    print(f"Trust density: {network_stats['trust_density']:.1%}")
    
    # Save results
    all_results = {
        "config": {
            "n_agents": args.n_agents,
            "n_timesteps": n_timesteps,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": metrics,
        "predictions": {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict)) or len(str(vv)) < 1000}
            for k, v in prediction_results.items()
        },
        "agent_stats": agent_stats,
        "network_stats": network_stats,
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path / 'results.json'}")
    
    # Generate plots
    if not args.no_plot:
        logger.info("Generating plots...")
        plot_time_series(metrics, output_path)
        plot_network_evolution(metrics, output_path)
        plot_prediction_results(prediction_results, output_path)
        logger.info(f"Plots saved to {output_path}")
    
    print("\n" + "=" * 60)
    print(f"Experiment complete! Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
