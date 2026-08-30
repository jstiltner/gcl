#!/usr/bin/env python3
"""
Experiment 03: Multi-Agent Coordination

This experiment validates Theorem 1 (Communication Complexity):
- Commitment-based coordination: O(n·k) operations
- Representation-based coordination: O(n·k²) operations
- Error rate for representation: O(ε·n·k²)

Where:
- n = number of tasks/subtasks
- k = number of agents
- ε = semantic drift between agents

Usage:
    python experiments/03_multiagent_coordination.py [--seed S]

Example:
    python experiments/03_multiagent_coordination.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcl.multiagent.agent import CommitmentAgent
from gcl.multiagent.market import Task
from gcl.multiagent.protocol import (
    CommitmentProtocol,
    RepresentationProtocol,
    compare_protocols,
    validate_theorem_1,
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
        description="Validate Theorem 1 (Communication Complexity)",
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
        default="results/03_multiagent",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting",
    )
    return parser.parse_args()


def run_complexity_experiment(
    n_values: list[int],
    k_values: list[int],
    seed: int = 42,
) -> dict:
    """
    Run experiment varying n (tasks) and k (agents).
    
    Tests Theorem 1a and 1b:
    - Commitment: O(n·k)
    - Representation: O(n·k²)
    """
    np.random.seed(seed)
    
    results = {
        "commitment": [],
        "representation": [],
    }
    
    for n in n_values:
        for k in k_values:
            logger.info(f"Testing n={n}, k={k}")
            
            # Create tasks and agents
            tasks = [
                Task(name=f"task_{i}", difficulty=0.5, min_stake=0.1)
                for i in range(n)
            ]
            
            # Run commitment protocol
            commit_agents = [CommitmentAgent(f"commit_{i}") for i in range(k)]
            commit_protocol = CommitmentProtocol()
            _, commit_metrics = commit_protocol.coordinate(
                [Task(**t.__dict__) for t in tasks],
                commit_agents,
            )
            
            results["commitment"].append({
                "n": n,
                "k": k,
                "operations": commit_metrics.verification_operations,
                "success_rate": commit_metrics.success_rate,
                "expected_order": n * k,
            })
            
            # Run representation protocol
            rep_agents = [CommitmentAgent(f"rep_{i}") for i in range(k)]
            rep_protocol = RepresentationProtocol(semantic_drift=0.1)
            _, rep_metrics = rep_protocol.coordinate(
                [Task(**t.__dict__) for t in tasks],
                rep_agents,
            )
            
            results["representation"].append({
                "n": n,
                "k": k,
                "operations": rep_metrics.interpretation_operations,
                "success_rate": rep_metrics.success_rate,
                "error_rate": rep_metrics.error_rate,
                "expected_order": n * k * k,
            })
    
    return results


def run_drift_experiment(
    n: int,
    k: int,
    drift_values: list[float],
    seed: int = 42,
) -> dict:
    """
    Run experiment varying semantic drift.
    
    Tests Theorem 1c: Error rate scales as O(ε·n·k²)
    """
    np.random.seed(seed)
    
    results = {
        "drift_values": drift_values,
        "commitment_success": [],
        "representation_success": [],
        "representation_errors": [],
    }
    
    tasks = [
        Task(name=f"task_{i}", difficulty=0.5, min_stake=0.1)
        for i in range(n)
    ]
    
    for drift in drift_values:
        logger.info(f"Testing drift={drift}")
        
        # Commitment (unaffected by drift)
        commit_agents = [CommitmentAgent(f"commit_{i}") for i in range(k)]
        commit_protocol = CommitmentProtocol()
        _, commit_metrics = commit_protocol.coordinate(
            [Task(**t.__dict__) for t in tasks],
            commit_agents,
        )
        results["commitment_success"].append(commit_metrics.success_rate)
        
        # Representation (affected by drift)
        rep_agents = [CommitmentAgent(f"rep_{i}") for i in range(k)]
        rep_protocol = RepresentationProtocol(semantic_drift=drift)
        _, rep_metrics = rep_protocol.coordinate(
            [Task(**t.__dict__) for t in tasks],
            rep_agents,
        )
        results["representation_success"].append(rep_metrics.success_rate)
        results["representation_errors"].append(rep_metrics.error_rate)
    
    return results


def plot_complexity_results(results: dict, output_path: Path) -> None:
    """Plot complexity scaling results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    commit_data = results["commitment"]
    rep_data = results["representation"]
    
    # Get unique n and k values
    n_values = sorted(set(d["n"] for d in commit_data))
    k_values = sorted(set(d["k"] for d in commit_data))
    
    # Plot 1: Operations vs n (for each k)
    ax = axes[0, 0]
    for k in k_values:
        commit_ops = [d["operations"] for d in commit_data if d["k"] == k]
        rep_ops = [d["operations"] for d in rep_data if d["k"] == k]
        
        ax.plot(n_values, commit_ops, 'o-', label=f"Commitment (k={k})")
        ax.plot(n_values, rep_ops, 's--', label=f"Representation (k={k})")
    
    ax.set_xlabel("Number of Tasks (n)")
    ax.set_ylabel("Communication Operations")
    ax.set_title("Communication Complexity vs Tasks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Operations vs k (for each n)
    ax = axes[0, 1]
    for n in n_values:
        commit_ops = [d["operations"] for d in commit_data if d["n"] == n]
        rep_ops = [d["operations"] for d in rep_data if d["n"] == n]
        
        ax.plot(k_values, commit_ops, 'o-', label=f"Commitment (n={n})")
        ax.plot(k_values, rep_ops, 's--', label=f"Representation (n={n})")
    
    ax.set_xlabel("Number of Agents (k)")
    ax.set_ylabel("Communication Operations")
    ax.set_title("Communication Complexity vs Agents")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Actual vs Expected (Commitment)
    ax = axes[1, 0]
    actual = [d["operations"] for d in commit_data]
    expected = [d["expected_order"] for d in commit_data]
    
    ax.scatter(expected, actual, alpha=0.7, s=100)
    max_val = max(max(actual), max(expected))
    ax.plot([0, max_val], [0, max_val], 'k--', label="y=x (perfect O(n·k))")
    ax.set_xlabel("Expected O(n·k)")
    ax.set_ylabel("Actual Operations")
    ax.set_title("Commitment: Actual vs Expected Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Actual vs Expected (Representation)
    ax = axes[1, 1]
    actual = [d["operations"] for d in rep_data]
    expected = [d["expected_order"] for d in rep_data]
    
    ax.scatter(expected, actual, alpha=0.7, s=100)
    # Fit line
    coeffs = np.polyfit(expected, actual, 1)
    fit_line = np.poly1d(coeffs)
    x_fit = np.linspace(0, max(expected), 100)
    ax.plot(x_fit, fit_line(x_fit), 'r-', label=f"Fit: {coeffs[0]:.2f}x + {coeffs[1]:.1f}")
    ax.set_xlabel("Expected O(n·k²)")
    ax.set_ylabel("Actual Operations")
    ax.set_title("Representation: Actual vs Expected Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "complexity_scaling.png", dpi=150)
    plt.close()


def plot_drift_results(results: dict, output_path: Path) -> None:
    """Plot semantic drift results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    drift_values = results["drift_values"]
    
    # Plot 1: Success rate vs drift
    ax = axes[0]
    ax.plot(drift_values, results["commitment_success"], 'o-', 
            label="Commitment", linewidth=2, markersize=8)
    ax.plot(drift_values, results["representation_success"], 's--', 
            label="Representation", linewidth=2, markersize=8)
    ax.set_xlabel("Semantic Drift (ε)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Coordination Success vs Semantic Drift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 2: Error rate vs drift
    ax = axes[1]
    ax.plot(drift_values, results["representation_errors"], 's-', 
            color='red', linewidth=2, markersize=8)
    ax.set_xlabel("Semantic Drift (ε)")
    ax.set_ylabel("Interpretation Error Rate")
    ax.set_title("Representation Error Rate vs Semantic Drift")
    ax.grid(True, alpha=0.3)
    
    # Add theoretical line (error ~ ε)
    ax2 = ax.twinx()
    theoretical = [d * 0.5 for d in drift_values]  # Scaled for visibility
    ax2.plot(drift_values, theoretical, 'k--', alpha=0.5, label="O(ε) theoretical")
    ax2.set_ylabel("Theoretical O(ε)")
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path / "drift_effects.png", dpi=150)
    plt.close()


def plot_comparison_summary(
    complexity_results: dict,
    drift_results: dict,
    output_path: Path,
) -> None:
    """Plot summary comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate average metrics
    commit_ops = np.mean([d["operations"] for d in complexity_results["commitment"]])
    rep_ops = np.mean([d["operations"] for d in complexity_results["representation"]])
    
    commit_success = np.mean([d["success_rate"] for d in complexity_results["commitment"]])
    rep_success = np.mean([d["success_rate"] for d in complexity_results["representation"]])
    
    # Bar chart
    x = np.arange(2)
    width = 0.35
    
    ops_bars = ax.bar(x - width/2, [commit_ops, rep_ops], width, 
                      label="Avg Operations", color=['blue', 'orange'], alpha=0.7)
    
    ax2 = ax.twinx()
    success_bars = ax2.bar(x + width/2, [commit_success, rep_success], width,
                           label="Success Rate", color=['green', 'red'], alpha=0.7)
    
    ax.set_ylabel("Communication Operations")
    ax2.set_ylabel("Success Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(["Commitment\nO(n·k)", "Representation\nO(n·k²)"])
    ax.set_title("Protocol Comparison Summary")
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path / "protocol_comparison.png", dpi=150)
    plt.close()


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Theorem 1 validation with seed={args.seed}")
    logger.info(f"Output directory: {output_path}")
    
    # Set seed
    np.random.seed(args.seed)
    
    # Experiment 1: Complexity scaling
    logger.info("Running complexity scaling experiment...")
    n_values = [5, 10, 20, 40]
    k_values = [2, 3, 5, 8]
    
    complexity_results = run_complexity_experiment(
        n_values=n_values,
        k_values=k_values,
        seed=args.seed,
    )
    
    # Experiment 2: Semantic drift effects
    logger.info("Running semantic drift experiment...")
    drift_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    drift_results = run_drift_experiment(
        n=20,
        k=5,
        drift_values=drift_values,
        seed=args.seed,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("THEOREM 1 VALIDATION RESULTS")
    print("=" * 60)
    
    print("\n--- Complexity Scaling (Theorem 1a, 1b) ---")
    print(f"{'n':>5} {'k':>5} {'Commit Ops':>12} {'Rep Ops':>12} {'Ratio':>8}")
    print("-" * 50)
    
    for c, r in zip(complexity_results["commitment"], complexity_results["representation"]):
        if c["n"] == r["n"] and c["k"] == r["k"]:
            ratio = r["operations"] / c["operations"] if c["operations"] > 0 else 0
            print(f"{c['n']:>5} {c['k']:>5} {c['operations']:>12} {r['operations']:>12} {ratio:>8.2f}x")
    
    print("\n--- Semantic Drift Effects (Theorem 1c) ---")
    print(f"{'Drift':>8} {'Commit Success':>15} {'Rep Success':>15} {'Rep Errors':>12}")
    print("-" * 55)
    
    for i, drift in enumerate(drift_values):
        print(f"{drift:>8.2f} {drift_results['commitment_success'][i]:>15.2%} "
              f"{drift_results['representation_success'][i]:>15.2%} "
              f"{drift_results['representation_errors'][i]:>12.2%}")
    
    # Summary statistics
    print("\n--- Summary ---")
    avg_commit_ops = np.mean([d["operations"] for d in complexity_results["commitment"]])
    avg_rep_ops = np.mean([d["operations"] for d in complexity_results["representation"]])
    
    print(f"Average Commitment Operations: {avg_commit_ops:.1f}")
    print(f"Average Representation Operations: {avg_rep_ops:.1f}")
    print(f"Complexity Ratio: {avg_rep_ops / avg_commit_ops:.2f}x")
    
    # Verify Theorem 1
    print("\n--- Theorem 1 Verification ---")
    
    # Check O(n·k) for commitment
    commit_data = complexity_results["commitment"]
    commit_ratios = [d["operations"] / d["expected_order"] for d in commit_data if d["expected_order"] > 0]
    avg_commit_ratio = np.mean(commit_ratios)
    print(f"Commitment: Actual/Expected(n·k) ratio = {avg_commit_ratio:.2f}")
    print(f"  → {'✓ Consistent with O(n·k)' if 0.5 <= avg_commit_ratio <= 3.0 else '✗ Not O(n·k)'}")
    
    # Check O(n·k²) for representation
    rep_data = complexity_results["representation"]
    rep_ratios = [d["operations"] / d["expected_order"] for d in rep_data if d["expected_order"] > 0]
    avg_rep_ratio = np.mean(rep_ratios)
    print(f"Representation: Actual/Expected(n·k²) ratio = {avg_rep_ratio:.2f}")
    print(f"  → {'✓ Consistent with O(n·k²)' if 0.1 <= avg_rep_ratio <= 2.0 else '✗ Not O(n·k²)'}")
    
    # Check drift threshold (Corollary 1)
    high_drift_idx = drift_values.index(0.5) if 0.5 in drift_values else -1
    if high_drift_idx >= 0:
        commit_at_drift = drift_results["commitment_success"][high_drift_idx]
        rep_at_drift = drift_results["representation_success"][high_drift_idx]
        print(f"\nAt ε=0.5: Commitment success={commit_at_drift:.2%}, Representation success={rep_at_drift:.2%}")
        if commit_at_drift > rep_at_drift:
            print("  → ✓ Corollary 1 validated: Commitment dominates at high drift")
        else:
            print("  → Corollary 1: Results inconclusive")
    
    # Save results
    all_results = {
        "config": {
            "seed": args.seed,
            "n_values": n_values,
            "k_values": k_values,
            "drift_values": drift_values,
        },
        "complexity": complexity_results,
        "drift": drift_results,
        "summary": {
            "avg_commit_ops": avg_commit_ops,
            "avg_rep_ops": avg_rep_ops,
            "complexity_ratio": avg_rep_ops / avg_commit_ops,
            "commit_order_ratio": avg_commit_ratio,
            "rep_order_ratio": avg_rep_ratio,
        },
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path / 'results.json'}")
    
    # Generate plots
    if not args.no_plot:
        logger.info("Generating plots...")
        plot_complexity_results(complexity_results, output_path)
        plot_drift_results(drift_results, output_path)
        plot_comparison_summary(complexity_results, drift_results, output_path)
        logger.info(f"Plots saved to {output_path}")
    
    print("\n" + "=" * 60)
    print(f"Experiment complete! Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
