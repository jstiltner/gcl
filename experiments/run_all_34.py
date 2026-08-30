"""
Run all Experiment 34 variants (A-E) and generate summary report.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from datetime import datetime


def run_all_experiments():
    """Run all 34A-E experiments and collect results."""
    print("=" * 80)
    print("RUNNING ALL EXPERIMENT 34 VARIANTS")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    results = {}
    
    # 34A: Self-Selection Mechanism Decomposition
    print("\n" + "=" * 80)
    print("EXPERIMENT 34A: Self-Selection Mechanism Decomposition")
    print("=" * 80)
    from experiments import exp_34a_self_selection_decomposition as exp_34a
    results["34a"] = exp_34a.run_experiment(n_rounds=100, n_seeds=10, n_agents=30)
    results["34a_analysis"] = exp_34a.analyze_results(results["34a"])
    
    # 34B: Team Task Coordination
    print("\n" + "=" * 80)
    print("EXPERIMENT 34B: Team Task Coordination")
    print("=" * 80)
    from experiments import exp_34b_team_tasks as exp_34b
    results["34b"] = exp_34b.run_experiment(n_rounds=100, n_seeds=10, n_agents=30)
    results["34b_analysis"] = exp_34b.analyze_results(results["34b"])
    
    # 34C: Sharing-Specialization Frontier
    print("\n" + "=" * 80)
    print("EXPERIMENT 34C: Sharing-Specialization Frontier")
    print("=" * 80)
    from experiments import exp_34c_sharing_frontier as exp_34c
    results["34c"] = exp_34c.run_experiment(n_rounds=100, n_seeds=10, n_agents=30)
    results["34c_analysis"] = exp_34c.analyze_results(results["34c"])
    
    # 34D: Long-Horizon Dynamics
    print("\n" + "=" * 80)
    print("EXPERIMENT 34D: Long-Horizon Dynamics")
    print("=" * 80)
    from experiments import exp_34d_long_horizon as exp_34d
    results["34d"] = exp_34d.run_experiment(n_seeds=5, n_agents=30)
    results["34d_analysis"] = exp_34d.analyze_results(results["34d"])
    
    # 34E: Task Scarcity
    print("\n" + "=" * 80)
    print("EXPERIMENT 34E: Task Scarcity")
    print("=" * 80)
    from experiments import exp_34e_task_scarcity as exp_34e
    results["34e"] = exp_34e.run_experiment(n_rounds=100, n_seeds=10, n_agents=30)
    results["34e_analysis"] = exp_34e.analyze_results(results["34e"])
    
    return results


def generate_summary(results: dict) -> str:
    """Generate summary report of all experiments."""
    summary = []
    summary.append("=" * 80)
    summary.append("EXPERIMENT 34 SERIES: SUMMARY REPORT")
    summary.append("=" * 80)
    summary.append("")
    
    # 34A Summary
    summary.append("## 34A: Self-Selection Mechanism Decomposition")
    analysis = results.get("34a_analysis", {})
    summary.append(f"- Information effect: {analysis.get('info_effect', 0):+.4f}")
    summary.append(f"- Effort effect: {analysis.get('effort_effect', 0):+.4f}")
    summary.append(f"- Learning effect: {analysis.get('learning_effect', 0):+.4f}")
    summary.append(f"- Risk effect: {analysis.get('risk_effect', 0):+.4f}")
    summary.append(f"- Best condition: {analysis.get('best_condition', 'N/A')}")
    summary.append("")
    
    # 34B Summary
    summary.append("## 34B: Team Task Coordination")
    analysis = results.get("34b_analysis", {})
    summary.append(f"- Self-selection avg: {analysis.get('overall_self_selection', 0):.3f}")
    summary.append(f"- Capability-matched avg: {analysis.get('overall_capability_matched', 0):.3f}")
    summary.append(f"- Self-selection advantage: {analysis.get('self_selection_advantage', 0):+.3f}")
    summary.append("")
    
    # 34C Summary
    summary.append("## 34C: Sharing-Specialization Frontier")
    analysis = results.get("34c_analysis", {})
    summary.append(f"- Best rate for cooperation: {analysis.get('best_coop_rate', 0)}%")
    summary.append(f"- Best rate for specialization: {analysis.get('best_spec_rate', 0)}%")
    summary.append(f"- Pareto optimal rates: {analysis.get('pareto_optimal_rates', [])}")
    summary.append("")
    
    # 34D Summary
    summary.append("## 34D: Long-Horizon Dynamics")
    analysis = results.get("34d_analysis", {})
    summary.append(f"- Specialization-time correlation: {analysis.get('specialization_time_correlation', 0):.3f}")
    summary.append(f"- Max specialization: {analysis.get('max_specialization', 0):.3f}")
    summary.append(f"- Stabilization horizon: {analysis.get('stabilization_horizon', 'N/A')}")
    summary.append("")
    
    # 34E Summary
    summary.append("## 34E: Task Scarcity")
    analysis = results.get("34e_analysis", {})
    summary.append(f"- Scarcity effect: {analysis.get('scarcity_effect', 0):+.3f}")
    summary.append(f"- Scarcity induces specialization: {analysis.get('scarcity_induces_specialization', False)}")
    summary.append(f"- Bonus attracts specialists: {analysis.get('bonus_attracts_specialists', False)}")
    summary.append("")
    
    # Key Findings
    summary.append("=" * 80)
    summary.append("KEY FINDINGS ACROSS ALL EXPERIMENTS")
    summary.append("=" * 80)
    summary.append("")
    
    return "\n".join(summary)


def main():
    """Run all experiments and generate report."""
    try:
        results = run_all_experiments()
        
        # Generate summary
        summary = generate_summary(results)
        print("\n" + summary)
        
        # Save summary
        os.makedirs("results", exist_ok=True)
        with open("results/experiment_34_summary.txt", "w") as f:
            f.write(summary)
        
        print(f"\nCompleted at: {datetime.now().isoformat()}")
        print("Summary saved to results/experiment_34_summary.txt")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Running experiments individually instead...")
        run_individually()


def run_individually():
    """Run each experiment as a separate script."""
    import subprocess
    
    experiments = [
        "34a_self_selection_decomposition",
        "34b_team_tasks",
        "34c_sharing_frontier",
        "34d_long_horizon",
        "34e_task_scarcity",
    ]
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp}...")
        print('='*60)
        result = subprocess.run(
            ["python", f"experiments/{exp}.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode != 0:
            print(f"Warning: {exp} exited with code {result.returncode}")


if __name__ == "__main__":
    # Run individually since imports may have issues
    run_individually()
