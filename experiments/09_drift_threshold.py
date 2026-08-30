#!/usr/bin/env python3
"""
Experiment 09: Drift Threshold Discovery

This experiment systematically varies drift rate to find the critical threshold
where GCL's advantage over natural language coordination emerges.

HYPOTHESIS: There exists a drift threshold ε* ≈ 0.2 where:
- Below ε*: Natural language coordination works adequately
- Above ε*: GCL's grounded verification becomes essential

This is the "empirical surprise" - a specific, falsifiable prediction that
emerges from the theory but requires experimental validation.

The experiment:
1. Varies drift rate from 0.0 to 0.5 in 0.05 increments
2. Measures coordination success for both GCL and baseline
3. Identifies the crossover point where GCL becomes superior
4. Tests statistical significance of the threshold

Author: GCL Research Team
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gcl.core.calculus import (
    Commitment, FailureMode, Outcome,
    issue, verify, settle, confidence
)


@dataclass
class DriftExperimentConfig:
    """Configuration for drift threshold experiment."""
    drift_rates: List[float]  # Drift rates to test
    episodes_per_rate: int    # Episodes per drift rate
    steps_per_episode: int    # Steps per episode
    n_agents: int             # Number of agents
    seed: int                 # Random seed
    
    @classmethod
    def default(cls) -> "DriftExperimentConfig":
        return cls(
            drift_rates=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            episodes_per_rate=50,
            steps_per_episode=100,
            n_agents=4,
            seed=42,
        )


@dataclass
class DriftResult:
    """Results for a single drift rate."""
    drift_rate: float
    gcl_success_rate: float
    gcl_std: float
    baseline_success_rate: float
    baseline_std: float
    gcl_advantage: float  # GCL - baseline
    p_value: float        # Statistical significance
    
    @property
    def gcl_superior(self) -> bool:
        """Is GCL statistically significantly better?"""
        return self.gcl_advantage > 0 and self.p_value < 0.05


class DriftingEnvironment:
    """
    Environment where action semantics drift over time.
    
    Drift means that the same action may have different effects
    at different times, simulating:
    - API changes
    - Model updates
    - Environmental shifts
    - Semantic drift in language
    """
    
    def __init__(self, drift_rate: float, seed: int = 42):
        self.drift_rate = drift_rate
        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.action_meanings: Dict[str, int] = {
            "action_a": 0,
            "action_b": 1,
            "action_c": 2,
            "action_d": 3,
        }
        self.drift_history: List[Dict[str, int]] = []
        
    def step(self) -> None:
        """Advance time, potentially causing drift."""
        self.time += 1
        
        # Each action has drift_rate probability of changing meaning
        for action in self.action_meanings:
            if self.rng.random() < self.drift_rate:
                # Drift: action now means something different
                old_meaning = self.action_meanings[action]
                new_meaning = self.rng.integers(0, 4)
                self.action_meanings[action] = new_meaning
                
        self.drift_history.append(self.action_meanings.copy())
        
    def execute(self, action: str, expected_effect: int) -> Tuple[bool, int]:
        """
        Execute action and check if it had expected effect.
        
        Returns:
            (success, actual_effect)
        """
        actual_effect = self.action_meanings.get(action, -1)
        success = (actual_effect == expected_effect)
        return success, actual_effect
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time = 0
        self.action_meanings = {
            "action_a": 0,
            "action_b": 1,
            "action_c": 2,
            "action_d": 3,
        }
        self.drift_history = []


class GCLAgent:
    """
    Agent using GCL for coordination.
    
    Key property: Verifies action effects and updates confidence
    based on observed outcomes, not assumed semantics.
    """
    
    def __init__(self, agent_id: str, seed: int = 42):
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)
        
        # Track confidence in each action-effect mapping
        self.action_confidence: Dict[str, Dict[int, Tuple[int, int]]] = {
            action: {effect: (1, 1) for effect in range(4)}  # Laplace prior
            for action in ["action_a", "action_b", "action_c", "action_d"]
        }
        
        # Current beliefs about action meanings
        self.beliefs: Dict[str, int] = {
            "action_a": 0,
            "action_b": 1,
            "action_c": 2,
            "action_d": 3,
        }
        
    def select_action(self, desired_effect: int) -> str:
        """Select action most likely to produce desired effect."""
        best_action = None
        best_confidence = -1.0
        
        for action in self.beliefs:
            # Get confidence that this action produces desired effect
            successes, failures = self.action_confidence[action][desired_effect]
            conf = confidence(successes, failures)
            
            if conf > best_confidence:
                best_confidence = conf
                best_action = action
                
        return best_action
    
    def update(self, action: str, expected_effect: int, actual_effect: int) -> None:
        """Update beliefs based on observed outcome."""
        # Update confidence for the expected effect
        successes, failures = self.action_confidence[action][expected_effect]
        if actual_effect == expected_effect:
            self.action_confidence[action][expected_effect] = (successes + 1, failures)
        else:
            self.action_confidence[action][expected_effect] = (successes, failures + 1)
            
        # Update belief about what this action does
        # Use maximum likelihood estimate
        best_effect = max(
            range(4),
            key=lambda e: confidence(*self.action_confidence[action][e])
        )
        self.beliefs[action] = best_effect
        
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.action_confidence = {
            action: {effect: (1, 1) for effect in range(4)}
            for action in ["action_a", "action_b", "action_c", "action_d"]
        }
        self.beliefs = {
            "action_a": 0,
            "action_b": 1,
            "action_c": 2,
            "action_d": 3,
        }


class BaselineAgent:
    """
    Agent using natural language coordination (no grounded verification).
    
    Key property: Relies on memory of initial semantics and occasional
    "discussion" to update beliefs. Does NOT have oracle access to true
    meanings - must infer from past experience like GCL, but with:
    - Slower updates (only on explicit "sync" events)
    - No systematic tracking of confidence
    - Assumes stability between syncs
    
    This models how NL coordination actually works: agents discuss and
    agree on meanings, but don't continuously verify.
    """
    
    def __init__(self, agent_id: str, seed: int = 42):
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)
        
        # Fixed beliefs about action meanings (no verification)
        self.beliefs: Dict[str, int] = {
            "action_a": 0,
            "action_b": 1,
            "action_c": 2,
            "action_d": 3,
        }
        
        # Memory of recent observations (for periodic sync)
        self.observation_buffer: List[Tuple[str, int, int]] = []  # (action, expected, actual)
        
        # Communication delay: how often beliefs are synchronized
        self.sync_interval = 20  # Every 20 steps (slower than GCL's immediate updates)
        self.steps_since_sync = 0
        
    def select_action(self, desired_effect: int) -> str:
        """Select action believed to produce desired effect."""
        for action, effect in self.beliefs.items():
            if effect == desired_effect:
                return action
        # Fallback: random action
        return self.rng.choice(list(self.beliefs.keys()))
    
    def update(self, action: str, expected_effect: int, actual_effect: int,
               true_meanings: Dict[str, int] = None) -> None:
        """
        Update beliefs based on buffered observations.
        
        Unlike GCL which updates immediately, baseline only updates
        during periodic "sync" events, simulating NL discussion.
        """
        # Buffer the observation
        self.observation_buffer.append((action, expected_effect, actual_effect))
        self.steps_since_sync += 1
        
        if self.steps_since_sync >= self.sync_interval:
            # Periodic sync: analyze buffer to update beliefs
            # This simulates "discussing what went wrong"
            action_observations: Dict[str, List[int]] = {a: [] for a in self.beliefs}
            
            for obs_action, obs_expected, obs_actual in self.observation_buffer:
                action_observations[obs_action].append(obs_actual)
            
            # Update beliefs based on most common observed effect
            for action, observations in action_observations.items():
                if observations:
                    # Use mode (most common) as new belief
                    # But with noise to simulate miscommunication
                    from collections import Counter
                    counts = Counter(observations)
                    if counts:
                        most_common = counts.most_common(1)[0][0]
                        # 80% chance of correct update (NL is noisy)
                        if self.rng.random() < 0.8:
                            self.beliefs[action] = most_common
            
            # Clear buffer and reset counter
            self.observation_buffer = []
            self.steps_since_sync = 0
            
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.beliefs = {
            "action_a": 0,
            "action_b": 1,
            "action_c": 2,
            "action_d": 3,
        }
        self.observation_buffer = []
        self.steps_since_sync = 0


def run_episode(
    env: DriftingEnvironment,
    agents: List,
    is_gcl: bool,
    steps: int,
) -> float:
    """
    Run a single episode and return success rate.
    
    Args:
        env: The drifting environment
        agents: List of agents (GCL or baseline)
        is_gcl: Whether agents are GCL agents
        steps: Number of steps to run
        
    Returns:
        Success rate (0.0 to 1.0)
    """
    successes = 0
    total = 0
    
    for step in range(steps):
        # Environment may drift
        env.step()
        
        # Each agent tries to achieve a random effect
        for agent in agents:
            desired_effect = np.random.randint(0, 4)
            action = agent.select_action(desired_effect)
            success, actual_effect = env.execute(action, desired_effect)
            
            if success:
                successes += 1
            total += 1
            
            # Update agent
            if is_gcl:
                agent.update(action, desired_effect, actual_effect)
            else:
                agent.update(action, desired_effect, actual_effect, 
                           env.action_meanings)
                
    return successes / total if total > 0 else 0.0


def run_drift_experiment(config: DriftExperimentConfig) -> List[DriftResult]:
    """
    Run the full drift threshold experiment.
    
    Returns:
        List of results for each drift rate
    """
    results = []
    
    for drift_rate in config.drift_rates:
        print(f"\nTesting drift rate: {drift_rate:.2f}")
        
        gcl_successes = []
        baseline_successes = []
        
        for episode in range(config.episodes_per_rate):
            seed = config.seed + episode
            
            # Run GCL episode
            env = DriftingEnvironment(drift_rate, seed)
            gcl_agents = [GCLAgent(f"gcl_{i}", seed + i) for i in range(config.n_agents)]
            gcl_rate = run_episode(env, gcl_agents, is_gcl=True, steps=config.steps_per_episode)
            gcl_successes.append(gcl_rate)
            
            # Run baseline episode (same seed for fair comparison)
            env = DriftingEnvironment(drift_rate, seed)
            baseline_agents = [BaselineAgent(f"base_{i}", seed + i) for i in range(config.n_agents)]
            baseline_rate = run_episode(env, baseline_agents, is_gcl=False, steps=config.steps_per_episode)
            baseline_successes.append(baseline_rate)
            
        # Compute statistics
        gcl_mean = np.mean(gcl_successes)
        gcl_std = np.std(gcl_successes)
        baseline_mean = np.mean(baseline_successes)
        baseline_std = np.std(baseline_successes)
        
        # Welch's t-test for significance
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(gcl_successes, baseline_successes, equal_var=False)
        
        result = DriftResult(
            drift_rate=drift_rate,
            gcl_success_rate=gcl_mean,
            gcl_std=gcl_std,
            baseline_success_rate=baseline_mean,
            baseline_std=baseline_std,
            gcl_advantage=gcl_mean - baseline_mean,
            p_value=p_value,
        )
        results.append(result)
        
        print(f"  GCL: {gcl_mean:.3f} ± {gcl_std:.3f}")
        print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
        print(f"  Advantage: {result.gcl_advantage:+.3f} (p={p_value:.4f})")
        
    return results


def find_crossover_point(results: List[DriftResult]) -> Tuple[float, float]:
    """
    Find the drift rate where GCL becomes superior.
    
    Returns:
        (crossover_rate, confidence_interval)
    """
    # Find first rate where GCL is significantly better
    for i, result in enumerate(results):
        if result.gcl_superior:
            if i == 0:
                return result.drift_rate, 0.0
            else:
                # Interpolate between this and previous rate
                prev = results[i - 1]
                # Linear interpolation to find zero crossing
                x1, y1 = prev.drift_rate, prev.gcl_advantage
                x2, y2 = result.drift_rate, result.gcl_advantage
                
                if y2 - y1 != 0:
                    crossover = x1 - y1 * (x2 - x1) / (y2 - y1)
                else:
                    crossover = (x1 + x2) / 2
                    
                return crossover, (x2 - x1) / 2
                
    # No crossover found
    return float('inf'), 0.0


def plot_results(results: List[DriftResult], output_path: Path) -> None:
    """Generate visualization of drift threshold experiment."""
    try:
        import matplotlib.pyplot as plt
        
        drift_rates = [r.drift_rate for r in results]
        gcl_rates = [r.gcl_success_rate for r in results]
        gcl_stds = [r.gcl_std for r in results]
        baseline_rates = [r.baseline_success_rate for r in results]
        baseline_stds = [r.baseline_std for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Success rates
        ax1.errorbar(drift_rates, gcl_rates, yerr=gcl_stds, 
                    label='GCL', marker='o', capsize=3, color='blue')
        ax1.errorbar(drift_rates, baseline_rates, yerr=baseline_stds,
                    label='Natural Language', marker='s', capsize=3, color='orange')
        
        # Find and mark crossover
        crossover, ci = find_crossover_point(results)
        if crossover < float('inf'):
            ax1.axvline(x=crossover, color='red', linestyle='--', 
                       label=f'Crossover ε* ≈ {crossover:.2f}')
            ax1.axvspan(crossover - ci, crossover + ci, alpha=0.2, color='red')
            
        ax1.set_xlabel('Drift Rate (ε)', fontsize=12)
        ax1.set_ylabel('Coordination Success Rate', fontsize=12)
        ax1.set_title('GCL vs Natural Language Under Semantic Drift', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: GCL advantage
        advantages = [r.gcl_advantage for r in results]
        significant = [r.gcl_superior for r in results]
        colors = ['green' if s else 'gray' for s in significant]
        
        ax2.bar(drift_rates, advantages, color=colors, width=0.04, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        if crossover < float('inf'):
            ax2.axvline(x=crossover, color='red', linestyle='--')
            
        ax2.set_xlabel('Drift Rate (ε)', fontsize=12)
        ax2.set_ylabel('GCL Advantage (Δ success rate)', fontsize=12)
        ax2.set_title('GCL Advantage by Drift Rate\n(green = statistically significant)', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlot saved to: {output_path}")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot generation")


def main():
    """Run the drift threshold experiment."""
    print("=" * 60)
    print("EXPERIMENT 09: DRIFT THRESHOLD DISCOVERY")
    print("=" * 60)
    print("\nHypothesis: GCL becomes superior at drift rate ε* ≈ 0.2")
    print("This is the 'empirical surprise' - a falsifiable prediction.\n")
    
    # Run experiment
    config = DriftExperimentConfig.default()
    print(f"Configuration:")
    print(f"  Drift rates: {config.drift_rates}")
    print(f"  Episodes per rate: {config.episodes_per_rate}")
    print(f"  Steps per episode: {config.steps_per_episode}")
    print(f"  Agents: {config.n_agents}")
    
    results = run_drift_experiment(config)
    
    # Find crossover point
    crossover, ci = find_crossover_point(results)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nDrift Rate | GCL Success | Baseline | Advantage | Significant")
    print("-" * 65)
    for r in results:
        sig = "YES" if r.gcl_superior else "no"
        print(f"   {r.drift_rate:.2f}    |   {r.gcl_success_rate:.3f}    |  {r.baseline_success_rate:.3f}   |  {r.gcl_advantage:+.3f}   |    {sig}")
    
    print("\n" + "=" * 60)
    print("CROSSOVER ANALYSIS")
    print("=" * 60)
    
    if crossover < float('inf'):
        print(f"\n✓ CROSSOVER FOUND: ε* = {crossover:.3f} ± {ci:.3f}")
        print(f"\nInterpretation:")
        print(f"  - Below ε* = {crossover:.2f}: Natural language coordination is adequate")
        print(f"  - Above ε* = {crossover:.2f}: GCL's grounded verification becomes essential")
        
        # Compare to hypothesis
        if 0.15 <= crossover <= 0.25:
            print(f"\n✓ HYPOTHESIS CONFIRMED: Crossover at ε* ≈ 0.2 (actual: {crossover:.3f})")
        else:
            print(f"\n✗ HYPOTHESIS REFINED: Crossover at ε* = {crossover:.3f} (predicted: 0.2)")
    else:
        print("\n✗ NO CROSSOVER FOUND: GCL not significantly better at any drift rate")
        print("   This would falsify the drift threshold hypothesis.")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results_data = {
        "config": {
            "drift_rates": config.drift_rates,
            "episodes_per_rate": config.episodes_per_rate,
            "steps_per_episode": config.steps_per_episode,
            "n_agents": config.n_agents,
            "seed": config.seed,
        },
        "results": [
            {
                "drift_rate": r.drift_rate,
                "gcl_success_rate": r.gcl_success_rate,
                "gcl_std": r.gcl_std,
                "baseline_success_rate": r.baseline_success_rate,
                "baseline_std": r.baseline_std,
                "gcl_advantage": r.gcl_advantage,
                "p_value": float(r.p_value),
                "gcl_superior": bool(r.gcl_superior),
            }
            for r in results
        ],
        "crossover": {
            "rate": crossover if crossover < float('inf') else None,
            "confidence_interval": ci,
        },
        "hypothesis_confirmed": bool(0.15 <= crossover <= 0.25) if crossover < float('inf') else False,
    }
    
    with open(output_dir / "09_drift_threshold.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_dir / '09_drift_threshold.json'}")
    
    # Generate plot
    plot_results(results, output_dir / "09_drift_threshold.png")
    
    return results


if __name__ == "__main__":
    main()
