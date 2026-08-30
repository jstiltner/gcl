"""
Experiment 34A: Self-Selection Mechanism Decomposition

Research Question: WHY does self-selection work better than capability-matching?

Hypotheses:
1. Information advantage: Agents know their own capability perfectly
2. Motivation: Volunteers try harder than assigned agents
3. Calibration: Self-selection is self-correcting over time
4. Risk management: Agents avoid tasks they might fail

Design: 2x2x2x2 factorial
- Information quality: perfect (1.0) vs noisy (0.5)
- Effort adjustment: enabled vs disabled
- Learning rate: fast (0.1) vs slow (0.01)
- Risk aversion: none (0.0) vs moderate (0.5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from itertools import product

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, create_population


@dataclass
class MechanismConfig:
    """Configuration for mechanism decomposition."""
    information_quality: float = 1.0  # 0-1, how well agents know own capability
    effort_adjustment: bool = True    # Can agents adjust effort based on selection?
    learning_rate: float = 0.1        # How fast agents calibrate
    risk_aversion: float = 0.0        # How much agents avoid risky tasks
    
    @property
    def name(self) -> str:
        parts = []
        parts.append(f"info{int(self.information_quality*100)}")
        parts.append("effort" if self.effort_adjustment else "noeffort")
        parts.append(f"learn{int(self.learning_rate*100)}")
        parts.append(f"risk{int(self.risk_aversion*100)}")
        return "_".join(parts)


class MechanismAgent:
    """Agent with configurable self-selection mechanisms."""
    
    def __init__(self, agent_id: str, base_capability: float, config: MechanismConfig):
        self.id = agent_id
        self.true_capability = base_capability
        self.config = config
        
        # Perceived capability (may differ from true if info is noisy)
        self.perceived_capability = self._perceive_capability()
        
        # Calibration state
        self.calibration_error = 0.0  # Difference between perceived and true
        self.success_history = []
        self.failure_history = []
        
        # Reputation
        self.reputation = 0.5
    
    def _perceive_capability(self) -> float:
        """Get perceived capability based on information quality."""
        if self.config.information_quality >= 1.0:
            return self.true_capability
        
        # Add noise inversely proportional to information quality
        noise_std = 0.2 * (1 - self.config.information_quality)
        noise = random.gauss(0, noise_std)
        perceived = self.true_capability + noise
        return max(0.1, min(0.9, perceived))
    
    def update_calibration(self, success: bool, task_difficulty: float):
        """Update perceived capability based on outcome."""
        if self.config.learning_rate <= 0:
            return
        
        # Expected success probability
        expected_prob = self.perceived_capability * 0.8 * (1 - task_difficulty * 0.6)
        
        # Actual outcome
        actual = 1.0 if success else 0.0
        
        # Prediction error
        error = actual - expected_prob
        
        # Update perceived capability
        adjustment = error * self.config.learning_rate
        self.perceived_capability += adjustment
        self.perceived_capability = max(0.1, min(0.9, self.perceived_capability))
        
        # Track calibration error
        self.calibration_error = abs(self.perceived_capability - self.true_capability)
    
    def calculate_volunteer_score(self, task: Task) -> float:
        """Calculate willingness to volunteer for a task."""
        # Base score from perceived capability
        base_score = self.perceived_capability - task.difficulty * 0.5
        
        # Risk adjustment
        if self.config.risk_aversion > 0:
            # Estimate failure probability
            success_prob = self.perceived_capability * 0.8 * (1 - task.difficulty * 0.6)
            failure_risk = 1 - success_prob
            risk_penalty = failure_risk * self.config.risk_aversion
            base_score -= risk_penalty
        
        return base_score
    
    def calculate_effort(self, volunteered: bool) -> float:
        """Calculate effort level."""
        base_effort = 0.8
        
        if not self.config.effort_adjustment:
            return base_effort
        
        if volunteered:
            # Volunteers try harder
            return min(1.0, base_effort + 0.1)
        else:
            # Assigned agents try less hard
            return base_effort - 0.1
    
    def attempt_task(self, task: Task, volunteered: bool) -> TaskOutcome:
        """Attempt a task."""
        effort = self.calculate_effort(volunteered)
        
        # Success probability based on TRUE capability (not perceived)
        success_prob = self.true_capability * effort * (1 - task.difficulty * 0.6)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        
        # Update calibration
        self.update_calibration(success, task.difficulty)
        
        # Update history
        if success:
            self.success_history.append(True)
            output_quality = self.true_capability * effort
        else:
            self.failure_history.append(True)
            output_quality = 0.0
        
        return TaskOutcome(
            success=success,
            agent=None,  # Not using base Agent
            task=task,
            output_quality=output_quality
        )


def run_condition(
    config: MechanismConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create agents
    agents = []
    for i in range(n_agents):
        base_cap = random.gauss(0.5, 0.15)
        base_cap = max(0.1, min(0.9, base_cap))
        agents.append(MechanismAgent(f"agent_{i}", base_cap, config))
    
    # Track metrics
    volunteer_successes = 0
    volunteer_attempts = 0
    assigned_successes = 0
    assigned_attempts = 0
    calibration_errors = []
    
    for round_num in range(n_rounds):
        # Generate task
        difficulty = random.uniform(0.3, 0.7)
        task = Task(id=f"task_{round_num}", difficulty=difficulty, reward=1.0)
        
        # Self-selection: agents volunteer based on score
        volunteers = []
        for agent in agents:
            score = agent.calculate_volunteer_score(task)
            if score > 0:
                volunteers.append((agent, score))
        
        if volunteers:
            # Pick best volunteer
            best_volunteer = max(volunteers, key=lambda x: x[1])[0]
            outcome = best_volunteer.attempt_task(task, volunteered=True)
            volunteer_attempts += 1
            if outcome.success:
                volunteer_successes += 1
        else:
            # Random assignment
            assigned = random.choice(agents)
            outcome = assigned.attempt_task(task, volunteered=False)
            assigned_attempts += 1
            if outcome.success:
                assigned_successes += 1
        
        # Track calibration
        for agent in agents:
            calibration_errors.append(agent.calibration_error)
    
    # Calculate metrics
    total_attempts = volunteer_attempts + assigned_attempts
    total_successes = volunteer_successes + assigned_successes
    cooperation_rate = total_successes / total_attempts if total_attempts > 0 else 0
    
    volunteer_rate = volunteer_attempts / total_attempts if total_attempts > 0 else 0
    volunteer_success_rate = volunteer_successes / volunteer_attempts if volunteer_attempts > 0 else 0
    assigned_success_rate = assigned_successes / assigned_attempts if assigned_attempts > 0 else 0
    
    mean_calibration_error = np.mean(calibration_errors) if calibration_errors else 0
    
    return {
        "cooperation_rate": cooperation_rate,
        "volunteer_rate": volunteer_rate,
        "volunteer_success_rate": volunteer_success_rate,
        "assigned_success_rate": assigned_success_rate,
        "calibration_error": mean_calibration_error,
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full factorial experiment."""
    print("=" * 70)
    print("EXPERIMENT 34A: SELF-SELECTION MECHANISM DECOMPOSITION")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Factorial design
    info_levels = [1.0, 0.5]
    effort_levels = [True, False]
    learn_levels = [0.1, 0.01]
    risk_levels = [0.0, 0.5]
    
    for info, effort, learn, risk in product(info_levels, effort_levels, learn_levels, risk_levels):
        config = MechanismConfig(
            information_quality=info,
            effort_adjustment=effort,
            learning_rate=learn,
            risk_aversion=risk
        )
        
        print(f"Running {config.name}...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(config, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        results[config.name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"vol_rate={aggregated['volunteer_rate']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze main effects of each factor."""
    analysis = {}
    
    # Extract cooperation rates
    coop_rates = {name: r["cooperation_rate"]["mean"] for name, r in results.items()}
    
    # Calculate main effects
    # Information effect
    info_high = [v for k, v in coop_rates.items() if "info100" in k]
    info_low = [v for k, v in coop_rates.items() if "info50" in k]
    analysis["info_effect"] = np.mean(info_high) - np.mean(info_low)
    
    # Effort effect
    effort_yes = [v for k, v in coop_rates.items() if "_effort_" in k]
    effort_no = [v for k, v in coop_rates.items() if "_noeffort_" in k]
    analysis["effort_effect"] = np.mean(effort_yes) - np.mean(effort_no)
    
    # Learning effect
    learn_fast = [v for k, v in coop_rates.items() if "learn10" in k]
    learn_slow = [v for k, v in coop_rates.items() if "learn1_" in k]
    analysis["learning_effect"] = np.mean(learn_fast) - np.mean(learn_slow)
    
    # Risk effect
    risk_none = [v for k, v in coop_rates.items() if "risk0" in k]
    risk_mod = [v for k, v in coop_rates.items() if "risk50" in k]
    analysis["risk_effect"] = np.mean(risk_mod) - np.mean(risk_none)
    
    # Find best and worst conditions
    analysis["best_condition"] = max(coop_rates, key=coop_rates.get)
    analysis["worst_condition"] = min(coop_rates, key=coop_rates.get)
    analysis["best_coop"] = coop_rates[analysis["best_condition"]]
    analysis["worst_coop"] = coop_rates[analysis["worst_condition"]]
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Self-Selection Mechanism Decomposition")
    print("=" * 70)
    
    # Main effects
    print("\n--- Main Effects on Cooperation ---")
    print(f"Information (perfect vs noisy):  {analysis['info_effect']:+.4f}")
    print(f"Effort (adjustable vs fixed):    {analysis['effort_effect']:+.4f}")
    print(f"Learning (fast vs slow):         {analysis['learning_effect']:+.4f}")
    print(f"Risk aversion (moderate vs none):{analysis['risk_effect']:+.4f}")
    
    # Rank effects
    effects = [
        ("Information", abs(analysis["info_effect"])),
        ("Effort", abs(analysis["effort_effect"])),
        ("Learning", abs(analysis["learning_effect"])),
        ("Risk", abs(analysis["risk_effect"])),
    ]
    effects.sort(key=lambda x: -x[1])
    
    print("\n--- Effect Ranking (by magnitude) ---")
    for i, (name, effect) in enumerate(effects, 1):
        print(f"{i}. {name}: {effect:.4f}")
    
    # Best/worst
    print(f"\n--- Best Condition ---")
    print(f"{analysis['best_condition']}: {analysis['best_coop']:.3f}")
    print(f"\n--- Worst Condition ---")
    print(f"{analysis['worst_condition']}: {analysis['worst_coop']:.3f}")
    
    # Key finding
    print("\n--- KEY FINDING ---")
    top_effect = effects[0]
    print(f"The most important mechanism is: {top_effect[0]} (effect = {top_effect[1]:.4f})")


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
    with open("results/experiment_34a_mechanism.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_34a_mechanism.json")


if __name__ == "__main__":
    main()
