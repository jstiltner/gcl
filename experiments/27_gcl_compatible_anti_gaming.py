r"""
Experiment 27: GCL-Compatible Anti-Gaming Mechanisms

Problem: Experiment 26 found peer assignment prevents gaming, but this
UNDERMINES GCL's core model where agents CHOOSE what to commit to.

From the paper (Section 3.3):
"Agents learn *what commitments to make* via reinforcement learning: π(s) → C"

Peer assignment removes agent choice, which is fundamental to GCL.

GCL-Compatible Anti-Gaming Mechanisms:
1. Difficulty-weighted reputation: Harder tasks = more reputation (preserves choice)
2. Commitment portfolio requirements: Must commit to diverse difficulty mix
3. Reputation decay for easy-only: Reputation decays if only easy tasks
4. Social pressure: Low-difficulty agents get fewer partnership offers
5. Stake scaling: Stakes scale with difficulty (more skin in game for easy)

Key insight: The agent still CHOOSES, but the incentive structure discourages gaming.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from scipy import stats
from collections import defaultdict
import json


@dataclass
class Task:
    """A task with difficulty and reward."""
    id: str
    difficulty: float  # 0-1
    reward: float
    task_type: str = "general"


@dataclass
class Commitment:
    """A GCL commitment - agent CHOOSES to make this."""
    task: Task
    stake: float
    agent_id: str
    confidence: float = 0.8


@dataclass
class GCLAgent:
    """
    Agent that makes GCL commitments with anti-gaming incentives.
    
    Key: Agent CHOOSES what to commit to (preserves GCL model).
    Anti-gaming works through incentive structure, not assignment.
    """
    
    id: str
    base_capability: float
    awareness_level: str = "social"  # Can see reputations
    
    # State
    reputation: float = 0.5
    resources: float = 1.0
    
    # History
    success_history: List[bool] = field(default_factory=list)
    failure_history: List[bool] = field(default_factory=list)
    tasks_attempted: int = 0
    
    # Difficulty tracking (for anti-gaming)
    difficulty_history: List[float] = field(default_factory=list)
    consecutive_easy: int = 0
    
    # Anti-gaming mechanism
    anti_gaming: str = "none"
    
    def get_own_reputation(self) -> float:
        if self.awareness_level == "blind":
            return 0.5
        return self.reputation
    
    def get_partner_reputation(self, partner_rep: float) -> float:
        if self.awareness_level in ["blind", "self"]:
            return 0.5
        return partner_rep
    
    def evaluate_commitment_value(self, task: Task, context: Dict[str, Any] = None) -> float:
        """
        Evaluate the value of making a commitment to this task.
        This is where anti-gaming incentives affect CHOICE.
        """
        context = context or {}
        my_rep = self.get_own_reputation()
        
        # Base value: expected reward
        success_prob = self.base_capability * (1 - task.difficulty * 0.6)
        base_value = task.reward * success_prob
        
        # Anti-gaming adjustments to perceived value
        
        if self.anti_gaming == "difficulty_weighted":
            # Harder tasks are MORE valuable (reputation multiplier)
            difficulty_bonus = 1.0 + task.difficulty * 2.0  # 1x to 3x
            base_value *= difficulty_bonus
        
        elif self.anti_gaming == "portfolio_requirement":
            # If too many easy tasks, hard tasks become more valuable
            if len(self.difficulty_history) >= 5:
                avg_difficulty = np.mean(self.difficulty_history[-5:])
                if avg_difficulty < 0.4 and task.difficulty > 0.5:
                    # Need hard tasks to balance portfolio
                    base_value *= 2.0
        
        elif self.anti_gaming == "reputation_decay":
            # Easy tasks give less value if reputation is decaying
            if self.consecutive_easy >= 3:
                if task.difficulty < 0.4:
                    base_value *= 0.5  # Easy tasks less valuable when decaying
                else:
                    base_value *= 1.5  # Hard tasks more valuable to stop decay
        
        elif self.anti_gaming == "social_pressure":
            # Low-difficulty agents get fewer partnership offers
            avg_difficulty = np.mean(self.difficulty_history) if self.difficulty_history else 0.5
            if avg_difficulty < 0.4:
                # Social penalty for being known as easy-task-only
                base_value *= 0.7
        
        elif self.anti_gaming == "stake_scaling":
            # Stakes scale inversely with difficulty
            # Easy tasks require MORE stake (more to lose)
            if task.difficulty < 0.3:
                base_value *= 0.6  # High stake requirement reduces value
        
        # Strategic awareness: consider reputation impact
        if self.awareness_level == "strategic":
            if my_rep < 0.3 and task.difficulty > 0.5:
                # Low rep: avoid hard tasks (risk averse)
                base_value *= 0.5
            elif my_rep > 0.7 and task.difficulty < 0.3:
                # High rep: easy tasks less valuable (already established)
                base_value *= 0.8
        
        return base_value
    
    def choose_commitment(self, available_tasks: List[Task], context: Dict[str, Any] = None) -> Optional[Commitment]:
        """
        CHOOSE which task to commit to (core GCL behavior).
        Agent evaluates all options and picks the best one.
        """
        if not available_tasks:
            return None
        
        # Evaluate all tasks
        task_values = [(task, self.evaluate_commitment_value(task, context)) 
                       for task in available_tasks]
        
        # Sort by value
        task_values.sort(key=lambda x: x[1], reverse=True)
        
        # Choose best task (with some randomness for exploration)
        if random.random() < 0.1:  # 10% exploration
            chosen_task = random.choice(available_tasks)
        else:
            chosen_task = task_values[0][0]
        
        # Calculate stake based on mechanism
        base_stake = 0.1
        if self.anti_gaming == "stake_scaling":
            # Inverse difficulty scaling: easy = high stake
            stake = base_stake * (2.0 - chosen_task.difficulty)
        else:
            stake = base_stake
        
        return Commitment(
            task=chosen_task,
            stake=stake,
            agent_id=self.id,
            confidence=self.base_capability
        )
    
    def attempt_task(self, task: Task) -> bool:
        """Attempt a committed task."""
        self.tasks_attempted += 1
        self.difficulty_history.append(task.difficulty)
        
        # Track consecutive easy
        if task.difficulty < 0.4:
            self.consecutive_easy += 1
        else:
            self.consecutive_easy = 0
        
        # Success probability
        success_prob = self.base_capability * (1 - task.difficulty * 0.6)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        
        if success:
            self.success_history.append(True)
        else:
            self.failure_history.append(True)
        
        return success
    
    def update_reputation(self, success: bool, task: Task):
        """Update reputation with anti-gaming adjustments."""
        base_change = 0.05
        
        if self.anti_gaming == "difficulty_weighted":
            # Harder tasks give more reputation
            multiplier = 1.0 + task.difficulty * 2.0
            if success:
                self.reputation += base_change * multiplier
            else:
                self.reputation -= base_change * (1.0 / multiplier)
        
        elif self.anti_gaming == "reputation_decay":
            if success:
                self.reputation += base_change
            else:
                self.reputation -= base_change
            
            # Decay for easy-only
            if self.consecutive_easy >= 3:
                self.reputation -= 0.02
        
        else:
            if success:
                self.reputation += base_change
            else:
                self.reputation -= base_change
        
        self.reputation = max(0.0, min(1.0, self.reputation))


def create_population(n_agents: int, anti_gaming: str) -> List[GCLAgent]:
    """Create a population with specified anti-gaming mechanism."""
    agents = []
    for i in range(n_agents):
        cap = random.gauss(0.5, 0.15)
        cap = max(0.1, min(0.9, cap))
        agents.append(GCLAgent(
            id=f"agent_{i}",
            base_capability=cap,
            anti_gaming=anti_gaming
        ))
    return agents


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


def run_simulation(
    n_agents: int = 50,
    n_rounds: int = 200,
    anti_gaming: str = "none",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run simulation with GCL-compatible anti-gaming."""
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    agents = create_population(n_agents, anti_gaming)
    history = []
    
    for round_num in range(n_rounds):
        round_successes = 0
        round_commitments = 0
        round_difficulties = []
        
        for agent in agents:
            # Generate available tasks (agent CHOOSES from these)
            available_tasks = [
                Task(
                    id=f"task_{round_num}_{agent.id}_{i}",
                    difficulty=random.uniform(0.2, 0.8),
                    reward=1.0
                )
                for i in range(3)  # 3 options to choose from
            ]
            
            # Agent CHOOSES commitment (core GCL behavior)
            commitment = agent.choose_commitment(available_tasks, {})
            
            if commitment is None:
                continue
            
            round_commitments += 1
            round_difficulties.append(commitment.task.difficulty)
            
            # Attempt the chosen task
            success = agent.attempt_task(commitment.task)
            
            if success:
                round_successes += 1
            
            agent.update_reputation(success, commitment.task)
        
        # Collect metrics
        reputations = [a.reputation for a in agents]
        
        history.append({
            "round": round_num,
            "cooperation_rate": round_successes / n_agents if n_agents > 0 else 0,
            "commitment_rate": round_commitments / n_agents if n_agents > 0 else 0,
            "mean_difficulty": np.mean(round_difficulties) if round_difficulties else 0,
            "mean_reputation": np.mean(reputations),
            "gini": calculate_gini(reputations),
        })
    
    # Final metrics
    final_reputations = [a.reputation for a in agents]
    all_difficulties = [d for a in agents for d in a.difficulty_history]
    easy_ratio = sum(1 for d in all_difficulties if d < 0.4) / len(all_difficulties) if all_difficulties else 0
    
    return {
        "anti_gaming": anti_gaming,
        "history": history,
        "final_metrics": {
            "cooperation_rate": np.mean([h["cooperation_rate"] for h in history[-20:]]),
            "mean_difficulty": np.mean([h["mean_difficulty"] for h in history[-20:]]),
            "mean_reputation": np.mean(final_reputations),
            "gini": calculate_gini(final_reputations),
            "easy_task_ratio": easy_ratio,
        },
    }


def run_full_experiment(n_runs: int = 10, n_agents: int = 50, n_rounds: int = 200):
    """Run experiment across all GCL-compatible anti-gaming mechanisms."""
    
    mechanisms = [
        "none",  # Baseline (no anti-gaming)
        "difficulty_weighted",  # Harder tasks = more reputation
        "portfolio_requirement",  # Must balance difficulty mix
        "reputation_decay",  # Reputation decays for easy-only
        "social_pressure",  # Low-difficulty agents get fewer offers
        "stake_scaling",  # Easy tasks require more stake
    ]
    
    print("=" * 70)
    print("EXPERIMENT 27: GCL-COMPATIBLE ANTI-GAMING")
    print("=" * 70)
    print("Key: Agents CHOOSE commitments (preserves GCL model)")
    print("Anti-gaming works through incentive structure, not assignment")
    print(f"\nMechanisms: {mechanisms}")
    print(f"Runs: {n_runs}, Agents: {n_agents}, Rounds: {n_rounds}")
    print()
    
    all_results = {m: [] for m in mechanisms}
    
    for mechanism in mechanisms:
        print(f"Running {mechanism}...", end=" ")
        for run in range(n_runs):
            result = run_simulation(
                n_agents=n_agents,
                n_rounds=n_rounds,
                anti_gaming=mechanism,
                seed=run * 1000 + hash(mechanism) % 1000
            )
            all_results[mechanism].append(result["final_metrics"])
        print("done")
    
    return all_results


def analyze_results(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze results and compare mechanisms."""
    
    analysis = {"summary": {}, "comparisons": {}, "best_mechanism": None}
    
    # Summary statistics
    for mechanism, results in all_results.items():
        analysis["summary"][mechanism] = {
            "cooperation_rate": {
                "mean": np.mean([r["cooperation_rate"] for r in results]),
                "std": np.std([r["cooperation_rate"] for r in results]),
            },
            "mean_difficulty": {
                "mean": np.mean([r["mean_difficulty"] for r in results]),
                "std": np.std([r["mean_difficulty"] for r in results]),
            },
            "easy_task_ratio": {
                "mean": np.mean([r["easy_task_ratio"] for r in results]),
                "std": np.std([r["easy_task_ratio"] for r in results]),
            },
            "gini": {
                "mean": np.mean([r["gini"] for r in results]),
                "std": np.std([r["gini"] for r in results]),
            },
        }
    
    # Compare each mechanism to baseline
    baseline_coop = [r["cooperation_rate"] for r in all_results["none"]]
    baseline_easy = [r["easy_task_ratio"] for r in all_results["none"]]
    baseline_diff = [r["mean_difficulty"] for r in all_results["none"]]
    
    for mechanism in all_results.keys():
        if mechanism == "none":
            continue
        
        mech_coop = [r["cooperation_rate"] for r in all_results[mechanism]]
        mech_easy = [r["easy_task_ratio"] for r in all_results[mechanism]]
        mech_diff = [r["mean_difficulty"] for r in all_results[mechanism]]
        
        # Test cooperation
        t_coop, p_coop = stats.ttest_ind(mech_coop, baseline_coop)
        
        # Test gaming reduction (lower easy ratio = less gaming)
        t_easy, p_easy = stats.ttest_ind(mech_easy, baseline_easy)
        gaming_reduced = np.mean(mech_easy) < np.mean(baseline_easy) and p_easy < 0.05
        
        # Test difficulty increase
        t_diff, p_diff = stats.ttest_ind(mech_diff, baseline_diff)
        difficulty_increased = np.mean(mech_diff) > np.mean(baseline_diff) and p_diff < 0.05
        
        analysis["comparisons"][mechanism] = {
            "cooperation_change": np.mean(mech_coop) - np.mean(baseline_coop),
            "cooperation_p": p_coop,
            "gaming_reduced": gaming_reduced,
            "gaming_change": np.mean(mech_easy) - np.mean(baseline_easy),
            "gaming_p": p_easy,
            "difficulty_increased": difficulty_increased,
            "difficulty_change": np.mean(mech_diff) - np.mean(baseline_diff),
            "difficulty_p": p_diff,
            "effective": gaming_reduced or difficulty_increased,
        }
    
    # Find best mechanism
    # Score = cooperation_rate + mean_difficulty - easy_task_ratio
    def score(mechanism):
        s = analysis["summary"][mechanism]
        return (s["cooperation_rate"]["mean"] 
                + s["mean_difficulty"]["mean"] 
                - s["easy_task_ratio"]["mean"])
    
    scores = {m: score(m) for m in all_results.keys()}
    analysis["scores"] = scores
    analysis["best_mechanism"] = max(scores, key=scores.get)
    
    return analysis


def print_results(all_results: Dict[str, List[Dict]], analysis: Dict[str, Any]):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("GCL-COMPATIBLE ANTI-GAMING RESULTS")
    print("=" * 70)
    print("(Agents CHOOSE commitments - preserves GCL model)")
    
    # Summary table
    print("\n--- Summary by Mechanism ---")
    print(f"{'Mechanism':<22} {'Cooperation':>12} {'Mean Diff':>10} {'Easy Ratio':>12} {'Score':>10}")
    print("-" * 70)
    
    for mechanism in all_results.keys():
        s = analysis["summary"][mechanism]
        score = analysis["scores"][mechanism]
        print(f"{mechanism:<22} {s['cooperation_rate']['mean']:>12.3f} "
              f"{s['mean_difficulty']['mean']:>10.3f} {s['easy_task_ratio']['mean']:>12.3f} "
              f"{score:>10.3f}")
    
    # Comparisons
    print("\n--- Comparison to Baseline (none) ---")
    for mechanism, comp in analysis["comparisons"].items():
        status = "✓ EFFECTIVE" if comp["effective"] else "✗ NOT EFFECTIVE"
        print(f"\n{mechanism}: {status}")
        print(f"  Cooperation: {comp['cooperation_change']:+.3f} (p={comp['cooperation_p']:.4f})")
        print(f"  Gaming: {comp['gaming_change']:+.3f} (p={comp['gaming_p']:.4f})")
        print(f"  Difficulty: {comp['difficulty_change']:+.3f} (p={comp['difficulty_p']:.4f})")
    
    # Best mechanism
    print("\n" + "=" * 70)
    print(f"BEST GCL-COMPATIBLE MECHANISM: {analysis['best_mechanism'].upper()}")
    print(f"Score: {analysis['scores'][analysis['best_mechanism']]:.3f}")
    print("=" * 70)
    
    # Key insight
    print("\n--- Key Insight ---")
    print("Unlike peer assignment, these mechanisms PRESERVE agent choice.")
    print("Agents still CHOOSE what to commit to (core GCL behavior).")
    print("Anti-gaming works through incentive structure, not forced assignment.")


if __name__ == "__main__":
    # Run experiment
    all_results = run_full_experiment(n_runs=10, n_agents=50, n_rounds=200)
    
    # Analyze
    analysis = analyze_results(all_results)
    
    # Print results
    print_results(all_results, analysis)
    
    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    
    output = {
        "results": {k: v for k, v in all_results.items()},
        "analysis": {
            "summary": analysis["summary"],
            "comparisons": analysis["comparisons"],
            "scores": {k: float(v) for k, v in analysis["scores"].items()},
            "best_mechanism": analysis["best_mechanism"],
        }
    }
    
    with open("results/experiment_27_gcl_compatible_anti_gaming.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to results/experiment_27_gcl_compatible_anti_gaming.json")
