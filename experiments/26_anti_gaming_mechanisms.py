"""
Experiment 26: Anti-Gaming Mechanisms for Reputation Awareness

Problem: Experiment 25 showed that strategic agents game the system by:
1. Refusing hard tasks (50% refusal rate)
2. Selecting easy tasks (96.6% easy task ratio)
3. Over-cautious behavior reducing cooperation

Goal: Design mechanisms that allow reputation awareness while preventing gaming.

Anti-Gaming Mechanisms to Test:
1. Difficulty-weighted reputation: Harder tasks give more reputation
2. Refusal penalty: Refusing tasks costs reputation
3. Minimum challenge requirement: Must attempt some hard tasks
4. Reputation decay: Reputation decays if not challenged
5. Peer assignment: Others assign your tasks (no cherry-picking)
6. Combined: Multiple mechanisms together
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
    """A commitment to complete a task."""
    task: Task
    stake: float
    agent_id: str


@dataclass
class AntiGamingAgent:
    """
    Agent with reputation awareness AND anti-gaming mechanisms.
    """
    
    id: str
    base_capability: float
    awareness_level: str = "strategic"  # All agents are strategic
    anti_gaming: str = "none"  # The mechanism being tested
    
    # State
    reputation: float = 0.5
    resources: float = 1.0
    
    # History
    success_history: List[bool] = field(default_factory=list)
    failure_history: List[bool] = field(default_factory=list)
    tasks_attempted: int = 0
    tasks_refused: int = 0
    easy_tasks_selected: int = 0
    hard_tasks_selected: int = 0
    
    # Anti-gaming tracking
    consecutive_easy: int = 0
    rounds_since_hard: int = 0
    
    # Tracking
    recently_failed: bool = False
    
    def get_own_reputation(self) -> float:
        return self.reputation
    
    def propose_commitment(self, task: Task, context: Dict[str, Any] = None) -> Optional[Commitment]:
        """
        Decide whether to commit to a task.
        Anti-gaming mechanisms modify the decision logic.
        """
        context = context or {}
        base_stake = 0.1
        my_reputation = self.get_own_reputation()
        
        # Check if task is assigned (peer assignment mechanism)
        if context.get("assigned_task"):
            # Must take assigned task
            return Commitment(task=task, stake=base_stake, agent_id=self.id)
        
        # Anti-gaming: Refusal penalty
        if self.anti_gaming == "refusal_penalty":
            # Refusing costs reputation, so be less picky
            if task.difficulty > 0.7 and my_reputation < 0.3:
                # Still refuse very hard tasks when struggling
                self.tasks_refused += 1
                return None
            # Otherwise take the task
            return Commitment(task=task, stake=base_stake, agent_id=self.id)
        
        # Anti-gaming: Minimum challenge requirement
        if self.anti_gaming == "min_challenge":
            # Must attempt hard task every N rounds
            if self.rounds_since_hard >= 5 and task.difficulty > 0.5:
                # Forced to take a challenging task
                self.hard_tasks_selected += 1
                return Commitment(task=task, stake=base_stake, agent_id=self.id)
        
        # Anti-gaming: Reputation decay
        if self.anti_gaming == "decay":
            # If only doing easy tasks, reputation decays
            # So need to take some hard tasks to maintain
            if self.consecutive_easy >= 3 and task.difficulty > 0.5:
                self.hard_tasks_selected += 1
                return Commitment(task=task, stake=base_stake, agent_id=self.id)
        
        # Default strategic behavior (with modifications)
        if my_reputation < 0.3:
            if task.difficulty > 0.5:
                self.tasks_refused += 1
                return None
            self.easy_tasks_selected += 1
        elif task.difficulty > 0.6:
            self.hard_tasks_selected += 1
        else:
            self.easy_tasks_selected += 1
        
        return Commitment(task=task, stake=base_stake, agent_id=self.id)
    
    def attempt_task(self, task: Task) -> bool:
        """Attempt a task."""
        self.tasks_attempted += 1
        
        # Track difficulty
        if task.difficulty > 0.5:
            self.rounds_since_hard = 0
            self.consecutive_easy = 0
        else:
            self.rounds_since_hard += 1
            self.consecutive_easy += 1
        
        # Success probability
        success_prob = self.base_capability * (1 - task.difficulty * 0.8)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        
        if success:
            self.success_history.append(True)
            self.recently_failed = False
        else:
            self.failure_history.append(True)
            self.recently_failed = True
        
        return success
    
    def update_reputation(self, success: bool, task: Task, anti_gaming: str):
        """Update reputation with anti-gaming adjustments."""
        base_change = 0.05
        
        # Mechanism: Difficulty-weighted reputation
        if anti_gaming == "difficulty_weighted":
            # Harder tasks give more reputation
            difficulty_multiplier = 1.0 + task.difficulty * 2.0  # 1x to 3x
            if success:
                self.reputation += base_change * difficulty_multiplier
            else:
                # Failure on hard tasks is less punishing
                self.reputation -= base_change * (1.0 / difficulty_multiplier)
        
        # Mechanism: Refusal penalty (applied elsewhere)
        elif anti_gaming == "refusal_penalty":
            if success:
                self.reputation += base_change
            else:
                self.reputation -= base_change
        
        # Mechanism: Reputation decay
        elif anti_gaming == "decay":
            if success:
                self.reputation += base_change
            else:
                self.reputation -= base_change
            
            # Decay if only doing easy tasks
            if self.consecutive_easy >= 3:
                self.reputation -= 0.02  # Decay penalty
        
        # Default
        else:
            if success:
                self.reputation += base_change
            else:
                self.reputation -= base_change
        
        # Clamp
        self.reputation = max(0.0, min(1.0, self.reputation))
    
    def apply_refusal_penalty(self):
        """Apply penalty for refusing a task."""
        self.reputation -= 0.02
        self.reputation = max(0.0, self.reputation)


def create_population(n_agents: int, anti_gaming: str) -> List[AntiGamingAgent]:
    """Create a population with specified anti-gaming mechanism."""
    agents = []
    for i in range(n_agents):
        cap = random.gauss(0.5, 0.15)
        cap = max(0.1, min(0.9, cap))
        agents.append(AntiGamingAgent(
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
    """Run simulation with specified anti-gaming mechanism."""
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    agents = create_population(n_agents, anti_gaming)
    history = []
    
    for round_num in range(n_rounds):
        round_successes = 0
        round_refusals = 0
        round_commitments = 0
        
        # Peer assignment: shuffle and assign tasks
        if anti_gaming == "peer_assignment":
            # Generate tasks first
            tasks = [Task(
                id=f"task_{round_num}_{i}",
                difficulty=random.uniform(0.2, 0.8),
                reward=1.0
            ) for i in range(n_agents)]
            
            # Shuffle and assign
            random.shuffle(tasks)
            for agent, task in zip(agents, tasks):
                context = {"assigned_task": True}
                commitment = agent.propose_commitment(task, context)
                
                if commitment:
                    round_commitments += 1
                    success = agent.attempt_task(task)
                    if success:
                        round_successes += 1
                    agent.update_reputation(success, task, anti_gaming)
        else:
            # Normal: each agent gets a task opportunity
            for agent in agents:
                difficulty = random.uniform(0.2, 0.8)
                task = Task(
                    id=f"task_{round_num}_{agent.id}",
                    difficulty=difficulty,
                    reward=1.0 + difficulty
                )
                
                commitment = agent.propose_commitment(task, {})
                
                if commitment is None:
                    round_refusals += 1
                    # Apply refusal penalty if mechanism is active
                    if anti_gaming == "refusal_penalty":
                        agent.apply_refusal_penalty()
                    continue
                
                round_commitments += 1
                success = agent.attempt_task(task)
                
                if success:
                    round_successes += 1
                
                agent.update_reputation(success, task, anti_gaming)
        
        # Collect metrics
        reputations = [a.reputation for a in agents]
        
        history.append({
            "round": round_num,
            "cooperation_rate": round_successes / n_agents if n_agents > 0 else 0,
            "commitment_rate": round_commitments / n_agents if n_agents > 0 else 0,
            "refusal_rate": round_refusals / n_agents if n_agents > 0 else 0,
            "mean_reputation": np.mean(reputations),
            "gini": calculate_gini(reputations),
        })
    
    # Final metrics
    final_reputations = [a.reputation for a in agents]
    total_easy = sum(a.easy_tasks_selected for a in agents)
    total_hard = sum(a.hard_tasks_selected for a in agents)
    easy_ratio = total_easy / (total_easy + total_hard) if (total_easy + total_hard) > 0 else 0
    total_refused = sum(a.tasks_refused for a in agents)
    
    return {
        "anti_gaming": anti_gaming,
        "history": history,
        "final_metrics": {
            "cooperation_rate": np.mean([h["cooperation_rate"] for h in history[-20:]]),
            "commitment_rate": np.mean([h["commitment_rate"] for h in history[-20:]]),
            "refusal_rate": np.mean([h["refusal_rate"] for h in history[-20:]]),
            "mean_reputation": np.mean(final_reputations),
            "gini": calculate_gini(final_reputations),
            "easy_task_ratio": easy_ratio,
            "total_refusals": total_refused,
        },
    }


def run_full_experiment(n_runs: int = 10, n_agents: int = 50, n_rounds: int = 200):
    """Run experiment across all anti-gaming mechanisms."""
    
    mechanisms = [
        "none",  # Baseline (strategic with no anti-gaming)
        "difficulty_weighted",  # Harder tasks = more reputation
        "refusal_penalty",  # Refusing costs reputation
        "min_challenge",  # Must attempt hard tasks periodically
        "decay",  # Reputation decays without challenge
        "peer_assignment",  # Others assign your tasks
    ]
    
    print("=" * 70)
    print("EXPERIMENT 26: ANTI-GAMING MECHANISMS")
    print("=" * 70)
    print(f"Mechanisms: {mechanisms}")
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
            "gini": {
                "mean": np.mean([r["gini"] for r in results]),
                "std": np.std([r["gini"] for r in results]),
            },
            "easy_task_ratio": {
                "mean": np.mean([r["easy_task_ratio"] for r in results]),
                "std": np.std([r["easy_task_ratio"] for r in results]),
            },
            "refusal_rate": {
                "mean": np.mean([r["refusal_rate"] for r in results]),
                "std": np.std([r["refusal_rate"] for r in results]),
            },
        }
    
    # Compare each mechanism to baseline (none)
    baseline_coop = [r["cooperation_rate"] for r in all_results["none"]]
    baseline_easy = [r["easy_task_ratio"] for r in all_results["none"]]
    
    for mechanism in all_results.keys():
        if mechanism == "none":
            continue
        
        mech_coop = [r["cooperation_rate"] for r in all_results[mechanism]]
        mech_easy = [r["easy_task_ratio"] for r in all_results[mechanism]]
        
        # Test cooperation improvement
        t_coop, p_coop = stats.ttest_ind(mech_coop, baseline_coop)
        coop_improved = np.mean(mech_coop) > np.mean(baseline_coop) and p_coop < 0.05
        
        # Test gaming reduction
        t_easy, p_easy = stats.ttest_ind(mech_easy, baseline_easy)
        gaming_reduced = np.mean(mech_easy) < np.mean(baseline_easy) and p_easy < 0.05
        
        analysis["comparisons"][mechanism] = {
            "cooperation_improved": coop_improved,
            "cooperation_change": np.mean(mech_coop) - np.mean(baseline_coop),
            "cooperation_p": p_coop,
            "gaming_reduced": gaming_reduced,
            "gaming_change": np.mean(mech_easy) - np.mean(baseline_easy),
            "gaming_p": p_easy,
            "effective": coop_improved or gaming_reduced,
        }
    
    # Find best mechanism
    # Score = cooperation_rate - easy_task_ratio * 0.5 - refusal_rate * 0.3
    def score(mechanism):
        s = analysis["summary"][mechanism]
        return (s["cooperation_rate"]["mean"] 
                - s["easy_task_ratio"]["mean"] * 0.5 
                - s["refusal_rate"]["mean"] * 0.3)
    
    scores = {m: score(m) for m in all_results.keys()}
    analysis["scores"] = scores
    analysis["best_mechanism"] = max(scores, key=scores.get)
    
    return analysis


def print_results(all_results: Dict[str, List[Dict]], analysis: Dict[str, Any]):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("ANTI-GAMING MECHANISMS RESULTS")
    print("=" * 70)
    
    # Summary table
    print("\n--- Summary by Mechanism ---")
    print(f"{'Mechanism':<20} {'Cooperation':>12} {'Easy Ratio':>12} {'Refusals':>10} {'Score':>10}")
    print("-" * 66)
    
    for mechanism in all_results.keys():
        s = analysis["summary"][mechanism]
        score = analysis["scores"][mechanism]
        print(f"{mechanism:<20} {s['cooperation_rate']['mean']:>12.3f} "
              f"{s['easy_task_ratio']['mean']:>12.3f} {s['refusal_rate']['mean']:>10.3f} "
              f"{score:>10.3f}")
    
    # Comparisons
    print("\n--- Comparison to Baseline (none) ---")
    for mechanism, comp in analysis["comparisons"].items():
        status = "✓ EFFECTIVE" if comp["effective"] else "✗ NOT EFFECTIVE"
        print(f"\n{mechanism}: {status}")
        print(f"  Cooperation: {comp['cooperation_change']:+.3f} (p={comp['cooperation_p']:.4f})")
        print(f"  Gaming: {comp['gaming_change']:+.3f} (p={comp['gaming_p']:.4f})")
    
    # Best mechanism
    print("\n" + "=" * 70)
    print(f"BEST MECHANISM: {analysis['best_mechanism'].upper()}")
    print(f"Score: {analysis['scores'][analysis['best_mechanism']]:.3f}")
    print("=" * 70)
    
    # Recommendations
    print("\n--- Recommendations ---")
    best = analysis["best_mechanism"]
    if best == "none":
        print("No anti-gaming mechanism improved over baseline.")
        print("Consider: Keeping agents blind to reputation may be best.")
    else:
        print(f"Implement {best.upper()} mechanism in base model.")
        s = analysis["summary"][best]
        print(f"Expected cooperation: {s['cooperation_rate']['mean']:.1%}")
        print(f"Expected gaming ratio: {s['easy_task_ratio']['mean']:.1%}")


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
    
    with open("results/experiment_26_anti_gaming.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to results/experiment_26_anti_gaming.json")
