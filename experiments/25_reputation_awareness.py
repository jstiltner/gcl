"""
Experiment 25: Reputation Awareness

Question: What happens when agents are aware of the reputation system 
and can factor it into decisions?

Conditions:
1. Blind (baseline): Agents don't see reputations
2. Self-aware: Agents see own reputation only
3. Socially-aware: Agents see own + others' reputations
4. Strategic: Agents see reputations AND know reputation affects task allocation

Hypotheses:
H1: Awareness increases cooperation (agents protect reputation)
H2: Awareness increases inequality (rich get richer)
H3: Strategic awareness increases gaming (metric optimization)
H4: Self-awareness without social visibility is optimal (effort without gaming)
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
    stake: float  # How much reputation is at risk
    agent_id: str


@dataclass
class ReputationAwareAgent:
    """
    Agent with configurable reputation awareness.
    
    Awareness levels:
    - blind: No visibility into reputation system
    - self: Can see own reputation only
    - social: Can see own + others' reputations
    - strategic: Full visibility + knows reputation affects allocation
    """
    
    id: str
    base_capability: float  # 0-1, innate ability
    awareness_level: str = "blind"  # blind, self, social, strategic
    
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
    redemption_attempts: int = 0
    
    # Tracking
    recently_failed: bool = False
    stake_adjustments: List[float] = field(default_factory=list)
    
    def get_own_reputation(self) -> float:
        """Get own reputation (if aware)."""
        if self.awareness_level == "blind":
            return 0.5  # Assume average
        return self.reputation
    
    def can_see_others(self) -> bool:
        """Can this agent see others' reputations?"""
        return self.awareness_level in ["social", "strategic"]
    
    def is_strategic(self) -> bool:
        """Does agent know reputation affects allocation?"""
        return self.awareness_level == "strategic"
    
    def propose_commitment(self, task: Task, context: Dict[str, Any] = None) -> Optional[Commitment]:
        """
        Decide whether to commit to a task and with what stake.
        
        Returns None if refusing the task.
        """
        context = context or {}
        
        # Base stake
        base_stake = 0.1
        
        # Blind agents just use base capability
        if self.awareness_level == "blind":
            # Simple capability-based decision
            if task.difficulty > self.base_capability + 0.2:
                return None  # Too hard
            return Commitment(task=task, stake=base_stake, agent_id=self.id)
        
        # Self-aware: factor in own reputation
        my_reputation = self.get_own_reputation()
        
        if my_reputation < 0.3:
            # Low reputation: be conservative, protect what's left
            if task.difficulty > 0.5:
                self.tasks_refused += 1
                return None  # Refuse risky tasks
            # Take easy tasks to rebuild
            base_stake *= 0.8  # Lower stake to protect
        elif my_reputation > 0.8:
            # High reputation: can afford some risk
            base_stake *= 1.2  # Higher stake, more confident
        
        # Socially-aware: factor in partner's reputation
        if self.awareness_level in ["social", "strategic"]:
            partner_reputation = context.get("partner_reputation")
            if partner_reputation is not None and partner_reputation < 0.3:
                # Partner is unreliable, reduce commitment stake
                base_stake *= 0.5
        
        # Strategic: actively manage reputation
        if self.awareness_level == "strategic":
            # Seek redemption opportunities if reputation is low
            if my_reputation < 0.5 and self.recently_failed:
                # Actively seek recovery with higher commitment
                base_stake *= 1.5  # Signal commitment to recovery
                self.redemption_attempts += 1
            
            # Gaming: prefer easy tasks when reputation is at risk
            if my_reputation < 0.4 and task.difficulty < 0.4:
                self.easy_tasks_selected += 1
            elif task.difficulty > 0.6:
                self.hard_tasks_selected += 1
        
        self.stake_adjustments.append(base_stake)
        return Commitment(task=task, stake=base_stake, agent_id=self.id)
    
    def attempt_task(self, task: Task) -> bool:
        """Attempt a task. Returns success/failure."""
        self.tasks_attempted += 1
        
        # Success probability based on capability and difficulty
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
    
    def update_reputation(self, success: bool, stake: float):
        """Update reputation based on outcome."""
        if success:
            self.reputation += stake * 0.5
        else:
            self.reputation -= stake
        
        # Clamp to [0, 1]
        self.reputation = max(0.0, min(1.0, self.reputation))


def create_population(n_agents: int, awareness_level: str, 
                      capability_mean: float = 0.5, 
                      capability_std: float = 0.15) -> List[ReputationAwareAgent]:
    """Create a population of agents with specified awareness level."""
    agents = []
    for i in range(n_agents):
        cap = random.gauss(capability_mean, capability_std)
        cap = max(0.1, min(0.9, cap))
        agents.append(ReputationAwareAgent(
            id=f"agent_{i}",
            base_capability=cap,
            awareness_level=awareness_level
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
    awareness_level: str = "blind",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a simulation with the specified awareness level.
    
    Returns metrics for analysis.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create population
    agents = create_population(n_agents, awareness_level)
    
    # Track metrics over time
    history = []
    
    for round_num in range(n_rounds):
        round_successes = 0
        round_refusals = 0
        round_commitments = 0
        
        # Each agent gets a task opportunity
        for agent in agents:
            # Generate task with varying difficulty
            difficulty = random.uniform(0.2, 0.8)
            task = Task(
                id=f"task_{round_num}_{agent.id}",
                difficulty=difficulty,
                reward=1.0 + difficulty
            )
            
            # Context for social/strategic agents
            context = {}
            if agent.can_see_others():
                # Pick a random "partner" to consider
                other = random.choice([a for a in agents if a.id != agent.id])
                context["partner_reputation"] = other.reputation
            
            # Agent decides whether to commit
            commitment = agent.propose_commitment(task, context)
            
            if commitment is None:
                round_refusals += 1
                continue
            
            round_commitments += 1
            
            # Agent attempts task
            success = agent.attempt_task(task)
            
            if success:
                round_successes += 1
            
            # Update reputation
            agent.update_reputation(success, commitment.stake)
        
        # Collect round metrics
        reputations = [a.reputation for a in agents]
        
        history.append({
            "round": round_num,
            "cooperation_rate": round_successes / n_agents if n_agents > 0 else 0,
            "commitment_rate": round_commitments / n_agents if n_agents > 0 else 0,
            "refusal_rate": round_refusals / n_agents if n_agents > 0 else 0,
            "mean_reputation": np.mean(reputations),
            "std_reputation": np.std(reputations),
            "gini": calculate_gini(reputations),
            "min_reputation": min(reputations),
            "max_reputation": max(reputations),
        })
    
    # Final metrics
    final_reputations = [a.reputation for a in agents]
    
    # Gaming indicators
    total_easy = sum(a.easy_tasks_selected for a in agents)
    total_hard = sum(a.hard_tasks_selected for a in agents)
    easy_ratio = total_easy / (total_easy + total_hard) if (total_easy + total_hard) > 0 else 0
    
    # Redemption seeking
    total_redemption = sum(a.redemption_attempts for a in agents)
    
    # Task refusal
    total_refused = sum(a.tasks_refused for a in agents)
    total_attempted = sum(a.tasks_attempted for a in agents)
    
    return {
        "awareness_level": awareness_level,
        "history": history,
        "final_metrics": {
            "cooperation_rate": np.mean([h["cooperation_rate"] for h in history[-20:]]),
            "commitment_rate": np.mean([h["commitment_rate"] for h in history[-20:]]),
            "refusal_rate": np.mean([h["refusal_rate"] for h in history[-20:]]),
            "mean_reputation": np.mean(final_reputations),
            "std_reputation": np.std(final_reputations),
            "gini": calculate_gini(final_reputations),
            "easy_task_ratio": easy_ratio,
            "redemption_attempts": total_redemption,
            "total_refusals": total_refused,
            "total_attempted": total_attempted,
        },
        "agents": agents,
    }


def run_full_experiment(n_runs: int = 10, n_agents: int = 50, n_rounds: int = 200):
    """
    Run the full experiment across all awareness levels.
    """
    awareness_levels = ["blind", "self", "social", "strategic"]
    
    print("=" * 70)
    print("EXPERIMENT 25: REPUTATION AWARENESS")
    print("=" * 70)
    print(f"Awareness levels: {awareness_levels}")
    print(f"Runs: {n_runs}, Agents: {n_agents}, Rounds: {n_rounds}")
    print()
    
    all_results = {level: [] for level in awareness_levels}
    
    for level in awareness_levels:
        print(f"Running {level}...", end=" ")
        for run in range(n_runs):
            result = run_simulation(
                n_agents=n_agents,
                n_rounds=n_rounds,
                awareness_level=level,
                seed=run * 1000 + hash(level) % 1000
            )
            all_results[level].append(result["final_metrics"])
        print("done")
    
    return all_results


def analyze_results(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze results and test hypotheses."""
    
    analysis = {
        "summary": {},
        "hypotheses": {},
        "statistical_tests": {}
    }
    
    # Summary statistics
    for level, results in all_results.items():
        analysis["summary"][level] = {
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
            "redemption_attempts": {
                "mean": np.mean([r["redemption_attempts"] for r in results]),
                "std": np.std([r["redemption_attempts"] for r in results]),
            },
        }
    
    # H1: Awareness increases cooperation
    blind_coop = [r["cooperation_rate"] for r in all_results["blind"]]
    self_coop = [r["cooperation_rate"] for r in all_results["self"]]
    social_coop = [r["cooperation_rate"] for r in all_results["social"]]
    strategic_coop = [r["cooperation_rate"] for r in all_results["strategic"]]
    
    # Test self vs blind
    t_stat, p_value = stats.ttest_ind(self_coop, blind_coop)
    h1_self_vs_blind = np.mean(self_coop) > np.mean(blind_coop) and p_value < 0.05
    
    analysis["hypotheses"]["H1_awareness_increases_cooperation"] = {
        "passed": h1_self_vs_blind,
        "blind_mean": np.mean(blind_coop),
        "self_mean": np.mean(self_coop),
        "social_mean": np.mean(social_coop),
        "strategic_mean": np.mean(strategic_coop),
        "self_vs_blind_p": p_value,
        "description": "Awareness increases cooperation (agents protect reputation)"
    }
    
    # H2: Awareness increases inequality
    blind_gini = [r["gini"] for r in all_results["blind"]]
    self_gini = [r["gini"] for r in all_results["self"]]
    social_gini = [r["gini"] for r in all_results["social"]]
    strategic_gini = [r["gini"] for r in all_results["strategic"]]
    
    t_stat, p_value = stats.ttest_ind(strategic_gini, blind_gini)
    h2_passed = np.mean(strategic_gini) > np.mean(blind_gini)
    
    analysis["hypotheses"]["H2_awareness_increases_inequality"] = {
        "passed": h2_passed,
        "blind_mean": np.mean(blind_gini),
        "self_mean": np.mean(self_gini),
        "social_mean": np.mean(social_gini),
        "strategic_mean": np.mean(strategic_gini),
        "strategic_vs_blind_p": p_value,
        "description": "Awareness increases inequality (rich get richer)"
    }
    
    # H3: Strategic awareness increases gaming
    blind_easy = [r["easy_task_ratio"] for r in all_results["blind"]]
    strategic_easy = [r["easy_task_ratio"] for r in all_results["strategic"]]
    
    t_stat, p_value = stats.ttest_ind(strategic_easy, blind_easy)
    h3_passed = np.mean(strategic_easy) > np.mean(blind_easy) and p_value < 0.05
    
    analysis["hypotheses"]["H3_strategic_increases_gaming"] = {
        "passed": h3_passed,
        "blind_easy_ratio": np.mean(blind_easy),
        "strategic_easy_ratio": np.mean(strategic_easy),
        "p_value": p_value,
        "description": "Strategic awareness increases gaming (metric optimization)"
    }
    
    # H4: Self-awareness is optimal
    # Optimal = high cooperation + low inequality + low gaming
    def score(level):
        coop = analysis["summary"][level]["cooperation_rate"]["mean"]
        gini = analysis["summary"][level]["gini"]["mean"]
        easy = analysis["summary"][level]["easy_task_ratio"]["mean"]
        # Higher cooperation is better, lower gini is better, lower easy ratio is better
        return coop - gini * 0.5 - easy * 0.3
    
    scores = {level: score(level) for level in all_results.keys()}
    best_level = max(scores, key=scores.get)
    h4_passed = best_level == "self"
    
    analysis["hypotheses"]["H4_self_awareness_optimal"] = {
        "passed": h4_passed,
        "scores": scores,
        "best_level": best_level,
        "description": "Self-awareness without social visibility is optimal"
    }
    
    # Summary
    passed = sum(1 for h in analysis["hypotheses"].values() if h["passed"])
    analysis["summary_stats"] = {
        "hypotheses_passed": passed,
        "hypotheses_total": 4,
    }
    
    return analysis


def print_results(all_results: Dict[str, List[Dict]], analysis: Dict[str, Any]):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("REPUTATION AWARENESS RESULTS")
    print("=" * 70)
    
    # Summary table
    print("\n--- Summary by Awareness Level ---")
    print(f"{'Level':<12} {'Cooperation':>12} {'Gini':>10} {'Easy Ratio':>12} {'Refusals':>10}")
    print("-" * 58)
    
    for level in ["blind", "self", "social", "strategic"]:
        s = analysis["summary"][level]
        print(f"{level:<12} {s['cooperation_rate']['mean']:>12.3f} "
              f"{s['gini']['mean']:>10.3f} {s['easy_task_ratio']['mean']:>12.3f} "
              f"{s['refusal_rate']['mean']:>10.3f}")
    
    # Hypotheses
    print("\n--- Hypothesis Testing ---")
    for h_name, h_data in analysis["hypotheses"].items():
        status = "✓ PASSED" if h_data["passed"] else "✗ FAILED"
        print(f"\n{h_name}: {status}")
        print(f"  {h_data['description']}")
        
        if "blind_mean" in h_data:
            print(f"  Blind: {h_data['blind_mean']:.3f}, Self: {h_data['self_mean']:.3f}, "
                  f"Social: {h_data['social_mean']:.3f}, Strategic: {h_data['strategic_mean']:.3f}")
        if "scores" in h_data:
            print(f"  Scores: {h_data['scores']}")
            print(f"  Best level: {h_data['best_level']}")
    
    # Overall
    print("\n" + "=" * 70)
    summary = analysis["summary_stats"]
    print(f"HYPOTHESES PASSED: {summary['hypotheses_passed']}/{summary['hypotheses_total']}")
    print("=" * 70)
    
    # Recommendations
    print("\n--- Recommendations for Base Model ---")
    
    best = analysis["hypotheses"]["H4_self_awareness_optimal"]["best_level"]
    print(f"\n1. Optimal awareness level: {best.upper()}")
    
    if analysis["hypotheses"]["H1_awareness_increases_cooperation"]["passed"]:
        print("2. Reputation awareness DOES improve cooperation - implement it")
    else:
        print("2. Reputation awareness effect on cooperation is weak")
    
    if analysis["hypotheses"]["H2_awareness_increases_inequality"]["passed"]:
        print("3. WARNING: Awareness increases inequality - consider mitigation")
    else:
        print("3. Inequality effect is manageable")
    
    if analysis["hypotheses"]["H3_strategic_increases_gaming"]["passed"]:
        print("4. WARNING: Strategic awareness enables gaming - limit visibility")
    else:
        print("4. Gaming risk is low")


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
    
    # Convert to serializable format
    serializable_results = {}
    for level, results in all_results.items():
        serializable_results[level] = results
    
    output = {
        "results": serializable_results,
        "analysis": {
            "summary": analysis["summary"],
            "hypotheses": {k: {kk: vv for kk, vv in v.items() if kk != "scores" or isinstance(vv, (int, float, str, bool))} 
                         for k, v in analysis["hypotheses"].items()},
            "summary_stats": analysis["summary_stats"],
        }
    }
    
    # Handle scores dict
    if "H4_self_awareness_optimal" in output["analysis"]["hypotheses"]:
        output["analysis"]["hypotheses"]["H4_self_awareness_optimal"]["scores"] = {
            k: float(v) for k, v in analysis["hypotheses"]["H4_self_awareness_optimal"]["scores"].items()
        }
    
    with open("results/experiment_25_reputation_awareness.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to results/experiment_25_reputation_awareness.json")
