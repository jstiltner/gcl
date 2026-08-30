"""
Experiment 26b: Reputation Awareness WITH Peer Assignment

Key Question: Can we enable social fabric (reputation awareness) while 
preventing gaming (peer assignment)?

This combines:
- Reputation awareness levels from Exp 25 (blind, self, social, strategic)
- Peer assignment mechanism from Exp 26 (best anti-gaming mechanism)

Hypotheses:
H1: With peer assignment, awareness level won't affect gaming (no choice = no gaming)
H2: Social awareness + peer assignment will enable trust-based cooperation
H3: Strategic awareness + peer assignment will NOT cause over-caution
H4: Combined system will outperform both blind and strategic-without-assignment
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
class AwareAgentWithAssignment:
    """
    Agent with reputation awareness AND peer assignment.
    """
    
    id: str
    base_capability: float
    awareness_level: str = "blind"  # blind, self, social, strategic
    
    # State
    reputation: float = 0.5
    resources: float = 1.0
    
    # History
    success_history: List[bool] = field(default_factory=list)
    failure_history: List[bool] = field(default_factory=list)
    tasks_attempted: int = 0
    tasks_completed: int = 0
    
    # Social tracking
    partner_history: Dict[str, List[bool]] = field(default_factory=lambda: defaultdict(list))
    
    def get_own_reputation(self) -> float:
        if self.awareness_level == "blind":
            return 0.5  # No awareness
        return self.reputation
    
    def get_partner_reputation(self, partner_rep: float) -> float:
        if self.awareness_level in ["blind", "self"]:
            return 0.5  # No awareness of others
        return partner_rep
    
    def accept_assigned_task(self, task: Task, context: Dict[str, Any] = None) -> Tuple[bool, float]:
        """
        Decide whether to accept an assigned task and how much effort to invest.
        With peer assignment, agents can't refuse but can vary effort.
        
        Returns: (accept, effort_level)
        """
        context = context or {}
        base_effort = 0.8
        
        my_rep = self.get_own_reputation()
        
        # Blind: Always full effort (no strategic adjustment)
        if self.awareness_level == "blind":
            return True, base_effort
        
        # Self-aware: Adjust effort based on own reputation
        if self.awareness_level == "self":
            if my_rep < 0.3:
                # Low rep: try harder to recover
                return True, min(1.0, base_effort + 0.15)
            elif my_rep > 0.7:
                # High rep: can coast a bit
                return True, base_effort - 0.1
            return True, base_effort
        
        # Social-aware: Consider partner reputation
        if self.awareness_level == "social":
            partner_rep = context.get("partner_reputation", 0.5)
            partner_rep = self.get_partner_reputation(partner_rep)
            
            # Trust high-rep partners more, invest more effort
            if partner_rep > 0.6:
                return True, min(1.0, base_effort + 0.1)
            elif partner_rep < 0.3:
                # Low-rep partner: still try but less invested
                return True, base_effort - 0.1
            return True, base_effort
        
        # Strategic: Full optimization
        if self.awareness_level == "strategic":
            partner_rep = context.get("partner_reputation", 0.5)
            partner_rep = self.get_partner_reputation(partner_rep)
            
            # Strategic calculation
            if my_rep < 0.3:
                # Must recover reputation
                return True, min(1.0, base_effort + 0.2)
            elif my_rep > 0.7 and partner_rep < 0.4:
                # High rep, low-value partner: minimal effort
                return True, base_effort - 0.15
            elif partner_rep > 0.6:
                # Good partner: invest more
                return True, min(1.0, base_effort + 0.1)
            return True, base_effort
        
        return True, base_effort
    
    def attempt_task(self, task: Task, effort: float) -> bool:
        """Attempt a task with given effort level."""
        self.tasks_attempted += 1
        
        # Success probability based on capability, difficulty, and effort
        success_prob = self.base_capability * effort * (1 - task.difficulty * 0.6)
        success_prob = max(0.0, min(1.0, success_prob))
        
        success = random.random() < success_prob
        
        if success:
            self.success_history.append(True)
            self.tasks_completed += 1
        else:
            self.failure_history.append(True)
        
        return success
    
    def update_reputation(self, success: bool, task: Task):
        """Update reputation based on outcome."""
        base_change = 0.05
        
        # Difficulty-weighted (from Exp 26)
        difficulty_multiplier = 1.0 + task.difficulty * 1.5
        
        if success:
            self.reputation += base_change * difficulty_multiplier
        else:
            self.reputation -= base_change * (1.0 / difficulty_multiplier)
        
        self.reputation = max(0.0, min(1.0, self.reputation))


def create_population(n_agents: int, awareness_level: str) -> List[AwareAgentWithAssignment]:
    """Create a population with specified awareness level."""
    agents = []
    for i in range(n_agents):
        cap = random.gauss(0.5, 0.15)
        cap = max(0.1, min(0.9, cap))
        agents.append(AwareAgentWithAssignment(
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
    use_peer_assignment: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run simulation with specified awareness and assignment mechanism."""
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    agents = create_population(n_agents, awareness_level)
    history = []
    
    for round_num in range(n_rounds):
        round_successes = 0
        round_efforts = []
        
        if use_peer_assignment:
            # Generate tasks
            tasks = [Task(
                id=f"task_{round_num}_{i}",
                difficulty=random.uniform(0.2, 0.8),
                reward=1.0
            ) for i in range(n_agents)]
            
            # Shuffle and assign
            random.shuffle(tasks)
            
            # Pair agents for context (social awareness needs partner info)
            agent_pairs = list(range(n_agents))
            random.shuffle(agent_pairs)
            
            for i, (agent, task) in enumerate(zip(agents, tasks)):
                # Get partner for context
                partner_idx = agent_pairs[(i + 1) % n_agents]
                partner = agents[partner_idx]
                
                context = {
                    "partner_reputation": partner.reputation,
                    "assigned": True
                }
                
                accept, effort = agent.accept_assigned_task(task, context)
                round_efforts.append(effort)
                
                if accept:
                    success = agent.attempt_task(task, effort)
                    if success:
                        round_successes += 1
                    agent.update_reputation(success, task)
        else:
            # No peer assignment - agents can refuse (for comparison)
            for agent in agents:
                task = Task(
                    id=f"task_{round_num}_{agent.id}",
                    difficulty=random.uniform(0.2, 0.8),
                    reward=1.0
                )
                
                # Without peer assignment, strategic agents might refuse
                if awareness_level == "strategic" and agent.reputation < 0.3 and task.difficulty > 0.5:
                    continue  # Refuse
                
                accept, effort = agent.accept_assigned_task(task, {})
                round_efforts.append(effort)
                
                if accept:
                    success = agent.attempt_task(task, effort)
                    if success:
                        round_successes += 1
                    agent.update_reputation(success, task)
        
        # Collect metrics
        reputations = [a.reputation for a in agents]
        
        history.append({
            "round": round_num,
            "cooperation_rate": round_successes / n_agents if n_agents > 0 else 0,
            "mean_effort": np.mean(round_efforts) if round_efforts else 0,
            "mean_reputation": np.mean(reputations),
            "gini": calculate_gini(reputations),
        })
    
    # Final metrics
    final_reputations = [a.reputation for a in agents]
    
    return {
        "awareness_level": awareness_level,
        "peer_assignment": use_peer_assignment,
        "history": history,
        "final_metrics": {
            "cooperation_rate": np.mean([h["cooperation_rate"] for h in history[-20:]]),
            "mean_effort": np.mean([h["mean_effort"] for h in history[-20:]]),
            "mean_reputation": np.mean(final_reputations),
            "gini": calculate_gini(final_reputations),
        },
    }


def run_full_experiment(n_runs: int = 10, n_agents: int = 50, n_rounds: int = 200):
    """Run experiment comparing awareness levels with and without peer assignment."""
    
    awareness_levels = ["blind", "self", "social", "strategic"]
    
    print("=" * 70)
    print("EXPERIMENT 26b: AWARENESS WITH PEER ASSIGNMENT")
    print("=" * 70)
    print(f"Awareness levels: {awareness_levels}")
    print(f"Runs: {n_runs}, Agents: {n_agents}, Rounds: {n_rounds}")
    print()
    
    all_results = {}
    
    # With peer assignment
    print("--- With Peer Assignment ---")
    for level in awareness_levels:
        key = f"{level}_assigned"
        all_results[key] = []
        print(f"Running {level} (assigned)...", end=" ")
        for run in range(n_runs):
            result = run_simulation(
                n_agents=n_agents,
                n_rounds=n_rounds,
                awareness_level=level,
                use_peer_assignment=True,
                seed=run * 1000 + hash(level) % 1000
            )
            all_results[key].append(result["final_metrics"])
        print("done")
    
    # Without peer assignment (for comparison)
    print("\n--- Without Peer Assignment ---")
    for level in awareness_levels:
        key = f"{level}_choice"
        all_results[key] = []
        print(f"Running {level} (choice)...", end=" ")
        for run in range(n_runs):
            result = run_simulation(
                n_agents=n_agents,
                n_rounds=n_rounds,
                awareness_level=level,
                use_peer_assignment=False,
                seed=run * 1000 + hash(level) % 1000
            )
            all_results[key].append(result["final_metrics"])
        print("done")
    
    return all_results


def analyze_results(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze results and test hypotheses."""
    
    analysis = {"summary": {}, "hypotheses": {}}
    
    # Summary statistics
    for condition, results in all_results.items():
        analysis["summary"][condition] = {
            "cooperation_rate": {
                "mean": np.mean([r["cooperation_rate"] for r in results]),
                "std": np.std([r["cooperation_rate"] for r in results]),
            },
            "mean_effort": {
                "mean": np.mean([r["mean_effort"] for r in results]),
                "std": np.std([r["mean_effort"] for r in results]),
            },
            "gini": {
                "mean": np.mean([r["gini"] for r in results]),
                "std": np.std([r["gini"] for r in results]),
            },
        }
    
    # Test hypotheses
    
    # H1: With peer assignment, awareness level won't affect gaming
    # (All assigned conditions should have similar cooperation)
    assigned_coops = {
        level: [r["cooperation_rate"] for r in all_results[f"{level}_assigned"]]
        for level in ["blind", "self", "social", "strategic"]
    }
    
    # ANOVA test
    f_stat, p_anova = stats.f_oneway(*assigned_coops.values())
    h1_passed = p_anova > 0.05  # No significant difference = hypothesis passes
    
    analysis["hypotheses"]["H1_no_gaming_difference"] = {
        "description": "With peer assignment, awareness level doesn't affect gaming",
        "f_statistic": f_stat,
        "p_value": p_anova,
        "passed": h1_passed,
        "interpretation": "No significant difference in cooperation across awareness levels" if h1_passed else "Awareness still affects cooperation even with peer assignment"
    }
    
    # H2: Social awareness + peer assignment enables trust-based cooperation
    social_assigned = [r["cooperation_rate"] for r in all_results["social_assigned"]]
    blind_assigned = [r["cooperation_rate"] for r in all_results["blind_assigned"]]
    t_stat, p_h2 = stats.ttest_ind(social_assigned, blind_assigned)
    h2_passed = np.mean(social_assigned) >= np.mean(blind_assigned) - 0.02  # At least as good
    
    analysis["hypotheses"]["H2_social_enables_trust"] = {
        "description": "Social awareness + peer assignment enables trust-based cooperation",
        "social_mean": np.mean(social_assigned),
        "blind_mean": np.mean(blind_assigned),
        "t_statistic": t_stat,
        "p_value": p_h2,
        "passed": h2_passed,
        "interpretation": "Social awareness maintains cooperation with peer assignment" if h2_passed else "Social awareness reduces cooperation even with peer assignment"
    }
    
    # H3: Strategic awareness + peer assignment won't cause over-caution
    strategic_assigned = [r["cooperation_rate"] for r in all_results["strategic_assigned"]]
    strategic_choice = [r["cooperation_rate"] for r in all_results["strategic_choice"]]
    t_stat, p_h3 = stats.ttest_ind(strategic_assigned, strategic_choice)
    h3_passed = np.mean(strategic_assigned) > np.mean(strategic_choice)
    
    analysis["hypotheses"]["H3_no_overcaution"] = {
        "description": "Strategic awareness + peer assignment doesn't cause over-caution",
        "assigned_mean": np.mean(strategic_assigned),
        "choice_mean": np.mean(strategic_choice),
        "t_statistic": t_stat,
        "p_value": p_h3,
        "passed": h3_passed,
        "interpretation": "Peer assignment prevents strategic over-caution" if h3_passed else "Strategic agents still show over-caution with peer assignment"
    }
    
    # H4: Combined system outperforms both blind and strategic-without-assignment
    social_assigned_mean = np.mean(social_assigned)
    blind_choice = [r["cooperation_rate"] for r in all_results["blind_choice"]]
    blind_choice_mean = np.mean(blind_choice)
    strategic_choice_mean = np.mean(strategic_choice)
    
    h4_passed = (social_assigned_mean > blind_choice_mean and 
                 social_assigned_mean > strategic_choice_mean)
    
    analysis["hypotheses"]["H4_combined_best"] = {
        "description": "Social+assignment outperforms blind and strategic-choice",
        "social_assigned": social_assigned_mean,
        "blind_choice": blind_choice_mean,
        "strategic_choice": strategic_choice_mean,
        "passed": h4_passed,
        "interpretation": "Combined system is optimal" if h4_passed else "Combined system doesn't outperform alternatives"
    }
    
    # Overall
    passed = sum(1 for h in analysis["hypotheses"].values() if h["passed"])
    analysis["overall"] = {
        "hypotheses_passed": passed,
        "total_hypotheses": 4,
        "conclusion": "VALIDATED" if passed >= 3 else "PARTIALLY VALIDATED" if passed >= 2 else "NOT VALIDATED"
    }
    
    return analysis


def print_results(all_results: Dict[str, List[Dict]], analysis: Dict[str, Any]):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 26b RESULTS: AWARENESS WITH PEER ASSIGNMENT")
    print("=" * 70)
    
    # Summary table
    print("\n--- Summary by Condition ---")
    print(f"{'Condition':<25} {'Cooperation':>12} {'Effort':>10} {'Gini':>10}")
    print("-" * 60)
    
    for condition in sorted(all_results.keys()):
        s = analysis["summary"][condition]
        print(f"{condition:<25} {s['cooperation_rate']['mean']:>12.3f} "
              f"{s['mean_effort']['mean']:>10.3f} {s['gini']['mean']:>10.3f}")
    
    # Hypotheses
    print("\n--- Hypothesis Tests ---")
    for h_name, h_result in analysis["hypotheses"].items():
        status = "✓ PASSED" if h_result["passed"] else "✗ FAILED"
        print(f"\n{h_name}: {status}")
        print(f"  {h_result['description']}")
        print(f"  {h_result['interpretation']}")
    
    # Overall
    print("\n" + "=" * 70)
    overall = analysis["overall"]
    print(f"OVERALL: {overall['conclusion']}")
    print(f"Hypotheses passed: {overall['hypotheses_passed']}/{overall['total_hypotheses']}")
    print("=" * 70)
    
    # Key insight
    print("\n--- Key Insight ---")
    social_assigned = analysis["summary"]["social_assigned"]["cooperation_rate"]["mean"]
    strategic_choice = analysis["summary"]["strategic_choice"]["cooperation_rate"]["mean"]
    blind_choice = analysis["summary"]["blind_choice"]["cooperation_rate"]["mean"]
    
    print(f"Social + Peer Assignment: {social_assigned:.1%} cooperation")
    print(f"Strategic + Choice:       {strategic_choice:.1%} cooperation")
    print(f"Blind + Choice:           {blind_choice:.1%} cooperation")
    
    if social_assigned > max(strategic_choice, blind_choice):
        print("\n✓ RECOMMENDATION: Use SOCIAL awareness with PEER ASSIGNMENT")
        print("  This enables social fabric while preventing gaming!")
    else:
        print("\n⚠ RECOMMENDATION: Further investigation needed")


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
            "hypotheses": analysis["hypotheses"],
            "overall": analysis["overall"],
        }
    }
    
    with open("results/experiment_26b_awareness_with_assignment.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to results/experiment_26b_awareness_with_assignment.json")
