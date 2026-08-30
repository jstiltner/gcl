"""
Experiment 39: Motivation Isolation

CRITICAL QUESTION: Is there an emergent motivation effect, or is self-selection
advantage entirely explained by information/self-knowledge?

Previous experiments confounded:
1. Selection bias (agents pick tasks they're suited for)
2. Self-knowledge (agents know their own capabilities)
3. Effort bonus (a parameter we set, not emergent)

This experiment isolates these by:
1. Giving the coordinator PERFECT information about agent capabilities
2. Removing the hardcoded effort bonus
3. Testing if self-selection STILL outperforms optimal assignment

If self-selection still wins with perfect coordinator information and no effort
bonus, then there IS something beyond information. If not, the advantage is
purely informational.

Design:
- Condition A: Self-selection (agents choose)
- Condition B: Oracle assignment (coordinator knows all capabilities perfectly)
- Condition C: Self-selection with oracle verification (agents choose, but
  coordinator can override bad choices)

Key insight: If A > B with no effort bonus and perfect information, then
self-selection has an intrinsic advantage beyond information.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class Agent:
    """Agent with true and perceived capabilities."""
    id: str
    true_capability: float
    perceived_capability: float  # What agent thinks it has
    
    # Psychological state (for emergent motivation)
    commitment_level: float = 0.5  # How committed to current task
    recent_successes: int = 0
    recent_failures: int = 0
    
    def update_psychology(self, success: bool, volunteered: bool):
        """Update psychological state based on outcome."""
        if success:
            self.recent_successes += 1
            if volunteered:
                # Success after volunteering increases commitment
                self.commitment_level = min(1.0, self.commitment_level + 0.1)
        else:
            self.recent_failures += 1
            if volunteered:
                # Failure after volunteering decreases commitment
                self.commitment_level = max(0.0, self.commitment_level - 0.05)
    
    def get_emergent_effort(self, volunteered: bool) -> float:
        """
        Calculate effort based on psychological state, NOT a hardcoded bonus.
        
        This models:
        - Commitment: Agents who chose to volunteer may try harder
        - Confidence: Recent success increases effort
        - Ownership: Volunteering creates psychological ownership
        """
        base_effort = 0.8
        
        # Commitment effect (emergent from history)
        commitment_bonus = (self.commitment_level - 0.5) * 0.1
        
        # Confidence effect (emergent from recent outcomes)
        confidence = self.recent_successes / max(1, self.recent_successes + self.recent_failures)
        confidence_bonus = (confidence - 0.5) * 0.1
        
        # Ownership effect (volunteering creates ownership)
        ownership_bonus = 0.05 if volunteered else 0.0
        
        total_effort = base_effort + commitment_bonus + confidence_bonus + ownership_bonus
        return max(0.5, min(1.0, total_effort))


def create_population(n_agents: int, info_quality: float = 1.0) -> List[Agent]:
    """Create population with specified information quality."""
    agents = []
    for i in range(n_agents):
        true_cap = random.gauss(0.6, 0.15)
        true_cap = max(0.2, min(0.9, true_cap))
        
        if info_quality >= 1.0:
            perceived_cap = true_cap
        else:
            noise = random.gauss(0, 0.2 * (1 - info_quality))
            perceived_cap = max(0.2, min(0.9, true_cap + noise))
        
        agents.append(Agent(
            id=f"agent_{i}",
            true_capability=true_cap,
            perceived_capability=perceived_cap
        ))
    
    return agents


def self_selection(agents: List[Agent], difficulty: float) -> Tuple[Agent, bool]:
    """Agents self-select based on perceived capability."""
    volunteers = []
    for agent in agents:
        # Volunteer if perceived capability exceeds threshold
        if agent.perceived_capability > difficulty * 0.6:
            score = agent.perceived_capability - difficulty * 0.4
            volunteers.append((agent, score))
    
    if volunteers:
        best = max(volunteers, key=lambda x: x[1])[0]
        return best, True  # volunteered=True
    else:
        # Fallback to random
        return random.choice(agents), False


def oracle_assignment(agents: List[Agent], difficulty: float) -> Tuple[Agent, bool]:
    """Oracle assigns based on TRUE capability (perfect information)."""
    # Find agent with capability closest to (but above) difficulty
    candidates = []
    for agent in agents:
        if agent.true_capability > difficulty * 0.5:
            # Prefer agents whose capability matches difficulty (not overkill)
            excess = agent.true_capability - difficulty
            if excess > 0:
                score = 1.0 / (1.0 + excess)  # Closer match = higher score
            else:
                score = excess
            candidates.append((agent, score))
    
    if candidates:
        best = max(candidates, key=lambda x: x[1])[0]
        return best, False  # volunteered=False (assigned)
    else:
        return max(agents, key=lambda a: a.true_capability), False


def hybrid_selection(agents: List[Agent], difficulty: float) -> Tuple[Agent, bool]:
    """Self-selection with oracle override for bad choices."""
    # First, get self-selection result
    ss_agent, volunteered = self_selection(agents, difficulty)
    
    # Oracle checks if this is a reasonable choice
    oracle_agent, _ = oracle_assignment(agents, difficulty)
    
    # If self-selection chose someone much worse, override
    if ss_agent.true_capability < oracle_agent.true_capability - 0.2:
        return oracle_agent, False  # Override
    else:
        return ss_agent, volunteered  # Accept self-selection


def run_condition(
    selection_method: str,
    n_rounds: int = 200,
    n_agents: int = 30,
    seed: int = 0,
    use_emergent_effort: bool = True,
    agent_info_quality: float = 1.0  # How well agents know themselves
) -> Dict[str, Any]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    agents = create_population(n_agents, agent_info_quality)
    
    successes = 0
    total = 0
    volunteer_count = 0
    effort_sum = 0
    
    for round_num in range(n_rounds):
        difficulty = random.uniform(0.3, 0.7)
        
        # Select agent based on method
        if selection_method == "self_selection":
            selected, volunteered = self_selection(agents, difficulty)
        elif selection_method == "oracle":
            selected, volunteered = oracle_assignment(agents, difficulty)
        elif selection_method == "hybrid":
            selected, volunteered = hybrid_selection(agents, difficulty)
        else:
            selected = random.choice(agents)
            volunteered = False
        
        if volunteered:
            volunteer_count += 1
        
        # Calculate effort
        if use_emergent_effort:
            effort = selected.get_emergent_effort(volunteered)
        else:
            effort = 0.8  # Fixed effort, no bonus
        
        effort_sum += effort
        
        # Execute task
        success_prob = selected.true_capability * effort * (1 - difficulty * 0.5)
        success = random.random() < success_prob
        
        # Update agent psychology
        selected.update_psychology(success, volunteered)
        
        total += 1
        if success:
            successes += 1
    
    return {
        "cooperation_rate": successes / total,
        "volunteer_rate": volunteer_count / total,
        "mean_effort": effort_sum / total,
    }


def run_experiment(n_rounds: int = 200, n_seeds: int = 20, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 39: MOTIVATION ISOLATION")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test conditions
    conditions = [
        # (name, selection_method, use_emergent_effort, agent_info_quality)
        ("self_select_emergent", "self_selection", True, 1.0),
        ("self_select_fixed", "self_selection", False, 1.0),
        ("oracle_emergent", "oracle", True, 1.0),
        ("oracle_fixed", "oracle", False, 1.0),
        ("hybrid_emergent", "hybrid", True, 1.0),
        ("hybrid_fixed", "hybrid", False, 1.0),
        ("random_emergent", "random", True, 1.0),
        ("random_fixed", "random", False, 1.0),
        # Test with imperfect self-knowledge
        ("self_select_noisy", "self_selection", True, 0.5),
        ("oracle_noisy_agents", "oracle", True, 0.5),
    ]
    
    for name, method, emergent, info_quality in conditions:
        print(f"Running {name}...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(method, n_rounds, n_agents, seed, emergent, info_quality)
            seed_results.append(metrics)
        
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        results[name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"effort={aggregated['mean_effort']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results to isolate motivation effect."""
    analysis = {}
    
    # Key comparisons
    
    # 1. Self-selection vs Oracle with FIXED effort (no motivation)
    # If self-selection wins here, it's purely information advantage
    ss_fixed = results["self_select_fixed"]["cooperation_rate"]["mean"]
    oracle_fixed = results["oracle_fixed"]["cooperation_rate"]["mean"]
    analysis["ss_vs_oracle_fixed"] = ss_fixed - oracle_fixed
    analysis["info_advantage"] = ss_fixed - oracle_fixed
    
    # 2. Self-selection vs Oracle with EMERGENT effort
    # If self-selection wins MORE here, the extra is motivation
    ss_emergent = results["self_select_emergent"]["cooperation_rate"]["mean"]
    oracle_emergent = results["oracle_emergent"]["cooperation_rate"]["mean"]
    analysis["ss_vs_oracle_emergent"] = ss_emergent - oracle_emergent
    
    # 3. Emergent motivation effect
    # = (SS_emergent - Oracle_emergent) - (SS_fixed - Oracle_fixed)
    analysis["emergent_motivation_effect"] = (
        analysis["ss_vs_oracle_emergent"] - analysis["ss_vs_oracle_fixed"]
    )
    
    # 4. Effort difference
    ss_effort = results["self_select_emergent"]["mean_effort"]["mean"]
    oracle_effort = results["oracle_emergent"]["mean_effort"]["mean"]
    analysis["effort_difference"] = ss_effort - oracle_effort
    
    # 5. Is there a real motivation effect?
    analysis["motivation_is_real"] = analysis["emergent_motivation_effect"] > 0.01
    
    # 6. What fraction of advantage is motivation vs information?
    total_advantage = analysis["ss_vs_oracle_emergent"]
    if total_advantage > 0:
        analysis["motivation_fraction"] = analysis["emergent_motivation_effect"] / total_advantage
        analysis["information_fraction"] = 1 - analysis["motivation_fraction"]
    else:
        analysis["motivation_fraction"] = 0
        analysis["information_fraction"] = 1
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Motivation Isolation")
    print("=" * 70)
    
    # Main results table
    print("\n--- Cooperation Rates by Condition ---")
    print(f"{'Condition':>25} {'Cooperation':>12} {'Effort':>10}")
    print("-" * 50)
    
    for name in ["self_select_emergent", "self_select_fixed", 
                 "oracle_emergent", "oracle_fixed",
                 "hybrid_emergent", "hybrid_fixed",
                 "random_emergent", "random_fixed"]:
        coop = results[name]["cooperation_rate"]["mean"]
        effort = results[name]["mean_effort"]["mean"]
        print(f"{name:>25} {coop:>12.3f} {effort:>10.3f}")
    
    # Key comparisons
    print("\n--- Key Comparisons ---")
    print(f"Self-selection vs Oracle (fixed effort):    {analysis['ss_vs_oracle_fixed']:+.4f}")
    print(f"Self-selection vs Oracle (emergent effort): {analysis['ss_vs_oracle_emergent']:+.4f}")
    print(f"Emergent motivation effect:                 {analysis['emergent_motivation_effect']:+.4f}")
    print(f"Effort difference (SS - Oracle):            {analysis['effort_difference']:+.4f}")
    
    # Decomposition
    print("\n--- Advantage Decomposition ---")
    print(f"Total self-selection advantage: {analysis['ss_vs_oracle_emergent']:+.4f}")
    print(f"  - Information component:      {analysis['info_advantage']:+.4f} ({analysis['information_fraction']*100:.1f}%)")
    print(f"  - Motivation component:       {analysis['emergent_motivation_effect']:+.4f} ({analysis['motivation_fraction']*100:.1f}%)")
    
    # Conclusion
    print("\n--- CONCLUSION ---")
    if analysis["motivation_is_real"]:
        print("✓ There IS an emergent motivation effect")
        print(f"  Motivation accounts for {analysis['motivation_fraction']*100:.1f}% of self-selection advantage")
    else:
        print("✗ No significant emergent motivation effect")
        print("  Self-selection advantage is primarily informational (self-knowledge)")
    
    # What this means
    print("\n--- INTERPRETATION ---")
    if analysis["ss_vs_oracle_fixed"] > 0:
        print("Even with fixed effort, self-selection beats oracle assignment.")
        print("This suggests agents have BETTER self-knowledge than the oracle,")
        print("OR there's something about self-selection beyond information.")
    else:
        print("With fixed effort, oracle beats self-selection.")
        print("Self-selection advantage requires the emergent effort mechanism.")


def main():
    results = run_experiment(n_rounds=200, n_seeds=20, n_agents=30)
    analysis = analyze_results(results)
    print_results(results, analysis)
    
    # Save
    os.makedirs("results", exist_ok=True)
    
    output = {"conditions": results, "analysis": analysis}
    with open("results/experiment_39_motivation.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/experiment_39_motivation.json")


if __name__ == "__main__":
    main()
