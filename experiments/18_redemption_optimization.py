"""
Experiment 18: Redemption Mechanism Optimization

Previous finding: Redemption (+30%) achieved 64.9% cooperation, beating no-consequences (51.9%)!

This experiment:
1. Sweep redemption bonus levels (10%, 20%, 30%, 40%, 50%, 60%)
2. Find the optimal redemption bonus
3. Test redemption + mild consequences vs pure redemption
4. Test redemption + fast decay combinations
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gcl.population.ablation import (
    AblationAgent,
    InfrastructureConfig,
)

RESULTS_DIR = Path("results/18_redemption_optimization")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RedemptionAgent(AblationAgent):
    """Agent with redemption mechanism."""
    
    redemption_boost: float = 0.3  # Extra cooperation boost when penalized
    penalty_reduction: float = 1.0  # Multiplier for penalty effect (1.0 = standard, 0.5 = mild)
    decay_multiplier: float = 1.0  # Multiplier for decay rate (1.0 = standard, 2.0 = fast)
    
    def decide_cooperation_with_redemption(self, partner_id: int) -> bool:
        """Decide cooperation with redemption boost when penalized."""
        # Base decision
        coop_prob = self.base_cooperation_rate
        
        # Standard adjustments from memory and reputation
        if self.config.memory_enabled:
            history = self.get_partner_history(partner_id)
            if history:
                recent = history[-1]
                if recent["outcome"] == "cooperated":
                    coop_prob += 0.2
                elif recent["outcome"] == "defected":
                    coop_prob -= 0.2
        
        if self.config.reputation_enabled:
            partner_rep = self.get_reputation_belief(partner_id)
            coop_prob += 0.3 * (partner_rep - 0.5)
        
        # REDEMPTION: if we are penalized by this partner, try harder to cooperate
        if self.config.consequences_enabled and self.has_penalty(partner_id):
            # Instead of reducing cooperation (standard), INCREASE it
            coop_prob += self.redemption_boost
            # But still apply some penalty effect (reduced by penalty_reduction)
            coop_prob -= self.config.violation_penalty * self.penalty_reduction * 0.5
        
        coop_prob = np.clip(coop_prob, 0.0, 1.0)
        return self.rng.random() < coop_prob
    
    def decay_penalties_fast(self):
        """Decay penalties with configurable speed."""
        decay_amount = int(self.decay_multiplier)
        for agent_id in list(self.active_penalties.keys()):
            self.active_penalties[agent_id] -= decay_amount
            if self.active_penalties[agent_id] <= 0:
                del self.active_penalties[agent_id]


def run_redemption_experiment(
    redemption_boost: float,
    penalty_reduction: float = 1.0,
    decay_multiplier: float = 1.0,
    violation_penalty: float = 0.5,
    penalty_duration: int = 10,
    n_agents: int = 50,
    n_rounds: int = 100,
    interactions_per_round: int = 100,
    seed: int = 42
) -> Dict:
    """Run experiment with specific redemption settings."""
    
    rng = np.random.default_rng(seed)
    
    # Create config
    config = InfrastructureConfig.full()
    config.violation_penalty = violation_penalty
    config.penalty_duration = penalty_duration
    
    # Create agents with redemption
    agents = []
    for i in range(n_agents):
        agent = RedemptionAgent(
            agent_id=i,
            config=config,
            base_cooperation_rate=rng.uniform(0.3, 0.7),
            redemption_boost=redemption_boost,
            penalty_reduction=penalty_reduction,
            decay_multiplier=decay_multiplier,
        )
        agents.append(agent)
    
    cooperation_rates = []
    penalty_counts = []
    
    for round_num in range(n_rounds):
        round_cooperations = 0
        round_interactions = 0
        round_penalties = sum(len(a.active_penalties) for a in agents)
        
        for _ in range(interactions_per_round):
            # Random pairing
            i, j = rng.choice(len(agents), size=2, replace=False)
            agent_a, agent_b = agents[i], agents[j]
            
            # Each agent decides whether to cooperate (with redemption)
            a_cooperates = agent_a.decide_cooperation_with_redemption(agent_b.agent_id)
            b_cooperates = agent_b.decide_cooperation_with_redemption(agent_a.agent_id)
            
            round_cooperations += int(a_cooperates) + int(b_cooperates)
            round_interactions += 2
            
            # Prisoner's dilemma payoffs
            if a_cooperates and b_cooperates:
                a_reward, b_reward = 3.0, 3.0
            elif a_cooperates and not b_cooperates:
                a_reward, b_reward = 0.0, 5.0
            elif not a_cooperates and b_cooperates:
                a_reward, b_reward = 5.0, 0.0
            else:
                a_reward, b_reward = 1.0, 1.0
            
            # Update memories
            agent_a.remember_interaction(
                agent_b.agent_id,
                "cooperate" if a_cooperates else "defect",
                "cooperated" if b_cooperates else "defected",
                a_reward
            )
            agent_b.remember_interaction(
                agent_a.agent_id,
                "cooperate" if b_cooperates else "defect",
                "cooperated" if a_cooperates else "defected",
                b_reward
            )
            
            # Update reputations
            agent_a.update_reputation_belief(agent_b.agent_id, b_cooperates)
            agent_b.update_reputation_belief(agent_a.agent_id, a_cooperates)
            
            # Apply penalties for defection
            if not a_cooperates:
                agent_b.apply_penalty(agent_a.agent_id)
            if not b_cooperates:
                agent_a.apply_penalty(agent_b.agent_id)
        
        # Decay penalties (with configurable speed)
        for agent in agents:
            agent.decay_penalties_fast()
        
        cooperation_rates.append(round_cooperations / round_interactions)
        penalty_counts.append(round_penalties / n_agents)
    
    return {
        'final_cooperation': cooperation_rates[-1],
        'avg_cooperation': np.mean(cooperation_rates),
        'cooperation_over_time': cooperation_rates,
        'avg_penalties': np.mean(penalty_counts),
        'final_penalties': penalty_counts[-1],
    }


def run_redemption_sweep(n_seeds: int = 5) -> Dict:
    """Sweep across redemption bonus levels."""
    
    print("=" * 60)
    print("PART 1: Redemption Bonus Sweep")
    print("=" * 60)
    
    redemption_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = {}
    
    for boost in redemption_levels:
        key = f"redemption_{boost:.0%}"
        print(f"\nTesting: {key}")
        
        seed_results = []
        for seed in range(n_seeds):
            result = run_redemption_experiment(
                redemption_boost=boost,
                seed=seed
            )
            seed_results.append(result)
        
        results[key] = {
            'redemption_boost': boost,
            'mean_cooperation': np.mean([r['final_cooperation'] for r in seed_results]),
            'std_cooperation': np.std([r['final_cooperation'] for r in seed_results]),
            'mean_penalties': np.mean([r['avg_penalties'] for r in seed_results]),
        }
        
        print(f"  Cooperation: {results[key]['mean_cooperation']:.1%} (±{results[key]['std_cooperation']:.1%})")
    
    return results


def run_combination_experiments(n_seeds: int = 5) -> Dict:
    """Test redemption + mild consequences combinations."""
    
    print("\n" + "=" * 60)
    print("PART 2: Redemption + Mild Consequences Combinations")
    print("=" * 60)
    
    # Test combinations of redemption with different penalty severities
    combinations = [
        # (redemption_boost, violation_penalty, penalty_duration, decay_mult, name)
        (0.3, 0.5, 10, 1, "Redemption 30% + Standard penalty"),
        (0.3, 0.3, 10, 1, "Redemption 30% + Mild penalty (30%)"),
        (0.3, 0.2, 10, 1, "Redemption 30% + Very mild penalty (20%)"),
        (0.3, 0.1, 10, 1, "Redemption 30% + Minimal penalty (10%)"),
        (0.3, 0.5, 5, 1, "Redemption 30% + Short duration (5)"),
        (0.3, 0.5, 10, 2, "Redemption 30% + Fast decay (2x)"),
        (0.3, 0.3, 5, 2, "Redemption 30% + Mild + Short + Fast"),
        (0.4, 0.3, 5, 2, "Redemption 40% + Mild + Short + Fast"),
        (0.5, 0.2, 5, 2, "Redemption 50% + Very mild + Short + Fast"),
        (0.0, 0.0, 0, 1, "No consequences (baseline)"),
    ]
    
    results = {}
    
    for boost, penalty, duration, decay, name in combinations:
        print(f"\nTesting: {name}")
        
        seed_results = []
        for seed in range(n_seeds):
            result = run_redemption_experiment(
                redemption_boost=boost,
                violation_penalty=penalty,
                penalty_duration=duration,
                decay_multiplier=decay,
                seed=seed
            )
            seed_results.append(result)
        
        results[name] = {
            'redemption_boost': boost,
            'violation_penalty': penalty,
            'penalty_duration': duration,
            'decay_multiplier': decay,
            'mean_cooperation': np.mean([r['final_cooperation'] for r in seed_results]),
            'std_cooperation': np.std([r['final_cooperation'] for r in seed_results]),
            'mean_penalties': np.mean([r['avg_penalties'] for r in seed_results]),
        }
        
        print(f"  Cooperation: {results[name]['mean_cooperation']:.1%} (±{results[name]['std_cooperation']:.1%})")
    
    return results


def analyze_results(sweep_results: Dict, combo_results: Dict) -> Dict:
    """Analyze all results to find optimal configuration."""
    
    # Find optimal redemption level
    sweep_data = [(k, v['redemption_boost'], v['mean_cooperation']) 
                  for k, v in sweep_results.items()]
    sweep_data.sort(key=lambda x: -x[2])
    
    optimal_boost = sweep_data[0][1]
    optimal_boost_coop = sweep_data[0][2]
    
    # Find best combination
    combo_data = [(k, v['mean_cooperation']) for k, v in combo_results.items()]
    combo_data.sort(key=lambda x: -x[1])
    
    best_combo = combo_data[0][0]
    best_combo_coop = combo_data[0][2] if len(combo_data[0]) > 2 else combo_data[0][1]
    
    # Get baselines
    no_consequences = combo_results.get('No consequences (baseline)', {}).get('mean_cooperation', 0)
    
    return {
        'optimal_redemption_boost': optimal_boost,
        'optimal_boost_cooperation': optimal_boost_coop,
        'best_combination': best_combo,
        'best_combination_cooperation': best_combo_coop,
        'no_consequences_baseline': no_consequences,
        'sweep_ranking': sweep_data,
        'combo_ranking': combo_data,
    }


def create_visualizations(sweep_results: Dict, combo_results: Dict, analysis: Dict):
    """Create visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Redemption bonus sweep
    ax1 = axes[0, 0]
    boosts = [v['redemption_boost'] for v in sweep_results.values()]
    coops = [v['mean_cooperation'] for v in sweep_results.values()]
    stds = [v['std_cooperation'] for v in sweep_results.values()]
    
    ax1.errorbar(boosts, coops, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax1.axhline(y=analysis['no_consequences_baseline'], color='green', linestyle='--', 
                alpha=0.7, label='No consequences baseline')
    ax1.axvline(x=analysis['optimal_redemption_boost'], color='red', linestyle='--',
                alpha=0.7, label=f'Optimal: {analysis["optimal_redemption_boost"]:.0%}')
    ax1.set_xlabel('Redemption Boost')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Cooperation vs Redemption Bonus Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combination comparison
    ax2 = axes[0, 1]
    combo_names = list(combo_results.keys())
    combo_coops = [combo_results[n]['mean_cooperation'] for n in combo_names]
    combo_stds = [combo_results[n]['std_cooperation'] for n in combo_names]
    
    # Sort by cooperation
    sorted_pairs = sorted(zip(combo_names, combo_coops, combo_stds), key=lambda x: -x[1])
    combo_names, combo_coops, combo_stds = zip(*sorted_pairs)
    
    colors = ['green' if 'baseline' in n else 'blue' for n in combo_names]
    ax2.barh(range(len(combo_names)), combo_coops, xerr=combo_stds, color=colors, alpha=0.7, capsize=3)
    ax2.set_yticks(range(len(combo_names)))
    ax2.set_yticklabels([n[:35] for n in combo_names], fontsize=8)
    ax2.set_xlabel('Cooperation Rate')
    ax2.set_title('Redemption + Consequences Combinations')
    ax2.axvline(x=analysis['no_consequences_baseline'], color='green', linestyle='--', alpha=0.5)
    
    # Plot 3: Penalties vs Cooperation
    ax3 = axes[1, 0]
    all_penalties = [v['mean_penalties'] for v in combo_results.values()]
    all_coops = [v['mean_cooperation'] for v in combo_results.values()]
    all_names = list(combo_results.keys())
    
    scatter = ax3.scatter(all_penalties, all_coops, s=100, c=range(len(all_names)), cmap='viridis')
    for i, name in enumerate(all_names):
        ax3.annotate(name[:20], (all_penalties[i], all_coops[i]), fontsize=7,
                    xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Average Active Penalties')
    ax3.set_ylabel('Cooperation Rate')
    ax3.set_title('Cooperation vs Penalty Load')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    summary_names = ['No Consequences', 'Standard\n(no redemption)', 
                     f'Optimal Redemption\n({analysis["optimal_redemption_boost"]:.0%})',
                     'Best Combination']
    summary_coops = [
        analysis['no_consequences_baseline'],
        sweep_results.get('redemption_0%', {}).get('mean_cooperation', 0.32),
        analysis['optimal_boost_cooperation'],
        analysis['best_combination_cooperation'],
    ]
    colors = ['green', 'red', 'blue', 'purple']
    
    bars = ax4.bar(range(len(summary_names)), summary_coops, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(summary_names)))
    ax4.set_xticklabels(summary_names, fontsize=9)
    ax4.set_ylabel('Cooperation Rate')
    ax4.set_title('Key Comparisons')
    
    for bar, coop in zip(bars, summary_coops):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{coop:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "redemption_optimization.png", dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {RESULTS_DIR / 'redemption_optimization.png'}")


def main():
    print("=" * 70)
    print("EXPERIMENT 18: Redemption Mechanism Optimization")
    print("=" * 70)
    
    # Part 1: Redemption bonus sweep
    sweep_results = run_redemption_sweep(n_seeds=5)
    
    # Part 2: Combination experiments
    combo_results = run_combination_experiments(n_seeds=5)
    
    # Analyze
    analysis = analyze_results(sweep_results, combo_results)
    
    # Visualize
    create_visualizations(sweep_results, combo_results, analysis)
    
    # Save results
    all_results = {
        'sweep': sweep_results,
        'combinations': combo_results,
        'analysis': analysis,
    }
    (RESULTS_DIR / "redemption_optimization_results.json").write_text(
        json.dumps(all_results, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n--- REDEMPTION BONUS SWEEP ---")
    print(f"{'Boost':<10} {'Cooperation':<15}")
    print("-" * 25)
    for name, boost, coop in analysis['sweep_ranking']:
        marker = " ← OPTIMAL" if boost == analysis['optimal_redemption_boost'] else ""
        print(f"{boost:<10.0%} {coop:<15.1%}{marker}")
    
    print("\n--- COMBINATION RANKING ---")
    print(f"{'Configuration':<45} {'Cooperation':<15}")
    print("-" * 60)
    for name, coop in analysis['combo_ranking'][:10]:
        print(f"{name:<45} {coop:<15.1%}")
    
    print(f"""
\n--- KEY FINDINGS ---

1. OPTIMAL REDEMPTION BONUS: {analysis['optimal_redemption_boost']:.0%}
   Cooperation: {analysis['optimal_boost_cooperation']:.1%}

2. BEST COMBINATION: {analysis['best_combination']}
   Cooperation: {analysis['best_combination_cooperation']:.1%}

3. NO CONSEQUENCES BASELINE: {analysis['no_consequences_baseline']:.1%}

4. IMPROVEMENT OVER NO CONSEQUENCES:
   Optimal redemption: {(analysis['optimal_boost_cooperation'] - analysis['no_consequences_baseline']):.1%}
   Best combination: {(analysis['best_combination_cooperation'] - analysis['no_consequences_baseline']):.1%}
""")
    
    # Interpretation
    if analysis['optimal_boost_cooperation'] > analysis['no_consequences_baseline'] + 0.1:
        print("""
✓ REDEMPTION SIGNIFICANTLY OUTPERFORMS NO-CONSEQUENCES!

The redemption mechanism transforms punishment from adversarial to cooperative:
- Penalized agents try HARDER to cooperate (to redeem themselves)
- This breaks the retaliation cycle
- Result: Higher cooperation than even no-consequences baseline

RECOMMENDATION: Implement redemption with {:.0%} boost for optimal results.
""".format(analysis['optimal_redemption_boost']))
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return all_results


if __name__ == "__main__":
    main()
