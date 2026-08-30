"""
Experiment 16: Consequence Severity Sweep

Test if there's an optimal consequence level, or if ANY consequences hurt.

The punishment spiral analysis showed:
- Recovery rates are 100% (no permanent exclusion)
- But cooperation drops from 49% to 26% with consequences
- Average active penalties: 17.39 with consequences

This suggests RETALIATION CYCLES, not spirals. Let's test if milder
consequences can provide accountability without triggering retaliation.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gcl.population.ablation import (
    AblationAgent,
    InfrastructureConfig,
)

RESULTS_DIR = Path("results/16_severity_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SeverityConfig:
    """Configuration for variable severity experiments."""
    violation_penalty: float  # How much cooperation probability drops when penalized
    penalty_duration: int     # How long penalties last
    
    def apply_to_config(self, config: InfrastructureConfig) -> InfrastructureConfig:
        """Apply severity settings to an infrastructure config."""
        config.violation_penalty = self.violation_penalty
        config.penalty_duration = self.penalty_duration
        return config


def run_severity_experiment(
    severity: SeverityConfig,
    n_agents: int = 50,
    n_rounds: int = 100,
    interactions_per_round: int = 100,
    seed: int = 42
) -> Dict:
    """Run experiment with specific severity settings."""
    
    rng = np.random.default_rng(seed)
    
    # Create config with specified severity
    config = InfrastructureConfig.full()
    config = severity.apply_to_config(config)
    
    # Create agents
    agents = []
    for i in range(n_agents):
        agent = AblationAgent(
            agent_id=i,
            config=config,
            base_cooperation_rate=rng.uniform(0.3, 0.7)
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
            
            # Each agent decides whether to cooperate
            a_cooperates = agent_a.decide_cooperation(agent_b.agent_id)
            b_cooperates = agent_b.decide_cooperation(agent_a.agent_id)
            
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
        
        # Decay penalties
        for agent in agents:
            agent.decay_penalties()
        
        cooperation_rates.append(round_cooperations / round_interactions)
        penalty_counts.append(round_penalties / n_agents)
    
    return {
        'final_cooperation': cooperation_rates[-1],
        'avg_cooperation': np.mean(cooperation_rates),
        'cooperation_over_time': cooperation_rates,
        'avg_penalties': np.mean(penalty_counts),
        'final_penalties': penalty_counts[-1],
    }


def run_severity_sweep(
    n_agents: int = 50,
    n_rounds: int = 100,
    n_seeds: int = 5
) -> Dict:
    """Sweep across consequence severity levels."""
    
    # Define severity levels to test
    # Original: penalty=0.5, duration=10
    severities = [
        SeverityConfig(violation_penalty=0.0, penalty_duration=0),   # No consequences
        SeverityConfig(violation_penalty=0.05, penalty_duration=3),  # Very mild
        SeverityConfig(violation_penalty=0.1, penalty_duration=5),   # Mild
        SeverityConfig(violation_penalty=0.2, penalty_duration=7),   # Moderate-low
        SeverityConfig(violation_penalty=0.3, penalty_duration=10),  # Moderate
        SeverityConfig(violation_penalty=0.5, penalty_duration=10),  # Original (high)
        SeverityConfig(violation_penalty=0.7, penalty_duration=15),  # Very high
        SeverityConfig(violation_penalty=1.0, penalty_duration=20),  # Extreme
    ]
    
    results = {}
    
    for severity in severities:
        key = f"penalty_{severity.violation_penalty:.2f}_duration_{severity.penalty_duration}"
        print(f"\nTesting: {key}")
        
        seed_results = []
        for seed in range(n_seeds):
            result = run_severity_experiment(
                severity=severity,
                n_agents=n_agents,
                n_rounds=n_rounds,
                seed=seed
            )
            seed_results.append(result)
        
        results[key] = {
            'violation_penalty': severity.violation_penalty,
            'penalty_duration': severity.penalty_duration,
            'mean_cooperation': np.mean([r['final_cooperation'] for r in seed_results]),
            'std_cooperation': np.std([r['final_cooperation'] for r in seed_results]),
            'mean_avg_cooperation': np.mean([r['avg_cooperation'] for r in seed_results]),
            'mean_penalties': np.mean([r['avg_penalties'] for r in seed_results]),
            'seeds': seed_results,
        }
        
        print(f"  Cooperation: {results[key]['mean_cooperation']:.1%} (±{results[key]['std_cooperation']:.1%})")
        print(f"  Avg penalties: {results[key]['mean_penalties']:.2f}")
    
    return results


def analyze_severity_results(results: Dict) -> Dict:
    """Find optimal severity and characterize the curve."""
    
    # Extract data points
    data_points = []
    for key, data in results.items():
        data_points.append({
            'penalty': data['violation_penalty'],
            'duration': data['penalty_duration'],
            'cooperation': data['mean_cooperation'],
            'std': data['std_cooperation'],
            'penalties': data['mean_penalties'],
        })
    
    # Sort by penalty severity
    data_points.sort(key=lambda x: x['penalty'])
    
    penalties = [d['penalty'] for d in data_points]
    cooperations = [d['cooperation'] for d in data_points]
    
    # Find optimal
    best_idx = np.argmax(cooperations)
    optimal_penalty = penalties[best_idx]
    max_cooperation = cooperations[best_idx]
    
    # Check if monotonic decreasing (any consequences hurt)
    is_monotonic_decreasing = all(
        cooperations[i] >= cooperations[i+1] - 0.02  # Allow 2% noise
        for i in range(len(cooperations)-1)
    )
    
    # Check for inverted-U (mild consequences help)
    has_inverted_u = (
        optimal_penalty > 0 and 
        optimal_penalty < penalties[-1] and
        cooperations[best_idx] > cooperations[0] + 0.02
    )
    
    # Linear regression
    slope = np.polyfit(penalties, cooperations, 1)[0]
    
    return {
        'data_points': data_points,
        'optimal_penalty': optimal_penalty,
        'max_cooperation': max_cooperation,
        'zero_penalty_cooperation': cooperations[0],
        'is_monotonic_decreasing': is_monotonic_decreasing,
        'has_inverted_u': has_inverted_u,
        'slope': slope,
        'interpretation': interpret_severity_curve(
            optimal_penalty, is_monotonic_decreasing, has_inverted_u, slope
        )
    }


def interpret_severity_curve(
    optimal: float, 
    monotonic: bool, 
    inverted_u: bool,
    slope: float
) -> str:
    """Interpret the severity-cooperation relationship."""
    
    if monotonic and optimal == 0:
        return """
ANY CONSEQUENCES HURT COOPERATION.

The relationship is monotonically decreasing: more severe consequences 
always lead to less cooperation. The optimal consequence level is zero.

This strongly supports removing punishment mechanisms entirely, or
implementing very different accountability approaches (e.g., positive
reinforcement for cooperation rather than punishment for defection).

The mechanism appears to be RETALIATION CYCLES:
1. Agent A defects → Agent B penalizes A
2. A's cooperation probability toward B drops
3. A more likely to defect against B → B penalizes again
4. Cycle continues, spreading through the population

Without penalties, defection doesn't trigger retaliation, so
cooperation can recover naturally through reputation and memory.
"""
    elif inverted_u:
        return f"""
MILD CONSEQUENCES MAY HELP.

Optimal severity is {optimal:.0%}. Very mild consequences might provide 
accountability benefits without triggering retaliation cycles.

This suggests a "gentle nudge" approach:
- Small, short-lived penalties for defection
- Enough to signal disapproval
- Not enough to trigger defensive retaliation

Consider implementing "warnings" rather than hard punishments.
"""
    elif optimal > 0:
        return f"""
SOME CONSEQUENCES HELP.

Optimal penalty is {optimal:.0%}. Consequences at this level improve
cooperation compared to no consequences.

This contradicts the simple "consequences hurt" narrative and suggests
the relationship is more nuanced. The key may be finding the right
balance between accountability and forgiveness.
"""
    else:
        return """
COMPLEX RELATIONSHIP.

The severity-cooperation curve is non-monotonic but doesn't show
a clear inverted-U pattern. Further investigation needed to understand
the dynamics.
"""


def create_visualizations(results: Dict, analysis: Dict):
    """Create visualizations for the severity sweep."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    data = analysis['data_points']
    penalties = [d['penalty'] for d in data]
    cooperations = [d['cooperation'] for d in data]
    stds = [d['std'] for d in data]
    avg_penalties = [d['penalties'] for d in data]
    
    # Plot 1: Cooperation vs Penalty Severity
    ax1 = axes[0, 0]
    ax1.errorbar(penalties, cooperations, yerr=stds, fmt='o-', capsize=5, 
                 linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=analysis['zero_penalty_cooperation'], color='green', 
                linestyle='--', alpha=0.7, label='No consequences baseline')
    ax1.axvline(x=analysis['optimal_penalty'], color='red', 
                linestyle='--', alpha=0.7, label=f'Optimal: {analysis["optimal_penalty"]:.0%}')
    ax1.set_xlabel('Violation Penalty (cooperation probability reduction)')
    ax1.set_ylabel('Final Cooperation Rate')
    ax1.set_title('Cooperation vs Consequence Severity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Active Penalties vs Severity
    ax2 = axes[0, 1]
    ax2.plot(penalties, avg_penalties, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Violation Penalty')
    ax2.set_ylabel('Average Active Penalties per Agent')
    ax2.set_title('Penalty Accumulation vs Severity')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cooperation vs Penalties (scatter)
    ax3 = axes[1, 0]
    ax3.scatter(avg_penalties, cooperations, s=100, c=penalties, cmap='RdYlGn_r')
    ax3.set_xlabel('Average Active Penalties')
    ax3.set_ylabel('Cooperation Rate')
    ax3.set_title('Cooperation vs Penalty Load')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Penalty Severity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary bar chart
    ax4 = axes[1, 1]
    labels = [f'{p:.0%}' for p in penalties]
    colors = ['green' if c == max(cooperations) else 'blue' for c in cooperations]
    bars = ax4.bar(labels, cooperations, color=colors, alpha=0.7)
    ax4.set_xlabel('Penalty Severity')
    ax4.set_ylabel('Cooperation Rate')
    ax4.set_title('Cooperation by Severity Level')
    ax4.axhline(y=analysis['zero_penalty_cooperation'], color='green', 
                linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar, coop in zip(bars, cooperations):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{coop:.0%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "severity_sweep_results.png", dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {RESULTS_DIR / 'severity_sweep_results.png'}")


def main():
    print("=" * 60)
    print("EXPERIMENT 16: Consequence Severity Sweep")
    print("=" * 60)
    
    results = run_severity_sweep(n_agents=50, n_rounds=100, n_seeds=5)
    analysis = analyze_severity_results(results)
    
    # Create visualizations
    create_visualizations(results, analysis)
    
    # Save results
    (RESULTS_DIR / "severity_results.json").write_text(
        json.dumps(results, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    (RESULTS_DIR / "severity_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nCooperation by Severity Level:")
    print("-" * 50)
    for d in analysis['data_points']:
        marker = " ← OPTIMAL" if d['penalty'] == analysis['optimal_penalty'] else ""
        print(f"  Penalty {d['penalty']:.0%}, Duration {d['duration']:2d}: "
              f"{d['cooperation']:.1%} (±{d['std']:.1%}){marker}")
    
    print(f"""
\nKEY FINDINGS:
  Optimal penalty: {analysis['optimal_penalty']:.0%}
  Max cooperation: {analysis['max_cooperation']:.1%}
  Zero-penalty cooperation: {analysis['zero_penalty_cooperation']:.1%}
  
  Monotonically decreasing: {analysis['is_monotonic_decreasing']}
  Has inverted-U shape: {analysis['has_inverted_u']}
  Slope: {analysis['slope']:.4f} (cooperation change per penalty unit)

{analysis['interpretation']}
""")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return results, analysis


if __name__ == "__main__":
    main()
