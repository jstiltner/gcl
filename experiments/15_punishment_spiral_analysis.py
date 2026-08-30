"""
Experiment 15: Punishment Spiral Analysis

Track agent trajectories after their first failure.
Do they recover or spiral into exclusion?

Key metrics:
- Time to first failure
- Reputation trajectory after first failure
- Number of future interactions after first failure
- Recovery rate (return to baseline reputation)

This experiment investigates WHY consequences hurt cooperation in our ablation study.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gcl.population.ablation import (
    AblationExperiment,
    AblationAgent,
    InfrastructureConfig,
    InfrastructureComponent,
)

RESULTS_DIR = Path("results/15_punishment_spirals")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentTrajectory:
    """Track an agent's trajectory over time."""
    agent_id: int
    reputation_history: List[float] = field(default_factory=list)
    cooperation_history: List[bool] = field(default_factory=list)  # Did this agent cooperate?
    partner_cooperation_history: List[bool] = field(default_factory=list)  # Did partner cooperate?
    penalty_count_history: List[int] = field(default_factory=list)  # Active penalties at each step
    
    @property
    def first_defection_time(self) -> Optional[int]:
        """When did this agent first defect?"""
        for i, coop in enumerate(self.cooperation_history):
            if not coop:
                return i
        return None
    
    @property
    def first_penalty_time(self) -> Optional[int]:
        """When did this agent first receive a penalty?"""
        for i, count in enumerate(self.penalty_count_history):
            if count > 0:
                return i
        return None
    
    @property
    def total_defections(self) -> int:
        return sum(1 for c in self.cooperation_history if not c)
    
    @property
    def total_cooperations(self) -> int:
        return sum(1 for c in self.cooperation_history if c)
    
    def reputation_after_first_defection(self, window: int = 50) -> Optional[List[float]]:
        """Get reputation trajectory for `window` steps after first defection."""
        if self.first_defection_time is None:
            return None
        start_idx = self.first_defection_time
        end_idx = min(start_idx + window, len(self.reputation_history))
        return self.reputation_history[start_idx:end_idx]
    
    def recovered(self, threshold: float = 0.9) -> bool:
        """Did agent recover to threshold of initial reputation?"""
        if self.first_defection_time is None or len(self.reputation_history) < 2:
            return True  # Never defected
        
        initial_rep = self.reputation_history[0] if self.reputation_history else 0.5
        post_defection_reps = self.reputation_history[self.first_defection_time:]
        
        if not post_defection_reps:
            return False
        
        max_recovery = max(post_defection_reps)
        return max_recovery >= initial_rep * threshold


class TrajectoryTracker:
    """Track all agent trajectories during simulation."""
    
    def __init__(self, agents: List[AblationAgent]):
        self.agents = agents
        self.trajectories: Dict[int, AgentTrajectory] = {
            agent.agent_id: AgentTrajectory(agent_id=agent.agent_id)
            for agent in agents
        }
        self.timestep = 0
        self.round_cooperations: List[float] = []
    
    def record_interaction(
        self,
        agent_a: AblationAgent,
        agent_b: AblationAgent,
        a_cooperated: bool,
        b_cooperated: bool
    ):
        """Record an interaction between two agents."""
        # Record for agent A
        traj_a = self.trajectories[agent_a.agent_id]
        traj_a.cooperation_history.append(a_cooperated)
        traj_a.partner_cooperation_history.append(b_cooperated)
        traj_a.penalty_count_history.append(len(agent_a.active_penalties))
        
        # Record for agent B
        traj_b = self.trajectories[agent_b.agent_id]
        traj_b.cooperation_history.append(b_cooperated)
        traj_b.partner_cooperation_history.append(a_cooperated)
        traj_b.penalty_count_history.append(len(agent_b.active_penalties))
    
    def record_round_end(self):
        """Record state at end of round."""
        self.timestep += 1
        
        # Record reputations (using reliability as proxy)
        for agent in self.agents:
            self.trajectories[agent.agent_id].reputation_history.append(agent.reliability)
        
        # Record round cooperation rate
        total_coop = sum(t.cooperation_history[-1] if t.cooperation_history else 0 
                        for t in self.trajectories.values())
        total_interactions = sum(1 for t in self.trajectories.values() if t.cooperation_history)
        if total_interactions > 0:
            self.round_cooperations.append(total_coop / total_interactions)
    
    def get_summary(self) -> Dict:
        """Summarize trajectory data."""
        
        agents_with_defections = [t for t in self.trajectories.values() 
                                  if t.first_defection_time is not None]
        agents_without_defections = [t for t in self.trajectories.values() 
                                     if t.first_defection_time is None]
        
        recovery_rate = (np.mean([t.recovered() for t in agents_with_defections]) 
                        if agents_with_defections else 1.0)
        
        # Average reputation trajectory after first defection
        post_defection_trajectories = [t.reputation_after_first_defection(50) 
                                       for t in agents_with_defections]
        post_defection_trajectories = [t for t in post_defection_trajectories 
                                       if t and len(t) > 5]
        
        if post_defection_trajectories:
            # Pad to same length
            max_len = max(len(t) for t in post_defection_trajectories)
            padded = [t + [t[-1]] * (max_len - len(t)) for t in post_defection_trajectories]
            avg_trajectory = np.mean(padded, axis=0).tolist()
        else:
            avg_trajectory = []
        
        # Penalty statistics
        avg_penalties = np.mean([
            np.mean(t.penalty_count_history) if t.penalty_count_history else 0
            for t in self.trajectories.values()
        ])
        
        return {
            'n_agents_with_defections': len(agents_with_defections),
            'n_agents_without_defections': len(agents_without_defections),
            'recovery_rate': recovery_rate,
            'avg_defections_per_agent': np.mean([t.total_defections for t in self.trajectories.values()]),
            'avg_cooperations_per_agent': np.mean([t.total_cooperations for t in self.trajectories.values()]),
            'avg_reputation_trajectory_after_defection': avg_trajectory,
            'final_cooperation_rate': self.round_cooperations[-1] if self.round_cooperations else 0,
            'avg_active_penalties': avg_penalties,
            'cooperation_over_time': self.round_cooperations,
        }


def run_tracked_experiment(
    config: InfrastructureConfig,
    n_agents: int = 50,
    n_rounds: int = 100,
    interactions_per_round: int = 100,
    seed: int = 42
) -> Tuple[TrajectoryTracker, Dict]:
    """Run experiment with detailed trajectory tracking."""
    
    rng = np.random.default_rng(seed)
    
    # Create agents
    agents = []
    for i in range(n_agents):
        agent = AblationAgent(
            agent_id=i,
            config=config,
            base_cooperation_rate=rng.uniform(0.3, 0.7)
        )
        agents.append(agent)
    
    tracker = TrajectoryTracker(agents)
    
    for round_num in range(n_rounds):
        for _ in range(interactions_per_round):
            # Random pairing
            i, j = rng.choice(len(agents), size=2, replace=False)
            agent_a, agent_b = agents[i], agents[j]
            
            # Each agent decides whether to cooperate
            a_cooperates = agent_a.decide_cooperation(agent_b.agent_id)
            b_cooperates = agent_b.decide_cooperation(agent_a.agent_id)
            
            # Record interaction BEFORE updating state
            tracker.record_interaction(agent_a, agent_b, a_cooperates, b_cooperates)
            
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
        
        tracker.record_round_end()
    
    return tracker, tracker.get_summary()


def run_spiral_comparison(n_rounds: int = 100, n_agents: int = 50, n_seeds: int = 5) -> Dict:
    """
    Compare agent trajectories with and without consequences.
    """
    results = {'with_consequences': [], 'without_consequences': []}
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed}...")
        
        # WITH consequences (full infrastructure)
        print("  With consequences...", end=" ")
        config_with = InfrastructureConfig.full()
        tracker_with, summary_with = run_tracked_experiment(
            config_with, n_agents=n_agents, n_rounds=n_rounds, seed=seed
        )
        results['with_consequences'].append(summary_with)
        print(f"recovery={summary_with['recovery_rate']:.2%}, coop={summary_with['final_cooperation_rate']:.2%}")
        
        # WITHOUT consequences
        print("  Without consequences...", end=" ")
        config_without = InfrastructureConfig.full()
        config_without.consequences_enabled = False
        tracker_without, summary_without = run_tracked_experiment(
            config_without, n_agents=n_agents, n_rounds=n_rounds, seed=seed
        )
        results['without_consequences'].append(summary_without)
        print(f"recovery={summary_without['recovery_rate']:.2%}, coop={summary_without['final_cooperation_rate']:.2%}")
    
    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze spiral comparison results."""
    
    analysis = {}
    
    for condition in ['with_consequences', 'without_consequences']:
        summaries = results[condition]
        
        analysis[condition] = {
            'mean_recovery_rate': np.mean([s['recovery_rate'] for s in summaries]),
            'std_recovery_rate': np.std([s['recovery_rate'] for s in summaries]),
            'mean_final_cooperation': np.mean([s['final_cooperation_rate'] for s in summaries]),
            'mean_defections_per_agent': np.mean([s['avg_defections_per_agent'] for s in summaries]),
            'mean_cooperations_per_agent': np.mean([s['avg_cooperations_per_agent'] for s in summaries]),
            'mean_active_penalties': np.mean([s['avg_active_penalties'] for s in summaries]),
        }
        
        # Average the trajectory curves
        trajectories = [s['avg_reputation_trajectory_after_defection'] for s in summaries 
                       if s['avg_reputation_trajectory_after_defection']]
        if trajectories:
            min_len = min(len(t) for t in trajectories)
            truncated = [t[:min_len] for t in trajectories]
            analysis[condition]['avg_trajectory'] = np.mean(truncated, axis=0).tolist()
        else:
            analysis[condition]['avg_trajectory'] = []
        
        # Average cooperation over time
        coop_curves = [s['cooperation_over_time'] for s in summaries if s['cooperation_over_time']]
        if coop_curves:
            min_len = min(len(c) for c in coop_curves)
            truncated = [c[:min_len] for c in coop_curves]
            analysis[condition]['avg_cooperation_curve'] = np.mean(truncated, axis=0).tolist()
        else:
            analysis[condition]['avg_cooperation_curve'] = []
    
    # Statistical comparison
    recovery_with = [s['recovery_rate'] for s in results['with_consequences']]
    recovery_without = [s['recovery_rate'] for s in results['without_consequences']]
    
    # Simple t-test approximation (avoid scipy dependency)
    mean_diff = np.mean(recovery_without) - np.mean(recovery_with)
    pooled_std = np.sqrt((np.var(recovery_with) + np.var(recovery_without)) / 2)
    n = len(recovery_with)
    t_stat = mean_diff / (pooled_std * np.sqrt(2/n)) if pooled_std > 0 else 0
    
    # Approximate p-value (two-tailed)
    # For n=5, df=8, t=2.31 gives p=0.05
    p_value = 0.05 if abs(t_stat) > 2.31 else 0.1 if abs(t_stat) > 1.86 else 0.5
    
    analysis['comparison'] = {
        'recovery_rate_difference': mean_diff,
        't_statistic': t_stat,
        'p_value_approx': p_value,
        'significant': abs(t_stat) > 2.31,
        'cooperation_difference': (analysis['without_consequences']['mean_final_cooperation'] - 
                                   analysis['with_consequences']['mean_final_cooperation']),
    }
    
    return analysis


def create_visualizations(results: Dict, analysis: Dict):
    """Create visualizations for the spiral analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Cooperation over time comparison
    ax1 = axes[0, 0]
    if analysis['with_consequences'].get('avg_cooperation_curve'):
        ax1.plot(analysis['with_consequences']['avg_cooperation_curve'], 
                'r-', label='With Consequences', linewidth=2)
    if analysis['without_consequences'].get('avg_cooperation_curve'):
        ax1.plot(analysis['without_consequences']['avg_cooperation_curve'], 
                'g-', label='Without Consequences', linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Cooperation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reputation trajectory after first defection
    ax2 = axes[0, 1]
    if analysis['with_consequences'].get('avg_trajectory'):
        ax2.plot(analysis['with_consequences']['avg_trajectory'], 
                'r-', label='With Consequences', linewidth=2)
    if analysis['without_consequences'].get('avg_trajectory'):
        ax2.plot(analysis['without_consequences']['avg_trajectory'], 
                'g-', label='Without Consequences', linewidth=2)
    ax2.set_xlabel('Steps After First Defection')
    ax2.set_ylabel('Reliability (Reputation)')
    ax2.set_title('Reputation Trajectory After First Defection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Recovery rate comparison
    ax3 = axes[1, 0]
    conditions = ['With\nConsequences', 'Without\nConsequences']
    recovery_rates = [
        analysis['with_consequences']['mean_recovery_rate'],
        analysis['without_consequences']['mean_recovery_rate']
    ]
    recovery_stds = [
        analysis['with_consequences']['std_recovery_rate'],
        analysis['without_consequences']['std_recovery_rate']
    ]
    colors = ['red', 'green']
    bars = ax3.bar(conditions, recovery_rates, yerr=recovery_stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax3.set_ylabel('Recovery Rate')
    ax3.set_title('Recovery Rate After First Defection')
    ax3.set_ylim(0, 1.1)
    for bar, rate in zip(bars, recovery_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=12)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    metrics = ['Final\nCooperation', 'Avg\nDefections', 'Avg Active\nPenalties']
    with_vals = [
        analysis['with_consequences']['mean_final_cooperation'],
        analysis['with_consequences']['mean_defections_per_agent'] / 100,  # Normalize
        analysis['with_consequences']['mean_active_penalties'] / 10,  # Normalize
    ]
    without_vals = [
        analysis['without_consequences']['mean_final_cooperation'],
        analysis['without_consequences']['mean_defections_per_agent'] / 100,
        analysis['without_consequences']['mean_active_penalties'] / 10,
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, with_vals, width, label='With Consequences', color='red', alpha=0.7)
    ax4.bar(x + width/2, without_vals, width, label='Without Consequences', color='green', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Summary Metrics Comparison')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "punishment_spiral_analysis.png", dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {RESULTS_DIR / 'punishment_spiral_analysis.png'}")


def interpret_results(analysis: Dict) -> str:
    """Generate interpretation of results."""
    
    recovery_diff = analysis['comparison']['recovery_rate_difference']
    significant = analysis['comparison']['significant']
    coop_diff = analysis['comparison']['cooperation_difference']
    
    if recovery_diff > 0.1 and significant:
        return """
PUNISHMENT SPIRAL HYPOTHESIS CONFIRMED.

Agents with consequences have significantly lower recovery rates after defection.
This explains why consequences hurt cooperation: early defections cascade into 
permanent reputation damage, reducing future cooperation opportunities.

The mechanism is a positive feedback loop:
1. Agent defects (possibly by chance or due to low initial cooperation rate)
2. Partner applies penalty → agent's cooperation probability drops further
3. Agent more likely to defect again → more penalties
4. Reputation (reliability) spirals downward
5. Other agents see low reliability → less likely to cooperate with this agent
6. Agent excluded from cooperative interactions → permanent low cooperation

Without consequences, agents can recover from occasional defections because:
- No penalty accumulation
- Each interaction is relatively independent
- Reputation can recover through subsequent cooperation
"""
    elif recovery_diff > 0 and coop_diff > 0.1:
        return """
PARTIAL SUPPORT for punishment spiral hypothesis.

Recovery rates are somewhat lower with consequences, and cooperation is 
significantly lower. The punishment spiral mechanism likely contributes,
but the effect size on recovery is moderate.

Other mechanisms may also be at play:
- Risk aversion (agents become conservative to avoid punishment)
- Interaction avoidance (agents avoid risky partners entirely)
"""
    else:
        return """
PUNISHMENT SPIRAL HYPOTHESIS NOT STRONGLY SUPPORTED.

Recovery rates are similar with and without consequences.
The cooperation reduction must be explained by a different mechanism:
- Risk aversion (agents commit less to avoid punishment)
- Interaction avoidance (agents avoid risky partners)
- Retaliation cycles (tit-for-tat with punishment escalates)

Further investigation needed.
"""


def main():
    print("=" * 60)
    print("EXPERIMENT 15: Punishment Spiral Analysis")
    print("=" * 60)
    
    # Run comparison
    results = run_spiral_comparison(n_rounds=100, n_agents=50, n_seeds=5)
    
    # Analyze
    analysis = analyze_results(results)
    
    # Create visualizations
    create_visualizations(results, analysis)
    
    # Save results
    (RESULTS_DIR / "spiral_results.json").write_text(
        json.dumps(results, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    (RESULTS_DIR / "spiral_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"""
Recovery Rate After First Defection:
  With consequences:    {analysis['with_consequences']['mean_recovery_rate']:.1%} (±{analysis['with_consequences']['std_recovery_rate']:.1%})
  Without consequences: {analysis['without_consequences']['mean_recovery_rate']:.1%} (±{analysis['without_consequences']['std_recovery_rate']:.1%})
  
  Difference: {analysis['comparison']['recovery_rate_difference']:.1%}
  t-statistic: {analysis['comparison']['t_statistic']:.2f}
  Significant (p<0.05): {analysis['comparison']['significant']}

Final Cooperation Rate:
  With consequences:    {analysis['with_consequences']['mean_final_cooperation']:.1%}
  Without consequences: {analysis['without_consequences']['mean_final_cooperation']:.1%}
  Difference: {analysis['comparison']['cooperation_difference']:.1%}

Average Active Penalties:
  With consequences:    {analysis['with_consequences']['mean_active_penalties']:.2f}
  Without consequences: {analysis['without_consequences']['mean_active_penalties']:.2f}

Interpretation:
{interpret_results(analysis)}
""")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return results, analysis


if __name__ == "__main__":
    main()
