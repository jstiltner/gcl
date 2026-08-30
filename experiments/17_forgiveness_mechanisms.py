"""
Experiment 17: Forgiveness Mechanisms

Test if adding reputation recovery mechanisms fixes the retaliation cycle problem.

Previous findings:
- Punishment spirals NOT confirmed (recovery rate = 100%)
- Retaliation cycles ARE the mechanism (17+ active penalties)
- ANY consequences hurt (monotonically decreasing)
- Slope: -36% cooperation per unit penalty severity

Mechanisms to test:
1. Time decay (penalty damage fades)
2. Redemption (successes after penalty boost cooperation extra)
3. Amnesty (periodic penalty resets)
4. Second chances (first N defections are forgiven)
5. Proportional (penalties scale with history)
6. Mutual forgiveness (both parties must agree to penalize)
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gcl.population.ablation import (
    AblationAgent,
    InfrastructureConfig,
)

RESULTS_DIR = Path("results/17_forgiveness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ForgivenessMechanism(Enum):
    """Types of forgiveness mechanisms."""
    NONE = auto()              # Standard consequences (baseline)
    NO_CONSEQUENCES = auto()   # No penalties at all
    TIME_DECAY = auto()        # Penalties decay faster
    REDEMPTION = auto()        # Extra cooperation boost after penalty
    AMNESTY = auto()           # Periodic penalty resets
    SECOND_CHANCES = auto()    # First N defections forgiven
    PROPORTIONAL = auto()      # Penalties scale with defection history
    COOLING_OFF = auto()       # Penalties only apply after repeated defection


@dataclass
class ForgivingAgent(AblationAgent):
    """Agent with configurable forgiveness mechanisms."""
    
    forgiveness_mechanism: ForgivenessMechanism = ForgivenessMechanism.NONE
    mechanism_params: Dict = field(default_factory=dict)
    
    # Additional tracking for forgiveness
    defection_count: Dict[int, int] = field(default_factory=dict)  # partner -> count
    recent_cooperation: Dict[int, List[bool]] = field(default_factory=dict)  # partner -> recent history
    
    def apply_penalty_with_forgiveness(self, violator_id: int) -> bool:
        """
        Apply penalty with forgiveness mechanism.
        Returns True if penalty was actually applied.
        """
        if not self.config.consequences_enabled:
            return False
        
        mechanism = self.forgiveness_mechanism
        params = self.mechanism_params
        
        # Track defection
        self.defection_count[violator_id] = self.defection_count.get(violator_id, 0) + 1
        
        if mechanism == ForgivenessMechanism.NO_CONSEQUENCES:
            return False
        
        elif mechanism == ForgivenessMechanism.SECOND_CHANCES:
            n_chances = params.get('n_chances', 3)
            if self.defection_count[violator_id] <= n_chances:
                return False  # Forgive first N defections
        
        elif mechanism == ForgivenessMechanism.COOLING_OFF:
            # Only penalize if defected multiple times recently
            window = params.get('window', 5)
            threshold = params.get('threshold', 2)
            
            if violator_id not in self.recent_cooperation:
                self.recent_cooperation[violator_id] = []
            
            self.recent_cooperation[violator_id].append(False)  # Record defection
            self.recent_cooperation[violator_id] = self.recent_cooperation[violator_id][-window:]
            
            recent_defections = sum(1 for c in self.recent_cooperation[violator_id] if not c)
            if recent_defections < threshold:
                return False  # Not enough recent defections to trigger penalty
        
        elif mechanism == ForgivenessMechanism.PROPORTIONAL:
            # Penalty duration scales with defection history
            base_duration = params.get('base_duration', 3)
            max_duration = params.get('max_duration', 15)
            
            # More defections = longer penalty
            duration = min(base_duration + self.defection_count[violator_id], max_duration)
            self.active_penalties[violator_id] = duration
            return True
        
        # Standard penalty application
        self.active_penalties[violator_id] = self.config.penalty_duration
        return True
    
    def record_cooperation(self, partner_id: int, cooperated: bool):
        """Record cooperation for cooling-off mechanism."""
        if partner_id not in self.recent_cooperation:
            self.recent_cooperation[partner_id] = []
        self.recent_cooperation[partner_id].append(cooperated)
        # Keep only recent history
        self.recent_cooperation[partner_id] = self.recent_cooperation[partner_id][-10:]
    
    def decay_penalties_with_forgiveness(self):
        """Decay penalties with optional faster decay."""
        mechanism = self.forgiveness_mechanism
        params = self.mechanism_params
        
        if mechanism == ForgivenessMechanism.TIME_DECAY:
            decay_rate = params.get('decay_rate', 2)  # Decay 2 rounds per step
            for agent_id in list(self.active_penalties.keys()):
                self.active_penalties[agent_id] -= decay_rate
                if self.active_penalties[agent_id] <= 0:
                    del self.active_penalties[agent_id]
        else:
            # Standard decay
            for agent_id in list(self.active_penalties.keys()):
                self.active_penalties[agent_id] -= 1
                if self.active_penalties[agent_id] <= 0:
                    del self.active_penalties[agent_id]
    
    def apply_amnesty(self, round_num: int):
        """Apply amnesty if it's an amnesty round."""
        if self.forgiveness_mechanism != ForgivenessMechanism.AMNESTY:
            return
        
        amnesty_period = self.mechanism_params.get('amnesty_period', 25)
        if round_num > 0 and round_num % amnesty_period == 0:
            # Clear all penalties
            self.active_penalties.clear()
            # Optionally reduce defection counts
            for partner in self.defection_count:
                self.defection_count[partner] = max(0, self.defection_count[partner] - 1)
    
    def decide_cooperation_with_redemption(self, partner_id: int) -> bool:
        """Decide cooperation with redemption boost."""
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
        
        # Redemption: if we were recently penalized, try harder to cooperate
        if self.forgiveness_mechanism == ForgivenessMechanism.REDEMPTION:
            if self.has_penalty(partner_id):
                redemption_boost = self.mechanism_params.get('redemption_boost', 0.2)
                coop_prob += redemption_boost  # Try to redeem ourselves
        
        # Standard penalty effect (reduced if we're trying to redeem)
        if self.config.consequences_enabled and self.has_penalty(partner_id):
            if self.forgiveness_mechanism != ForgivenessMechanism.REDEMPTION:
                coop_prob -= self.config.violation_penalty
        
        coop_prob = np.clip(coop_prob, 0.0, 1.0)
        return self.rng.random() < coop_prob


def run_forgiveness_experiment(
    mechanism: ForgivenessMechanism,
    mechanism_params: Dict,
    n_agents: int = 50,
    n_rounds: int = 100,
    interactions_per_round: int = 100,
    seed: int = 42
) -> Dict:
    """Run experiment with specific forgiveness mechanism."""
    
    rng = np.random.default_rng(seed)
    
    # Create config
    config = InfrastructureConfig.full()
    
    # Create agents with forgiveness
    agents = []
    for i in range(n_agents):
        agent = ForgivingAgent(
            agent_id=i,
            config=config,
            base_cooperation_rate=rng.uniform(0.3, 0.7),
            forgiveness_mechanism=mechanism,
            mechanism_params=mechanism_params,
        )
        agents.append(agent)
    
    cooperation_rates = []
    penalty_counts = []
    
    for round_num in range(n_rounds):
        round_cooperations = 0
        round_interactions = 0
        
        # Apply amnesty at start of round
        for agent in agents:
            agent.apply_amnesty(round_num)
        
        round_penalties = sum(len(a.active_penalties) for a in agents)
        
        for _ in range(interactions_per_round):
            # Random pairing
            i, j = rng.choice(len(agents), size=2, replace=False)
            agent_a, agent_b = agents[i], agents[j]
            
            # Each agent decides whether to cooperate
            if mechanism == ForgivenessMechanism.REDEMPTION:
                a_cooperates = agent_a.decide_cooperation_with_redemption(agent_b.agent_id)
                b_cooperates = agent_b.decide_cooperation_with_redemption(agent_a.agent_id)
            else:
                a_cooperates = agent_a.decide_cooperation(agent_b.agent_id)
                b_cooperates = agent_b.decide_cooperation(agent_a.agent_id)
            
            round_cooperations += int(a_cooperates) + int(b_cooperates)
            round_interactions += 2
            
            # Record cooperation for cooling-off
            agent_a.record_cooperation(agent_b.agent_id, b_cooperates)
            agent_b.record_cooperation(agent_a.agent_id, a_cooperates)
            
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
            
            # Apply penalties for defection (with forgiveness)
            if not a_cooperates:
                agent_b.apply_penalty_with_forgiveness(agent_a.agent_id)
            if not b_cooperates:
                agent_a.apply_penalty_with_forgiveness(agent_b.agent_id)
        
        # Decay penalties (with forgiveness)
        for agent in agents:
            agent.decay_penalties_with_forgiveness()
        
        cooperation_rates.append(round_cooperations / round_interactions)
        penalty_counts.append(round_penalties / n_agents)
    
    return {
        'final_cooperation': cooperation_rates[-1],
        'avg_cooperation': np.mean(cooperation_rates),
        'cooperation_over_time': cooperation_rates,
        'avg_penalties': np.mean(penalty_counts),
        'final_penalties': penalty_counts[-1],
    }


def run_forgiveness_comparison(
    n_agents: int = 50,
    n_rounds: int = 100,
    n_seeds: int = 5
) -> Dict:
    """Compare all forgiveness mechanisms."""
    
    mechanisms = [
        (ForgivenessMechanism.NONE, {}, "Standard consequences"),
        (ForgivenessMechanism.NO_CONSEQUENCES, {}, "No consequences"),
        (ForgivenessMechanism.TIME_DECAY, {'decay_rate': 2}, "Fast decay (2x)"),
        (ForgivenessMechanism.TIME_DECAY, {'decay_rate': 3}, "Very fast decay (3x)"),
        (ForgivenessMechanism.REDEMPTION, {'redemption_boost': 0.2}, "Redemption (+20%)"),
        (ForgivenessMechanism.REDEMPTION, {'redemption_boost': 0.3}, "Strong redemption (+30%)"),
        (ForgivenessMechanism.AMNESTY, {'amnesty_period': 25}, "Amnesty (every 25 rounds)"),
        (ForgivenessMechanism.AMNESTY, {'amnesty_period': 10}, "Frequent amnesty (every 10)"),
        (ForgivenessMechanism.SECOND_CHANCES, {'n_chances': 2}, "2 chances"),
        (ForgivenessMechanism.SECOND_CHANCES, {'n_chances': 5}, "5 chances"),
        (ForgivenessMechanism.COOLING_OFF, {'window': 5, 'threshold': 2}, "Cooling off (2/5)"),
        (ForgivenessMechanism.COOLING_OFF, {'window': 3, 'threshold': 2}, "Strict cooling (2/3)"),
        (ForgivenessMechanism.PROPORTIONAL, {'base_duration': 2, 'max_duration': 10}, "Proportional"),
    ]
    
    results = {}
    
    for mechanism, params, name in mechanisms:
        print(f"\nTesting: {name}")
        
        seed_results = []
        for seed in range(n_seeds):
            result = run_forgiveness_experiment(
                mechanism=mechanism,
                mechanism_params=params,
                n_agents=n_agents,
                n_rounds=n_rounds,
                seed=seed
            )
            seed_results.append(result)
        
        results[name] = {
            'mechanism': mechanism.name,
            'params': params,
            'mean_cooperation': np.mean([r['final_cooperation'] for r in seed_results]),
            'std_cooperation': np.std([r['final_cooperation'] for r in seed_results]),
            'mean_avg_cooperation': np.mean([r['avg_cooperation'] for r in seed_results]),
            'mean_penalties': np.mean([r['avg_penalties'] for r in seed_results]),
            'seeds': seed_results,
        }
        
        print(f"  Cooperation: {results[name]['mean_cooperation']:.1%} (±{results[name]['std_cooperation']:.1%})")
        print(f"  Avg penalties: {results[name]['mean_penalties']:.2f}")
    
    return results


def analyze_forgiveness_results(results: Dict) -> Dict:
    """Find best forgiveness mechanism."""
    
    # Extract key metrics
    summary = []
    for name, data in results.items():
        summary.append({
            'name': name,
            'cooperation': data['mean_cooperation'],
            'std': data['std_cooperation'],
            'penalties': data['mean_penalties'],
        })
    
    # Sort by cooperation
    summary.sort(key=lambda x: -x['cooperation'])
    
    # Get baselines
    no_consequences = next((s for s in summary if s['name'] == 'No consequences'), None)
    standard = next((s for s in summary if s['name'] == 'Standard consequences'), None)
    
    no_consequences_coop = no_consequences['cooperation'] if no_consequences else 0
    standard_coop = standard['cooperation'] if standard else 0
    
    # Find best mechanism (excluding no consequences)
    best_with_consequences = next(
        (s for s in summary if s['name'] not in ['No consequences', 'Standard consequences']),
        None
    )
    
    return {
        'ranking': summary,
        'no_consequences_baseline': no_consequences_coop,
        'standard_consequences': standard_coop,
        'best_mechanism': best_with_consequences['name'] if best_with_consequences else None,
        'best_cooperation': best_with_consequences['cooperation'] if best_with_consequences else 0,
        'improvement_over_standard': (best_with_consequences['cooperation'] - standard_coop) if best_with_consequences else 0,
        'gap_to_no_consequences': (no_consequences_coop - best_with_consequences['cooperation']) if best_with_consequences else 0,
    }


def create_visualizations(results: Dict, analysis: Dict):
    """Create visualizations for forgiveness comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ranking = analysis['ranking']
    names = [r['name'] for r in ranking]
    cooperations = [r['cooperation'] for r in ranking]
    stds = [r['std'] for r in ranking]
    penalties = [r['penalties'] for r in ranking]
    
    # Plot 1: Cooperation ranking
    ax1 = axes[0, 0]
    colors = ['green' if n == 'No consequences' else 'red' if n == 'Standard consequences' else 'blue' 
              for n in names]
    bars = ax1.barh(range(len(names)), cooperations, xerr=stds, color=colors, alpha=0.7, capsize=3)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Cooperation Rate')
    ax1.set_title('Forgiveness Mechanisms Ranked by Cooperation')
    ax1.axvline(x=analysis['no_consequences_baseline'], color='green', linestyle='--', alpha=0.5)
    ax1.axvline(x=analysis['standard_consequences'], color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Cooperation vs Penalties scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(penalties, cooperations, s=100, c=range(len(names)), cmap='viridis')
    for i, name in enumerate(names):
        ax2.annotate(name[:15], (penalties[i], cooperations[i]), fontsize=7, 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Average Active Penalties')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('Cooperation vs Penalty Load')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Improvement over standard
    ax3 = axes[1, 0]
    improvements = [c - analysis['standard_consequences'] for c in cooperations]
    colors = ['green' if i > 0 else 'red' for i in improvements]
    ax3.barh(range(len(names)), improvements, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel('Improvement over Standard Consequences')
    ax3.set_title('Relative Performance')
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 4: Summary comparison
    ax4 = axes[1, 1]
    key_mechanisms = ['Standard consequences', 'No consequences', 
                      analysis['best_mechanism'] if analysis['best_mechanism'] else 'N/A']
    key_coops = [
        analysis['standard_consequences'],
        analysis['no_consequences_baseline'],
        analysis['best_cooperation']
    ]
    colors = ['red', 'green', 'blue']
    bars = ax4.bar(range(len(key_mechanisms)), key_coops, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(key_mechanisms)))
    ax4.set_xticklabels(['Standard', 'No Consequences', f'Best:\n{analysis["best_mechanism"][:20]}'], fontsize=9)
    ax4.set_ylabel('Cooperation Rate')
    ax4.set_title('Key Comparison')
    
    for bar, coop in zip(bars, key_coops):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{coop:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "forgiveness_comparison.png", dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {RESULTS_DIR / 'forgiveness_comparison.png'}")


def main():
    print("=" * 60)
    print("EXPERIMENT 17: Forgiveness Mechanisms")
    print("=" * 60)
    
    results = run_forgiveness_comparison(n_agents=50, n_rounds=100, n_seeds=5)
    analysis = analyze_forgiveness_results(results)
    
    # Create visualizations
    create_visualizations(results, analysis)
    
    # Save results
    (RESULTS_DIR / "forgiveness_results.json").write_text(
        json.dumps(results, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    (RESULTS_DIR / "forgiveness_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    
    print("\n" + "=" * 60)
    print("MECHANISM RANKING (by cooperation rate)")
    print("=" * 60)
    
    for rank, item in enumerate(analysis['ranking'], 1):
        marker = ""
        if item['name'] == 'No consequences':
            marker = " [BASELINE]"
        elif item['name'] == 'Standard consequences':
            marker = " [ORIGINAL]"
        elif item['name'] == analysis['best_mechanism']:
            marker = " [BEST WITH CONSEQUENCES]"
        print(f"  {rank:2d}. {item['name']:<30} {item['cooperation']:.1%} (±{item['std']:.1%}){marker}")
    
    print(f"""
\nKEY COMPARISONS:
  No consequences baseline:  {analysis['no_consequences_baseline']:.1%}
  Standard consequences:     {analysis['standard_consequences']:.1%}
  Best forgiveness:          {analysis['best_cooperation']:.1%} ({analysis['best_mechanism']})
  
  Improvement over standard: {analysis['improvement_over_standard']:.1%}
  Gap to no-consequences:    {analysis['gap_to_no_consequences']:.1%}
""")
    
    # Interpretation
    if analysis['gap_to_no_consequences'] < 0.05:
        print("✓ FORGIVENESS CAN RECOVER MOST OF THE COOPERATION LOSS!")
        print("  The best mechanism nearly matches no-consequences performance.")
    elif analysis['improvement_over_standard'] > 0.1:
        print("◐ FORGIVENESS HELPS SIGNIFICANTLY BUT DOESN'T FULLY SOLVE THE PROBLEM.")
        print("  Consider combining multiple mechanisms or further tuning.")
    elif analysis['improvement_over_standard'] > 0.05:
        print("◐ FORGIVENESS PROVIDES MODEST IMPROVEMENT.")
        print("  Some benefit, but consequences still hurt overall.")
    else:
        print("✗ FORGIVENESS MECHANISMS DON'T SIGNIFICANTLY HELP.")
        print("  The retaliation cycle problem may be fundamental to this design.")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return results, analysis


if __name__ == "__main__":
    main()
