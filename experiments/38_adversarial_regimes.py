"""
Experiment 38: Adversarial Regime Changes

Tests GCL's advantage when regimes are FUNDAMENTALLY different,
not just parameter shifts. MARL's learned policies become invalid.

Regime types:
1. Cooperate vs Compete: Volunteering helps vs hurts
2. Majority vs Minority: Need many volunteers vs need few
3. Inverse Capability: High capability helps vs hurts
4. Reward Inversion: Success = reward vs Success = penalty

Hypothesis: GCL will show LARGE advantage because it adapts
based on current state, not learned policies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class AdversarialRegime:
    """Configuration for an adversarial regime."""
    name: str
    volunteer_helps: bool = True      # Does volunteering help the task?
    need_many: bool = False           # Need many volunteers (majority)?
    capability_positive: bool = True  # Does high capability help?
    reward_positive: bool = True      # Is success rewarded (vs penalized)?
    
    def describe(self) -> str:
        parts = []
        parts.append("volunteer+" if self.volunteer_helps else "volunteer-")
        parts.append("many" if self.need_many else "few")
        parts.append("cap+" if self.capability_positive else "cap-")
        parts.append("reward+" if self.reward_positive else "reward-")
        return "_".join(parts)


# Define adversarial regime pairs
ADVERSARIAL_PAIRS = {
    "cooperate_compete": (
        AdversarialRegime("cooperate", volunteer_helps=True),
        AdversarialRegime("compete", volunteer_helps=False),
    ),
    "majority_minority": (
        AdversarialRegime("majority", need_many=True),
        AdversarialRegime("minority", need_many=False),
    ),
    "capability_inversion": (
        AdversarialRegime("cap_positive", capability_positive=True),
        AdversarialRegime("cap_negative", capability_positive=False),
    ),
    "reward_inversion": (
        AdversarialRegime("reward_positive", reward_positive=True),
        AdversarialRegime("reward_negative", reward_positive=False),
    ),
}


class AdversarialEnvironment:
    """Environment with adversarial regime changes."""
    
    def __init__(self, n_agents: int, regime_type: str = "cooperate_compete"):
        self.n_agents = n_agents
        self.regime_type = regime_type
        self.regime_pair = ADVERSARIAL_PAIRS[regime_type]
        self.current_regime_idx = 0
        self.current_regime = self.regime_pair[0]
        
        # Agent capabilities
        self.capabilities = np.random.uniform(0.3, 0.9, n_agents)
    
    def switch_regime(self):
        """Switch to the other regime."""
        self.current_regime_idx = 1 - self.current_regime_idx
        self.current_regime = self.regime_pair[self.current_regime_idx]
    
    def execute_task(self, actions: List[int]) -> float:
        """Execute task under current regime."""
        volunteers = [i for i, a in enumerate(actions) if a == 1]
        n_volunteers = len(volunteers)
        
        regime = self.current_regime
        
        # Determine base success
        if regime.volunteer_helps:
            # Normal: volunteers help
            if n_volunteers == 0:
                base_success = 0.0
            else:
                best_cap = max(self.capabilities[i] for i in volunteers)
                if regime.capability_positive:
                    base_success = best_cap * 0.9
                else:
                    # Inverse: lower capability is better
                    worst_cap = min(self.capabilities[i] for i in volunteers)
                    base_success = (1 - worst_cap) * 0.9
        else:
            # Compete: volunteers HURT (e.g., too many cooks)
            if n_volunteers == 0:
                base_success = 0.8  # No volunteers = good
            else:
                # More volunteers = worse
                penalty = n_volunteers * 0.1
                base_success = max(0, 0.8 - penalty)
        
        # Majority/minority modifier
        if regime.need_many:
            # Need majority to volunteer
            if n_volunteers >= self.n_agents // 2:
                base_success *= 1.2
            else:
                base_success *= 0.5
        else:
            # Need minority (few volunteers)
            if n_volunteers <= 3:
                base_success *= 1.2
            else:
                base_success *= 0.5
        
        # Determine success
        success = random.random() < min(1.0, base_success)
        
        # Reward based on regime
        if regime.reward_positive:
            return 1.0 if success else 0.0
        else:
            # Inverse: success is penalized
            return 0.0 if success else 1.0


class AdaptiveGCLAgent:
    """GCL agent that adapts to regime based on recent feedback."""
    
    def __init__(self, n_agents: int, env: AdversarialEnvironment):
        self.n_agents = n_agents
        self.env = env
        self.episode_rewards = []
        
        # Track what works in current regime
        self.volunteer_success = 0.5  # Running estimate
        self.many_success = 0.5
        self.learning_rate = 0.3
    
    def get_actions(self) -> List[int]:
        """Adaptive self-selection based on recent feedback."""
        actions = []
        
        # Decide volunteering strategy based on recent success
        volunteer_prob = 0.5 + 0.3 * (self.volunteer_success - 0.5)
        target_volunteers = 3 if self.many_success < 0.5 else self.n_agents // 2
        
        for i in range(self.n_agents):
            cap = self.env.capabilities[i]
            
            # Base probability from capability
            base_prob = cap * volunteer_prob
            
            # Adjust based on how many we need
            if len([a for a in actions if a == 1]) < target_volunteers:
                base_prob *= 1.2
            else:
                base_prob *= 0.5
            
            volunteer = random.random() < min(0.9, base_prob)
            actions.append(1 if volunteer else 0)
        
        return actions
    
    def step(self) -> float:
        """Execute one episode and learn from feedback."""
        actions = self.get_actions()
        reward = self.env.execute_task(actions)
        
        # Update estimates based on outcome
        n_volunteers = sum(actions)
        
        if reward > 0.5:
            # Success - what worked?
            if n_volunteers > 0:
                self.volunteer_success += self.learning_rate * (1 - self.volunteer_success)
            else:
                self.volunteer_success += self.learning_rate * (0 - self.volunteer_success)
            
            if n_volunteers >= self.n_agents // 2:
                self.many_success += self.learning_rate * (1 - self.many_success)
            else:
                self.many_success += self.learning_rate * (0 - self.many_success)
        else:
            # Failure - what didn't work?
            if n_volunteers > 0:
                self.volunteer_success += self.learning_rate * (0 - self.volunteer_success)
            else:
                self.volunteer_success += self.learning_rate * (1 - self.volunteer_success)
            
            if n_volunteers >= self.n_agents // 2:
                self.many_success += self.learning_rate * (0 - self.many_success)
            else:
                self.many_success += self.learning_rate * (1 - self.many_success)
        
        self.episode_rewards.append(reward)
        return reward
    
    def train(self):
        pass
    
    def on_regime_change(self):
        """Reset estimates on regime change."""
        # Partial reset - keep some memory
        self.volunteer_success = 0.5
        self.many_success = 0.5


class IQLAgent:
    """Independent Q-Learning agent."""
    
    def __init__(self, n_agents: int, env: AdversarialEnvironment):
        self.n_agents = n_agents
        self.env = env
        
        # Simple state: just track recent reward
        self.n_states = 5
        self.q_tables = [np.zeros((self.n_states, 2)) for _ in range(n_agents)]
        
        self.lr = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.recent_reward = 0.5
        self.episode_rewards = []
    
    def get_state(self) -> int:
        """Discretize recent reward into state."""
        return min(int(self.recent_reward * self.n_states), self.n_states - 1)
    
    def get_actions(self) -> List[int]:
        state = self.get_state()
        actions = []
        
        for i in range(self.n_agents):
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                action = int(np.argmax(self.q_tables[i][state]))
            actions.append(action)
        
        return actions
    
    def step(self) -> float:
        state = self.get_state()
        actions = self.get_actions()
        reward = self.env.execute_task(actions)
        
        # Update Q-values
        volunteers = [i for i, a in enumerate(actions) if a == 1]
        for i in range(self.n_agents):
            old_q = self.q_tables[i][state, actions[i]]
            if actions[i] == 1:
                ind_reward = reward if i in volunteers else 0
            else:
                ind_reward = reward if not volunteers else 0
            self.q_tables[i][state, actions[i]] = old_q + self.lr * (ind_reward - old_q)
        
        # Update recent reward
        self.recent_reward = 0.9 * self.recent_reward + 0.1 * reward
        
        self.episode_rewards.append(reward)
        return reward
    
    def train(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def on_regime_change(self):
        """Reset epsilon but keep Q-tables (this is the problem!)."""
        self.epsilon = min(1.0, self.epsilon + 0.3)


class QMIXAgent:
    """QMIX agent."""
    
    def __init__(self, n_agents: int, env: AdversarialEnvironment):
        self.n_agents = n_agents
        self.env = env
        
        self.n_states = 5
        self.q_tables = [np.zeros((self.n_states, 2)) for _ in range(n_agents)]
        self.mixing_weights = np.ones(n_agents) / n_agents
        
        self.lr = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.recent_reward = 0.5
        self.episode_rewards = []
    
    def get_state(self) -> int:
        return min(int(self.recent_reward * self.n_states), self.n_states - 1)
    
    def get_actions(self) -> List[int]:
        state = self.get_state()
        actions = []
        
        for i in range(self.n_agents):
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                action = int(np.argmax(self.q_tables[i][state]))
            actions.append(action)
        
        return actions
    
    def step(self) -> float:
        state = self.get_state()
        actions = self.get_actions()
        reward = self.env.execute_task(actions)
        
        # QMIX update
        q_values = np.array([self.q_tables[i][state, actions[i]] for i in range(self.n_agents)])
        q_tot = np.sum(self.mixing_weights * q_values)
        td_error = reward - q_tot
        
        for i in range(self.n_agents):
            self.q_tables[i][state, actions[i]] += self.lr * td_error * self.mixing_weights[i]
        
        self.recent_reward = 0.9 * self.recent_reward + 0.1 * reward
        
        self.episode_rewards.append(reward)
        return reward
    
    def train(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def on_regime_change(self):
        self.epsilon = min(1.0, self.epsilon + 0.3)


def run_adversarial_experiment(
    method_class,
    regime_type: str,
    change_frequency: int,
    n_episodes: int,
    n_agents: int,
    seed: int
) -> Dict[str, Any]:
    """Run a single adversarial regime experiment."""
    random.seed(seed)
    np.random.seed(seed)
    
    env = AdversarialEnvironment(n_agents, regime_type)
    agent = method_class(n_agents, env)
    
    results = {
        "episode_rewards": [],
        "regime_indices": [],
        "regime_change_episodes": [],
    }
    
    for episode in range(n_episodes):
        if episode > 0 and episode % change_frequency == 0:
            env.switch_regime()
            agent.on_regime_change()
            results["regime_change_episodes"].append(episode)
        
        reward = agent.step()
        results["episode_rewards"].append(reward)
        results["regime_indices"].append(env.current_regime_idx)
        
        agent.train()
    
    return results


def compute_metrics(results: Dict[str, Any], change_frequency: int) -> Dict[str, float]:
    """Compute metrics from results."""
    rewards = np.array(results["episode_rewards"])
    
    avg_coop = np.mean(rewards)
    
    # Performance by regime
    regime_0_mask = np.array(results["regime_indices"]) == 0
    regime_1_mask = ~regime_0_mask
    coop_regime_0 = np.mean(rewards[regime_0_mask]) if np.any(regime_0_mask) else 0
    coop_regime_1 = np.mean(rewards[regime_1_mask]) if np.any(regime_1_mask) else 0
    
    # Recovery time
    window = 20
    recovery_times = []
    for change_ep in results["regime_change_episodes"]:
        if change_ep < window:
            continue
        pre_change = np.mean(rewards[max(0, change_ep - window):change_ep])
        threshold = 0.6 * pre_change  # Lower threshold for adversarial
        
        for i in range(change_ep, min(change_ep + change_frequency, len(rewards))):
            post_window = rewards[max(change_ep, i - window//2):i + window//2]
            if len(post_window) > 0 and np.mean(post_window) >= threshold:
                recovery_times.append(i - change_ep)
                break
        else:
            recovery_times.append(change_frequency)
    
    avg_recovery = np.mean(recovery_times) if recovery_times else change_frequency
    
    # Worst case
    worst_case = 1.0
    for i in range(0, len(rewards) - window, window // 2):
        window_coop = np.mean(rewards[i:i + window])
        worst_case = min(worst_case, window_coop)
    
    return {
        "avg_cooperation": float(avg_coop),
        "coop_regime_0": float(coop_regime_0),
        "coop_regime_1": float(coop_regime_1),
        "avg_recovery_time": float(avg_recovery),
        "worst_case": float(worst_case),
    }


def run_full_experiment(
    regime_type: str = "cooperate_compete",
    change_frequencies: List[int] = None,
    n_episodes: int = 2000,
    n_seeds: int = 10,
    n_agents: int = 30
) -> Dict[str, Any]:
    """Run full adversarial experiment."""
    if change_frequencies is None:
        change_frequencies = [50, 100, 200]
    
    print("=" * 70)
    print(f"EXPERIMENT 38: ADVERSARIAL REGIME CHANGES ({regime_type})")
    print("=" * 70)
    print(f"Episodes: {n_episodes}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    methods = {
        "gcl": AdaptiveGCLAgent,
        "iql": IQLAgent,
        "qmix": QMIXAgent,
    }
    
    all_results = {}
    
    for freq in change_frequencies:
        print(f"\n--- Change frequency: {freq} episodes ---")
        all_results[freq] = {}
        
        for method_name, method_class in methods.items():
            print(f"Running {method_name}...", end=" ", flush=True)
            
            seed_metrics = []
            for seed in range(n_seeds):
                results = run_adversarial_experiment(
                    method_class, regime_type, freq, n_episodes, n_agents, seed
                )
                metrics = compute_metrics(results, freq)
                seed_metrics.append(metrics)
            
            aggregated = {}
            for key in seed_metrics[0].keys():
                values = [m[key] for m in seed_metrics]
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            
            all_results[freq][method_name] = aggregated
            print(f"coop={aggregated['avg_cooperation']['mean']:.3f}")
    
    return all_results


def print_results(results: Dict[str, Any], regime_type: str):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print(f"RESULTS: Adversarial Regime Changes ({regime_type})")
    print("=" * 70)
    
    frequencies = sorted(results.keys())
    
    print("\n--- Average Cooperation by Change Frequency ---")
    print(f"{'Frequency':>10} {'GCL':>10} {'IQL':>10} {'QMIX':>10} {'GCL Adv':>10}")
    print("-" * 55)
    
    for f in frequencies:
        gcl = results[f]["gcl"]["avg_cooperation"]["mean"]
        iql = results[f]["iql"]["avg_cooperation"]["mean"]
        qmix = results[f]["qmix"]["avg_cooperation"]["mean"]
        best_marl = max(iql, qmix)
        adv = gcl - best_marl
        print(f"{f:>10} {gcl:>10.3f} {iql:>10.3f} {qmix:>10.3f} {adv:>+10.3f}")
    
    print("\n--- Recovery Time (episodes) ---")
    print(f"{'Frequency':>10} {'GCL':>10} {'IQL':>10} {'QMIX':>10}")
    print("-" * 45)
    
    for f in frequencies:
        gcl = results[f]["gcl"]["avg_recovery_time"]["mean"]
        iql = results[f]["iql"]["avg_recovery_time"]["mean"]
        qmix = results[f]["qmix"]["avg_recovery_time"]["mean"]
        print(f"{f:>10} {gcl:>10.1f} {iql:>10.1f} {qmix:>10.1f}")
    
    print("\n--- Worst-Case Cooperation ---")
    print(f"{'Frequency':>10} {'GCL':>10} {'IQL':>10} {'QMIX':>10}")
    print("-" * 45)
    
    for f in frequencies:
        gcl = results[f]["gcl"]["worst_case"]["mean"]
        iql = results[f]["iql"]["worst_case"]["mean"]
        qmix = results[f]["qmix"]["worst_case"]["mean"]
        print(f"{f:>10} {gcl:>10.3f} {iql:>10.3f} {qmix:>10.3f}")
    
    # Calculate average advantage
    advantages = []
    for f in frequencies:
        gcl = results[f]["gcl"]["avg_cooperation"]["mean"]
        best_marl = max(results[f]["iql"]["avg_cooperation"]["mean"],
                       results[f]["qmix"]["avg_cooperation"]["mean"])
        advantages.append(gcl - best_marl)
    
    avg_advantage = np.mean(advantages)
    
    print("\n--- KEY FINDINGS ---")
    print(f"✓ Average GCL advantage: {avg_advantage:+.3f} ({avg_advantage*100:+.1f}%)")
    
    if avg_advantage > 0.05:
        print("✓ GCL shows LARGE advantage in adversarial regimes")
    elif avg_advantage > 0:
        print("✓ GCL shows moderate advantage in adversarial regimes")
    else:
        print("○ MARL performs better in adversarial regimes")
    
    print("\n--- THEORETICAL INSIGHT ---")
    print("In adversarial regimes, MARL's learned policies become HARMFUL.")
    print("GCL adapts based on current feedback, not past learning.")


def main():
    # Test all adversarial regime types
    regime_types = ["cooperate_compete", "majority_minority", "capability_inversion", "reward_inversion"]
    
    all_type_results = {}
    
    for regime_type in regime_types:
        results = run_full_experiment(
            regime_type=regime_type,
            change_frequencies=[50, 100, 200],
            n_episodes=2000,
            n_seeds=10,
            n_agents=30
        )
        all_type_results[regime_type] = results
        print_results(results, regime_type)
    
    # Summary across all types
    print("\n" + "=" * 70)
    print("SUMMARY: GCL Advantage Across All Adversarial Regime Types")
    print("=" * 70)
    
    print(f"\n{'Regime Type':>25} {'GCL Advantage':>15}")
    print("-" * 45)
    
    for regime_type in regime_types:
        results = all_type_results[regime_type]
        advantages = []
        for f in results.keys():
            gcl = results[f]["gcl"]["avg_cooperation"]["mean"]
            best_marl = max(results[f]["iql"]["avg_cooperation"]["mean"],
                          results[f]["qmix"]["avg_cooperation"]["mean"])
            advantages.append(gcl - best_marl)
        avg_adv = np.mean(advantages)
        print(f"{regime_type:>25} {avg_adv:>+15.3f}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    with open("results/experiment_38_adversarial.json", "w") as f:
        json.dump(all_type_results, f, indent=2)
    
    print("\nResults saved to results/experiment_38_adversarial.json")


if __name__ == "__main__":
    main()
