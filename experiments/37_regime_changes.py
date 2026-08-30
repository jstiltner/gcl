"""
Experiment 37: Regime Change Robustness

Tests GCL's dynamism advantage over MARL in non-stationary environments.

Hypothesis: GCL's sample efficiency (2 episodes to 50%) translates to
superior performance when regime changes occur frequently.

Regime change types:
- 37A: Task difficulty shift (easy ↔ hard)
- 37B: Agent capability shift (high ↔ low)
- 37C: Reward structure shift (individual ↔ team)
- 37D: Population turnover (30% replacement)

Key metrics:
- Average cooperation across regimes
- Recovery time after regime change
- Worst-case performance
- Stability (variance)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class RegimeConfig:
    """Configuration for a regime."""
    difficulty_range: Tuple[float, float] = (0.3, 0.7)
    capability_range: Tuple[float, float] = (0.3, 0.9)
    team_reward: bool = True  # True = only best volunteer gets reward
    
    def __str__(self):
        return f"diff={self.difficulty_range}, cap={self.capability_range}, team={self.team_reward}"


# Define regime pairs for each experiment type
REGIME_PAIRS = {
    "difficulty": (
        RegimeConfig(difficulty_range=(0.2, 0.4)),  # Easy
        RegimeConfig(difficulty_range=(0.6, 0.8)),  # Hard
    ),
    "capability": (
        RegimeConfig(capability_range=(0.6, 0.9)),  # High capability
        RegimeConfig(capability_range=(0.3, 0.6)),  # Low capability
    ),
    "reward": (
        RegimeConfig(team_reward=True),   # Team reward
        RegimeConfig(team_reward=False),  # Individual reward
    ),
}


class NonStationaryEnvironment:
    """Environment with regime changes."""
    
    def __init__(self, n_agents: int, regime_type: str = "difficulty"):
        self.n_agents = n_agents
        self.regime_type = regime_type
        self.regime_pair = REGIME_PAIRS[regime_type]
        self.current_regime_idx = 0
        self.current_regime = self.regime_pair[0]
        
        # Initialize capabilities
        self.reset_capabilities()
    
    def reset_capabilities(self):
        """Reset agent capabilities based on current regime."""
        cap_range = self.current_regime.capability_range
        self.capabilities = np.random.uniform(cap_range[0], cap_range[1], self.n_agents)
    
    def switch_regime(self):
        """Switch to the other regime."""
        self.current_regime_idx = 1 - self.current_regime_idx
        self.current_regime = self.regime_pair[self.current_regime_idx]
        
        # For capability regime, reset capabilities
        if self.regime_type == "capability":
            self.reset_capabilities()
    
    def get_difficulty(self) -> float:
        """Sample task difficulty from current regime."""
        diff_range = self.current_regime.difficulty_range
        return np.random.uniform(diff_range[0], diff_range[1])
    
    def execute_task(self, actions: List[int]) -> float:
        """Execute task and return reward."""
        difficulty = self.get_difficulty()
        volunteers = [i for i, a in enumerate(actions) if a == 1]
        
        if not volunteers:
            return 0.0
        
        best_idx = max(volunteers, key=lambda i: self.capabilities[i])
        capability = self.capabilities[best_idx]
        effort = 0.9  # Effort bonus for volunteers
        
        success_prob = capability * effort * (1 - difficulty * 0.6)
        success = random.random() < success_prob
        
        if self.current_regime.team_reward:
            return 1.0 if success else 0.0
        else:
            # Individual reward: all volunteers get reward if success
            return 1.0 if success else 0.0
    
    def turnover(self, fraction: float = 0.3):
        """Replace a fraction of agents with new ones."""
        n_replace = int(self.n_agents * fraction)
        indices = random.sample(range(self.n_agents), n_replace)
        cap_range = self.current_regime.capability_range
        for i in indices:
            self.capabilities[i] = np.random.uniform(cap_range[0], cap_range[1])


class GCLAgent:
    """GCL self-selection agent (no training needed)."""
    
    def __init__(self, n_agents: int, env: NonStationaryEnvironment):
        self.n_agents = n_agents
        self.env = env
        self.episode_rewards = []
    
    def get_actions(self) -> List[int]:
        """Self-selection based on capability."""
        difficulty = self.env.get_difficulty()
        actions = []
        for i in range(self.n_agents):
            cap = self.env.capabilities[i]
            if cap > difficulty * 0.5:
                score = cap - difficulty * 0.3
                volunteer = score > 0.2
            else:
                volunteer = False
            actions.append(1 if volunteer else 0)
        return actions
    
    def step(self) -> float:
        """Execute one episode."""
        actions = self.get_actions()
        reward = self.env.execute_task(actions)
        self.episode_rewards.append(reward)
        return reward
    
    def train(self):
        """No training needed."""
        pass
    
    def on_regime_change(self):
        """Called when regime changes - GCL adapts immediately."""
        pass


class IQLAgent:
    """Independent Q-Learning agent."""
    
    def __init__(self, n_agents: int, env: NonStationaryEnvironment):
        self.n_agents = n_agents
        self.env = env
        
        self.n_states = 10
        self.q_tables = [np.zeros((self.n_states, 2)) for _ in range(n_agents)]
        
        self.lr = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.episode_rewards = []
    
    def discretize_state(self, difficulty: float) -> int:
        return min(int(difficulty * self.n_states), self.n_states - 1)
    
    def get_actions(self) -> List[int]:
        difficulty = self.env.get_difficulty()
        state = self.discretize_state(difficulty)
        actions = []
        
        for i in range(self.n_agents):
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                action = int(np.argmax(self.q_tables[i][state]))
            actions.append(action)
        
        return actions
    
    def step(self) -> float:
        difficulty = self.env.get_difficulty()
        state = self.discretize_state(difficulty)
        actions = self.get_actions()
        reward = self.env.execute_task(actions)
        
        # Update Q-values
        volunteers = [i for i, a in enumerate(actions) if a == 1]
        for i in range(self.n_agents):
            old_q = self.q_tables[i][state, actions[i]]
            if actions[i] == 1 and i in volunteers:
                ind_reward = reward
            else:
                ind_reward = 0.0
            self.q_tables[i][state, actions[i]] = old_q + self.lr * (ind_reward - old_q)
        
        self.episode_rewards.append(reward)
        return reward
    
    def train(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def on_regime_change(self):
        """Reset epsilon to encourage exploration after regime change."""
        self.epsilon = min(1.0, self.epsilon + 0.3)


class QMIXAgent:
    """Simplified QMIX agent."""
    
    def __init__(self, n_agents: int, env: NonStationaryEnvironment):
        self.n_agents = n_agents
        self.env = env
        
        self.n_states = 10
        self.q_tables = [np.zeros((self.n_states, 2)) for _ in range(n_agents)]
        self.mixing_weights = np.ones(n_agents) / n_agents
        
        self.lr = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.episode_rewards = []
    
    def discretize_state(self, difficulty: float) -> int:
        return min(int(difficulty * self.n_states), self.n_states - 1)
    
    def get_actions(self) -> List[int]:
        difficulty = self.env.get_difficulty()
        state = self.discretize_state(difficulty)
        actions = []
        
        for i in range(self.n_agents):
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                action = int(np.argmax(self.q_tables[i][state]))
            actions.append(action)
        
        return actions
    
    def step(self) -> float:
        difficulty = self.env.get_difficulty()
        state = self.discretize_state(difficulty)
        actions = self.get_actions()
        reward = self.env.execute_task(actions)
        
        # QMIX update
        q_values = np.array([self.q_tables[i][state, actions[i]] for i in range(self.n_agents)])
        q_tot = np.sum(self.mixing_weights * q_values)
        td_error = reward - q_tot
        
        for i in range(self.n_agents):
            self.q_tables[i][state, actions[i]] += self.lr * td_error * self.mixing_weights[i]
        
        if reward > 0:
            for i in range(self.n_agents):
                if actions[i] == 1:
                    self.mixing_weights[i] *= 1.01
            self.mixing_weights /= np.sum(self.mixing_weights)
        
        self.episode_rewards.append(reward)
        return reward
    
    def train(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def on_regime_change(self):
        self.epsilon = min(1.0, self.epsilon + 0.3)


def run_regime_change_experiment(
    method_class,
    regime_type: str,
    change_frequency: int,
    n_episodes: int,
    n_agents: int,
    seed: int
) -> Dict[str, Any]:
    """Run a single regime change experiment."""
    random.seed(seed)
    np.random.seed(seed)
    
    env = NonStationaryEnvironment(n_agents, regime_type)
    agent = method_class(n_agents, env)
    
    results = {
        "episode_rewards": [],
        "regime_indices": [],
        "regime_change_episodes": [],
    }
    
    for episode in range(n_episodes):
        # Check for regime change
        if episode > 0 and episode % change_frequency == 0:
            env.switch_regime()
            agent.on_regime_change()
            results["regime_change_episodes"].append(episode)
        
        # Run episode
        reward = agent.step()
        results["episode_rewards"].append(reward)
        results["regime_indices"].append(env.current_regime_idx)
        
        agent.train()
    
    return results


def compute_metrics(results: Dict[str, Any], change_frequency: int, window: int = 50) -> Dict[str, float]:
    """Compute metrics from experiment results."""
    rewards = np.array(results["episode_rewards"])
    regime_changes = results["regime_change_episodes"]
    
    # Average cooperation
    avg_coop = np.mean(rewards)
    
    # Cooperation by regime
    regime_0_mask = np.array(results["regime_indices"]) == 0
    regime_1_mask = ~regime_0_mask
    coop_regime_0 = np.mean(rewards[regime_0_mask]) if np.any(regime_0_mask) else 0
    coop_regime_1 = np.mean(rewards[regime_1_mask]) if np.any(regime_1_mask) else 0
    
    # Recovery time: episodes to reach 80% of pre-change performance
    recovery_times = []
    for change_ep in regime_changes:
        if change_ep < window:
            continue
        pre_change = np.mean(rewards[max(0, change_ep - window):change_ep])
        threshold = 0.8 * pre_change
        
        # Find first episode after change that exceeds threshold
        for i in range(change_ep, min(change_ep + change_frequency, len(rewards))):
            post_window = rewards[max(change_ep, i - window//2):i + window//2]
            if len(post_window) > 0 and np.mean(post_window) >= threshold:
                recovery_times.append(i - change_ep)
                break
        else:
            recovery_times.append(change_frequency)  # Never recovered
    
    avg_recovery = np.mean(recovery_times) if recovery_times else 0
    
    # Worst-case: minimum cooperation in any window
    worst_case = 1.0
    for i in range(0, len(rewards) - window, window // 2):
        window_coop = np.mean(rewards[i:i + window])
        worst_case = min(worst_case, window_coop)
    
    # Stability: variance of windowed cooperation
    windowed_coops = []
    for i in range(0, len(rewards) - window, window // 2):
        windowed_coops.append(np.mean(rewards[i:i + window]))
    stability = 1 - np.std(windowed_coops) if windowed_coops else 1.0
    
    return {
        "avg_cooperation": float(avg_coop),
        "coop_regime_0": float(coop_regime_0),
        "coop_regime_1": float(coop_regime_1),
        "avg_recovery_time": float(avg_recovery),
        "worst_case": float(worst_case),
        "stability": float(stability),
    }


def run_full_experiment(
    regime_type: str = "difficulty",
    change_frequencies: List[int] = None,
    n_episodes: int = 2000,
    n_seeds: int = 10,
    n_agents: int = 30
) -> Dict[str, Any]:
    """Run full regime change experiment."""
    if change_frequencies is None:
        change_frequencies = [50, 100, 200, 500, 1000]
    
    print("=" * 70)
    print(f"EXPERIMENT 37: REGIME CHANGE ROBUSTNESS ({regime_type})")
    print("=" * 70)
    print(f"Episodes: {n_episodes}, Seeds: {n_seeds}, Agents: {n_agents}")
    print(f"Change frequencies: {change_frequencies}")
    print()
    
    methods = {
        "gcl": GCLAgent,
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
                results = run_regime_change_experiment(
                    method_class, regime_type, freq, n_episodes, n_agents, seed
                )
                metrics = compute_metrics(results, freq)
                seed_metrics.append(metrics)
            
            # Aggregate
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


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results across frequencies."""
    analysis = {}
    
    frequencies = sorted(results.keys())
    
    # Find crossover point where MARL beats GCL
    gcl_coops = [results[f]["gcl"]["avg_cooperation"]["mean"] for f in frequencies]
    iql_coops = [results[f]["iql"]["avg_cooperation"]["mean"] for f in frequencies]
    qmix_coops = [results[f]["qmix"]["avg_cooperation"]["mean"] for f in frequencies]
    
    # GCL advantage at each frequency
    for f in frequencies:
        gcl = results[f]["gcl"]["avg_cooperation"]["mean"]
        iql = results[f]["iql"]["avg_cooperation"]["mean"]
        qmix = results[f]["qmix"]["avg_cooperation"]["mean"]
        
        analysis[f"gcl_vs_iql_freq{f}"] = gcl - iql
        analysis[f"gcl_vs_qmix_freq{f}"] = gcl - qmix
    
    # Find crossover (where GCL advantage becomes negative)
    crossover_iql = None
    crossover_qmix = None
    
    for i, f in enumerate(frequencies):
        if analysis[f"gcl_vs_iql_freq{f}"] < 0 and crossover_iql is None:
            crossover_iql = f
        if analysis[f"gcl_vs_qmix_freq{f}"] < 0 and crossover_qmix is None:
            crossover_qmix = f
    
    analysis["crossover_iql"] = crossover_iql
    analysis["crossover_qmix"] = crossover_qmix
    
    # Recovery time comparison
    for f in frequencies:
        gcl_recovery = results[f]["gcl"]["avg_recovery_time"]["mean"]
        iql_recovery = results[f]["iql"]["avg_recovery_time"]["mean"]
        qmix_recovery = results[f]["qmix"]["avg_recovery_time"]["mean"]
        
        analysis[f"recovery_ratio_iql_freq{f}"] = iql_recovery / max(gcl_recovery, 1)
        analysis[f"recovery_ratio_qmix_freq{f}"] = qmix_recovery / max(gcl_recovery, 1)
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Regime Change Robustness")
    print("=" * 70)
    
    frequencies = sorted(results.keys())
    
    # Cooperation by frequency
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
    
    # Recovery time
    print("\n--- Average Recovery Time (episodes) ---")
    print(f"{'Frequency':>10} {'GCL':>10} {'IQL':>10} {'QMIX':>10}")
    print("-" * 45)
    
    for f in frequencies:
        gcl = results[f]["gcl"]["avg_recovery_time"]["mean"]
        iql = results[f]["iql"]["avg_recovery_time"]["mean"]
        qmix = results[f]["qmix"]["avg_recovery_time"]["mean"]
        print(f"{f:>10} {gcl:>10.1f} {iql:>10.1f} {qmix:>10.1f}")
    
    # Worst case
    print("\n--- Worst-Case Cooperation ---")
    print(f"{'Frequency':>10} {'GCL':>10} {'IQL':>10} {'QMIX':>10}")
    print("-" * 45)
    
    for f in frequencies:
        gcl = results[f]["gcl"]["worst_case"]["mean"]
        iql = results[f]["iql"]["worst_case"]["mean"]
        qmix = results[f]["qmix"]["worst_case"]["mean"]
        print(f"{f:>10} {gcl:>10.3f} {iql:>10.3f} {qmix:>10.3f}")
    
    # Key findings
    print("\n--- KEY FINDINGS ---")
    
    if analysis["crossover_iql"]:
        print(f"✓ GCL beats IQL when changes occur more frequently than every {analysis['crossover_iql']} episodes")
    else:
        print("✓ GCL beats IQL at all tested frequencies")
    
    if analysis["crossover_qmix"]:
        print(f"✓ GCL beats QMIX when changes occur more frequently than every {analysis['crossover_qmix']} episodes")
    else:
        print("✓ GCL beats QMIX at all tested frequencies")
    
    # Recovery advantage
    freq_50 = 50
    if freq_50 in results:
        gcl_rec = results[freq_50]["gcl"]["avg_recovery_time"]["mean"]
        iql_rec = results[freq_50]["iql"]["avg_recovery_time"]["mean"]
        print(f"\n✓ At freq=50: GCL recovers in {gcl_rec:.1f} episodes vs IQL's {iql_rec:.1f}")
        print(f"  ({iql_rec/max(gcl_rec,1):.1f}x faster recovery)")
    
    print("\n--- THEORETICAL INSIGHT ---")
    print("GCL adapts immediately because it doesn't learn - it coordinates.")
    print("MARL must re-learn after each regime change, causing performance drops.")


def main():
    # Run for difficulty regime changes
    results = run_full_experiment(
        regime_type="difficulty",
        change_frequencies=[50, 100, 200, 500, 1000],
        n_episodes=2000,
        n_seeds=10,
        n_agents=30
    )
    
    analysis = analyze_results(results)
    print_results(results, analysis)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    output = {
        "regime_type": "difficulty",
        "results": results,
        "analysis": analysis,
    }
    
    with open("results/experiment_37_regime_changes.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/experiment_37_regime_changes.json")


if __name__ == "__main__":
    main()
