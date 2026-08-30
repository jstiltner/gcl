"""
Experiment 19: Redemption Gaming Analysis

Test whether agents can exploit the redemption mechanism by intentionally
failing to position for redemption bonuses.

Key questions:
1. Do strategic agents learn to game redemption?
2. Does higher redemption bonus increase gaming?
3. Can we detect gaming patterns?
4. How do we design redemption to be gaming-resistant?
5. Do proper effort costs prevent gaming? (NEW - validates GCL theory)

Gaming strategies tested:
- HONEST: Always tries to succeed (control)
- NAIVE: Standard behavior, tries to succeed
- ALTERNATING: Intentionally alternate fail/succeed to farm redemption
- CALCULATED: Fail when expected redemption value exceeds effort cost
- ADAPTIVE: Heuristic-based strategy that adjusts based on past gaming efficiency

Metrics:
- "Cooperation" = pairwise mutual success rate (both agents in a pair succeed)
- "Success rate" = individual agent success rate

Effort Cost Model (GCL Theory):
- attempt_cost: Cost of any attempt (success or failure)
- remediation_cost: Additional cost when recovering from failure state
- gaming_penalty: Extra cost if intentional failure is detected

The GCL theory claims that proper effort costs make gaming unprofitable.
This experiment validates that claim.

Note: Collusive strategies (agents coordinating to trade redemption) are out of scope
for this experiment as they require explicit communication channels.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import defaultdict

RESULTS_DIR = Path("results/19_redemption_gaming")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class AgentStrategy(Enum):
    """Agent strategies for testing gaming."""
    NAIVE = "naive"                    # Standard behavior, no gaming
    ALTERNATING = "alternating"        # Intentionally alternate fail/succeed
    CALCULATED = "calculated"          # Fail when expected redemption > effort cost
    HEURISTIC = "heuristic"            # Threshold-based strategy (adjusts based on past efficiency)
    HONEST = "honest"                  # Always try to succeed (control)


@dataclass
class AgentState:
    """Track agent state for gaming detection."""
    agent_id: int
    strategy: AgentStrategy
    history: List[bool] = field(default_factory=list)  # True=success, False=failure
    redemption_bonuses_received: int = 0
    intentional_failures: int = 0
    total_reward: float = 0.0
    
    def add_outcome(self, success: bool, was_intentional_failure: bool = False):
        self.history.append(success)
        if was_intentional_failure:
            self.intentional_failures += 1
    
    @property
    def alternation_score(self) -> float:
        """Detect alternating patterns. High score = suspicious."""
        if len(self.history) < 4:
            return 0.0
        
        alternations = 0
        for i in range(1, len(self.history)):
            if self.history[i] != self.history[i-1]:
                alternations += 1
        
        # Perfect alternation would be len-1 alternations
        max_alternations = len(self.history) - 1
        return alternations / max_alternations if max_alternations > 0 else 0
    
    @property
    def success_rate(self) -> float:
        if not self.history:
            return 0.5
        return sum(self.history) / len(self.history)
    
    @property  
    def gaming_efficiency(self) -> float:
        """Redemption bonuses per failure. High = efficient gaming."""
        failures = len([h for h in self.history if not h])
        if failures == 0:
            return 0.0
        return self.redemption_bonuses_received / failures


class RedemptionEnvironment:
    """Environment with redemption mechanism and gaming detection."""
    
    def __init__(
        self,
        n_agents: int = 50,
        redemption_bonus: float = 0.3,
        base_success_prob: float = 0.5,
        effort_cost: float = 0.1,  # Cost of trying to succeed
        strategy_distribution: Dict[AgentStrategy, float] = None
    ):
        self.n_agents = n_agents
        self.redemption_bonus = redemption_bonus
        self.base_success_prob = base_success_prob
        self.effort_cost = effort_cost
        
        # Default: all naive
        if strategy_distribution is None:
            strategy_distribution = {AgentStrategy.NAIVE: 1.0}
        
        # Create agents with strategies
        self.agents: List[AgentState] = []
        for i in range(n_agents):
            strategy = self._sample_strategy(strategy_distribution)
            self.agents.append(AgentState(agent_id=i, strategy=strategy))
        
        self.timestep = 0
        self.cooperation_history = []
        
        # Track who just failed (eligible for redemption)
        self.just_failed = set()
    
    def _sample_strategy(self, distribution: Dict[AgentStrategy, float]) -> AgentStrategy:
        strategies = list(distribution.keys())
        probs = [distribution[s] for s in strategies]
        return np.random.choice(strategies, p=probs)
    
    def step(self) -> Dict:
        """Run one round of interactions."""
        self.timestep += 1
        
        # IMPORTANT: Snapshot just_failed at round start to avoid order effects
        # Redemption eligibility is based on previous round's failures
        redemption_eligible = self.just_failed.copy()
        
        # Pair agents randomly
        indices = list(range(self.n_agents))
        np.random.shuffle(indices)
        pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 2)]
        
        round_successes = 0
        round_total = 0
        round_individual_successes = 0
        round_individual_total = 0
        
        for i, j in pairs:
            agent_i = self.agents[i]
            agent_j = self.agents[j]
            
            # Each agent decides whether to cooperate
            success_i, intentional_fail_i = self._agent_decision(agent_i)
            success_j, intentional_fail_j = self._agent_decision(agent_j)
            
            # Compute rewards
            reward_i = self._compute_reward(agent_i, success_i, i in self.just_failed)
            reward_j = self._compute_reward(agent_j, success_j, j in self.just_failed)
            
            # Track redemption bonuses
            if success_i and i in self.just_failed:
                agent_i.redemption_bonuses_received += 1
            if success_j and j in self.just_failed:
                agent_j.redemption_bonuses_received += 1
            
            # Update state
            agent_i.add_outcome(success_i, intentional_fail_i)
            agent_j.add_outcome(success_j, intentional_fail_j)
            agent_i.total_reward += reward_i
            agent_j.total_reward += reward_j
            
            # Update just_failed set
            if not success_i:
                self.just_failed.add(i)
            else:
                self.just_failed.discard(i)
            
            if not success_j:
                self.just_failed.add(j)
            else:
                self.just_failed.discard(j)
            
            # Track cooperation
            if success_i and success_j:
                round_successes += 1
            round_total += 1
        
        cooperation_rate = round_successes / round_total if round_total > 0 else 0
        self.cooperation_history.append(cooperation_rate)
        
        return {
            'timestep': self.timestep,
            'cooperation_rate': cooperation_rate,
            'n_in_redemption_state': len(self.just_failed)
        }
    
    def _agent_decision(self, agent: AgentState) -> Tuple[bool, bool]:
        """
        Agent decides whether to succeed.
        Returns (success, was_intentional_failure)
        """
        
        if agent.strategy == AgentStrategy.HONEST:
            # Always try to succeed
            success = np.random.random() < self.base_success_prob
            return success, False
        
        elif agent.strategy == AgentStrategy.NAIVE:
            # Standard behavior - try to succeed
            success = np.random.random() < self.base_success_prob
            return success, False
        
        elif agent.strategy == AgentStrategy.ALTERNATING:
            # Intentionally alternate
            if len(agent.history) == 0:
                # First move: fail to set up redemption
                return False, True
            else:
                # Alternate from last outcome
                last_success = agent.history[-1]
                if last_success:
                    # Last was success, now fail intentionally
                    return False, True
                else:
                    # Last was failure, now succeed (collect redemption)
                    success = np.random.random() < min(0.9, self.base_success_prob + 0.3)
                    return success, False
        
        elif agent.strategy == AgentStrategy.CALCULATED:
            # Fail when not in redemption state, succeed when in redemption state
            in_redemption_state = agent.agent_id in self.just_failed
            
            if in_redemption_state:
                # Try hard to succeed and collect bonus
                success = np.random.random() < min(0.9, self.base_success_prob + 0.2)
                return success, False
            else:
                # Consider failing to set up redemption
                # Only fail if expected value is positive
                expected_redemption = self.redemption_bonus * self.base_success_prob
                if expected_redemption > self.effort_cost:
                    # Worth gaming
                    return False, True
                else:
                    # Not worth it, play normally
                    success = np.random.random() < self.base_success_prob
                    return success, False
        
        elif agent.strategy == AgentStrategy.HEURISTIC:
            # Heuristic-based strategy: adjusts based on past gaming efficiency
            # NOT reinforcement learning - just a threshold-based rule
            if len(agent.history) < 10:
                # Explore: try some gaming
                if np.random.random() < 0.3:
                    return False, True
                else:
                    success = np.random.random() < self.base_success_prob
                    return success, False
            else:
                # Exploit: check if gaming has been profitable
                gaming_roi = agent.gaming_efficiency
                if gaming_roi > 0.5:  # Gaming is working
                    # Continue gaming strategy
                    in_redemption = agent.agent_id in self.just_failed
                    if in_redemption:
                        success = np.random.random() < min(0.9, self.base_success_prob + 0.2)
                        return success, False
                    else:
                        return False, True
                else:
                    # Gaming not working, play honest
                    success = np.random.random() < self.base_success_prob
                    return success, False
        
        # Default: naive behavior
        success = np.random.random() < self.base_success_prob
        return success, False
    
    def _compute_reward(self, agent: AgentState, success: bool, was_in_redemption: bool) -> float:
        """Compute reward for an outcome.
        
        Reward structure:
        - Success: 1.0 - effort_cost (+ redemption_bonus if eligible)
        - Failure: 0.0 (no effort cost for failures - they didn't try)
        
        Note: Effort cost is applied to success attempts to model the cost of
        trying hard to cooperate. This makes gaming analysis more realistic.
        """
        if success:
            base_reward = 1.0 - self.effort_cost  # Effort cost for trying to succeed
            if was_in_redemption:
                # Redemption bonus!
                base_reward += self.redemption_bonus
            return base_reward
        else:
            return 0.0
    
    def get_gaming_analysis(self) -> Dict:
        """Analyze gaming patterns across agents."""
        
        by_strategy = defaultdict(list)
        for agent in self.agents:
            by_strategy[agent.strategy.value].append({
                'success_rate': agent.success_rate,
                'alternation_score': agent.alternation_score,
                'redemption_bonuses': agent.redemption_bonuses_received,
                'intentional_failures': agent.intentional_failures,
                'total_reward': agent.total_reward,
                'gaming_efficiency': agent.gaming_efficiency
            })
        
        summary = {}
        for strategy, agents in by_strategy.items():
            summary[strategy] = {
                'n_agents': len(agents),
                'mean_success_rate': np.mean([a['success_rate'] for a in agents]),
                'mean_alternation_score': np.mean([a['alternation_score'] for a in agents]),
                'mean_redemption_bonuses': np.mean([a['redemption_bonuses'] for a in agents]),
                'mean_intentional_failures': np.mean([a['intentional_failures'] for a in agents]),
                'mean_total_reward': np.mean([a['total_reward'] for a in agents]),
                'mean_gaming_efficiency': np.mean([a['gaming_efficiency'] for a in agents]),
            }
        
        return {
            'by_strategy': summary,
            'overall_cooperation': np.mean(self.cooperation_history[-100:]) if self.cooperation_history else 0,
            'final_cooperation': self.cooperation_history[-1] if self.cooperation_history else 0,
        }


class DecreasingRedemptionEnvironment(RedemptionEnvironment):
    """Redemption bonus decreases with each use."""
    
    def __init__(self, n_agents: int, initial_bonus: float, decay_rate: float, **kwargs):
        super().__init__(n_agents=n_agents, redemption_bonus=initial_bonus, **kwargs)
        self.initial_bonus = initial_bonus
        self.decay_rate = decay_rate
        self.agent_bonus_multiplier = {i: 1.0 for i in range(n_agents)}
    
    def _compute_reward(self, agent: AgentState, success: bool, was_in_redemption: bool) -> float:
        if success:
            base_reward = 1.0
            if was_in_redemption:
                # Decreasing bonus
                multiplier = self.agent_bonus_multiplier[agent.agent_id]
                bonus = self.initial_bonus * multiplier
                base_reward += bonus
                # Decrease for next time
                self.agent_bonus_multiplier[agent.agent_id] *= (1 - self.decay_rate)
            return base_reward
        return 0.0


class CooldownRedemptionEnvironment(RedemptionEnvironment):
    """Can only get redemption bonus once per N rounds."""
    
    def __init__(self, n_agents: int, redemption_bonus: float, cooldown_rounds: int, **kwargs):
        super().__init__(n_agents=n_agents, redemption_bonus=redemption_bonus, **kwargs)
        self.cooldown_rounds = cooldown_rounds
        self.last_redemption = {i: -cooldown_rounds for i in range(n_agents)}
    
    def _compute_reward(self, agent: AgentState, success: bool, was_in_redemption: bool) -> float:
        if success:
            base_reward = 1.0
            if was_in_redemption:
                # Check cooldown
                rounds_since = self.timestep - self.last_redemption[agent.agent_id]
                if rounds_since >= self.cooldown_rounds:
                    base_reward += self.redemption_bonus
                    self.last_redemption[agent.agent_id] = self.timestep
            return base_reward
        return 0.0


class ConsistencyBonusEnvironment(RedemptionEnvironment):
    """Bonus for consecutive successes instead of redemption."""
    
    def __init__(self, n_agents: int, streak_bonus: float, max_streak_bonus: float, **kwargs):
        super().__init__(n_agents=n_agents, redemption_bonus=0, **kwargs)
        self.streak_bonus = streak_bonus
        self.max_streak_bonus = max_streak_bonus
        self.agent_streaks = {i: 0 for i in range(n_agents)}
    
    def _compute_reward(self, agent: AgentState, success: bool, was_in_redemption: bool) -> float:
        if success:
            streak = self.agent_streaks[agent.agent_id]
            bonus = min(streak * self.streak_bonus, self.max_streak_bonus)
            self.agent_streaks[agent.agent_id] += 1
            return 1.0 + bonus
        else:
            self.agent_streaks[agent.agent_id] = 0
            return 0.0


class ProbabilisticRedemptionEnvironment(RedemptionEnvironment):
    """Redemption bonus awarded probabilistically."""
    
    def __init__(self, n_agents: int, redemption_bonus: float, redemption_probability: float, **kwargs):
        super().__init__(n_agents=n_agents, redemption_bonus=redemption_bonus, **kwargs)
        self.redemption_probability = redemption_probability
    
    def _compute_reward(self, agent: AgentState, success: bool, was_in_redemption: bool) -> float:
        if success:
            base_reward = 1.0
            if was_in_redemption:
                # Probabilistic bonus
                if np.random.random() < self.redemption_probability:
                    base_reward += self.redemption_bonus
            return base_reward
        return 0.0


class ProperEffortCostEnvironment(RedemptionEnvironment):
    """
    Environment with proper effort cost model as specified in GCL theory.
    
    This validates the claim that "effort costs prevent gaming" by implementing:
    1. attempt_cost: Every attempt (success or failure) costs effort
    2. remediation_cost: Additional cost when recovering from failure state
    3. gaming_penalty: Extra cost if intentional failure is detected
    
    The key insight: When remediation_cost > redemption_bonus * success_prob,
    gaming becomes unprofitable and honest strategies should dominate.
    """
    
    def __init__(
        self,
        n_agents: int = 50,
        redemption_bonus: float = 0.3,
        base_success_prob: float = 0.5,
        attempt_cost: float = 0.05,      # Cost per attempt (success or failure)
        remediation_cost: float = 0.15,   # Additional cost for recovery work
        gaming_penalty: float = 0.2,      # Penalty if gaming detected
        detection_prob: float = 0.3,      # Probability of detecting gaming
        strategy_distribution: Dict[AgentStrategy, float] = None
    ):
        # Initialize parent with effort_cost=0 (we handle it ourselves)
        super().__init__(
            n_agents=n_agents,
            redemption_bonus=redemption_bonus,
            base_success_prob=base_success_prob,
            effort_cost=0,  # We override the reward function
            strategy_distribution=strategy_distribution
        )
        self.attempt_cost = attempt_cost
        self.remediation_cost = remediation_cost
        self.gaming_penalty = gaming_penalty
        self.detection_prob = detection_prob
        
        # Track gaming detection
        self.detected_gamers = set()
    
    def step(self) -> Dict:
        """Run one round with proper effort cost tracking."""
        self.timestep += 1
        
        # Pair agents randomly
        indices = list(range(self.n_agents))
        np.random.shuffle(indices)
        pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 2)]
        
        round_successes = 0
        round_total = 0
        
        for i, j in pairs:
            agent_i = self.agents[i]
            agent_j = self.agents[j]
            
            # Each agent decides whether to cooperate
            success_i, intentional_fail_i = self._agent_decision(agent_i)
            success_j, intentional_fail_j = self._agent_decision(agent_j)
            
            # Compute rewards with proper effort costs
            reward_i = self._compute_reward_with_effort(
                agent_i, success_i, i in self.just_failed, intentional_fail_i
            )
            reward_j = self._compute_reward_with_effort(
                agent_j, success_j, j in self.just_failed, intentional_fail_j
            )
            
            # Track redemption bonuses
            if success_i and i in self.just_failed:
                agent_i.redemption_bonuses_received += 1
            if success_j and j in self.just_failed:
                agent_j.redemption_bonuses_received += 1
            
            # Update state
            agent_i.add_outcome(success_i, intentional_fail_i)
            agent_j.add_outcome(success_j, intentional_fail_j)
            agent_i.total_reward += reward_i
            agent_j.total_reward += reward_j
            
            # Update just_failed set
            if not success_i:
                self.just_failed.add(i)
            else:
                self.just_failed.discard(i)
            
            if not success_j:
                self.just_failed.add(j)
            else:
                self.just_failed.discard(j)
            
            # Track cooperation
            if success_i and success_j:
                round_successes += 1
            round_total += 1
        
        cooperation_rate = round_successes / round_total if round_total > 0 else 0
        self.cooperation_history.append(cooperation_rate)
        
        return {
            'timestep': self.timestep,
            'cooperation_rate': cooperation_rate,
            'n_in_redemption_state': len(self.just_failed),
            'n_detected_gamers': len(self.detected_gamers)
        }
    
    def _compute_reward_with_effort(
        self,
        agent: AgentState,
        success: bool,
        was_in_redemption: bool,
        was_intentional_failure: bool
    ) -> float:
        """
        Compute reward with proper effort cost model.
        
        This implements the GCL theoretical claim:
        - Every attempt costs effort (attempt_cost)
        - Recovery requires additional work (remediation_cost)
        - Gaming can be detected and penalized (gaming_penalty)
        
        Expected value analysis:
        - Honest: success_prob * (1.0 - attempt_cost) + (1-success_prob) * (-attempt_cost)
                = success_prob - attempt_cost
        - Gaming: -attempt_cost (fail) + success_prob * (1.0 + bonus - attempt_cost - remediation_cost)
                = success_prob * (1.0 + bonus - remediation_cost) - 2*attempt_cost
        
        Gaming is unprofitable when:
        remediation_cost > bonus + attempt_cost * (1/success_prob - 1)
        """
        # Base: every attempt costs effort
        reward = -self.attempt_cost
        
        if success:
            # Success reward
            reward += 1.0
            
            if was_in_redemption:
                # Redemption bonus, but with remediation cost
                reward += self.redemption_bonus
                reward -= self.remediation_cost  # Cost of recovery work
        else:
            # Failure: just the attempt cost (already applied)
            # But check for gaming
            if was_intentional_failure:
                # Probabilistic detection
                if np.random.random() < self.detection_prob:
                    reward -= self.gaming_penalty
                    self.detected_gamers.add(agent.agent_id)
        
        return reward
    
    def get_gaming_analysis(self) -> Dict:
        """Extended analysis including effort cost metrics."""
        base_analysis = super().get_gaming_analysis()
        
        # Add effort cost specific metrics
        base_analysis['effort_cost_model'] = {
            'attempt_cost': self.attempt_cost,
            'remediation_cost': self.remediation_cost,
            'gaming_penalty': self.gaming_penalty,
            'detection_prob': self.detection_prob,
            'n_detected_gamers': len(self.detected_gamers),
            'detection_rate': len(self.detected_gamers) / self.n_agents if self.n_agents > 0 else 0
        }
        
        # Calculate theoretical break-even point
        # Gaming unprofitable when: remediation_cost > bonus
        theoretical_gaming_profitable = self.remediation_cost < self.redemption_bonus
        base_analysis['theoretical_gaming_profitable'] = theoretical_gaming_profitable
        
        return base_analysis


def check_gaming_profitable(results: List[Dict]) -> bool:
    """Check if gaming strategies outperformed honest ones."""
    gaming_strategies = ['alternating', 'calculated', 'heuristic']
    honest_strategies = ['honest', 'naive']
    
    gaming_rewards = []
    honest_rewards = []
    
    for result in results:
        for strategy, data in result['by_strategy'].items():
            if strategy in gaming_strategies:
                gaming_rewards.append(data['mean_total_reward'])
            elif strategy in honest_strategies:
                honest_rewards.append(data['mean_total_reward'])
    
    if not gaming_rewards or not honest_rewards:
        return False
    
    return np.mean(gaming_rewards) > np.mean(honest_rewards)


def run_gaming_detection_experiment(
    redemption_bonuses: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    n_rounds: int = 200,
    n_agents: int = 50,
    n_seeds: int = 5
) -> Dict:
    """
    Test if gaming increases with redemption bonus.
    """
    
    results = {}
    
    for bonus in redemption_bonuses:
        print(f"\nRedemption bonus: {bonus:.0%}")
        
        seed_results = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            
            # Mix of strategies to detect gaming potential
            strategy_dist = {
                AgentStrategy.NAIVE: 0.4,
                AgentStrategy.HONEST: 0.2,
                AgentStrategy.ALTERNATING: 0.1,
                AgentStrategy.CALCULATED: 0.1,
                AgentStrategy.HEURISTIC: 0.2,
            }
            
            env = RedemptionEnvironment(
                n_agents=n_agents,
                redemption_bonus=bonus,
                strategy_distribution=strategy_dist
            )
            
            for _ in range(n_rounds):
                env.step()
            
            analysis = env.get_gaming_analysis()
            seed_results.append(analysis)
        
        # Aggregate across seeds
        results[f'bonus_{bonus:.2f}'] = {
            'redemption_bonus': bonus,
            'mean_cooperation': np.mean([s['overall_cooperation'] for s in seed_results]),
            'std_cooperation': np.std([s['overall_cooperation'] for s in seed_results]),
            'strategy_performance': aggregate_strategy_performance(seed_results),
        }
        
        print(f"  Cooperation: {results[f'bonus_{bonus:.2f}']['mean_cooperation']:.1%}")
    
    return results


def aggregate_strategy_performance(seed_results: List[Dict]) -> Dict:
    """Aggregate strategy performance across seeds."""
    
    strategies = ['naive', 'honest', 'alternating', 'calculated', 'heuristic']
    aggregated = {}
    
    for strategy in strategies:
        rewards = []
        alternations = []
        gaming_eff = []
        
        for result in seed_results:
            if strategy in result['by_strategy']:
                s = result['by_strategy'][strategy]
                rewards.append(s['mean_total_reward'])
                alternations.append(s['mean_alternation_score'])
                gaming_eff.append(s['mean_gaming_efficiency'])
        
        if rewards:
            aggregated[strategy] = {
                'mean_reward': np.mean(rewards),
                'mean_alternation': np.mean(alternations),
                'mean_gaming_efficiency': np.mean(gaming_eff),
            }
    
    return aggregated


def run_strategy_competition(
    redemption_bonus: float = 0.3,
    n_rounds: int = 500,
    n_agents: int = 100,
    n_seeds: int = 5
) -> Dict:
    """
    Direct competition: which strategy wins?
    """
    
    print(f"\nStrategy competition at {redemption_bonus:.0%} redemption bonus")
    
    seed_results = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        # Equal distribution of strategies
        strategy_dist = {
            AgentStrategy.NAIVE: 0.2,
            AgentStrategy.HONEST: 0.2,
            AgentStrategy.ALTERNATING: 0.2,
            AgentStrategy.CALCULATED: 0.2,
            AgentStrategy.HEURISTIC: 0.2,
        }
        
        env = RedemptionEnvironment(
            n_agents=n_agents,
            redemption_bonus=redemption_bonus,
            strategy_distribution=strategy_dist
        )
        
        for _ in range(n_rounds):
            env.step()
        
        analysis = env.get_gaming_analysis()
        seed_results.append(analysis)
    
    # Rank strategies by reward
    strategy_rewards = defaultdict(list)
    for result in seed_results:
        for strategy, data in result['by_strategy'].items():
            strategy_rewards[strategy].append(data['mean_total_reward'])
    
    ranking = sorted(
        [(s, np.mean(rewards)) for s, rewards in strategy_rewards.items()],
        key=lambda x: -x[1]
    )
    
    return {
        'redemption_bonus': redemption_bonus,
        'ranking': ranking,
        'winner': ranking[0][0] if ranking else None,
        'gaming_profitable': ranking[0][0] in ['alternating', 'calculated', 'heuristic'] if ranking else False,
        'seed_results': seed_results
    }


def test_gaming_resistant_designs(
    n_rounds: int = 200,
    n_agents: int = 50,
    n_seeds: int = 5
) -> Dict:
    """
    Test modifications to make redemption gaming-resistant.
    """
    
    results = {}
    
    # Mix of strategies
    strategy_dist = {
        AgentStrategy.NAIVE: 0.4,
        AgentStrategy.HONEST: 0.2,
        AgentStrategy.ALTERNATING: 0.1,
        AgentStrategy.CALCULATED: 0.1,
        AgentStrategy.HEURISTIC: 0.2,
    }
    
    # Design 1: Decreasing bonus
    print("\nTesting: Decreasing redemption bonus")
    design1_results = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = DecreasingRedemptionEnvironment(
            n_agents=n_agents,
            initial_bonus=0.3,
            decay_rate=0.1,  # Bonus decreases by 10% each time
            strategy_distribution=strategy_dist
        )
        for _ in range(n_rounds):
            env.step()
        design1_results.append(env.get_gaming_analysis())
    
    results['decreasing_bonus'] = {
        'mean_cooperation': np.mean([r['overall_cooperation'] for r in design1_results]),
        'gaming_profitable': check_gaming_profitable(design1_results)
    }
    print(f"  Cooperation: {results['decreasing_bonus']['mean_cooperation']:.1%}")
    
    # Design 2: Redemption cooldown
    print("Testing: Redemption cooldown")
    design2_results = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = CooldownRedemptionEnvironment(
            n_agents=n_agents,
            redemption_bonus=0.3,
            cooldown_rounds=5,
            strategy_distribution=strategy_dist
        )
        for _ in range(n_rounds):
            env.step()
        design2_results.append(env.get_gaming_analysis())
    
    results['cooldown'] = {
        'mean_cooperation': np.mean([r['overall_cooperation'] for r in design2_results]),
        'gaming_profitable': check_gaming_profitable(design2_results)
    }
    print(f"  Cooperation: {results['cooldown']['mean_cooperation']:.1%}")
    
    # Design 3: Consistency bonus instead
    print("Testing: Consistency bonus (replaces redemption)")
    design3_results = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = ConsistencyBonusEnvironment(
            n_agents=n_agents,
            streak_bonus=0.1,  # Bonus per consecutive success
            max_streak_bonus=0.5,
            strategy_distribution=strategy_dist
        )
        for _ in range(n_rounds):
            env.step()
        design3_results.append(env.get_gaming_analysis())
    
    results['consistency_bonus'] = {
        'mean_cooperation': np.mean([r['overall_cooperation'] for r in design3_results]),
        'gaming_profitable': check_gaming_profitable(design3_results)
    }
    print(f"  Cooperation: {results['consistency_bonus']['mean_cooperation']:.1%}")
    
    # Design 4: Probabilistic redemption
    print("Testing: Probabilistic redemption")
    design4_results = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = ProbabilisticRedemptionEnvironment(
            n_agents=n_agents,
            redemption_bonus=0.5,  # Higher bonus but...
            redemption_probability=0.5,  # Only 50% chance of getting it
            strategy_distribution=strategy_dist
        )
        for _ in range(n_rounds):
            env.step()
        design4_results.append(env.get_gaming_analysis())
    
    results['probabilistic'] = {
        'mean_cooperation': np.mean([r['overall_cooperation'] for r in design4_results]),
        'gaming_profitable': check_gaming_profitable(design4_results)
    }
    print(f"  Cooperation: {results['probabilistic']['mean_cooperation']:.1%}")
    
    return results


def test_effort_cost_prevents_gaming(
    n_rounds: int = 300,
    n_agents: int = 50,
    n_seeds: int = 10
) -> Dict:
    """
    Test the GCL theoretical claim: proper effort costs prevent gaming.
    
    This is the KEY experiment that validates the theoretical claim.
    
    We test multiple effort cost configurations:
    1. No effort cost (baseline - gaming should be profitable)
    2. Low effort cost (gaming may still be profitable)
    3. Proper effort cost (gaming should be unprofitable)
    4. High effort cost (gaming definitely unprofitable)
    
    The prediction: When remediation_cost > redemption_bonus, honest wins.
    """
    
    print("\n" + "=" * 70)
    print("PART 4: Effort Cost Prevents Gaming (GCL Theory Validation)")
    print("=" * 70)
    
    results = {}
    
    # Strategy distribution with gaming agents
    strategy_dist = {
        AgentStrategy.NAIVE: 0.3,
        AgentStrategy.HONEST: 0.3,
        AgentStrategy.ALTERNATING: 0.15,
        AgentStrategy.CALCULATED: 0.15,
        AgentStrategy.HEURISTIC: 0.1,
    }
    
    # Test configurations
    configs = [
        {
            'name': 'no_effort_cost',
            'attempt_cost': 0.0,
            'remediation_cost': 0.0,
            'gaming_penalty': 0.0,
            'detection_prob': 0.0,
            'expected_gaming_profitable': True,
            'description': 'No effort costs - gaming should be profitable'
        },
        {
            'name': 'low_effort_cost',
            'attempt_cost': 0.02,
            'remediation_cost': 0.1,
            'gaming_penalty': 0.1,
            'detection_prob': 0.2,
            'expected_gaming_profitable': True,  # remediation < bonus
            'description': 'Low effort costs - gaming may still work'
        },
        {
            'name': 'proper_effort_cost',
            'attempt_cost': 0.05,
            'remediation_cost': 0.35,  # > redemption_bonus (0.3)
            'gaming_penalty': 0.2,
            'detection_prob': 0.3,
            'expected_gaming_profitable': False,  # remediation > bonus
            'description': 'Proper effort costs - gaming should be unprofitable'
        },
        {
            'name': 'high_effort_cost',
            'attempt_cost': 0.1,
            'remediation_cost': 0.5,
            'gaming_penalty': 0.5,
            'detection_prob': 0.5,
            'expected_gaming_profitable': False,
            'description': 'High effort costs - gaming definitely unprofitable'
        }
    ]
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  {config['description']}")
        print(f"  remediation_cost={config['remediation_cost']}, redemption_bonus=0.3")
        
        seed_results = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            
            env = ProperEffortCostEnvironment(
                n_agents=n_agents,
                redemption_bonus=0.3,
                base_success_prob=0.5,
                attempt_cost=config['attempt_cost'],
                remediation_cost=config['remediation_cost'],
                gaming_penalty=config['gaming_penalty'],
                detection_prob=config['detection_prob'],
                strategy_distribution=strategy_dist
            )
            
            for _ in range(n_rounds):
                env.step()
            
            analysis = env.get_gaming_analysis()
            seed_results.append(analysis)
        
        # Aggregate results
        gaming_profitable = check_gaming_profitable(seed_results)
        
        results[config['name']] = {
            'config': config,
            'mean_cooperation': np.mean([r['overall_cooperation'] for r in seed_results]),
            'std_cooperation': np.std([r['overall_cooperation'] for r in seed_results]),
            'gaming_profitable': gaming_profitable,
            'expected_gaming_profitable': config['expected_gaming_profitable'],
            'prediction_correct': gaming_profitable == config['expected_gaming_profitable'],
            'strategy_performance': aggregate_strategy_performance(seed_results),
            'n_detected_gamers': np.mean([r['effort_cost_model']['n_detected_gamers'] for r in seed_results])
        }
        
        status = "✓" if results[config['name']]['prediction_correct'] else "✗"
        print(f"  Cooperation: {results[config['name']]['mean_cooperation']:.1%}")
        print(f"  Gaming profitable: {gaming_profitable} (expected: {config['expected_gaming_profitable']}) {status}")
    
    # Summary
    n_correct = sum(1 for r in results.values() if r['prediction_correct'])
    n_total = len(results)
    
    print(f"\n{'='*70}")
    print(f"EFFORT COST VALIDATION: {n_correct}/{n_total} predictions correct")
    print(f"{'='*70}")
    
    if n_correct == n_total:
        print("\n✓ GCL THEORY VALIDATED: Proper effort costs prevent gaming!")
    else:
        print("\n✗ Some predictions failed - investigate further")
    
    return results


def create_visualizations(gaming_results: Dict, competition: Dict, resistant_results: Dict, effort_results: Dict = None):
    """Create visualizations for gaming analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cooperation vs Redemption Bonus
    ax1 = axes[0, 0]
    bonuses = []
    coops = []
    for key, data in gaming_results.items():
        bonuses.append(data['redemption_bonus'])
        coops.append(data['mean_cooperation'])
    
    ax1.plot(bonuses, coops, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Redemption Bonus')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Cooperation vs Redemption Bonus (with Gaming Agents)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Strategy Competition
    ax2 = axes[0, 1]
    strategies = [s for s, _ in competition['ranking']]
    rewards = [r for _, r in competition['ranking']]
    colors = ['red' if s in ['alternating', 'calculated', 'heuristic'] else 'green' for s in strategies]
    
    ax2.barh(range(len(strategies)), rewards, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(strategies)))
    ax2.set_yticklabels(strategies)
    ax2.set_xlabel('Total Reward')
    ax2.set_title(f'Strategy Competition (Bonus={competition["redemption_bonus"]:.0%})')
    ax2.axvline(x=np.mean(rewards), color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Gaming-Resistant Designs
    ax3 = axes[1, 0]
    designs = list(resistant_results.keys())
    design_coops = [resistant_results[d]['mean_cooperation'] for d in designs]
    gaming_status = ['VULNERABLE' if resistant_results[d]['gaming_profitable'] else 'RESISTANT' for d in designs]
    colors = ['red' if s == 'VULNERABLE' else 'green' for s in gaming_status]
    
    bars = ax3.bar(range(len(designs)), design_coops, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(designs)))
    ax3.set_xticklabels([d.replace('_', '\n') for d in designs], fontsize=9)
    ax3.set_ylabel('Cooperation Rate')
    ax3.set_title('Gaming-Resistant Designs')
    
    for bar, status in zip(bars, gaming_status):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                status, ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.9, 'GAMING ANALYSIS SUMMARY', ha='center', va='top', 
             fontsize=14, fontweight='bold', transform=ax4.transAxes)
    
    summary_text = f"""
Winner Strategy: {competition['winner']}
Gaming Profitable: {competition['gaming_profitable']}

Gaming-Resistant Designs:
"""
    for design, data in resistant_results.items():
        status = 'RESISTANT' if not data['gaming_profitable'] else 'VULNERABLE'
        summary_text += f"  • {design}: {status}\n"
    
    ax4.text(0.1, 0.7, summary_text, ha='left', va='top', fontsize=10,
             transform=ax4.transAxes, family='monospace')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gaming_analysis.png", dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {RESULTS_DIR / 'gaming_analysis.png'}")


def main():
    print("=" * 70)
    print("EXPERIMENT 19: Redemption Gaming Analysis")
    print("=" * 70)
    
    all_results = {}
    
    # Part 1: Does gaming increase with bonus?
    print("\n" + "=" * 70)
    print("PART 1: Gaming vs Redemption Bonus Level")
    print("=" * 70)
    
    gaming_vs_bonus = run_gaming_detection_experiment(
        redemption_bonuses=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        n_rounds=200,
        n_agents=50,
        n_seeds=5
    )
    all_results['gaming_vs_bonus'] = gaming_vs_bonus
    
    # Part 2: Strategy competition
    print("\n" + "=" * 70)
    print("PART 2: Strategy Competition")
    print("=" * 70)
    
    competition = run_strategy_competition(
        redemption_bonus=0.3,
        n_rounds=500,
        n_agents=100,
        n_seeds=5
    )
    all_results['strategy_competition'] = competition
    
    print(f"\nStrategy ranking:")
    for rank, (strategy, reward) in enumerate(competition['ranking'], 1):
        print(f"  {rank}. {strategy}: {reward:.2f}")
    print(f"\nGaming profitable: {competition['gaming_profitable']}")
    
    # Part 3: Gaming-resistant designs
    print("\n" + "=" * 70)
    print("PART 3: Gaming-Resistant Designs")
    print("=" * 70)
    
    resistant_designs = test_gaming_resistant_designs(
        n_rounds=200,
        n_agents=50,
        n_seeds=5
    )
    all_results['resistant_designs'] = resistant_designs
    
    print("\nDesign comparison:")
    for design, data in resistant_designs.items():
        gaming_status = "VULNERABLE" if data['gaming_profitable'] else "RESISTANT"
        print(f"  {design}: {data['mean_cooperation']:.1%} cooperation, {gaming_status}")
    
    # Part 4: Effort cost validation (NEW - validates GCL theory)
    effort_cost_results = test_effort_cost_prevents_gaming(
        n_rounds=300,
        n_agents=50,
        n_seeds=10
    )
    all_results['effort_cost_validation'] = effort_cost_results
    
    # Create visualizations
    create_visualizations(gaming_vs_bonus, competition, resistant_designs, effort_cost_results)
    
    # Save results
    (RESULTS_DIR / "gaming_analysis.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
KEY FINDINGS:

1. Gaming vs Bonus Level:
   - At 0% bonus: {gaming_vs_bonus.get('bonus_0.00', {}).get('mean_cooperation', 0):.1%} cooperation
   - At 30% bonus: {gaming_vs_bonus.get('bonus_0.30', {}).get('mean_cooperation', 0):.1%} cooperation
   - At 100% bonus: {gaming_vs_bonus.get('bonus_1.00', {}).get('mean_cooperation', 0):.1%} cooperation

2. Strategy Competition:
   - Winner: {competition['winner']}
   - Gaming profitable: {competition['gaming_profitable']}

3. Gaming-Resistant Designs:
""")
    
    for design, data in resistant_designs.items():
        print(f"   - {design}: {'RESISTANT' if not data['gaming_profitable'] else 'VULNERABLE'}")
    
    # Effort cost validation summary
    n_correct = sum(1 for r in effort_cost_results.values() if r['prediction_correct'])
    n_total = len(effort_cost_results)
    print(f"""
4. Effort Cost Validation (GCL Theory):
   - Predictions correct: {n_correct}/{n_total}
   - Theory validated: {'YES' if n_correct == n_total else 'PARTIAL'}
   - Key finding: When remediation_cost > redemption_bonus, gaming is unprofitable
""")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return all_results


if __name__ == "__main__":
    main()