"""
Experiment 29: Boundary Conditions via Environmental Enrichment

Find conditions where meritocracy (or other structures) beats Ubuntu.
Test various market conditions to identify crossover points.

Environmental Conditions:
A. Reward Distribution (winner_take_all)
B. Volatility (task_difficulty_variance)
C. Market Relationship (repeat_partnership_bonus)
D. Scarcity (tasks_per_round)
E. Task Type (aggregation: max vs sum)
F. Scale (n_agents)
"""

import sys
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.social_structures.agents.agent import Agent, Task, create_population
from experiments.social_structures.competition.firm import Firm, create_firm, FirmTaskOutcome
from experiments.social_structures.structures.implementations import (
    Meritocracy, Guild, ObligationNetwork, Ubuntu, RotatingLeadership, Monastic
)


@dataclass
class BoundaryConfig:
    """Configuration for boundary condition experiments."""
    
    # Base settings
    n_agents_per_firm: int = 30
    n_rounds: int = 100
    tasks_per_round: int = 10
    n_trials: int = 5
    starting_resources: float = 50.0
    
    # A. Reward Distribution
    winner_take_all: float = 1.0  # 1.0 = winner gets all, 0.5 = proportional
    win_reward: float = 3.0
    loss_penalty: float = 1.0
    
    # B. Volatility
    task_difficulty_mean: float = 0.5
    task_difficulty_variance: float = 0.2
    
    # C. Market Relationship
    repeat_partnership_bonus: float = 0.0  # Bonus for repeated partnerships
    
    # D. Scarcity (tasks_per_round above)
    
    # E. Task Aggregation
    task_aggregation: str = "sum"  # "sum" = team output, "max" = best individual
    
    # F. Scale (n_agents_per_firm above)


class BoundaryExperiment:
    """Run boundary condition experiments."""
    
    def __init__(self, config: BoundaryConfig):
        self.config = config
        self.partnership_history: Dict[str, Dict[str, int]] = {}  # firm -> {partner -> count}
    
    def create_firms(self) -> Dict[str, Firm]:
        """Create firms with all 6 structures."""
        structures = ["meritocracy", "guild", "obligation", "ubuntu", "rotating", "monastic"]
        firms = {}
        
        for struct_name in structures:
            firm = create_firm(
                firm_id=f"firm_{struct_name}",
                structure_type=struct_name,
                n_agents=self.config.n_agents_per_firm,
                starting_resources=self.config.starting_resources
            )
            firms[struct_name] = firm
            self.partnership_history[struct_name] = {}
        
        return firms
    
    def generate_task(self, round_num: int) -> Task:
        """Generate a task with configurable volatility."""
        # Base difficulty with variance
        base_difficulty = self.config.task_difficulty_mean
        variance = self.config.task_difficulty_variance
        
        difficulty = base_difficulty + random.gauss(0, variance)
        difficulty = max(0.1, min(0.9, difficulty))
        
        return Task(
            id=f"task_{round_num}_{random.randint(0, 10000)}",
            difficulty=difficulty,
            reward=self.config.win_reward
        )
    
    def calculate_rewards(
        self, 
        outcomes: Dict[str, FirmTaskOutcome],
        firms: Dict[str, Firm]
    ) -> Dict[str, float]:
        """Calculate rewards based on winner_take_all setting."""
        if not outcomes:
            return {}
        
        # Get outputs
        outputs = {name: outcome.output for name, outcome in outcomes.items()}
        total_output = sum(outputs.values())
        
        if total_output == 0:
            return {name: 0.0 for name in outcomes}
        
        # Determine winner
        winner = max(outputs.items(), key=lambda x: x[1])[0]
        
        rewards = {}
        for name in outcomes:
            if self.config.winner_take_all >= 0.99:
                # Pure winner-take-all
                if name == winner:
                    rewards[name] = self.config.win_reward
                else:
                    rewards[name] = -self.config.loss_penalty
            else:
                # Proportional rewards
                proportion = outputs[name] / total_output if total_output > 0 else 0
                winner_share = self.config.winner_take_all
                proportional_share = 1 - winner_share
                
                if name == winner:
                    rewards[name] = self.config.win_reward * (winner_share + proportional_share * proportion)
                else:
                    rewards[name] = self.config.win_reward * proportional_share * proportion - self.config.loss_penalty * (1 - proportion)
        
        return rewards
    
    def aggregate_output(self, outcomes: List[Any]) -> float:
        """Aggregate outputs based on task_aggregation setting."""
        if not outcomes:
            return 0.0
        
        outputs = [o.output_quality for o in outcomes if hasattr(o, 'output_quality')]
        
        if not outputs:
            return 0.0
        
        if self.config.task_aggregation == "max":
            return max(outputs)
        else:  # "sum"
            return sum(outputs)
    
    def run_competition_round(
        self, 
        firms: Dict[str, Firm], 
        round_num: int
    ) -> Dict[str, Any]:
        """Run one round of competition."""
        round_results = {
            "round": round_num,
            "tasks": [],
            "firm_states": {}
        }
        
        # Run internal updates
        for firm in firms.values():
            if not firm.is_bankrupt():
                firm.run_internal_round()
        
        # Compete on tasks
        for task_idx in range(self.config.tasks_per_round):
            task = self.generate_task(round_num)
            
            # Each firm attempts the task
            outcomes = {}
            for name, firm in firms.items():
                if not firm.is_bankrupt():
                    outcome = firm.attempt_task(task)
                    # Override output with custom aggregation
                    outcome.output = self.aggregate_output(outcome.agent_outcomes)
                    outcomes[name] = outcome
            
            # Calculate rewards
            rewards = self.calculate_rewards(outcomes, firms)
            
            # Apply rewards
            for name, reward in rewards.items():
                if reward > 0:
                    firms[name].add_resources(reward)
                else:
                    firms[name].remove_resources(abs(reward))
            
            # Track partnerships for relationship bonus
            if len(outcomes) >= 2:
                active_firms = list(outcomes.keys())
                for i, f1 in enumerate(active_firms):
                    for f2 in active_firms[i+1:]:
                        if f2 not in self.partnership_history[f1]:
                            self.partnership_history[f1][f2] = 0
                        self.partnership_history[f1][f2] += 1
            
            round_results["tasks"].append({
                "task_id": task.id,
                "outputs": {n: o.output for n, o in outcomes.items()},
                "rewards": rewards
            })
        
        # End-of-round updates
        for name, firm in firms.items():
            if not firm.is_bankrupt():
                firm.end_internal_round()
                firm.structure.maybe_share_knowledge(firm.agents)
            
            round_results["firm_states"][name] = {
                "resources": firm.resources,
                "bankrupt": firm.is_bankrupt(),
                "avg_capability": firm.get_average_capability(),
                "wins": firm.wins,
                "losses": firm.losses
            }
        
        return round_results
    
    def run_trial(self, trial_num: int) -> Dict[str, Any]:
        """Run a single trial."""
        random.seed(42 + trial_num)
        
        firms = self.create_firms()
        trial_results = {
            "trial": trial_num,
            "rounds": [],
            "final_standings": {},
            "extinction_rounds": {},
            "survival": {}
        }
        
        for round_num in range(self.config.n_rounds):
            round_results = self.run_competition_round(firms, round_num)
            trial_results["rounds"].append(round_results)
            
            # Track extinctions
            for name, firm in firms.items():
                if firm.is_bankrupt() and name not in trial_results["extinction_rounds"]:
                    trial_results["extinction_rounds"][name] = round_num
        
        # Final standings
        for name, firm in firms.items():
            trial_results["final_standings"][name] = {
                "resources": firm.resources,
                "wins": firm.wins,
                "losses": firm.losses,
                "survived": not firm.is_bankrupt(),
                "final_capability": firm.get_average_capability()
            }
            trial_results["survival"][name] = not firm.is_bankrupt()
        
        return trial_results
    
    def run_experiment(self, condition_name: str) -> Dict[str, Any]:
        """Run all trials for a condition."""
        print(f"  Running: {condition_name}...")
        
        all_trials = []
        for trial in range(self.config.n_trials):
            trial_results = self.run_trial(trial)
            all_trials.append(trial_results)
        
        # Aggregate
        return self._aggregate_trials(all_trials, condition_name)
    
    def _aggregate_trials(self, trials: List[Dict], condition_name: str) -> Dict[str, Any]:
        """Aggregate results across trials."""
        structures = ["meritocracy", "guild", "obligation", "ubuntu", "rotating", "monastic"]
        
        aggregated = {
            "condition_name": condition_name,
            "n_trials": len(trials),
            "survival_rates": {},
            "avg_wins": {},
            "avg_final_resources": {},
            "win_counts": {}
        }
        
        for struct in structures:
            survivals = [t["survival"][struct] for t in trials]
            wins = [t["final_standings"][struct]["wins"] for t in trials]
            resources = [t["final_standings"][struct]["resources"] for t in trials]
            
            aggregated["survival_rates"][struct] = sum(survivals) / len(survivals)
            aggregated["avg_wins"][struct] = sum(wins) / len(wins)
            aggregated["avg_final_resources"][struct] = sum(resources) / len(resources)
        
        # Count trial wins
        for struct in structures:
            aggregated["win_counts"][struct] = 0
        
        for trial in trials:
            winner = max(
                trial["final_standings"].items(),
                key=lambda x: x[1]["resources"]
            )[0]
            aggregated["win_counts"][winner] += 1
        
        return aggregated


def run_boundary_experiments():
    """Run all boundary condition experiments."""
    results = {}
    
    print("\n" + "="*70)
    print("EXPERIMENT 29: BOUNDARY CONDITIONS")
    print("="*70)
    
    # A. Reward Distribution
    print("\n[A] REWARD DISTRIBUTION")
    
    for wta in [0.50, 0.70, 0.85, 0.95, 1.0]:
        config = BoundaryConfig(winner_take_all=wta)
        exp = BoundaryExperiment(config)
        results[f"wta_{wta}"] = exp.run_experiment(f"Winner-Take-All = {wta}")
    
    # B. Volatility
    print("\n[B] VOLATILITY")
    
    for var in [0.05, 0.1, 0.2, 0.3, 0.5]:
        config = BoundaryConfig(task_difficulty_variance=var)
        exp = BoundaryExperiment(config)
        results[f"volatility_{var}"] = exp.run_experiment(f"Volatility = {var}")
    
    # C. Scarcity
    print("\n[C] SCARCITY")
    
    for tasks in [3, 5, 10, 15, 20]:
        config = BoundaryConfig(tasks_per_round=tasks)
        exp = BoundaryExperiment(config)
        results[f"tasks_{tasks}"] = exp.run_experiment(f"Tasks/Round = {tasks}")
    
    # D. Task Aggregation
    print("\n[D] TASK AGGREGATION")
    
    for agg in ["sum", "max"]:
        config = BoundaryConfig(task_aggregation=agg)
        exp = BoundaryExperiment(config)
        results[f"agg_{agg}"] = exp.run_experiment(f"Aggregation = {agg}")
    
    # E. Scale
    print("\n[E] SCALE")
    
    for n_agents in [10, 20, 30, 50, 100]:
        config = BoundaryConfig(n_agents_per_firm=n_agents, n_rounds=50)  # Fewer rounds for large scale
        exp = BoundaryExperiment(config)
        results[f"scale_{n_agents}"] = exp.run_experiment(f"N Agents = {n_agents}")
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary of boundary conditions."""
    print("\n" + "="*70)
    print("SUMMARY: BOUNDARY CONDITIONS")
    print("="*70)
    
    structures = ["meritocracy", "guild", "obligation", "ubuntu", "rotating", "monastic"]
    
    # Find conditions where Ubuntu doesn't dominate
    print("\n[CROSSOVER ANALYSIS]")
    print("-" * 50)
    
    crossovers = []
    
    for condition, data in results.items():
        ubuntu_wins = data["win_counts"]["ubuntu"]
        total = data["n_trials"]
        ubuntu_rate = ubuntu_wins / total
        
        # Find the winner
        winner = max(data["win_counts"].items(), key=lambda x: x[1])[0]
        winner_wins = data["win_counts"][winner]
        
        if winner != "ubuntu":
            crossovers.append({
                "condition": condition,
                "winner": winner,
                "winner_rate": winner_wins / total,
                "ubuntu_rate": ubuntu_rate
            })
            print(f"  *** {data['condition_name']}: {winner.upper()} wins ({winner_wins}/{total})")
        else:
            print(f"      {data['condition_name']}: Ubuntu wins ({ubuntu_wins}/{total})")
    
    # Detailed crossover analysis
    print("\n[CROSSOVER POINTS]")
    print("-" * 50)
    
    # Winner-take-all crossover
    print("\nA. Winner-Take-All Crossover:")
    for wta in [0.50, 0.70, 0.85, 0.95, 1.0]:
        key = f"wta_{wta}"
        if key in results:
            data = results[key]
            ubuntu_wins = data["win_counts"]["ubuntu"]
            meritocracy_wins = data["win_counts"]["meritocracy"]
            total = data["n_trials"]
            print(f"   WTA={wta}: Ubuntu {ubuntu_wins}/{total}, Meritocracy {meritocracy_wins}/{total}")
    
    # Volatility crossover
    print("\nB. Volatility Crossover:")
    for var in [0.05, 0.1, 0.2, 0.3, 0.5]:
        key = f"volatility_{var}"
        if key in results:
            data = results[key]
            ubuntu_wins = data["win_counts"]["ubuntu"]
            meritocracy_wins = data["win_counts"]["meritocracy"]
            total = data["n_trials"]
            print(f"   Var={var}: Ubuntu {ubuntu_wins}/{total}, Meritocracy {meritocracy_wins}/{total}")
    
    # Scarcity crossover
    print("\nC. Scarcity Crossover:")
    for tasks in [3, 5, 10, 15, 20]:
        key = f"tasks_{tasks}"
        if key in results:
            data = results[key]
            ubuntu_wins = data["win_counts"]["ubuntu"]
            meritocracy_wins = data["win_counts"]["meritocracy"]
            total = data["n_trials"]
            print(f"   Tasks={tasks}: Ubuntu {ubuntu_wins}/{total}, Meritocracy {meritocracy_wins}/{total}")
    
    # Aggregation crossover
    print("\nD. Aggregation Crossover:")
    for agg in ["sum", "max"]:
        key = f"agg_{agg}"
        if key in results:
            data = results[key]
            ubuntu_wins = data["win_counts"]["ubuntu"]
            meritocracy_wins = data["win_counts"]["meritocracy"]
            total = data["n_trials"]
            print(f"   Agg={agg}: Ubuntu {ubuntu_wins}/{total}, Meritocracy {meritocracy_wins}/{total}")
    
    # Scale crossover
    print("\nE. Scale Crossover:")
    for n in [10, 20, 30, 50, 100]:
        key = f"scale_{n}"
        if key in results:
            data = results[key]
            ubuntu_wins = data["win_counts"]["ubuntu"]
            meritocracy_wins = data["win_counts"]["meritocracy"]
            total = data["n_trials"]
            print(f"   N={n}: Ubuntu {ubuntu_wins}/{total}, Meritocracy {meritocracy_wins}/{total}")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    if crossovers:
        print("\nConditions where Ubuntu LOSES:")
        for c in crossovers:
            print(f"  - {c['condition']}: {c['winner']} wins {c['winner_rate']*100:.0f}%")
    else:
        print("\nUbuntu dominates ALL conditions tested!")
    
    # Hypothesis validation
    print("\n[HYPOTHESIS VALIDATION]")
    print("-" * 50)
    
    hypotheses = [
        ("Winner-take-all (0.95)", "meritocracy", "wta_0.95"),
        ("Proportional (0.50)", "ubuntu", "wta_0.5"),
        ("High volatility (0.5)", "ubuntu", "volatility_0.5"),
        ("Low volatility (0.05)", "meritocracy", "volatility_0.05"),
        ("Scarcity (3 tasks)", "meritocracy", "tasks_3"),
        ("Abundance (20 tasks)", "ubuntu", "tasks_20"),
        ("Max aggregation", "meritocracy", "agg_max"),
        ("Sum aggregation", "ubuntu", "agg_sum"),
        ("Small scale (10)", "tie", "scale_10"),
        ("Large scale (100)", "meritocracy", "scale_100"),
    ]
    
    passed = 0
    for name, predicted, key in hypotheses:
        if key in results:
            data = results[key]
            actual_winner = max(data["win_counts"].items(), key=lambda x: x[1])[0]
            
            if predicted == "tie":
                # Check if close
                ubuntu_wins = data["win_counts"]["ubuntu"]
                meritocracy_wins = data["win_counts"]["meritocracy"]
                is_tie = abs(ubuntu_wins - meritocracy_wins) <= 1
                status = "✓" if is_tie else "✗"
                if is_tie:
                    passed += 1
            else:
                status = "✓" if actual_winner == predicted else "✗"
                if actual_winner == predicted:
                    passed += 1
            
            print(f"  {status} {name}: Predicted {predicted}, Actual {actual_winner}")
    
    print(f"\nHypotheses passed: {passed}/{len(hypotheses)}")


if __name__ == "__main__":
    results = run_boundary_experiments()
