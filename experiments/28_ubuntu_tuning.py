"""
Experiment 28: Ubuntu Dominance Tuning

Validates whether Ubuntu's dominance in social structures is robust
and identifies the key mechanisms.

Questions:
1. Is Ubuntu's win due to knowledge sharing or collective identity?
2. Is the early extinction (round 6-7) an artifact of low starting resources?
3. Is capability growth bounded realistically?
4. Is Ubuntu vulnerable to free-riders?

Experiments:
A. Disable knowledge sharing for all structures
B. Higher starting resources (100 vs 10)
C. Cap capability growth with diminishing returns
D. Add free-rider agents to Ubuntu
"""

import sys
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.social_structures.agents.agent import Agent, Task, create_population
from experiments.social_structures.competition.firm import Firm, create_firm, FirmTaskOutcome
from experiments.social_structures.structures.implementations import (
    Meritocracy, Guild, ObligationNetwork, Ubuntu, RotatingLeadership, Monastic
)


@dataclass
class TuningConfig:
    """Configuration for tuning experiments."""
    
    # Base settings
    n_agents_per_firm: int = 30
    n_rounds: int = 50
    tasks_per_round: int = 10
    n_trials: int = 5
    
    # Experiment A: Knowledge sharing
    disable_knowledge_sharing: bool = False
    
    # Experiment B: Starting resources
    starting_resources: float = 10.0
    
    # Experiment C: Capability caps
    max_capability: float = 1.0
    template_diminishing_returns: bool = False
    diminishing_factor: float = 0.5  # Each template gives 50% less than previous
    
    # Experiment D: Free riders
    free_rider_fraction: float = 0.0  # Fraction of agents that are free-riders
    
    # Competition settings
    win_reward: float = 2.0
    loss_penalty: float = 1.0
    bankruptcy_threshold: float = 0.0


class TuningExperiment:
    """Run tuning experiments to validate Ubuntu dominance."""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.results = {}
    
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
            
            # Apply capability cap if configured
            if self.config.max_capability < 1.0:
                for agent in firm.agents:
                    agent.base_capability = min(agent.base_capability, self.config.max_capability)
            
            # Add free-riders to Ubuntu if configured
            if struct_name == "ubuntu" and self.config.free_rider_fraction > 0:
                n_free_riders = int(len(firm.agents) * self.config.free_rider_fraction)
                for i in range(n_free_riders):
                    firm.agents[i].is_free_rider = True
            
            firms[struct_name] = firm
        
        return firms
    
    def generate_task(self, round_num: int) -> Task:
        """Generate a task with varying difficulty."""
        difficulty = 0.3 + 0.4 * (round_num / self.config.n_rounds)  # Increases over time
        return Task(
            id=f"task_{round_num}_{random.randint(0, 10000)}",
            difficulty=min(0.9, difficulty + random.uniform(-0.1, 0.1)),
            reward=self.config.win_reward
        )
    
    def run_competition_round(
        self, 
        firms: Dict[str, Firm], 
        round_num: int
    ) -> Dict[str, Any]:
        """Run one round of competition between firms."""
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
                    outcomes[name] = outcome
            
            # Determine winner (highest output)
            if outcomes:
                winner = max(outcomes.items(), key=lambda x: x[1].output)
                winner_name = winner[0]
                
                # Distribute rewards/penalties
                for name, outcome in outcomes.items():
                    if name == winner_name:
                        firms[name].add_resources(self.config.win_reward)
                    else:
                        firms[name].remove_resources(self.config.loss_penalty)
                
                round_results["tasks"].append({
                    "task_id": task.id,
                    "winner": winner_name,
                    "outputs": {n: o.output for n, o in outcomes.items()}
                })
        
        # End-of-round updates
        for name, firm in firms.items():
            if not firm.is_bankrupt():
                firm.end_internal_round()
                
                # Apply knowledge sharing control
                if self.config.disable_knowledge_sharing:
                    # Skip knowledge sharing
                    pass
                else:
                    firm.structure.maybe_share_knowledge(firm.agents)
                
                # Apply capability cap with diminishing returns
                if self.config.template_diminishing_returns:
                    for agent in firm.agents:
                        self._apply_diminishing_returns(agent)
            
            # Record firm state
            round_results["firm_states"][name] = {
                "resources": firm.resources,
                "bankrupt": firm.is_bankrupt(),
                "avg_capability": firm.get_average_capability(),
                "avg_reputation": firm.get_average_reputation(),
                "total_templates": firm.get_total_templates(),
                "wins": firm.wins,
                "losses": firm.losses
            }
        
        return round_results
    
    def _apply_diminishing_returns(self, agent: Agent):
        """Apply diminishing returns to template capability boosts."""
        if not agent.template_library:
            return
        
        # Sort templates by capability boost (highest first)
        sorted_templates = sorted(
            agent.template_library, 
            key=lambda t: t.capability_boost, 
            reverse=True
        )
        
        # Apply diminishing factor
        total_boost = 0.0
        for i, template in enumerate(sorted_templates):
            factor = self.config.diminishing_factor ** i
            total_boost += template.capability_boost * factor
        
        # Cap total boost
        agent.learned_capability = min(
            total_boost, 
            self.config.max_capability - agent.base_capability
        )
    
    def run_trial(self, trial_num: int) -> Dict[str, Any]:
        """Run a single trial of the experiment."""
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
                "final_capability": firm.get_average_capability(),
                "final_reputation": firm.get_average_reputation()
            }
            trial_results["survival"][name] = not firm.is_bankrupt()
        
        return trial_results
    
    def run_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Run all trials for an experiment configuration."""
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        all_trials = []
        for trial in range(self.config.n_trials):
            print(f"  Trial {trial + 1}/{self.config.n_trials}...")
            trial_results = self.run_trial(trial)
            all_trials.append(trial_results)
        
        # Aggregate results
        aggregated = self._aggregate_trials(all_trials)
        aggregated["experiment_name"] = experiment_name
        aggregated["config"] = {
            "disable_knowledge_sharing": self.config.disable_knowledge_sharing,
            "starting_resources": self.config.starting_resources,
            "max_capability": self.config.max_capability,
            "template_diminishing_returns": self.config.template_diminishing_returns,
            "free_rider_fraction": self.config.free_rider_fraction
        }
        
        return aggregated
    
    def _aggregate_trials(self, trials: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across trials."""
        structures = ["meritocracy", "guild", "obligation", "ubuntu", "rotating", "monastic"]
        
        aggregated = {
            "n_trials": len(trials),
            "survival_rates": {},
            "avg_wins": {},
            "avg_final_resources": {},
            "avg_final_capability": {},
            "avg_extinction_round": {},
            "win_counts": {}  # How many trials each structure "won" (most resources)
        }
        
        for struct in structures:
            survivals = [t["survival"][struct] for t in trials]
            wins = [t["final_standings"][struct]["wins"] for t in trials]
            resources = [t["final_standings"][struct]["resources"] for t in trials]
            capabilities = [t["final_standings"][struct]["final_capability"] for t in trials]
            
            aggregated["survival_rates"][struct] = sum(survivals) / len(survivals)
            aggregated["avg_wins"][struct] = sum(wins) / len(wins)
            aggregated["avg_final_resources"][struct] = sum(resources) / len(resources)
            aggregated["avg_final_capability"][struct] = sum(capabilities) / len(capabilities)
            
            # Extinction rounds (only for trials where they went extinct)
            extinction_rounds = [
                t["extinction_rounds"].get(struct, self.config.n_rounds) 
                for t in trials
            ]
            aggregated["avg_extinction_round"][struct] = sum(extinction_rounds) / len(extinction_rounds)
        
        # Count trial wins
        for struct in structures:
            aggregated["win_counts"][struct] = 0
        
        for trial in trials:
            # Winner is the one with most resources at end
            winner = max(
                trial["final_standings"].items(),
                key=lambda x: x[1]["resources"]
            )[0]
            aggregated["win_counts"][winner] += 1
        
        return aggregated


def run_all_tuning_experiments():
    """Run all tuning experiments."""
    results = {}
    
    # Baseline (original settings)
    print("\n" + "="*70)
    print("EXPERIMENT 28: UBUNTU DOMINANCE TUNING")
    print("="*70)
    
    # Baseline
    config = TuningConfig()
    exp = TuningExperiment(config)
    results["baseline"] = exp.run_experiment("Baseline (Original Settings)")
    
    # Experiment A: Disable Knowledge Sharing
    config_a = TuningConfig(disable_knowledge_sharing=True)
    exp_a = TuningExperiment(config_a)
    results["no_knowledge_sharing"] = exp_a.run_experiment("A: No Knowledge Sharing")
    
    # Experiment B: Higher Starting Resources
    config_b = TuningConfig(starting_resources=100.0)
    exp_b = TuningExperiment(config_b)
    results["high_resources"] = exp_b.run_experiment("B: High Starting Resources (100)")
    
    # Experiment C: Capability Cap with Diminishing Returns
    config_c = TuningConfig(
        max_capability=0.8,
        template_diminishing_returns=True,
        diminishing_factor=0.5
    )
    exp_c = TuningExperiment(config_c)
    results["capped_capability"] = exp_c.run_experiment("C: Capped Capability + Diminishing Returns")
    
    # Experiment D: Free Riders in Ubuntu
    config_d = TuningConfig(free_rider_fraction=0.2)
    exp_d = TuningExperiment(config_d)
    results["free_riders"] = exp_d.run_experiment("D: 20% Free Riders in Ubuntu")
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary of all experiments."""
    print("\n" + "="*70)
    print("SUMMARY: UBUNTU DOMINANCE TUNING RESULTS")
    print("="*70)
    
    structures = ["meritocracy", "guild", "obligation", "ubuntu", "rotating", "monastic"]
    
    for exp_name, exp_results in results.items():
        print(f"\n{exp_results['experiment_name']}")
        print("-" * 50)
        
        # Win counts
        print("Trial Wins:")
        for struct in structures:
            wins = exp_results["win_counts"][struct]
            total = exp_results["n_trials"]
            pct = wins / total * 100
            marker = "***" if struct == "ubuntu" else ""
            print(f"  {struct:15s}: {wins}/{total} ({pct:5.1f}%) {marker}")
        
        # Survival rates
        print("\nSurvival Rates:")
        for struct in structures:
            rate = exp_results["survival_rates"][struct] * 100
            print(f"  {struct:15s}: {rate:5.1f}%")
        
        # Average final capability
        print("\nAvg Final Capability:")
        for struct in structures:
            cap = exp_results["avg_final_capability"][struct]
            print(f"  {struct:15s}: {cap:.3f}")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Compare Ubuntu wins across experiments
    baseline_ubuntu_wins = results["baseline"]["win_counts"]["ubuntu"]
    no_sharing_ubuntu_wins = results["no_knowledge_sharing"]["win_counts"]["ubuntu"]
    high_res_ubuntu_wins = results["high_resources"]["win_counts"]["ubuntu"]
    capped_ubuntu_wins = results["capped_capability"]["win_counts"]["ubuntu"]
    free_rider_ubuntu_wins = results["free_riders"]["win_counts"]["ubuntu"]
    
    total_trials = results["baseline"]["n_trials"]
    
    print(f"\nUbuntu Win Rate Comparison:")
    print(f"  Baseline:              {baseline_ubuntu_wins}/{total_trials} ({baseline_ubuntu_wins/total_trials*100:.0f}%)")
    print(f"  No Knowledge Sharing:  {no_sharing_ubuntu_wins}/{total_trials} ({no_sharing_ubuntu_wins/total_trials*100:.0f}%)")
    print(f"  High Resources:        {high_res_ubuntu_wins}/{total_trials} ({high_res_ubuntu_wins/total_trials*100:.0f}%)")
    print(f"  Capped Capability:     {capped_ubuntu_wins}/{total_trials} ({capped_ubuntu_wins/total_trials*100:.0f}%)")
    print(f"  With Free Riders:      {free_rider_ubuntu_wins}/{total_trials} ({free_rider_ubuntu_wins/total_trials*100:.0f}%)")
    
    # Interpretations
    print("\nInterpretations:")
    
    if no_sharing_ubuntu_wins < baseline_ubuntu_wins * 0.5:
        print("  ✓ A: Knowledge sharing IS the key mechanism (Ubuntu loses without it)")
    else:
        print("  ✗ A: Knowledge sharing is NOT the key mechanism (Ubuntu still wins)")
    
    if high_res_ubuntu_wins >= baseline_ubuntu_wins * 0.8:
        print("  ✓ B: Finding is robust to resource levels")
    else:
        print("  ✗ B: Finding is sensitive to resource levels")
    
    if capped_ubuntu_wins >= baseline_ubuntu_wins * 0.8:
        print("  ✓ C: Collective coordination matters, not just capability growth")
    else:
        print("  ✗ C: Capability growth is the main driver")
    
    if free_rider_ubuntu_wins < baseline_ubuntu_wins * 0.5:
        print("  ✓ D: Ubuntu IS vulnerable to free-riders")
    else:
        print("  ✗ D: Ubuntu is robust to free-riders")


if __name__ == "__main__":
    results = run_all_tuning_experiments()
