"""
Experiment 31: Template Sharing Rate Optimization

Research Question: What's the optimal template sharing rate?
Is there a point of diminishing returns?

Design: Test sharing rates from 0% to 100% in 25% increments.

Hypotheses:
H1: Diminishing returns - early sharing gains more than later
H2: 50% achieves >80% of full pooling benefit
H3: Capability scales more linearly than cooperation
H4: Efficiency (cooperation/overhead) peaks at moderate sharing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import json

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, create_population
from experiments.social_structures.structures.base import BaseStructure
from experiments.social_structures.competition.firm import Firm


@dataclass
class SharingRateConfig:
    """Configuration for sharing rate experiment."""
    sharing_rate: float  # 0.0 to 1.0
    success_reward: float = 0.1
    failure_penalty: float = 0.3
    redemption_bonus: float = 0.2


class SharingRateStructure(BaseStructure):
    """
    Structure with configurable template sharing rate.
    
    sharing_rate = 0.0: No sharing (baseline)
    sharing_rate = 1.0: Full pooling (Ubuntu-style)
    sharing_rate = 0.5: Each agent shares 50% of templates with 50% of peers
    """
    
    def __init__(self, config: SharingRateConfig):
        self.config = config
        self.sharing_rate = config.sharing_rate
        self._name = f"sharing_{int(config.sharing_rate * 100)}pct"
        self.templates_transferred = 0  # Track overhead
    
    @property
    def name(self) -> str:
        return self._name
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """Agents volunteer based on capability."""
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        volunteers = []
        for agent in available:
            if agent.effective_capability > task.difficulty * 0.5:
                volunteers.append(agent)
        
        if not volunteers:
            return [max(available, key=lambda a: a.effective_capability)]
        
        return [max(volunteers, key=lambda a: a.effective_capability)]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Standard difficulty-weighted reputation update."""
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            delta = self.config.success_reward * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + delta)
            agent.learn_template(outcome.task, True)
            
            if agent.is_struggling():
                agent.reputation += self.config.redemption_bonus
                agent.recovered_recently = True
        else:
            delta = self.config.failure_penalty / difficulty_mult
            agent.reputation = max(0.0, agent.reputation - delta)
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """Share templates at configured rate."""
        if self.sharing_rate == 0:
            return  # No sharing
        
        if self.sharing_rate >= 1.0:
            # Full pooling (Ubuntu-style)
            all_template_ids = set()
            template_map = {}
            
            for agent in agents:
                for t in agent.template_library:
                    if t.id not in all_template_ids:
                        all_template_ids.add(t.id)
                        template_map[t.id] = t
            
            for agent in agents:
                agent_ids = {t.id for t in agent.template_library}
                for tid, template in template_map.items():
                    if tid not in agent_ids:
                        agent.template_library.append(template)
                        self.templates_transferred += 1
        else:
            # Partial sharing
            for agent in agents:
                if not agent.template_library:
                    continue
                
                # Select templates to share (fraction of library)
                n_to_share = max(1, int(len(agent.template_library) * self.sharing_rate))
                if n_to_share > len(agent.template_library):
                    n_to_share = len(agent.template_library)
                templates_to_share = random.sample(agent.template_library, n_to_share)
                
                # Share with fraction of other agents
                other_agents = [a for a in agents if a.id != agent.id]
                if not other_agents:
                    continue
                n_recipients = max(1, int(len(other_agents) * self.sharing_rate))
                if n_recipients > len(other_agents):
                    n_recipients = len(other_agents)
                recipients = random.sample(other_agents, n_recipients)
                
                for template in templates_to_share:
                    for recipient in recipients:
                        if recipient.receive_template(template):
                            self.templates_transferred += 1
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        if from_agent.template_library:
            template = random.choice(from_agent.template_library)
            if to_agent.receive_template(template):
                self.templates_transferred += 1
                return True
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        return self.status_from_percentile(agent, all_agents)
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        pass


def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient."""
    if not values or len(values) <= 1:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
    return (2 * cumsum) / (n * total) - (n + 1) / n


def run_condition(
    sharing_rate: float,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition with given sharing rate."""
    random.seed(seed)
    np.random.seed(seed)
    
    config = SharingRateConfig(sharing_rate=sharing_rate)
    structure = SharingRateStructure(config)
    agents = create_population(n_agents, prefix=f"s{int(sharing_rate*100)}_{seed}")
    
    firm = Firm(
        id=f"sharing_{int(sharing_rate*100)}_{seed}",
        structure=structure,
        agents=agents,
        resources=10.0
    )
    
    # Run simulation
    for round_num in range(n_rounds):
        firm.run_internal_round()
        
        difficulty = random.uniform(0.3, 0.7)
        task = Task(id=f"task_{round_num}", difficulty=difficulty, reward=1.0)
        firm.attempt_task(task)
        
        firm.end_internal_round()
    
    # Collect metrics
    n = len(agents)
    total_tasks = sum(len(a.success_history) + len(a.failure_history) for a in agents)
    successes = sum(len(a.success_history) for a in agents)
    cooperation_rate = successes / total_tasks if total_tasks > 0 else 0
    
    reps = [a.reputation for a in agents]
    gini = calculate_gini(reps)
    
    mean_capability = sum(a.effective_capability for a in agents) / n if n > 0 else 0
    
    total_templates = sum(len(a.template_library) for a in agents)
    unique_templates = len(set(t.id for a in agents for t in a.template_library))
    diffusion = total_templates / (unique_templates * n) if unique_templates > 0 and n > 0 else 0
    
    return {
        "cooperation_rate": cooperation_rate,
        "mean_capability": mean_capability,
        "gini": gini,
        "knowledge_diffusion": diffusion,
        "total_templates": total_templates,
        "unique_templates": unique_templates,
        "templates_transferred": structure.templates_transferred,
    }


def run_experiment(
    sharing_rates: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    n_rounds: int = 100,
    n_seeds: int = 10,
    n_agents: int = 30
) -> Dict[str, Any]:
    """Run the full sharing rate experiment."""
    print("=" * 70)
    print("EXPERIMENT 31: TEMPLATE SHARING RATE OPTIMIZATION")
    print("=" * 70)
    print(f"Sharing rates: {sharing_rates}")
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    for rate in sharing_rates:
        print(f"Running {int(rate*100)}% sharing...", end=" ")
        seed_results = []
        
        for seed in range(n_seeds):
            metrics = run_condition(rate, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        # Aggregate
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        results[f"rate_{int(rate*100)}"] = aggregated
        print(f"done (coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"cap={aggregated['mean_capability']['mean']:.3f})")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results for diminishing returns and efficiency."""
    rates = [0, 25, 50, 75, 100]
    
    # Extract means
    cooperation = [results[f"rate_{r}"]["cooperation_rate"]["mean"] for r in rates]
    capability = [results[f"rate_{r}"]["mean_capability"]["mean"] for r in rates]
    overhead = [results[f"rate_{r}"]["templates_transferred"]["mean"] for r in rates]
    
    # Calculate marginal gains
    marginal_coop = [cooperation[i] - cooperation[i-1] for i in range(1, len(cooperation))]
    marginal_cap = [capability[i] - capability[i-1] for i in range(1, len(capability))]
    
    # Calculate efficiency (cooperation gain per 1000 transfers)
    efficiency = []
    for i in range(1, len(rates)):
        coop_gain = cooperation[i] - cooperation[0]
        if overhead[i] > 0:
            eff = (coop_gain / overhead[i]) * 1000
        else:
            eff = 0
        efficiency.append(eff)
    
    # Find optimal rate (highest efficiency)
    if efficiency:
        optimal_idx = efficiency.index(max(efficiency))
        optimal_rate = rates[optimal_idx + 1]
    else:
        optimal_rate = 0
    
    # Calculate % of full benefit at 50%
    full_benefit = cooperation[-1] - cooperation[0]
    half_benefit = cooperation[2] - cooperation[0]  # 50% rate
    pct_at_50 = (half_benefit / full_benefit * 100) if full_benefit > 0 else 0
    
    return {
        "rates": rates,
        "cooperation": cooperation,
        "capability": capability,
        "overhead": overhead,
        "marginal_cooperation": marginal_coop,
        "marginal_capability": marginal_cap,
        "efficiency": efficiency,
        "optimal_rate": optimal_rate,
        "pct_benefit_at_50": pct_at_50,
    }


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Template Sharing Rate Optimization")
    print("=" * 70)
    
    # Main results table
    print("\n--- Results by Sharing Rate ---")
    print(f"{'Rate':>8} {'Cooperation':>12} {'Capability':>12} {'Overhead':>12} {'Efficiency':>12}")
    print("-" * 60)
    
    rates = [0, 25, 50, 75, 100]
    for i, rate in enumerate(rates):
        coop = results[f"rate_{rate}"]["cooperation_rate"]["mean"]
        cap = results[f"rate_{rate}"]["mean_capability"]["mean"]
        overhead = results[f"rate_{rate}"]["templates_transferred"]["mean"]
        
        if i > 0:
            eff = analysis["efficiency"][i-1]
            print(f"{rate:>7}% {coop:>12.3f} {cap:>12.3f} {overhead:>12.0f} {eff:>12.3f}")
        else:
            print(f"{rate:>7}% {coop:>12.3f} {cap:>12.3f} {overhead:>12.0f} {'N/A':>12}")
    
    # Marginal gains
    print("\n--- Marginal Gains (per 25% increment) ---")
    print(f"{'Increment':>15} {'Δ Cooperation':>15} {'Δ Capability':>15}")
    print("-" * 50)
    
    increments = ["0→25%", "25→50%", "50→75%", "75→100%"]
    for i, inc in enumerate(increments):
        d_coop = analysis["marginal_cooperation"][i]
        d_cap = analysis["marginal_capability"][i]
        print(f"{inc:>15} {d_coop:>+15.4f} {d_cap:>+15.4f}")
    
    # Hypothesis evaluation
    print("\n--- Hypothesis Evaluation ---")
    
    # H1: Diminishing returns
    marginal = analysis["marginal_cooperation"]
    h1_result = marginal[0] > marginal[-1]  # First increment > last increment
    print(f"H1 (Diminishing returns): {'SUPPORTED' if h1_result else 'NOT SUPPORTED'}")
    print(f"    First increment: {marginal[0]:+.4f}, Last increment: {marginal[-1]:+.4f}")
    
    # H2: 50% achieves >80% of benefit
    h2_result = analysis["pct_benefit_at_50"] > 80
    print(f"H2 (50% achieves >80% of benefit): {'SUPPORTED' if h2_result else 'NOT SUPPORTED'}")
    print(f"    50% achieves {analysis['pct_benefit_at_50']:.1f}% of full benefit")
    
    # H3: Capability scales more linearly
    # Compare variance of marginal gains
    coop_var = np.var(analysis["marginal_cooperation"])
    cap_var = np.var(analysis["marginal_capability"])
    h3_result = cap_var < coop_var
    print(f"H3 (Capability more linear): {'SUPPORTED' if h3_result else 'NOT SUPPORTED'}")
    print(f"    Cooperation variance: {coop_var:.6f}, Capability variance: {cap_var:.6f}")
    
    # H4: Efficiency peaks at moderate sharing
    h4_result = analysis["optimal_rate"] in [25, 50]
    print(f"H4 (Efficiency peaks at moderate sharing): {'SUPPORTED' if h4_result else 'NOT SUPPORTED'}")
    print(f"    Optimal rate: {analysis['optimal_rate']}%")
    
    # Summary
    print("\n--- SUMMARY ---")
    print(f"Optimal sharing rate for efficiency: {analysis['optimal_rate']}%")
    print(f"50% sharing achieves {analysis['pct_benefit_at_50']:.1f}% of full pooling benefit")
    
    if analysis["pct_benefit_at_50"] > 80:
        print("\nRECOMMENDATION: Moderate sharing (50%) is near-optimal.")
        print("Full pooling has diminishing returns relative to overhead.")
    else:
        print("\nRECOMMENDATION: Higher sharing rates may be needed for full benefit.")


def main():
    """Run the full experiment."""
    # Run experiment
    results = run_experiment(
        sharing_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_rounds=100,
        n_seeds=10,
        n_agents=30
    )
    
    # Analyze
    analysis = analyze_results(results)
    
    # Print results
    print_results(results, analysis)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    output = {
        "conditions": results,
        "analysis": analysis,
    }
    
    with open("results/experiment_31_sharing_rate.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_31_sharing_rate.json")
    
    return output


if __name__ == "__main__":
    main()
