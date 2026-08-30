"""
Experiment 34B: Team Task Coordination

Research Question: Does self-selection work for multi-agent tasks?

Task Types:
1. Solo: Single agent (baseline from Exp 32)
2. Additive: 2 agents, success = avg(capabilities)
3. Complementary: 2 agents, both must succeed
4. Weakest link: 3 agents, success = min(capabilities)
5. Best shot: 3 agents, success = max(capabilities)

Hypotheses:
H1: Self-selection degrades for team tasks
H2: Specialization emerges for complementary tasks
H3: Team formation becomes the bottleneck
H4: Visibility matters more for teams
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, Template, create_population


class TeamTaskType(Enum):
    SOLO = "solo"                    # 1 agent
    ADDITIVE = "additive"            # 2 agents, avg capability
    COMPLEMENTARY = "complementary"  # 2 agents, both must succeed
    WEAKEST_LINK = "weakest_link"    # 3 agents, min capability
    BEST_SHOT = "best_shot"          # 3 agents, max capability


@dataclass
class TeamTask:
    """Task that may require multiple agents."""
    id: str
    difficulty: float
    reward: float
    task_type: TeamTaskType
    required_agents: int
    
    @staticmethod
    def create(task_id: str, task_type: TeamTaskType, difficulty: float = None) -> 'TeamTask':
        if difficulty is None:
            difficulty = random.uniform(0.3, 0.7)
        
        required = {
            TeamTaskType.SOLO: 1,
            TeamTaskType.ADDITIVE: 2,
            TeamTaskType.COMPLEMENTARY: 2,
            TeamTaskType.WEAKEST_LINK: 3,
            TeamTaskType.BEST_SHOT: 3,
        }
        
        return TeamTask(
            id=task_id,
            difficulty=difficulty,
            reward=1.0,
            task_type=task_type,
            required_agents=required[task_type]
        )


@dataclass
class TeamOutcome:
    """Result of a team task attempt."""
    success: bool
    team: List[Agent]
    task: TeamTask
    individual_successes: List[bool]
    team_capability: float


class TeamCoordinator:
    """Coordinates team formation and task execution."""
    
    def __init__(self, visibility: str = "public"):
        self.visibility = visibility  # "public" or "private"
        self.team_history: List[Tuple[List[str], bool]] = []
    
    def get_observed_capability(self, agent: Agent, observer: Optional[Agent] = None) -> float:
        """Get capability as observed by another agent."""
        if self.visibility == "public":
            return agent.effective_capability
        else:
            # Private: only know own capability
            if observer is None or observer.id == agent.id:
                return agent.effective_capability
            return 0.5  # Assume average
    
    def form_team_self_selection(
        self, 
        task: TeamTask, 
        agents: List[Agent]
    ) -> List[Agent]:
        """Form team through self-selection (volunteering)."""
        available = [a for a in agents if a.reputation > 0.1]
        if len(available) < task.required_agents:
            available = agents
        
        # Agents volunteer based on own capability
        volunteers = []
        for agent in available:
            own_cap = agent.effective_capability
            if own_cap > task.difficulty * 0.4:
                # Volunteer score based on capability and task difficulty
                score = own_cap - task.difficulty * 0.3
                volunteers.append((agent, score))
        
        # Sort by score and take top N
        volunteers.sort(key=lambda x: -x[1])
        team = [v[0] for v in volunteers[:task.required_agents]]
        
        # If not enough volunteers, add random agents
        while len(team) < task.required_agents:
            remaining = [a for a in available if a not in team]
            if remaining:
                team.append(random.choice(remaining))
            else:
                break
        
        return team
    
    def form_team_capability_matched(
        self,
        task: TeamTask,
        agents: List[Agent]
    ) -> List[Agent]:
        """Form team through capability matching (centralized)."""
        available = [a for a in agents if a.reputation > 0.1]
        if len(available) < task.required_agents:
            available = agents
        
        # Sort by observed capability
        sorted_agents = sorted(
            available,
            key=lambda a: self.get_observed_capability(a),
            reverse=True
        )
        
        # Take top N agents
        return sorted_agents[:task.required_agents]
    
    def form_team_random(
        self,
        task: TeamTask,
        agents: List[Agent]
    ) -> List[Agent]:
        """Form team randomly."""
        available = [a for a in agents if a.reputation > 0.1]
        if len(available) < task.required_agents:
            available = agents
        
        return random.sample(available, min(task.required_agents, len(available)))
    
    def execute_team_task(
        self,
        team: List[Agent],
        task: TeamTask
    ) -> TeamOutcome:
        """Execute a team task."""
        # Calculate individual success probabilities
        individual_probs = []
        for agent in team:
            prob = agent.effective_capability * 0.8 * (1 - task.difficulty * 0.6)
            prob = max(0.0, min(1.0, prob))
            individual_probs.append(prob)
        
        # Determine individual successes
        individual_successes = [random.random() < p for p in individual_probs]
        
        # Calculate team success based on task type
        if task.task_type == TeamTaskType.SOLO:
            team_success = individual_successes[0] if individual_successes else False
            team_capability = team[0].effective_capability if team else 0
            
        elif task.task_type == TeamTaskType.ADDITIVE:
            # Average capability determines success
            team_capability = sum(a.effective_capability for a in team) / len(team)
            team_prob = team_capability * 0.8 * (1 - task.difficulty * 0.6)
            team_success = random.random() < team_prob
            
        elif task.task_type == TeamTaskType.COMPLEMENTARY:
            # Both must succeed
            team_success = all(individual_successes)
            team_capability = min(a.effective_capability for a in team)
            
        elif task.task_type == TeamTaskType.WEAKEST_LINK:
            # Success determined by weakest member
            team_capability = min(a.effective_capability for a in team)
            team_prob = team_capability * 0.8 * (1 - task.difficulty * 0.6)
            team_success = random.random() < team_prob
            
        elif task.task_type == TeamTaskType.BEST_SHOT:
            # Success determined by best member
            team_capability = max(a.effective_capability for a in team)
            team_prob = team_capability * 0.8 * (1 - task.difficulty * 0.6)
            team_success = random.random() < team_prob
        
        else:
            team_success = False
            team_capability = 0
        
        # Update agent histories
        for agent, success in zip(team, individual_successes):
            if success:
                agent.success_history.append(True)
            else:
                agent.failure_history.append(True)
        
        # Record team history
        self.team_history.append(([a.id for a in team], team_success))
        
        return TeamOutcome(
            success=team_success,
            team=team,
            task=task,
            individual_successes=individual_successes,
            team_capability=team_capability
        )


@dataclass
class TeamExperimentConfig:
    """Configuration for team experiment."""
    task_type: TeamTaskType
    formation_method: str  # "self_selection", "capability_matched", "random"
    visibility: str = "public"  # "public" or "private"
    sharing_enabled: bool = True
    
    @property
    def name(self) -> str:
        return f"{self.task_type.value}_{self.formation_method}_{self.visibility}"


def run_condition(
    config: TeamExperimentConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    agents = create_population(n_agents, prefix=f"{config.name}_{seed}")
    coordinator = TeamCoordinator(visibility=config.visibility)
    
    successes = 0
    total = 0
    team_capabilities = []
    team_sizes = []
    
    for round_num in range(n_rounds):
        # Generate task
        task = TeamTask.create(f"task_{round_num}", config.task_type)
        
        # Form team
        if config.formation_method == "self_selection":
            team = coordinator.form_team_self_selection(task, agents)
        elif config.formation_method == "capability_matched":
            team = coordinator.form_team_capability_matched(task, agents)
        else:
            team = coordinator.form_team_random(task, agents)
        
        # Execute task
        if len(team) >= task.required_agents:
            outcome = coordinator.execute_team_task(team, task)
            total += 1
            if outcome.success:
                successes += 1
            team_capabilities.append(outcome.team_capability)
            team_sizes.append(len(team))
        
        # Share templates if enabled
        if config.sharing_enabled:
            all_templates = {}
            for agent in agents:
                for t in agent.template_library:
                    if t.id not in all_templates:
                        all_templates[t.id] = t
            for agent in agents:
                agent_ids = {t.id for t in agent.template_library}
                for tid, template in all_templates.items():
                    if tid not in agent_ids:
                        agent.template_library.append(template)
    
    cooperation_rate = successes / total if total > 0 else 0
    mean_team_capability = np.mean(team_capabilities) if team_capabilities else 0
    
    return {
        "cooperation_rate": cooperation_rate,
        "mean_team_capability": mean_team_capability,
        "mean_team_size": np.mean(team_sizes) if team_sizes else 0,
        "total_tasks": total,
    }


def run_experiment(n_rounds: int = 100, n_seeds: int = 10, n_agents: int = 30) -> Dict[str, Any]:
    """Run the full experiment."""
    print("=" * 70)
    print("EXPERIMENT 34B: TEAM TASK COORDINATION")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test all combinations
    task_types = list(TeamTaskType)
    formation_methods = ["self_selection", "capability_matched", "random"]
    
    for task_type in task_types:
        for method in formation_methods:
            config = TeamExperimentConfig(
                task_type=task_type,
                formation_method=method,
                visibility="public"
            )
            
            print(f"Running {config.name}...", end=" ", flush=True)
            
            seed_results = []
            for seed in range(n_seeds):
                metrics = run_condition(config, n_rounds, n_agents, seed)
                seed_results.append(metrics)
            
            aggregated = {}
            for key in seed_results[0].keys():
                values = [r[key] for r in seed_results]
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            
            results[config.name] = aggregated
            print(f"coop={aggregated['cooperation_rate']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results."""
    analysis = {}
    
    # Extract cooperation rates
    coop_rates = {name: r["cooperation_rate"]["mean"] for name, r in results.items()}
    
    # Compare formation methods by task type
    for task_type in TeamTaskType:
        type_name = task_type.value
        self_sel = coop_rates.get(f"{type_name}_self_selection_public", 0)
        cap_match = coop_rates.get(f"{type_name}_capability_matched_public", 0)
        rand = coop_rates.get(f"{type_name}_random_public", 0)
        
        analysis[f"{type_name}_self_vs_cap"] = self_sel - cap_match
        analysis[f"{type_name}_self_vs_random"] = self_sel - rand
        analysis[f"{type_name}_best_method"] = max(
            [("self_selection", self_sel), ("capability_matched", cap_match), ("random", rand)],
            key=lambda x: x[1]
        )[0]
    
    # Overall: does self-selection still win?
    self_sel_avg = np.mean([v for k, v in coop_rates.items() if "self_selection" in k])
    cap_match_avg = np.mean([v for k, v in coop_rates.items() if "capability_matched" in k])
    random_avg = np.mean([v for k, v in coop_rates.items() if "_random_" in k])
    
    analysis["overall_self_selection"] = self_sel_avg
    analysis["overall_capability_matched"] = cap_match_avg
    analysis["overall_random"] = random_avg
    analysis["self_selection_advantage"] = self_sel_avg - cap_match_avg
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Team Task Coordination")
    print("=" * 70)
    
    # Results matrix
    print("\n--- Cooperation Rate by Task Type and Formation Method ---")
    print(f"{'Task Type':>15} {'Self-Select':>12} {'Cap-Match':>12} {'Random':>12} {'Best':>15}")
    print("-" * 70)
    
    for task_type in TeamTaskType:
        type_name = task_type.value
        self_sel = results.get(f"{type_name}_self_selection_public", {}).get("cooperation_rate", {}).get("mean", 0)
        cap_match = results.get(f"{type_name}_capability_matched_public", {}).get("cooperation_rate", {}).get("mean", 0)
        rand = results.get(f"{type_name}_random_public", {}).get("cooperation_rate", {}).get("mean", 0)
        best = analysis.get(f"{type_name}_best_method", "")
        
        print(f"{type_name:>15} {self_sel:>12.3f} {cap_match:>12.3f} {rand:>12.3f} {best:>15}")
    
    # Overall comparison
    print("\n--- Overall Formation Method Comparison ---")
    print(f"Self-selection:     {analysis['overall_self_selection']:.3f}")
    print(f"Capability-matched: {analysis['overall_capability_matched']:.3f}")
    print(f"Random:             {analysis['overall_random']:.3f}")
    print(f"Self-selection advantage: {analysis['self_selection_advantage']:+.3f}")
    
    # Hypothesis evaluation
    print("\n--- Hypothesis Evaluation ---")
    
    # H1: Self-selection degrades for team tasks
    solo_self = results.get("solo_self_selection_public", {}).get("cooperation_rate", {}).get("mean", 0)
    team_self = np.mean([
        results.get(f"{t.value}_self_selection_public", {}).get("cooperation_rate", {}).get("mean", 0)
        for t in TeamTaskType if t != TeamTaskType.SOLO
    ])
    h1 = team_self < solo_self
    print(f"H1 (Self-selection degrades for teams): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    print(f"    Solo: {solo_self:.3f}, Team avg: {team_self:.3f}")
    
    # H2: Does self-selection still beat capability-matching for teams?
    h2 = analysis["self_selection_advantage"] > 0
    print(f"H2 (Self-selection still wins for teams): {'SUPPORTED' if h2 else 'NOT SUPPORTED'}")
    print(f"    Advantage: {analysis['self_selection_advantage']:+.3f}")
    
    # Key finding
    print("\n--- KEY FINDING ---")
    if h1 and h2:
        print("Self-selection degrades for teams but STILL outperforms capability-matching")
    elif h1 and not h2:
        print("Self-selection degrades for teams and capability-matching becomes better")
    elif not h1 and h2:
        print("Self-selection works equally well for teams and still outperforms")
    else:
        print("Self-selection works equally well for teams but capability-matching catches up")


def main():
    results = run_experiment(n_rounds=100, n_seeds=10, n_agents=30)
    analysis = analyze_results(results)
    print_results(results, analysis)
    
    # Save
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
    
    output = {"conditions": results, "analysis": analysis}
    with open("results/experiment_34b_teams.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_34b_teams.json")


if __name__ == "__main__":
    main()
