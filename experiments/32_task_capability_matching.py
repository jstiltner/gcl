"""
Experiment 32: Task-Capability Matching Factorial

Research Question: How do we convert capability into cooperation?

Background: Experiments 30-31 revealed that template sharing increases capability (+86%)
but cooperation stays flat (~55%). This suggests task-capability matching is the missing
mechanism that converts capability into cooperation.

Design: 4×4×4 factorial testing:
- Factor 1: Capability Visibility (private, public, reputation_proxied, separate_signals)
- Factor 2: Matching Mechanism (random, self_selection, capability_matched, difficulty_weighted)
- Factor 3: Task Distribution (fixed_medium, uniform, adaptive, bimodal)

Phase 1: Priority subset (16 conditions) - Visibility × Mechanism with uniform tasks
Phase 2: Best mechanism × Task Distribution (4 conditions)
Phase 3: Key interactions (selected)

Hypotheses:
H1: Public visibility > Private visibility for cooperation
H2: Capability-matched > Self-selection > Random
H3: Adaptive tasks unlock capability benefits
H4: Reputation-proxied ≈ Public capability
H5: Difficulty-weighted reputation is robust across conditions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from itertools import product

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, Template, create_population
from experiments.social_structures.structures.base import BaseStructure
from experiments.social_structures.competition.firm import Firm


class CapabilityVisibility(Enum):
    """How capability information is distributed."""
    PRIVATE = "private"           # Only agent knows own capability
    PUBLIC = "public"             # All agents know all capabilities
    REPUTATION_PROXIED = "rep_proxied"  # Reputation tracks capability (with noise)
    SEPARATE_SIGNALS = "separate"  # Capability and reputation are distinct, both visible


class MatchingMechanism(Enum):
    """How tasks are matched to agents."""
    RANDOM = "random"             # Tasks assigned randomly
    SELF_SELECTION = "self_select"  # Agents volunteer based on assessment
    CAPABILITY_MATCHED = "cap_match"  # Tasks assigned to best-fit agent
    DIFFICULTY_WEIGHTED = "diff_weight"  # Harder tasks → higher reputation reward


class TaskDistribution(Enum):
    """How task difficulties are distributed."""
    FIXED_MEDIUM = "fixed"        # All tasks difficulty = 0.5
    UNIFORM = "uniform"           # Difficulty ~ U(0.3, 0.7)
    ADAPTIVE = "adaptive"         # Difficulty scales with mean capability
    BIMODAL = "bimodal"           # 50% easy (0.2-0.4), 50% hard (0.6-0.8)


@dataclass
class MatchingConfig:
    """Configuration for a single experimental condition."""
    visibility: CapabilityVisibility
    mechanism: MatchingMechanism
    task_dist: TaskDistribution
    sharing_rate: float = 1.0  # Full sharing to isolate matching effects
    success_reward: float = 0.1
    failure_penalty: float = 0.3
    redemption_bonus: float = 0.2
    
    @property
    def condition_name(self) -> str:
        return f"{self.visibility.value}_{self.mechanism.value}_{self.task_dist.value}"


@dataclass
class AgentCapabilityInfo:
    """Capability information available about an agent."""
    agent_id: str
    true_capability: float
    observed_capability: float  # What others see (may differ from true)
    reputation: float
    
    
class MatchingStructure(BaseStructure):
    """
    Structure implementing different visibility and matching mechanisms.
    """
    
    def __init__(self, config: MatchingConfig):
        self.config = config
        self._name = config.condition_name
        self.templates_transferred = 0
        
        # Track capability observations (for public/separate visibility)
        self.capability_observations: Dict[str, float] = {}
        
        # Track matching quality
        self.matching_scores: List[float] = []
        
    @property
    def name(self) -> str:
        return self._name
    
    def get_observed_capability(self, agent: Agent, observer: Optional[Agent] = None) -> float:
        """
        Get the capability of an agent as observed by another agent (or system).
        
        Visibility rules:
        - PRIVATE: Only self knows true capability
        - PUBLIC: Everyone knows true capability
        - REPUTATION_PROXIED: Capability ≈ reputation (with noise)
        - SEPARATE_SIGNALS: Both capability and reputation visible
        """
        if self.config.visibility == CapabilityVisibility.PRIVATE:
            if observer is None or observer.id == agent.id:
                return agent.effective_capability
            else:
                return 0.5  # Unknown, assume average
                
        elif self.config.visibility == CapabilityVisibility.PUBLIC:
            return agent.effective_capability
            
        elif self.config.visibility == CapabilityVisibility.REPUTATION_PROXIED:
            # Reputation is a noisy proxy for capability
            # Add noise proportional to how different rep is from capability
            noise = random.gauss(0, 0.1)
            return max(0.0, min(1.0, agent.reputation + noise))
            
        elif self.config.visibility == CapabilityVisibility.SEPARATE_SIGNALS:
            # Both are visible - return true capability
            return agent.effective_capability
            
        return agent.effective_capability
    
    def generate_task(self, round_num: int, population_capability: float) -> Task:
        """Generate a task based on the task distribution."""
        
        if self.config.task_dist == TaskDistribution.FIXED_MEDIUM:
            difficulty = 0.5
            
        elif self.config.task_dist == TaskDistribution.UNIFORM:
            difficulty = random.uniform(0.3, 0.7)
            
        elif self.config.task_dist == TaskDistribution.ADAPTIVE:
            # Difficulty scales with population capability
            # Base difficulty + scaling factor * (capability - 0.5)
            base = 0.5
            scaling = 0.6  # How responsive difficulty is to capability
            difficulty = base + scaling * (population_capability - 0.5)
            difficulty = max(0.2, min(0.8, difficulty))  # Clamp
            
        elif self.config.task_dist == TaskDistribution.BIMODAL:
            # 50% easy, 50% hard
            if random.random() < 0.5:
                difficulty = random.uniform(0.2, 0.4)  # Easy
            else:
                difficulty = random.uniform(0.6, 0.8)  # Hard
                
        else:
            difficulty = 0.5
            
        return Task(
            id=f"task_{round_num}",
            difficulty=difficulty,
            reward=1.0
        )
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        Select agent(s) for a task based on matching mechanism.
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        if self.config.mechanism == MatchingMechanism.RANDOM:
            # Random assignment
            return [random.choice(available)]
            
        elif self.config.mechanism == MatchingMechanism.SELF_SELECTION:
            # Agents volunteer based on their own assessment
            volunteers = []
            for agent in available:
                # Agent knows own capability (always)
                own_cap = agent.effective_capability
                
                # Agent estimates task difficulty (with noise if not public)
                if self.config.visibility in [CapabilityVisibility.PUBLIC, 
                                               CapabilityVisibility.SEPARATE_SIGNALS]:
                    perceived_difficulty = task.difficulty
                else:
                    # Noisy estimate
                    perceived_difficulty = task.difficulty + random.gauss(0, 0.1)
                    perceived_difficulty = max(0.1, min(0.9, perceived_difficulty))
                
                # Volunteer if capability seems sufficient
                if own_cap > perceived_difficulty * 0.6:
                    volunteers.append(agent)
            
            if not volunteers:
                # No volunteers - pick highest capability
                return [max(available, key=lambda a: a.effective_capability)]
            
            # Among volunteers, pick highest capability
            return [max(volunteers, key=lambda a: a.effective_capability)]
            
        elif self.config.mechanism == MatchingMechanism.CAPABILITY_MATCHED:
            # System matches task to best-fit agent based on observed capability
            # Best fit = capability closest to (but above) difficulty
            
            def match_score(agent: Agent) -> float:
                obs_cap = self.get_observed_capability(agent)
                if obs_cap < task.difficulty * 0.5:
                    return -1  # Too low capability
                # Prefer agents whose capability is just above difficulty
                excess = obs_cap - task.difficulty
                if excess < 0:
                    return excess  # Negative = below difficulty
                return 1.0 / (1.0 + excess)  # Higher score for closer match
            
            scored = [(a, match_score(a)) for a in available]
            scored = [(a, s) for a, s in scored if s > 0]
            
            if not scored:
                # No good matches - pick highest capability
                return [max(available, key=lambda a: self.get_observed_capability(a))]
            
            # Pick best match
            best = max(scored, key=lambda x: x[1])
            return [best[0]]
            
        elif self.config.mechanism == MatchingMechanism.DIFFICULTY_WEIGHTED:
            # Agents volunteer, but harder tasks give more reputation
            # This incentivizes capable agents to take hard tasks
            
            volunteers = []
            for agent in available:
                own_cap = agent.effective_capability
                
                # Expected reputation gain from this task
                if own_cap > task.difficulty * 0.5:
                    # Higher difficulty = higher potential reward
                    expected_value = task.difficulty * self.config.success_reward * 2
                    # Volunteer if expected value is positive
                    if expected_value > 0.05:
                        volunteers.append((agent, expected_value))
            
            if not volunteers:
                return [max(available, key=lambda a: a.effective_capability)]
            
            # Pick agent with highest expected value (usually highest capability)
            best = max(volunteers, key=lambda x: x[1])
            return [best[0]]
            
        return [random.choice(available)]
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """
        Update reputation based on outcome.
        
        For DIFFICULTY_WEIGHTED mechanism, harder tasks give more reputation.
        For REPUTATION_PROXIED visibility, reputation tracks capability.
        """
        if self.config.mechanism == MatchingMechanism.DIFFICULTY_WEIGHTED:
            # Difficulty-weighted reputation update
            difficulty_mult = 1.0 + difficulty * 2.0  # Stronger weighting
        else:
            # Standard difficulty weighting
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
        
        # For REPUTATION_PROXIED, also nudge reputation toward capability
        if self.config.visibility == CapabilityVisibility.REPUTATION_PROXIED:
            # Slow drift toward true capability
            cap_diff = agent.effective_capability - agent.reputation
            agent.reputation += cap_diff * 0.1
        
        # Track matching quality
        match_quality = 1.0 - abs(agent.effective_capability - difficulty)
        self.matching_scores.append(match_quality)
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """Share templates at configured rate (default: full sharing)."""
        if self.config.sharing_rate == 0:
            return
        
        if self.config.sharing_rate >= 1.0:
            # Full pooling
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
            # Partial sharing (same as Exp 31)
            for agent in agents:
                if not agent.template_library:
                    continue
                
                n_to_share = max(1, int(len(agent.template_library) * self.config.sharing_rate))
                if n_to_share > len(agent.template_library):
                    n_to_share = len(agent.template_library)
                templates_to_share = random.sample(agent.template_library, n_to_share)
                
                other_agents = [a for a in agents if a.id != agent.id]
                if not other_agents:
                    continue
                n_recipients = max(1, int(len(other_agents) * self.config.sharing_rate))
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
    
    def get_mean_matching_quality(self) -> float:
        """Get average matching quality across all tasks."""
        if not self.matching_scores:
            return 0.0
        return sum(self.matching_scores) / len(self.matching_scores)


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
    config: MatchingConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, float]:
    """Run a single experimental condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    structure = MatchingStructure(config)
    agents = create_population(n_agents, prefix=f"{config.condition_name}_{seed}")
    
    firm = Firm(
        id=f"{config.condition_name}_{seed}",
        structure=structure,
        agents=agents,
        resources=10.0
    )
    
    # Track metrics over time
    cooperation_by_round = []
    capability_by_round = []
    
    for round_num in range(n_rounds):
        firm.run_internal_round()
        
        # Calculate population capability for adaptive tasks
        pop_capability = sum(a.effective_capability for a in agents) / len(agents)
        
        # Generate task based on distribution
        task = structure.generate_task(round_num, pop_capability)
        
        # Attempt task
        firm.attempt_task(task)
        
        firm.end_internal_round()
        
        # Track per-round metrics
        round_successes = sum(1 for a in agents if a.success_history and a.success_history[-1] if len(a.success_history) > round_num)
        cooperation_by_round.append(round_successes / n_agents if n_agents > 0 else 0)
        capability_by_round.append(pop_capability)
    
    # Collect final metrics
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
    
    # Capability utilization: actual success / expected success given capability
    expected_success = sum(a.effective_capability * 0.8 * 0.7 for a in agents)  # Assuming avg difficulty 0.5
    capability_utilization = successes / expected_success if expected_success > 0 else 0
    
    return {
        "cooperation_rate": cooperation_rate,
        "mean_capability": mean_capability,
        "gini": gini,
        "knowledge_diffusion": diffusion,
        "total_templates": total_templates,
        "unique_templates": unique_templates,
        "templates_transferred": structure.templates_transferred,
        "matching_quality": structure.get_mean_matching_quality(),
        "capability_utilization": capability_utilization,
        "final_cooperation": np.mean(cooperation_by_round[-10:]) if len(cooperation_by_round) >= 10 else cooperation_rate,
    }


def run_phase1_experiment(
    n_rounds: int = 100,
    n_seeds: int = 10,
    n_agents: int = 30
) -> Dict[str, Any]:
    """
    Phase 1: Visibility × Mechanism factorial with uniform tasks.
    16 conditions.
    """
    print("=" * 70)
    print("EXPERIMENT 32 PHASE 1: Visibility × Mechanism Factorial")
    print("=" * 70)
    print(f"Conditions: 4 visibility × 4 mechanism = 16")
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    visibilities = list(CapabilityVisibility)
    mechanisms = list(MatchingMechanism)
    
    for vis, mech in product(visibilities, mechanisms):
        config = MatchingConfig(
            visibility=vis,
            mechanism=mech,
            task_dist=TaskDistribution.UNIFORM  # Fixed for Phase 1
        )
        
        print(f"Running {config.condition_name}...", end=" ", flush=True)
        seed_results = []
        
        for seed in range(n_seeds):
            metrics = run_condition(config, n_rounds, n_agents, seed)
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
        
        results[config.condition_name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"cap={aggregated['mean_capability']['mean']:.3f}, "
              f"match={aggregated['matching_quality']['mean']:.3f}")
    
    return results


def run_phase2_experiment(
    best_visibility: CapabilityVisibility,
    best_mechanism: MatchingMechanism,
    n_rounds: int = 100,
    n_seeds: int = 10,
    n_agents: int = 30
) -> Dict[str, Any]:
    """
    Phase 2: Best mechanism × Task Distribution.
    4 conditions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 32 PHASE 2: Best Mechanism × Task Distribution")
    print("=" * 70)
    print(f"Best visibility: {best_visibility.value}")
    print(f"Best mechanism: {best_mechanism.value}")
    print(f"Task distributions: 4")
    print()
    
    results = {}
    
    for task_dist in TaskDistribution:
        config = MatchingConfig(
            visibility=best_visibility,
            mechanism=best_mechanism,
            task_dist=task_dist
        )
        
        print(f"Running {config.condition_name}...", end=" ", flush=True)
        seed_results = []
        
        for seed in range(n_seeds):
            metrics = run_condition(config, n_rounds, n_agents, seed)
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
        
        results[config.condition_name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"cap={aggregated['mean_capability']['mean']:.3f}")
    
    return results


def analyze_phase1(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Phase 1 results to find best visibility and mechanism."""
    
    # Extract cooperation rates by visibility and mechanism
    vis_means = {v.value: [] for v in CapabilityVisibility}
    mech_means = {m.value: [] for m in MatchingMechanism}
    
    for condition, metrics in results.items():
        # Parse condition name: visibility_mechanism_taskdist
        # e.g., "private_self_select_uniform" or "rep_proxied_cap_match_uniform"
        
        # Find which visibility this condition has
        vis = None
        for v in CapabilityVisibility:
            if condition.startswith(v.value + "_"):
                vis = v.value
                break
        
        # Find which mechanism this condition has
        mech = None
        for m in MatchingMechanism:
            if "_" + m.value + "_" in condition:
                mech = m.value
                break
        
        if vis is None or mech is None:
            print(f"Warning: Could not parse condition {condition}")
            continue
        
        coop = metrics["cooperation_rate"]["mean"]
        vis_means[vis].append(coop)
        mech_means[mech].append(coop)
    
    # Average by factor
    vis_avg = {v: np.mean(vals) for v, vals in vis_means.items()}
    mech_avg = {m: np.mean(vals) for m, vals in mech_means.items()}
    
    # Find best
    best_vis = max(vis_avg, key=vis_avg.get)
    best_mech = max(mech_avg, key=mech_avg.get)
    
    # Find best overall condition
    best_condition = max(results.keys(), key=lambda k: results[k]["cooperation_rate"]["mean"])
    
    # Calculate main effects
    grand_mean = np.mean([m["cooperation_rate"]["mean"] for m in results.values()])
    
    vis_effects = {v: avg - grand_mean for v, avg in vis_avg.items()}
    mech_effects = {m: avg - grand_mean for m, avg in mech_avg.items()}
    
    return {
        "visibility_means": vis_avg,
        "mechanism_means": mech_avg,
        "best_visibility": best_vis,
        "best_mechanism": best_mech,
        "best_condition": best_condition,
        "grand_mean": grand_mean,
        "visibility_effects": vis_effects,
        "mechanism_effects": mech_effects,
    }


def print_phase1_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print Phase 1 results."""
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS: Visibility × Mechanism")
    print("=" * 70)
    
    # Results matrix
    print("\n--- Cooperation Rate Matrix ---")
    print(f"{'':>15}", end="")
    for mech in MatchingMechanism:
        print(f"{mech.value:>12}", end="")
    print()
    print("-" * 65)
    
    for vis in CapabilityVisibility:
        print(f"{vis.value:>15}", end="")
        for mech in MatchingMechanism:
            condition = f"{vis.value}_{mech.value}_uniform"
            coop = results[condition]["cooperation_rate"]["mean"]
            print(f"{coop:>12.3f}", end="")
        print()
    
    # Main effects
    print("\n--- Main Effects ---")
    print("\nVisibility Effects:")
    for vis, effect in sorted(analysis["visibility_effects"].items(), key=lambda x: -x[1]):
        print(f"  {vis:>15}: {effect:+.4f}")
    
    print("\nMechanism Effects:")
    for mech, effect in sorted(analysis["mechanism_effects"].items(), key=lambda x: -x[1]):
        print(f"  {mech:>15}: {effect:+.4f}")
    
    # Best condition
    print(f"\n--- Best Condition ---")
    print(f"Best visibility: {analysis['best_visibility']}")
    print(f"Best mechanism: {analysis['best_mechanism']}")
    print(f"Best overall: {analysis['best_condition']}")
    best_coop = results[analysis['best_condition']]["cooperation_rate"]["mean"]
    print(f"Best cooperation: {best_coop:.3f}")
    
    # Hypothesis evaluation
    print("\n--- Hypothesis Evaluation ---")
    
    # H1: Public > Private
    pub_coop = analysis["visibility_means"]["public"]
    priv_coop = analysis["visibility_means"]["private"]
    h1 = pub_coop > priv_coop
    print(f"H1 (Public > Private): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    print(f"    Public: {pub_coop:.3f}, Private: {priv_coop:.3f}")
    
    # H2: Capability-matched > Self-selection > Random
    cap_coop = analysis["mechanism_means"]["cap_match"]
    self_coop = analysis["mechanism_means"]["self_select"]
    rand_coop = analysis["mechanism_means"]["random"]
    h2 = cap_coop > self_coop > rand_coop
    print(f"H2 (Cap-matched > Self-select > Random): {'SUPPORTED' if h2 else 'NOT SUPPORTED'}")
    print(f"    Cap-matched: {cap_coop:.3f}, Self-select: {self_coop:.3f}, Random: {rand_coop:.3f}")
    
    # H4: Reputation-proxied ≈ Public
    rep_coop = analysis["visibility_means"]["rep_proxied"]
    h4 = abs(rep_coop - pub_coop) < 0.05
    print(f"H4 (Rep-proxied ≈ Public): {'SUPPORTED' if h4 else 'NOT SUPPORTED'}")
    print(f"    Rep-proxied: {rep_coop:.3f}, Public: {pub_coop:.3f}, Diff: {abs(rep_coop - pub_coop):.3f}")
    
    # H5: Difficulty-weighted is robust
    diff_coop = analysis["mechanism_means"]["diff_weight"]
    h5 = diff_coop >= cap_coop * 0.95  # Within 5% of best
    print(f"H5 (Diff-weighted robust): {'SUPPORTED' if h5 else 'NOT SUPPORTED'}")
    print(f"    Diff-weighted: {diff_coop:.3f}, Cap-matched: {cap_coop:.3f}")


def print_phase2_results(results: Dict[str, Any]):
    """Print Phase 2 results."""
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS: Task Distribution")
    print("=" * 70)
    
    print("\n--- Results by Task Distribution ---")
    print(f"{'Distribution':>15} {'Cooperation':>12} {'Capability':>12} {'Utilization':>12}")
    print("-" * 55)
    
    for condition, metrics in results.items():
        dist = condition.split("_")[-1]
        coop = metrics["cooperation_rate"]["mean"]
        cap = metrics["mean_capability"]["mean"]
        util = metrics["capability_utilization"]["mean"]
        print(f"{dist:>15} {coop:>12.3f} {cap:>12.3f} {util:>12.3f}")
    
    # H3: Adaptive tasks unlock capability benefits
    adaptive_key = [k for k in results.keys() if "adaptive" in k][0]
    uniform_key = [k for k in results.keys() if "uniform" in k][0]
    
    adaptive_coop = results[adaptive_key]["cooperation_rate"]["mean"]
    uniform_coop = results[uniform_key]["cooperation_rate"]["mean"]
    adaptive_util = results[adaptive_key]["capability_utilization"]["mean"]
    uniform_util = results[uniform_key]["capability_utilization"]["mean"]
    
    print("\n--- Hypothesis H3 Evaluation ---")
    h3 = adaptive_util > uniform_util
    print(f"H3 (Adaptive unlocks capability): {'SUPPORTED' if h3 else 'NOT SUPPORTED'}")
    print(f"    Adaptive utilization: {adaptive_util:.3f}, Uniform: {uniform_util:.3f}")
    print(f"    Adaptive cooperation: {adaptive_coop:.3f}, Uniform: {uniform_coop:.3f}")


def main():
    """Run the full experiment."""
    
    # Phase 1: Visibility × Mechanism
    phase1_results = run_phase1_experiment(
        n_rounds=100,
        n_seeds=10,
        n_agents=30
    )
    
    # Analyze Phase 1
    phase1_analysis = analyze_phase1(phase1_results)
    print_phase1_results(phase1_results, phase1_analysis)
    
    # Determine best visibility and mechanism for Phase 2
    best_vis_str = phase1_analysis["best_visibility"]
    best_mech_str = phase1_analysis["best_mechanism"]
    
    # Convert strings back to enums
    best_vis = next(v for v in CapabilityVisibility if v.value == best_vis_str)
    best_mech = next(m for m in MatchingMechanism if m.value == best_mech_str)
    
    # Phase 2: Best mechanism × Task Distribution
    phase2_results = run_phase2_experiment(
        best_visibility=best_vis,
        best_mechanism=best_mech,
        n_rounds=100,
        n_seeds=10,
        n_agents=30
    )
    
    print_phase2_results(phase2_results)
    
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
        "phase1": {
            "conditions": phase1_results,
            "analysis": phase1_analysis,
        },
        "phase2": {
            "conditions": phase2_results,
            "best_visibility": best_vis_str,
            "best_mechanism": best_mech_str,
        },
    }
    
    with open("results/experiment_32_matching.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 32 COMPLETE")
    print("=" * 70)
    print(f"Results saved to results/experiment_32_matching.json")
    
    # Summary
    print("\n--- KEY FINDINGS ---")
    print(f"Best visibility: {best_vis_str}")
    print(f"Best mechanism: {best_mech_str}")
    
    best_cond = phase1_analysis["best_condition"]
    best_coop = phase1_results[best_cond]["cooperation_rate"]["mean"]
    print(f"Best Phase 1 cooperation: {best_coop:.3f} ({best_cond})")
    
    # Compare to baseline (random/private)
    baseline_cond = "private_random_uniform"
    if baseline_cond in phase1_results:
        baseline_coop = phase1_results[baseline_cond]["cooperation_rate"]["mean"]
        improvement = (best_coop - baseline_coop) / baseline_coop * 100
        print(f"Improvement over baseline: {improvement:+.1f}%")
    
    return output


if __name__ == "__main__":
    main()