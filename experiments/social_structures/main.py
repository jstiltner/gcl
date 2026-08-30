"""
Social Structures Experiment - Main Runner

Tests which internal organizational structure produces optimal coordination
when competing within a capitalist meta-network.

Phases:
1. Internal Baseline - Test each structure in isolation
2. Head-to-Head - Direct competition between structure pairs
3. Population Ecology - Multiple firms of each type compete
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
from itertools import combinations
from scipy import stats
import json

from experiments.social_structures.agents.agent import Agent, Task, create_population
from experiments.social_structures.structures.implementations import create_structure
from experiments.social_structures.competition.firm import Firm, create_firm
from experiments.social_structures.competition.arena import CapitalistArena
from experiments.social_structures.config.capitalism import CapitalistMetaNetwork, MarketConditions


STRUCTURE_TYPES = ["meritocracy", "guild", "obligation", "ubuntu", "rotating", "monastic"]


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


def collect_internal_metrics(firm: Firm) -> Dict[str, float]:
    """Collect internal metrics for a firm."""
    agents = firm.agents
    n = len(agents)
    
    # Cooperation rate (success rate)
    total_tasks = sum(len(a.success_history) + len(a.failure_history) for a in agents)
    successes = sum(len(a.success_history) for a in agents)
    cooperation_rate = successes / total_tasks if total_tasks > 0 else 0
    
    # Gini coefficient
    reps = [a.reputation for a in agents]
    gini = calculate_gini(reps)
    
    # Mobility (status changes)
    status_changes = sum(1 for a in agents if a.status_changed_recently)
    mobility = status_changes / n if n > 0 else 0
    
    # Hard task attempts
    hard_attempts = sum(1 for a in agents if a.attempted_hard_task_recently)
    hard_rate = hard_attempts / n if n > 0 else 0
    
    # Recovery rate
    failed_agents = [a for a in agents if len(a.failure_history) > 0]
    recovered = [a for a in failed_agents if a.recovered_recently]
    recovery_rate = len(recovered) / len(failed_agents) if failed_agents else 1.0
    
    # Knowledge diffusion
    total_templates = sum(len(a.template_library) for a in agents)
    unique_templates = len(set(t.id for a in agents for t in a.template_library))
    diffusion = total_templates / (unique_templates * n) if unique_templates > 0 and n > 0 else 0
    
    # Underclass
    struggling = [a for a in agents if a.status_level == "struggling"]
    underclass_size = len(struggling) / n if n > 0 else 0
    
    # Mean capability
    mean_capability = sum(a.effective_capability for a in agents) / n if n > 0 else 0
    
    return {
        "cooperation_rate": cooperation_rate,
        "gini": gini,
        "mobility": mobility,
        "hard_task_rate": hard_rate,
        "recovery_rate": recovery_rate,
        "knowledge_diffusion": diffusion,
        "underclass_size": underclass_size,
        "mean_capability": mean_capability,
        "total_templates": total_templates,
        "unique_templates": unique_templates,
    }


# ============================================================================
# Phase 1: Internal Baseline (No Competition)
# ============================================================================

def run_phase1_internal_baseline(n_rounds: int = 200, n_seeds: int = 10, n_agents: int = 50):
    """
    Test each structure in isolation.
    No competition - just internal coordination.
    """
    print("=" * 70)
    print("PHASE 1: INTERNAL BASELINE (No Competition)")
    print("=" * 70)
    print(f"Structures: {STRUCTURE_TYPES}")
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    for structure_type in STRUCTURE_TYPES:
        print(f"Testing {structure_type}...", end=" ")
        seed_results = []
        
        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)
            
            # Create isolated firm
            firm = create_firm(
                firm_id=f"{structure_type}_{seed}",
                structure_type=structure_type,
                n_agents=n_agents,
                starting_resources=10.0
            )
            
            # Run internal tasks (no competition)
            for round_num in range(n_rounds):
                firm.run_internal_round()
                
                # Generate and attempt task
                difficulty = random.uniform(0.3, 0.7)
                task = Task(id=f"task_{round_num}", difficulty=difficulty, reward=1.0)
                firm.attempt_task(task)
                
                firm.end_internal_round()
            
            # Collect metrics
            metrics = collect_internal_metrics(firm)
            seed_results.append(metrics)
        
        # Aggregate across seeds
        aggregated = {}
        for key in seed_results[0].keys():
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        results[structure_type] = aggregated
        print("done")
    
    return results


def print_phase1_results(results: Dict[str, Any]):
    """Print Phase 1 results."""
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS: Internal Baseline")
    print("=" * 70)
    
    # Key metrics table
    metrics = ["cooperation_rate", "gini", "recovery_rate", "underclass_size", "knowledge_diffusion"]
    
    print(f"\n{'Structure':<15}", end="")
    for m in metrics:
        print(f"{m[:12]:>14}", end="")
    print()
    print("-" * 85)
    
    for structure in STRUCTURE_TYPES:
        print(f"{structure:<15}", end="")
        for m in metrics:
            val = results[structure][m]["mean"]
            print(f"{val:>14.3f}", end="")
        print()
    
    # Find winners for each metric
    print("\n--- Winners by Metric ---")
    for metric in metrics:
        values = [(s, results[s][metric]["mean"]) for s in STRUCTURE_TYPES]
        if metric == "gini" or metric == "underclass_size":
            # Lower is better
            winner = min(values, key=lambda x: x[1])
            print(f"{metric}: {winner[0]} ({winner[1]:.3f}) [lowest]")
        else:
            # Higher is better
            winner = max(values, key=lambda x: x[1])
            print(f"{metric}: {winner[0]} ({winner[1]:.3f}) [highest]")


# ============================================================================
# Phase 2: Head-to-Head Competition
# ============================================================================

def run_phase2_head_to_head(n_rounds: int = 300, n_seeds: int = 10, n_agents: int = 30):
    """
    Direct competition between structure pairs.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: HEAD-TO-HEAD COMPETITION")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents per firm: {n_agents}")
    print()
    
    # All pairs
    pairs = list(combinations(STRUCTURE_TYPES, 2))
    results = {}
    
    for struct_a, struct_b in pairs:
        pair_key = f"{struct_a}_vs_{struct_b}"
        print(f"Testing {pair_key}...", end=" ")
        seed_results = []
        
        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)
            
            # Create two firms
            firm_a = create_firm(f"{struct_a}_{seed}", struct_a, n_agents)
            firm_b = create_firm(f"{struct_b}_{seed}", struct_b, n_agents)
            
            # Create arena
            arena = CapitalistArena(
                config=CapitalistMetaNetwork(),
                market=MarketConditions()
            )
            arena.add_firm(firm_a)
            arena.add_firm(firm_b)
            
            # Run competition
            sim_result = arena.run_simulation(n_rounds)
            
            # Determine winner
            if firm_a.resources > firm_b.resources:
                winner = struct_a
            elif firm_b.resources > firm_a.resources:
                winner = struct_b
            else:
                winner = "tie"
            
            seed_results.append({
                "winner": winner,
                "a_resources": firm_a.resources,
                "b_resources": firm_b.resources,
                "a_survived": not firm_a.is_bankrupt(),
                "b_survived": not firm_b.is_bankrupt(),
                "rounds": sim_result.rounds,
            })
        
        # Aggregate
        wins_a = sum(1 for r in seed_results if r["winner"] == struct_a)
        wins_b = sum(1 for r in seed_results if r["winner"] == struct_b)
        ties = sum(1 for r in seed_results if r["winner"] == "tie")
        
        results[pair_key] = {
            "structure_a": struct_a,
            "structure_b": struct_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": wins_a / n_seeds,
            "win_rate_b": wins_b / n_seeds,
            "avg_resources_a": np.mean([r["a_resources"] for r in seed_results]),
            "avg_resources_b": np.mean([r["b_resources"] for r in seed_results]),
            "survival_rate_a": np.mean([r["a_survived"] for r in seed_results]),
            "survival_rate_b": np.mean([r["b_survived"] for r in seed_results]),
        }
        print("done")
    
    return results


def print_phase2_results(results: Dict[str, Any]):
    """Print Phase 2 results."""
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS: Head-to-Head Competition")
    print("=" * 70)
    
    # Win matrix
    win_matrix = defaultdict(lambda: defaultdict(int))
    for pair_key, data in results.items():
        a, b = data["structure_a"], data["structure_b"]
        win_matrix[a][b] = data["wins_a"]
        win_matrix[b][a] = data["wins_b"]
    
    print(f"\n{'Win Matrix':<15}", end="")
    for s in STRUCTURE_TYPES:
        print(f"{s[:8]:>10}", end="")
    print(f"{'TOTAL':>10}")
    print("-" * 85)
    
    for s1 in STRUCTURE_TYPES:
        print(f"{s1:<15}", end="")
        total = 0
        for s2 in STRUCTURE_TYPES:
            if s1 == s2:
                print(f"{'---':>10}", end="")
            else:
                wins = win_matrix[s1][s2]
                total += wins
                print(f"{wins:>10}", end="")
        print(f"{total:>10}")
    
    # Overall ranking
    print("\n--- Overall Ranking (by total wins) ---")
    totals = [(s, sum(win_matrix[s].values())) for s in STRUCTURE_TYPES]
    totals.sort(key=lambda x: x[1], reverse=True)
    for rank, (structure, wins) in enumerate(totals, 1):
        print(f"{rank}. {structure}: {wins} wins")


# ============================================================================
# Phase 3: Population Ecology
# ============================================================================

def run_phase3_population_ecology(n_rounds: int = 500, n_seeds: int = 10, 
                                   firms_per_type: int = 5, n_agents: int = 20):
    """
    Multiple firms of each type compete in ecosystem.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: POPULATION ECOLOGY")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}")
    print(f"Firms per type: {firms_per_type}, Agents per firm: {n_agents}")
    print()
    
    results = []
    
    for seed in range(n_seeds):
        print(f"Running seed {seed + 1}/{n_seeds}...", end=" ")
        random.seed(seed)
        np.random.seed(seed)
        
        arena = CapitalistArena(
            config=CapitalistMetaNetwork(),
            market=MarketConditions()
        )
        
        # Add firms of each type
        for structure_type in STRUCTURE_TYPES:
            for i in range(firms_per_type):
                firm = create_firm(
                    f"{structure_type}_{i}",
                    structure_type,
                    n_agents
                )
                arena.add_firm(firm)
        
        # Run evolution
        sim_result = arena.run_simulation(n_rounds)
        
        # Count survivors by type
        survivor_counts = defaultdict(int)
        survivor_resources = defaultdict(float)
        for firm in sim_result.survivors:
            survivor_counts[firm.structure.name] += 1
            survivor_resources[firm.structure.name] += firm.resources
        
        # Track extinction order
        extinction_order = []
        for round_result in sim_result.history:
            for dead_firm in round_result.deaths:
                extinction_order.append({
                    "structure": dead_firm.structure.name,
                    "round": round_result.round_num
                })
        
        results.append({
            "survivor_counts": dict(survivor_counts),
            "survivor_resources": dict(survivor_resources),
            "total_rounds": sim_result.rounds,
            "extinction_order": extinction_order,
            "winner": sim_result.winner.structure.name if sim_result.winner else None,
        })
        print("done")
    
    return results


def print_phase3_results(results: List[Dict[str, Any]]):
    """Print Phase 3 results."""
    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS: Population Ecology")
    print("=" * 70)
    
    # Average survivors by type
    avg_survivors = defaultdict(list)
    avg_resources = defaultdict(list)
    wins = defaultdict(int)
    
    for r in results:
        for s in STRUCTURE_TYPES:
            avg_survivors[s].append(r["survivor_counts"].get(s, 0))
            avg_resources[s].append(r["survivor_resources"].get(s, 0))
        if r["winner"]:
            wins[r["winner"]] += 1
    
    print(f"\n{'Structure':<15} {'Avg Survivors':>15} {'Avg Resources':>15} {'Wins':>10}")
    print("-" * 60)
    
    for s in STRUCTURE_TYPES:
        avg_surv = np.mean(avg_survivors[s])
        avg_res = np.mean(avg_resources[s])
        w = wins[s]
        print(f"{s:<15} {avg_surv:>15.2f} {avg_res:>15.2f} {w:>10}")
    
    # Ranking by survival
    print("\n--- Ranking by Average Survivors ---")
    ranking = [(s, np.mean(avg_survivors[s])) for s in STRUCTURE_TYPES]
    ranking.sort(key=lambda x: x[1], reverse=True)
    for rank, (structure, avg) in enumerate(ranking, 1):
        print(f"{rank}. {structure}: {avg:.2f} avg survivors")


# ============================================================================
# Main
# ============================================================================

def run_all_phases():
    """Run all experimental phases."""
    print("=" * 70)
    print("SOCIAL STRUCTURES EXPERIMENT")
    print("=" * 70)
    print("\nResearch Question: Which internal organizational structure")
    print("produces optimal coordination when competing in capitalism?")
    print()
    
    # Phase 1 - reduced for speed
    phase1_results = run_phase1_internal_baseline(n_rounds=100, n_seeds=5, n_agents=30)
    print_phase1_results(phase1_results)
    
    # Phase 2 - reduced for speed
    phase2_results = run_phase2_head_to_head(n_rounds=100, n_seeds=5, n_agents=20)
    print_phase2_results(phase2_results)
    
    # Phase 3 - reduced for speed
    phase3_results = run_phase3_population_ecology(n_rounds=200, n_seeds=5,
                                                    firms_per_type=3, n_agents=15)
    print_phase3_results(phase3_results)
    
    # Save results
    all_results = {
        "phase1": phase1_results,
        "phase2": phase2_results,
        "phase3": phase3_results,
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/social_structures_experiment.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to results/social_structures_experiment.json")
    
    return all_results


if __name__ == "__main__":
    run_all_phases()
