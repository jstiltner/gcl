"""
Experiment 40: Corrected Oracle and the Observability Phase Boundary

MOTIVATION: Experiment 39's "oracle" was NOT optimal. It scored candidates by
closeness-of-fit (score = 1/(1+excess)), deliberately penalizing over-qualified
agents. But the success model

    success_prob = true_capability * effort * (1 - difficulty * 0.5)

is monotonically increasing in capability, so "overkill" is free and the
anti-overkill oracle is strictly suboptimal. The reported "+81% over optimal
external matching" therefore conflated (a) genuine information asymmetry with
(b) the oracle optimizing the wrong objective.

This experiment corrects that:

Part A (confound demonstration): Compare self-selection against BOTH the
  original matching oracle (Exp 39) and a true argmax oracle, under fixed and
  emergent effort, with perfect self-knowledge.

Part B (observability sweep): Vary agent self-knowledge noise (sigma_self) and
  oracle observation noise (sigma_oracle) on a grid. Fixed effort isolates the
  information channel. This maps WHERE decentralized self-selection beats
  centralized assignment as a function of relative information asymmetry.

Part C (effort decomposition, corrected): Re-estimate the emergent-effort
  contribution against the corrected argmax oracle.

Statistics: 50 seeds per condition, bootstrap 95% CIs on mean differences,
Cohen's d across seeds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Agents (harness reused from experiments/39_motivation_isolation.py)
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Agent with true, self-perceived, and oracle-observed capabilities."""
    id: str
    true_capability: float
    perceived_capability: float   # what the agent believes (noise sigma_self)
    oracle_observed: float        # what the coordinator observes (noise sigma_oracle)

    commitment_level: float = 0.5
    recent_successes: int = 0
    recent_failures: int = 0

    def update_psychology(self, success: bool, volunteered: bool):
        if success:
            self.recent_successes += 1
            if volunteered:
                self.commitment_level = min(1.0, self.commitment_level + 0.1)
        else:
            self.recent_failures += 1
            if volunteered:
                self.commitment_level = max(0.0, self.commitment_level - 0.05)

    def get_emergent_effort(self, volunteered: bool) -> float:
        base_effort = 0.8
        commitment_bonus = (self.commitment_level - 0.5) * 0.1
        confidence = self.recent_successes / max(1, self.recent_successes + self.recent_failures)
        confidence_bonus = (confidence - 0.5) * 0.1
        ownership_bonus = 0.05 if volunteered else 0.0
        return max(0.5, min(1.0, base_effort + commitment_bonus + confidence_bonus + ownership_bonus))


def create_population(n_agents: int, sigma_self: float, sigma_oracle: float) -> List[Agent]:
    """Create population with independent self-perception and oracle-observation noise."""
    agents = []
    for i in range(n_agents):
        true_cap = max(0.2, min(0.9, random.gauss(0.6, 0.15)))

        if sigma_self <= 0:
            perceived = true_cap
        else:
            perceived = max(0.2, min(0.9, true_cap + random.gauss(0, sigma_self)))

        if sigma_oracle <= 0:
            observed = true_cap
        else:
            observed = max(0.2, min(0.9, true_cap + random.gauss(0, sigma_oracle)))

        agents.append(Agent(
            id=f"agent_{i}",
            true_capability=true_cap,
            perceived_capability=perceived,
            oracle_observed=observed,
        ))
    return agents


# ---------------------------------------------------------------------------
# Selection mechanisms
# ---------------------------------------------------------------------------

def self_selection(agents: List[Agent], difficulty: float) -> Tuple[Agent, bool]:
    """Agents self-select on perceived capability (identical to Exp 39)."""
    volunteers = []
    for agent in agents:
        if agent.perceived_capability > difficulty * 0.6:
            score = agent.perceived_capability - difficulty * 0.4
            volunteers.append((agent, score))
    if volunteers:
        return max(volunteers, key=lambda x: x[1])[0], True
    return random.choice(agents), False


def matching_oracle(agents: List[Agent], difficulty: float) -> Tuple[Agent, bool]:
    """Exp 39's original anti-overkill oracle (kept to document the confound).

    Scores by closeness-of-fit, penalizing excess capability, even though the
    success model rewards capability monotonically. NOT optimal.
    """
    candidates = []
    for agent in agents:
        if agent.oracle_observed > difficulty * 0.5:
            excess = agent.oracle_observed - difficulty
            score = 1.0 / (1.0 + excess) if excess > 0 else excess
            candidates.append((agent, score))
    if candidates:
        return max(candidates, key=lambda x: x[1])[0], False
    return max(agents, key=lambda a: a.oracle_observed), False


def argmax_oracle(agents: List[Agent], difficulty: float) -> Tuple[Agent, bool]:
    """Corrected oracle: assign the agent with maximal observed capability.

    Under success_prob = capability * effort * (1 - difficulty/2), this is the
    success-probability-maximizing assignment given the oracle's information.
    """
    return max(agents, key=lambda a: a.oracle_observed), False


SELECTORS = {
    "self_selection": self_selection,
    "matching_oracle": matching_oracle,
    "argmax_oracle": argmax_oracle,
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_condition(
    selection_method: str,
    n_rounds: int = 200,
    n_agents: int = 30,
    seed: int = 0,
    use_emergent_effort: bool = False,
    sigma_self: float = 0.0,
    sigma_oracle: float = 0.0,
) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(n_agents, sigma_self, sigma_oracle)
    selector = SELECTORS[selection_method]

    successes = 0
    volunteer_count = 0
    effort_sum = 0.0

    for _ in range(n_rounds):
        difficulty = random.uniform(0.3, 0.7)
        selected, volunteered = selector(agents, difficulty)
        if volunteered:
            volunteer_count += 1

        effort = selected.get_emergent_effort(volunteered) if use_emergent_effort else 0.8
        effort_sum += effort

        success_prob = selected.true_capability * effort * (1 - difficulty * 0.5)
        success = random.random() < success_prob
        selected.update_psychology(success, volunteered)
        if success:
            successes += 1

    return {
        "cooperation_rate": successes / n_rounds,
        "volunteer_rate": volunteer_count / n_rounds,
        "mean_effort": effort_sum / n_rounds,
    }


def run_seeds(n_seeds: int, **kwargs) -> np.ndarray:
    """Return per-seed cooperation rates."""
    return np.array([
        run_condition(seed=s, **kwargs)["cooperation_rate"] for s in range(n_seeds)
    ])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                      alpha: float = 0.05, seed: int = 12345) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b). Returns (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return float(a.mean() - b.mean()), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# ---------------------------------------------------------------------------
# Experiment parts
# ---------------------------------------------------------------------------

def part_a_confound(n_seeds: int, n_rounds: int, n_agents: int) -> Dict[str, Any]:
    print("\n--- Part A: Confound demonstration (perfect information) ---")
    out: Dict[str, Any] = {}
    runs = {}
    for method in ["self_selection", "matching_oracle", "argmax_oracle"]:
        for effort_mode, emergent in [("fixed", False), ("emergent", True)]:
            key = f"{method}_{effort_mode}"
            runs[key] = run_seeds(
                n_seeds, selection_method=method, n_rounds=n_rounds,
                n_agents=n_agents, use_emergent_effort=emergent,
                sigma_self=0.0, sigma_oracle=0.0,
            )
            out[key] = {"mean": float(runs[key].mean()), "std": float(runs[key].std(ddof=1))}
            print(f"  {key:>28}: coop={runs[key].mean():.3f} (sd {runs[key].std(ddof=1):.3f})")

    comparisons = {}
    for label, (x, y) in {
        "ss_vs_matching_fixed": ("self_selection_fixed", "matching_oracle_fixed"),
        "ss_vs_argmax_fixed": ("self_selection_fixed", "argmax_oracle_fixed"),
        "ss_vs_argmax_emergent": ("self_selection_emergent", "argmax_oracle_emergent"),
        "argmax_vs_matching_fixed": ("argmax_oracle_fixed", "matching_oracle_fixed"),
    }.items():
        diff, lo, hi = bootstrap_diff_ci(runs[x], runs[y])
        comparisons[label] = {
            "diff": diff, "ci95": [lo, hi],
            "cohens_d": cohens_d(runs[x], runs[y]),
            "significant": bool(lo > 0 or hi < 0),
        }
        print(f"  {label:>28}: {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] d={comparisons[label]['cohens_d']:+.2f}")

    out["comparisons"] = comparisons
    return out


def part_b_observability_sweep(n_seeds: int, n_rounds: int, n_agents: int) -> Dict[str, Any]:
    print("\n--- Part B: Observability sweep (fixed effort, argmax oracle) ---")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    grid = []
    print(f"  {'sig_self':>8} {'sig_orac':>8} {'ss':>7} {'oracle':>7} {'adv':>8} {'sig':>4}")
    for sigma_self in noise_levels:
        for sigma_oracle in noise_levels:
            ss = run_seeds(
                n_seeds, selection_method="self_selection", n_rounds=n_rounds,
                n_agents=n_agents, use_emergent_effort=False,
                sigma_self=sigma_self, sigma_oracle=sigma_oracle,
            )
            orc = run_seeds(
                n_seeds, selection_method="argmax_oracle", n_rounds=n_rounds,
                n_agents=n_agents, use_emergent_effort=False,
                sigma_self=sigma_self, sigma_oracle=sigma_oracle,
            )
            diff, lo, hi = bootstrap_diff_ci(ss, orc)
            cell = {
                "sigma_self": sigma_self,
                "sigma_oracle": sigma_oracle,
                "self_selection": float(ss.mean()),
                "argmax_oracle": float(orc.mean()),
                "advantage": diff,
                "ci95": [lo, hi],
                "cohens_d": cohens_d(ss, orc),
                "significant": bool(lo > 0 or hi < 0),
            }
            grid.append(cell)
            mark = "*" if cell["significant"] else ""
            print(f"  {sigma_self:>8.2f} {sigma_oracle:>8.2f} {ss.mean():>7.3f} "
                  f"{orc.mean():>7.3f} {diff:>+8.4f} {mark:>4}")
    return {"noise_levels": noise_levels, "grid": grid}


def part_c_effort_decomposition(n_seeds: int, n_rounds: int, n_agents: int) -> Dict[str, Any]:
    print("\n--- Part C: Effort decomposition vs corrected oracle ---")
    ss_fixed = run_seeds(n_seeds, selection_method="self_selection", n_rounds=n_rounds,
                         n_agents=n_agents, use_emergent_effort=False,
                         sigma_self=0.0, sigma_oracle=0.0)
    ss_em = run_seeds(n_seeds, selection_method="self_selection", n_rounds=n_rounds,
                      n_agents=n_agents, use_emergent_effort=True,
                      sigma_self=0.0, sigma_oracle=0.0)
    orc_fixed = run_seeds(n_seeds, selection_method="argmax_oracle", n_rounds=n_rounds,
                          n_agents=n_agents, use_emergent_effort=False,
                          sigma_self=0.0, sigma_oracle=0.0)
    orc_em = run_seeds(n_seeds, selection_method="argmax_oracle", n_rounds=n_rounds,
                       n_agents=n_agents, use_emergent_effort=True,
                       sigma_self=0.0, sigma_oracle=0.0)

    info_diff, info_lo, info_hi = bootstrap_diff_ci(ss_fixed, orc_fixed)
    total_diff, total_lo, total_hi = bootstrap_diff_ci(ss_em, orc_em)
    out = {
        "information_component": {"diff": info_diff, "ci95": [info_lo, info_hi]},
        "total_advantage": {"diff": total_diff, "ci95": [total_lo, total_hi]},
        "emergent_effort_component": total_diff - info_diff,
        "oracle_emergent_vs_fixed": {
            "diff": float(orc_em.mean() - orc_fixed.mean()),
        },
    }
    print(f"  Information component (fixed effort): {info_diff:+.4f} [{info_lo:+.4f}, {info_hi:+.4f}]")
    print(f"  Total advantage (emergent effort):    {total_diff:+.4f} [{total_lo:+.4f}, {total_hi:+.4f}]")
    print(f"  Emergent effort component:            {total_diff - info_diff:+.4f}")
    return out


def main():
    n_seeds, n_rounds, n_agents = 50, 200, 30
    print("=" * 70)
    print("EXPERIMENT 40: CORRECTED ORACLE + OBSERVABILITY PHASE BOUNDARY")
    print("=" * 70)
    print(f"Seeds: {n_seeds}, Rounds: {n_rounds}, Agents: {n_agents}")

    results = {
        "config": {"n_seeds": n_seeds, "n_rounds": n_rounds, "n_agents": n_agents},
        "part_a_confound": part_a_confound(n_seeds, n_rounds, n_agents),
        "part_b_observability": part_b_observability_sweep(n_seeds, n_rounds, n_agents),
        "part_c_effort": part_c_effort_decomposition(n_seeds, n_rounds, n_agents),
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_40_corrected_oracle.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Headline summary
    a = results["part_a_confound"]["comparisons"]
    print("\n--- HEADLINE ---")
    print(f"Self-selection vs Exp-39 matching oracle (fixed): {a['ss_vs_matching_fixed']['diff']:+.4f}")
    print(f"Self-selection vs corrected argmax oracle (fixed): {a['ss_vs_argmax_fixed']['diff']:+.4f}")
    if a["ss_vs_argmax_fixed"]["significant"] and a["ss_vs_argmax_fixed"]["diff"] > 0:
        print("=> Self-selection advantage survives even a true argmax oracle.")
    else:
        print("=> Exp 39's 'information asymmetry' advantage does not survive a")
        print("   truly optimal oracle: the prior result was driven by the")
        print("   anti-overkill objective, not privileged self-knowledge.")


if __name__ == "__main__":
    main()
