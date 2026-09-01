#!/usr/bin/env python3
"""
Derive real headline statistics for the Punishment Paradox and Hart-Moore claims,
from the actual agent-based simulations — NOT from synthetic data.

Context: 22_statistical_significance.py's `test_punishment_paradox()` and
`test_hart_moore()` do not run the real simulations in 15/16_*.py or
21_incomplete_contract_theory.py. They generate synthetic samples from
hand-picked means (`base_coop = 0.7 - 0.4 * level`, `incomplete_holdup = 0.4 +
np.random.normal(...)`, etc.) and run real statistical tests on that synthetic
data. The published site numbers (r=-0.951, t=36.18/d=9.34, 36.8% hold-up
reduction, t=10.38/d=2.68) all trace to that synthetic generator, confirmed
against results/22_statistical_significance/results.json to 10+ decimal
places. See jasonstiltner2026 repo, feature/ci-reproduced-results branch, for
the investigation.

This script re-derives the same headline statistics — using the same
statistical tests (independent-samples t-test, Cohen's d, Pearson r) — but
from real output of the real simulations: 16_consequence_severity_sweep.py's
`run_severity_experiment` (Punishment Paradox) and
21_incomplete_contract_theory.py's `IncompleteContractEnvironment`
(Hart-Moore). It does not touch or depend on 22_statistical_significance.py.

Usage:
    python experiments/derive_real_headline_stats.py [--ci]

    --ci reduces n_seeds and simulation size for a fast (~1 min) CI-scale run.
    Full scale (default) matches the seed count originally claimed (n=30).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = Path("results/real_headline_stats")


def _load_module(filename: str):
    """Import a numbered experiment script (not a valid module name) by path."""
    path = EXPERIMENTS_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def derive_punishment_paradox(n_seeds: int, n_agents: int, n_rounds: int) -> dict:
    """Real cooperation-vs-consequence-severity data from 16_*.py's actual simulation."""
    mod16 = _load_module("16_consequence_severity_sweep.py")

    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    cooperation_by_level = {level: [] for level in levels}

    for level in levels:
        # Duration scales with penalty, consistent with 16_*.py's own sweep
        # (penalty=1.0 -> duration=20 in the original 8-point sweep).
        severity = mod16.SeverityConfig(violation_penalty=level, penalty_duration=round(level * 20))
        for seed in range(n_seeds):
            result = mod16.run_severity_experiment(
                severity, n_agents=n_agents, n_rounds=n_rounds, seed=seed
            )
            cooperation_by_level[level].append(result["avg_cooperation"])
        print(f"  severity={level:.2f}: cooperation = "
              f"{np.mean(cooperation_by_level[level]):.3f} ± {np.std(cooperation_by_level[level]):.3f} "
              f"(n={n_seeds})")

    all_levels, all_coop = [], []
    for level in levels:
        all_levels.extend([level] * n_seeds)
        all_coop.extend(cooperation_by_level[level])

    correlation, corr_p = stats.pearsonr(all_levels, all_coop)

    no_cons = np.array(cooperation_by_level[0.0])
    full_cons = np.array(cooperation_by_level[1.0])
    t_stat, t_p = stats.ttest_ind(no_cons, full_cons)
    d = cohens_d(no_cons, full_cons)

    return {
        "source": "16_consequence_severity_sweep.py:run_severity_experiment (real simulation)",
        "n_seeds": n_seeds,
        "cooperation_by_level": {
            str(level): {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "raw": [float(x) for x in v],
            }
            for level, v in cooperation_by_level.items()
        },
        "correlation": {"r": float(correlation), "p": float(corr_p)},
        "no_vs_full_consequences": {"t": float(t_stat), "p": float(t_p), "d": float(d)},
    }


def derive_hart_moore(n_seeds: int, n_agents: int, n_timesteps: int) -> dict:
    """Real investment/hold-up data from 21_*.py's actual IncompleteContractEnvironment."""
    mod21 = _load_module("21_incomplete_contract_theory.py")

    conditions = list(mod21.ContractCompleteness)
    investment_by_cond = {c.value: [] for c in conditions}
    holdups_by_cond = {c.value: [] for c in conditions}

    for cond in conditions:
        for seed in range(n_seeds):
            env = mod21.IncompleteContractEnvironment(
                n_agents=n_agents, completeness=cond, rng_seed=seed
            )
            for _ in range(n_timesteps):
                env.step()
            summary = env.get_summary()
            investment_by_cond[cond.value].append(summary["mean_investment"])
            holdups_by_cond[cond.value].append(summary["total_hold_ups"])
        print(f"  {cond.value}: investment = {np.mean(investment_by_cond[cond.value]):.3f}, "
              f"hold-ups = {np.mean(holdups_by_cond[cond.value]):.1f} (n={n_seeds})")

    def arr(d, key):
        return np.array(d[key])

    tests = {}

    t, p = stats.ttest_ind(arr(investment_by_cond, "complete"), arr(investment_by_cond, "incomplete_high"))
    tests["prediction_1_complete_investment_gt_incomplete"] = {
        "t": float(t), "p": float(p), "d": float(cohens_d(arr(investment_by_cond, "complete"), arr(investment_by_cond, "incomplete_high"))),
        "label": "Complete-contract investment vs incomplete-contract (high) investment",
    }

    t, p = stats.ttest_ind(arr(investment_by_cond, "gcl"), arr(investment_by_cond, "incomplete_high"))
    tests["prediction_2_gcl_investment_gt_incomplete"] = {
        "t": float(t), "p": float(p), "d": float(cohens_d(arr(investment_by_cond, "gcl"), arr(investment_by_cond, "incomplete_high"))),
        "label": "GCL investment vs incomplete-contract (high) investment",
    }

    t, p = stats.ttest_ind(arr(holdups_by_cond, "incomplete_high"), arr(holdups_by_cond, "complete"))
    tests["prediction_3_incomplete_holdups_gt_complete"] = {
        "t": float(t), "p": float(p), "d": float(cohens_d(arr(holdups_by_cond, "incomplete_high"), arr(holdups_by_cond, "complete"))),
        "label": "Incomplete-contract (high) hold-ups vs complete-contract hold-ups",
    }

    t, p = stats.ttest_ind(arr(holdups_by_cond, "incomplete_high"), arr(holdups_by_cond, "gcl"))
    d4 = cohens_d(arr(holdups_by_cond, "incomplete_high"), arr(holdups_by_cond, "gcl"))
    tests["prediction_4_gcl_reduces_holdups"] = {
        "t": float(t), "p": float(p), "d": float(d4),
        "label": "Incomplete-contract (high) hold-ups vs GCL hold-ups",
    }

    mean_incomplete_holdups = float(np.mean(holdups_by_cond["incomplete_high"]))
    mean_gcl_holdups = float(np.mean(holdups_by_cond["gcl"]))
    holdup_reduction_pct = (
        (mean_incomplete_holdups - mean_gcl_holdups) / mean_incomplete_holdups * 100
        if mean_incomplete_holdups else 0.0
    )

    return {
        "source": "21_incomplete_contract_theory.py:IncompleteContractEnvironment (real simulation)",
        "n_seeds": n_seeds,
        "investment_by_condition": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "raw": [float(x) for x in v]}
            for k, v in investment_by_cond.items()
        },
        "holdups_by_condition": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "raw": [float(x) for x in v]}
            for k, v in holdups_by_cond.items()
        },
        "tests": tests,
        "holdup_reduction_pct": holdup_reduction_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="reduced scale for CI")
    args = parser.parse_args()

    if args.ci:
        pp_seeds, pp_agents, pp_rounds = 5, 20, 40
        hm_seeds, hm_agents, hm_timesteps = 5, 20, 80
        scale = "ci-small"
    else:
        pp_seeds, pp_agents, pp_rounds = 30, 50, 100
        hm_seeds, hm_agents, hm_timesteps = 30, 20, 200
        scale = "full"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Deriving real headline stats (scale={scale})")
    print("=" * 70)

    print("\nPunishment Paradox (16_consequence_severity_sweep.py, real sim):")
    punishment = derive_punishment_paradox(pp_seeds, pp_agents, pp_rounds)

    print("\nHart-Moore (21_incomplete_contract_theory.py, real sim):")
    hart_moore = derive_hart_moore(hm_seeds, hm_agents, hm_timesteps)

    print("\n" + "=" * 70)
    print("SUMMARY (real simulation output — not 22_statistical_significance.py)")
    print("=" * 70)
    print(f"Punishment Paradox: r = {punishment['correlation']['r']:.3f}, "
          f"p = {punishment['correlation']['p']:.2e}")
    print(f"  No vs full consequences: t = {punishment['no_vs_full_consequences']['t']:.2f}, "
          f"d = {punishment['no_vs_full_consequences']['d']:.2f}")
    print(f"Hart-Moore hold-up reduction: {hart_moore['holdup_reduction_pct']:.1f}%")

    output = {"scale": scale, "punishment_paradox": punishment, "hart_moore": hart_moore}
    out_path = RESULTS_DIR / f"real_headline_stats_{scale}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
