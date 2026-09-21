#!/usr/bin/env python3
"""
Experiment 42: Theorem 5 (Analogical Transfer) against a real task.

Theorem 5 states:

    E[Success] >= confidence * similarity

Experiment 02 already reports this as validated. It is not a test. Its transfer
loop (``02_template_induction.py:355-364``) does this:

    noise = np.random.normal(0, 0.1)
    actual_success_prob = confidence * similarity + noise     # "actual" success
    predicted = confidence * similarity                       # the bound
    theorem_satisfied = actual_success_prob >= predicted * 0.9

No task is performed. The outcome is the bound plus Gaussian noise, so the check
reduces to ``noise >= -0.1 * confidence * similarity`` and its 0.79 "satisfaction
rate" is the probability that a normal draw clears a small negative number. The
theorem cannot fail there.

This experiment performs the task.

WHAT MAKES THIS A TEST
----------------------
Success is produced by an environment that has never heard of ``similarity``:

  - The environment holds a hidden optimal control parameter theta*(context).
  - A template carries a concrete theta, found by real search over observed
    outcomes in a source region.
  - Transfer applies that theta to a target context. Quality falls off with
    |theta - theta*(target)|, which depends on the environment's ground truth.
  - Success is decided by evaluating the template's own VerificationSchema
    predicate against the resulting state.

``similarity`` is computed by ``ContextRegionSpec.similarity`` -- a normalised
distance from learned interval bounds, averaged with a 0/1 categorical match.
``quality`` is computed from how far the transferred parameter sits from the
target's true optimum. These are different functions of the context, and nothing
constrains the second to dominate the first. The bound can therefore be violated,
and the experiment's job is to report how often it is and where.

Confidence is *measured*, not set: it is the induced template's success rate over
a held-out in-region validation sample.

TWO BOUNDS, NOT ONE
-------------------
The theorem as written is ``default_confidence * similarity``. The package does
not implement that. ``CommitmentTemplate.compute_confidence`` (template.py:431-433)
already multiplies by similarity when the state is outside the validated region,
and ``expected_success_rate`` (template.py:534-535) then multiplies by similarity
*again*, so the implemented bound is ``default_confidence * similarity**2``. That
is strictly weaker wherever similarity < 1. Both are reported separately below.

PRE-REGISTERED HYPOTHESES
-------------------------
Fixed before the first run; outcomes recorded whatever they are.

  H1  The theorem as stated holds: fewer than 5% of target context cells have an
      empirical success rate significantly below default_confidence * similarity.
  H2  Similarity is informative about transfer: Spearman rho between cell
      similarity and cell empirical success rate is positive and significant.
  H3  The stated and implemented bounds agree on which cells violate.
  H4  The bound is not vacuous: in at least 10% of cells the empirical success
      rate is close to the bound (within 0.10) rather than far above it. A bound
      that is never approached is true but carries no information.

"Significantly below" means the upper end of the cell's 95% bootstrap CI for the
empirical success rate is still under the bound. Raw point-violations are also
reported, unadjusted.

Usage:
    python experiments/42_theorem5_real_task.py
    python experiments/42_theorem5_real_task.py --n-seeds 20 --trials-per-cell 30
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gcl.core.commitment import (  # noqa: E402
    ActionSpec,
    VerificationResult,
    VerificationStatus,
)
from gcl.templates.induction import (  # noqa: E402
    InductionConfig,
    TemplateInducer,
    TemplateLibrary,
)
from gcl.templates.template import (  # noqa: E402
    ActionSchema,
    CommitmentTemplate,
    ContextRegionSpec,
    TriggerSchema,
    VerificationSchema,
)

RESULTS_PATH = Path(__file__).parent.parent / "results" / "experiment_42_theorem5_real_task.json"

# Source region the template is induced in. Everything outside is transfer.
SOURCE_DOMAIN = "alpha"
SOURCE_LOAD = (0.20, 0.60)
SOURCE_DIFFICULTY = (0.10, 0.40)

SUCCESS_THRESHOLD = 0.60
SUCCESS_EXPRESSION = f"quality >= {SUCCESS_THRESHOLD}"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TuningEnvironment:
    """
    A parameter-tuning task with a hidden optimum.

    The agent must supply a control parameter ``theta``. Quality is a Gaussian
    kernel on the distance between theta and the context's true optimum, plus
    observation noise. Success is ``quality >= SUCCESS_THRESHOLD``.

    Deliberately written without any reference to context regions, similarity,
    confidence, or templates. It is given a context and a theta and returns a
    quality. This separation is the whole point of the experiment: if the
    environment could see the bound, the test would be circular in the way
    Experiment 02 is.
    """

    # Per-domain offsets in the true optimum. Unknown to the agent.
    DOMAIN_OFFSET = {"alpha": 0.30, "beta": 0.55, "gamma": 0.15}

    NOISE_SD = 0.05

    def __init__(self, rng: np.random.Generator, tolerance: float) -> None:
        self.rng = rng
        self.tolerance = tolerance
        self.evaluations = 0

    def optimal_theta(self, context: dict[str, Any]) -> float:
        """The hidden ground truth. Never exposed to the agent."""
        offset = self.DOMAIN_OFFSET[context["domain"]]
        raw = offset + 0.60 * context["load"] - 0.35 * context["difficulty"] ** 2
        return float(np.clip(raw, 0.0, 1.0))

    def run(self, context: dict[str, Any], theta: float) -> dict[str, Any]:
        """
        Execute the task and return the resulting state.

        The returned dict is what the VerificationSchema predicate is evaluated
        against, so the outcome of the commitment is decided by the environment's
        own dynamics rather than being drawn from the bound.
        """
        self.evaluations += 1
        error = theta - self.optimal_theta(context)
        quality = float(np.exp(-(error**2) / (2 * self.tolerance**2)))
        quality += float(self.rng.normal(0.0, self.NOISE_SD))
        quality = float(np.clip(quality, 0.0, 1.0))
        return {**context, "theta": theta, "quality": quality}


def sample_source_context(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "domain": SOURCE_DOMAIN,
        "load": float(rng.uniform(*SOURCE_LOAD)),
        "difficulty": float(rng.uniform(*SOURCE_DIFFICULTY)),
    }


# ---------------------------------------------------------------------------
# Phase 1: find a theta by real search over observed outcomes
# ---------------------------------------------------------------------------


def calibrate_theta(
    env: TuningEnvironment,
    rng: np.random.Generator,
    n_candidates: int = 41,
    probes_per_candidate: int = 25,
) -> tuple[float, float, int]:
    """
    Search for the theta that performs best in the source region.

    The agent only observes qualities it actually produced; it never reads
    ``optimal_theta``. Returns the chosen theta, its observed success rate, and
    the number of candidates tied at that rate -- when the task is forgiving
    enough that every candidate succeeds, the tuning problem is degenerate and
    the chosen theta is an artifact of argmax tie-breaking rather than a result.
    """
    candidates = np.linspace(0.0, 1.0, n_candidates)
    contexts = [sample_source_context(rng) for _ in range(probes_per_candidate)]

    rates = []
    for theta in candidates:
        successes = sum(
            env.run(ctx, float(theta))["quality"] >= SUCCESS_THRESHOLD for ctx in contexts
        )
        rates.append(successes / len(contexts))

    rates_arr = np.array(rates)
    best_rate = float(rates_arr.max())
    n_tied = int((rates_arr == best_rate).sum())
    best_theta = float(candidates[int(rates_arr.argmax())])
    return best_theta, best_rate, n_tied


# ---------------------------------------------------------------------------
# Phase 2: induce a template through the package's own induction path
# ---------------------------------------------------------------------------


def build_seed_template(theta: float) -> CommitmentTemplate:
    """A hand-built template used only to generate commitments for induction."""
    return CommitmentTemplate(
        trigger_schema=TriggerSchema(name="tuning_available", expression="load >= 0.0"),
        action_schema=ActionSchema(action_type="tune", static_parameters={"theta": theta}),
        verification_schema=VerificationSchema(
            name="tuning_success", success_expression=SUCCESS_EXPRESSION
        ),
        context_region=ContextRegionSpec(bounds={}, allowed_values={}),
        default_confidence=0.5,
    )


def induce_template(
    env: TuningEnvironment,
    rng: np.random.Generator,
    theta: float,
    n_experiences: int = 120,
) -> tuple[CommitmentTemplate | None, dict[str, Any]]:
    """
    Run commitments in the source region and induce a template from the outcomes.

    Uses ``TemplateInducer``, so the learned context region and the template's
    confidence both come from the package rather than from this script.
    """
    seed_template = build_seed_template(theta)
    inducer = TemplateInducer(
        library=TemplateLibrary(),
        config=InductionConfig(min_commitments=20, min_success_rate=0.5),
    )

    for _ in range(n_experiences):
        context = sample_source_context(rng)
        commitment = seed_template.instantiate(context, issuer="tuner")
        applied = float(commitment.promised_behavior.parameters["theta"])
        post_state = env.run(context, applied)

        predicate = seed_template.verification_schema.create_success_predicate()
        succeeded = bool(predicate.evaluate(post_state))

        inducer.observe(
            commitment=commitment,
            result=VerificationResult(
                status=VerificationStatus.SUCCESS if succeeded else VerificationStatus.FAILURE,
                commitment_id=commitment.id,
            ),
            context=context,
        )

    induced = inducer.induce()
    if not induced:
        return None, {"induced": 0}

    template = induced[0]

    # TemplateCandidate.to_template() rebuilds the action schema from
    # ``action_type`` alone and discards promised_behavior.parameters, so an
    # induced template cannot reproduce the action it was induced from. The
    # tuned theta is re-attached here. Recorded as a finding, not hidden.
    template.action_schema = ActionSchema(
        action_type=template.action_schema.action_type,
        static_parameters={"theta": theta},
    )

    return template, {
        "induced": len(induced),
        "learned_bounds": {k: list(v) for k, v in template.context_region.bounds.items()},
        "learned_values": {
            k: sorted(str(x) for x in v) for k, v in template.context_region.allowed_values.items()
        },
        "induced_confidence": float(template.default_confidence),
    }


def measure_confidence(
    env: TuningEnvironment,
    rng: np.random.Generator,
    template: CommitmentTemplate,
    n_validation: int = 400,
) -> float:
    """Held-out in-region success rate. This is what the bound is built on."""
    predicate = template.verification_schema.create_success_predicate()
    theta = float(template.action_schema.static_parameters["theta"])
    successes = sum(
        bool(predicate.evaluate(env.run(sample_source_context(rng), theta)))
        for _ in range(n_validation)
    )
    return successes / n_validation


# ---------------------------------------------------------------------------
# Phase 3: transfer
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    domain: str
    load: float
    difficulty: float
    similarity: float
    in_region: bool
    stated_bound: float
    implemented_bound: float
    successes: int
    trials: int

    @property
    def empirical(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


def transfer_grid(
    env: TuningEnvironment,
    rng: np.random.Generator,
    template: CommitmentTemplate,
    confidence: float,
    trials_per_cell: int,
    grid_points: int,
) -> list[Cell]:
    """
    Apply the template across a grid of target contexts and verify each outcome.

    For every cell: instantiate the template into a commitment, pull the action
    parameter out of the instantiated ActionSpec, run the environment, and decide
    success with the template's verification predicate.
    """
    predicate = template.verification_schema.create_success_predicate()
    loads = np.linspace(0.0, 1.0, grid_points)
    difficulties = np.linspace(0.0, 1.0, grid_points)

    cells: list[Cell] = []
    for domain in TuningEnvironment.DOMAIN_OFFSET:
        for load in loads:
            for difficulty in difficulties:
                context = {
                    "domain": domain,
                    "load": float(load),
                    "difficulty": float(difficulty),
                }
                similarity = template.context_similarity(context)
                in_region = template.is_valid_in_context(context)

                commitment = template.instantiate(context, issuer="tuner")
                action: ActionSpec = commitment.promised_behavior
                theta = float(action.parameters["theta"])

                successes = sum(
                    bool(predicate.evaluate(env.run(context, theta)))
                    for _ in range(trials_per_cell)
                )

                cells.append(
                    Cell(
                        domain=domain,
                        load=float(load),
                        difficulty=float(difficulty),
                        similarity=float(similarity),
                        in_region=bool(in_region),
                        # Theorem as written.
                        stated_bound=float(confidence * similarity),
                        # What the package computes.
                        implemented_bound=float(template.expected_success_rate(context)),
                        successes=successes,
                        trials=trials_per_cell,
                    )
                )
    return cells


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def bootstrap_upper(successes: int, trials: int, rng: np.random.Generator, n_boot: int = 2000) -> float:
    """Upper end of a 95% bootstrap CI for a cell's success rate."""
    if trials == 0:
        return 0.0
    draws = rng.binomial(trials, successes / trials, size=n_boot) / trials
    return float(np.percentile(draws, 97.5))


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """
    Ranks with ties averaged.

    Plain argsort-of-argsort breaks ties by array position, which manufactures a
    correlation when a variable is near-constant -- and empirical success is
    exactly 1.0 in almost every cell once the task is forgiving enough.
    """
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho without a scipy dependency. Returns 0.0 if either side is constant."""
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def analyse(all_cells: list[list[Cell]], rng: np.random.Generator) -> dict[str, Any]:
    flat = [c for seed_cells in all_cells for c in seed_cells]

    stated_point = [c for c in flat if c.empirical < c.stated_bound]
    impl_point = [c for c in flat if c.empirical < c.implemented_bound]

    stated_sig = [
        c
        for c in stated_point
        if bootstrap_upper(c.successes, c.trials, rng) < c.stated_bound
    ]
    impl_sig = [
        c
        for c in impl_point
        if bootstrap_upper(c.successes, c.trials, rng) < c.implemented_bound
    ]

    sims = np.array([c.similarity for c in flat])
    emps = np.array([c.empirical for c in flat])
    rho = spearman(sims, emps)
    # Permutation test for rho.
    perm = np.array([spearman(sims, rng.permutation(emps)) for _ in range(500)])
    rho_p = float((np.abs(perm) >= abs(rho)).mean())

    # A lower bound is informative when the truth sits just above it. A cell
    # where the truth is *below* the bound is a violation, not a tight bound,
    # so tightness is measured only over cells the bound actually holds in.
    slack = emps - np.array([c.stated_bound for c in flat])
    tight = float(((slack >= 0.0) & (slack <= 0.10)).mean())
    ties = float((emps == emps[0]).mean())

    # Where the violations concentrate.
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in TuningEnvironment.DOMAIN_OFFSET:
        d_cells = [c for c in flat if c.domain == domain]
        d_viol = [c for c in stated_sig if c.domain == domain]
        by_domain[domain] = {
            "cells": len(d_cells),
            "mean_similarity": float(np.mean([c.similarity for c in d_cells])),
            "mean_empirical": float(np.mean([c.empirical for c in d_cells])),
            "significant_violations": len(d_viol),
            "violation_rate": len(d_viol) / len(d_cells) if d_cells else 0.0,
        }

    in_region = [c for c in flat if c.in_region]
    out_region = [c for c in flat if not c.in_region]

    worst = sorted(stated_sig, key=lambda c: c.empirical - c.stated_bound)[:10]

    return {
        "cells_total": len(flat),
        "overall_mean_empirical": float(np.mean(emps)),
        "overall_mean_similarity": float(np.mean(sims)),
        "h1_stated_bound": {
            "point_violations": len(stated_point),
            "point_violation_rate": len(stated_point) / len(flat),
            "significant_violations": len(stated_sig),
            "significant_violation_rate": len(stated_sig) / len(flat),
            "threshold": 0.05,
            "passed": (len(stated_sig) / len(flat)) < 0.05,
        },
        "h2_similarity_informative": {
            "spearman_rho": rho,
            "permutation_p": rho_p,
            "modal_success_share": ties,
            "degenerate": ties > 0.90,
            "passed": rho > 0 and rho_p < 0.05 and ties <= 0.90,
        },
        "h3_bounds_agree": {
            "stated_significant_violations": len(stated_sig),
            "implemented_significant_violations": len(impl_sig),
            "note": (
                "compute_confidence already scales by similarity outside the region "
                "(template.py:431-433) and expected_success_rate scales by it again "
                "(template.py:534-535), so the implemented bound is confidence * "
                "similarity**2"
            ),
            "passed": len(stated_sig) == len(impl_sig),
        },
        "h4_bound_informative": {
            "fraction_within_0.10_above_bound": tight,
            "mean_slack": float(np.mean(slack)),
            "median_slack": float(np.median(slack)),
            "threshold": 0.10,
            "passed": tight >= 0.10,
        },
        "by_domain": by_domain,
        "in_region": {
            "cells": len(in_region),
            "mean_empirical": float(np.mean([c.empirical for c in in_region])) if in_region else None,
            "mean_similarity": float(np.mean([c.similarity for c in in_region])) if in_region else None,
        },
        "out_of_region": {
            "cells": len(out_region),
            "mean_empirical": float(np.mean([c.empirical for c in out_region])) if out_region else None,
            "mean_similarity": float(np.mean([c.similarity for c in out_region])) if out_region else None,
        },
        "worst_violations": [
            {
                "domain": c.domain,
                "load": round(c.load, 3),
                "difficulty": round(c.difficulty, 3),
                "similarity": round(c.similarity, 4),
                "stated_bound": round(c.stated_bound, 4),
                "empirical": round(c.empirical, 4),
                "shortfall": round(c.empirical - c.stated_bound, 4),
            }
            for c in worst
        ],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_regime(
    tolerance: float,
    args: argparse.Namespace,
) -> tuple[list[list[Cell]], list[dict[str, Any]]]:
    """Run the full pipeline at one task-sensitivity setting."""
    all_cells: list[list[Cell]] = []
    per_seed: list[dict[str, Any]] = []

    for i in range(args.n_seeds):
        rng = np.random.default_rng(args.seed + i)
        env = TuningEnvironment(rng, tolerance=tolerance)

        theta, observed_rate, n_tied = calibrate_theta(env, rng)
        template, induction_info = induce_template(env, rng, theta)
        if template is None:
            continue

        confidence = measure_confidence(env, rng, template)
        template.default_confidence = confidence

        all_cells.append(
            transfer_grid(
                env, rng, template, confidence, args.trials_per_cell, args.grid_points
            )
        )
        per_seed.append(
            {
                "seed": args.seed + i,
                "tuned_theta": theta,
                "calibration_success_rate": observed_rate,
                "calibration_tied_candidates": n_tied,
                "validated_confidence": confidence,
                "env_evaluations": env.evaluations,
                **induction_info,
            }
        )
    return all_cells, per_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--trials-per-cell", type=int, default=30)
    parser.add_argument("--grid-points", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.40, 0.80, 1.60],
        help="Task sensitivity sweep. Larger = more forgiving task.",
    )
    args = parser.parse_args()

    print("Experiment 42: Theorem 5 against a real task")
    print(f"  seeds={args.n_seeds} trials/cell={args.trials_per_cell} grid={args.grid_points}")
    print(f"  sensitivity sweep: {args.tolerances}")

    regimes: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    for tolerance in args.tolerances:
        all_cells, per_seed = run_regime(tolerance, args)
        if not all_cells:
            skipped.append(
                {
                    "tolerance": tolerance,
                    "reason": (
                        "induction produced no template: in-region success rate fell "
                        "below InductionConfig.min_success_rate, so there is nothing "
                        "to transfer and Theorem 5 has no subject"
                    ),
                }
            )
            print(f"  tolerance={tolerance}: no seeds produced a template, skipped")
            continue
        analysis = analyse(all_cells, np.random.default_rng(args.seed + 9999))
        tied = [p["calibration_tied_candidates"] for p in per_seed]
        regimes[f"{tolerance:g}"] = {
            "tolerance": tolerance,
            "seeds_used": len(per_seed),
            "seeds_requested": args.n_seeds,
            # Fewer than 5 surviving seeds is not a measurement.
            "underpowered": len(per_seed) < 5,
            # When every candidate theta ties, the source-region tuning problem
            # is vacuous and the template's theta is an argmax tie-break.
            "calibration_degenerate": bool(np.mean(tied) > 0.5 * 41),
            "mean_tied_candidates": float(np.mean(tied)),
            "per_seed": per_seed,
            "analysis": analysis,
        }
        h1 = analysis["h1_stated_bound"]
        h4 = analysis["h4_bound_informative"]
        print(
            f"  tolerance={tolerance:<5g} "
            f"violations={h1['significant_violation_rate']:.1%}  "
            f"mean_success={analysis['overall_mean_empirical']:.3f}  "
            f"tight={h4['fraction_within_0.10_above_bound']:.1%}  "
            f"rho={analysis['h2_similarity_informative']['spearman_rho']:+.3f}"
        )

    if not regimes:
        raise SystemExit("no regime produced a template")

    # A regime is "safe" for the theorem if the bound is not significantly
    # violated. It is "informative" if the bound is also approached. The
    # question is whether any regime is both.
    usable = {k: v for k, v in regimes.items() if not v["underpowered"]}
    safe = [k for k, v in usable.items() if v["analysis"]["h1_stated_bound"]["passed"]]
    informative = [k for k, v in usable.items() if v["analysis"]["h4_bound_informative"]["passed"]]
    both = sorted(set(safe) & set(informative))
    # A regime where the tuning problem is degenerate cannot support a claim
    # about transfer either way.
    non_degenerate_safe = [k for k in safe if not regimes[k]["calibration_degenerate"]]

    output = {
        "config": {
            "n_seeds": args.n_seeds,
            "trials_per_cell": args.trials_per_cell,
            "grid_points": args.grid_points,
            "base_seed": args.seed,
            "tolerances": args.tolerances,
            "source_region": {
                "domain": SOURCE_DOMAIN,
                "load": list(SOURCE_LOAD),
                "difficulty": list(SOURCE_DIFFICULTY),
            },
            "success_threshold": SUCCESS_THRESHOLD,
            "environment": {
                "domain_offsets": TuningEnvironment.DOMAIN_OFFSET,
                "noise_sd": TuningEnvironment.NOISE_SD,
                "optimal_theta": "clip(offset[domain] + 0.60*load - 0.35*difficulty**2, 0, 1)",
                "quality": "exp(-(theta - optimal_theta)**2 / (2 * tolerance**2)) + N(0, noise_sd)",
            },
        },
        "preregistered_hypotheses": {
            "H1": "stated bound holds; significant violation rate < 5% of cells",
            "H2": "Spearman rho(similarity, empirical success) > 0 and significant",
            "H3": "stated and implemented bounds flag the same violating cells",
            "H4": "bound is approached (within 0.10) in >= 10% of cells, i.e. not vacuous",
        },
        "design_note": (
            "Task sensitivity is swept rather than chosen. A single fixed tolerance "
            "would let the author pick the verdict; the sweep reports the regime "
            "boundary instead. Tolerance is the width of the band of control "
            "parameters that count as succeeding, so small values mean an "
            "unforgiving task and large values mean one where almost any parameter "
            "works."
        ),
        "regimes": regimes,
        "skipped_regimes": skipped,
        "summary": {
            "underpowered_regimes": [k for k, v in regimes.items() if v["underpowered"]],
            "degenerate_calibration_regimes": [
                k for k, v in regimes.items() if v["calibration_degenerate"]
            ],
            "regimes_where_bound_holds": safe,
            "regimes_where_bound_holds_and_tuning_is_non_degenerate": non_degenerate_safe,
            "regimes_where_bound_is_informative": informative,
            "regimes_where_both": both,
            "verdict": (
                "no tested regime has the bound both true and informative"
                if not both
                else f"bound is both true and informative at tolerance(s) {both}"
            ),
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2))

    print("\n--- Summary ---")
    print(f"  bound holds in:      {safe or 'no regime'}")
    print(f"  bound informative in: {informative or 'no regime'}")
    print(f"  both:                {both or 'no regime'}")
    print(f"\n  written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
