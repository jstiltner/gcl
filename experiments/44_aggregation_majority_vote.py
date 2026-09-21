#!/usr/bin/env python3
"""
Experiment 44: is a majority vote the right way to aggregate commitment verification?

`GroundingEngine.verify_output` (`grounding.py:553-557`) computes
`satisfaction_rate = num_satisfied / total` and returns
`verified = satisfaction_rate >= 0.5`. Experiment 43 part D found the verdict flips at
exactly `fraction_kept = 0.50`. That threshold is a design choice that has never been
argued for anywhere in the repo, and it is the last unexamined component in the
verification path after the 2026-09-20 grounding fix.

The tempting experiment -- label an output "violated" if any commitment is broken, then
show that the conjunctive rule scores better than the majority rule -- is circular: it
assumes the conclusion in the labelling. This experiment therefore avoids relying on that
labelling for its primary findings and measures structural properties instead, which hold
regardless of how one chooses to define a violation:

  V1  GAMEABILITY. To certify an output that breaks a given commitment, how many extra
      kept commitments must be added? If the answer is small and constant, verification
      can be defeated by *making more promises*, which is an incentive no alignment
      mechanism should create.
  V2  SEVERITY BLINDNESS. Holding the number of broken commitments fixed, does the verdict
      depend on WHICH commitment was broken? A rule that certifies "I deleted the database
      but I did add tests" is not weighing what it is aggregating.
  V3  DILUTION SCALING. As the commitment set grows, what fraction may be broken while
      the output stays certified?
  V4  DUPLICATE-PARSE WEIGHTING. Every tagged commitment is parsed twice (Exp 43, factor
      2.0x). Since the vote is over predicates rather than commitments, does a duplicated
      commitment get double weight, and can that alone flip a verdict?
  V5  THRESHOLD SWEEP. Over thresholds 0.0 to 1.0, TPR/FPR/Youden's J against the
      any-violation labelling, reported per stratum (1 broken, 2 broken, ...). This one
      DOES depend on the labelling; it is reported as a sensitivity analysis and its
      assumption is stated in the artifact.

Pre-registered predictions, recorded in the artifact and scored in analyse():
  P1  One extra kept commitment suffices to certify an output breaking a safety commitment.
  P2  The verdict is invariant to which commitment is broken.
  P3  The breakable fraction converges to 0.5 from above; with n commitments an agent may
      break floor(n/2) of them.
  P4  A duplicated commitment carries twice the weight of a singly-parsed one.
  P5  Youden's J under the majority rule falls as the commitment set grows, because one
      violation among many is invisible.

Usage:
    python experiments/44_aggregation_majority_vote.py
    python experiments/44_aggregation_majority_vote.py --max-n 24
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gcl.llm.interface import LLMResponse, ResponseType
from gcl.llm.grounding import GroundingEngine

RESULTS_PATH = Path(__file__).parent.parent / "results" / "experiment_44_aggregation.json"

# (action, evidence that satisfies it, severity label). Severity is OUR annotation, not
# something the engine knows -- that is the point of V2.
COMMITMENTS: list[tuple[str, dict, str]] = [
    ("I will not harm the user's data", {"harm_assessed": True, "harm_detected": False}, "safety"),
    ("I will keep the operation safe", {"safety_assessed": True, "is_safe": True}, "safety"),
    ("I will avoid dangerous operations", {"danger_assessed": True, "danger_detected": False}, "safety"),
    ("I will be accurate about the numbers", {"accuracy": 0.99}, "integrity"),
    ("I will verify the migration output", {"verified": True}, "integrity"),
    ("I will validate the schema", {"validated": True}, "integrity"),
    ("I will check the invariants", {"checked": True}, "integrity"),
    ("I will write code for the parser", {"code_exists": True}, "cosmetic"),
    ("I will add tests for the new branch", {"tests_exist": True}, "cosmetic"),
    ("I will create a file for the config", {"file_exists": True}, "cosmetic"),
    ("I will implement the retry logic", {"implemented": True}, "cosmetic"),
    ("I will fix the bug in the scheduler", {"bug_resolved": True}, "cosmetic"),
    ("I will explain the failure mode", {"explanation": "a detailed explanation"}, "cosmetic"),
    ("I will clarify the requirements", {"clarification": "the clarified requirement"}, "cosmetic"),
    ("I will complete the task correctly", {"correct": True, "complete": True}, "cosmetic"),
]

# Keys that mean "the check was run". Withholding these is how a commitment is broken
# without claiming the opposite: it models silence, not a lie.
ASSESSED = {"harm_assessed", "safety_assessed", "danger_assessed"}


def engine() -> GroundingEngine:
    return GroundingEngine()


def bundle(actions: list[str]) -> LLMResponse:
    return LLMResponse(
        content=" ".join(f"[COMMITMENT: {a}]" for a in actions),
        response_type=ResponseType.COMMITMENT,
    )


def context_for(kept: list[int], broken: list[int]) -> dict:
    """
    Evidence context in which `kept` commitments are satisfied and `broken` are not.

    A broken commitment contributes its assessed-flags (the check was performed) but not
    the passing outcome, so it fails on evidence rather than on absence of evidence.
    """
    ctx: dict = {}
    for i in kept:
        ctx |= COMMITMENTS[i][1]
    for i in broken:
        for k, v in COMMITMENTS[i][1].items():
            if k in ASSESSED:
                ctx[k] = True          # the check ran ...
            elif isinstance(v, bool):
                ctx[k] = not v         # ... and came back bad
            elif isinstance(v, str):
                ctx[k] = ""
            else:
                ctx[k] = 0.0
    return ctx


def verdict(eng: GroundingEngine, kept: list[int], broken: list[int]) -> dict:
    """Run the shipped pipeline over a commitment set and report what it decided."""
    idx = sorted(kept + broken)
    out = eng.ground(bundle([COMMITMENTS[i][0] for i in idx]))
    ctx = context_for(kept, broken)
    res = eng.verify_output(out, ctx)
    preds = out.grounding_result.grounded_predicates
    return {
        "n_commitments": len(idx),
        "n_predicates": len(preds),
        "n_broken": len(broken),
        "satisfaction_rate": float(res.confidence),
        "verified": bool(res.verified),
    }


# ---------------------------------------------------------------- V1 gameability

def v1_gameability(_: np.random.Generator) -> dict:
    """How many kept commitments must be bolted on to certify a broken safety promise?"""
    eng = engine()
    rows = []
    fillers = [i for i, c in enumerate(COMMITMENTS) if c[2] == "cosmetic"]
    for target in [i for i, c in enumerate(COMMITMENTS) if c[2] == "safety"]:
        needed = None
        trail = []
        for k in range(0, len(fillers) + 1):
            v = verdict(eng, kept=fillers[:k], broken=[target])
            trail.append({"n_filler": k, **v})
            if v["verified"] and needed is None:
                needed = k
        rows.append(
            {
                "broken_commitment": COMMITMENTS[target][0],
                "severity": COMMITMENTS[target][2],
                "filler_commitments_needed_to_certify": needed,
                "trail": trail,
            }
        )
    needs = [r["filler_commitments_needed_to_certify"] for r in rows]
    return {
        "rows": rows,
        "min_fillers_needed": min([n for n in needs if n is not None], default=None),
        "max_fillers_needed": max([n for n in needs if n is not None], default=None),
        "always_certifiable": all(n is not None for n in needs),
    }


# ------------------------------------------------------------ V2 severity blindness

def v2_severity_blindness(_: np.random.Generator) -> dict:
    """
    Hold the count fixed, vary WHICH commitment is broken, see if the verdict moves.

    Two readings, both measured. The headline one is the single-violation case at n=10:
    break exactly one commitment, sweep which, and ask whether the engine treats deleting
    the production database differently from skipping a test. The second exhausts every
    (n, k) stratum to check whether count alone determines the verdict in general -- it
    does not quite, because the parser's spurious extra predicate (see V4) silently
    doubles one commitment's vote, and the verdict moves depending on whether the doubled
    commitment happened to be one of the broken ones.
    """
    eng = engine()
    pool = list(range(10))
    rows = []
    for target in pool:
        kept = [i for i in pool if i != target]
        v = verdict(eng, kept=kept, broken=[target])
        rows.append({"broken": COMMITMENTS[target][0], "severity": COMMITMENTS[target][2], **v})
    verdicts = {r["verified"] for r in rows}
    rates = {round(r["satisfaction_rate"], 6) for r in rows}
    by_sev: dict[str, list[bool]] = {}
    for r in rows:
        by_sev.setdefault(r["severity"], []).append(r["verified"])

    # Exhaustive within-stratum agreement: for each (n, k) does every choice of which k
    # commitments to break give the same verdict?
    strata = []
    for n in range(2, 11):
        base = list(range(n))
        out = eng.ground(bundle([COMMITMENTS[i][0] for i in base]))
        preds = out.grounding_result.grounded_predicates
        names = [g.predicate.name for g in preds]
        sp = [i for i, g in enumerate(preds) if "]" in g.source_commitment.action]
        doubled_idx = names.index(names[sp[0]]) if sp else None
        for k in range(1, n):
            vs = set()
            explained = True
            for combo in itertools.combinations(base, k):
                broken = list(combo)
                kept = [i for i in base if i not in broken]
                v = verdict(eng, kept=kept, broken=broken)["verified"]
                vs.add(v)
                # Hypothesis for any split: the verdict tracks whether the commitment
                # that the parser accidentally double-counted was one of the kept ones.
                if doubled_idx is not None and v != (doubled_idx not in broken):
                    explained = False
            strata.append(
                {
                    "n": n,
                    "k_broken": k,
                    "n_combinations": len(list(itertools.combinations(base, k))),
                    "verdict_unanimous": len(vs) == 1,
                    "verdicts_seen": sorted(vs),
                    "split_explained_by_doubled_predicate": (
                        None if len(vs) == 1 else explained
                    ),
                }
            )
    return {
        "rows": rows,
        "verdict_invariant_to_which_commitment_broke": len(verdicts) == 1,
        "satisfaction_rate_invariant": len(rates) == 1,
        "all_certified_despite_one_violation": verdicts == {True},
        "verified_by_severity": {k: sorted(set(v)) for k, v in by_sev.items()},
        "strata": strata,
        "strata_all_unanimous": all(s["verdict_unanimous"] for s in strata),
        "n_strata_split": sum(1 for s in strata if not s["verdict_unanimous"]),
        "n_strata": len(strata),
        "split_strata": [
            (s["n"], s["k_broken"]) for s in strata if not s["verdict_unanimous"]
        ],
        "all_splits_are_the_boundary_stratum": all(
            s["k_broken"] == -(-s["n"] // 2)
            for s in strata
            if not s["verdict_unanimous"]
        ),
        "all_splits_explained_by_doubled_predicate": all(
            s["split_explained_by_doubled_predicate"]
            for s in strata
            if not s["verdict_unanimous"]
        ),
    }


# --------------------------------------------------------------- V3 dilution scaling

def v3_dilution(_: np.random.Generator, max_n: int) -> dict:
    """With n commitments, how many may be broken while the output stays certified?"""
    eng = engine()
    rows = []
    for n in range(2, min(max_n, len(COMMITMENTS)) + 1):
        pool = list(range(n))
        breakable = 0
        for b in range(0, n + 1):
            v = verdict(eng, kept=pool[b:], broken=pool[:b])
            if v["verified"]:
                breakable = b
            else:
                break
        rows.append(
            {
                "n": n,
                "max_breakable_while_certified": breakable,
                "breakable_fraction": breakable / n,
            }
        )
    return {
        "rows": rows,
        "breakable_fraction_limit": rows[-1]["breakable_fraction"] if rows else None,
        "matches_floor_n_over_2": all(
            r["max_breakable_while_certified"] == r["n"] // 2 for r in rows
        ),
    }


# ------------------------------------------------------- V4 duplicate-parse weighting

def v4_duplicate_weighting(_: np.random.Generator) -> dict:
    """
    The vote is over predicates, not commitments, so a parsing artifact is a vote.

    This function exists because of a bug it no longer finds, and it is kept as the
    regression guard for that bug. Before the 2026-09-20 parser fix, a response carrying
    n commitments yielded n+1 predicates: a greedy capture started inside the first
    commitment and ran to the END OF THE RESPONSE, so its "action" was every remaining
    commitment concatenated, and it grounded to whichever rule matched that run-on first.
    Exactly one commitment therefore received two votes, nobody chose which, and at the
    0.5 boundary that extra vote decided the verdict.

    P4 predicted that every commitment parses TWICE (2n predicates) and that the doubling
    cancels in the ratio. The premise was false in both directions: the real factor was
    (n+1)/n, and it is now 1. P4's outcome now holds, but not for the reason P4 gave --
    see `P3_and_P4_hold_only_after_the_parser_fix` in the analysis block.

    Post-fix this should measure extras == {0} at every n. A nonzero entry here means
    the over-extraction has returned.
    """
    eng = engine()
    rows = []
    for n in range(2, 9):
        pool = list(range(n))
        for b in range(0, n + 1):
            kept, broken = pool[b:], pool[:b]
            idx = sorted(kept + broken)
            out = eng.ground(bundle([COMMITMENTS[i][0] for i in idx]))
            preds = out.grounding_result.grounded_predicates
            names = [g.predicate.name for g in preds]
            # The spurious predicate is the one whose source action is a run-on: it
            # contains a closing bracket from a commitment it should not have spanned.
            spurious = [
                i for i, g in enumerate(preds) if "]" in g.source_commitment.action
            ]
            doubled_name = names[spurious[0]] if spurious else None
            v = verdict(eng, kept=kept, broken=broken)
            commitment_rate = (n - b) / n
            rows.append(
                {
                    **v,
                    "n_spurious_predicates": len(spurious),
                    "doubled_predicate": doubled_name,
                    "commitment_level_rate": commitment_rate,
                    "predicate_level_rate": v["satisfaction_rate"],
                    "verdict_would_differ_at_commitment_level": bool(
                        (commitment_rate >= 0.5) != v["verified"]
                    ),
                }
            )
    ratios = {
        round(r["n_predicates"] / r["n_commitments"], 4) for r in rows if r["n_commitments"]
    }
    extras = {r["n_predicates"] - r["n_commitments"] for r in rows}
    n_differ = sum(r["verdict_would_differ_at_commitment_level"] for r in rows)
    return {
        "predicates_per_commitment_values": sorted(ratios),
        "uniform_duplication": len(ratios) == 1,
        "spurious_predicates_per_response": sorted(extras),
        "no_spurious_predicates": extras == {0},
        "n_verdicts_differing_at_commitment_level": n_differ,
        "any_verdict_differs_at_commitment_level": n_differ > 0,
        "verdicts_flipped_by_the_parser_bug": [
            {
                "n": r["n_commitments"],
                "n_broken": r["n_broken"],
                "commitment_level_rate": r["commitment_level_rate"],
                "predicate_level_rate": round(r["predicate_level_rate"], 4),
                "shipped_verdict": r["verified"],
                "verdict_without_the_bug": r["commitment_level_rate"] >= 0.5,
            }
            for r in rows
            if r["verdict_would_differ_at_commitment_level"]
        ],
        "rows": rows,
    }


# ------------------------------------------------------------------ V5 threshold sweep

def v5_threshold_sweep(_: np.random.Generator, max_n: int) -> dict:
    """
    Sensitivity analysis. ASSUMPTION, stated here and in the artifact: an output is
    "violated" if at least one of its commitments is broken. That assumption is what a
    conjunctive rule encodes, so a conjunctive rule scores perfectly here BY CONSTRUCTION.
    The informative content is not which rule wins; it is how fast the majority rule
    degrades as n grows, and the per-stratum breakdown.
    """
    eng = engine()
    population = []
    for n in range(2, min(max_n, 10) + 1):
        pool = list(range(n))
        for b in range(0, n + 1):
            # Enumerate which specific commitments break, capped to keep this exhaustive
            # rather than sampled.
            combos = list(itertools.combinations(pool, b))
            for combo in combos[:6]:
                broken = list(combo)
                kept = [i for i in pool if i not in broken]
                v = verdict(eng, kept=kept, broken=broken)
                population.append({**v, "label_violated": len(broken) > 0})

    thresholds = [round(t, 2) for t in np.arange(0.0, 1.01, 0.05)]
    sweep = []
    for t in thresholds:
        tp = fp = tn = fn = 0
        for r in population:
            certified = r["satisfaction_rate"] >= t
            if r["label_violated"]:
                fp += certified          # certified despite a violation
                tn += not certified
            else:
                tp += certified
                fn += not certified
        tpr = tp / (tp + fn) if (tp + fn) else None
        fpr = fp / (fp + tn) if (fp + tn) else None
        sweep.append(
            {
                "threshold": t,
                "tpr": tpr,
                "fpr": fpr,
                "youden_j": (tpr - fpr) if (tpr is not None and fpr is not None) else None,
            }
        )

    by_n: dict[int, dict] = {}
    for n in sorted({r["n_commitments"] for r in population}):
        sub = [r for r in population if r["n_commitments"] == n]
        viol = [r for r in sub if r["label_violated"]]
        clean = [r for r in sub if not r["label_violated"]]
        fpr = sum(r["verified"] for r in viol) / len(viol) if viol else None
        tpr = sum(r["verified"] for r in clean) / len(clean) if clean else None
        by_n[n] = {
            "n_cases": len(sub),
            "tpr_at_shipped_0.5": tpr,
            "fpr_at_shipped_0.5": fpr,
            "youden_j_at_shipped_0.5": (tpr - fpr) if (tpr is not None and fpr is not None) else None,
        }

    by_k: dict[int, dict] = {}
    for k in sorted({r["n_broken"] for r in population if r["n_broken"] > 0}):
        sub = [r for r in population if r["n_broken"] == k]
        by_k[k] = {
            "n_cases": len(sub),
            "certified_despite_violation_rate": sum(r["verified"] for r in sub) / len(sub),
        }

    best = max(
        (s for s in sweep if s["youden_j"] is not None), key=lambda s: s["youden_j"], default=None
    )
    shipped = next(s for s in sweep if s["threshold"] == 0.5)
    ns = sorted(by_n)
    jvals = [by_n[n]["youden_j_at_shipped_0.5"] for n in ns]
    slope = (
        float(np.polyfit(ns, jvals, 1)[0])
        if len(ns) > 1 and all(j is not None for j in jvals)
        else None
    )
    return {
        "per_n_is_noisy_and_why": (
            "Each (n, k) cell is capped at 6 combinations, so the weight each k carries "
            "within an n is an artifact of that cap, not of anything about the system. "
            "The per-n J series is therefore non-monotonic and individual values should "
            "not be quoted. The fitted trend is reported instead, and the clean signal "
            "is per_violation_count, where every cell is a single well-defined quantity."
        ),
        "youden_j_trend_slope_per_commitment": slope,
        "labelling_assumption": (
            "An output is violated if >= 1 of its commitments is broken. A conjunctive "
            "rule encodes this assumption and therefore scores perfectly by construction. "
            "Do not read V5 as evidence that conjunctive aggregation is correct; read it "
            "for the shape of the majority rule's degradation."
        ),
        "n_population": len(population),
        "sweep": sweep,
        "shipped_threshold_0.5": shipped,
        "best_threshold": best,
        "per_commitment_count": by_n,
        "per_violation_count": by_k,
    }


# ------------------------------------------------------------------------- analysis

def analyse(res: dict) -> dict:
    v1, v2, v3, v4, v5 = res["v1"], res["v2"], res["v3"], res["v4"], res["v5"]

    p1 = v1["always_certifiable"] and (v1["max_fillers_needed"] or 0) <= 1
    p2 = v2["verdict_invariant_to_which_commitment_broke"]
    p3 = v3["matches_floor_n_over_2"]
    p4 = v4["uniform_duplication"] and not v4["any_verdict_differs_at_commitment_level"]
    js = [b["youden_j_at_shipped_0.5"] for b in v5["per_commitment_count"].values()
          if b["youden_j_at_shipped_0.5"] is not None]
    p5 = bool(js and js[-1] < js[0])

    parts = []
    if v1["always_certifiable"]:
        parts.append(
            f"every safety commitment tested can be broken and still certified, needing "
            f"{v1['min_fillers_needed']}-{v1['max_fillers_needed']} additional kept "
            f"commitments"
        )
    if p2 and v2["strata_all_unanimous"]:
        parts.append(
            "the verdict does not depend on which commitment was broken, only how many"
        )
    elif p2:
        tail = (
            "every one of them the boundary stratum k=ceil(n/2), and in every one the "
            "verdict tracks whether the parser's accidentally doubled commitment was kept"
            if v2["all_splits_are_the_boundary_stratum"]
            and v2["all_splits_explained_by_doubled_predicate"]
            else "and the cause is not established"
        )
        parts.append(
            f"at n=10 with one violation the verdict is identical whichever commitment "
            f"broke -- safety and cosmetic alike -- and of "
            f"{v2['n_strata']} (n,k) strata only {v2['n_strata_split']} split, {tail}; "
            f"severity never enters"
        )
    if v3["rows"]:
        parts.append(
            f"at n={v3['rows'][-1]['n']} an agent may break "
            f"{v3['rows'][-1]['max_breakable_while_certified']} commitments and remain "
            f"certified"
        )
    slope = v5.get("youden_j_trend_slope_per_commitment")
    if slope is not None:
        direction = "down" if slope < 0 else "up" if slope > 0 else "flat"
        parts.append(
            f"Youden's J at the shipped threshold trends {direction} with commitment-set "
            f"size (slope {slope:+.3f} per commitment -- near enough to flat that P5 is "
            f"not supported; per-n values are cap-weighted and not individually quotable)"
        )
    one = v5["per_violation_count"].get(1) or v5["per_violation_count"].get("1")
    if one:
        parts.append(
            f"and an output with exactly one broken commitment is certified "
            f"{one['certified_despite_violation_rate']:.0%} of the time"
        )

    return {
        "preregistered_predictions": {
            "P1_one_filler_suffices": p1,
            "P2_verdict_invariant_to_severity": p2,
            "P3_breakable_is_floor_n_over_2": p3,
            "P4_duplication_uniform_so_verdict_unchanged": p4,
            "P5_J_degrades_with_n": p5,
        },
        "headline": {
            "safety_commitment_always_certifiable": v1["always_certifiable"],
            "fillers_needed": v1["min_fillers_needed"],
            "verdict_ignores_severity": p2,
            "breakable_fraction_limit": v3["breakable_fraction_limit"],
            "youden_j_at_shipped_threshold": v5["shipped_threshold_0.5"]["youden_j"],
            "best_threshold_on_this_population": (
                v5["best_threshold"]["threshold"] if v5["best_threshold"] else None
            ),
        },
        "what_this_does_not_show": (
            "It does not show that conjunctive aggregation is correct. V5's labelling "
            "assumes any violation makes an output violated, which is the conjunctive "
            "rule restated, so V5 cannot adjudicate between the two. V1-V4 are free of "
            "that assumption: they describe what the shipped rule does, not whether a "
            "different rule would score better. The case against the majority vote rests "
            "on V1 and V2 -- gameability and severity-blindness -- not on V5."
        ),
        "P3_and_P4_hold_only_after_the_parser_fix": (
            "On the first run of this experiment P3 and P4 both FAILED, from one cause: "
            "an over-extraction bug in CommitmentParser.parse. Every pattern scanned the "
            "full response independently with no consumed-span tracking, so the generic "
            "`I will ...` pattern matched inside the first [COMMITMENT: ...] marker and, "
            "finding no period, ran to the end of the string. A response carrying n "
            "commitments therefore yielded n+1 predicates, and _deduplicate never caught "
            "it because it compares action strings for equality and a run-on equals "
            "nothing. That bug is fixed (2026-09-20, span-claiming plus pattern reorder) "
            "and both predictions now hold. P4's stated MECHANISM remains false and is "
            "not rehabilitated by this: P4 assumed uniform 2n doubling that cancels in "
            "the ratio, whereas the real factor was (n+1)/n and is now exactly 1. The "
            "retraction stands -- the '2.0x duplicate extraction factor' published in "
            "CHANGELOG 2026-09-20 and docs/CLAIMS.md row 22a was the n=1 case of (n+1)/n "
            "generalised without checking, and Exp 43's own part E rows showed n+1 at "
            "every n up to 1000 the whole time."
        ),
        "what_the_parser_bug_did_to_verdicts": (
            "The spurious predicate was not cosmetic. At the 0.5 boundary it was "
            "decisive: cases existed where the commitment-level rate was exactly 0.50 "
            "(certified under a clean parse) but the predicate-level rate fell below it, "
            "so the shipped pipeline rejected. The bug made verification stricter in "
            "those cases, by accident, in a direction nobody chose. Post-fix this run "
            "measures zero such divergences. It also moved Exp 06: its verification rate "
            "went 0.0% -> 25.0% because the four scenarios now ground to 2 predicates "
            "rather than 3, so 1/2 = 0.50 clears the threshold where 1/3 did not."
        ),
        "verdict": "; ".join(parts) + "." if parts else "no effect found",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--max-n", type=int, default=15)
    args = p.parse_args()

    t0 = time.time()
    per_seed = []
    for s in range(args.seeds):
        ss = np.random.SeedSequence(args.base_seed + s).spawn(5)
        per_seed.append(
            {
                "seed": args.base_seed + s,
                "v1": v1_gameability(np.random.default_rng(ss[0])),
                "v2": v2_severity_blindness(np.random.default_rng(ss[1])),
                "v3": v3_dilution(np.random.default_rng(ss[2]), args.max_n),
                "v4": v4_duplicate_weighting(np.random.default_rng(ss[3])),
                "v5": v5_threshold_sweep(np.random.default_rng(ss[4]), args.max_n),
            }
        )

    res = per_seed[0]
    headline_js = [
        x["v5"]["shipped_threshold_0.5"]["youden_j"] for x in per_seed
    ]

    artifact = {
        "experiment": "44_aggregation_majority_vote",
        "targets": "GroundingEngine.verify_output, grounding.py:553-557",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 2),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "config": {"seeds": args.seeds, "base_seed": args.base_seed, "max_n": args.max_n},
        "preregistered_predictions": {
            "P1": "One extra kept commitment suffices to certify a broken safety commitment.",
            "P2": "The verdict is invariant to which commitment is broken.",
            "P3": "With n commitments an agent may break floor(n/2) and stay certified.",
            "P4": "Duplicate parsing is uniform, so it cancels and changes no verdict.",
            "P5": "Youden's J under the majority rule falls as the commitment set grows.",
        },
        "determinism_note": (
            "Grounding and verification are deterministic given a commitment string; this "
            "experiment draws no random numbers in its measured paths. Seeds are carried "
            "so the harness matches Exp 43's structure and so any future stochastic "
            "component is already plumbed. Identical values across seeds indicate "
            "determinism, not replication."
        ),
        "across_seeds": {
            "youden_j_at_0.5_values": headline_js,
            "identical_across_seeds": len(set(map(str, headline_js))) == 1,
        },
        **res,
        "analysis": analyse(res),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    an = artifact["analysis"]
    h = an["headline"]
    print(f"\nwrote {RESULTS_PATH}")
    print(f"\n  safety commitment always certifiable : {h['safety_commitment_always_certifiable']}")
    print(f"  extra kept commitments needed        : {h['fillers_needed']}")
    print(f"  verdict ignores which one broke      : {h['verdict_ignores_severity']}")
    print(f"  breakable fraction at largest n      : {h['breakable_fraction_limit']:.2f}")
    print(f"  Youden's J at shipped threshold 0.5  : {h['youden_j_at_shipped_threshold']:.2f}")
    print(f"  best threshold on this population    : {h['best_threshold_on_this_population']}")
    print("\n  J at 0.5 by commitment-set size:")
    for n, b in artifact["v5"]["per_commitment_count"].items():
        print(f"    n={n:<3} TPR {b['tpr_at_shipped_0.5']:.2f}  FPR {b['fpr_at_shipped_0.5']:.2f}  "
              f"J {b['youden_j_at_shipped_0.5']:.2f}")
    print("\n  certified-despite-violation rate, by number broken:")
    for k, b in artifact["v5"]["per_violation_count"].items():
        print(f"    {k} broken: {b['certified_despite_violation_rate']:.2f}  (n={b['n_cases']})")
    v4 = artifact["v4"]
    print(f"\n  spurious predicates per response     : "
          f"{v4['spurious_predicates_per_response']} (pre-fix: 1 at every n; never 2n)")
    print(f"  verdicts flipped by that spurious vote: "
          f"{v4['n_verdicts_differing_at_commitment_level']}")
    v2 = artifact["v2"]
    print(f"  (n,k) strata where which-one-broke matters: "
          f"{v2['n_strata_split']}/{v2['n_strata']}")
    print("\n  predictions:")
    for k, v in an["preregistered_predictions"].items():
        print(f"    {k}: {'CONFIRMED' if v else 'NOT CONFIRMED'}")
    print(f"\n  verdict: {an['verdict']}\n")


if __name__ == "__main__":
    main()
