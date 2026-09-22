#!/usr/bin/env python3
"""
Experiment 47: what do claims 1, 2 and 3 measure once the arms are made comparable?

All three rest on `40_corrected_oracle.py`, and all three inherit the same structural
problem: the two arms being compared are not two mechanisms. They are one selection
rule, run twice, wearing different labels.

  S1  SAME PICK.  At sigma_self = sigma_oracle = 0, `create_population` (:85-93) sets
      perceived == observed == true_capability. `self_selection` (:108-117) keeps
      agents with `perceived > difficulty*0.6` and takes the argmax of
      `perceived - difficulty*0.4`; difficulty is a single scalar per round, so the
      subtracted term is a constant across agents and the rule is argmax on perceived.
      `argmax_oracle` (:137-143) is argmax on observed. Same signal, same argmax, same
      agent -- verified 1000/1000 rounds in the audit.

  S2  THE ONLY DIFFERENCE IS A BOOLEAN.  `self_selection` returns `(agent, True)`;
      `argmax_oracle` returns `(agent, False)`. That flag feeds two things and nothing
      else: `ownership_bonus = 0.05 if volunteered` (:75) and the `commitment_level`
      update, which `update_psychology` (:60-68) applies only when volunteered. Both
      reach the outcome solely through `get_emergent_effort`. Under FIXED effort the
      flag is inert, which is why claim 2 reports a perfect null: with effort fixed the
      two arms are bit-identical arrays, and its published CI [-0.013, +0.013] is a
      bootstrap of one array against itself.

  S3  MISSING ARM.  Part B varies which SIGNAL each arm reads -- self-selection reads
      `perceived` (noise sigma_self), the oracle reads `observed` (noise sigma_oracle).
      No arm holds the signal fixed and varies only who decides. So the grid cannot
      distinguish "decentralised self-selection wins" from "argmax on the less noisy
      estimate wins", and those are very different claims.

This experiment adds the arms that make the comparison identifying, and separates three
channels the shipped design confounds: the SIGNAL read, the SELECTION RULE, and the
MOTIVATION LABEL.

Pre-registered predictions, scored in analyse():

  P1  (harness validity) The shipped Part A/C comparison reproduces: self-selection
      with emergent effort beats the argmax oracle by ~ +0.065 with d ~ 1.68.

  P2  Relabel only -- give the oracle's pick `volunteered=True`, changing nothing about
      which agent is chosen -- and the advantage is EXACTLY zero: `np.array_equal` on
      the per-seed arrays returns True. Claim 1's effect is the boolean, not the
      mechanism.

  P3  A central oracle reading the agents' OWN `perceived` signal, with the same label,
      is bit-identical to self-selection at every cell of the Part B grid. If so, the
      "observability phase boundary" is a signal-quality boundary and carries no
      information about centralisation vs self-selection -- claim 3 is true as stated
      only because "self-selection" is the name given to whichever arm reads the
      lower-noise signal.

  P4  Claim 1's effect size is a readout of two author-chosen constants. Sweeping
      `ownership_bonus` traces the advantage linearly, and with `ownership_bonus = 0`
      AND the commitment update ungated (so the label reaches nothing at all) the arms
      become bit-identical. Any target d can be produced by choosing a different 0.05.

  P5  Claim 2's HOLDING is nonetheless testable, and this experiment tests it properly.
      On the diagonal of the noise grid (sigma_self == sigma_oracle > 0) the two arms
      read equally noisy but INDEPENDENTLY drawn signals, so they genuinely differ and
      neither should win. Predicted: no significant advantage on any diagonal cell
      under fixed effort. That is the null claim 2 asserts -- just not the one its
      cited sigma = 0 evidence establishes.

What this does not do: it does not show `ownership_bonus` is the wrong modelling
assumption. "People who choose a task try harder at it" is a defensible premise. It
shows that claim 1 reports it as a finding when it is an input, and that its CI and d
describe noise around a number fixed before any agent acted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM_PATH = ROOT / "experiments" / "40_corrected_oracle.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The constant at 40_corrected_oracle.py:75. Duplicated deliberately: if the simulation
# is ever edited, this experiment should start disagreeing rather than track it.
SHIPPED_OWNERSHIP_BONUS = 0.05


def load_sim():
    spec = importlib.util.spec_from_file_location("corrected_oracle_sim", SIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["corrected_oracle_sim"] = mod
    spec.loader.exec_module(mod)
    return mod


def install_knobs(mod) -> dict[str, Any]:
    """
    Replace the two label-gated methods with versions carrying explicit knobs, so the
    motivation channel can be switched off without touching the selection rule.

    `ownership_bonus` defaults to the shipped 0.05 and `commitment_gated` to True, so
    with defaults every method is arithmetically identical to the shipped code.
    """
    knobs = {"ownership_bonus": SHIPPED_OWNERSHIP_BONUS, "commitment_gated": True}

    def update_psychology(self, success: bool, volunteered: bool):
        # Shipped (:60-68) moves commitment_level only when volunteered. Ungating it
        # removes the second channel through which the boolean reaches the outcome.
        apply_commitment = volunteered or not knobs["commitment_gated"]
        if success:
            self.recent_successes += 1
            if apply_commitment:
                self.commitment_level = min(1.0, self.commitment_level + 0.1)
        else:
            self.recent_failures += 1
            if apply_commitment:
                self.commitment_level = max(0.0, self.commitment_level - 0.05)

    def get_emergent_effort(self, volunteered: bool) -> float:
        base_effort = 0.8
        commitment_bonus = (self.commitment_level - 0.5) * 0.1
        denom = max(1, self.recent_successes + self.recent_failures)
        confidence = self.recent_successes / denom
        confidence_bonus = (confidence - 0.5) * 0.1
        ownership_bonus = knobs["ownership_bonus"] if volunteered else 0.0
        return max(0.5, min(1.0,
                            base_effort + commitment_bonus + confidence_bonus
                            + ownership_bonus))

    mod.Agent.update_psychology = update_psychology
    mod.Agent.get_emergent_effort = get_emergent_effort
    return knobs


def install_selectors(mod) -> None:
    """
    Add the arms the shipped design is missing. Each is a (signal, rule, label) triple;
    the shipped pair varies all three at once, so none of them is identified.

    None of these consume the RNG, matching `argmax_oracle`. `self_selection` consumes
    one `random.choice` only on its no-volunteer fallback (:117); bit-identity between
    arms is asserted empirically rather than assumed, so if that fallback ever fires the
    equality tests will simply fail rather than quietly mislead.
    """

    def by(signal: str, label: bool):
        def selector(agents, difficulty):
            return max(agents, key=lambda a: getattr(a, signal)), label
        return selector

    mod.SELECTORS.update({
        # Same pick as the shipped oracle, relabelled as a volunteer. Isolates S2.
        "argmax_oracle_labelled": by("oracle_observed", True),
        # Same pick as self-selection, relabelled as an assignment.
        "self_signal_unlabelled": by("perceived_capability", False),
        # A CENTRAL oracle reading the agents' own signal, labelled as they are.
        # Differs from self_selection in nothing but the name. Isolates S3.
        "perceived_oracle_labelled": by("perceived_capability", True),
    })


def run(mod, method: str, emergent: bool, n_seeds: int, n_rounds: int, n_agents: int,
        sigma_self: float = 0.0, sigma_oracle: float = 0.0) -> np.ndarray:
    return mod.run_seeds(
        n_seeds, selection_method=method, n_rounds=n_rounds, n_agents=n_agents,
        use_emergent_effort=emergent, sigma_self=sigma_self, sigma_oracle=sigma_oracle,
    )


def compare(mod, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff, lo, hi = mod.bootstrap_diff_ci(a, b)
    return {
        "diff": diff,
        "ci95": [lo, hi],
        "cohens_d": mod.cohens_d(a, b),
        "significant": bool(lo > 0 or hi < 0),
        # The decisive test. "Close to zero" and "exactly zero" mean different things:
        # exactly zero at every seed means one computation was run twice.
        "arrays_bit_identical": bool(np.array_equal(a, b)),
        "max_abs_per_seed_difference": float(np.max(np.abs(a - b))),
    }


def part_label(mod, knobs, args) -> dict[str, Any]:
    """P1, P2: is claim 1's effect the mechanism or the boolean?"""
    knobs["ownership_bonus"] = SHIPPED_OWNERSHIP_BONUS
    knobs["commitment_gated"] = True
    kw = dict(n_seeds=args.seeds, n_rounds=args.rounds, n_agents=args.agents)

    ss_em = run(mod, "self_selection", True, **kw)
    orc_em = run(mod, "argmax_oracle", True, **kw)
    orc_em_lab = run(mod, "argmax_oracle_labelled", True, **kw)
    ss_em_unlab = run(mod, "self_signal_unlabelled", True, **kw)
    ss_fx = run(mod, "self_selection", False, **kw)
    orc_fx = run(mod, "argmax_oracle", False, **kw)

    return {
        "shipped_claim1_ss_vs_oracle_emergent": compare(mod, ss_em, orc_em),
        "shipped_claim2_ss_vs_oracle_fixed": compare(mod, ss_fx, orc_fx),
        "relabelled_ss_vs_oracle_labelled_volunteer": compare(mod, ss_em, orc_em_lab),
        "relabelled_ss_unlabelled_vs_oracle": compare(mod, ss_em_unlab, orc_em),
        "mean_cooperation": {
            "self_selection_emergent": float(ss_em.mean()),
            "argmax_oracle_emergent": float(orc_em.mean()),
            "argmax_oracle_emergent_relabelled_volunteer": float(orc_em_lab.mean()),
            "self_selection_fixed": float(ss_fx.mean()),
            "argmax_oracle_fixed": float(orc_fx.mean()),
        },
    }


def part_bonus_sweep(mod, knobs, args) -> dict[str, Any]:
    """P4: trace claim 1's effect as a function of the constant that produces it."""
    kw = dict(n_seeds=args.seeds, n_rounds=args.rounds, n_agents=args.agents)
    rows = []
    for gated in (True, False):
        for bonus in args.bonuses:
            knobs["ownership_bonus"] = bonus
            knobs["commitment_gated"] = gated
            ss = run(mod, "self_selection", True, **kw)
            orc = run(mod, "argmax_oracle", True, **kw)
            c = compare(mod, ss, orc)
            rows.append({
                "ownership_bonus": bonus,
                "commitment_update_gated_on_volunteered": gated,
                "is_shipped_configuration": bool(
                    gated and bonus == SHIPPED_OWNERSHIP_BONUS
                ),
                **c,
            })
    knobs["ownership_bonus"] = SHIPPED_OWNERSHIP_BONUS
    knobs["commitment_gated"] = True
    return {"sweep": rows}


def part_signal_grid(mod, knobs, args) -> dict[str, Any]:
    """
    P3, P5: re-run Part B with the missing control, and test claim 2's holding where
    it can actually fail.
    """
    knobs["ownership_bonus"] = SHIPPED_OWNERSHIP_BONUS
    knobs["commitment_gated"] = True
    kw = dict(n_seeds=args.seeds, n_rounds=args.rounds, n_agents=args.agents)
    levels = args.noise
    grid = []
    for s_self in levels:
        for s_orac in levels:
            n = dict(sigma_self=s_self, sigma_oracle=s_orac)
            ss = run(mod, "self_selection", False, **kw, **n)
            orc = run(mod, "argmax_oracle", False, **kw, **n)
            # The control the shipped grid omits: central assignment, agents' signal.
            central = run(mod, "perceived_oracle_labelled", False, **kw, **n)
            grid.append({
                "sigma_self": s_self,
                "sigma_oracle": s_orac,
                "on_diagonal": bool(s_self == s_orac),
                "shipped_ss_vs_oracle": compare(mod, ss, orc),
                "control_ss_vs_central_oracle_same_signal": compare(mod, ss, central),
            })
    return {"noise_levels": levels, "grid": grid}


def analyse(label: dict, sweep: dict, grid: dict) -> dict[str, Any]:
    c1 = label["shipped_claim1_ss_vs_oracle_emergent"]
    p1 = bool(c1["diff"] > 0.05 and c1["cohens_d"] > 1.0)

    p2 = bool(
        label["relabelled_ss_vs_oracle_labelled_volunteer"]["arrays_bit_identical"]
    )

    controls = [
        cell["control_ss_vs_central_oracle_same_signal"]["arrays_bit_identical"]
        for cell in grid["grid"]
    ]
    p3 = bool(controls) and all(controls)

    neutral = [
        r for r in sweep["sweep"]
        if r["ownership_bonus"] == 0.0
        and not r["commitment_update_gated_on_volunteered"]
    ]
    monotone = sorted(
        (r for r in sweep["sweep"] if r["commitment_update_gated_on_volunteered"]),
        key=lambda r: r["ownership_bonus"],
    )
    diffs = [r["diff"] for r in monotone]
    p4 = bool(
        neutral
        and all(r["arrays_bit_identical"] for r in neutral)
        and all(x <= y + 1e-12 for x, y in zip(diffs, diffs[1:]))
    )

    diag = [c for c in grid["grid"] if c["on_diagonal"] and c["sigma_self"] > 0]
    p5 = bool(diag) and not any(
        c["shipped_ss_vs_oracle"]["significant"] for c in diag
    )

    off = [c for c in grid["grid"] if not c["on_diagonal"]]
    boundary_holds = all(
        (c["shipped_ss_vs_oracle"]["diff"] > 0)
        == (c["sigma_oracle"] > c["sigma_self"])
        for c in off
        if c["shipped_ss_vs_oracle"]["significant"]
    )

    return {
        "preregistered_predictions": {
            "P1_shipped_claim1_reproduces": p1,
            "P2_claim1_effect_is_the_volunteered_boolean": p2,
            "P3_phase_boundary_is_signal_quality_not_centralization": p3,
            "P4_effect_size_is_a_readout_of_the_ownership_bonus_constant": p4,
            "P5_claim2_holding_survives_at_matched_nonzero_noise": p5,
        },
        "phase_boundary_direction_holds_where_significant": bool(boundary_holds),
        "shipped_ownership_bonus": SHIPPED_OWNERSHIP_BONUS,
        "verdict": _verdict(p1, p2, p3, p4, p5),
    }


def _verdict(p1: bool, p2: bool, p3: bool, p4: bool, p5: bool) -> str:
    if not p1:
        return (
            "UNEXPECTED. The shipped claim-1 comparison did not reproduce, so the "
            "decomposition below is not anchored to the published result and nothing "
            "here should be used to amend the record until that is explained."
        )
    parts = []
    if p2 and p4:
        parts.append(
            "CLAIM 1 RETRACT. The +0.065 / d = 1.68 advantage is the `volunteered` "
            "boolean. Relabel the oracle's pick -- same agent, same round, same seed -- "
            "and the per-seed arrays become bit-identical, difference exactly 0.0000. "
            "The magnitude is set by the hardcoded `ownership_bonus = 0.05` at "
            "40_corrected_oracle.py:75 and traces it monotonically; zero the bonus and "
            "ungate the commitment update and the two arms are the same computation. "
            "The premise 'agents who choose a task try harder' is a legitimate "
            "modelling assumption, but it is an INPUT here, and claim 1 reports it as "
            "an output with a CI and an effect size attached."
        )
    elif p2:
        parts.append(
            "CLAIM 1 RETRACT. The advantage is the `volunteered` boolean -- relabelling "
            "alone makes the arms bit-identical -- though the constant sweep did not "
            "behave as predicted and should be inspected before it is cited."
        )
    if p5:
        parts.append(
            "CLAIM 2 HOLDING SURVIVES, EVIDENCE DOES NOT. At sigma = 0 the two arms are "
            "one computation run twice, so the published '+0.000, CI [-0.013, +0.013]' "
            "is a bootstrap of an array against itself and cannot be evidence of "
            "anything. Tested where it can fail -- equal but independently drawn noise "
            "on the diagonal of the grid -- the null does hold. Strike the number, keep "
            "the conclusion, and restate it against the diagonal cells."
        )
    else:
        parts.append(
            "CLAIM 2 RETRACT. Its cited sigma = 0 evidence is one computation run "
            "twice, and at matched non-zero noise -- where the arms genuinely differ -- "
            "the advantage is significant on at least one diagonal cell, so the holding "
            "does not survive either."
        )
    if p3:
        parts.append(
            "CLAIM 3 RESTATE. A central oracle reading the agents' own `perceived` "
            "signal is bit-identical to self-selection at every grid cell, so the "
            "boundary is about WHICH SIGNAL IS LESS NOISY and says nothing about "
            "self-selection or decentralisation. The finding is real but it is "
            "'argmax on the better estimate wins', which is not what the row claims."
        )
    else:
        parts.append(
            "CLAIM 3 UNRESOLVED. The signal-matched central oracle was not bit-identical "
            "to self-selection, so the boundary is not purely a signal-quality effect "
            "and the remaining difference needs locating before the row is amended."
        )
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--bonuses", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    p.add_argument("--noise", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.2, 0.3])
    args = p.parse_args()

    t0 = time.time()
    mod = load_sim()
    knobs = install_knobs(mod)
    install_selectors(mod)

    label = part_label(mod, knobs, args)
    sweep = part_bonus_sweep(mod, knobs, args)
    grid = part_signal_grid(mod, knobs, args)
    an = analyse(label, sweep, grid)

    artifact = {
        "experiment": "47_oracle_decomposition",
        "targets": (
            "claims 1, 2, 3; experiments/40_corrected_oracle.py:75 (ownership_bonus), "
            ":60-68 (label-gated commitment), :108-117 vs :137-143 (same argmax), "
            "Part B grid (no signal-matched control)"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 2),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pythonhashseed": __import__("os").environ.get("PYTHONHASHSEED", "unset"),
        },
        "config": {
            "n_seeds": args.seeds, "n_rounds": args.rounds, "n_agents": args.agents,
            "ownership_bonuses": args.bonuses, "noise_levels": args.noise,
        },
        "part_label_decomposition": label,
        "part_ownership_bonus_sweep": sweep,
        "part_signal_grid": grid,
        "analysis": an,
    }

    out = ROOT / "results" / "experiment_47_oracle_decomposition.json"
    out.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out}\n")

    print("  -- claim 1: is it the mechanism or the boolean? --")
    for k, v in label.items():
        if k == "mean_cooperation":
            continue
        lo, hi = v["ci95"]
        print(f"    {k:46s} {v['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"d={v['cohens_d']:+6.2f}  identical={v['arrays_bit_identical']}")

    print("\n  -- claim 1: effect vs the constant that produces it --")
    print(f"    {'bonus':>6} {'commit_gated':>13} {'diff':>9} {'d':>7}  identical")
    for r in sweep["sweep"]:
        star = "  <- shipped" if r["is_shipped_configuration"] else ""
        print(f"    {r['ownership_bonus']:>6.2f} {str(r['commitment_update_gated_on_volunteered']):>13} "
              f"{r['diff']:>+9.4f} {r['cohens_d']:>+7.2f}  {r['arrays_bit_identical']}{star}")

    print("\n  -- claim 3: shipped grid vs the signal-matched control --")
    print(f"    {'sig_self':>8} {'sig_orac':>8} {'shipped':>9} {'sig':>4} "
          f"{'ctrl_diff':>10}  ctrl_identical")
    for c in grid["grid"]:
        s = c["shipped_ss_vs_oracle"]
        ct = c["control_ss_vs_central_oracle_same_signal"]
        print(f"    {c['sigma_self']:>8.2f} {c['sigma_oracle']:>8.2f} "
              f"{s['diff']:>+9.4f} {'*' if s['significant'] else '':>4} "
              f"{ct['diff']:>+10.4f}  {ct['arrays_bit_identical']}")

    print()
    for k, v in an["preregistered_predictions"].items():
        print(f"    {k}: {'CONFIRMED' if v else 'NOT CONFIRMED'}")
    print(f"\n  {an['verdict']}")


if __name__ == "__main__":
    main()
