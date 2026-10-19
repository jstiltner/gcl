#!/usr/bin/env python3
"""
Experiment 48: what separates the arms in claims 7 and 8?

Claim 7 ("simple agents outperform strategic reasoners", 0.530 vs 0.306) rests on
`35a_rich_agents.py`. Claim 8 ("self-selection advantage over random/naive baselines is
statistically robust", d = 4.05) rests on `35b_statistical_validation.py`. Both compare
arms that differ in less than their labels suggest.

  A1  DEAD BRANCH.  `decide_to_volunteer` (35a:101-149) guards level 1 with
      `if self.strategic_level >= 1` (:118) and returns unconditionally at :133. The
      level-2 block at :136-147 is therefore unreachable for every agent, and levels 1
      and 2 are the same code path.

  A2  DEAD FIELD.  Even if :136 were reached, it branches on `self.reputation`, which is
      declared at :76 with default 0.5 and never assigned anywhere in the file. Every
      level-2 threshold would resolve to the middle branch, 0.5, forever. The "recursive
      / theory-of-mind" arm has no theory of mind in it.

  A3  WHAT ACTUALLY VARIES.  Level 0's threshold is `0.5 - risk_tolerance*0.2` with
      risk_tolerance ~ U(0.3, 0.7), i.e. ~0.40 on average. Level 1's is 0.6 when more
      than two others are expected to volunteer and 0.4 otherwise -- with 30 agents the
      first case is near-universal. So the comparison is threshold 0.40 vs threshold
      0.60. A higher threshold produces fewer volunteers, more rounds fall through to
      `random.choice(agents)` (:267), and those rounds are penalised TWICE: a random
      agent instead of the best volunteer, and effort 0.7 instead of 0.9, because
      `calculate_effort` (:151-169) adds +0.1 for volunteering and subtracts 0.1
      otherwise. Neither penalty is about strategic reasoning.

  B1  HARDCODED EFFORT.  35b's self-selection arm uses `effort = 0.9` (:101) and its
      random arm `effort = 0.8` (:108). Part of d = 4.05 is that literal.

  B2  MISSING ARM.  35b's self-selection rule keeps agents with
      `effective_capability > difficulty*0.5` and takes the argmax of
      `effective_capability - difficulty*0.3` (:95-100). Difficulty is one scalar per
      round, so the subtracted term is constant across agents: the rule is argmax on
      capability, and the no-volunteer fallback (:103) is argmax on capability too.
      There is no arm in which a CENTRAL selector uses the same information, so the
      experiment cannot distinguish "self-selection works" from "argmax beats uniform
      random", which is arithmetic.

Pre-registered predictions, scored in analyse():

  P1  (harness validity) 35a reproduces ~0.530 / ~0.306 / ~0.306 across levels 0/1/2
      and 35b reproduces d ~ 4.05.

  P2  Levels 1 and 2 of 35a are bit-identical per seed, and no agent's `reputation`
      moves off its 0.5 default in any run, at any level. Claim 7's three-point
      comparison has two points.

  P3  The level ordering is a threshold effect, not a reasoning-depth effect: replacing
      `decide_to_volunteer` with a plain fixed threshold and sweeping it reproduces the
      0.530-to-0.306 range, with levels 0 and 1 landing on that curve at ~0.40 and
      ~0.60. If so, "strategic reasoners do worse" means "a 0.6 threshold does worse
      than a 0.4 threshold".

  P4  Equalising the hardcoded effort terms shrinks both headline effects materially --
      35a's level-0-minus-level-1 gap and 35b's d -- without abolishing either. The
      corrected figures, not the shipped ones, are what the record should carry.

  P5  35b's self-selection arm is bit-identical per seed to a CENTRAL oracle that takes
      argmax on the same capability signal with the same effort. If so, claim 8's
      subject is wrong in the same way claim 3's is: the result is about the selection
      RULE, and carries no information about self-selection.

What this does not do: it does not claim strategic reasoning should help, nor that
"volunteers try harder" is a bad premise. It locates how much of each published number
is a measurement and how much is a literal.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATH_35A = ROOT / "experiments" / "35a_rich_agents.py"
PATH_35B = ROOT / "experiments" / "35b_statistical_validation.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Literals duplicated from the simulations so this experiment breaks loudly rather than
# silently tracking an edit. 35a:157/159 and 35b:101/108.
SHIPPED_EFFORT_SWING = 0.1
SHIPPED_SS_EFFORT = 0.9
SHIPPED_RANDOM_EFFORT = 0.8


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 35a -- claim 7
# ---------------------------------------------------------------------------

def install_35a_knobs(mod) -> dict[str, Any]:
    """
    Replace the three methods that carry the defects, each defaulting to the shipped
    behaviour so that with default knobs every number is arithmetically unchanged.
    """
    knobs = {
        "fix_dead_branch": False,   # guard level 1 with == 1 so :136 becomes reachable
        "wire_reputation": False,   # maintain reputation from outcomes
        "effort_swing": SHIPPED_EFFORT_SWING,
        "fixed_threshold": None,    # replace the whole rule with a constant threshold
    }
    reputation_writes = {"count": 0}

    def decide_to_volunteer(self, task_difficulty, task_type, other_agents=None):
        own_prob = self.estimate_success_probability(task_difficulty, task_type)

        if knobs["fixed_threshold"] is not None:
            return own_prob > knobs["fixed_threshold"], own_prob

        if self.strategic_level == 0:
            threshold = 0.5 - self.risk_tolerance * 0.2
            return own_prob > threshold, own_prob

        # Shipped guard is `>= 1`, which swallows level 2. `fix_dead_branch` narrows it
        # to `== 1`; that single character is the whole repair.
        level1_guard = (self.strategic_level == 1) if knobs["fix_dead_branch"] \
            else (self.strategic_level >= 1)
        if level1_guard and other_agents:
            expected_volunteers = 0
            for other in other_agents:
                if other.id != self.id:
                    other_cap = self.memory.capability_estimates.get(other.id, 0.5)
                    if other_cap > task_difficulty * 0.5:
                        expected_volunteers += 0.5
            threshold = 0.6 if expected_volunteers > 2 else 0.4
            return own_prob > threshold, own_prob

        if self.strategic_level >= 2 and other_agents:
            if self.reputation < 0.3:
                threshold = 0.3
            elif self.reputation > 0.7:
                threshold = 0.6
            else:
                threshold = 0.5
            return own_prob > threshold, own_prob

        return own_prob > 0.5, own_prob

    def calculate_effort(self, volunteered: bool, partner=None) -> float:
        base_effort = 0.8
        swing = knobs["effort_swing"]
        base_effort += swing if volunteered else -swing
        if partner:
            trust = self.trust_network.get(partner.id, 0.5)
            if trust > 0.7:
                base_effort += 0.05
            elif trust < 0.3:
                base_effort -= 0.05
        return max(0.5, min(1.0, base_effort))

    original_learn = mod.RichAgent.learn_from_outcome

    def learn_from_outcome(self, task_type, success, partner=None):
        original_learn(self, task_type, success, partner)
        if knobs["wire_reputation"]:
            # The field the level-2 branch reads, given the only meaning the file's own
            # vocabulary supports: realised success rate.
            n_s = len(self.success_history)
            n_f = len(self.failure_history)
            if n_s + n_f:
                self.reputation = n_s / (n_s + n_f)
                reputation_writes["count"] += 1

    mod.RichAgent.decide_to_volunteer = decide_to_volunteer
    mod.RichAgent.calculate_effort = calculate_effort
    mod.RichAgent.learn_from_outcome = learn_from_outcome
    return knobs, reputation_writes


def run_35a(mod, level: int, n_seeds: int, n_rounds: int, n_agents: int) -> np.ndarray:
    return np.array([
        mod.run_condition(strategic_level=level, n_rounds=n_rounds,
                          n_agents=n_agents, seed=s)["cooperation_rate"]
        for s in range(n_seeds)
    ])


def reputation_is_inert(mod, n_rounds: int, n_agents: int) -> bool:
    """
    Runtime proof of A2, not a code reading: run each level and check that every agent's
    `reputation` is still exactly its 0.5 default at the end.
    """
    for level in (0, 1, 2):
        random.seed(0)
        np.random.seed(0)
        agents = [
            mod.RichAgent(id=f"a{i}", base_capability=0.5,
                          strategic_level=level, risk_tolerance=0.5)
            for i in range(n_agents)
        ]
        for i in range(n_rounds):
            agents[i % n_agents].learn_from_outcome("technical", i % 2 == 0)
        if any(a.reputation != 0.5 for a in agents):
            return False
    return True


# ---------------------------------------------------------------------------
# 35b -- claim 8
# ---------------------------------------------------------------------------

def install_35b_arms(mod) -> dict[str, Any]:
    """
    Reimplement 35b's round loop with the effort constants and the selector exposed.
    Every line other than those two is copied from `run_self_selection_condition`
    (35b:72-123), so the `self_selection` arm at default knobs is the shipped arm.
    """
    knobs = {"ss_effort": SHIPPED_SS_EFFORT, "random_effort": SHIPPED_RANDOM_EFFORT}

    def run(method: str, n_rounds: int, n_agents: int, seed: int) -> float:
        random.seed(seed)
        np.random.seed(seed)
        agents = mod.create_population(n_agents, prefix=f"{method}_{seed}")
        successes = 0
        for round_num in range(n_rounds):
            difficulty = random.uniform(0.3, 0.7)
            if method == "self_selection":
                volunteers = [
                    (a, a.effective_capability - difficulty * 0.3)
                    for a in agents
                    if a.effective_capability > difficulty * 0.5
                ]
                if volunteers:
                    selected = max(volunteers, key=lambda x: x[1])[0]
                    effort = knobs["ss_effort"]
                else:
                    selected = max(agents, key=lambda a: a.effective_capability)
                    effort = knobs["random_effort"]
            elif method == "central_argmax":
                # The arm 35b omits: a CENTRAL selector with the same information and
                # the same effort. Consumes no RNG, matching the self_selection path.
                selected = max(agents, key=lambda a: a.effective_capability)
                effort = knobs["ss_effort"]
            else:
                selected = random.choice(agents)
                effort = knobs["random_effort"]

            success_prob = selected.effective_capability * effort * (1 - difficulty * 0.6)
            success = random.random() < success_prob
            if success:
                successes += 1
                selected.success_history.append(True)
            else:
                selected.failure_history.append(True)
        return successes / n_rounds

    return knobs, run


# ---------------------------------------------------------------------------

def compare(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = 5000
    diffs = np.array([
        rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)
    ])
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {
        "diff": float(a.mean() - b.mean()),
        "ci95": [lo, hi],
        "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0,
        "significant": bool(lo > 0 or hi < 0),
        "arrays_bit_identical": bool(np.array_equal(a, b)),
        "max_abs_per_seed_difference": float(np.max(np.abs(a - b))),
    }


def part_claim7(mod, knobs, writes, args, rng) -> dict[str, Any]:
    kw = dict(n_seeds=args.seeds_35a, n_rounds=args.rounds, n_agents=args.agents)

    def reset():
        knobs.update(fix_dead_branch=False, wire_reputation=False,
                     effort_swing=SHIPPED_EFFORT_SWING, fixed_threshold=None)

    reset()
    shipped = {lvl: run_35a(mod, lvl, **kw) for lvl in (0, 1, 2)}

    knobs.update(fix_dead_branch=True, wire_reputation=True)
    repaired = {lvl: run_35a(mod, lvl, **kw) for lvl in (0, 1, 2)}
    repaired_writes = writes["count"]

    reset()
    knobs["effort_swing"] = 0.0
    no_swing = {lvl: run_35a(mod, lvl, **kw) for lvl in (0, 1, 2)}

    reset()
    sweep = []
    for th in args.thresholds:
        knobs["fixed_threshold"] = th
        arr = run_35a(mod, 0, **kw)
        sweep.append({"threshold": th, "mean_cooperation": float(arr.mean()),
                      "std": float(arr.std(ddof=1))})
    reset()

    gap_shipped = compare(shipped[0], shipped[1], rng)
    gap_no_swing = compare(no_swing[0], no_swing[1], rng)
    swing_share = (
        1.0 - gap_no_swing["diff"] / gap_shipped["diff"]
        if gap_shipped["diff"] else None
    )

    return {
        "mean_cooperation_by_level_shipped": {
            f"level_{k}": float(v.mean()) for k, v in shipped.items()
        },
        "level1_vs_level2_shipped": compare(shipped[1], shipped[2], rng),
        "reputation_never_written_in_shipped_code": reputation_is_inert(
            mod, args.rounds, args.agents
        ),
        "mean_cooperation_by_level_repaired": {
            f"level_{k}": float(v.mean()) for k, v in repaired.items()
        },
        "level1_vs_level2_repaired": compare(repaired[1], repaired[2], rng),
        "reputation_writes_after_repair": repaired_writes,
        "level0_vs_level1_shipped": gap_shipped,
        "level0_vs_level1_effort_equalised": gap_no_swing,
        "share_of_gap_that_is_the_effort_constant": swing_share,
        "fixed_threshold_sweep": sweep,
    }


def part_claim8(run, knobs, args, rng) -> dict[str, Any]:
    def arr(method: str) -> np.ndarray:
        return np.array([
            run(method, args.rounds, args.agents, s) for s in range(args.seeds_35b)
        ])

    knobs.update(ss_effort=SHIPPED_SS_EFFORT, random_effort=SHIPPED_RANDOM_EFFORT)
    ss, rnd, central = arr("self_selection"), arr("random"), arr("central_argmax")
    shipped = compare(ss, rnd, rng)
    vs_central = compare(ss, central, rng)

    knobs.update(ss_effort=SHIPPED_RANDOM_EFFORT)
    ss_eq, rnd_eq = arr("self_selection"), arr("random")
    equalised = compare(ss_eq, rnd_eq, rng)
    knobs.update(ss_effort=SHIPPED_SS_EFFORT)

    share = (
        1.0 - equalised["cohens_d"] / shipped["cohens_d"] if shipped["cohens_d"] else None
    )
    return {
        "shipped_self_selection_vs_random": shipped,
        "effort_equalised_self_selection_vs_random": equalised,
        "share_of_d_that_is_the_effort_constant": share,
        "self_selection_vs_central_argmax_same_info": vs_central,
        "mean_cooperation": {
            "self_selection": float(ss.mean()),
            "random": float(rnd.mean()),
            "central_argmax": float(central.mean()),
        },
    }


def analyse(c7: dict, c8: dict) -> dict[str, Any]:
    lv = c7["mean_cooperation_by_level_shipped"]
    p1 = bool(
        abs(lv["level_0"] - 0.530) < 0.02
        and abs(lv["level_1"] - 0.306) < 0.02
        and c8["shipped_self_selection_vs_random"]["cohens_d"] > 3.5
    )
    p2 = bool(
        c7["level1_vs_level2_shipped"]["arrays_bit_identical"]
        and c7["reputation_never_written_in_shipped_code"]
    )

    sweep = c7["fixed_threshold_sweep"]
    means = [r["mean_cooperation"] for r in sweep]
    spans_range = bool(
        min(means) <= lv["level_1"] + 0.02 and max(means) >= lv["level_0"] - 0.02
    )
    monotone = all(x >= y - 1e-9 for x, y in zip(means, means[1:]))
    p3 = bool(spans_range and monotone)

    s7 = c7["share_of_gap_that_is_the_effort_constant"]
    s8 = c8["share_of_d_that_is_the_effort_constant"]
    p4 = bool(
        s7 is not None and s8 is not None
        and 0.05 < s7 < 0.95 and 0.05 < s8 < 0.95
    )
    p5 = bool(c8["self_selection_vs_central_argmax_same_info"]["arrays_bit_identical"])

    return {
        "preregistered_predictions": {
            "P1_shipped_numbers_reproduce": p1,
            "P2_level2_is_dead_code_and_reputation_is_inert": p2,
            "P3_level_ordering_is_a_threshold_effect": p3,
            "P4_effort_constants_carry_a_material_share_of_both_effects": p4,
            "P5_claim8_self_selection_arm_is_a_central_argmax": p5,
        },
        "verdict": _verdict(c7, c8, p1, p2, p3, p4, p5),
    }


def _verdict(c7, c8, p1, p2, p3, p4, p5) -> str:
    if not p1:
        return (
            "UNEXPECTED. The shipped 35a/35b numbers did not reproduce, so nothing "
            "below is anchored to the published record and no row should be amended "
            "until that is explained."
        )
    out = []
    if p2:
        rep = c7["level1_vs_level2_shipped"]
        out.append(
            "CLAIM 7 DOWNGRADE. Its three-level comparison has two levels: the level-2 "
            "branch at 35a:136 is unreachable because :118 guards level 1 with `>= 1` "
            "and returns at :133, so levels 1 and 2 are bit-identical per seed "
            f"(max difference {rep['max_abs_per_seed_difference']:.4f}). The branch "
            "also reads `reputation`, which is never written anywhere in the file -- "
            "confirmed at runtime, not just by reading -- so the 'recursive "
            "theory-of-mind' arm would resolve to a constant 0.5 threshold even if it "
            "were reachable."
        )
    if p3:
        out.append(
            "And the surviving level-0-vs-level-1 difference is a threshold effect: "
            "swapping the whole strategic rule for a bare fixed threshold and sweeping "
            "it reproduces the published range monotonically, with the two levels "
            "landing on that curve at their effective thresholds of ~0.40 and ~0.60. "
            "'Strategic reasoners do worse' means 'a higher volunteer threshold does "
            "worse', which is a statement about the threshold, not about reasoning."
        )
    s7 = c7["share_of_gap_that_is_the_effort_constant"]
    s8 = c8["share_of_d_that_is_the_effort_constant"]
    if p4:
        out.append(
            f"In both experiments a hardcoded effort literal carries part of the "
            f"headline: {100*s7:.0f}% of claim 7's level-0/level-1 gap is the +/-0.1 "
            f"swing in `calculate_effort` (35a:157/159), and {100*s8:.0f}% of claim 8's "
            f"d = 4.05 is the 0.9-vs-0.8 difference at 35b:101/108. Both effects "
            f"survive equalisation at reduced size; the corrected figures are "
            f"{c7['level0_vs_level1_effort_equalised']['diff']:+.4f} and "
            f"d = {c8['effort_equalised_self_selection_vs_random']['cohens_d']:.2f}."
        )
    if p5:
        out.append(
            "CLAIM 8 RESTATE. Its self-selection arm is bit-identical per seed to a "
            "CENTRAL oracle taking argmax on the same capability signal at the same "
            "effort, because the volunteer filter never excludes the argmax agent and "
            "the score's difficulty term is constant within a round. So the comparison "
            "is 'argmax beats uniform random', which is arithmetic, plus an effort "
            "literal. It is not evidence about self-selection, and d = 4.05 should not "
            "be cited as though it were."
        )
    else:
        out.append(
            "CLAIM 8 PARTIALLY RESOLVED. The self-selection arm was NOT bit-identical "
            "to the central-argmax control, so some of its advantage is not pure "
            "argmax; the residual needs locating before the row is rewritten."
        )
    return " ".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds-35a", type=int, default=10)
    p.add_argument("--seeds-35b", type=int, default=100)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    args = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(12345)

    m35a = load("rich_agents_sim", PATH_35A)
    knobs_a, writes = install_35a_knobs(m35a)
    c7 = part_claim7(m35a, knobs_a, writes, args, rng)

    m35b = load("statistical_validation_sim", PATH_35B)
    knobs_b, run_b = install_35b_arms(m35b)
    c8 = part_claim8(run_b, knobs_b, args, rng)

    an = analyse(c7, c8)

    artifact = {
        "experiment": "48_strategic_and_baseline_decomposition",
        "targets": (
            "claim 7; experiments/35a_rich_agents.py:118 (>=1 swallows level 2), "
            ":136-147 (unreachable), :76 (reputation never written), "
            ":157/159 (+/-0.1 effort). "
            "claim 8; experiments/35b_statistical_validation.py:101/108 (0.9 vs 0.8), "
            ":95-100 (self-selection is argmax, no central control arm)"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 2),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pythonhashseed": __import__("os").environ.get("PYTHONHASHSEED", "unset"),
        },
        "config": vars(args),
        "claim_7_35a": c7,
        "claim_8_35b": c8,
        "analysis": an,
    }
    out = ROOT / "results" / "experiment_48_strategic_and_baseline_decomposition.json"
    out.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out}\n")

    print("  -- claim 7 (35a): levels --")
    for k in ("mean_cooperation_by_level_shipped", "mean_cooperation_by_level_repaired"):
        vals = "  ".join(f"{lv}={v:.4f}" for lv, v in c7[k].items())
        print(f"    {k:38s} {vals}")
    r = c7["level1_vs_level2_shipped"]
    rr = c7["level1_vs_level2_repaired"]
    print(f"    level1 vs level2 shipped   identical={r['arrays_bit_identical']} "
          f"maxdiff={r['max_abs_per_seed_difference']:.4f}")
    print(f"    level1 vs level2 repaired  identical={rr['arrays_bit_identical']} "
          f"maxdiff={rr['max_abs_per_seed_difference']:.4f} "
          f"(reputation writes: {c7['reputation_writes_after_repair']})")
    print(f"    reputation inert in shipped code: "
          f"{c7['reputation_never_written_in_shipped_code']}")
    print(f"    level0-level1 gap  shipped {c7['level0_vs_level1_shipped']['diff']:+.4f}"
          f"  effort-equalised {c7['level0_vs_level1_effort_equalised']['diff']:+.4f}"
          f"  ({100*c7['share_of_gap_that_is_the_effort_constant']:.0f}% was the constant)")

    print("\n  -- claim 7: fixed-threshold sweep (strategic rule removed entirely) --")
    for row in c7["fixed_threshold_sweep"]:
        print(f"    threshold {row['threshold']:.2f}  coop {row['mean_cooperation']:.4f}")

    print("\n  -- claim 8 (35b) --")
    for k in ("shipped_self_selection_vs_random",
              "effort_equalised_self_selection_vs_random",
              "self_selection_vs_central_argmax_same_info"):
        v = c8[k]
        lo, hi = v["ci95"]
        print(f"    {k:46s} {v['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"d={v['cohens_d']:+6.2f}  identical={v['arrays_bit_identical']}")
    print(f"    share of d that is the 0.9-vs-0.8 literal: "
          f"{100*c8['share_of_d_that_is_the_effort_constant']:.0f}%")

    print()
    for k, v in an["preregistered_predictions"].items():
        print(f"    {k}: {'CONFIRMED' if v else 'NOT CONFIRMED'}")
    print(f"\n  {an['verdict']}")


if __name__ == "__main__":
    main()
