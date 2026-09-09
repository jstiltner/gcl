# Changelog

## 2026-09-09 — Closed the 2026-09-01 open items; retracted six further figures

**Both open items from the 2026-09-01 entry were contaminated.** They stayed live for a week while
the correction above them read as complete. That is the main lesson of this entry: a partial
correction, presented honestly, still lends credibility to the uncorrected material beside it. Open
items need a deadline, not just a flag.

**Open item 1 — `test_redemption_mechanism()` (Claim #13).** Confirmed synthetic. It draws
`no_redemption = 0.35 + N(0, 0.08)` and `with_redemption = 0.60 + N(0, 0.08)`, and the published
+52.7% matches `results/22_statistical_significance/results.json` to ten decimal places
(0.392622152890477 / 0.5995294236143346 / 52.69882740966349). The real experiment exists and is
**stronger**: `results/17_forgiveness/forgiveness_analysis.json` (n = 5 seeds) gives standard
consequences 0.319 → Redemption(+20%) **0.619 (+94.0%)**, Strong redemption(+30%) 0.649 (+103.4%),
against a no-consequences baseline of 0.519. **The claim survives; the number did not.**

**Open item 2 — `test_population_dynamics()` (Claim #15).** Row 15 was *independently sourced* and
its numbers are correct — it cites Exp 07 directly, as the 2026-09-01 entry hoped. The contamination
was downstream: the published write-ups and jasonstiltner.com carried exp22's numbers (82.3%
convergence, clustering 0.699, Gini 0.745, 26.5% efficiency, "all p < 0.001", "1200+ runs") rather
than this table's. **The site did not match its own claims table**, and nothing checked that it did.

Row 15 needed annotating for a separate reason: Exp 07's own `summary` is `passed: 3, total: 4`.
**Prediction 3 (Template Replicator Dynamics) failed** all three sub-checks (r = −0.299, wrong sign).
"Four properties emerge consistently" was never true. The trust network is also **dense** (density
0.429, 4,248 edges), not sparse as described.

### Three further problems the Exp 22 issue had masked

**Selected comparison (Claims 9 and 10, both retracted).** Neither number was miscalculated.
`0.5335 / 0.5525 = 96.6%`, so "~97% of MARL" is arithmetically right — while the same `analysis`
block records `gcl_rank: 3` and `gcl_is_best: false`. Comparing GCL only against the best baseline
restates "third of five" as "97% of MARL". Likewise "25–50×" is the range across 2 of 4 baselines;
`sample_efficiency_vs_mappo: 1.00` and `sample_efficiency_vs_random: 1.20` were dropped, MAPPO is
MARL, and the discarded *random* baseline shows the threshold barely discriminates. Both arms are
outlier-driven (`episodes_to_50`: QMIX 102.4 ± 435.4, IQL 51.75 ± 214.6, CI lower bound 0.8 on
each). **A verified number inside a selected comparison is still a false claim.**

**Two experiments had no row in `docs/CLAIMS.md` at all** — Exp 08 and Exp 37 — in direct violation
of that file's stated rule that every published claim must match an entry. Both then contradicted
what was being said about them:

- **Exp 08**: GCL is **last of six** on efficiency (0.645; CNP 0.824, FIPA-ACL 0.818, MARL-IQL
  0.755, Auction 0.661). It is lowest on messages (84.0) but **strictly dominated by MARL-IQL**,
  which is non-communicating and sends **zero** messages at 0.755 efficiency — so GCL is not on the
  Pareto frontier. `success_rate` is 1.0 for all six and does not discriminate.
- **Exp 37**: sweeps **five** change frequencies, not four. Every GCL−baseline delta (0.17–2.57 pp)
  sits inside per-arm std of 1.0–2.5 pp at n = 10, no significance test was run, `crossover_iql` and
  `crossover_qmix` are both `null`, and the two baselines **disagree on the sign of the trend** —
  `change_frequency` is episodes *between* shifts, so the QMIX series rises toward the most *stable*
  end, the opposite of the stated "MARL must relearn, GCL adapts" mechanism.

**The absence of a row was itself the signal.** Rule #1 of `docs/CLAIMS.md` was doing useful work
precisely where it was being ignored.

### New: an unresolved contradiction, recorded rather than resolved

Exp 07 (100 agents, 1 seed) reports trust clustering **0.7475**. Exp 23 (7 population sizes 5→200 ×
10 seeds = 70 runs) reports `clustering_coefficient: 0.0` in **every single run, zero variance**, and
correctly concludes "small-world: not detected". Both cannot be right, and neither write-up mentions
the other. Recorded as **open** in `docs/CLAIMS.md` row 16a. No claim depending on trust-network
topology should be published until it is settled.

### What held up

Punishment Paradox (r = −0.972) and Hart-Moore (40.4%, CI [37.2%, 43.5%]) — the 2026-09-01
corrections, now CI-reproduced on every push against a real `commit_sha` and `workflow_run_url`.
The self-selection work (Claims 1–3, Exp 40/41) is the model for the rest of this repo: it carries
its own retraction of the earlier "+81% / 75% from information asymmetry" result and volunteers a
null (Exp 41b, McNemar p = 0.52). Note that `src/gcl/population/` sits at **0% test coverage**, and
it is the subsystem behind the section where the fabricated figures lived.

`22_statistical_significance.py` stays in the tree, unchanged, for the reason given on 2026-09-01:
the fix is to correct the record, not to remove the evidence.

## 2026-09-01 — Corrected Punishment Paradox and Hart-Moore headline stats

**What was wrong:** `experiments/22_statistical_significance.py`'s `test_punishment_paradox()`
and `test_hart_moore()` do not run the real agent-based simulations in
`15_punishment_spiral_analysis.py` / `16_consequence_severity_sweep.py` /
`21_incomplete_contract_theory.py`. They generate samples from hand-picked means with
Gaussian noise (e.g. `base_coop = 0.7 - 0.4 * level`; `incomplete_holdup = 0.4 +
np.random.normal(0, 0.05)`) and run real statistical tests (t-test, Cohen's d, Pearson r) on
that synthetic data. The script says so in a comment: *"We don't need these imports for the
statistical tests which use simulated data for rigorous statistical analysis."*

The published headline numbers — Punishment Paradox `r = -0.951`, Hart-Moore `36.8%`
hold-up reduction (and every t/d value in that section) — all traced to
`results/22_statistical_significance/results.json`, matching to 10+ decimal places. None of
them came from the real simulations, despite being presented (on jasonstiltner.com and in
this repo's own `docs/CLAIMS.md`) as validated simulation output.

This was found while building CI-based reproduction for these claims (see
jasonstiltner2026 repo, `feature/ci-reproduced-results` branch) — trying to wire a `--ci`
flag into the "real" experiment scripts surfaced that they don't actually produce the
published numbers.

**What was fixed:** `experiments/derive_real_headline_stats.py` is a new, standalone script
that imports the real simulation code from `16_consequence_severity_sweep.py` (Punishment
Paradox) and `21_incomplete_contract_theory.py` (Hart-Moore) directly, runs it at n=30
seeds, and computes the same statistical tests on the real output. It does not touch or
depend on `22_statistical_significance.py`.

Corrected numbers (`results/real_headline_stats/real_headline_stats_full.json`):

| Claim | Before (synthetic) | After (real simulation) |
|---|---|---|
| Punishment Paradox correlation | r = −0.951 | **r = −0.972**, p = 1.8e-94 |
| No vs. full consequences | t=36.18, d=9.34 | **t=52.10, d=13.45** |
| Hart-Moore hold-up reduction | 36.8% [28.4%, 45.2%] | **40.4%** [37.2%, 43.5%] (bootstrapped) |
| Hart-Moore Predictions 1 & 2 (investment) | t/d reported | **t/d no longer reported** — this model sets investment as a fixed multiplier per completeness condition (plus small per-interaction noise), so a t/d here reflects the model's construction, not an emergent effect. Real means (0.500 / 0.425 / 0.200 across conditions) are reported instead. |
| Hart-Moore Prediction 3 (hold-ups, incomplete vs. complete) | "4.2× more" | Real: 89.7 hold-up incidents vs. **zero** — hold-ups are structurally impossible under complete contracts in this model, so this is a real stochastic count against a real structural zero, not a ratio. |
| Hart-Moore Prediction 4 (hold-up reduction, real) | — | **t=19.74, d=5.10** |

`docs/CLAIMS.md` rows 12 and 14 updated to match, per this repo's own stated policy
("revised or retracted claims are updated here, not silently rewritten in place").
jasonstiltner.com's Punishment Paradox and Hart-Moore sections corrected to match.

**Open items, not yet re-verified** (same script, different sections — flagged, not fixed,
in this pass; see `docs/CLAIMS.md` row 13):

- `22_statistical_significance.py`'s `test_redemption_mechanism()` (backs Claim #13,
  "Redemption mechanism improves cooperation, +52.7%") uses the same synthetic-data pattern
  (`no_redemption = 0.35 + noise`, `with_redemption = 0.60 + noise`). Not yet checked against
  the real redemption experiments (`17_forgiveness_mechanisms.py`,
  `18_redemption_optimization.py`).
- `22_statistical_significance.py`'s `test_population_dynamics()` also generates synthetic
  data internally, though `docs/CLAIMS.md` row 15 attributes its numbers to Experiment 07
  directly rather than to Experiment 22 — not yet confirmed whether row 15 is actually
  affected or independently sourced.

**Why this matters, stated plainly:** this repo backs a job-search portfolio. A future
reader running these numbers down would have found the same thing in about ten minutes of
reading the code. The fix is to correct the record, not to remove the evidence — this
CHANGELOG stays, `22_statistical_significance.py` stays as-is (it is still useful as a
worked example of the correct statistical machinery, just not as a source of these two
claims), and the corrected script and its real output are the new source of truth for
Punishment Paradox and Hart-Moore going forward, including in CI.
