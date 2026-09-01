# Changelog

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
