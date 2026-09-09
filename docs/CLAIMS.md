# Canonical Claims Table

**This file is the single source of truth for all quantitative claims made in
this repository.** Every claim in the README, paper draft, and findings
documents must match an entry here. When a claim is revised or retracted, this
table is updated and superseded documents are annotated — not silently rewritten.

Last updated: August 2026 (post-Experiment 41).

| # | Claim | Experiment | Effect size | 95% CI / stats | Status |
|---|---|---|---|---|---|
| 1 | Emergent motivation: self-selection with emergent effort beats a truly optimal (argmax) oracle | 40 (Part A/C) | +0.065 cooperation | [+0.050, +0.080], d = 1.68 | **Validated** |
| 2 | Information asymmetry gives self-selection no advantage over an optimal oracle when effort is fixed | 40 (Part A) | +0.000 | [−0.013, +0.013], d = 0.00 | **Validated (null)** |
| 3 | Observability phase boundary: self-selection wins iff coordinator observation noise exceeds agent self-knowledge noise | 40 (Part B) | up to ±0.127 | significant when noise gap ≥ 0.1 | **Validated** |
| 4 | Self-selection beats optimal external matching by 81%; information asymmetry ≈ 75% of advantage | 39 | — | — | **RETRACTED** — Exp 39's oracle was not optimal (anti-overkill objective); see Exp 40 |
| 5 | Effort > information mechanism decomposition (+0.052 vs +0.045) | 34A | — | — | **RETRACTED** — effort was a hardcoded parameter, not emergent; see Exp 39/40 |
| 6 | Specialization does NOT emerge naturally | 33, 33b, 34C-E | HHI < 0.02 | robust across sharing rates, horizons, scarcity | **Validated** |
| 7 | Simple agents outperform strategic reasoners | 35A | 0.530 vs 0.306 | — | **Validated** |
| 8 | Self-selection advantage over random/naive baselines is statistically robust | 35B | d = 4.05 | p < 10⁻⁷², power = 1.0 | Validated (note: baseline set predates corrected oracle) |
| 9 | GCL achieves ~97% of MARL performance | 36 | 0.534 vs 0.552 | — | **RETRACTED (2026-09-09)** — arithmetic correct, comparison selected. Exp 36 records `gcl_rank: 3`, `gcl_is_best: false`. Full ranking: IQL 0.5525 > QMIX 0.542 > **GCL 0.5335** > MAPPO 0.522 > random 0.4745. Comparing only against the best baseline restates "third of five" as "97% of MARL". See row 9a |
| 9a | GCL is mid-field against MARL baselines — above MAPPO and random, below IQL and QMIX | 36 | 0.5335 vs 0.4745–0.5525 | d = −0.29 vs IQL, −0.13 vs QMIX, +0.19 vs MAPPO, +1.10 vs random | **Validated** — this is what Exp 36's `analysis` block actually reports |
| 10 | GCL is 25–50× more sample efficient than MARL | 36 | 2 vs 52–102 episodes | — | **RETRACTED (2026-09-09)** — the range spans 2 of 4 baselines. `sample_efficiency_vs_mappo: 1.00` and `sample_efficiency_vs_random: 1.20` were omitted, and MAPPO is MARL. Also outlier-driven: `episodes_to_50` is QMIX 102.4 ± 435.4 and IQL 51.75 ± 214.6, both with a CI lower bound of 0.8. See row 10a |
| 10a | GCL reaches the 50%-cooperation threshold without training; two of four MARL baselines take longer, with high variance | 36 | GCL 2.05 episodes; QMIX 49.95×, IQL 25.24×, MAPPO 1.00×, random 1.20× | per-arm std ≈ 4× the mean on both slow arms | **Validated, weak** — the *random* baseline scores 0.475, so a 50% threshold discriminates poorly and should not carry a headline |
| 11 | GCL advantage under drift is significant for ε ≥ 0.15 | 09 | +0.027 to +0.036 | significant at ε ∈ {0.15, 0.20, 0.25} | **Validated** — note GCL slightly *underperforms* at ε = 0 (−0.060) and shows no significant advantage at ε ∈ {0.05, 0.10} |
| 12 | Punishment paradox: harsher consequences decrease cooperation | 15–16 | r = −0.972 | p < 0.001 | **Validated (corrected 2026-09-01)** — original r=−0.951 traced to a synthetic-data generator in Exp 22 (`test_punishment_paradox`), not Exp 15/16's real simulation; see CHANGELOG |
| 13 | Redemption mechanism improves cooperation | 17–18 | ~~+52.7%~~ **+94.0%** | std 0.035, n = 5 seeds | **Validated (corrected 2026-09-09)** — the 2026-09-01 open item is now closed and **confirmed contaminated**: +52.7% traced to Exp 22's `test_redemption_mechanism`, which draws `no_redemption = 0.35 + N(0,0.08)` and `with_redemption = 0.60 + N(0,0.08)`. The real experiment is *stronger*: Exp 17 gives standard consequences 0.319 → Redemption(+20%) **0.619 (+94.0%)**, Strong redemption(+30%) 0.649 (+103.4%). The claim survives; the number did not |
| 14 | Failure-first specification reduces hold-up vs incomplete contracts | 21 | −40.4% | [−43.5%, −37.2%], p < 0.001 | **Validated (corrected 2026-09-01)** — original −36.8% traced to the same Exp 22 issue (`test_hart_moore`); see CHANGELOG |
| 15 | Population predictions: protocol convergence, small-world trust, template fitness, specialization Gini | 07 | α=0.16 R²=0.78; clustering=0.75; r=−0.30 p<0.001; specialization 0.223→0.755 | 100 agents, 5000 steps, **1 seed** | **Validated in part (annotated 2026-09-09)** — Exp 07's own `summary` is `passed: 3, total: 4`. **Prediction 3 (Template Replicator Dynamics) FAILED** all three sub-checks (`correlation_passes`, `dominance_passes`, `usage_ratio_passes` all `False`; r = −0.299, wrong sign). This row's numbers are correct and were *not* affected by Exp 22 — but "four properties emerge" was never true, and the trust network is **dense** (density 0.429, 4,248 edges), not sparse. Contradicted by Exp 23; see row 16 |
| 16 | Dunbar-like coordination limit near ~100 agents | 23 | — | — | Validated (single model; not stress-tested) |
| 16a | **UNRESOLVED CONTRADICTION (raised 2026-09-09)**: Exp 07 and Exp 23 disagree on trust-network clustering | 07 vs 23 | 0.7475 vs 0.0 | Exp 07: 100 agents, 1 seed. Exp 23: 7 population sizes (5→200) × 10 seeds = 70 runs, `clustering_coefficient: 0.0` in **every** run, zero variance | **OPEN** — Exp 23 correctly reports "small-world: not detected", while Exp 07 reports small-world clustering. Both cannot be right. Neither experiment's write-up acknowledges the other. No claim depending on trust-network topology should be published until this is resolved |
| 17a | MAS protocol comparison: GCL vs CNP, FIPA-ACL, auction and MARL-IQL baselines | 08 | efficiency: CNP 0.824 > FIPA-ACL 0.818 > MARL-IQL 0.755 > Auction 0.661 > **GCL 0.645** (last of six) | messages: GCL 84.0, CNP 109.5, Auction 175.6, FIPA-ACL 190.2, **MARL-IQL 0.0** (non-communicating); n_trials = 5, seed 42 | **Added 2026-09-09** — previously had no row here, in violation of this file's own rule. GCL is **lowest on messages and last on efficiency**; it is strictly dominated by MARL-IQL (fewer messages *and* higher efficiency) and is not on the Pareto frontier. `success_rate` is 1.0 for all six and does not discriminate. Exp 08's `key_findings` gives `gcl_message_reduction_vs_cnp: 23.3%` |
| 17b | Non-stationary environments: GCL vs IQL/QMIX across change frequencies | 37 | GCL − IQL: +0.61/+0.90/+0.21/+0.27/+0.17 pp; GCL − QMIX: +0.74/+1.38/+1.33/+1.54/+2.57 pp (change_frequency 50/100/200/500/1000) | per-arm std 1.0–2.5 pp at n = 10 seeds; `crossover_iql` and `crossover_qmix` both `null` | **NOT SUPPORTED (added 2026-09-09)** — previously had no row, also in violation of this file's rule. Exp 37 sweeps **five** frequencies, not four. Every delta sits inside noise, no significance test was run, and the two baselines **disagree on the sign of the trend**: `change_frequency` is episodes *between* shifts, so the QMIX series rises toward the most *stable* end — the opposite of the "MARL must relearn, GCL adapts" mechanism |
| 17 | 6 theorems (closure, convergence, Nash equilibrium, ...) | tests + sims | — | — | **Empirically validated propositions** — NOT formal mathematical proofs |
| 18 | Real LLMs have privileged self-knowledge exploitable by self-selection | 41 (Part 1/2) | Brier external < self for 4/4 agents | e.g., gpt_direct 0.281 vs 0.484 | **NOT SUPPORTED** — external assessment better calibrated; LLMs sit on the central-assignment side of the Exp 40 phase boundary |
| 19 | Structured selection (self or external) beats random assignment for LLM agents | 41 (Part 2) | +0.183 to +0.200 success | n = 60/condition, single run | Supported (preliminary) |
| 20 | Choice framing ("volunteered" vs "assigned") increases LLM success | 41 (Part 3), 41b | +0.033 (GPT arm), +0.017 pooled | McNemar p = 0.52 / 0.56; n = 120 paired per agent | **NOT DETECTED** — powered re-test (41b) finds no significant effect; effect bounded above ≈ +0.10; sim motivation effect (claim #1) does not transfer to prompted LLMs at this power |

## Retraction / revision history

- **2026-09-09 (claim audit)**: Claims 9 and 10 retracted, claim 13 corrected, claims 15 and 16
  annotated, and rows 9a/10a/16a/17a/17b added. This pass closed the two open items left by the
  2026-09-01 correction — **both were contaminated** — and found three further problems that the
  Exp 22 issue had masked:
  - **Selected comparison.** Claims 9 and 10 were arithmetically correct and still false. Each
    compared GCL against a favourable subset of Exp 36's baselines while the same `analysis` block
    recorded `gcl_rank: 3` and `gcl_is_best: false`. A verified number inside a selected comparison
    is still a false claim.
  - **Missing rows.** Experiments 08 and 37 were published without any row in this table, in direct
    violation of its stated rule. Both turned out to contradict what was being said about them — Exp
    08 places GCL *last of six* on efficiency, and Exp 37's deltas are entirely inside noise. The
    absence of a row was itself the warning sign.
  - **A partial correction reads as a completed one.** The 2026-09-01 entry was honest and correctly
    scoped, and its two deferred open items then sat live for a week beside it. A visible correction
    lends credibility to the uncorrected material next to it. Open items need a deadline, not just a
    flag.
- **Aug 2026 (Exp 40)**: Claims 4–5 retracted. Exp 39's "perfect information
  oracle" scored by closeness-of-fit, penalizing over-qualified agents under a
  success model monotonic in capability — it was not optimal. Against a true
  argmax oracle the information advantage is +0.000; the surviving mechanism is
  emergent motivation (+0.065, d = 1.68). See `docs/EXPERIMENT_40_FINDINGS.md`.
- **Jan 2026 (Exp 39)**: Exp 34A's "effort > information" decomposition
  retracted; effort was a hardcoded parameter. See
  `docs/RESEARCH_REPORT_EXPERIMENTS_30_39_REVISED.md`.

## Editorial rules

1. New quantitative claims require an entry here before appearing elsewhere.
2. Any baseline described as "optimal" must state the objective it optimizes
   and why that objective is optimal under the success model (lesson of Exp 40).
3. Superseded reports move to `docs/archive/` with a pointer, preserving the
   revision trail.
