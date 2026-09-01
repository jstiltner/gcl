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
| 9 | GCL achieves ~97% of MARL performance | 36 | 0.534 vs 0.552 | — | **Validated** |
| 10 | GCL is 25–50× more sample efficient than MARL | 36 | 2 vs 52–102 episodes | — | **Validated** |
| 11 | GCL advantage under drift is significant for ε ≥ 0.15 | 09 | +0.027 to +0.036 | significant at ε ∈ {0.15, 0.20, 0.25} | **Validated** — note GCL slightly *underperforms* at ε = 0 (−0.060) and shows no significant advantage at ε ∈ {0.05, 0.10} |
| 12 | Punishment paradox: harsher consequences decrease cooperation | 15–16 | r = −0.972 | p < 0.001 | **Validated (corrected 2026-09-01)** — original r=−0.951 traced to a synthetic-data generator in Exp 22 (`test_punishment_paradox`), not Exp 15/16's real simulation; see CHANGELOG |
| 13 | Redemption mechanism improves cooperation | 17–18 | +52.7% | p < 0.001 | Validated — **not yet re-verified against Exp 22's synthetic-data pattern (`test_redemption_mechanism`); see CHANGELOG open item** |
| 14 | Failure-first specification reduces hold-up vs incomplete contracts | 21 | −40.4% | [−43.5%, −37.2%], p < 0.001 | **Validated (corrected 2026-09-01)** — original −36.8% traced to the same Exp 22 issue (`test_hart_moore`); see CHANGELOG |
| 15 | Population predictions: protocol convergence, small-world trust, template fitness, specialization Gini | 07 | α=0.16 R²=0.78; clustering=0.75; r=−0.30 p<0.001; Gini=0.78 | 100 agents, 5000 steps | Validated |
| 16 | Dunbar-like coordination limit near ~100 agents | 23 | — | — | Validated (single model; not stress-tested) |
| 17 | 6 theorems (closure, convergence, Nash equilibrium, ...) | tests + sims | — | — | **Empirically validated propositions** — NOT formal mathematical proofs |
| 18 | Real LLMs have privileged self-knowledge exploitable by self-selection | 41 (Part 1/2) | Brier external < self for 4/4 agents | e.g., gpt_direct 0.281 vs 0.484 | **NOT SUPPORTED** — external assessment better calibrated; LLMs sit on the central-assignment side of the Exp 40 phase boundary |
| 19 | Structured selection (self or external) beats random assignment for LLM agents | 41 (Part 2) | +0.183 to +0.200 success | n = 60/condition, single run | Supported (preliminary) |
| 20 | Choice framing ("volunteered" vs "assigned") increases LLM success | 41 (Part 3), 41b | +0.033 (GPT arm), +0.017 pooled | McNemar p = 0.52 / 0.56; n = 120 paired per agent | **NOT DETECTED** — powered re-test (41b) finds no significant effect; effect bounded above ≈ +0.10; sim motivation effect (claim #1) does not transfer to prompted LLMs at this power |

## Retraction / revision history

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
