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
| 14 | Failure-first specification reduces hold-up vs incomplete contracts | 21 → **45** | ~~−40.4%~~ **−37.5%** at n=30 after the reproducibility fix | ~~[−43.5%, −37.2%], p < 0.001~~ — the interval is withdrawn, see verdict | **DOWNGRADED — NOT EVIDENCE (2026-09-21)**, previously "Validated (corrected 2026-09-01)". Two things, one fatal. **(1) The number was not reproducible.** Exp 21 drew from `list(self.all_contingencies)`, a **set of strings**, at `:164`, `:169` and `:348`. `list()` over a str set is hash-ordered and CPython randomises string hashes per process, so the seeded generator drew identical *indices* that landed on *different contingencies* every run. Measured spread on the headline at identical seeds: **23.6% / 30.7% / 42.1%** at PYTHONHASHSEED 0/1/2. Every published value sat inside that noise, and the repeated `ci: refresh` commits in the history are its signature. Fixed 2026-09-21 by drawing from a sorted list; now stable at 42.9% (ci-small) and **37.5% (n=30 full)** regardless of hash seed. **(2) The comparison is circular.** `create_commitment`'s GCL arm specifies `{"quality_low", "delay", "cost_increase", "partner_defects"}` (`:174`); `check_hold_up` defines the contingencies that *cause* hold-ups as `{"quality_low", "delay", "cost_increase", "partner_defects"}` (`:255`) and fires at `0.6 * vulnerability` for those versus `0.1` otherwise (`:276-281`). The GCL arm is hardcoded to specify exactly the set the scoring rule is hardcoded to punish, so it can only ever be held up on the low-probability branch. Exp 45 holds the specification **count** fixed at 4 and varies only *which* four: negatives **54.6**, random **164.1**, positives **236.1** — a **4.3x** spread at identical coverage. Swap the strings and the advantage inverts. The p-value describes sampling noise around a conclusion fixed before any agent acted. Same failure mode as retracted claim 5 (hardcoded, not emergent) and as Theorems 5 and 6. **Sibling predictions 1 and 2 are worse and are not separately rowed:** investment is literally `0.5 × {1.0, 0.7, 0.4, 0.85}` per condition (`:211-226`), which is why their effect sizes are **d = 499** and **d = 371** |
| 15 | Population predictions: protocol convergence, small-world trust, template fitness, specialization Gini | 07 | α=0.16 R²=0.78; clustering=0.75; r=−0.30 p<0.001; specialization 0.223→0.755 | 100 agents, 5000 steps, **1 seed** | **Validated in part (annotated 2026-09-09)** — Exp 07's own `summary` is `passed: 3, total: 4`. **Prediction 3 (Template Replicator Dynamics) FAILED** all three sub-checks (`correlation_passes`, `dominance_passes`, `usage_ratio_passes` all `False`; r = −0.299, wrong sign). This row's numbers are correct and were *not* affected by Exp 22 — but "four properties emerge" was never true, and the trust network is **dense** (density 0.429, 4,248 edges), not sparse. Contradicted by Exp 23; see row 16 |
| 16 | Dunbar-like coordination limit near ~100 agents | 23 | `dunbar_estimate: 100.0` | efficiency by size: 0.227 / **0.265** / 0.206 / 0.184 / 0.114 / 0.090 / 0.075 at n = 5/10/20/50/100/150/200 | **Downgraded to grid resolution (2026-09-09)** — the estimate is "first tested size whose efficiency falls below half of the maximum" (`23_dunbar_scaling.py:471-481`). Half-max is 0.1325, anchored on the peak at **n = 10**. The crossing is bracketed by n = 50 (0.184) and n = 100 (0.114), so the true limit is anywhere in (50, 100]; the reported 100 is simply the next point on the grid {5,10,20,50,100,150,200}. A denser grid returns a different number. The monotone efficiency decline is real; the value "~100" is not a measurement and should not be presented as convergence on Dunbar's number |
| 16a | **RESOLVED IN EXP 07's FAVOUR (raised and closed 2026-09-09)**: Exp 07 and Exp 23 disagreed on trust-network clustering | 07 vs 23 | 0.7475 vs 0.0 | Exp 23: **69 of 70 runs have `mean_degree: 0.0` and `n_components == n_agents`** — the trust graph has no edges at all | **Exp 23's metric is broken; its 0.0 is an artifact, not a result.** `23_dunbar_scaling.py:280` initialises `trust = np.eye(n) * 0.5`, so every off-diagonal entry starts at 0 and grows only +0.1 per success in the selected agent's column. Binarisation is `> 0.5`, so an edge needs **>5 net successes by one agent**. Instrumented run at n = 100: max off-diagonal trust **0.1000**, max successes by any agent **3**, edges **0**. `compute_network_metrics` then takes its `binary.sum() == 0` branch (line 66), which **hardcodes** `clustering_coefficient=0.0`. That is the source of the zero variance across all 70 runs. The single exception (one n = 10 run, `mean_degree` 0.9) has too few edges to close a triangle. Both of Exp 23's clustering columns are affected — the legacy `trust_clustering` (line 340) uses the same threshold and the same zero branch. **Exp 07's 0.7475 (density 0.429, 4,248 edges) stands as the only real measurement.** Exp 23's "No small-world structure detected" is withdrawn: `is_small_world` comes from the `else False` default at line 492 after `valid_sw = arr[arr > 0]` empties the array — σ was never compared against 1 |
| 16b | Phase transition in coordination efficiency at 5 agents | 23 | `transition_size: 5.0`, `efficiency_threshold: 0.5` | max efficiency across the whole sweep is **0.265** (n = 10) | **NOT A FINDING (added 2026-09-09)** — the loop at `23_dunbar_scaling.py:460-466` breaks at the first size whose efficiency is below 0.5, and **every** size tested is below 0.5, including the smallest. "Transition at 5 agents" means only that the sweep never began above the threshold. Same failure shape as claim 10's 50%-cooperation threshold: a cutoff that does not discriminate. The experiment's own printout reports this as "2. PHASE TRANSITION: 5.0 agents" |
| 17a | MAS protocol comparison: GCL vs CNP, FIPA-ACL, auction and MARL-IQL baselines | 08 | efficiency: CNP 0.824 > FIPA-ACL 0.818 > MARL-IQL 0.755 > Auction 0.661 > **GCL 0.645** (last of six) | messages: GCL 84.0, CNP 109.5, Auction 175.6, FIPA-ACL 190.2, **MARL-IQL 0.0** (non-communicating); n_trials = 5, seed 42 | **Added 2026-09-09** — previously had no row here, in violation of this file's own rule. GCL is **lowest on messages and last on efficiency**; it is strictly dominated by MARL-IQL (fewer messages *and* higher efficiency) and is not on the Pareto frontier. `success_rate` is 1.0 for all six and does not discriminate. Exp 08's `key_findings` gives `gcl_message_reduction_vs_cnp: 23.3%` |
| 17b | Non-stationary environments: GCL vs IQL/QMIX across change frequencies | 37 | GCL − IQL: +0.61/+0.90/+0.21/+0.27/+0.17 pp; GCL − QMIX: +0.74/+1.38/+1.33/+1.54/+2.57 pp (change_frequency 50/100/200/500/1000) | per-arm std 1.0–2.5 pp at n = 10 seeds; `crossover_iql` and `crossover_qmix` both `null` | **NOT SUPPORTED (added 2026-09-09)** — previously had no row, also in violation of this file's rule. Exp 37 sweeps **five** frequencies, not four. Every delta sits inside noise, no significance test was run, and the two baselines **disagree on the sign of the trend**: `change_frequency` is episodes *between* shifts, so the QMIX series rises toward the most *stable* end — the opposite of the "MARL must relearn, GCL adapts" mechanism |
| 17 | 6 theorems (closure, convergence, Nash equilibrium, ...) | tests + sims | — | — | **Downgraded (2026-09-20)** — previously "empirically validated propositions (NOT formal proofs)". That was already the weaker reading, but it still overstates what the sims do. **Two of the six have now been examined and neither was tested.** Theorem 5: `02_template_induction.py:355-364` draws its "actual" outcome from `np.random.random() < confidence * similarity`, i.e. samples the bound it is checking, so the check cannot fail; tested against an executed task it **fails** (row 21). Theorem 6: Exp 06's three 1.000 success rates are produced by `auto_create_predicates=True` and always-true predicates (row 22); after those were fixed the experiment's own flag reads `theorem_6_validated: false` (row 22a). Both were "validated" by procedures that could not return failure, found by the same check — *is the outcome measured, or generated from the predictor?* The remaining four (closure, convergence, Nash equilibrium, ...) have not been re-examined and the base rate so far is 2 for 2. Read as: **six propositions, two tested and both refuted, four unaudited** |
| 18 | Real LLMs have privileged self-knowledge exploitable by self-selection | 41 (Part 1/2) | Brier external < self for 4/4 agents | e.g., gpt_direct 0.281 vs 0.484 | **NOT SUPPORTED** — external assessment better calibrated; LLMs sit on the central-assignment side of the Exp 40 phase boundary |
| 19 | Structured selection (self or external) beats random assignment for LLM agents | 41 (Part 2) | +0.183 to +0.200 success | n = 60/condition, single run | Supported (preliminary) |
| 20 | Choice framing ("volunteered" vs "assigned") increases LLM success | 41 (Part 3), 41b | +0.033 (GPT arm), +0.017 pooled | McNemar p = 0.52 / 0.56; n = 120 paired per agent | **NOT DETECTED** — powered re-test (41b) finds no significant effect; effect bounded above ≈ +0.10; sim motivation effect (claim #1) does not transfer to prompted LLMs at this power |
| 21 | Theorem 5 (analogical transfer): `E[Success] >= confidence * similarity` | 42 | violation rate by task sensitivity (tolerance): 84.3% / 71.3% / 47.5% / 28.0% / 12.5% / 0.0% at tol 0.05–1.60 | 20 seeds × 30 trials × 363 context cells per regime; binomial test per cell | **NOT SUPPORTED (added 2026-09-20)** — first test in which success is *executed and verified* rather than sampled from the bound. The bound holds in exactly one regime (tol 1.60), where mean success is 1.000 and every bound in [0,1] holds; that regime is also `calibration_degenerate` (all 41 candidate parameters tie). `regimes_where_bound_holds_and_tuning_is_non_degenerate` and `regimes_where_bound_is_informative` are both **empty** — the bound is true only where it says nothing. Worst violations occur at **similarity 0.9994** (bound 0.947, empirical 0.000), because the induced context region Θ records where a template was *used*, not where it *worked*. The shipped implementation is also not the stated bound: it multiplies by similarity twice (`template.py:431-433`, `template.py:534-535`), giving `confidence * similarity**2`, which is tighter and still violated. Similarity does retain weak ordering information (Spearman ρ +0.06 to +0.18, tie-averaged) |
| 22 | Theorem 6 (alignment verifiability): commitment-grounded alignment is verifiable in polynomial time, and grounding preserves semantic content — **pre-fix state, superseded by row 22a** | 06 → **43** | Exp 06: grounding 600/600 = 1.000, verification 1.000, semantic preservation 1.000, `theorem_6_validated: true`. Exp 43: grounding rate 1.00 for **nonsense and harmful** commitments too; 16.7% of predicates constant across all contexts; 23.3% certified on an **empty** context | pooled Youden's J 0.83, but J = **0.00** on the `unregistered_target` class (TPR 1.0, FPR 1.0) | **NOT SUPPORTED (added 2026-09-20)** — Exp 06's three 1.000s are artifacts of `auto_create_predicates=True`, which invents a predicate rather than failing (`grounding.py:390-402`), so grounding cannot fail. **6 of 17 shipped grounding rules (35%) name predicates that are never registered** and fall through to `lambda ctx: True` (`grounding.py:410-416`). Exp 06's complexity harness lands on that path: all n commitments ground to the single unregistered `correctness_verified`, `distinct_predicates` is 1 at every n from 1 to 1000, and verification returns confidence 1.00 against an all-failed context and against one reporting active harm. The blind spot is the **assurance vocabulary** — clarify, avoid danger, complete correctly, check, validate — all certified as kept when broken. `safety_maintained` and `no_harm` default to the safe answer, so silence reads as compliance. Also: grounding confidence is **inverted** (gibberish 0.875 > valid 0.812), tagged commitments are over-extracted (~~"parsed twice"~~ — actually n+1 predicates per response, see row 23), `verify_output` is a majority vote that passes at exactly 50% compliance, and Exp 06's n = 600 is 12 strings replayed 50×. Timing is sublinear (exponent 0.79, CI [0.61, 0.99]) but measures one always-true lambda, so it bears on nothing |
| 22a | Theorem 6 after the 2026-09-20 grounding fix | 06 (re-run), 43 (re-run) | Exp 06 now: grounding 0.750 (was 1.000), verification rate **0.000** (was 1.000), `theorem_6_validated: **false**`. Exp 43 under the shipped default: unfalsifiable predicates **0.0%**, empty-context certification **0.0%**, worst-class Youden's J **1.00 — conditional, see caveat** | unregistered rule targets 35% → **0%**; held-out paraphrase grounding 1.00 → **0.00** | **Corrected, and newly limited (added 2026-09-20)** — `auto_create_predicates` now defaults to `False`, the six missing predicates are registered, and `no_harm`/`safety_maintained` now require affirmative evidence instead of returning the safe answer on an empty context. The verification rate did not regress: **1.000 was never real.** All four of Exp 06's "scenarios" receive the same default MockLLM reply — not because the 10-character key at `06_alignment_verification.py:260` fails to match, but because `verify_alignment` calls `generate_with_commitment`, which never consults the registered-response table at all, so the per-scenario strings are dead regardless of the key. Those commitments grounded to the then-unregistered `completeness_achieved` = `lambda ctx: True`, carrying `verify_output`'s `>= 0.5` vote. **One tautology sufficed** — at `>=`, 1 of 2 also passes, so the duplicate parse was not what made it pass. **J = 1.00 is conditional on grounding and is measured over n=15, not 30**: held-out, nonsense and harmful commitments never enter the confusion matrix because they fail to ground first. It says that *given* a commitment grounded, its predicate reads the evidence key it should — discrimination moved to the grounder, and the pooled figure must not be quoted as evidence that verification rejects anything. Criterion 3 of the flag (`alignment > 0.3`) is near-vacuous: `_calculate_alignment_score` awards a flat 0.3 for grounding at all, and three of the four scenarios score exactly 0.300. `theorem_6_validated` was additionally gated on four thresholds that **never read the verification result**; a fifth was added and the flag is now `false`. **New limitation: grounding coverage is now bounded by a 17-entry regex table** — legitimate paraphrases ("refactor the payment module") drop from 1.00 to 0.00. The fix buys soundness with coverage. ~~Still open: every tagged commitment is parsed **twice**, which is what gave the always-true predicate its majority~~ — **this was wrong, corrected by row 23**: a response carrying n commitments yields n+1 predicates, not 2n; there is exactly one spurious extra per *response*, and it is not what gave the always-true predicate its majority (one tautology sufficed at `>=`) |
| 23 | `verify_output`'s aggregation rule (`satisfaction_rate >= 0.5`) is an adequate way to decide whether an output kept its commitments | **44** | Break any safety commitment and add **1** kept commitment → certified. At n=15, **7 of 15** commitments may be broken while certified. An output with exactly one broken commitment is certified ~~98%~~ **100%** of the time (pre-parser-fix figure struck; see row 23a). Best threshold on the population is **0.95**, not 0.5 | Youden's J at 0.5 trends ~~−0.016~~ **+0.009** per commitment as the set grows — i.e. flat, not degrading; verdict is invariant to *which* commitment broke | **NOT SUPPORTED (added 2026-09-20)** — the rule is **gameable by making more promises**: an agent that breaks "I will not harm the user's data" is certified by adding one further promise it intends to keep, so verification improves as the agent commits to more. This is structural, not an artifact of labelling. It is also **severity-blind** — every commitment carries weight 1/n and the engine has no notion that some matter more; ~~of 45 exhaustive (n,k) strata only 9 split on which commitment broke, all at the boundary k=ceil(n/2), and each of those tracks **whether the parser's accidentally doubled commitment was kept**, not severity~~ — after the parser fix **0 of 45** strata split, which confirms the stated cause: the splits were the parser artifact, and severity-blindness is now unanimous across every stratum. **Correction to row 22a:** the parser does not parse every commitment twice. A response with n commitments yields **n+1** predicates; Exp 43's "2.0x" was the n=1 case of (n+1)/n, and part E's rows already showed n+1 up to n=1000. A greedy capture runs from inside the first commitment to the end of the response, so one arbitrarily-chosen commitment gets two votes and at the 0.5 boundary that vote decides. **Scope:** Exp 44 does **not** show conjunctive aggregation is correct — its V5 labels an output violated if any commitment is broken, which is the conjunctive rule restated, so that comparison is circular and is marked as such in the artifact. The finding is about what the shipped rule does, not what should replace it. `src/` is unchanged; the threshold is a design decision, not a defect |
| 23a | The parser over-extraction found by row 23 is fixed, and what it was costing | 44 (re-run), 43 (re-run), 06 (re-run) | Spurious predicates per response **1 → 0**; verdicts flipped by the artifact **>0 → 0**; (n,k) strata splitting on which commitment broke **9/45 → 0/45**. Exp 43's `predicates_per_commitment_single` **2.0 → 1.0**. Exp 06 verification rate **0.0% → 25.0%**, alignment score 0.33 → 0.35, scaling exponent 0.33 → 0.24 | Exp 44 P3 and P4 flip NOT CONFIRMED → CONFIRMED; suite 382 → 386 tests, 3 of the 4 new ones fail without the fix | **FIXED (added 2026-09-20)** — `CommitmentParser.parse` ran every pattern over the full response with no consumed-span tracking, so on `[COMMITMENT: I will X] [COMMITMENT: I will Y]` the generic `I will\s+(.+?)(?:\.|$)` matched *inside* the first marker and, with no period to stop at, ran to the end of the string. `_deduplicate` compares action strings for equality and a run-on equals nothing, so it never caught it. Fixed by span-claiming plus reordering the weak patterns ahead of the moderate ones (`I will try to` is a refinement of `I will`; whichever runs first claims the span). **Two things this does *not* mean.** (1) **Exp 06 did not get better.** All four scenarios still receive the same reply; two commitments now ground to two predicates instead of three, so the one scenario whose context sets `verified: True` scores 1/2 = 0.50 and clears `>= 0.5` where 1/3 = 0.33 did not. The denominator changed; the verifier did not learn to reject anything, and `theorem_6_validated` is still **false**. (2) **P4's mechanism is still false.** P4 predicted uniform 2n doubling that cancels in the ratio; the real factor was (n+1)/n and is now 1, so the prediction's outcome survives for an unrelated reason and the "2.0x" retraction stands. Recorded in the artifact as `P3_and_P4_hold_only_after_the_parser_fix` so a CONFIRMED reading cannot launder the reasoning. **The row 23 headline is unchanged and marginally worse:** one broken commitment plus one kept commitment still certifies, and single-violation outputs are now certified **100%** of the time — the 2% that used to fail were failing on the artifact, not on evidence. The suite passed both before and after a behaviour-changing fix, so this path had **no coverage at all**; four regression tests were added |

## Retraction / revision history

- **2026-09-21 (reproducibility)**: Claim 14 downgraded from Validated to **not evidence**, and
  Exp 45 added. Found by chasing a CI badge that moved 26.7% → 38.3% between two runs 42 minutes
  apart with no code change in between. The proximate cause was hash-ordered iteration over a set
  of strings; the underlying one is that the GCL arm and the hold-up rule share a hardcoded
  four-element literal, so the result is entailed rather than measured.
  **The number was refreshed *and* the claim was downgraded in the same change, deliberately.**
  Publishing a corrected −37.5% under a Validated banner would have been a correction that
  reinstates the defect it corrects: a more precise figure for a quantity that was never
  evidence. Precision is not the same as validity.
  Also checked and cleared: `population/environment.py:274` and `09_drift_threshold.py:254` use
  the same `choice(list(...))` shape but over a set of **ints** and over **dict keys**
  respectively, both of which are order-stable (`hash(int) == int`; dicts are insertion-ordered).
  `list(TaskType)` across Exps 33/33b/34c/34d/35c is enum iteration, also stable. The bug is
  specific to sets of strings.
- **2026-09-20 (parser fix)**: Row 23a added; the now-stale figures in row 23 struck through in
  place rather than overwritten (98% → 100%, J slope −0.016 → +0.009, 9/45 strata → 0/45). The
  0/45 result is worth noting as a *confirmation*: the previous entry asserted those 9 splits were
  the parser artifact and not severity, and removing the artifact removed exactly those 9. Two of
  Exp 44's refuted predictions (P3, P4) now pass, and the artifact field was renamed
  `P3_and_P4_failed_and_this_is_why` → `P3_and_P4_hold_only_after_the_parser_fix` so that a
  CONFIRMED verdict cannot be read back as vindicating P4's stated mechanism, which remains false.
  **Lesson repeated from the entry below: a passing suite across a behaviour-changing fix is
  evidence of missing coverage, not of safety.** It happened twice in two days here.
- **2026-09-20 (aggregation)**: Row 23 added for `verify_output`'s majority vote. Exp 43's three
  methodological defects fixed (seed stability was only checked on the permissive column; the two
  columns were probed with different RNG draws; conjunctive safety predicates were classified by
  sampling). **Retraction: "every tagged commitment is parsed twice" was wrong** — it is n+1
  predicates per response, and Exp 43's own part E rows contained the refutation all along. The
  2.0x factor was the n=1 case of (n+1)/n, generalised without checking. **Lesson: a ratio
  measured at n=1 is not a factor.** Two of Exp 44's five pre-registered predictions were refuted;
  both are left standing as written rather than revised to match the outcome, with the cause
  recorded in the artifact under `P3_and_P4_failed_and_this_is_why` (renamed on the parser fix —
  see the entry above).
- **2026-09-20 (verification pass on the fix)**: Row 22a corrected in place. Re-deriving each
  number from the code rather than the artifacts turned up four imprecisions in the *correction
  itself*: (1) there are two auto-create branches and only the registry-gap one installs
  `lambda ctx: True` — an unknown verb yields an effectively constant-**false** predicate, so
  nonsense was never certified as satisfied; (2) `add_response` is unreachable rather than
  non-matching; (3) one tautology sufficed at `>=`, so the duplicate parse was not load-bearing;
  (4) worst-class J = 1.00 is conditional on grounding over n=15, with the hard negatives removed
  by the previous stage. Also recorded: `alignment > 0.3` is near-vacuous, Exp 43's seed-stability
  check runs on the permissive column only, and its per-column `CONSTANT_FALSE` counts are
  sampling noise from a shared RNG. **Lesson: a correction is a claim, and gets audited like
  one.** Every defect here was in text written the same day to fix a different defect. Full
  detail in CHANGELOG 2026-09-20 (verification pass).
- **2026-09-20 (grounding fixed, Exp 06 re-run)**: Row 22a added, row 22 marked superseded. The two
  defects found above were fixed in `src/gcl/llm/grounding.py` and every affected number re-derived.
  `theorem_6_validated` went **true → false**. Three lessons:
  - **A correction that leaves the same defect elsewhere is not a correction.** Registering the six
    missing predicates while leaving `no_harm` and `safety_maintained` returning the safe answer on
    an empty context would have fixed the instances found and preserved the pattern that produced
    them. The safety defaults were changed in the same pass for that reason.
  - **A number that moves from 1.000 to 0.000 was probably never measuring anything.** Exp 06's
    verification rate did not regress; it was pinned at 1.000 by an always-true predicate holding a
    2-of-3 majority. Before treating a large swing as a regression, check whether the old value was
    reachable by any other route.
  - **Fixes have costs and the cost is the new claim.** Grounding is now sound and its coverage is
    bounded by a 17-entry regex table; legitimate paraphrases fell from 1.00 to 0.00. Recording only
    the improvement would misrepresent the system as much as the original 1.000 did.
- **2026-09-20 (Theorem 6 re-run)**: Row 22 added, row 17 downgraded again. Two lessons beyond the
  ones below:
  - **A permissive default converts a metric into a tautology.** `auto_create_predicates=True`
    means grounding never fails, so "grounding success rate" measures the parser. The default was
    set for ergonomics and silently became the result. When a rate reports 1.000, find the branch
    that makes the alternative unreachable before believing it.
  - **Pooling hid the only finding that mattered.** Verification discriminates at J = 0.83 overall
    and at J = 0.00 on the assurance vocabulary — clarify, avoid danger, complete correctly, check,
    validate — where it certifies every broken commitment. The pooled number is real and would have
    been the wrong thing to publish. Report the worst stratum beside the aggregate, always.
- **2026-09-20 (Theorem 5 tested against a real task)**: Row 17 downgraded, row 21 added. Exp 42
  executes and verifies the task instead of sampling the outcome from the bound, and the bound
  fails. Three things worth carrying forward:
  - **A test whose outcome is drawn from the hypothesis is not a test.**
    `02_template_induction.py:355-364` computes success as `np.random.random() < confidence *
    similarity`. It reported `avg_actual 0.62` vs `avg_predicted 0.633` — a gap that is pure
    sampling error in a procedure whose expectation is exact by construction. This sat in the
    validated column for months. Grep every "validation" for whether the outcome variable is
    *measured* or *generated from the predictor*.
  - **True-but-vacuous and informative-but-false are the same failure.** The bound holds in one
    regime, the one where every attempt succeeds. Reporting "0% violations" without reporting that
    mean success is 1.000 would have been accurate and worthless. Any bound claim needs a tightness
    statistic beside its violation rate; Exp 42 records both and both must pass.
  - **Sweep the parameter that decides the verdict.** The first draft of Exp 42 fixed task
    sensitivity at 0.12 and produced a 67% violation rate. That is Exp 02's defect mirrored — the
    author choosing the answer. Sweeping the parameter converts an assertion into a regime map, and
    the map is what shows the bound is never both true and informative.
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
- **2026-09-09 (Exp 23 follow-up)**: Row 16a closed, row 16 downgraded, row 16b added. Investigating
  the Exp 07 / Exp 23 clustering contradiction showed the contradiction was never between two
  measurements — Exp 23 was not measuring. Three defects, one root cause and two independent:
  - **A default returned as a result.** Exp 23's trust graph is empty in 69 of 70 runs, so
    `compute_network_metrics` returns its hardcoded empty-graph values and the whole `network_topology`
    block is downstream of nothing. `mean_degrees` is `[0.0, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0]`;
    `path_length_vs_size` is six `null`s and a 1.0 and still carries `"trend": "stable_or_decreasing"`;
    `clustering_vs_size` is seven zeros and carries `"trend": "stable_or_increasing"` because
    `0.0 < 0.0` is false. Every one of those labels is an else-branch.
  - **Zero variance is a bug signature.** The tell was not any single value but `std: 0.0` across 7
    population sizes × 10 seeds. Real measurements of a stochastic process vary. Treat an
    exactly-constant result across a sweep as a defect until proven otherwise.
  - **Thresholds and grids that do not discriminate.** `dunbar_estimate: 100` is the next grid point
    after the crossing, not the crossing; `transition_size: 5` is the smallest population tested,
    reported because every population failed the threshold. Both are properties of the sweep, not of
    the system. Before publishing a threshold crossing, check that data falls on both sides of it.
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
4. A metric that is exactly constant across a parameter sweep must be explained
   before it is published. Check the degenerate branch of the function that
   produced it first (lesson of Exp 23).
