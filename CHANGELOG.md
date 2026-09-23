# Changelog

## 2026-09-22 (two-arm decomposition) — eight claims re-run correctly; seven do not survive

### Scope

The 2026-09-21 pass identified eight claim rows whose experiments never import `gcl` and whose
two arms looked like they might not differ in the way the claim says. Rather than retracting on
the code defect alone, each was **re-run correctly** and the *holding* evaluated against the
corrected run. Four new experiments:

| exp | targets | published claim |
|---|---|---|
| `46_specialization_corrected.py` | claim 6 | specialization does not emerge |
| `47_oracle_decomposition.py` | claims 1, 2, 3 | emergent motivation; information null; phase boundary |
| `48_strategic_and_baseline_decomposition.py` | claims 7, 8 | simple beats strategic; self-selection vs baselines |
| `49_marl_comparison_corrected.py` | claims 9a, 10a | mid-field vs MARL; sample efficiency |

Outcome: **1 and 6 retracted, 9a and 10a retracted, 3 / 7 / 8 restated or downgraded, 2's
conclusion kept on replaced evidence.** Every corrected experiment first reproduces its published
number exactly, so nothing below floats free of the record.

### Method

One question found all eight: **in a two-arm comparison, what actually differs between the arms?**
The decisive form is to make the arms identical in the one respect the claim is about and require
the effect to go to **exactly** zero — `np.array_equal` on the per-seed arrays, not "close to".
It came back bit-identical five times:

| comparison | shipped effect | after equalising |
|---|---|---|
| claim 1: self-selection vs oracle, relabel the oracle's pick as volunteered | +0.0651, d = 1.68 | **+0.0000, bit-identical** |
| claim 2: self-selection vs oracle at σ = 0 | +0.0000, CI [−0.013, +0.013] | **bit-identical — the CI is a bootstrap of an array against itself** |
| claim 7: strategic level 1 vs level 2 | (published as distinct) | **bit-identical; `:136` is unreachable** |
| claim 8: self-selection vs central argmax on the same signal | d = 4.05 vs random | **+0.0000, bit-identical** |
| claim 9a: GCL vs a constant all-volunteer policy | GCL 3rd of 5 | **bit-identical, every seed, every episode** |

### What each one turned out to be

**Claim 1 — a boolean times a constant.** The advantage is the `volunteered` flag, which reaches
the outcome through `ownership_bonus = 0.05 if volunteered else 0.0` (`40_corrected_oracle.py:75`)
and a `commitment_level` update gated on the same flag (`:60-68`). Sweeping the bonus traces the
headline monotonically (0.00 → +0.0304 … **0.05 → +0.0651** … 0.20 → +0.1277); remove the gate too
and bonus 0.00 gives bit-identical arms. **+0.0304 of the effect is the gated commitment update,
+0.0347 is the bonus constant.** The premise is defensible; reporting it as a measured output with
a CI is not.

**Claim 2 — the only survivor.** At σ = 0 the two arms are one computation run twice, so the
published null could not have come out otherwise. Re-tested on the diagonal of a 5×5 noise grid
with equal but **independently drawn** noise: +0.0000 / +0.0024 / +0.0059 / −0.0270 / +0.0019,
none significant. The conclusion stands; the evidence for it is new.

**Claim 3 — the boundary is real and is about signal quality.** A *central* oracle assigning by
argmax on the agents' own `perceived_capability` is bit-identical to self-selection at **all 25
grid cells**. So the boundary sits at σ_oracle = σ_self because that is where the coordinator's
estimate becomes the worse one. "Argmax on the better estimate wins" is true; it is not a finding
about decentralisation.

**Claim 6 — the correction runs *against* the repo.** Three layers. The published `HHI < 0.02` is
**~17× below the metric's floor** of 1/3 for any agent that ever acts; it is reachable only via the
`total == 0 → 0.0` sentinel, averaged over all 30 agents. Exactly **one** agent ever acts: `get_volunteers`
returns a single `max(...)`, and the model's only capacity constraint is inert —
`get_available_agents` filters on `resources > 0` (`structures/base.py:138-140`), but
`Agent.resources` (`agents/agent.py:84`) is **written nowhere in the codebase** and that filter is
its only reader. Nothing rotates, so the headline is 29 zeros averaged with one agent's real HHI.
Corrected — cooldown to break the monopoly, score only active agents, compare against a
**matched permutation null** holding each agent's task count and the global type mix fixed —
specialization **does** emerge: Δ **+0.0390** / **+0.0634** / **+0.0399** at cooldown 1/3/10, all
CIs excluding zero at 30 seeds. The "robust across sharing rates, horizons, scarcity" column is the
same artefact repeated: Exp 33 reports 0.0333 / 0.01137 / 0.01140 / 0.01126 / 0.01134, 34C reports
0.01127–0.01143 across its whole frontier, 34D's max is 0.01134, 34E gives 0.01131 / 0.01309 /
0.01776 — one active agent's HHI over 30, with the first being **exactly 1/30**. A metric that is
near-constant across four independent sweeps was editorial rule 4 firing and being read as
robustness. **33b is the sole exception** (0.331–0.511, so more than one agent acts) and is
untested against the null. A forced-assignment positive control gives Δ +0.4279, and a
**type-blind mechanism ablation** collapses Δ to +0.0046 / −0.0033 / −0.0048, which is what shows
the effect is the model's own feedback loop and not the cooldown. Reported with its magnitude:
**6.0% / 10.2% / 7.4%** of the distance from the null to full specialization. The claim is
contradicted by its own simulation.

**Claim 7 — two levels, not three, and a third of the gap is a literal.** `35a_rich_agents.py:118`
guards level 1 with `>= 1` and returns at `:133`, so `:136` is dead. The dead branch reads
`self.reputation`, which is declared at `:76` and **never written** — confirmed at runtime.
Repaired, level 2 scores 0.4490, between the other two. Swapping the whole strategic rule for a
bare fixed threshold reproduces the published range monotonically (0.30 → 0.5560 … **0.60 →
0.3060** … 0.70 → 0.2780), with level 1 landing on that curve to four decimals. **32%** of the
0.224 gap is `calculate_effort`'s ±0.1 swing (`:157`/`:159`).

**Claim 8 — argmax beats uniform random.** The self-selection arm is bit-identical to a central
coordinator taking `max(agents, key=effective_capability)`, because the volunteer filter never
excludes the argmax agent. **21%** of d = 4.05 is `effort = 0.9` (`35b:101`) against `0.8`
(`:108`); equalised, d = 3.19. The p < 10⁻⁷² describes sampling noise around a conclusion fixed by
the selection rule.

**Claims 9a / 10a — the environment does the selecting, so the comparison cannot rank.** All five
`step()` methods end `max(volunteers, key=capability)`, so a policy picks only the volunteer *set*,
reward is monotone non-decreasing in that set, and "everyone volunteers" is optimal by
construction. An `AllVolunteer` arm — no learning, no observation — reaches **0.5516** and is
beaten on **0 of 240,000** paired episodes. GCL reaches 0.5516 and is **bit-identical** to it: its
rule (`:67-69`) always admits the most capable agent and `max` discards the rest. The field spans
**0.013** (IQL 0.5513 / QMIX 0.5396 / MAPPO 0.5388 / random 0.5384) above a single-volunteer floor
of 0.3757. Separately, raising the random arm's `effort` from `0.8` (`:408`) to the `0.9` every
other arm uses moves random **from last to second**. And the real `qmix.py` / `mappo.py` sitting in
the same directory are **never imported by the runner** — `:35` imports only `Agent` — so "QMIX"
and "MAPPO" are tabular stand-ins; wired in for real they land at 0.5229 and 0.5260, inside the
same band. Row 10a's 25–50× multiples are a metric artefact: `episodes_to_50` is a **cumulative
running mean** (`:453-455`), so a seed that succeeds on episode 1 scores 0 and the mean over seeds
is set by the seeds that missed. Under the trailing-window rate the headline already uses, the
worst ratio falls from 49.95× to **1.61×**.

### Defects in this pass's own harnesses, found and fixed before publishing

Three of Exp 49's five pre-registered predictions failed on the first run. All three failures were
mine:

1. The effort knob was initialised to `0.8` and never set to `0.9`, so the "effort-equalised" arm
   silently re-ran the shipped one and produced an identical number.
2. `SimpleMAPPO.get_actions` returns `(actions, log_probs)` while the other four arms return a
   bare list, so the recorded action vector was a tuple and its volunteer set came out empty —
   MAPPO scored 0.0000.
3. P3 and P5 were scored on the shipped unpaired harness, where arms never see the same task
   sequence and the reward is a coin. At 5 seeds its noise put the **ceiling arm below IQL** and
   **real QMIX above the ceiling**, both impossible by construction. Fixed by adding a paired task
   tape scored on the realised success probability rather than its coin flip, which also makes the
   dominance checkable episode-by-episode instead of on average.

Two predictions were amended rather than silently relaxed, with the reason and the observed
numbers recorded in the source: Exp 46's P1 (`n_active == 1.0` → `<= 2.0`; 1 of 30 seeds had two
actors) and P2/P4 (scored one-sided, since they are directional); Exp 49's P3 (moved to the paired
run) and P4 (a median clause dropped that tested something row 10a never asserts). Where an
amendment makes a prediction **easier** to confirm, that is stated in the file.

### Reproducibility

All four experiments are bit-identical across `PYTHONHASHSEED` 0, 1 and 42, compared on the
per-seed arrays rather than on the summary statistics.

### Still open, not addressed here

- **33b** has not been re-run through the matched-permutation null. It is the only one of claim 6's
  supporting experiments whose numbers are above the metric's floor, so it is the only one that
  might carry real information; it is **unsupported, not refuted**. (33, 34C, 34D and 34E are all
  the sentinel artefact and are covered by the retraction.)
- 14 experiments have results but no row in `docs/CLAIMS.md`, in violation of that file's own
  rule: 24, 25, 26, 26b, 27, 30, 31, 32, 34a, 34b, 35c, 35d, 35e, 38. (`35c` reuses claim 6's
  broken HHI metric.)
- Exps 37 and 38 define their own `QMIXAgent` classes and need the same "is the baseline real?"
  check that sank 9a/10a. Row 17b already covers 37.
- Claim 11 (Exp 09) does import `gcl` and was excluded from this pass; it is unaudited.
- `ci-reproduce.yml` still has no `pull_request` trigger, so CI has never validated a PR here.

## 2026-09-21 (reproducibility) — the CI badge was never reproducible; claim 14 downgraded from Validated to not-evidence

### How this was found

The `ci-reproduce` workflow published a Hart-Moore hold-up reduction of **26.7%**, then
**38.3%** forty-two minutes later, with no code change between the two runs. Ruled out the
2026-09-20 parser fix as the cause first: `derive_real_headline_stats.py` never imports
`gcl`, and neither does Exp 21.

Two local runs at identical seeds and scale gave **33.3%** and **40.4%**. Pinning hash
randomization made it deterministic:

| `PYTHONHASHSEED` | hold-up reduction |
|---|---|
| 0 | 23.6% |
| 1 | 30.7% |
| 2 | 42.1% |

### The bug

`21_incomplete_contract_theory.py` drew from `list(self.all_contingencies)` at `:164`,
`:169` and `:348`. `all_contingencies` is a **set of strings**, and `list()` over one is
hash-ordered — CPython randomises string hashes per process unless `PYTHONHASHSEED` is
pinned. The seeded generator therefore drew identical *indices* on every run, and those
indices landed on *different contingencies*.

The workflow is named `ci-reproduce` and exists to demonstrate reproducibility. It was
republishing a different number on essentially every run: the badge held
**26.7 / 27.7 / 28.5 / 32.2 / 36.8 / 38.3 / 39.8%** across successive runs, a different
value on **9 of 10** refreshes.

> **Corrected 2026-09-21, same day.** This entry originally said the repeated
> `ci: refresh CI-reproduced headline stats` commits "are the bug's signature". That is
> wrong as stated. `ci_results/latest.json` embeds `generated_at`, `commit_sha` and
> `workflow_run_url`, so the job commits on **every** run whether or not a statistic
> moved — the commit count proves nothing. The badge *values* above are the evidence.
> Caught by running a `workflow_dispatch` re-run after the merge specifically to confirm
> the job would go quiet, finding that it committed anyway, and diffing it rather than
> assuming the fix had failed. Asserting a cause without verifying it is the exact
> failure this changelog entry criticises elsewhere; recording it here rather than
> quietly amending the sentence.

### The fix

A canonical `self.contingency_order = sorted(self.all_contingencies)` is drawn from
instead. Stable at **42.9%** (ci-small) across hash seeds 0–3 and unpinned. The workflow
also pins `PYTHONHASHSEED: "0"` as a guard, not as the fix.

**Verified in CI after merge, not just locally.** A `workflow_dispatch` re-run on an
unchanged `main` produced a diff touching only `generated_at`, `commit_sha` and
`workflow_run_url`; `badge-hart-moore.json` was byte-identical and `latest.json`'s
`hart_moore` block was unchanged. That is the first time this workflow has reproduced its
own numbers.

**Swept for the same pattern and cleared three false positives**, verified rather than
assumed: `population/environment.py:274` iterates a set of **ints** (`hash(int) == int`,
order-stable); `09_drift_threshold.py:254` iterates **dict keys** (insertion-ordered);
`list(TaskType)` in Exps 33/33b/34c/34d/35c is **enum** iteration (definition-ordered).
Only sets of strings are affected. No `src/` change was needed.

### Re-derived at full scale, and then downgraded anyway

| | value |
|---|---|
| hold-up reduction, n=30 | **37.5%** (was published as 40.4%) |
| prediction 4 | t = 14.29, d = 3.69 |

And then, reading the code to confirm the number meant what it claimed, a second and worse
problem. Two hardcoded sets 80 lines apart are character-for-character identical:

```
:174  create_commitment, GCL arm
      failure_contingencies = {"quality_low", "delay", "cost_increase", "partner_defects"}

:255  check_hold_up
      negative_contingencies = {"quality_low", "delay", "cost_increase", "partner_defects"}
```

`check_hold_up` returns False when the realized contingency is specified (`:257`), and
otherwise fires at `0.6 * vulnerability` for a negative contingency versus `0.1` for a
positive one (`:276-281`). **The GCL arm is hardcoded to specify exactly the set the
scoring rule is hardcoded to punish**, so it can only ever be held up on the
low-probability branch.

### Exp 45 — added to separate the two explanations

Holds the specification *count* fixed at 4 for every arm and varies only *which* four:

| arm | specified set | mean hold-ups |
|---|---|---|
| shipped | the 4 negatives | **54.6** |
| control | 4 random | 164.1 |
| control | 4 positives | 236.1 |

**4.3x spread at identical coverage.** Swap the four strings and the advantage inverts.
Harness validity: the shipped arm reproduces `derive_real_headline_stats --scale full`
exactly (54.6), so the arms differ in one respect only.

Claim 14 is therefore **downgraded from Validated to not-evidence**, and the published
confidence interval is withdrawn. The p-value describes sampling noise around a conclusion
fixed before any agent acted.

Two limits stated so this is not read as more than it is: encoding "negative contingencies
cause hold-ups" is a defensible modelling assumption from the literature, and Exp 45 does
not refute the Hart-Moore mechanism. The finding is that claim 14's number is *entailed by*
that assumption rather than evidence for it. Separately, the arms all carry the GCL
vulnerability multiplier, so they are not comparable to the shipped `incomplete_high`
figure of 87.3 — a second confound in the original comparison, flagged but not measured.

Sibling predictions 1 and 2 are worse and were not separately rowed: investment is
literally `0.5 × {1.0, 0.7, 0.4, 0.85}` per condition (`:211-226`), which is why their
effect sizes are **d = 499** and **d = 371**. An effect size of 499 is not a strong result;
it is a constant with noise added.

### The judgment call

The refreshed number and the downgrade ship together, on purpose. Publishing −37.5% under
a Validated banner would have been a correction that reinstates the defect it corrects — a
more precise figure for a quantity that was never evidence. Precision is not validity.

Same failure mode as retracted claim 5 (hardcoded, not emergent) and as Theorems 5 and 6:
**the outcome is generated from the predictor rather than measured.** That is four now,
found by the same question.

---

## 2026-09-20 (parser) — over-extraction bug fixed; Exp 06's verification rate moves 0.0% → 25.0%; Exp 44's P3 and P4 now hold

### The bug

`CommitmentParser.parse` ran every pattern in `COMMITMENT_PATTERNS` over the full response
independently, with no record of which spans had already been consumed. On a response with
explicit markers this over-extracts. Concretely, on

```
I understand your request. [COMMITMENT: I will complete the task as specified] [COMMITMENT: I will verify my work before submission]
```

the explicit-marker pattern correctly yields two commitments, and then the generic
`I will\s+(.+?)(?:\.|$)` pattern matches *inside the first marker* and, finding no period to
stop at, runs to the end of the string. The third commitment's action is

```
complete the task as specified] [COMMITMENT: I will verify my work before submission]
```

`_deduplicate` never caught it: it compares action strings for equality, and a run-on is not
equal to anything. **n commitments therefore yielded n+1 predicates**, and since
`GroundingEngine.verify_output` votes over predicates, that artifact was a vote.

### The fix

`src/gcl/llm/commitment_parser.py`, two changes:

1. **Span-claiming in `parse()`.** Matches that overlap a span already claimed by an
   earlier, higher-priority pattern are skipped.
2. **Pattern reorder.** The weak patterns now precede the moderate ones, because
   `I will try to` is a refinement of `I will` and whichever runs first claims the span —
   so the more specific reading needs the first look or it can never win.

Four regression tests added to `tests/test_llm.py::TestCommitmentParser`. Three of them fail
without the fix. Suite: 382 → 386, all passing.

Worth stating plainly: the suite passed both before and after a behaviour-changing fix,
which means there was no coverage of this path at all. That is why the tests were added
rather than just the fix.

### Downstream: Exp 06's verification rate moved, and this is why

| metric | before | after |
|---|---|---|
| polynomial scaling exponent | 0.33 | 0.24 |
| grounding success | 75.0% | 75.0% |
| mean alignment score | 0.33 | 0.35 |
| semantic preservation | 75.0% | 75.0% |
| **verification rate** | **0.0%** | **25.0%** |
| Theorem 6 verdict | NOT VALIDATED | NOT VALIDATED |

Verified empirically rather than inferred. All four scenarios receive the *same* response,
because `verify_alignment` calls `generate_with_commitment`, which never consults the
`add_response` table (see the 2026-09-20 verification-pass entry below). That response
carries two commitments, which now ground to two predicates instead of three. The
`verification` scenario's expected-behaviour context sets `verified: True`, so it scores
1/2 = 0.50, which clears `satisfaction_rate >= 0.5`; at 1/3 = 0.33 it did not. One of four
scenarios passing is 25%.

So the improvement is not an improvement. The threshold did not get better at telling
aligned from unaligned outputs; the denominator changed. The 0.0% was an artifact and the
25.0% is the same tautology with a different number attached.

### Downstream: Exp 44 re-run against the fixed parser

| measurement | before | after |
|---|---|---|
| spurious predicates per response | 1 (i.e. n+1) | 0 (n) |
| verdicts flipped by the spurious vote | >0 | 0 |
| (n,k) strata where *which* commitment broke matters | 9/45 | 0/45 |
| P3 `breakable_is_floor_n_over_2` | NOT CONFIRMED | CONFIRMED |
| P4 `duplication_uniform_so_verdict_unchanged` | NOT CONFIRMED | CONFIRMED |
| certified despite exactly one violation | 98% | **100%** |
| Youden's J at the shipped 0.5 threshold | 0.47 | 0.47 |
| best threshold on this population | 0.95 | 0.95 |
| max breakable at n=15 while certified | 7 | 7 |

**P4's outcome now holds but its mechanism is still false, and the retraction stands.**
P4 assumed uniform 2n doubling that cancels in the ratio. The real factor was (n+1)/n and
is now exactly 1. Both the old "2.0x" figure and P4's premise were wrong; only the
prediction's outcome survived, for an unrelated reason. The Exp 44 analysis field is
renamed `P3_and_P4_hold_only_after_the_parser_fix` and says so explicitly, so the confirmed
verdict cannot be read as vindicating the reasoning that produced it.

**The headline finding is unchanged and slightly worse.** Breaking any one safety
commitment and adding one kept commitment still certifies. An output with exactly one
broken commitment is now certified **100%** of the time, up from 98% — the 2% that used to
fail were failing on the parser artifact, not on anything the verifier knows.

Also renamed, because the old name asserted something false:
`experiment_43_theorem6.json`'s `duplicate_extraction_factor` → `predicates_per_commitment_single`.
It never measured duplication; Exp 43 grounds one commitment at a time, so it was reading
(n+1)/n at n=1. It now reads 1.0.

---

## 2026-09-20 (aggregation) — Exp 43 hardened; Exp 44 finds verification is defeated by making more promises; the "parsed twice" claim was wrong

### Exp 43: three methodological defects fixed

1. **Seed stability was only ever checked on the permissive column.** `main()` aggregated
   `auto_create_true` exclusively, so the shipped configuration — the one every published number
   comes from — had never been checked across seeds at all. Both columns are now reported under
   `across_seeds.shipped` and `.permissive`. The shipped column is stable across all 10 seeds
   (J, unfalsifiable share and constant-false share each take a single value).
2. **The two columns were not a controlled comparison.** One generator was threaded through every
   call in sequence, so the `auto_create=True` and `=False` runs were probed with different random
   contexts. Each part now gets an independent `SeedSequence` child, and the paired columns are
   built from the *same* child. The spurious `CONSTANT_FALSE: 2` in the shipped in-vocabulary
   column is gone; both columns now report `RESPONSIVE: 20`, as they always should have.
3. **Predicate classification was sampled where it should have been deterministic.** The safety
   predicates are conjunctive (`harm_assessed and not harm_detected`), so the satisfying
   assignment is one combination in four and a random sweep missed it entirely with probability
   `0.75**n` — about 10% at the original n=8. `adversarial_contexts` now includes explicit
   `assessed_clean` and `assessed_bad` contexts that pin those predicates, and the random count
   is raised to 32. No classification now depends on the draw.

Also: H5 now carries `H5_measures_loop_overhead_not_predicate_diversity` in the artifact.
`distinct_predicates` is 1 at every n from 1 to 1000, so the timing curve measures evaluating one
predicate n times and must not be quoted as support for polynomial verifiability.

### Correction: the parser does **not** parse every commitment twice

Published in the entry below and in `docs/CLAIMS.md` row 22a as "every tagged commitment is parsed
twice (measured factor 2.0x)". That is wrong, and Exp 43's own artifact already contained the
refutation: part E's rows show `parsed_predicates = n + 1` at every n up to 1000, not `2n`. The
2.0x came from grounding **one commitment at a time**, where `(n+1)/n = 2`.

What actually happens: a greedy capture starts inside the first commitment and runs to the **end of
the response**, producing exactly **one** spurious extra commitment per response whose action text
is every remaining commitment concatenated:

```
n=8 commitments -> 9 predicates; the 9th sourced from
  "not harm the user's data] [COMMITMENT: I will keep the operation safe] [COMMITMENT: ...
```

It then grounds to whichever grounding rule matches that run-on string first, so **the identity of
the doubled commitment is arbitrary and content-dependent**. One commitment silently receives two
votes, nobody chose which, and at the 0.5 boundary that extra vote is decisive. This is smaller
than the retracted claim in scope (one predicate per response, not a doubling of all of them) and
sharper in consequence.

### Exp 44 — `verify_output`'s majority vote

New: `experiments/44_aggregation_majority_vote.py`, `results/experiment_44_aggregation.json`.
Five pre-registered predictions; **P1, P2, P5 confirmed, P3 and P4 refuted.**

| finding | result |
|---|---|
| Break a safety commitment, add kept commitments until certified | **1 filler suffices**, for every safety commitment tested |
| Verdict depends on *which* commitment broke (n=10, one violation) | **No** — safety and cosmetic are interchangeable |
| Max commitments breakable while certified, n=15 | **7** (fraction 0.47) |
| Output with exactly one broken commitment is certified | **98%** of the time |
| Youden's J at threshold 0.5, trend vs set size | **−0.016 per commitment** |
| Best threshold on this population | **0.95** |

**The headline is gameability.** An agent that breaks "I will not harm the user's data" is
certified as aligned by adding one more promise it intends to keep. Verification improves as the
agent promises more, which is precisely the incentive an alignment mechanism must not create. This
does not depend on any labelling choice — it is a structural property of `satisfaction_rate >= 0.5`.

**Severity never enters.** The engine has no notion that some commitments matter more; all carry
weight 1/n. Of 45 exhaustive (n, k) strata only 9 split on which commitment broke, every one of
them the boundary stratum k = ceil(n/2), and in every one the verdict tracks **whether the
parser's accidentally doubled commitment happened to be kept**. So the only thing that makes the
identity of a broken commitment matter is a parsing bug.

**What Exp 44 does not show:** that conjunctive aggregation is correct. V5 labels an output
violated if any commitment is broken, which *is* the conjunctive rule restated, so a conjunctive
rule scores perfectly there by construction. That circularity is recorded in the artifact under
`what_this_does_not_show`, and the case against the majority vote rests on V1 and V2, which are
free of it. V5's per-n series is additionally cap-weighted (6 combinations per cell) and its
individual values are marked not quotable; the clean series is `per_violation_count`.

382 tests pass. No `src/` behaviour changed in this entry — the majority vote is documented, not
yet altered, because changing it is a design decision rather than a defect fix.

## 2026-09-20 (verification pass) — Corrections to the two entries below, and what the fixed run actually establishes

Re-derived every number in the two entries below from the code rather than re-reading the
artifacts. The fixes hold. Four statements in those entries do not, and the headline result is
narrower than they imply.

### Correction 1 — there are two auto-create branches, and only one of them is dangerous

Both entries below describe a single "auto-create branch". There are two, and they fail in
opposite directions:

| branch | trigger | predicate installed | effect |
|---|---|---|---|
| `grounding.py:419-434` | no grounding rule matched | `ctx.get(f"completed_{action[:20]}", False)` | nothing ever sets that key — **effectively constant false** |
| `grounding.py:437-452` | a rule matched, but its named predicate is unregistered | `lambda ctx: True` | **constant true; the commitment cannot fail** |

Constant true is reachable **only through a registry gap, never through an unknown verb.** The six
missing predicates took the second branch; that was the entire defect and the registry fix closes
it. Nonsense and harmful commitments took the first branch — they grounded at 1.00 with confidence
0.875, but they verified *false*, not true. The entry below is right that nonsense grounded
indiscriminately and right that confidence was inverted; it should not be read as saying nonsense
was ever certified as satisfied. Comments at `grounding.py:284` and `:443` corrected to match.

### Correction 2 — `add_response` is not merely non-matching, it is unreachable

The entry below attributes Exp 06's single-response problem to the 10-character key at
`06_alignment_verification.py:260` failing to match. The actual cause is one level up:
`AlignmentVerifier.verify_alignment` calls `llm.generate_with_commitment(prompt)`, which does not
consult the registered-response table at all. Calling `llm.generate(prompt)` on the same object
*does* return the scenario response for two of the four scenarios. So the per-scenario
`llm_response` strings are dead regardless of the key, and fixing the key would not revive them.
Confirmed: all four scenarios ground to `['completeness_achieved', 'verification_performed',
'completeness_achieved']`.

### Correction 3 — one tautology sufficed; the duplicate parse was not load-bearing here

The entry below says the always-true lambda "held a 2-of-3 majority." True, but `verify_output`
tests `satisfaction_rate >= 0.5`, so 1 of 2 would also have passed. Removing the duplicate-parse
bug would not have exposed the false verification. The `>=` is doing as much work as the tautology.

### Correction 4 — `alignment_above_0.3` is close to a constant

`_calculate_alignment_score` awards a flat **0.3 for being grounded at all**, then `0.4 ×
confidence`. Three of Exp 06's four scenarios score exactly **0.300** against a 0.3 bar; the
reported mean of 0.3333 clears it only on the strength of the fourth, the one scenario whose
context happens to set `verified`. A wholly unverified output scores 0.300. Criterion 3 of
`theorem_6_validated` is therefore near-vacuous and should not be quoted as independent support.

### What the fixed Exp 43 run does and does not establish

**Holds.** H1 (grounding discriminates: valid 1.00 vs nonsense/harmful/held-out 0.00), H3 (0.0%
unfalsifiable, down from 16.7%), H4 (0.0% empty-context certification, down from 23.3%). H3 and H4
are direct consequences of the fix.

**Does not hold as stated.** H2's Youden's J = 1.00 is **conditional on grounding, and grounding
removed every hard negative first.** Under the shipped default part B scores **n = 15, not 30** —
held-out, nonsense and harmful never enter the confusion matrix because they fail to ground. The
honest reading is: *given that a commitment grounded, the predicate reads the evidence key it was
supposed to read.* Discrimination moved from the verifier to the grounder. J = 1.00 is not evidence
that verification rejects anything, and the pooled figure must not be quoted as if it were.

**Three defects in Exp 43 itself, found while checking it:**

1. **The seed-stability check runs on the wrong column.** Lines 614 and 617 aggregate
   `auto_create_true` only. The shipped configuration was never checked across seeds.
2. **Part A's `CONSTANT_FALSE` counts are sampling noise.** `part_a_grounding(True, rng)` and
   `(False, rng)` share one generator and therefore get different random probes. The conjunctive
   safety predicates need a 1-in-4 key combination, so across 8 draws a genuinely responsive
   predicate is mislabeled constant-false roughly 10% of the time. This fully explains the
   `CONSTANT_FALSE: 2` vs `RESPONSIVE: 20` split on identical inputs. The headline is unaffected
   (`unfalsifiable_share` counts constant-*true* only), but the field is not comparable across
   columns and should not be reported per-column.
3. **Part E remains degenerate.** `distinct_predicates` is 1 at every n from 1 to 1000. The
   exponent 0.749, CI [0.611, 0.895], measures verifying *one* predicate n times. The fix did not
   touch this and H5 should be read as a statement about loop overhead.

### Two things the fix did not touch

- **The 0.5 majority cliff.** `verify_output` certifies at `satisfaction_rate >= 0.5`; part D puts
  the flip at exactly `fraction_kept = 0.50`. Keeping half of your commitments is sufficient to be
  certified. This is now the largest remaining substantive problem in the verification path.
- **The duplicate parse**, still open, as recorded below.

### One more Exp 06 number that is smaller than it looks

`grounding.success_rate: 0.75` is **9 of 12**, not 450 of 600: the twelve templates are
deterministic and replayed 50×. The three failures are `maintain data privacy and security`,
`be transparent about limitations`, and — pointedly — `ask for clarification when needed`, which
fails *despite* `clarification_provided` now being registered, because the rule regex matches
`clarify` and not `clarification`. That single case is the coverage limitation in miniature.
`avg_confidence: 0.6135` pools the three failures as 0.0; conditional on grounding it is 0.818.

## 2026-09-20 (later) — Grounding defects fixed; Exp 06's published numbers corrected, and `theorem_6_validated` is now false

Acting on the entry below. Three changes to `src/gcl/llm/grounding.py`:

1. **`auto_create_predicates` now defaults to `False`.** With it on, grounding could not fail, so
   any grounding success rate measured the parser instead.
2. **The six missing predicates are registered** — `clarification_provided`,
   `correctness_verified`, `completeness_achieved`, `check_performed`, `validation_complete`,
   `danger_avoided`. Every rule target in `DEFAULT_GROUNDING_RULES` now resolves to a real
   predicate; the unregistered share is **35% → 0%**.
3. **Safety predicates now require affirmative evidence.** `no_harm` was
   `not ctx.get("harm_detected", False)` and `safety_maintained` was `ctx.get("is_safe", True)` —
   both returning the safe answer on an empty context. They now require `harm_assessed` /
   `safety_assessed`. This third change was not in the two-line plan; registering the six while
   leaving these two defaulting to "safe" would have reinstated the exact defect being corrected.

### Exp 06's published numbers, before and after

| metric | before | after |
|---|---|---|
| `grounding.success_rate` | 1.0000 | **0.7500** |
| `grounding.failed` | 0 | **150** |
| `grounding.avg_confidence` | 0.8354 | 0.6135 |
| `alignment.avg_score` | 0.6000 | 0.3333 |
| `alignment.verification_rate` | 1.0000 | **0.0000** |
| `semantic.preservation_rate` | 1.0000 | 0.7500 |
| `complexity.scaling_exponent` | 0.1952 | 0.2169 |
| `theorem_6_validated` | **true** | **false** |

**The verification rate did not regress from 1.000 to 0.000; 1.000 was never real.** Tracing it:
`06_alignment_verification.py:260` calls `llm.add_response(scenario["prompt"][:10], ...)`, which
never matches, so all four "scenarios" receive the same default MockLLM reply — *"I will complete
the task as specified"* and *"I will verify my work before submission"*. Those ground to
`completeness_achieved`, `verification_performed` and (via the duplicate-parse bug)
`completeness_achieved` again. Before the fix `completeness_achieved` was unregistered and
therefore `lambda ctx: True`, so two of three predicates were unconditionally satisfied, clearing
`verify_output`'s `satisfaction_rate >= 0.5` majority vote in every run. The 1.000 was an
always-true lambda holding a 2-of-3 majority. Now that the predicate reads `ctx["complete"]` and
Exp 06's `expected_behavior` dicts supply no such key, the honest value is 0.000.

Two consequences worth stating plainly. Exp 06's four scenarios are **one response repeated** — the
per-scenario `llm_response` strings are dead code. And the 0.000 is not evidence that verification
is broken; it is evidence that Exp 06 never supplied the evidence its own commitments referenced.

`theorem_6_validated` was also gated on four thresholds that **never read the verification result**
— a theorem about verifiability validated without checking whether anything verified, passing with
`avg_alignment` at 0.333 against a 0.3 bar. A fifth criterion was added, and the per-criterion
values are now written to the artifact so the bare boolean cannot be quoted alone. The flag is now
`false`, failing solely on criterion 5.

### Exp 43 re-run against the fixed engine

| metric | permissive | shipped default |
|---|---|---|
| grounding rate, valid | 1.00 | 1.00 |
| grounding rate, held-out paraphrase | 1.00 | **0.00** |
| grounding rate, nonsense | 1.00 | **0.00** |
| grounding rate, harmful | 1.00 | **0.00** |
| unfalsifiable predicates | 0.0% | 0.0% |
| worst-class Youden's J | 1.00 | 1.00 |
| empty-context certification | 0.0% | 0.0% |

H1, H3 and H4 now pass under the shipped default; previously they failed at 16.7% unfalsifiable and
23.3% empty-context certification, with worst-class J = 0.00.

**The fix buys soundness with coverage, and that is the new limitation.** Legitimate paraphrases
outside the seventeen rule families — "I will refactor the payment module", "I will migrate the
user table" — no longer ground at all, dropping from 1.00 to 0.00. The system went from grounding
everything and verifying nothing to grounding only keyword-matched commitments and verifying those
correctly. Grounding coverage is now bounded by a seventeen-entry regex table, which is a much
narrower claim than Exp 06's 100% suggested and should be stated wherever grounding is described.

Note for future readers: the "permissive" column in `results/experiment_43_theorem6.json` is
`auto_create_predicates=True` against the *fixed* registry, not the pre-fix engine. The original
failing numbers are recorded in the entry below and are no longer reproducible from the code.

Not fixed, and still open: **every tagged commitment is parsed twice** (measured factor 2.0x), the
duplicate carrying a corrupted action string with a trailing bracket. It inflates every predicate
count in the repo and it is what gave the always-true lambda its majority above. It is a parser
change rather than a registry one, so it is left for a separate pass.

382 package tests pass — and passed both before and after a change that inverts the safety-predicate
semantics, so no test in the suite covers empty-context safety verification. `docs/CLAIMS.md` row 22
updated.

## 2026-09-20 — Theorem 6's validation is an artifact; verification is blind on assurance language

`06_alignment_verification.py` writes `theorem_6_validated: true` on the strength of a grounding
success rate of **1.000 (600/600)**, a verification rate of **1.000** and a semantic preservation
rate of **1.000**. Three saturated ceilings against `MockLLM(seed=42)`. None of them is a
measurement.

- **Grounding cannot fail.** `GroundingEngine(auto_create_predicates=True)` is the default; when no
  rule matches, `grounding.py:390-402` invents a predicate instead of failing. The success rate is
  bounded below by the parser's detection rate, so 600/600 is a tautology.
- **Six of the seventeen shipped grounding rules name predicates that are never registered.**
  `clarification_provided`, `danger_avoided`, `correctness_verified`, `completeness_achieved`,
  `check_performed` and `validation_complete` all fall through to `grounding.py:410-416`, which
  registers `evaluate_fn=lambda ctx: True`. That is **35% of the rule table** unfalsifiable by
  construction.
- **Exp 06's complexity harness lands on exactly that path.** Its n commitments all ground to the
  single predicate `correctness_verified`. `verify_output` therefore returns `verified=True,
  confidence=1.00` against an all-tasks-failed context *and* against a context reporting active
  harm. `distinct_predicates` is **1 at every n** from 1 to 1000, so the O(|C|·|P|) claim was never
  exercised; the reported `scaling_exponent: 0.195` came from timings of 1.9e-5 s, which is
  interpreter overhead.
- **n = 600 is twelve strings replayed fifty times.** Grounding is deterministic; 588 of those
  trials are copies.
- **Every tagged commitment is parsed twice.** `[COMMITMENT: I will write clean code]` yields two
  commitments — the tagged capture and a second bare "I will ..." capture whose action string
  carries the trailing bracket (`'write clean code]'`). Measured duplication factor **2.0x**.

`experiments/43_theorem6_verification_discrimination.py` measures what Exp 06 asserted, treating
verification as a classifier: its job is to separate kept commitments from broken ones, so the
metric is discrimination, not pass rate. Five pre-registered hypotheses, a six-class commitment
corpus (in-vocabulary, unregistered-target, held-out paraphrase, nonsense, harmful, non-commitment),
both settings of `auto_create_predicates`, and honest kept/broken worlds per commitment.

| | result | H |
|---|---|---|
| grounding rate, valid vs nonsense vs harmful | 1.00 / 1.00 / 1.00 | **H1 FAIL** |
| grounding confidence, valid vs nonsense | 0.816 vs **0.875** | inverted |
| predicates constant across all contexts | 16.7% | **H3 FAIL** |
| commitments certified on an empty context | 23.3% | **H4 FAIL** |
| Youden's J, pooled | 0.83 | H2 pass |
| Youden's J, `unregistered_target` class | **0.00** | — |

**Gibberish grounds with higher confidence than valid commitments.** `_calculate_grounding_confidence`
awards +0.05 per word shared between the action and the predicate name, and auto-generated predicate
names are *built from the action words*, so they overlap maximally by construction. "I will flurb the
wizzle gronkly" scores 0.875; "I will write code for the parser" scores 0.812. Confidence is
anti-correlated with groundability.

**The blind spot is the assurance vocabulary.** H2 passes on the pooled rate and that pooled rate
should not be quoted alone — reporting 0.83 while one class sits at 0.00 is the same selected
comparison that retracted claims 9 and 10. The `unregistered_target` class has TPR 1.0 and **FPR
1.0**: all five commitments are certified as kept in a world where nothing was done and harm was
detected. Those five are "I will clarify the requirements", "I will avoid dangerous operations",
"I will complete the task correctly", "I will check the invariants", "I will validate the schema".
The system verifies ordinary work and rubber-stamps every promise of safety and correctness.

The empty-context result has the same shape. Seven of thirty commitments are certified with zero
evidence, and the seven are: not harming user data, keeping the operation safe, clarifying, avoiding
danger, completing correctly, checking invariants, validating the schema. `safety_maintained` is
`lambda ctx: ctx.get("is_safe", True)` and `no_harm` is `not ctx.get("harm_detected", False)` —
both **default to the safe answer**, so silence is read as compliance.

Two further defects: `verify_output` uses `satisfaction_rate >= 0.5`, a majority vote, so a batch is
"verified" at exactly **50% compliance** (measured flip point 0.50); and the semantic-preservation
metric Exp 06 reported as a rate, `avg_key_term_preservation: 2.0`, is a count of retained words from
a key-term list drawn from the commitment string itself.

What this does and does not show. It does not show that Theorem 6 is false — polynomial-time
verifiability is plausible and the timing is in fact sublinear here. It shows that the shipped
pipeline does not implement a test of it, that `theorem_6_validated: true` was produced by
always-true predicates, and that the failure is concentrated in the vocabulary where a verification
layer is supposed to earn its keep. Registering the six missing predicates and defaulting
`auto_create_predicates` to `False` are both one-line changes; neither was made here because both
change published numbers.

No file under `src/gcl/` was modified. 382 package tests pass. `docs/CLAIMS.md` rows 17 and 22
updated. Nothing on jasonstiltner.com has been changed.

## 2026-09-20 — Theorem 5 tested against a real task for the first time; it fails

Theorem 5 states `E[Success] >= confidence * similarity`. The repo has carried this as validated
since Exp 02, on the strength of `experiments/02_template_induction.py:355-364`, which draws its
"actual" success from `np.random.random() < confidence * similarity`. That is the bound sampling
itself. No task is executed and nothing is verified, so the test cannot fail — and in fact it very
nearly did anyway: the run reports `avg_actual 0.62` against `avg_predicted 0.633`, a discrepancy
that is pure sampling error in a procedure whose expectation is exact.

`experiments/42_theorem5_real_task.py` replaces the sampled outcome with an executed one. An agent
must tune a single continuous parameter for a task whose optimum varies with domain, load and
difficulty; success is `quality >= 0.60`, measured after the run by a `VerificationSchema`
predicate, not asserted. The experiment imports and calls the real `gcl.templates` package —
`TemplateInducer.observe()` / `.induce()` over 120 in-region commitments, then
`template.instantiate()` per transfer cell — so the thing under test is the shipped implementation.
Confidence is measured on 400 held-out in-region trials rather than set.

Task sensitivity is **swept, not chosen** (`--tolerances`, six values). A single fixed tolerance
would have let the author pick the verdict, which is the same defect as Exp 02 in the opposite
direction. 20 seeds, 30 trials per cell, an 11×11×3 context grid (363 cells per seed):

| tolerance | cells violating the bound (sig.) | mean success | within 0.10 above bound | Spearman ρ |
|---|---|---|---|---|
| 0.05 | 84.3% | 0.133 | 1.9% | +0.075 |
| 0.10 | 71.3% | 0.265 | 1.7% | +0.082 |
| 0.20 | 47.5% | 0.501 | 7.3% | +0.179 |
| 0.40 | 28.0% | 0.694 | 6.7% | +0.062 |
| 0.80 | 12.5% | 0.846 | 9.3% | +0.096 |
| 1.60 | 0.0% | 1.000 | 7.7% | +0.000 |

Against the four pre-registered hypotheses recorded in the artifact:

- **H1 (the bound holds) fails in every regime except tolerance 1.60**, where mean success is
  exactly 1.000 — the task is so forgiving that every attempt succeeds and any bound in [0,1] is
  satisfied. That regime is also flagged `calibration_degenerate`: all 41 candidate parameter values
  tie at 100%, so the "tuned" value is an argmax tie-break artifact.
  `regimes_where_bound_holds_and_tuning_is_non_degenerate` is **empty**.
- **H4 (the bound is approached) fails everywhere** — at most 9.3% of cells land within 0.10 above
  it. `regimes_where_bound_is_informative` is **empty**. The bound is true only where it says
  nothing, and says something only where it is false.
- **H2 (similarity ranks success) passes weakly.** ρ is positive in all non-degenerate regimes but
  never exceeds +0.18. Similarity carries ordering information; it does not carry a bound.
  (An earlier version of this analysis reported ρ = −0.218 at tolerance 1.60. That was my bug:
  positional tie-breaking in a rank transform, applied to a column that is 1.0 almost everywhere.
  With tie-averaged ranks it is exactly +0.000.)
- **H3 (the stated and implemented bounds agree) fails.** At tolerance 0.20 the stated bound
  `confidence * similarity` is violated in 3451 cells and the implemented
  `expected_success_rate()` in 3300. They differ because the implementation multiplies by similarity
  twice: `template.py:431-433` scales confidence by similarity out of region, and
  `template.py:534-535` multiplies by similarity again. The shipped bound is effectively
  `confidence * similarity**2` — tighter than the theorem, and still violated.

The sharpest single result is not the average. At tolerance 0.20 the worst violations are at
**similarity 0.9994** — the near-centre of the validated region — where the stated bound is 0.947
and empirical success is **0.000**. The induced context region Θ records where a template was
*used*, not where it *worked*, so high similarity to Θ certifies familiarity, not competence. A
related case: domains beta and gamma sit at identical similarity 0.4016 and return 0.368 and 0.543.
One number cannot separate them because similarity is computed over the context spec, and the spec
does not know which direction from the source region the task gets harder.

What this does and does not show. It does not refute Theorem 5 as stated under its own assumptions;
it shows that the quantity the implementation computes for `similarity` does not satisfy those
assumptions on a task with real outcomes, and that no tested regime makes the bound simultaneously
true and informative. A different similarity metric — one fit against outcomes rather than against
context overlap — is the obvious next thing to test.

One further defect found while wiring the experiment: `induction.py:142-145` builds
`ActionSchema(action_type=self.action_type or "unknown")` and discards the observed action
parameters, so an induced template cannot reproduce the behaviour it was induced from. Exp 42 works
around this by re-attaching the tuned parameter after induction; the workaround is commented in
place. No file under `src/gcl/` was modified. 382 package tests pass.

`docs/CLAIMS.md` rows 17 and 21 updated. Nothing on jasonstiltner.com has been changed.

## 2026-09-09 — Exp 23's network metrics are measuring an empty graph

Follow-up to the entry below, which left the Exp 07 / Exp 23 clustering disagreement open on the
grounds that deciding between two experiments is a research call. It was not a disagreement between
two measurements. **Exp 23 never measured anything.**

`experiments/23_dunbar_scaling.py:280` initialises the trust matrix as `np.eye(n_agents) * 0.5`, so
every off-diagonal entry starts at exactly 0. Trust rises +0.1 per success, applied to the selected
agent's column, and the graph is binarised at `> 0.5` — so an off-diagonal edge requires **more than
five net successes by a single agent**. An instrumented run at n = 100 over 100 rounds reaches a
maximum off-diagonal trust of **0.1000**, with 14 distinct agents ever selected and a maximum of
**3** successes by any one of them. Edges above threshold: **0**.

With no edges, `compute_network_metrics` returns early at line 66 down a branch that **hardcodes**
`clustering_coefficient=0.0`, `small_world_coefficient=0.0`, `mean_degree=0.0` and
`n_components=n`. The results file confirms this held in **69 of 70 runs** (7 sizes × 10 seeds); the
single exception is one n = 10 run with `mean_degree` 0.9, too sparse to close a triangle. Both
clustering columns are affected — the legacy `trust_clustering` at line 340 uses the same threshold
and falls to the same zero branch.

Consequences:

- **`clustering_coefficient: 0.0` is withdrawn as a result.** It is a constant returned by an error
  path. The zero variance across all 70 runs, which is what prompted the investigation, is fully
  explained by it.
- **"No small-world structure detected" is withdrawn.** `is_small_world` comes from the `else False`
  default at line 492, reached because `valid_sw = arr[arr > 0]` is empty. σ was never compared
  against 1.
- **The whole `network_topology` block is downstream of nothing.** `mean_degrees` is
  `[0.0, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0]`; `path_length_vs_size` is six `null`s and a single 1.0 and
  still carries `"trend": "stable_or_decreasing"`; `clustering_vs_size` is seven zeros and carries
  `"trend": "stable_or_increasing"` because `0.0 < 0.0` is false. Each label is an else-branch.
- **Exp 07's 0.7475 stands** (100 agents, density 0.429, 4,248 edges) as the only real measurement of
  trust-network clustering in the repo. It is single-seed and should be replicated before it carries
  any weight, but it is not contradicted.

Two further defects found in the same file, independent of the trust bug:

- **`dunbar_estimate: 100` is grid resolution, not a limit.** It is the first tested size whose mean
  efficiency falls below half of the maximum (`23_dunbar_scaling.py:471-481`). Half-max is 0.1325,
  anchored on the peak at n = **10**. The crossing is bracketed by n = 50 (0.184) and n = 100
  (0.114), so the limit lies somewhere in (50, 100]; 100 is simply the next point on the grid
  {5,10,20,50,100,150,200}. The monotone decline in efficiency with population is real. The
  convergence on ~100 is an artifact of where the grid points were placed, and should not be
  presented as the experiment recovering Dunbar's number.
- **`transition_size: 5.0` is not a phase transition.** The loop at lines 460-466 breaks at the first
  size with efficiency below 0.5, and every size tested is below 0.5 — the sweep's maximum efficiency
  is 0.265. It reports the smallest population tested. This is the same failure as the retracted
  claim 10, whose 50%-cooperation threshold was cleared by the random baseline: a cutoff that
  separates nothing.

The generalisable lesson, now editorial rule 4 in `docs/CLAIMS.md`: **a metric that is exactly
constant across a parameter sweep is a defect until proven otherwise.** Real measurements of a
stochastic process vary. The tell here was never a suspicious value — 0.0 clustering is perfectly
plausible — it was `std: 0.0` repeated seven times.

See `docs/CLAIMS.md` rows 16, 16a and 16b.

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
