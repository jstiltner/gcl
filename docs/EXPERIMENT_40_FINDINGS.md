# Experiment 40: Corrected Oracle and the Observability Phase Boundary

**Date**: August 2026
**Script**: `experiments/40_corrected_oracle.py`
**Results**: `results/experiment_40_corrected_oracle.json`
**Config**: 50 seeds/condition, 200 rounds, 30 agents; bootstrap 95% CIs; Cohen's d

---

## Why this experiment exists (correction of Experiment 39)

Experiment 39's "oracle with perfect information" was **not optimal**. It scored
candidates by closeness-of-fit (`score = 1/(1+excess)`), penalizing
over-qualified agents — but the success model
(`success_prob = capability × effort × (1 − difficulty/2)`) is monotonically
increasing in capability, so excess capability is free. The previously reported
"+81% over optimal external matching" and the "~75% information asymmetry"
decomposition were artifacts of the oracle optimizing the wrong objective.

Experiment 40 replaces it with a true success-maximizing oracle
(`argmax(observed capability)`) and adds an observability sweep.

---

## Part A: Confound demonstration (perfect information, both sides)

| Condition | Cooperation |
|---|---|
| Self-selection (fixed effort) | 0.531 |
| Self-selection (emergent effort) | **0.599** |
| Matching oracle — Exp 39's (fixed) | 0.305 |
| Argmax oracle — corrected (fixed) | 0.531 |
| Argmax oracle — corrected (emergent) | 0.534 |

| Comparison | Δ | 95% CI | d | Significant |
|---|---|---|---|---|
| SS vs matching oracle (fixed) | +0.227 | [+0.215, +0.239] | 7.22 | yes |
| **SS vs argmax oracle (fixed)** | **+0.000** | [−0.013, +0.013] | 0.00 | **no** |
| SS vs argmax oracle (emergent) | +0.065 | [+0.050, +0.080] | 1.68 | yes |
| Argmax vs matching oracle (fixed) | +0.227 | [+0.215, +0.239] | 7.22 | yes |

### Finding 1 — The "information asymmetry" advantage was an artifact

With effort fixed and perfect information on both sides, self-selection and the
corrected oracle are **statistically indistinguishable** (Δ = +0.000). The
entire +0.24 gap reported in Experiment 39 is reproduced as the gap between the
*corrected* and *flawed* oracles (+0.227). **RETRACTED**: "self-selection beats
optimal external matching by 81%"; "information asymmetry accounts for ~75% of
the advantage."

### Finding 2 — The emergent effort effect is real and survives correction

Self-selection with emergent effort beats the truly optimal oracle by
**+0.065 [+0.050, +0.080], d = 1.68** (a ~12% relative improvement). Assigned
agents do not develop the same commitment/ownership dynamics
(oracle emergent ≈ oracle fixed: +0.003). The corrected decomposition
**inverts** Experiment 39's conclusion:

| Component | Exp 39 (flawed oracle) | Exp 40 (corrected) |
|---|---|---|
| Information asymmetry | +0.240 (~75%) | **+0.000 (~0%)** |
| Emergent effort/motivation | +0.082 (~25%) | **+0.065 (~100%)** |

The honest claim: **choice creates commitment**. Volunteering triggers an
ownership/commitment feedback loop that assignment does not, and this — not
privileged self-knowledge — is the mechanism behind the self-selection
advantage in this model.

---

## Part B: The observability phase boundary

With fixed effort (information channel isolated), sweeping self-knowledge noise
(σ_self) × oracle observation noise (σ_oracle) over {0, 0.05, 0.1, 0.2, 0.3}:

- **Self-selection wins iff σ_oracle > σ_self** (advantage up to +0.127 at
  σ_oracle = 0.3, σ_self = 0; significant whenever the gap ≥ 0.1).
- **Centralized assignment wins iff σ_self > σ_oracle** (symmetrically, down to
  −0.127).
- Along the diagonal (equal noise), differences are indistinguishable from 0.

### Finding 3 — Neither mechanism dominates; relative observability decides

There is no intrinsic informational advantage to self-selection. The correct
design principle: **delegate task selection to whichever party has the less
noisy view of agent capability** — plus a bonus to self-selection from the
motivation channel (Part A) that shifts the practical boundary in its favor by
roughly one noise level.

---

## Implications

1. **For the GCL narrative**: The defensible claims are (a) the emergent
   motivation effect (+0.065, d = 1.68), and (b) the observability phase
   boundary. The "information asymmetry" framing must be dropped from all
   documents (see `docs/CLAIMS.md`).
2. **For real systems**: Whether self-selection helps becomes an empirical
   question about where a system sits relative to the boundary: do agents
   (human or LLM) assess their own task-success probability more accurately
   than an external observer can? This directly motivates Experiment 41.
3. **For research integrity**: This is the second self-correction in this
   program (34A → 39 → 40). Both prior errors were oracle/parameter constructions
   that smuggled the conclusion into the design. Future mechanism claims should
   pre-specify the optimality criterion of any baseline coordinator.

---

## Gate decision for Experiment 41 (LLM validation)

**Gate passes.** There is a real, asymmetry-dependent advantage region and a
validated motivation mechanism worth testing with real LLMs. The key LLM
questions:

1. Does an LLM's stated confidence predict its own task success better than an
   external model's assessment of it (which side of the phase boundary are real
   LLM systems on)?
2. Does "chosen vs assigned" framing change LLM output quality (does the
   motivation effect have an analogue in prompted agents)?

**Outcome (Aug 2026)**: Experiment 41 was run. Answer to (1): external
assessment beat self-confidence for all 4 LLM agents tested — real LLMs sit on
the central-assignment side of the phase boundary. Answer to (2): the powered
re-test (Experiment 41b, paired design, n = 120) found no significant framing
effect (+0.033, p = 0.52) — the simulated motivation effect is not detected
in prompted LLMs. See `docs/EXPERIMENT_41_FINDINGS.md`.
