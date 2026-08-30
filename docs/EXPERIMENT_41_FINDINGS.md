# Experiment 41: LLM Self-Knowledge and Task Selection — Findings

**Date**: August 2026
**Script**: `experiments/41_llm_self_selection.py`
**Results**: `results/experiment_41_llm_self_selection.json`
**Models**: claude-sonnet-4-6, gpt-4o (SDK-cached, temperature 0.0)
**Design**: 60 deterministically verifiable tasks (multi-digit multiplication,
letter counting, modular-arithmetic chains) × 4 heterogeneous agents
({Claude, GPT} × {reasoning allowed, direct-answer-only}).

## Purpose

Experiment 40 established (in simulation) that self-selection beats central
assignment **iff agents know themselves better than the assigner knows them**
(σ_self < σ_oracle), and that the surviving mechanism from the retracted
"information asymmetry" claim is emergent motivation. Experiment 41 asks the
empirical question the simulation cannot answer: **which side of that phase
boundary do real LLMs sit on?**

Three parts:
1. **Calibration probe**: Does an LLM's stated confidence predict its own
   success better than an external LLM assessor's prediction? (Brier scores;
   lower = better.)
2. **Selection mechanisms**: Self-selection vs external assignment vs random
   on the same task bank.
3. **Motivation framing**: "You volunteered for this task" vs "You have been
   assigned this task" — does the simulated motivation effect appear in LLMs?

## Results

### Part 1: External assessment beats self-assessment for every agent

| Agent | n | Accuracy | Brier (self) | Brier (external) | Better predictor |
|---|---|---|---|---|---|
| claude_reasoning | 33 | 1.000 | 0.027 | **0.010** | external |
| claude_direct | 41 | 0.098 | 0.689 | **0.394** | external |
| gpt_reasoning | 60 | 0.600 | 0.361 | **0.333** | external |
| gpt_direct | 60 | 0.433 | 0.484 | **0.281** | external |

(n < 60 for Claude agents reflects responses that did not parse to the
required `ANSWER:` format and were excluded from calibration scoring.)

**Finding**: An external LLM assessor predicted each agent's success better
than the agent's own stated confidence, for all four agents. The direct-answer
agents were badly overconfident about arithmetic they cannot do without
scratch space (claude_direct: 9.8% accuracy against high self-confidence,
Brier 0.689).

**Interpretation**: Current LLMs sit on the **central-assignment side** of
Experiment 40's phase boundary: σ_self > σ_oracle. Stated self-confidence is
not privileged information — it is noisier than an outside observer's
assessment. This is the opposite of the (retracted) information-asymmetry
story, and consistent with the LLM calibration literature.

### Part 2: Structured selection beats random; self ≈ external

| Mechanism | Success | Picks |
|---|---|---|
| External assignment | **0.717** | gpt_reasoning 54, gpt_direct 6 |
| Self-selection | 0.700 | gpt_reasoning 39, claude_reasoning 14, others 7 |
| Random | 0.517 | uniform |

**Finding**: Both structured mechanisms beat random by ~+0.19; external
assignment edges out self-selection by +0.017 (not meaningful at n=60).
Notably, self-selection routed most tasks to gpt_reasoning on the strength of
stated confidence, even though claude_reasoning had the highest measured
accuracy (1.000 in Part 1) — confidence-based volunteering rewards the most
*overconfident* competent agent, not the most competent one. This is exactly
the failure mode predicted for the σ_self > σ_oracle regime.

### Part 3: No motivation framing effect (ceiling-limited)

| Framing | Success | n |
|---|---|---|
| "You volunteered" | 0.967 | 30 |
| "You have been assigned" | 1.000 | 30 |

**Finding**: No positive framing effect (−0.033). Caveat: the probe used
claude_reasoning, which is at ceiling on this task bank (Part 1 accuracy
1.000), so the test has essentially no power. The simulation's emergent
motivation effect (+0.065, d = 1.68, Exp 40) is **neither replicated nor
refuted** here; a harder task bank would be needed.

### Part 3 re-test: Experiment 41b (powered, paired design)

**Script**: `experiments/41b_motivation_framing.py`
**Results**: `results/experiment_41b_motivation_framing.json`
**Design**: 120 harder tasks (5x5 and 6x6-digit multiplication, nested modular
chains, long letter-counting), each attempted under three framings
(chosen / neutral / assigned) by two reasoning agents — a paired design
scored with the exact McNemar test.

| Agent | chosen | neutral | assigned | chosen − assigned | McNemar p |
|---|---|---|---|---|---|
| claude_reasoning | 0.967 | 0.992 | 0.967 | +0.000 | 1.000 (ceiling again) |
| gpt_reasoning | 0.267 | 0.250 | 0.233 | +0.033 | 0.523 |
| **Pooled** | 0.617 | 0.621 | 0.600 | +0.017 | 0.557 |

**Finding**: No significant framing effect. claude-sonnet-4-6 remained at
ceiling even on 6-digit multiplication, so only the GPT arm is informative:
it landed at ideal mid-range accuracy (0.23–0.27) and shows a small,
directionally positive, non-significant effect (+0.033, discordant pairs
13/9, p = 0.52). The 95% CI includes both zero and the simulated effect size
(+0.065). **Verdict: the emergent motivation effect found in simulation
(Exp 40) is not detected in prompted LLMs at this power; it is bounded above
at roughly +0.10 for this setting.** Choice framing should not be relied on
as a mechanism in LLM-agent systems pending stronger evidence.

## Conclusions

1. **The phase-boundary framework transfers**: Exp 40 predicts self-selection
   only wins when self-knowledge is better than external observation. For
   current LLMs on verifiable tasks it is not — external assessment is
   better calibrated for all four agents tested — so central assignment
   should (and does, weakly) match or beat self-selection.
2. **Honest negative result for naive LLM self-selection**: letting LLM
   agents volunteer by stated confidence selects for overconfidence. GCL-style
   systems using LLM agents should ground selection in verified track records
   (commitments settled), not self-reports — which is precisely what the GCL
   reputation mechanism does.
3. **Motivation effect not detected in LLMs**: the one simulated mechanism
   that survived Exp 40 did not replicate in the powered 41b re-test
   (+0.033, p = 0.52, n = 120 paired). The simulation-to-LLM transfer of the
   motivation channel is unsupported; treat choice framing as a null-to-small
   effect for prompted agents.

## Limitations

- Single run per condition (deterministic decoding + cache); n=60 tasks.
- Narrow task family (arithmetic/counting); results may differ on open-ended
  work where self-knowledge of style/strength matters more.
- Part 3 ceiling effect (see above).
- Two model families only.

## Status of claims

- "LLMs have privileged self-knowledge exploitable by self-selection":
  **NOT SUPPORTED** (Part 1, Part 2).
- "Structured selection (any) beats random assignment": **SUPPORTED** (+0.19).
- "Choice framing increases LLM success": **NOT DETECTED** (41b: +0.033,
  p = 0.52, powered paired design; effect bounded above ≈ +0.10).

See `docs/CLAIMS.md` for the canonical claims table.
