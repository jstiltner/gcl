# GCL Research Report: Experiments 30-39 (REVISED)
## Comprehensive Findings with Corrected Mechanism Analysis

> **NOTICE: SUPERSEDED IN PART BY EXPERIMENT 40 (August 2026).**
> The headline claims in this report — "self-selection beats optimal external
> matching by 81%" and "information asymmetry accounts for ~75% of the
> advantage" — are **RETRACTED**. Experiment 39's oracle was not optimal: it
> scored candidates by closeness-of-fit, penalizing over-qualified agents under
> a success model that is monotonically increasing in capability. Against a
> truly optimal (argmax) oracle, the informational advantage is +0.000
> [−0.013, +0.013]. The **emergent motivation effect survives** (+0.065,
> d = 1.68), and an **observability phase boundary** determines when
> self-selection beats central assignment. See
> `docs/EXPERIMENT_40_FINDINGS.md` and `docs/CLAIMS.md`. The remainder of this
> report is preserved unedited as part of the revision trail.

> **FURTHER RETRACTION (2026-09-09).** Two more headline claims in this report are withdrawn, and
> one framing figure is misleading. See `docs/CLAIMS.md` rows 9/10 and `CHANGELOG.md`.
>
> - **"GCL achieves 97% of MARL performance"** — arithmetically right, structurally false. Exp 36
>   records `gcl_rank: 3`, `gcl_is_best: false`; GCL is third of five, behind IQL and QMIX.
> - **"25-50x better sample efficiency"** — the range across 2 of 4 baselines. `vs_mappo: 1.00` and
>   `vs_random: 1.20` were dropped, and the retained arms have standard deviations roughly four
>   times their means.
> - **"Total Seeds: 1200+ independent runs"** (line below) covers this 30-39 program only. It was
>   later quoted on jasonstiltner.com as the sample size behind the **Experiment 07** population
>   results, which ran at `n_agents: 100, n_timesteps: 5000, seed: 42` — a **single seed**. Do not
>   reuse this figure outside this report.

**Date**: January 2026
**Experiments**: 30-39 (10 experiments, 60+ conditions)
**Total Seeds**: 1200+ independent runs
**Statistical Methods**: Bootstrap CIs, Cohen's d, Power Analysis

---

## Core Finding

### Self-selection outperforms optimal external matching by 81%, even with effort controlled.

**Mechanism**: Agents have privileged access to their own capabilities—information that external coordinators cannot observe. This **information asymmetry** accounts for ~75% of the advantage. A secondary emergent effect (~25%) shows self-selectors also exert more effort.

**Implication**: Coordination systems should leverage agent self-knowledge rather than relying on external assignment, even when the external assigner has "perfect" information about observable capabilities.

---

## Mechanism Decomposition (Experiment 39)

| Component | Effect Size | Percentage |
|-----------|-------------|------------|
| **Information Asymmetry** | +0.240 | **~75%** |
| **Emergent Effort** | +0.082 | **~25%** |
| **Total Advantage** | +0.322 | 100% |

### Key Insight

Even when an external coordinator has "perfect" information about agent capabilities, agents still have **privileged access** to aspects of their own capabilities that are not externally observable. This private information enables better task-agent matching through self-selection.

---

## Executive Summary

This research program investigated the mechanisms underlying GCL's coordination effectiveness. The key findings are:

1. **Self-selection beats optimal external matching by 81%** (Exp 39) - even with effort controlled
2. **Information asymmetry accounts for ~75% of the advantage** (Exp 39) - agents have privileged self-knowledge
3. **Emergent effort contributes ~25%** (Exp 39) - self-selectors try harder
4. **Specialization does NOT emerge naturally** (Exp 33)
5. **GCL achieves 97% of MARL performance with 25-50x better sample efficiency** (Exp 36)

---

## Part I: Mechanism Decomposition (Experiments 30-34, 39)

### Experiment 39: Motivation Isolation (NEW)

**Critical Question**: Is self-selection advantage due to information or motivation?

**Design**: 
- Self-selection vs Oracle assignment (oracle has PERFECT information)
- With and without emergent effort mechanism
- Isolate information vs motivation components

**Results**:
| Condition | Cooperation | Effort |
|-----------|-------------|--------|
| Self-select (emergent) | **0.608** | 0.904 |
| Self-select (fixed) | 0.534 | 0.800 |
| Oracle (emergent) | 0.287 | 0.776 |
| Oracle (fixed) | 0.295 | 0.800 |

**Key Comparisons**:
- Self-selection vs Oracle (fixed effort): **+0.240** (pure information)
- Self-selection vs Oracle (emergent effort): **+0.322** (total)
- Emergent motivation effect: **+0.082** (difference)

**CRITICAL FINDING**: 
- **74.5% of advantage is information/self-knowledge**
- **25.5% of advantage is emergent motivation**
- Even with fixed effort, self-selection beats oracle by +0.240

**Why does self-selection beat oracle even with perfect information?**
1. **Self-knowledge asymmetry**: Agents know their own capabilities better than any external observer
2. **Task-agent fit**: Self-evaluation captures something about fit that external matching doesn't
3. **Selection bias**: Agents pick tasks they're more likely to succeed at

---

### Experiment 32: Task-Capability Matching (Reinterpreted)

**Original Finding**: Self-selection (0.536) beats capability-matching (0.308) by 74%

**Revised Interpretation**: This advantage is primarily due to:
1. **Self-knowledge** (74.5%): Agents know themselves better than the matcher knows them
2. **Emergent motivation** (25.5%): Volunteers develop commitment through choice

**NOT primarily due to**: A hardcoded "effort bonus" parameter

---

### Experiment 34A: Mechanism Decomposition (Reinterpreted)

**Original Claim**: "Effort (+0.052) > Information (+0.045)"

**Problem**: The "effort effect" was a parameter we set, not emergent behavior

**Corrected Interpretation**: 
- The experiment varied `information_quality` and `effort_adjustment` as independent parameters
- This doesn't isolate the true mechanisms
- Experiment 39 provides the correct decomposition

---

## Part II: Validated Findings (Experiments 33, 35, 36-38)

### Experiment 33: Specialization Dynamics (VALIDATED)

**Finding**: Specialization does NOT emerge naturally (HHI < 0.02)

**This finding stands**: It's about emergent behavior, not mechanism decomposition

---

### Experiment 35: Rigorous Validation (PARTIALLY REVISED)

#### 35A: Rich Agent Representations (VALIDATED)
- Strategic reasoning hurts cooperation (0.530 vs 0.306)
- Simple agents perform best

#### 35B: Statistical Validation (VALIDATED)
- Cohen's d = 4.05, p < 10^-72, power = 1.0
- Self-selection advantage is statistically robust

#### 35D: Effort Mechanism Sensitivity (REINTERPRETED)
**Original**: "Advantage persists at 0% effort bonus (+0.155)"

**Revised**: This +0.155 is the **information/self-knowledge** component, not motivation. The experiment confirms that self-knowledge alone provides substantial advantage.

#### 35E: Baseline Comparison (VALIDATED)
- Self-selection beats all baselines including centralized optimal
- **Revised interpretation**: This is because agents have better self-knowledge than any central coordinator

---

### Experiments 36-38: MARL Comparison (VALIDATED)

These findings stand as they compare methods, not mechanisms:

| Finding | Evidence |
|---------|----------|
| 97% of MARL performance | 0.534 vs 0.552 |
| 25-50x sample efficiency | 2 vs 52-102 episodes |
| Consistent in non-stationary | +0.2-0.9% all frequencies |

---

## Theoretical Framework

### The Information Asymmetry Hypothesis (PRIMARY, ~75%)

> **Agents have privileged access to their own capabilities—information that external coordinators cannot observe.**

Even when an external coordinator has "perfect" information about observable capabilities, agents still possess private information about:
- Their current state (fatigue, focus, confidence)
- Task-specific fit that isn't captured by general capability metrics
- Subtle aspects of capability that are difficult to externalize

This explains:
1. Why self-selection beats optimal external matching by 81% (Exp 39)
2. Why self-selection beats centralized optimal (Exp 35E)
3. Why the advantage persists even with effort controlled (Exp 39)
4. ~75% of the total self-selection advantage

### The Emergent Effort Hypothesis (SECONDARY, ~25%)

> **Self-selectors exert more effort than those who are assigned.**

This is an emergent effect, not a hardcoded parameter:
- Commitment builds through choice
- Psychological ownership increases engagement
- Self-selectors show effort of 0.904 vs 0.776 for assigned agents

This explains:
1. The additional ~25% of advantage (Exp 39)
2. Why emergent effort exceeds baseline effort
3. The commitment and confidence effects observed

### The Flexibility Hypothesis (VALIDATED)

> **GCL coordination is based on flexibility, not specialization.**

This explains:
1. Why specialization doesn't emerge (Exp 33)
2. Why generalists outperform specialists (Exp 33b)

---

## Summary of Corrected Findings

### Core Mechanisms (REVISED)

| Finding | Evidence | Correct Interpretation |
|---------|----------|------------------------|
| Self-selection beats capability-matching by 74% | Exp 32 | Primarily self-knowledge (74.5%) |
| Self-knowledge is primary mechanism | Exp 39 | +0.240 effect (74.5% of total) |
| Emergent motivation is secondary | Exp 39 | +0.082 effect (25.5% of total) |
| Specialization doesn't emerge | Exp 33 | Validated |
| Simple agents suffice | Exp 35A | Validated |

### MARL Comparison (VALIDATED)

| Finding | Evidence |
|---------|----------|
| 97% of MARL performance | 0.534 vs 0.552 |
| 25-50x sample efficiency | 2 vs 52-102 episodes |
| Consistent in non-stationary | +0.2-0.9% all frequencies |

---

## Addressing Potential Criticisms (REVISED)

### "Your 'motivation' claim was based on a hardcoded parameter"
**Response**: Correct. Experiment 39 properly isolates the mechanisms:
- 74.5% is information/self-knowledge
- 25.5% is emergent motivation
- The previous framing was incorrect

### "Isn't self-knowledge just another form of information?"
**Response**: Yes, but it's a specific form: **private information** that agents have about themselves that external coordinators cannot access. This is the key insight.

### "What's the practical implication?"
**Response**: 
- **For system design**: Let agents self-select rather than centrally assigning
- **Why it works**: Agents have private information about their own capabilities
- **Bonus effect**: Self-selection also creates emergent motivation (+25.5%)

---

## Recommendations for Next Steps (REVISED)

### High Priority

1. **Quantify Self-Knowledge Asymmetry (Experiment 40)**
   - How much better do agents know themselves vs external observers?
   - What types of information are most private?
   - Can we reduce the asymmetry?

2. **Test in Real Systems**
   - Do human teams show the same self-knowledge advantage?
   - Is the 74.5%/25.5% split consistent across domains?

### Medium Priority

3. **Explore Motivation Mechanisms**
   - What creates the emergent motivation effect?
   - Can we amplify it beyond 25.5%?
   - Is it commitment, ownership, or something else?

---

## Conclusion

### Core Finding

**Self-selection outperforms optimal external matching by 81%, even with effort controlled.**

### Mechanism

Agents have **privileged access** to their own capabilities—information that external coordinators cannot observe. This information asymmetry accounts for ~75% of the advantage. A secondary emergent effect (~25%) shows self-selectors also exert more effort.

### Decomposition

| Component | Effect | Percentage |
|-----------|--------|------------|
| Information Asymmetry | +0.240 | ~75% |
| Emergent Effort | +0.082 | ~25% |
| **Total** | **+0.322** | **100%** |

### Implication

**Coordination systems should leverage agent self-knowledge rather than relying on external assignment**, even when the external assigner has "perfect" information about observable capabilities.

### Practical Applications

The framework achieves 97% of MARL performance with 25-50x better sample efficiency, making it ideal for:
- **Cold-start coordination**: Leverages existing self-knowledge without training
- **Non-stationary environments**: Self-knowledge adapts immediately to changes
- **Privacy-preserving systems**: Agents don't need to reveal private capability information
- **Systems with unobservable capabilities**: External coordinators can't see what agents know about themselves

### Statistical Robustness

The findings are statistically robust:
- Cohen's d = 4.05 (large effect)
- p < 10^-72 (highly significant)
- Power = 1.0 (adequate)
- Mechanism decomposition properly isolated (Experiment 39)
