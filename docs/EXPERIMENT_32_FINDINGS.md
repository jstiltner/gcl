# Experiment 32 Findings: Task-Capability Matching

> **NOTICE (Aug 2026)**: The interpretation of this result was revised by
> Experiments 39 and 40. The "capability-matching" baseline here (like Exp 39's
> oracle) optimizes a fit/anti-overkill objective that is suboptimal under the
> success model. Against a truly optimal argmax oracle, the informational
> advantage of self-selection is zero; the surviving mechanism is emergent
> motivation. See `docs/EXPERIMENT_40_FINDINGS.md` and `docs/CLAIMS.md`.
> Preserved unedited below as part of the revision trail.

## Executive Summary

**The results overturn our hypotheses and reveal a fundamental insight about coordination.**

### Key Finding

**Self-selection beats capability-matching by 74%** (0.536 vs 0.308 cooperation rate)

This is the opposite of what we predicted. We expected that matching tasks to agents based on capability would improve cooperation. Instead, letting agents choose their own tasks dramatically outperforms centralized matching.

---

## Phase 1 Results: Visibility × Mechanism

### Cooperation Rate Matrix

|                | Random | Self-Select | Cap-Match | Diff-Weight |
|----------------|--------|-------------|-----------|-------------|
| **Private**    | 0.349  | 0.524       | 0.296     | 0.401       |
| **Public**     | 0.349  | 0.547       | 0.296     | 0.401       |
| **Rep-Proxied**| 0.354  | 0.524       | 0.344     | 0.401       |
| **Separate**   | 0.349  | 0.547       | 0.296     | 0.401       |

### Main Effects

**Mechanism Effects (dominant factor):**
- Self-select: +0.137
- Diff-weight: +0.002
- Random: -0.048
- Cap-match: -0.091

**Visibility Effects (minimal impact):**
- Rep-proxied: +0.007
- Public: -0.000
- Separate: -0.000
- Private: -0.006

### Hypothesis Evaluation

| Hypothesis | Prediction | Result | Status |
|------------|------------|--------|--------|
| H1: Public > Private | Public visibility improves cooperation | 0.398 vs 0.393 | ✓ SUPPORTED (barely) |
| H2: Cap-match > Self-select > Random | Centralized matching is best | 0.308 < 0.536 > 0.350 | ✗ REJECTED |
| H4: Rep-proxied ≈ Public | Reputation is good proxy | 0.406 ≈ 0.398 | ✓ SUPPORTED |
| H5: Diff-weighted is robust | Difficulty weighting helps | 0.401 > 0.308 | ✓ SUPPORTED |

---

## Phase 2 Results: Task Distribution

Using best mechanism (self-select) with rep-proxied visibility:

| Distribution | Cooperation | Capability | Utilization |
|--------------|-------------|------------|-------------|
| Fixed (0.5)  | 0.496       | 0.861      | 3.480       |
| Uniform      | 0.524       | 0.882      | 3.597       |
| Adaptive     | 0.438       | 0.871      | 3.065       |
| Bimodal      | 0.531       | 0.923      | 3.443       |

### H3: Adaptive Tasks Unlock Capability

**REJECTED.** Adaptive tasks actually *reduce* cooperation (0.438 vs 0.524).

This is counterintuitive. We expected that scaling task difficulty with capability would convert capability gains into cooperation gains. Instead, it makes things worse.

---

## Interpretation

### Why Self-Selection Wins

1. **Information Advantage**: Agents know their own capability perfectly. Centralized matching uses observed capability, which may be noisy or outdated.

2. **Motivation**: Agents who volunteer are more committed. Assigned agents may lack intrinsic motivation.

3. **Calibration**: Self-selection is self-calibrating. Agents learn which tasks they can handle through experience.

4. **Avoiding Mismatches**: Capability-matching tries to find "optimal" matches, but this can assign agents to tasks just barely within their capability, leading to high failure rates.

### Why Capability-Matching Fails

The capability-matching algorithm tries to assign tasks to agents whose capability is "just above" the task difficulty. This seems optimal but:

1. **Tight margins = high variance**: An agent with capability 0.55 on a 0.50 difficulty task has a narrow success margin.

2. **No slack**: Self-selection naturally builds in slack because agents only volunteer when they feel confident.

3. **Ignores effort**: Capability-matching ignores that agents adjust effort based on context. Self-selected agents may try harder.

### Why Visibility Doesn't Matter Much

Visibility effects are tiny (±0.007) compared to mechanism effects (±0.137). This suggests:

1. **Agents already know what matters**: Their own capability is the key information, and they always have that.

2. **Others' capability is less relevant**: For individual task success, knowing others' capabilities doesn't help much.

3. **Reputation is sufficient**: When others' information matters (for trust), reputation is a good enough proxy.

### Why Adaptive Tasks Hurt

Adaptive tasks scale difficulty with population capability. This seems like it should help, but:

1. **Removes easy wins**: When tasks get harder, even capable agents fail more often.

2. **Reduces learning opportunities**: Harder tasks mean fewer successes, which means fewer templates learned.

3. **Discourages volunteering**: If tasks are always hard, agents may be less willing to volunteer.

---

## Implications for GCL

### 1. Agent Autonomy is Crucial

GCL's core tenet is that agents choose their own commitments: π(s) → C. This experiment validates that design choice. Centralized assignment underperforms self-selection.

**Implication**: Don't try to "optimize" task assignment. Let agents choose.

### 2. Reputation as Proxy is Sufficient

We don't need separate capability tracking. Reputation-proxied visibility performs as well as public capability visibility.

**Implication**: GCL's reputation system is sufficient for coordination. No need for explicit capability signaling.

### 3. Difficulty-Weighted Reputation Works

Difficulty-weighted reputation (0.401) outperforms random (0.350) and capability-matching (0.308). It's robust across visibility conditions.

**Implication**: The difficulty-weighted reputation update from earlier experiments is validated.

### 4. Task Difficulty Should Be Varied, Not Adaptive

Bimodal tasks (0.531) slightly outperform uniform (0.524), and both beat adaptive (0.438).

**Implication**: Offer a mix of easy and hard tasks. Don't try to match difficulty to capability.

---

## Revised Understanding

### The Capability-Cooperation Gap (Exp 30-31)

We found that capability increases with sharing but cooperation doesn't. We hypothesized that task-capability matching was the missing mechanism.

### The New Finding (Exp 32)

Task-capability matching makes things *worse*. Self-selection is the key mechanism.

### Synthesis

The capability-cooperation gap exists because:

1. **Capability is necessary but not sufficient**: You need capability to succeed, but having more capability than needed doesn't help.

2. **Self-selection is the matching mechanism**: Agents naturally match themselves to appropriate tasks through volunteering.

3. **Centralized matching disrupts this**: Trying to "optimize" matching removes the self-calibration that makes self-selection work.

### The Real Insight

**Coordination emerges from agent autonomy, not from centralized optimization.**

This is a core GCL principle, and Experiment 32 provides empirical support.

---

## Next Steps

### 1. Investigate Self-Selection Dynamics

What makes self-selection work? Is it:
- Better information (agents know own capability)?
- Higher motivation (volunteers try harder)?
- Self-calibration (learning from experience)?

### 2. Test Hybrid Approaches

Can we combine self-selection with light guidance?
- Suggest tasks but let agents choose
- Provide capability feedback to improve self-assessment
- Reputation-based task recommendations

### 3. Explore Failure Modes

When does self-selection fail?
- When agents overestimate capability?
- When all agents want the same tasks?
- When tasks require coordination?

### 4. Multi-Agent Tasks

Current experiments use single-agent tasks. What happens with team tasks?
- Does self-selection still work?
- How do agents form teams?
- Does capability visibility matter more for team formation?

---

## Summary Table

| Finding | Implication |
|---------|-------------|
| Self-selection > Capability-matching | Let agents choose their own commitments |
| Visibility effects are minimal | Reputation is sufficient for coordination |
| Adaptive tasks hurt | Offer varied tasks, don't match to capability |
| Difficulty-weighting is robust | Keep difficulty-weighted reputation updates |
| Bimodal tasks work well | Mix of easy and hard tasks is optimal |

---

## Connection to Published Work

This finding strengthens the GCL framework's emphasis on agent autonomy. The published work argues that agents should choose their own commitments. Experiment 32 provides empirical evidence that this is not just philosophically appealing but practically superior.

The "Ubuntu effect" from earlier experiments can now be reinterpreted:
- Ubuntu's advantage wasn't just knowledge sharing
- It was also the collective identity that enabled better self-selection
- Agents in Ubuntu may have been more willing to volunteer for tasks that benefit the collective

This suggests a refined framing: **Ubuntu works because it aligns individual self-selection with collective benefit, not because it enables centralized coordination.**
