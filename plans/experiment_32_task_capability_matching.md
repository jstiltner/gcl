# Experiment 32: Task-Capability Matching Mechanisms

## The Problem

Experiments 30-31 revealed a critical gap:
- **Template sharing increases capability** (+86%)
- **But cooperation stays flat** (~55%)

This means **capability alone doesn't improve coordination**. The missing piece is **task-capability matching** - how do agents get assigned to tasks that match their capability?

## Key Questions

1. **Who knows capability?** (Information)
   - Self only (agent knows own capability)
   - Public (everyone knows everyone's capability)
   - Reputation-proxied (reputation ≈ capability)
   - Distinct signals (capability vs reputation are separate)

2. **How are tasks matched?** (Mechanism)
   - Random (no matching)
   - Self-selection (agents volunteer)
   - Centralized assignment (coordinator assigns)
   - Market (agents bid/compete)

3. **What's the task distribution?** (Environment)
   - Fixed difficulty (all tasks same)
   - Uniform (0.3-0.7)
   - Adaptive (difficulty scales with population capability)
   - Heterogeneous (mix of easy and hard)

## Factorial Design

### Factor 1: Capability Visibility
| Level | Description |
|-------|-------------|
| **Private** | Only agent knows own capability |
| **Public** | All agents know all capabilities |
| **Reputation-Proxied** | Reputation tracks capability (with noise) |
| **Separate Signals** | Capability and reputation are distinct, both visible |

### Factor 2: Matching Mechanism
| Level | Description |
|-------|-------------|
| **Random** | Tasks assigned randomly |
| **Self-Selection** | Agents volunteer based on assessment |
| **Capability-Matched** | Tasks assigned to best-fit agent |
| **Difficulty-Weighted** | Harder tasks → higher reputation reward |

### Factor 3: Task Distribution
| Level | Description |
|-------|-------------|
| **Fixed-Medium** | All tasks difficulty = 0.5 |
| **Uniform** | Difficulty ~ U(0.3, 0.7) |
| **Adaptive** | Difficulty scales with mean capability |
| **Bimodal** | 50% easy (0.2-0.4), 50% hard (0.6-0.8) |

## Full Combinatorics

4 × 4 × 4 = **64 conditions**

This is too many. Let's prioritize based on expected value.

## High-EV Subset

### Priority 1: Capability Visibility × Matching Mechanism (16 conditions)
Hold task distribution constant (Uniform). This tests the core question:
- Does knowing capability help?
- Does matching mechanism matter?

### Priority 2: Best Mechanism × Task Distribution (4 conditions)
Take the best mechanism from Priority 1, vary task distribution.

### Priority 3: Interaction Effects (8 conditions)
Test key interactions:
- Public capability + Capability-matched + Adaptive tasks
- Private capability + Self-selection + Bimodal tasks

## Hypotheses

### H1: Capability Visibility Matters
Public capability > Private capability for cooperation.
Rationale: Better information enables better matching.

### H2: Matching Mechanism Matters More Than Visibility
Capability-matched > Self-selection > Random, regardless of visibility.
Rationale: Mechanism determines how information is used.

### H3: Adaptive Tasks Unlock Capability Benefits
With adaptive tasks, high-capability agents face harder tasks, so capability gains translate to cooperation gains.

### H4: Reputation-Proxied ≈ Public Capability
If reputation tracks capability well, proxied visibility should perform similarly to public.

### H5: Separate Signals Enable Gaming
If capability and reputation are distinct, agents may game reputation while neglecting capability.

## Implementation Plan

### Phase 1: Core Factorial (Priority 1)
```
Visibility: [private, public, reputation_proxied, separate]
Mechanism: [random, self_selection, capability_matched, difficulty_weighted]
Tasks: uniform (fixed)
```
16 conditions × 10 seeds × 100 rounds = 16,000 runs

### Phase 2: Task Distribution (Priority 2)
```
Visibility: best from Phase 1
Mechanism: best from Phase 1
Tasks: [fixed_medium, uniform, adaptive, bimodal]
```
4 conditions × 10 seeds × 100 rounds = 4,000 runs

### Phase 3: Interaction Effects (Priority 3)
Selected combinations based on Phase 1-2 results.

## Expected Findings

### Prediction 1: Capability-Matched + Public Visibility Wins
When the system knows capabilities and matches tasks accordingly, cooperation should improve.

### Prediction 2: Self-Selection Underperforms
Agents may not accurately assess their own capability, leading to mismatches.

### Prediction 3: Adaptive Tasks Are Key
Without adaptive tasks, capability gains are wasted. With adaptive tasks, capability → cooperation.

### Prediction 4: Difficulty-Weighted Reputation Is Robust
Even without perfect capability information, difficulty-weighted reputation incentivizes appropriate task selection.

## Connection to GCL

This experiment directly tests the **commitment selection mechanism** in GCL:
- **π(s) → C**: How do agents choose what commitments to make?
- **Information**: What do agents know when choosing?
- **Incentives**: How does the structure reward good choices?

The finding will inform:
- Whether GCL needs explicit capability tracking (vs reputation-proxied)
- Whether self-selection is sufficient (vs centralized matching)
- How to design task distributions for optimal coordination

## Metrics

1. **Cooperation Rate**: Task success rate
2. **Capability Utilization**: (actual success) / (expected success given capability)
3. **Matching Quality**: Correlation between task difficulty and agent capability
4. **Efficiency**: Cooperation per unit of capability
5. **Fairness**: Gini coefficient of success distribution

## Files to Create

- `experiments/32_task_capability_matching.py`
- `experiments/social_structures/structures/matching.py`
- `results/experiment_32_matching.json`

## Timeline

1. Implement matching mechanisms (4 types)
2. Implement visibility levels (4 types)
3. Run Phase 1 (16 conditions)
4. Analyze and identify best combination
5. Run Phase 2 (4 conditions)
6. Run Phase 3 (selected interactions)
7. Synthesize findings

## Key Insight to Test

**The hypothesis**: Capability sharing (templates) is necessary but not sufficient. Task-capability matching is the missing mechanism that converts capability into cooperation.

If confirmed, this explains:
- Why Ubuntu "won" (it had implicit matching via collective identity)
- Why sharing alone doesn't help (capability without matching is wasted)
- What GCL systems need (both knowledge sharing AND task matching)
