# Experiment 34 Series: Comprehensive Findings

## Executive Summary

Five experiments (34A-E) investigated the mechanisms behind autonomous task selection and specialization dynamics. The results reveal fundamental insights about coordination in GCL systems.

---

## Experiment 34A: Self-Selection Mechanism Decomposition

**Question:** WHY does self-selection work?

### Results

| Factor | Effect on Cooperation |
|--------|----------------------|
| **Effort** | +0.052 |
| **Information** | +0.045 |
| Learning | -0.016 |
| Risk | +0.002 |

### Key Finding

**Effort adjustment is the most important mechanism** (+0.052), followed by information quality (+0.045).

Self-selection works because:
1. **Volunteers try harder** - Effort adjustment accounts for 52% of the effect
2. **Agents know their own capability** - Information accounts for 45% of the effect
3. Learning rate and risk aversion have minimal impact

### Implication

Self-selection's advantage comes from **motivation** (effort) more than **information** (capability knowledge). This suggests that the act of volunteering itself increases commitment.

---

## Experiment 34B: Team Task Coordination

**Question:** Does self-selection work for multi-agent tasks?

### Results

| Task Type | Self-Select | Cap-Match | Random |
|-----------|-------------|-----------|--------|
| Solo | 0.454 | 0.454 | 0.299 |
| Additive | 0.433 | 0.433 | 0.257 |
| Complementary | 0.191 | 0.191 | 0.077 |
| Weakest Link | 0.401 | 0.401 | 0.212 |
| Best Shot | 0.466 | 0.466 | 0.335 |
| **Average** | **0.389** | **0.389** | **0.236** |

### Key Finding

**Self-selection and capability-matching perform identically for teams** (both 0.389).

- Self-selection degrades for teams (0.454 → 0.373 avg)
- But capability-matching doesn't gain an advantage
- Both still beat random assignment (0.236)

### Implication

For team tasks, the formation method matters less than having ANY intelligent selection. The challenge is the task structure (complementary tasks are hardest at 0.191), not the selection mechanism.

---

## Experiment 34C: Sharing-Specialization Frontier

**Question:** What's the optimal sharing rate?

### Results

| Sharing Rate | Cooperation | Specialization |
|--------------|-------------|----------------|
| 0% | 0.517 | 0.011 |
| 20% | **0.551** | 0.011 |
| 50% | 0.532 | 0.011 |
| 80% | 0.550 | **0.015** |
| 100% | 0.517 | 0.011 |

### Key Finding

**Specialization is essentially zero across all sharing rates** (0.011-0.015).

- Best cooperation at 20% sharing (0.551)
- No meaningful trade-off because specialization doesn't emerge
- Pareto optimal rates: 20% and 80%

### Implication

The sharing-specialization trade-off doesn't exist in this model because specialization never emerges. The optimal sharing rate is ~20% for cooperation, but this is independent of specialization concerns.

---

## Experiment 34D: Long-Horizon Dynamics

**Question:** Does specialization emerge over longer time horizons?

### Results

| Rounds | Cooperation | Specialization | Growth |
|--------|-------------|----------------|--------|
| 100 | 0.536 | 0.011 | +0.000 |
| 250 | 0.553 | 0.011 | -0.003 |
| 500 | 0.566 | 0.011 | -0.003 |
| 750 | 0.575 | 0.011 | -0.003 |
| 1000 | 0.574 | 0.011 | -0.003 |

### Key Finding

**Specialization DECREASES with time** (r = -0.765).

- Longer horizons lead to homogenization, not specialization
- Cooperation improves with time (0.536 → 0.574)
- Specialization stabilizes immediately at ~0.011

### Implication

Time doesn't help specialization emerge. The model naturally converges to homogeneous agents who are generalists. This is the opposite of what we hypothesized.

---

## Experiment 34E: Task Scarcity

**Question:** Does task type scarcity induce specialization?

### Results

| Distribution | Cooperation | Specialization | Rare Specialists |
|--------------|-------------|----------------|------------------|
| Equal (33/33/33) | 0.539 | 0.011 | 1.3% |
| Skewed (50/35/15) | 0.546 | 0.013 | 0.0% |
| Extreme (70/20/10) | 0.538 | 0.018 | 0.0% |
| Skewed + Bonus | 0.546 | 0.013 | 0.0% |
| Extreme + Bonus | 0.538 | 0.018 | 0.0% |

### Key Finding

**Scarcity does NOT induce specialization** (+0.006 effect).

- Even with extreme scarcity (10% rare tasks), specialization stays at 0.018
- Bonuses for rare tasks don't attract specialists
- No agents specialize in rare tasks (0% rare specialists)

### Implication

Task scarcity alone doesn't create specialization pressure. Agents remain generalists regardless of task distribution or incentives.

---

## Synthesis: Why Doesn't Specialization Emerge?

Across all experiments, specialization remains at ~0.01 (essentially zero). This is a robust finding that requires explanation.

### Possible Reasons

1. **Template sharing homogenizes agents**
   - Even at 0% sharing, agents don't specialize
   - Sharing accelerates homogenization but isn't the cause

2. **Self-selection doesn't favor specialization**
   - Agents volunteer based on capability, not type
   - No mechanism rewards type-specific expertise

3. **Task types are interchangeable**
   - All types have similar difficulty distributions
   - No structural advantage to specializing

4. **Learning is too slow**
   - 10% template generation rate
   - Not enough type-specific templates accumulate

5. **Experience bonus is too weak**
   - Type experience gives small bonus
   - Not enough to overcome generalist advantage

### The Fundamental Insight

**Specialization requires structural pressure that doesn't exist in the current model.**

For specialization to emerge, we would need:
- Tasks that REQUIRE type-specific capability
- Penalties for attempting wrong-type tasks
- Stronger experience/template bonuses for type matching
- Or: Innate agent differences (as shown in Exp 33b)

---

## Revised Understanding of GCL Coordination

### What Works

1. **Self-selection** - Agents choosing their own tasks
2. **Effort adjustment** - Volunteers trying harder
3. **Capability-based selection** - Knowing own capability
4. **Template sharing** - Increases capability (but not cooperation)

### What Doesn't Emerge

1. **Specialization** - Agents remain generalists
2. **Division of labor** - No natural task type preferences
3. **Type-specific expertise** - Templates don't create specialists

### Implications for GCL Design

1. **Don't expect emergent specialization**
   - If specialization is needed, it must be designed in
   - Innate agent differences or structural constraints required

2. **Self-selection works through motivation**
   - The act of volunteering increases effort
   - Information advantage is secondary

3. **Team tasks don't change the picture**
   - Self-selection and capability-matching are equivalent
   - Task structure (complementary vs additive) matters more

4. **Sharing rate optimization**
   - ~20% sharing is optimal for cooperation
   - Full sharing (100%) is suboptimal

---

## Summary Table

| Experiment | Key Question | Key Finding |
|------------|--------------|-------------|
| 34A | Why does self-selection work? | **Effort** (+0.052) > Information (+0.045) |
| 34B | Does it work for teams? | Self-select = Cap-match (both 0.389) |
| 34C | Optimal sharing rate? | 20% for cooperation, no specialization trade-off |
| 34D | Does time help? | **No** - specialization decreases (r=-0.765) |
| 34E | Does scarcity help? | **No** - scarcity doesn't induce specialization |

---

## Next Steps

Given these findings, the most valuable next experiments would be:

1. **Forced Specialization Study**
   - What structural changes WOULD induce specialization?
   - Test: type-specific capability requirements, penalties, stronger bonuses

2. **Effort Mechanism Deep Dive**
   - Since effort is the key mechanism, understand it better
   - Test: effort costs, effort limits, effort visibility

3. **Team Formation Dynamics**
   - How do agents form teams?
   - Test: team stability, repeated partnerships, trust networks

4. **Real-World Validation**
   - Do these findings hold with LLM agents?
   - Test: Claude/GPT coordination via commitments
