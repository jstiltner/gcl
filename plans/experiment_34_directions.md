# Next Research Directions After Experiments 32-33

## What We've Learned

1. **Self-selection beats centralized matching** (Exp 32)
2. **Specialization doesn't emerge naturally** (Exp 33)
3. **Sharing helps cooperation but prevents specialization** (Exp 33b)
4. **Agents choose based on capability, not type** (Exp 33)

## The Open Questions

### Q1: WHY does self-selection work?

We know self-selection beats capability-matching, but we don't know WHY. Hypotheses:

1. **Information advantage**: Agents know their own capability perfectly
2. **Motivation**: Volunteers try harder than assigned agents
3. **Calibration**: Self-selection is self-correcting over time
4. **Risk management**: Agents avoid tasks they might fail

**Experiment 34A: Self-Selection Mechanism Decomposition**
- Test each hypothesis in isolation
- Vary: information quality, effort adjustment, learning rate, risk aversion

### Q2: What happens with TEAM tasks?

All experiments so far use single-agent tasks. Real coordination requires teams.

**Experiment 34B: Team Task Coordination**
- Tasks require 2-3 agents with complementary capabilities
- Test: Does self-selection still work? Does specialization emerge?
- Hypothesis: Team tasks may require specialization that single tasks don't

### Q3: What's the optimal sharing rate?

We tested 0% and 100% sharing. What about intermediate rates?

**Experiment 34C: Sharing-Specialization Frontier**
- Test sharing rates: 0%, 25%, 50%, 75%, 100%
- Measure: cooperation AND specialization
- Find: Pareto frontier of sharing vs specialization

### Q4: Does time horizon matter?

Current experiments run 100 rounds. Specialization may emerge slowly.

**Experiment 34D: Long-Horizon Dynamics**
- Run 500-1000 rounds
- Track: When does specialization emerge? Does it stabilize?
- Hypothesis: Specialization may emerge after ~300 rounds

### Q5: What about task scarcity?

Current tasks are equally distributed. What if some types are rare?

**Experiment 34E: Task Scarcity and Specialization**
- Make one task type rare (10% vs 45%/45%)
- Test: Do agents specialize in rare tasks?
- Hypothesis: Scarcity creates specialization pressure

---

## Prioritized Research Agenda

### Tier 1: Highest EV (Do Next)

**Experiment 34B: Team Task Coordination**

Why highest priority:
- Tests a fundamentally different task structure
- May reveal when specialization IS needed
- Directly relevant to real-world coordination
- Could overturn or confirm self-selection findings

Design:
```
Task types:
- Solo tasks (baseline)
- Pair tasks (2 agents, additive)
- Complementary tasks (2 agents, both must succeed)
- Team tasks (3 agents, weakest link)

Metrics:
- Cooperation rate
- Team formation patterns
- Specialization emergence
- Self-selection vs assignment
```

### Tier 2: High EV (Do Soon)

**Experiment 34A: Self-Selection Mechanism Decomposition**

Why high priority:
- Explains the core finding from Exp 32
- Identifies which mechanism to preserve/enhance
- Informs GCL design decisions

Design:
```
Factors:
- Information: perfect vs noisy capability knowledge
- Effort: fixed vs adjustable based on selection
- Learning: fast vs slow calibration
- Risk: risk-neutral vs risk-averse agents

2x2x2x2 = 16 conditions
```

### Tier 3: Medium EV (Do If Time)

**Experiment 34C: Sharing-Specialization Frontier**

Why medium priority:
- Refines existing findings
- May find optimal balance
- Less likely to overturn current understanding

**Experiment 34D: Long-Horizon Dynamics**

Why medium priority:
- Tests robustness of findings
- May reveal slow dynamics
- Computationally expensive

### Tier 4: Lower EV (Future Work)

**Experiment 34E: Task Scarcity**

Why lower priority:
- Specific environmental manipulation
- Less generalizable
- May not change core findings

---

## Recommended Next Experiment: 34B Team Tasks

### Rationale

The most important open question is: **Does self-selection work for team tasks?**

Current findings show self-selection works for single-agent tasks because agents know their own capability. But team tasks require:
- Knowing others' capabilities
- Coordinating complementary skills
- Forming effective teams

This is where specialization might actually matter.

### Design

```python
class TeamTask:
    task_type: TaskType
    required_roles: List[Role]  # e.g., [TECHNICAL, SOCIAL]
    combination_rule: str  # "additive", "complementary", "weakest_link"
    difficulty: float
    reward: float

class Role:
    name: str
    required_capability: float
    task_type_affinity: TaskType
```

### Conditions

1. **Solo baseline**: Single-agent tasks (from Exp 32-33)
2. **Additive pairs**: 2 agents, success = avg(capabilities)
3. **Complementary pairs**: 2 agents, both must succeed
4. **Weakest link teams**: 3 agents, success = min(capabilities)
5. **Best shot teams**: 3 agents, success = max(capabilities)

### Hypotheses

**H1: Self-selection degrades for team tasks**
- Solo: self-selection works
- Teams: need coordination beyond self-selection

**H2: Specialization emerges for complementary tasks**
- Additive: no specialization needed
- Complementary: specialization helps

**H3: Team formation becomes the bottleneck**
- Solo: task selection is key
- Teams: team formation is key

**H4: Visibility matters more for teams**
- Solo: visibility doesn't matter (Exp 32)
- Teams: need to know others' capabilities

### Expected Findings

If H1-H4 are supported:
- Self-selection is sufficient for solo tasks
- Team tasks require additional coordination mechanisms
- Specialization emerges when tasks require complementary skills
- GCL needs team formation protocols, not just commitment selection

If H1-H4 are NOT supported:
- Self-selection is robust even for teams
- Agents naturally form effective teams
- Specialization still doesn't emerge
- GCL's current design is sufficient

---

## Alternative: Mechanism Decomposition (34A)

If you prefer to understand WHY self-selection works before testing teams:

### Design

```python
class SelectionMechanism:
    information_quality: float  # 0 = noisy, 1 = perfect
    effort_adjustment: bool     # Can agents adjust effort?
    learning_rate: float        # How fast do agents calibrate?
    risk_aversion: float        # How much do agents avoid failure?
```

### Conditions

| Condition | Info | Effort | Learning | Risk |
|-----------|------|--------|----------|------|
| Baseline | 1.0 | Yes | 0.1 | 0.0 |
| Noisy info | 0.5 | Yes | 0.1 | 0.0 |
| Fixed effort | 1.0 | No | 0.1 | 0.0 |
| Slow learning | 1.0 | Yes | 0.01 | 0.0 |
| Risk averse | 1.0 | Yes | 0.1 | 0.5 |
| All degraded | 0.5 | No | 0.01 | 0.0 |

### Expected Findings

- **Information**: If noisy info hurts, self-selection relies on self-knowledge
- **Effort**: If fixed effort hurts, motivation matters
- **Learning**: If slow learning hurts, calibration matters
- **Risk**: If risk aversion helps, agents are avoiding failure

---

## My Recommendation

**Start with Experiment 34B (Team Tasks)** because:

1. It tests a fundamentally different scenario
2. It could reveal when specialization IS needed
3. It's directly relevant to real-world coordination
4. It builds on rather than just refines current findings

If team tasks show different dynamics, we'll have a richer understanding of when self-selection works and when it doesn't.

If team tasks show the same dynamics, we'll have strong evidence that self-selection is robust across task structures.

Either way, we learn something important.
