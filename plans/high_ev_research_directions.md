# High-EV Research Directions After Experiments 30-31

## The Core Finding

**Capability ≠ Cooperation**

Template sharing increases capability (+86%) but cooperation stays flat (~55%). This is a fundamental insight that opens several research directions.

---

## Direction 1: Task-Capability Matching (Experiment 32)

**Question**: How do we convert capability into cooperation?

**Approach**: Factorial design testing visibility × matching mechanism × task distribution

**EV Assessment**:
- High theoretical value (explains the gap)
- High practical value (actionable for system design)
- Medium implementation effort (16-64 conditions)
- **Overall EV: HIGH**

---

## Direction 2: Dynamic Task Difficulty

**Question**: What if tasks adapt to agent capability?

**Approach**: Implement adaptive task generation where difficulty scales with population capability.

**Hypothesis**: If tasks get harder as agents get more capable, capability gains will translate to cooperation gains.

**Implementation**:
```python
def generate_task(population_capability):
    base_difficulty = 0.5
    scaling = 0.3  # How much difficulty responds to capability
    difficulty = base_difficulty + scaling * (population_capability - 0.5)
    return Task(difficulty=difficulty)
```

**EV Assessment**:
- High theoretical value (tests binding constraint hypothesis)
- Medium practical value (real-world tasks may not adapt)
- Low implementation effort (simple modification)
- **Overall EV: HIGH**

---

## Direction 3: Capability Signaling

**Question**: How do agents communicate capability to enable matching?

**Approach**: Test different signaling mechanisms:
1. **Cheap talk**: Agents claim capability (can lie)
2. **Costly signaling**: Agents demonstrate capability (costly but credible)
3. **Reputation-based**: Past performance signals capability
4. **Direct observation**: Agents observe each other's performance

**Hypothesis**: Costly signaling and direct observation outperform cheap talk.

**EV Assessment**:
- High theoretical value (connects to signaling theory)
- High practical value (real systems need signaling)
- Medium implementation effort
- **Overall EV: HIGH**

---

## Direction 4: Specialization vs Generalization

**Question**: Should agents specialize in certain task types or generalize?

**Approach**: Introduce task types (e.g., technical, social, creative) and test:
1. **Generalist**: All agents learn all templates
2. **Specialist**: Agents focus on one task type
3. **Hybrid**: Some specialists, some generalists

**Hypothesis**: Specialization + matching outperforms generalization.

**EV Assessment**:
- High theoretical value (division of labor)
- High practical value (real organizations specialize)
- Medium implementation effort
- **Overall EV: MEDIUM-HIGH**

---

## Direction 5: Commitment Granularity

**Question**: Does the size/scope of commitments matter?

**Approach**: Test different commitment structures:
1. **Atomic**: One task per commitment
2. **Bundled**: Multiple related tasks per commitment
3. **Hierarchical**: Commitments contain sub-commitments
4. **Flexible**: Agents choose granularity

**Hypothesis**: Optimal granularity depends on capability distribution.

**EV Assessment**:
- Medium theoretical value (less novel)
- High practical value (real systems vary granularity)
- Medium implementation effort
- **Overall EV: MEDIUM**

---

## Direction 6: Information Asymmetry

**Question**: What happens when agents have different information about tasks/capabilities?

**Approach**: Test information structures:
1. **Symmetric**: All agents know everything
2. **Private capability**: Agents know own capability only
3. **Private task info**: Task difficulty revealed only to assigned agent
4. **Asymmetric**: Some agents know more than others

**Hypothesis**: Information asymmetry creates coordination failures that matching mechanisms can address.

**EV Assessment**:
- High theoretical value (connects to mechanism design)
- High practical value (real systems have asymmetry)
- Medium implementation effort
- **Overall EV: HIGH**

---

## Direction 7: Endogenous Task Generation

**Question**: What if agents create tasks for each other?

**Approach**: Allow agents to:
1. Create tasks (with difficulty based on their capability)
2. Assign tasks to others
3. Evaluate task completion

**Hypothesis**: Endogenous task generation creates natural capability matching.

**EV Assessment**:
- High theoretical value (emergent coordination)
- Medium practical value (some systems have this)
- High implementation effort
- **Overall EV: MEDIUM**

---

## Direction 8: Multi-Agent Commitments

**Question**: What if tasks require multiple agents?

**Approach**: Introduce team tasks requiring coordination:
1. **Additive**: Success = sum of individual contributions
2. **Complementary**: Success requires diverse capabilities
3. **Weakest link**: Success limited by lowest contributor
4. **Best shot**: Success determined by best contributor

**Hypothesis**: Complementary tasks create strongest incentives for capability sharing.

**EV Assessment**:
- High theoretical value (team production)
- High practical value (real tasks often require teams)
- Medium implementation effort
- **Overall EV: HIGH**

---

## Prioritized Research Agenda

### Tier 1: Immediate (Next Experiment)
1. **Direction 2: Dynamic Task Difficulty** - Simplest test of binding constraint hypothesis
2. **Direction 1: Task-Capability Matching** - Core mechanism question

### Tier 2: Short-term (Next 2-3 Experiments)
3. **Direction 6: Information Asymmetry** - Connects to mechanism design
4. **Direction 8: Multi-Agent Commitments** - Team production

### Tier 3: Medium-term (Future Work)
5. **Direction 3: Capability Signaling** - Signaling theory
6. **Direction 4: Specialization** - Division of labor

### Tier 4: Long-term (If Resources Allow)
7. **Direction 5: Commitment Granularity**
8. **Direction 7: Endogenous Task Generation**

---

## Recommended Next Step

**Experiment 32A: Dynamic Task Difficulty**

This is the simplest, highest-EV test. If tasks adapt to capability, we should see capability gains translate to cooperation gains.

**Design**:
- Control: Fixed difficulty (0.5)
- Treatment: Adaptive difficulty (scales with mean capability)
- Both with 100% template sharing

**Prediction**: Treatment shows higher cooperation because capability is no longer wasted.

**If confirmed**: Proceed to full matching mechanism study (Experiment 32B)
**If not confirmed**: Revise theory - maybe capability isn't the binding constraint

---

## Alternative: Simplest Possible Test

Before the full factorial, run a **2×2 pilot**:

| | Fixed Tasks | Adaptive Tasks |
|---|---|---|
| **No Sharing** | Baseline | Adaptive-only |
| **Full Sharing** | Sharing-only | Both |

This tests the interaction between sharing and task adaptation with minimal conditions.

**Prediction**:
- Sharing-only: High capability, flat cooperation (confirmed in Exp 31)
- Adaptive-only: Same capability, higher cooperation
- Both: High capability, high cooperation

If the interaction is significant, proceed to full factorial.
