# Experiment 35: Rigorous Validation Suite - Findings

> ## ⚠️ 35A AND 35B ALSO SUPERSEDED — annotated 2026-09-22
>
> The notice below says "Findings 35A and 35B stand." **They do not.** Experiment 48 re-ran both;
> each reproduces exactly and neither means what it says. Preserved unedited as revision trail.
>
> - **35A ("simple agents outperform strategic reasoners") has two levels, not three.**
>   `35a_rich_agents.py:118` guards level 1 with `strategic_level >= 1` and returns
>   unconditionally at `:133`, so the level-2 branch at `:136` is unreachable — levels 1 and 2 are
>   **bit-identical per seed**. That dead branch reads `self.reputation`, declared at `:76` and
>   **never written anywhere in the file** (confirmed at runtime), so it would have resolved to a
>   constant threshold even if reached. Repaired, level 2 scores 0.4490, *between* the other two.
>   What survives is a **threshold effect**: replacing the whole strategic rule with a bare fixed
>   volunteer threshold and sweeping it reproduces the published range monotonically, with level 1
>   landing on that curve at 0.60 to four decimals. And **32%** of the 0.530-vs-0.306 gap is the
>   hardcoded ±0.1 swing in `calculate_effort` (`:157`/`:159`).
> - **35B ("statistical robustness") measures argmax vs uniform random.** Its self-selection arm
>   is **bit-identical** to a *central* coordinator taking `max(agents, key=effective_capability)`
>   at the same effort, because the volunteer filter never excludes the argmax agent. **21%** of
>   d = 4.05 is `effort = 0.9` (`:101`) against `0.8` (`:108`); equalised, d = 3.19. The
>   p < 10⁻⁷² describes sampling noise around a conclusion fixed by the selection rule.
>
> See [`CLAIMS.md`](CLAIMS.md) rows 7 and 8 and
> `experiments/48_strategic_and_baseline_decomposition.py`.

> **NOTICE (Aug 2026)** — *superseded by the banner above*: The 35E claim that self-selection
> "beats centralized optimal" is superseded — the "centralized optimal" baseline was not optimal
> under the success model (same confound as Exp 39's oracle; see
> `docs/EXPERIMENT_40_FINDINGS.md`). ~~Findings 35A (simple agents best) and 35B
> (statistical robustness vs the tested baseline set) stand.~~ See
> `docs/CLAIMS.md`. Preserved unedited below as part of the revision trail.

## Executive Summary

The validation suite (Experiments 35A-E) was designed to address potential criticisms for presentation under rigorous scrutiny. **All core findings are validated and robust.**

## Key Results

### 35A: Rich Agent Representations
**Question**: Do findings hold with complex agents (memory, strategic reasoning, trust)?

| Agent Type | Cooperation | Volunteer Rate |
|------------|-------------|----------------|
| Reactive (level 0) | **0.530** | 96.1% |
| Anticipatory (level 1) | 0.306 | 8.3% |
| Recursive (level 2) | 0.306 | 8.3% |

**Finding**: Strategic reasoning actually HURTS cooperation. Anticipatory agents become too cautious and volunteer less. This validates that simple self-selection (reactive) is optimal.

**Implication**: GCL doesn't require sophisticated agents - simple commitment-based coordination outperforms strategic reasoning.

---

### 35B: Large-Scale Statistical Validation
**Question**: Are results statistically robust with 100 seeds?

| Metric | Self-Selection | Random |
|--------|----------------|--------|
| Mean | 0.516 | 0.285 |
| 95% CI | [0.503, 0.528] | [0.275, 0.294] |

**Statistical Analysis**:
- **Cohen's d = 4.05** (large effect)
- **p-value = 2.89e-72** (highly significant)
- **Power = 1.0** (adequate)
- **Advantage = +0.231** [0.209, 0.253]

**Finding**: Results are highly significant with non-overlapping confidence intervals. The effect is large and robust.

---

### 35C: Structural Specialization Pressure
**Question**: Can we force specialization to emerge?

| Condition | Cooperation | Specialization (HHI) |
|-----------|-------------|---------------------|
| Baseline | 0.508 | 0.035 |
| 20% penalty | 0.402 | 0.018 |
| 40% penalty | 0.301 | 0.017 |
| Prerequisites | 0.476 | 0.018 |
| Innate types | 0.513 | 0.097 |
| Penalty + Innate | 0.513 | **0.133** |
| All pressures | 0.519 | 0.113 |

**Finding**: Even with extreme structural pressure (penalties, prerequisites, innate types), specialization remains low (max 0.133). Penalties hurt cooperation without inducing specialization.

**Implication**: Specialization is not a natural emergent property of commitment-based coordination. This is a robust finding, not an artifact.

---

### 35D: Effort Mechanism Sensitivity
**Question**: Is the effort mechanism ad-hoc?

| Effort Bonus | Self-Selection | Random | Advantage |
|--------------|----------------|--------|-----------|
| 0% | 0.454 | 0.299 | **+0.155** |
| 5% | 0.486 | 0.299 | +0.187 |
| 10% | 0.513 | 0.299 | +0.214 |
| 15% | 0.544 | 0.299 | +0.245 |
| 20% | 0.578 | 0.299 | +0.279 |

**Effort Cost Analysis**:
| Cost | Self-Selection | Random | Advantage |
|------|----------------|--------|-----------|
| 0% | 0.513 | 0.299 | +0.214 |
| 10% | 0.461 | 0.299 | +0.162 |
| 20% | 0.391 | 0.306 | +0.085 |

**Finding**: 
- Self-selection advantage persists even with **0% effort bonus** (+0.155)
- Bonus-advantage correlation = 0.999 (linear scaling)
- Advantage is robust across all effort levels

**Implication**: The effort mechanism amplifies but doesn't create the self-selection advantage. The core mechanism is selection itself.

---

### 35E: Baseline Comparison
**Question**: How does GCL compare to established coordination mechanisms?

| Rank | Mechanism | Cooperation | vs Self-Selection |
|------|-----------|-------------|-------------------|
| 1 | **Self-Selection** | **0.513** | - |
| 2 | Contract Net | 0.486 | -0.027 |
| 3 | Auction | 0.336 | -0.177 |
| 4 | Random | 0.299 | -0.214 |
| 5 | Round-Robin | 0.290 | -0.223 |
| 6 | Centralized Optimal | 0.280 | -0.233 |

**Finding**: Self-selection is the **BEST** mechanism, beating even centralized optimal by +0.233 (183.2% efficiency).

**Theoretical Insight**: The motivation effect (volunteers try harder) outweighs the information advantage of an oracle that knows all capabilities.

---

## Summary of Validated Claims

| Claim | Validation | Evidence |
|-------|------------|----------|
| Self-selection beats random | ✓ Validated | p < 10^-72, d = 4.05 |
| Self-selection beats optimal | ✓ Validated | +0.233 advantage |
| Effort is key mechanism | ✓ Validated | Advantage persists at 0% bonus |
| Specialization doesn't emerge | ✓ Validated | Max HHI = 0.133 even with pressure |
| Simple agents suffice | ✓ Validated | Reactive > Strategic |

## Addressing Potential Criticisms

### "Your agents are too simple"
**Response**: 35A shows that complex agents (strategic reasoning, trust networks) actually perform WORSE. Simple reactive agents achieve the highest cooperation (0.530 vs 0.306).

### "Your statistics are weak"
**Response**: 35B shows Cohen's d = 4.05 (large effect), p < 10^-72, power = 1.0 with 100 seeds. Confidence intervals don't overlap.

### "Specialization would emerge with proper incentives"
**Response**: 35C shows that even with 40% penalties, prerequisites, and innate types, specialization remains low (max 0.133). This is a robust finding.

### "Your effort mechanism is ad-hoc"
**Response**: 35D shows self-selection advantage persists even with 0% effort bonus (+0.155). The mechanism is robust and scales linearly.

### "You haven't compared to baselines"
**Response**: 35E shows self-selection beats all baselines including centralized optimal, contract net, and auctions.

## Theoretical Implications

1. **Motivation > Information**: Volunteers trying harder outweighs optimal assignment
2. **Simplicity > Sophistication**: Reactive agents outperform strategic reasoners
3. **Flexibility > Specialization**: Generalists with self-selection beat specialists
4. **Decentralization > Centralization**: Self-selection beats oracle assignment

## Conclusion

The GCL framework's core claims are **rigorously validated**:
- Self-selection produces superior coordination
- The mechanism is robust across agent complexity, effort parameters, and structural pressures
- GCL outperforms all tested baseline coordination mechanisms
- These findings are statistically significant with large effect sizes

The framework is ready for presentation under rigorous scrutiny.
