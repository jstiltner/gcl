# Experiment 38: Adversarial Regime Changes - Findings

## Executive Summary

Experiment 38 tests GCL against MARL in **fundamentally adversarial** regime changes where learned policies become harmful. The results are **nuanced**: GCL shows advantage only in specific adversarial scenarios.

## Adversarial Regime Types Tested

| Type | Regime 0 | Regime 1 | What Changes |
|------|----------|----------|--------------|
| cooperate_compete | Volunteering helps | Volunteering hurts | Action meaning inverts |
| majority_minority | Need many volunteers | Need few volunteers | Optimal quantity changes |
| capability_inversion | High capability helps | Low capability helps | Capability meaning inverts |
| reward_inversion | Success = reward | Success = penalty | Reward meaning inverts |

## Results Summary

### GCL Advantage by Regime Type

| Regime Type | GCL Advantage | Interpretation |
|-------------|---------------|----------------|
| cooperate_compete | **+2.3%** | GCL wins |
| majority_minority | **-25.1%** | MARL wins significantly |
| capability_inversion | -2.6% | MARL wins slightly |
| reward_inversion | -0.4% | Essentially tied |

### Detailed Results: cooperate_compete (GCL Wins)

| Frequency | GCL | IQL | QMIX | GCL Adv |
|-----------|-----|-----|------|---------|
| 50 | 0.223 | 0.201 | 0.200 | **+0.022** |
| 100 | 0.224 | 0.199 | 0.197 | **+0.025** |
| 200 | 0.226 | 0.198 | 0.204 | **+0.021** |

**Why GCL wins**: When the regime switches from "cooperate" to "compete", MARL's learned "volunteer" policy becomes actively harmful. GCL adapts based on feedback.

### Detailed Results: majority_minority (MARL Wins)

| Frequency | GCL | IQL | QMIX | GCL Adv |
|-----------|-----|-----|------|---------|
| 50 | 0.391 | 0.605 | 0.571 | **-0.214** |
| 100 | 0.393 | 0.654 | 0.577 | **-0.261** |
| 200 | 0.392 | 0.668 | 0.620 | **-0.276** |

**Why MARL wins**: The majority/minority regime doesn't invert the meaning of actions - it just changes the optimal quantity. MARL can learn this more effectively than GCL's simple feedback mechanism.

## Key Findings

### 1. GCL Advantage is Regime-Specific

GCL only shows advantage when:
- **Action meaning inverts** (cooperate → compete)
- MARL's learned policy becomes **actively harmful**

GCL does NOT show advantage when:
- Only **parameters change** (quantity, thresholds)
- MARL's learned policy remains **partially useful**

### 2. MARL is More Robust Than Expected

Even in adversarial regimes, MARL methods:
- Adapt through exploration (epsilon reset)
- Retain useful partial knowledge
- Learn new patterns quickly

### 3. The "Adversarial" Framing Matters

Not all regime changes are equally adversarial:
- **True adversarial**: Action meaning inverts (cooperate_compete)
- **Parametric change**: Optimal values shift (majority_minority)
- **Semantic change**: Interpretation changes (capability_inversion)

## Theoretical Implications

### When GCL Excels

1. **Action meaning inversion**: When "good" becomes "bad"
2. **No partial transfer**: When old knowledge is harmful
3. **Rapid feedback**: When environment provides clear signals

### When MARL Excels

1. **Parametric changes**: When optimal values shift
2. **Partial transfer**: When old knowledge is partially useful
3. **Complex patterns**: When optimal policy is non-obvious

### The Fundamental Trade-off

| Aspect | GCL | MARL |
|--------|-----|------|
| Learning | None (reactive) | Extensive |
| Adaptation | Immediate | Gradual |
| Knowledge transfer | None | Partial |
| Harmful transfer | Impossible | Possible |

## Comparison with Experiments 36-37

| Experiment | Environment | GCL vs MARL |
|------------|-------------|-------------|
| 36 | Stationary | MARL +3% |
| 37 | Parametric changes | GCL +0.5% |
| 38 | Adversarial (cooperate_compete) | GCL +2.3% |
| 38 | Adversarial (majority_minority) | MARL +25% |

## Practical Recommendations

### Use GCL When:
1. Environment may fundamentally change (cooperate ↔ compete)
2. Old knowledge could become harmful
3. Simplicity and interpretability matter
4. Worst-case performance matters

### Use MARL When:
1. Environment changes are parametric
2. Old knowledge remains partially useful
3. Maximum performance is needed
4. Training time is available

### Hybrid Approach:
1. **Detect regime type**: Is this a fundamental or parametric change?
2. **If fundamental**: Reset to GCL
3. **If parametric**: Continue MARL with increased exploration

## Conclusion

The adversarial regime experiments reveal that **GCL's advantage is context-dependent**:

- **GCL wins** when action meanings invert (cooperate_compete: +2.3%)
- **MARL wins** when only parameters change (majority_minority: -25.1%)

This suggests GCL is best positioned as a **safety mechanism** for environments where fundamental regime changes are possible, rather than a universal replacement for MARL.

The key insight is:
> **GCL prevents harmful knowledge transfer, but also prevents beneficial knowledge transfer.**

In environments where old knowledge is likely to be harmful, GCL's "no transfer" approach is advantageous. In environments where old knowledge is likely to be useful, MARL's learning approach is superior.
