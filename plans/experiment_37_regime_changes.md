# Experiment 37: Regime Change Robustness

## Hypothesis

GCL's sample efficiency advantage (25-50x faster than MARL) should translate to **superior performance in non-stationary environments** where regime changes occur. MARL methods must re-learn after each change, while GCL adapts immediately.

## Theoretical Basis

From Experiment 36:
- GCL reaches 50% cooperation in ~2 episodes
- QMIX requires ~102 episodes
- IQL requires ~52 episodes

If the environment changes every N episodes:
- **N < 50**: MARL never converges before next change
- **N ~ 100**: MARL barely converges before next change
- **N > 500**: MARL has time to converge and may outperform

## Regime Change Types

### 37A: Task Difficulty Shift
- Regime 1: Easy tasks (difficulty 0.2-0.4)
- Regime 2: Hard tasks (difficulty 0.6-0.8)
- Alternating every N episodes

### 37B: Agent Capability Shift
- Regime 1: High-capability agents (0.6-0.9)
- Regime 2: Low-capability agents (0.3-0.6)
- Simulates workforce changes

### 37C: Reward Structure Shift
- Regime 1: Individual rewards (each volunteer gets reward)
- Regime 2: Team rewards (only best volunteer gets reward)
- Tests adaptation to incentive changes

### 37D: Population Turnover
- Every N episodes, replace 30% of agents
- New agents have random capabilities
- Tests robustness to membership changes

### 37E: Combined Regime Changes
- Multiple changes occurring simultaneously
- Most realistic scenario

## Experimental Design

### Independent Variables
1. **Method**: GCL, QMIX, MAPPO, IQL
2. **Regime change frequency**: Every 50, 100, 200, 500, 1000 episodes
3. **Regime change type**: 37A-E

### Dependent Variables
1. **Average cooperation rate** across all regimes
2. **Recovery time** after regime change (episodes to return to 80% of pre-change performance)
3. **Stability** (variance in cooperation rate)
4. **Worst-case performance** (minimum cooperation in any window)

### Metrics

```
Adaptation Score = (Performance after change) / (Performance before change)
Recovery Time = Episodes until Adaptation Score > 0.8
Robustness = 1 - Variance(cooperation across regimes)
```

## Expected Results

### Regime Change Every 50 Episodes
- **GCL**: Maintains ~0.53 cooperation (immediate adaptation)
- **MARL**: Drops to ~0.30 (never converges)
- **Advantage**: GCL +0.23 (77% better)

### Regime Change Every 100 Episodes
- **GCL**: Maintains ~0.53 cooperation
- **MARL**: Achieves ~0.40 (partial convergence)
- **Advantage**: GCL +0.13 (33% better)

### Regime Change Every 500 Episodes
- **GCL**: Maintains ~0.53 cooperation
- **MARL**: Achieves ~0.50 (near convergence)
- **Advantage**: GCL +0.03 (6% better)

### Regime Change Every 1000 Episodes
- **GCL**: Maintains ~0.53 cooperation
- **MARL**: Achieves ~0.55 (full convergence)
- **Advantage**: MARL +0.02 (4% better)

## Key Predictions

1. **Crossover point**: GCL outperforms MARL when regime changes occur more frequently than every ~300 episodes

2. **Recovery time**: GCL recovers in 1-2 episodes; MARL requires 50-100 episodes

3. **Worst-case**: GCL's worst-case performance is much better than MARL's

4. **Stability**: GCL has lower variance across regimes

## Implementation Plan

```python
# Pseudocode for regime change experiment

def run_regime_change_experiment(method, change_frequency, change_type, n_episodes):
    agent = create_agent(method)
    env = create_environment()
    
    results = []
    current_regime = 0
    
    for episode in range(n_episodes):
        # Check for regime change
        if episode > 0 and episode % change_frequency == 0:
            current_regime = 1 - current_regime  # Toggle regime
            env.apply_regime(current_regime, change_type)
        
        # Run episode
        reward = agent.step(env)
        results.append({
            'episode': episode,
            'regime': current_regime,
            'reward': reward,
            'episodes_since_change': episode % change_frequency
        })
        
        agent.train()
    
    return analyze_results(results)
```

## Statistical Analysis

1. **Two-way ANOVA**: Method × Change Frequency
2. **Post-hoc tests**: Tukey HSD for pairwise comparisons
3. **Effect sizes**: Cohen's d for GCL vs each MARL method
4. **Confidence intervals**: 95% bootstrap CIs

## Presentation-Ready Outputs

1. **Line plot**: Cooperation vs Episode with regime change markers
2. **Heatmap**: Method × Change Frequency → Average Cooperation
3. **Bar chart**: Recovery time by method
4. **Box plot**: Cooperation distribution by method and regime

## Addressing Potential Criticisms

### "MARL could use transfer learning"
**Response**: True, but transfer learning requires knowing when regimes change. GCL doesn't need this information.

### "Real regime changes are gradual"
**Response**: We can test gradual transitions (37F) where parameters shift linearly over N episodes.

### "MARL could detect regime changes"
**Response**: This adds complexity. GCL's advantage is simplicity - no change detection needed.

## Conclusion

Experiment 37 will demonstrate that GCL's sample efficiency translates to **superior robustness in non-stationary environments**. The key insight is:

> **GCL adapts immediately because it doesn't learn - it coordinates.**

This positions GCL as the preferred method for:
- Dynamic environments
- Uncertain regime durations
- Systems requiring consistent performance
- Applications where worst-case matters more than average-case
