# Experiment 30: Ubuntu Decomposition Study

## Motivation

The Ubuntu structure dominates all other social structures in Experiments 27-29, winning 99% of pairwise comparisons across 22 boundary conditions. However, this "dominance" may be confounded by two mechanical factors:

1. **Knowledge Pooling**: Ubuntu shares ALL templates with ALL agents every round (100% diffusion)
2. **Reputation Pooling**: All agents share one collective reputation (Gini = 0 by construction)

These factors compound to give Ubuntu agents ~2x the capability of other structures and zero stratification. The question is: **Is Ubuntu's advantage due to its philosophy ("I am because we are") or simply due to pooling mechanics?**

## Research Questions

1. How much of Ubuntu's advantage comes from knowledge pooling alone?
2. How much comes from reputation pooling alone?
3. Is there an interaction effect (pooling both > sum of parts)?
4. What is the "pure Ubuntu" effect after controlling for pooling?

## Experimental Design

### 2x2 Factorial Design

| Condition | Knowledge | Reputation | Description |
|-----------|-----------|------------|-------------|
| **Baseline** | Individual | Individual | Standard meritocracy (no pooling) |
| **K-Pool** | Pooled | Individual | Full knowledge sharing, individual reputation |
| **R-Pool** | Individual | Pooled | Individual knowledge, shared reputation |
| **Full-Pool** | Pooled | Pooled | Ubuntu-style (both pooled) |

### Operationalization

#### Knowledge Pooling Levels
- **Individual**: Agents keep templates to themselves (transfer only via explicit mechanisms)
- **Pooled**: All templates shared with all agents every round (Ubuntu-style)

#### Reputation Pooling Levels
- **Individual**: Each agent has own reputation, updated based on own outcomes
- **Pooled**: Single collective reputation, all agents share same value

### Implementation

```python
class DecompositionStructure(BaseStructure):
    """Parameterized structure for decomposition study."""
    
    def __init__(
        self,
        knowledge_pooling: bool = False,  # True = share all templates
        reputation_pooling: bool = False,  # True = collective reputation
        config: StructureConfig = None
    ):
        self.knowledge_pooling = knowledge_pooling
        self.reputation_pooling = reputation_pooling
        self.collective_reputation = 0.5  # Used if reputation_pooling=True
        self.config = config or StructureConfig()
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """Share knowledge based on pooling setting."""
        if self.knowledge_pooling:
            # Full pooling: share all with all
            all_templates = {}
            for agent in agents:
                for t in agent.template_library:
                    all_templates[t.id] = t
            
            for agent in agents:
                agent_ids = {t.id for t in agent.template_library}
                for tid, template in all_templates.items():
                    if tid not in agent_ids:
                        agent.template_library.append(template)
        # else: no automatic sharing (individual)
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Process outcome based on reputation pooling setting."""
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if self.reputation_pooling:
            # Collective reputation
            n = len(all_agents)
            if outcome.success:
                delta = (self.config.success_reward * difficulty_mult) / n
                self.collective_reputation = min(1.0, self.collective_reputation + delta)
            else:
                delta = (self.config.failure_penalty / difficulty_mult) / n
                self.collective_reputation = max(0.0, self.collective_reputation - delta)
            
            # Sync all agents
            for a in all_agents:
                a.reputation = self.collective_reputation
        else:
            # Individual reputation
            if outcome.success:
                delta = self.config.success_reward * difficulty_mult
                agent.reputation = min(1.0, agent.reputation + delta)
            else:
                delta = self.config.failure_penalty / difficulty_mult
                agent.reputation = max(0.0, agent.reputation - delta)
        
        # Template learning (always individual)
        if outcome.success:
            agent.learn_template(outcome.task, True)
```

## Metrics

### Primary Metrics
1. **Cooperation Rate**: Task success rate
2. **Mean Capability**: Average agent capability at end
3. **Gini Coefficient**: Inequality in reputation/resources
4. **Total Output**: Aggregate task completion value

### Secondary Metrics
1. **Knowledge Diffusion**: Fraction of templates shared across agents
2. **Recovery Rate**: How often struggling agents recover
3. **Underclass Size**: Fraction of agents with reputation < 0.3

## Hypotheses

### H1: Knowledge Pooling is Primary Driver
- K-Pool will achieve ~80% of Full-Pool's advantage over Baseline
- R-Pool alone will achieve ~20% of the advantage

### H2: Reputation Pooling Reduces Variance
- R-Pool will have Gini ≈ 0 (by construction)
- K-Pool will have Gini similar to Baseline

### H3: Interaction Effect Exists
- Full-Pool > K-Pool + R-Pool (superadditive)
- Knowledge enables capability; shared reputation enables risk-taking

### H4: "Ubuntu Philosophy" Effect is Small
- After controlling for pooling, the residual "Ubuntu" effect will be < 10%

## Analysis Plan

### Main Effects
```
Effect_Knowledge = (K-Pool + Full-Pool) / 2 - (Baseline + R-Pool) / 2
Effect_Reputation = (R-Pool + Full-Pool) / 2 - (Baseline + K-Pool) / 2
```

### Interaction Effect
```
Interaction = Full-Pool - K-Pool - R-Pool + Baseline
```

### Decomposition
```
Ubuntu_Advantage = Full-Pool - Baseline
Knowledge_Contribution = K-Pool - Baseline
Reputation_Contribution = R-Pool - Baseline
Interaction_Contribution = Interaction
Residual = Ubuntu_Advantage - Knowledge_Contribution - Reputation_Contribution - Interaction_Contribution
```

If Residual ≈ 0, then Ubuntu's advantage is fully explained by pooling mechanics.
If Residual > 0, there's something else about Ubuntu (the "philosophy").

## Expected Results

Based on the current data:
- Ubuntu: cooperation 0.562, capability 0.958, Gini 0.0
- Meritocracy: cooperation 0.354, capability 0.499, Gini 0.125

**Prediction**: K-Pool will achieve cooperation ~0.50, capability ~0.90, Gini ~0.10
This would suggest knowledge pooling is the primary driver (~75% of effect).

## Implications for Publication

### If H1 Confirmed (Knowledge is Primary)
Reframe finding as: "Information sharing, not collective identity, drives coordination advantage. Structures that hoard knowledge create artificial scarcity and underperform."

### If H3 Confirmed (Interaction Exists)
Reframe as: "Knowledge pooling and risk-sharing interact synergistically. Neither alone achieves the full benefit—collective identity enables both."

### If H4 Rejected (Residual > 10%)
Keep Ubuntu framing but specify: "Beyond pooling mechanics, the Ubuntu model's emphasis on collective identity provides additional coordination benefits."

## Implementation Timeline

1. Create `DecompositionStructure` class with parameterized pooling
2. Run 4 conditions × 5 replications × 100 rounds
3. Compute main effects and interaction
4. Update publication framing based on results

## Code Location

- Experiment: `experiments/30_ubuntu_decomposition.py`
- Structure: `experiments/social_structures/structures/decomposition.py`
- Results: `results/experiment_30_decomposition.json`

## Statistical Tests

- ANOVA for main effects and interaction
- Effect sizes (Cohen's d) for each comparison
- Bootstrap confidence intervals for decomposition percentages
