# Experiment 31: Template Sharing Rate Optimization

## Motivation

Experiment 30 showed that knowledge pooling (template sharing) is the primary driver of coordination advantage. But it only tested two extremes:
- 0% sharing (Baseline)
- 100% sharing (K-Pool)

**Question**: What's the optimal sharing rate? Is there a point of diminishing returns?

## Research Questions

1. How does cooperation scale with sharing rate?
2. Is there a "sweet spot" that achieves most of the benefit with less overhead?
3. Does capability scale linearly or show diminishing returns?
4. What's the efficiency frontier (cooperation per unit of sharing)?

## Experimental Design

### Sharing Rate Conditions

| Condition | Sharing Rate | Description |
|-----------|--------------|-------------|
| S0 | 0% | No sharing (baseline) |
| S25 | 25% | Each agent shares 25% of templates |
| S50 | 50% | Each agent shares 50% of templates |
| S75 | 75% | Each agent shares 75% of templates |
| S100 | 100% | Full pooling (all templates shared) |

### Implementation

```python
def maybe_share_knowledge(self, agents: List[Agent], sharing_rate: float):
    """Share templates at specified rate."""
    if sharing_rate == 0:
        return  # No sharing
    
    if sharing_rate == 1.0:
        # Full pooling (existing Ubuntu implementation)
        all_templates = {}
        for agent in agents:
            for t in agent.template_library:
                all_templates[t.id] = t
        
        for agent in agents:
            agent_ids = {t.id for t in agent.template_library}
            for tid, template in all_templates.items():
                if tid not in agent_ids:
                    agent.template_library.append(template)
    else:
        # Partial sharing: each agent shares fraction of their templates
        for agent in agents:
            if not agent.template_library:
                continue
            
            # Select templates to share
            n_to_share = max(1, int(len(agent.template_library) * sharing_rate))
            templates_to_share = random.sample(agent.template_library, n_to_share)
            
            # Share with random subset of other agents
            other_agents = [a for a in agents if a.id != agent.id]
            n_recipients = max(1, int(len(other_agents) * sharing_rate))
            recipients = random.sample(other_agents, n_recipients)
            
            for template in templates_to_share:
                for recipient in recipients:
                    recipient.receive_template(template)
```

### Metrics

1. **Cooperation Rate**: Task success rate
2. **Mean Capability**: Average effective capability
3. **Knowledge Diffusion**: Templates per agent / unique templates
4. **Sharing Overhead**: Number of template transfers per round
5. **Efficiency**: Cooperation gain per unit of sharing overhead

## Hypotheses

### H1: Diminishing Returns
Cooperation will show diminishing returns with sharing rate:
- 0→25%: Large gain
- 25→50%: Moderate gain
- 50→75%: Small gain
- 75→100%: Minimal gain

### H2: 50% is Near-Optimal
50% sharing will achieve >80% of the benefit of 100% sharing.

### H3: Capability Scales Linearly
Unlike cooperation, capability will scale roughly linearly with sharing rate.

### H4: Efficiency Peaks at Moderate Sharing
The efficiency metric (cooperation/overhead) will peak around 25-50%.

## Expected Results

```
Sharing Rate | Cooperation | Capability | Efficiency
-------------|-------------|------------|------------
0%           | 0.56        | 0.55       | N/A
25%          | 0.60        | 0.70       | HIGH
50%          | 0.63        | 0.82       | MEDIUM
75%          | 0.65        | 0.90       | LOW
100%         | 0.57        | 0.96       | LOWEST
```

Note: Cooperation may actually DECREASE at very high sharing rates due to:
- Reduced specialization incentives
- Free-rider effects
- Capability ceiling effects

## Analysis Plan

### 1. Dose-Response Curve
Plot cooperation and capability vs sharing rate.

### 2. Marginal Returns
Calculate marginal gain for each 25% increment.

### 3. Efficiency Frontier
Plot cooperation vs overhead to find optimal operating point.

### 4. Regression Analysis
Fit curves to determine functional form:
- Linear: y = a + bx
- Logarithmic: y = a + b*log(x)
- Saturation: y = a * (1 - e^(-bx))

## Practical Implications

If H2 is confirmed (50% achieves 80% of benefit):
- **Recommendation**: Implement moderate sharing policies
- **Rationale**: Full pooling has overhead costs (communication, storage)
- **Actionable**: "Share half your templates with half your peers"

## Implementation Timeline

1. Create `SharingRateStructure` class with configurable rate
2. Run 5 conditions × 10 seeds × 100 rounds
3. Compute metrics and efficiency curves
4. Generate visualizations
5. Update findings document

## Files to Create

- `experiments/31_sharing_rate_optimization.py`
- `experiments/social_structures/structures/sharing_rate.py`
- `results/experiment_31_sharing_rate.json`

## Connection to GCL

This experiment directly tests the **template hierarchy** mechanism from the GCL framework:
- Templates are the unit of knowledge transfer
- Sharing policies determine knowledge diffusion
- Optimal sharing rate is a design parameter for multi-agent systems

The finding will inform recommendations for:
- How much knowledge to share in commitment-based coordination
- Trade-offs between capability and overhead
- Practical deployment of GCL in real systems
