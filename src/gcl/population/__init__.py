"""
Population-scale commitment experiments.

This module implements Track 2 from the experimental strategy:
- Lightweight agents for fast local experimentation (50-500 agents)
- Population environment for emergent dynamics
- Metrics for tracking institutional emergence

Key components:
- LightweightCommitmentAgent: Fast neural network policy
- PopulationEnvironment: Manages agent interactions
- PopulationMetrics: Tracks emergent phenomena

Falsifiable Predictions:
1. Protocol Convergence: Entropy decreases logarithmically
2. Trust Network Structure: Small-world properties emerge
3. Template Fitness Dynamics: Replicator dynamics
4. Specialization Emergence: Gini coefficient increases
5. Critical Population Size: N* ≈ 20-50
"""

from gcl.population.lightweight_agent import (
    AgentConfig,
    AgentStats,
    CommitmentEncoder,
    CommitmentOutcome,
    CommitmentRecord,
    LearnedTemplate,
    LightweightCommitment,
    LightweightCommitmentAgent,
)
from gcl.population.environment import (
    PopulationConfig,
    PopulationEnvironment,
    Task,
)
from gcl.population.metrics import (
    PopulationMetrics,
)
from gcl.population.predictions import (
    test_prediction_1_protocol_convergence,
    test_prediction_2_trust_network,
    test_prediction_3_template_dynamics,
    test_prediction_4_specialization,
    test_prediction_5_critical_threshold,
    test_all_predictions,
)

__all__ = [
    # Agent
    "AgentConfig",
    "AgentStats",
    "CommitmentEncoder",
    "CommitmentOutcome",
    "CommitmentRecord",
    "LearnedTemplate",
    "LightweightCommitment",
    "LightweightCommitmentAgent",
    # Environment
    "PopulationConfig",
    "PopulationEnvironment",
    "Task",
    # Metrics
    "PopulationMetrics",
    # Predictions
    "test_prediction_1_protocol_convergence",
    "test_prediction_2_trust_network",
    "test_prediction_3_template_dynamics",
    "test_prediction_4_specialization",
    "test_prediction_5_critical_threshold",
    "test_all_predictions",
]
