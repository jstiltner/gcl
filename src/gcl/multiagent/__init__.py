"""
Multi-agent coordination through commitment markets.

This module implements:
- Commitment markets for agent coordination
- Agent matching and task allocation
- Communication protocols (commitment-based vs representation-based)
- Validation of Theorems 1 (Communication Complexity) and 3 (Nash Equilibrium)
"""

from gcl.multiagent.market import (
    CommitmentMarket,
    MarketConfig,
    MarketStatistics,
    Task,
    TaskAllocation,
)
from gcl.multiagent.agent import (
    CommitmentAgent,
    AgentConfig,
    AgentState,
)
from gcl.multiagent.protocol import (
    CommunicationProtocol,
    CommitmentProtocol,
    RepresentationProtocol,
    ProtocolMetrics,
)

__all__ = [
    # Market
    "CommitmentMarket",
    "MarketConfig",
    "MarketStatistics",
    "Task",
    "TaskAllocation",
    # Agent
    "CommitmentAgent",
    "AgentConfig",
    "AgentState",
    # Protocol
    "CommunicationProtocol",
    "CommitmentProtocol",
    "RepresentationProtocol",
    "ProtocolMetrics",
]
