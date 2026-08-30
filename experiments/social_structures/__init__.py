"""
Social Structures Experiment

Tests which internal organizational structure produces optimal coordination
when competing within a capitalist meta-network.

Structures tested:
1. Meritocracy - Individual performance determines status
2. Guild - Collective reputation, apprenticeship progression
3. ObligationNetwork - Status from giving, not accumulating
4. Ubuntu - Collective identity, shared reputation
5. RotatingLeadership - No permanent hierarchy
6. Monastic - Formalized failure and redemption
"""

from .agents.agent import Agent, Task, TaskOutcome, Template, create_agent, create_population
from .structures.implementations import (
    Meritocracy, Guild, ObligationNetwork, Ubuntu, 
    RotatingLeadership, Monastic, create_structure
)
from .competition.firm import Firm, FirmTaskOutcome, create_firm
from .competition.arena import CapitalistArena, SimulationResult
from .config.structures import (
    MeritocracyConfig, GuildConfig, ObligationNetworkConfig,
    UbuntuConfig, RotatingLeadershipConfig, MonasticConfig
)
from .config.capitalism import CapitalistMetaNetwork, MarketConditions

__all__ = [
    # Agents
    "Agent", "Task", "TaskOutcome", "Template", "create_agent", "create_population",
    # Structures
    "Meritocracy", "Guild", "ObligationNetwork", "Ubuntu", 
    "RotatingLeadership", "Monastic", "create_structure",
    # Competition
    "Firm", "FirmTaskOutcome", "create_firm",
    "CapitalistArena", "SimulationResult",
    # Config
    "MeritocracyConfig", "GuildConfig", "ObligationNetworkConfig",
    "UbuntuConfig", "RotatingLeadershipConfig", "MonasticConfig",
    "CapitalistMetaNetwork", "MarketConditions",
]
