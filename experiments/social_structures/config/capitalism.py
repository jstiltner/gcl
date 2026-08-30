"""
Configuration for the capitalist meta-network (competition environment).

All internal structures compete within this fixed external environment.
"""

from dataclasses import dataclass


@dataclass
class CapitalistMetaNetwork:
    """The competitive environment firms exist within."""
    
    # Competition
    competition_type: str = "market"
    winner_determination: str = "output_comparison"
    # Firms compete on tasks; higher output wins
    
    # Rewards
    winner_reward: float = 1.0  # Winner gets this
    loser_cost: float = 0.2  # Loser loses this
    market_share_accumulation: bool = True
    # Winners accumulate resources over time
    
    # Failure
    bankruptcy_threshold: float = 0.0  # Resources <= this = death
    bankruptcy_consequence: str = "elimination"
    # No redemption at meta-level (harsh capitalism)
    
    # Knowledge
    inter_firm_sharing: str = "none"  # No sharing between competitors
    ip_protection: bool = True
    # Knowledge dies with bankrupt firms
    
    # No obligations between firms
    inter_firm_obligations: str = "contractual_only"
    # Only explicit contracts, no social obligation


@dataclass
class MarketConditions:
    """
    Vary these to test context-dependence.
    Different market conditions may favor different structures.
    """
    
    # Reward distribution
    winner_take_all: float = 0.9  # Winner gets this fraction (vs distributed)
    # 0.9 = winner-take-all, 0.5 = proportional
    
    # Volatility
    task_difficulty_variance: float = 0.3  # How much difficulty varies
    # High = volatile market, low = stable
    
    # Relationship value
    repeat_partnership_bonus: float = 0.0  # Extra reward for working with same partner
    # 0 = transactional, high = relationship-based
    
    # Market size
    tasks_per_round: int = 10  # How many opportunities exist
    # Low = zero-sum, high = abundance
    
    # Task difficulty
    task_difficulty_mean: float = 0.5  # Average task difficulty


# Pre-defined market condition variants for Phase 4
MARKET_CONDITIONS = {
    "winner_take_all": MarketConditions(winner_take_all=0.95),
    "distributed": MarketConditions(winner_take_all=0.5),
    "volatile": MarketConditions(task_difficulty_variance=0.5),
    "stable": MarketConditions(task_difficulty_variance=0.1),
    "relationship": MarketConditions(repeat_partnership_bonus=0.3),
    "transactional": MarketConditions(repeat_partnership_bonus=0.0),
    "scarce": MarketConditions(tasks_per_round=3),
    "abundant": MarketConditions(tasks_per_round=20),
}
