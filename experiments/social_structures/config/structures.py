"""
Configuration classes for the six social structures.

Each structure encodes different approaches to:
- Reputation (individual vs collective)
- Failure handling (personal vs shared cost)
- Redemption pathways
- Status/mobility
- Obligations
- Knowledge transfer
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class TemplateSharingPolicy(Enum):
    """How knowledge (templates) are shared within a structure."""
    NONE = "none"
    VOLUNTARY = "voluntary"
    WITHIN_GROUP = "within_group"
    CREATES_OBLIGATION = "creates_obligation"
    MANDATORY = "mandatory"
    FULLY_SHARED = "fully_shared"
    HANDOFF = "handoff"
    TEACHING = "teaching"


@dataclass
class MeritocracyConfig:
    """
    Standard Western corporate model.
    Individual performance determines status. Work hard, get ahead.
    """
    
    # Reputation
    reputation_unit: str = "individual"
    reputation_update: str = "personal_performance"
    
    # Failure handling
    failure_cost_bearer: str = "individual"
    failure_penalty: float = 0.3  # Reputation hit
    redemption_pathway: str = "earn_back_via_success"
    redemption_bonus: float = 0.0  # No special bonus for recovery
    
    # Status/mobility
    status_basis: str = "accumulated_reputation"
    mobility: str = "performance_based"
    
    # Obligations
    high_status_obligation: str = "none"
    
    # Knowledge transfer
    template_sharing: TemplateSharingPolicy = TemplateSharingPolicy.NONE
    
    # Success reward
    success_reward: float = 0.1


@dataclass
class GuildConfig:
    """
    Medieval guild / professional association model.
    Collective reputation, apprenticeship progression, mutual protection.
    """
    
    # Reputation
    reputation_unit: str = "guild"  # Collective, not individual
    reputation_update: str = "guild_average_performance"
    
    # Failure handling
    failure_cost_bearer: str = "shared"  # Guild absorbs 50% of cost
    failure_absorption_rate: float = 0.5
    failure_penalty: float = 0.15  # Reduced individual hit (0.3 * 0.5)
    redemption_pathway: str = "guild_supported_recovery"
    redemption_bonus: float = 0.2  # Guild helps you recover
    
    # Status/mobility
    status_basis: str = "tenure_plus_skill"
    mobility: str = "apprentice_journeyman_master"
    progression_time: int = 50  # Rounds to advance a level
    
    # Obligations
    high_status_obligation: str = "mentor_apprentices"
    mentorship_requirement: int = 1  # Must mentor at least 1 lower-status
    
    # Knowledge transfer
    template_sharing: TemplateSharingPolicy = TemplateSharingPolicy.WITHIN_GROUP
    
    # Success reward
    success_reward: float = 0.1


@dataclass
class ObligationNetworkConfig:
    """
    Potlatch / Kula Ring / Guanxi model.
    Status comes from giving, not accumulating. You rise by helping others.
    """
    
    # Reputation
    reputation_unit: str = "relational"  # Network of debts
    reputation_update: str = "obligations_created"
    
    # Failure handling
    failure_cost_bearer: str = "creditor_obligated_to_help"
    failure_penalty: float = 0.1  # Low individual penalty
    redemption_pathway: str = "receive_help_then_pay_forward"
    redemption_bonus: float = 0.3  # Strong recovery bonus
    
    # Status/mobility
    status_basis: str = "outstanding_obligations_owed_to_you"
    mobility: str = "generosity_based"
    
    # Obligations
    high_status_obligation: str = "must_help_debtors"
    hoarding_penalty: float = 0.2  # Lose status if you don't help
    helping_reward: float = 0.3  # Gain status when you help
    
    # Knowledge transfer
    template_sharing: TemplateSharingPolicy = TemplateSharingPolicy.CREATES_OBLIGATION
    
    # Success reward
    success_reward: float = 0.1


@dataclass
class UbuntuConfig:
    """
    Southern African communal philosophy model.
    "I am because we are." Individual identity is collective identity.
    """
    
    # Reputation
    reputation_unit: str = "collective"  # One shared reputation
    reputation_update: str = "collective_performance"
    
    # Failure handling
    failure_cost_bearer: str = "collective"
    failure_penalty_divisor: int = 50  # Distributed across all agents (N)
    failure_penalty: float = 0.006  # 0.3 / 50
    redemption_pathway: str = "community_restoration"
    redemption_bonus: float = 0.2  # Community celebrates recovery
    
    # Status/mobility
    status_basis: str = "contribution_to_collective"
    mobility: str = "recognized_by_community"
    
    # Obligations
    high_status_obligation: str = "serve_collective"
    
    # Knowledge transfer
    template_sharing: TemplateSharingPolicy = TemplateSharingPolicy.FULLY_SHARED
    
    # Success reward
    success_reward: float = 0.1


@dataclass
class RotatingLeadershipConfig:
    """
    Athenian democracy / some indigenous council models.
    No permanent hierarchy. Status is assigned, not earned.
    """
    
    # Reputation
    reputation_unit: str = "minimal"  # Reputation barely matters
    reputation_update: str = "slight_competence_tracking"
    
    # Failure handling
    failure_cost_bearer: str = "role_not_person"
    failure_penalty: float = 0.1  # Minimal personal consequence
    redemption_pathway: str = "automatic_role_rotation"
    redemption_bonus: float = 0.0  # Not needed—role changes anyway
    
    # Status/mobility
    status_basis: str = "lottery_or_rotation"
    mobility: str = "guaranteed_by_design"
    rotation_period: int = 20  # Rounds before roles rotate
    
    # Obligations
    high_status_obligation: str = "serve_during_term"
    
    # Knowledge transfer
    template_sharing: TemplateSharingPolicy = TemplateSharingPolicy.HANDOFF
    
    # Success reward
    success_reward: float = 0.1


@dataclass
class MonasticConfig:
    """
    Benedictine / Buddhist Sangha model.
    Formalized failure and redemption. Humility as virtue.
    """
    
    # Reputation
    reputation_unit: str = "spiritual_standing"
    reputation_update: str = "humility_and_service"
    
    # Failure handling
    failure_cost_bearer: str = "expected_and_ritualized"
    failure_penalty: float = 0.05  # Very low—failure is expected
    redemption_pathway: str = "confession_penance_restoration"
    redemption_ritual: bool = True  # Explicit multi-step process
    redemption_bonus: float = 0.4  # Strong bonus for completing ritual
    
    # Status/mobility
    status_basis: str = "service_and_humility"
    mobility: str = "spiritual_progression"
    
    # Obligations
    high_status_obligation: str = "greater_service"
    accumulation_forbidden: bool = True
    
    # Knowledge transfer
    template_sharing: TemplateSharingPolicy = TemplateSharingPolicy.TEACHING
    
    # Success reward
    success_reward: float = 0.1


def get_config(structure_type: str):
    """Get configuration for a structure type."""
    configs = {
        "meritocracy": MeritocracyConfig(),
        "guild": GuildConfig(),
        "obligation": ObligationNetworkConfig(),
        "ubuntu": UbuntuConfig(),
        "rotating": RotatingLeadershipConfig(),
        "monastic": MonasticConfig(),
    }
    return configs.get(structure_type.lower())
