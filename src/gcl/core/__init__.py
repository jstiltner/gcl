"""
GCL Core Module

Contains the fundamental abstractions for grounded commitments.
"""

from gcl.core.commitment import (
    ActionSpec,
    CommitmentPortfolio,
    Consequence,
    ContextRegion,
    FailureMode,
    GroundedCommitment,
    VerificationResult,
    VerificationStatus,
)
from gcl.core.predicates import Predicate
from gcl.core.registry import VerificationRegistry, verification_registry
from gcl.core.reputation import (
    AgentReputation,
    ReputationConfig,
    ReputationEvent,
    ReputationEventType,
    ReputationTracker,
    StakeAccount,
    StakeConfig,
    StakeManager,
    StakeTransaction,
)
from gcl.core.verification import (
    BatchVerificationResult,
    VerificationContext,
    VerificationEngine,
    VerificationReport,
)

__all__ = [
    # Commitment types
    "ActionSpec",
    "CommitmentPortfolio",
    "Consequence",
    "ContextRegion",
    "FailureMode",
    "GroundedCommitment",
    "Predicate",
    "VerificationResult",
    "VerificationStatus",
    # Registry
    "VerificationRegistry",
    "verification_registry",
    # Verification
    "BatchVerificationResult",
    "VerificationContext",
    "VerificationEngine",
    "VerificationReport",
    # Reputation
    "AgentReputation",
    "ReputationConfig",
    "ReputationEvent",
    "ReputationEventType",
    "ReputationTracker",
    # Stake
    "StakeAccount",
    "StakeConfig",
    "StakeManager",
    "StakeTransaction",
]
