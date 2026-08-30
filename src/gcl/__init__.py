"""
Grounded Commitment Learning (GCL)

A framework for AI alignment through verifiable behavioral contracts.
"""

__version__ = "0.1.0"

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
from gcl.core.registry import verification_registry
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
    # Core types
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
