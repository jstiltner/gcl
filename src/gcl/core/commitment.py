"""
Core commitment dataclasses for the GCL framework.

This module defines the fundamental data structures for grounded commitments,
including failure modes, verification results, and commitment portfolios.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from gcl.core.predicates import Predicate


class VerificationStatus(str, Enum):
    """Status of a commitment verification."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class ActionSpec(BaseModel):
    """
    Specification of an action to be performed.
    
    Actions are the promised behaviors in a commitment. They specify
    what the agent commits to doing when trigger conditions are met.
    
    Attributes:
        action_type: Type/category of the action (e.g., "respond", "delegate", "compute").
        parameters: Action-specific parameters as key-value pairs.
        timeout_seconds: Maximum time allowed for action completion.
        description: Human-readable description of the action.
        
    Example:
        >>> action = ActionSpec(
        ...     action_type="respond",
        ...     parameters={"format": "json", "max_tokens": 1000},
        ...     timeout_seconds=30.0,
        ...     description="Generate a JSON response"
        ... )
    """

    action_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type/category of the action",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters",
    )
    timeout_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Maximum time allowed for action completion",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Human-readable description of the action",
    )

    model_config = ConfigDict(frozen=True)


class Consequence(BaseModel):
    """
    The consequence of a failure mode.
    
    Consequences define what happens when a commitment fails in a specific way.
    They are used to compute rewards/penalties and guide remediation.
    
    Attributes:
        consequence_type: Category of consequence (e.g., "stake_slash", "reputation_loss").
        magnitude: Severity of the consequence (0.0 to 1.0).
        description: Human-readable description.
        metadata: Additional consequence-specific data.
        
    Example:
        >>> consequence = Consequence(
        ...     consequence_type="stake_slash",
        ...     magnitude=0.5,
        ...     description="Lose 50% of staked amount"
        ... )
    """

    consequence_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Category of consequence",
    )
    magnitude: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Severity of the consequence (0.0 to 1.0)",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable description",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional consequence-specific data",
    )

    model_config = ConfigDict(frozen=True)


class FailureMode(BaseModel):
    """
    A specific way a commitment can fail.
    
    Commitments are written FAILURE-FIRST: they enumerate all the ways
    they can fail, with consequences for each. Success is whatever
    remains after all failure conditions are excluded.
    
    Attributes:
        id: Unique identifier for this failure mode.
        name: Human-readable name.
        condition: Predicate that triggers this failure mode.
        consequence: What happens if this failure occurs.
        severity: How bad this failure is (0.0 to 1.0).
        remediation: Optional action to recover from this failure.
        
    Example:
        >>> failure = FailureMode(
        ...     name="timeout",
        ...     condition=Predicate(name="timeout", expression="elapsed_time > timeout_limit"),
        ...     consequence=Consequence(
        ...         consequence_type="stake_slash",
        ...         magnitude=0.3,
        ...         description="Partial stake loss for timeout"
        ...     ),
        ...     severity=0.3,
        ... )
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this failure mode",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name",
    )
    condition: Predicate = Field(
        ...,
        description="Predicate that triggers this failure mode",
    )
    consequence: Consequence = Field(
        ...,
        description="What happens if this failure occurs",
    )
    severity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How bad this failure is (0.0 to 1.0)",
    )
    remediation: ActionSpec | None = Field(
        default=None,
        description="Optional action to recover from this failure",
    )

    model_config = ConfigDict(frozen=True)

    def is_triggered(self, context: dict[str, Any]) -> bool:
        """Check if this failure mode is triggered by the given context."""
        return self.condition.evaluate(context)


class ContextRegion(BaseModel):
    """
    Defines the valid context bounds for a commitment.
    
    Context regions specify where a commitment is valid. This enables
    template matching and transfer learning across similar contexts.
    
    Attributes:
        description: Human-readable description of the context.
        embedding: Optional vector embedding for similarity matching.
        bounds: Dictionary of context constraints (e.g., {"temperature": (0, 100)}).
        tags: List of tags for categorization.
        
    Example:
        >>> context = ContextRegion(
        ...     description="Question-answering tasks",
        ...     tags=["qa", "text"],
        ...     bounds={"max_tokens": (0, 4096)}
        ... )
    """

    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable description of the context",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Optional vector embedding for similarity matching",
    )
    bounds: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of context constraints",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for categorization",
    )

    model_config = ConfigDict(frozen=True)

    def matches_context(self, context: dict[str, Any]) -> bool:
        """
        Check if a context falls within this region's bounds.
        
        Args:
            context: The context to check.
            
        Returns:
            True if the context matches all bounds, False otherwise.
        """
        for key, bound in self.bounds.items():
            if key not in context:
                continue  # Missing keys don't violate bounds

            value = context[key]

            # Handle tuple bounds (min, max)
            if isinstance(bound, (list, tuple)) and len(bound) == 2:
                min_val, max_val = bound
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
            # Handle set bounds (allowed values)
            elif isinstance(bound, set):
                if value not in bound:
                    return False
            # Handle exact match
            else:
                if value != bound:
                    return False

        return True


class VerificationResult(BaseModel):
    """
    Result of verifying a commitment.
    
    Verification results capture the outcome of checking whether a
    commitment was fulfilled, including any triggered failure modes.
    
    Attributes:
        status: The verification status (success, failure, etc.).
        commitment_id: ID of the commitment that was verified.
        triggered_failure_mode: ID of the failure mode that was triggered (if any).
        details: Additional verification details.
        timestamp: When the verification occurred.
        verification_duration_ms: How long verification took.
        
    Example:
        >>> result = VerificationResult(
        ...     status=VerificationStatus.SUCCESS,
        ...     commitment_id="abc-123",
        ...     details={"accuracy": 0.95}
        ... )
    """

    status: VerificationStatus = Field(
        ...,
        description="The verification status",
    )
    commitment_id: str = Field(
        ...,
        description="ID of the commitment that was verified",
    )
    triggered_failure_mode: str | None = Field(
        default=None,
        description="ID of the failure mode that was triggered (if any)",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional verification details",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the verification occurred",
    )
    verification_duration_ms: float | None = Field(
        default=None,
        ge=0,
        description="How long verification took in milliseconds",
    )

    model_config = ConfigDict(frozen=True)

    @property
    def is_success(self) -> bool:
        """Check if verification was successful."""
        return self.status == VerificationStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Check if verification resulted in failure."""
        return self.status == VerificationStatus.FAILURE


class GroundedCommitment(BaseModel):
    """
    A verifiable behavioral contract.
    
    Grounded commitments are the core abstraction in GCL. They represent
    promises that an agent makes about its behavior, with explicit
    trigger conditions, success criteria, and failure modes.
    
    Key principles:
    - FAILURE-FIRST: Enumerate failure modes explicitly. Success is the complement.
    - VERIFIABLE: All conditions can be checked against observable state.
    - STAKEABLE: Agents put stake behind their commitments.
    
    Attributes:
        id: Unique identifier for this commitment.
        issuer: ID of the agent making this commitment.
        trigger_conditions: Conditions that must hold for commitment to activate.
        promised_behavior: The action the agent commits to performing.
        success_condition: Predicate defining successful completion.
        failure_modes: List of ways this commitment can fail (sorted by severity).
        stake: Amount the agent stakes on this commitment.
        confidence: Agent's confidence in fulfilling this commitment (0.0 to 1.0).
        valid_contexts: Context region where this commitment is valid.
        verification_fn_name: Name of registered verification function.
        created_at: When this commitment was created.
        expires_at: When this commitment expires (optional).
        metadata: Additional commitment-specific data.
        
    Example:
        >>> commitment = GroundedCommitment(
        ...     issuer="agent-1",
        ...     trigger_conditions=[
        ...         Predicate(name="is_question", expression="query_type == 'question'")
        ...     ],
        ...     promised_behavior=ActionSpec(action_type="answer", parameters={}),
        ...     success_condition=Predicate(name="accurate", expression="accuracy > 0.8"),
        ...     failure_modes=[
        ...         FailureMode(
        ...             name="inaccurate",
        ...             condition=Predicate(name="low_accuracy", expression="accuracy <= 0.8"),
        ...             consequence=Consequence(
        ...                 consequence_type="stake_slash",
        ...                 magnitude=0.5,
        ...                 description="Lose half stake for inaccuracy"
        ...             ),
        ...             severity=0.5,
        ...         )
        ...     ],
        ...     stake=1.0,
        ...     confidence=0.85,
        ...     valid_contexts=ContextRegion(description="General QA"),
        ... )
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this commitment",
    )
    issuer: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ID of the agent making this commitment",
    )
    trigger_conditions: list[Predicate] = Field(
        ...,
        min_length=1,
        description="Conditions that must hold for commitment to activate",
    )
    promised_behavior: ActionSpec = Field(
        ...,
        description="The action the agent commits to performing",
    )
    success_condition: Predicate = Field(
        ...,
        description="Predicate defining successful completion",
    )
    failure_modes: list[FailureMode] = Field(
        ...,
        min_length=1,
        description="List of ways this commitment can fail",
    )
    stake: float = Field(
        ...,
        ge=0.0,
        description="Amount the agent stakes on this commitment",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in fulfilling this commitment",
    )
    valid_contexts: ContextRegion = Field(
        ...,
        description="Context region where this commitment is valid",
    )
    verification_fn_name: str | None = Field(
        default=None,
        description="Name of registered verification function",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this commitment was created",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When this commitment expires",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional commitment-specific data",
    )

    @field_validator("failure_modes")
    @classmethod
    def sort_failure_modes_by_severity(cls, v: list[FailureMode]) -> list[FailureMode]:
        """Ensure failure modes are sorted by severity (descending)."""
        return sorted(v, key=lambda fm: fm.severity, reverse=True)

    @model_validator(mode="after")
    def validate_expiration(self) -> "GroundedCommitment":
        """Ensure expires_at is after created_at if set."""
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self

    def is_expired(self) -> bool:
        """Check if this commitment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def triggers_match(self, context: dict[str, Any]) -> bool:
        """
        Check if all trigger conditions are met.
        
        Args:
            context: The context to check against trigger conditions.
            
        Returns:
            True if all trigger conditions are satisfied.
        """
        return all(pred.evaluate(context) for pred in self.trigger_conditions)

    def is_valid_in_context(self, context: dict[str, Any]) -> bool:
        """
        Check if this commitment is valid in the given context.
        
        Args:
            context: The context to check.
            
        Returns:
            True if the context falls within valid_contexts bounds.
        """
        return self.valid_contexts.matches_context(context)

    def check_failure_modes(self, context: dict[str, Any]) -> FailureMode | None:
        """
        Check if any failure mode is triggered.
        
        Failure modes are checked in order of severity (highest first).
        
        Args:
            context: The context to check against failure conditions.
            
        Returns:
            The first triggered failure mode, or None if no failures.
        """
        for failure_mode in self.failure_modes:
            if failure_mode.is_triggered(context):
                return failure_mode
        return None

    def check_success(self, context: dict[str, Any]) -> bool:
        """
        Check if the success condition is met.
        
        Args:
            context: The context to check.
            
        Returns:
            True if the success condition is satisfied.
        """
        return self.success_condition.evaluate(context)

    def get_verification_fn(self) -> Any:
        """
        Get the verification function from the registry.
        
        Returns:
            The verification function, or None if not set or not found.
        """
        if self.verification_fn_name is None:
            return None

        from gcl.core.registry import verification_registry

        return verification_registry.get(self.verification_fn_name)

    def expected_value(self) -> float:
        """
        Calculate the expected value of this commitment.
        
        Expected value = confidence * stake - (1 - confidence) * expected_loss
        
        Returns:
            The expected value of making this commitment.
        """
        # Calculate expected loss from failure modes
        # Weight by severity (assuming uniform probability across failure modes)
        if not self.failure_modes:
            expected_loss = 0.0
        else:
            avg_severity = sum(fm.severity for fm in self.failure_modes) / len(
                self.failure_modes
            )
            expected_loss = self.stake * avg_severity

        return self.confidence * self.stake - (1 - self.confidence) * expected_loss


class CommitmentPortfolio(BaseModel):
    """
    A collection of commitments with validation and management.
    
    Portfolios allow agents to manage multiple active commitments,
    track total stake exposure, and enforce resource constraints.
    
    Attributes:
        id: Unique identifier for this portfolio.
        owner: ID of the agent that owns this portfolio.
        commitments: List of commitments in the portfolio.
        created_at: When this portfolio was created.
        max_total_stake: Maximum allowed total stake (optional).
        
    Example:
        >>> portfolio = CommitmentPortfolio(
        ...     owner="agent-1",
        ...     max_total_stake=10.0
        ... )
        >>> portfolio.add_commitment(commitment)
        >>> print(portfolio.total_stake)
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this portfolio",
    )
    owner: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ID of the agent that owns this portfolio",
    )
    commitments: list[GroundedCommitment] = Field(
        default_factory=list,
        description="List of commitments in the portfolio",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this portfolio was created",
    )
    max_total_stake: float | None = Field(
        default=None,
        ge=0,
        description="Maximum allowed total stake",
    )

    @property
    def total_stake(self) -> float:
        """Calculate the total stake across all commitments."""
        return sum(c.stake for c in self.commitments)

    @property
    def active_commitments(self) -> list[GroundedCommitment]:
        """Get commitments that haven't expired."""
        return [c for c in self.commitments if not c.is_expired()]

    @property
    def expired_commitments(self) -> list[GroundedCommitment]:
        """Get commitments that have expired."""
        return [c for c in self.commitments if c.is_expired()]

    @property
    def average_confidence(self) -> float:
        """Calculate the average confidence across all commitments."""
        if not self.commitments:
            return 0.0
        return sum(c.confidence for c in self.commitments) / len(self.commitments)

    @property
    def expected_portfolio_value(self) -> float:
        """Calculate the total expected value of the portfolio."""
        return sum(c.expected_value() for c in self.commitments)

    def add_commitment(self, commitment: GroundedCommitment) -> None:
        """
        Add a commitment to the portfolio.
        
        Args:
            commitment: The commitment to add.
            
        Raises:
            ValueError: If adding would exceed max_total_stake.
        """
        if self.max_total_stake is not None:
            new_total = self.total_stake + commitment.stake
            if new_total > self.max_total_stake:
                raise ValueError(
                    f"Adding commitment would exceed max stake "
                    f"({new_total:.2f} > {self.max_total_stake:.2f})"
                )
        self.commitments.append(commitment)

    def remove_commitment(self, commitment_id: str) -> bool:
        """
        Remove a commitment by ID.
        
        Args:
            commitment_id: The ID of the commitment to remove.
            
        Returns:
            True if the commitment was found and removed, False otherwise.
        """
        for i, c in enumerate(self.commitments):
            if c.id == commitment_id:
                self.commitments.pop(i)
                return True
        return False

    def get_commitment(self, commitment_id: str) -> GroundedCommitment | None:
        """
        Get a commitment by ID.
        
        Args:
            commitment_id: The ID of the commitment to retrieve.
            
        Returns:
            The commitment if found, None otherwise.
        """
        for c in self.commitments:
            if c.id == commitment_id:
                return c
        return None

    def get_commitments_by_issuer(self, issuer: str) -> list[GroundedCommitment]:
        """Get all commitments from a specific issuer."""
        return [c for c in self.commitments if c.issuer == issuer]

    def get_commitments_matching_context(
        self, context: dict[str, Any]
    ) -> list[GroundedCommitment]:
        """Get all commitments that are valid in the given context."""
        return [c for c in self.commitments if c.is_valid_in_context(context)]

    def get_triggered_commitments(
        self, context: dict[str, Any]
    ) -> list[GroundedCommitment]:
        """Get all commitments whose trigger conditions are met."""
        return [
            c
            for c in self.active_commitments
            if c.triggers_match(context) and c.is_valid_in_context(context)
        ]

    def prune_expired(self) -> list[GroundedCommitment]:
        """
        Remove all expired commitments from the portfolio.
        
        Returns:
            List of removed commitments.
        """
        expired = self.expired_commitments
        self.commitments = self.active_commitments
        return expired

    def validate_stake_budget(self, additional_stake: float = 0.0) -> bool:
        """
        Check if the portfolio is within stake budget.
        
        Args:
            additional_stake: Additional stake to consider.
            
        Returns:
            True if within budget, False otherwise.
        """
        if self.max_total_stake is None:
            return True
        return self.total_stake + additional_stake <= self.max_total_stake

    def remaining_stake_budget(self) -> float | None:
        """
        Calculate remaining stake budget.
        
        Returns:
            Remaining budget, or None if no max is set.
        """
        if self.max_total_stake is None:
            return None
        return max(0.0, self.max_total_stake - self.total_stake)

    def __len__(self) -> int:
        """Return the number of commitments in the portfolio."""
        return len(self.commitments)

    def __iter__(self):
        """Iterate over commitments in the portfolio."""
        return iter(self.commitments)

    def __contains__(self, commitment_id: str) -> bool:
        """Check if a commitment ID is in the portfolio."""
        return any(c.id == commitment_id for c in self.commitments)
