"""
Commitment templates for reusable commitment patterns.

This module implements Definition 8 from the theoretical framework:
A template T = (τ, α, ψ, Θ) where:
- τ: S → {0, 1} is the trigger schema
- α: S → A is the action schema
- ψ: S × S × A → {0, 1} is the verification schema
- Θ ⊆ S is the validated context region
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator

from gcl.core.commitment import (
    ActionSpec,
    Consequence,
    ContextRegion,
    FailureMode,
    GroundedCommitment,
)
from gcl.core.predicates import Predicate


class TriggerSchema(BaseModel):
    """
    Schema for when a template should be triggered.
    
    Implements τ: S → {0, 1} from Definition 8.
    The trigger schema is a parameterized predicate that determines
    when the template applies to a given state.
    """
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="")
    
    # Predicate expression with placeholders
    expression: str = Field(..., min_length=1)
    
    # Required context keys for evaluation
    required_keys: list[str] = Field(default_factory=list)
    
    # Optional default values for missing keys
    defaults: dict[str, Any] = Field(default_factory=dict)
    
    def evaluate(self, state: dict[str, Any]) -> bool:
        """
        Evaluate whether the trigger condition is met.
        
        Args:
            state: Current state dictionary.
            
        Returns:
            True if the template should trigger.
        """
        # Fill in defaults for missing keys
        eval_state = {**self.defaults, **state}
        
        # Check required keys
        for key in self.required_keys:
            if key not in eval_state:
                return False
        
        # Create predicate and evaluate
        predicate = Predicate(name=self.name, expression=self.expression)
        try:
            return predicate.evaluate(eval_state)
        except Exception:
            return False
    
    def __call__(self, state: dict[str, Any]) -> bool:
        """Shorthand for evaluate."""
        return self.evaluate(state)


class ActionSchema(BaseModel):
    """
    Schema for the action to take when template triggers.
    
    Implements α: S → A from Definition 8.
    The action schema is parameterized by state, allowing
    context-dependent action specification.
    """
    
    action_type: str = Field(..., min_length=1)
    description: str = Field(default="")
    
    # Parameter templates (can reference state variables)
    parameter_templates: dict[str, str] = Field(default_factory=dict)
    
    # Static parameters (constant values)
    static_parameters: dict[str, Any] = Field(default_factory=dict)
    
    # Timeout specification
    timeout_seconds: float | None = Field(default=None, ge=0)
    
    def instantiate(self, state: dict[str, Any]) -> ActionSpec:
        """
        Instantiate the action schema with concrete state.
        
        Args:
            state: Current state dictionary.
            
        Returns:
            Concrete ActionSpec.
        """
        # Start with static parameters
        parameters = dict(self.static_parameters)
        
        # Fill in templated parameters
        for key, template in self.parameter_templates.items():
            # Simple template substitution: {var} -> state[var]
            value = template
            for var_name, var_value in state.items():
                placeholder = f"{{{var_name}}}"
                if placeholder in value:
                    value = value.replace(placeholder, str(var_value))
            parameters[key] = value
        
        return ActionSpec(
            action_type=self.action_type,
            parameters=parameters,
            timeout=timedelta(seconds=self.timeout_seconds) if self.timeout_seconds else None,
        )


class VerificationSchema(BaseModel):
    """
    Schema for verifying commitment success.
    
    Implements ψ: S × S × A → {0, 1} from Definition 8.
    The verification schema defines how to check if the
    commitment was fulfilled.
    """
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="")
    
    # Success condition expression
    success_expression: str = Field(..., min_length=1)
    
    # Failure mode specifications
    failure_modes: list[dict[str, Any]] = Field(default_factory=list)
    
    # Verification function name (from registry)
    verification_fn_name: str | None = Field(default=None)
    
    def create_success_predicate(self) -> Predicate:
        """Create the success condition predicate."""
        return Predicate(name=f"{self.name}_success", expression=self.success_expression)
    
    def create_failure_modes(self) -> list[FailureMode]:
        """Create failure mode objects from specifications."""
        modes = []
        for i, spec in enumerate(self.failure_modes):
            mode = FailureMode(
                name=spec.get("name", f"failure_{i}"),
                condition=Predicate(
                    name=spec.get("name", f"failure_{i}"),
                    expression=spec.get("condition", "False"),
                ),
                consequence=Consequence(
                    consequence_type=spec.get("consequence_type", "penalty"),
                    magnitude=spec.get("magnitude", 0.5),
                    description=spec.get("description") or f"Failure mode {i}",
                ),
                severity=spec.get("severity", 0.5),
                remediation=spec.get("remediation"),
            )
            modes.append(mode)
        return modes


class ContextRegionSpec(BaseModel):
    """
    Specification for the validated context region Θ.
    
    Defines the region of state space where the template
    has been validated to work correctly.
    """
    
    description: str = Field(default="")
    
    # Bounds on continuous variables
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    
    # Allowed values for discrete variables
    allowed_values: dict[str, set[Any]] = Field(default_factory=dict)
    
    # Tags for categorical matching
    tags: set[str] = Field(default_factory=set)
    
    # Embedding for similarity computation
    embedding: list[float] | None = Field(default=None)
    
    @field_validator("allowed_values", mode="before")
    @classmethod
    def convert_allowed_values(cls, v: Any) -> dict[str, set[Any]]:
        """Convert lists to sets for allowed values."""
        if isinstance(v, dict):
            return {k: set(vals) if isinstance(vals, list) else vals for k, vals in v.items()}
        return v
    
    def contains(self, state: dict[str, Any]) -> bool:
        """
        Check if a state is within the validated region.
        
        Args:
            state: State to check.
            
        Returns:
            True if state is in the validated region.
        """
        # Check bounds
        for key, (low, high) in self.bounds.items():
            if key in state:
                value = state[key]
                if not (low <= value <= high):
                    return False
        
        # Check allowed values
        for key, allowed in self.allowed_values.items():
            if key in state:
                if state[key] not in allowed:
                    return False
        
        return True
    
    def similarity(self, state: dict[str, Any]) -> float:
        """
        Compute similarity between state and validated region.
        
        Used for analogical transfer (Theorem 5).
        
        Args:
            state: State to compare.
            
        Returns:
            Similarity score in [0, 1].
        """
        if not self.bounds and not self.allowed_values:
            return 1.0  # No constraints = always similar
        
        scores = []
        
        # Continuous similarity (distance from bounds)
        for key, (low, high) in self.bounds.items():
            if key in state:
                value = state[key]
                if low <= value <= high:
                    scores.append(1.0)
                else:
                    # Distance from nearest bound, normalized
                    dist = min(abs(value - low), abs(value - high))
                    range_size = high - low if high > low else 1.0
                    scores.append(max(0.0, 1.0 - dist / range_size))
        
        # Discrete similarity (exact match)
        for key, allowed in self.allowed_values.items():
            if key in state:
                scores.append(1.0 if state[key] in allowed else 0.0)
        
        return np.mean(scores) if scores else 1.0
    
    def to_context_region(self) -> ContextRegion:
        """Convert to a ContextRegion for commitment creation."""
        return ContextRegion(
            description=self.description or "Template context region",
            bounds=self.bounds,
            allowed_values={k: v for k, v in self.allowed_values.items()},
            tags=self.tags,
            embedding=self.embedding,
        )


class TemplateMetadata(BaseModel):
    """Metadata about a commitment template."""
    
    # Identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, max_length=200)
    version: str = Field(default="1.0.0")
    
    # Provenance
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = Field(default=None)
    source: str = Field(default="manual")  # manual, induced, composed
    
    # Statistics
    usage_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    
    # Categorization
    tags: set[str] = Field(default_factory=set)
    domain: str | None = Field(default=None)
    
    @property
    def success_rate(self) -> float:
        """Compute success rate from usage statistics."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5
    
    def record_usage(self, success: bool) -> None:
        """Record a template usage."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1


@dataclass
class CommitmentTemplate:
    """
    A reusable commitment pattern.
    
    Implements Definition 8: T = (τ, α, ψ, Θ) where:
    - τ (trigger_schema): When does this template apply?
    - α (action_schema): What action to take?
    - ψ (verification_schema): How to verify success?
    - Θ (context_region): Where has this been validated?
    
    Templates enable:
    1. Reuse of successful commitment patterns
    2. Analogical transfer to new contexts (Theorem 5)
    3. Composition into complex behaviors (Theorem 4)
    
    Example:
        >>> template = CommitmentTemplate(
        ...     trigger_schema=TriggerSchema(
        ...         name="task_available",
        ...         expression="task_type == 'classification' and difficulty < 0.7",
        ...     ),
        ...     action_schema=ActionSchema(
        ...         action_type="classify",
        ...         parameter_templates={"input": "{task_input}"},
        ...     ),
        ...     verification_schema=VerificationSchema(
        ...         name="classification_success",
        ...         success_expression="accuracy > 0.8",
        ...     ),
        ...     context_region=ContextRegionSpec(
        ...         bounds={"difficulty": (0.0, 0.7)},
        ...     ),
        ... )
        >>> commitment = template.instantiate(state, issuer="agent-1")
    """
    
    # Core components (Definition 8)
    trigger_schema: TriggerSchema
    action_schema: ActionSchema
    verification_schema: VerificationSchema
    context_region: ContextRegionSpec
    
    # Metadata
    metadata: TemplateMetadata = field(default_factory=lambda: TemplateMetadata(name="unnamed"))
    
    # Default stake and confidence functions
    default_stake: float = 1.0
    default_confidence: float = 0.5
    
    # Stake/confidence adjustment functions (optional)
    stake_fn: Callable[[dict[str, Any]], float] | None = None
    confidence_fn: Callable[[dict[str, Any]], float] | None = None
    
    def triggers(self, state: dict[str, Any]) -> bool:
        """
        Check if the template triggers for a given state.
        
        Args:
            state: Current state dictionary.
            
        Returns:
            True if the template should trigger.
        """
        return self.trigger_schema.evaluate(state)
    
    def is_valid_in_context(self, state: dict[str, Any]) -> bool:
        """
        Check if the state is within the validated context region.
        
        Args:
            state: State to check.
            
        Returns:
            True if state is in validated region.
        """
        return self.context_region.contains(state)
    
    def context_similarity(self, state: dict[str, Any]) -> float:
        """
        Compute similarity between state and validated region.
        
        Used for analogical transfer bound (Theorem 5):
        E[Success] >= confidence * similarity
        
        Args:
            state: State to compare.
            
        Returns:
            Similarity score in [0, 1].
        """
        return self.context_region.similarity(state)
    
    def compute_stake(self, state: dict[str, Any]) -> float:
        """Compute stake for a given state."""
        if self.stake_fn is not None:
            return self.stake_fn(state)
        return self.default_stake
    
    def compute_confidence(self, state: dict[str, Any]) -> float:
        """
        Compute confidence for a given state.
        
        For states outside validated region, confidence is
        scaled by context similarity (Theorem 5).
        """
        base_confidence = self.default_confidence
        if self.confidence_fn is not None:
            base_confidence = self.confidence_fn(state)
        
        # Scale by context similarity for transfer
        if not self.is_valid_in_context(state):
            similarity = self.context_similarity(state)
            base_confidence *= similarity
        
        return np.clip(base_confidence, 0.0, 1.0)
    
    def instantiate(
        self,
        state: dict[str, Any],
        issuer: str,
        stake: float | None = None,
        confidence: float | None = None,
    ) -> GroundedCommitment:
        """
        Instantiate the template into a concrete commitment.
        
        Implements Definition 9: Instantiate(T, s) produces a commitment.
        
        Args:
            state: Current state for parameterization.
            issuer: Agent making the commitment.
            stake: Override stake (uses compute_stake if None).
            confidence: Override confidence (uses compute_confidence if None).
            
        Returns:
            A grounded commitment instance.
        """
        # Compute stake and confidence
        actual_stake = stake if stake is not None else self.compute_stake(state)
        actual_confidence = confidence if confidence is not None else self.compute_confidence(state)
        
        # Create trigger conditions from schema
        trigger_predicate = Predicate(
            name=self.trigger_schema.name,
            expression=self.trigger_schema.expression,
        )
        
        # Create action spec
        action_spec = self.action_schema.instantiate(state)
        
        # Create success condition and failure modes
        success_predicate = self.verification_schema.create_success_predicate()
        failure_modes = self.verification_schema.create_failure_modes()
        
        # Ensure at least one failure mode
        if not failure_modes:
            failure_modes = [
                FailureMode(
                    name="default_failure",
                    condition=Predicate(name="default_fail", expression="True"),
                    consequence=Consequence(
                        consequence_type="penalty",
                        magnitude=0.5,
                        description="Default failure consequence",
                    ),
                    severity=0.5,
                )
            ]
        
        # Create context region
        context = self.context_region.to_context_region()
        
        return GroundedCommitment(
            issuer=issuer,
            trigger_conditions=[trigger_predicate],
            promised_behavior=action_spec,
            success_condition=success_predicate,
            failure_modes=failure_modes,
            stake=actual_stake,
            confidence=actual_confidence,
            valid_contexts=context,
            verification_fn_name=self.verification_schema.verification_fn_name,
            metadata={
                "template_id": self.metadata.id,
                "template_name": self.metadata.name,
                "instantiation_state": state,
            },
        )
    
    def record_outcome(self, success: bool) -> None:
        """Record the outcome of a commitment from this template."""
        self.metadata.record_usage(success)
    
    def expected_success_rate(self, state: dict[str, Any]) -> float:
        """
        Estimate expected success rate for a state.
        
        Implements the analogical transfer bound (Theorem 5):
        E[Success] >= confidence * similarity
        
        Args:
            state: State to estimate for.
            
        Returns:
            Expected success probability.
        """
        confidence = self.compute_confidence(state)
        
        if self.is_valid_in_context(state):
            # In validated region: use historical success rate
            return max(confidence, self.metadata.success_rate)
        else:
            # Outside region: use transfer bound
            similarity = self.context_similarity(state)
            return confidence * similarity
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize template to dictionary."""
        return {
            "trigger_schema": self.trigger_schema.model_dump(),
            "action_schema": self.action_schema.model_dump(),
            "verification_schema": self.verification_schema.model_dump(),
            "context_region": self.context_region.model_dump(),
            "metadata": self.metadata.model_dump(),
            "default_stake": self.default_stake,
            "default_confidence": self.default_confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommitmentTemplate:
        """Deserialize template from dictionary."""
        return cls(
            trigger_schema=TriggerSchema(**data["trigger_schema"]),
            action_schema=ActionSchema(**data["action_schema"]),
            verification_schema=VerificationSchema(**data["verification_schema"]),
            context_region=ContextRegionSpec(**data["context_region"]),
            metadata=TemplateMetadata(**data["metadata"]),
            default_stake=data.get("default_stake", 1.0),
            default_confidence=data.get("default_confidence", 0.5),
        )


def create_simple_template(
    name: str,
    trigger_expression: str,
    action_type: str,
    success_expression: str,
    failure_modes: list[dict[str, Any]] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    default_stake: float = 1.0,
    default_confidence: float = 0.5,
) -> CommitmentTemplate:
    """
    Create a simple commitment template with minimal configuration.
    
    Args:
        name: Template name.
        trigger_expression: When to trigger (predicate expression).
        action_type: Type of action to take.
        success_expression: How to verify success.
        failure_modes: Optional failure mode specifications.
        bounds: Optional context bounds.
        default_stake: Default stake amount.
        default_confidence: Default confidence level.
        
    Returns:
        A configured CommitmentTemplate.
    """
    return CommitmentTemplate(
        trigger_schema=TriggerSchema(
            name=f"{name}_trigger",
            expression=trigger_expression,
        ),
        action_schema=ActionSchema(
            action_type=action_type,
        ),
        verification_schema=VerificationSchema(
            name=f"{name}_verify",
            success_expression=success_expression,
            failure_modes=failure_modes or [],
        ),
        context_region=ContextRegionSpec(
            bounds=bounds or {},
        ),
        metadata=TemplateMetadata(name=name),
        default_stake=default_stake,
        default_confidence=default_confidence,
    )
