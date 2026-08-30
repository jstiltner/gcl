"""
Template induction from commitment experience.

This module provides algorithms for:
1. Inducing templates from successful commitments
2. Generalizing templates across contexts
3. Managing a library of learned templates
4. Analogical transfer to new contexts (Theorem 5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from gcl.core.commitment import GroundedCommitment, VerificationResult, VerificationStatus
from gcl.templates.template import (
    ActionSchema,
    CommitmentTemplate,
    ContextRegionSpec,
    TemplateMetadata,
    TriggerSchema,
    VerificationSchema,
)
from gcl.templates.composition import compute_template_similarity


class InductionConfig(BaseModel):
    """Configuration for template induction."""
    
    # Minimum commitments needed to induce a template
    min_commitments: int = Field(default=5, ge=1)
    
    # Minimum success rate to consider a pattern
    min_success_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Similarity threshold for grouping commitments
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Maximum templates in library
    max_templates: int = Field(default=100, ge=1)
    
    # Confidence decay for unused templates
    confidence_decay: float = Field(default=0.99, ge=0.0, le=1.0)
    
    # Minimum confidence to keep template
    min_confidence: float = Field(default=0.1, ge=0.0, le=1.0)


@dataclass
class CommitmentExperience:
    """Record of a commitment and its outcome."""
    
    commitment: GroundedCommitment
    result: VerificationResult
    context: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success(self) -> bool:
        """Whether the commitment succeeded."""
        return self.result.status == VerificationStatus.SUCCESS


@dataclass
class TemplateCandidate:
    """A candidate template being induced from experience."""
    
    experiences: list[CommitmentExperience] = field(default_factory=list)
    
    # Extracted patterns
    action_type: str | None = None
    trigger_pattern: str | None = None
    success_pattern: str | None = None
    
    # Context bounds (learned from experiences)
    context_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    context_values: dict[str, set[Any]] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Compute success rate from experiences."""
        if not self.experiences:
            return 0.0
        successes = sum(1 for e in self.experiences if e.success)
        return successes / len(self.experiences)
    
    @property
    def count(self) -> int:
        """Number of experiences."""
        return len(self.experiences)
    
    def add_experience(self, exp: CommitmentExperience) -> None:
        """Add an experience and update patterns."""
        self.experiences.append(exp)
        self._update_patterns(exp)
    
    def _update_patterns(self, exp: CommitmentExperience) -> None:
        """Update extracted patterns from new experience."""
        commitment = exp.commitment
        
        # Update action type
        if self.action_type is None:
            self.action_type = commitment.promised_behavior.action_type
        
        # Update trigger pattern (use first trigger condition)
        if self.trigger_pattern is None and commitment.trigger_conditions:
            self.trigger_pattern = commitment.trigger_conditions[0].expression
        
        # Update success pattern
        if self.success_pattern is None:
            self.success_pattern = commitment.success_condition.expression
        
        # Update context bounds
        for key, value in exp.context.items():
            if isinstance(value, (int, float)):
                if key not in self.context_bounds:
                    self.context_bounds[key] = (value, value)
                else:
                    low, high = self.context_bounds[key]
                    self.context_bounds[key] = (min(low, value), max(high, value))
            else:
                if key not in self.context_values:
                    self.context_values[key] = set()
                self.context_values[key].add(value)
    
    def to_template(self, name: str | None = None) -> CommitmentTemplate:
        """Convert candidate to a template."""
        template_name = name or f"induced_{uuid4().hex[:8]}"
        
        # Create trigger schema
        trigger = TriggerSchema(
            name=f"{template_name}_trigger",
            expression=self.trigger_pattern or "True",
        )
        
        # Create action schema
        action = ActionSchema(
            action_type=self.action_type or "unknown",
        )
        
        # Create verification schema
        verification = VerificationSchema(
            name=f"{template_name}_verify",
            success_expression=self.success_pattern or "True",
        )
        
        # Create context region from learned bounds
        context = ContextRegionSpec(
            description=f"Induced from {self.count} experiences",
            bounds=dict(self.context_bounds),
            allowed_values={k: v for k, v in self.context_values.items()},
        )
        
        # Create metadata
        metadata = TemplateMetadata(
            name=template_name,
            source="induced",
            usage_count=self.count,
            success_count=sum(1 for e in self.experiences if e.success),
            failure_count=sum(1 for e in self.experiences if not e.success),
            tags={"induced"},
        )
        
        return CommitmentTemplate(
            trigger_schema=trigger,
            action_schema=action,
            verification_schema=verification,
            context_region=context,
            metadata=metadata,
            default_confidence=self.success_rate,
        )


class TemplateLibrary:
    """
    Library of learned commitment templates.
    
    Manages a collection of templates with:
    - Retrieval by similarity
    - Automatic pruning of low-confidence templates
    - Statistics tracking
    
    Example:
        >>> library = TemplateLibrary()
        >>> library.add(template)
        >>> best_match = library.find_best_match(state)
        >>> if best_match:
        ...     commitment = best_match.instantiate(state, issuer="agent-1")
    """
    
    def __init__(self, config: InductionConfig | None = None) -> None:
        """
        Initialize the template library.
        
        Args:
            config: Configuration for the library.
        """
        self.config = config or InductionConfig()
        self.templates: dict[str, CommitmentTemplate] = {}
        self._usage_timestamps: dict[str, datetime] = {}
    
    def add(self, template: CommitmentTemplate) -> None:
        """
        Add a template to the library.
        
        Args:
            template: Template to add.
        """
        # Check capacity
        if len(self.templates) >= self.config.max_templates:
            self._prune_lowest_confidence()
        
        self.templates[template.metadata.id] = template
        self._usage_timestamps[template.metadata.id] = datetime.now(timezone.utc)
    
    def remove(self, template_id: str) -> bool:
        """
        Remove a template from the library.
        
        Args:
            template_id: ID of template to remove.
            
        Returns:
            True if template was removed.
        """
        if template_id in self.templates:
            del self.templates[template_id]
            self._usage_timestamps.pop(template_id, None)
            return True
        return False
    
    def get(self, template_id: str) -> CommitmentTemplate | None:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def find_matching(
        self,
        state: dict[str, Any],
        min_confidence: float = 0.0,
    ) -> list[CommitmentTemplate]:
        """
        Find all templates that trigger for a state.
        
        Args:
            state: Current state.
            min_confidence: Minimum confidence threshold.
            
        Returns:
            List of matching templates.
        """
        matches = []
        for template in self.templates.values():
            if template.triggers(state):
                confidence = template.compute_confidence(state)
                if confidence >= min_confidence:
                    matches.append(template)
        
        # Sort by confidence (descending)
        matches.sort(key=lambda t: t.compute_confidence(state), reverse=True)
        return matches
    
    def find_best_match(
        self,
        state: dict[str, Any],
        min_confidence: float = 0.0,
    ) -> CommitmentTemplate | None:
        """
        Find the best matching template for a state.
        
        Args:
            state: Current state.
            min_confidence: Minimum confidence threshold.
            
        Returns:
            Best matching template, or None.
        """
        matches = self.find_matching(state, min_confidence)
        if matches:
            # Update usage timestamp
            best = matches[0]
            self._usage_timestamps[best.metadata.id] = datetime.now(timezone.utc)
            return best
        return None
    
    def find_similar(
        self,
        template: CommitmentTemplate,
        threshold: float = 0.5,
    ) -> list[tuple[CommitmentTemplate, float]]:
        """
        Find templates similar to a given template.
        
        Args:
            template: Template to compare against.
            threshold: Minimum similarity threshold.
            
        Returns:
            List of (template, similarity) tuples.
        """
        similar = []
        for t in self.templates.values():
            if t.metadata.id != template.metadata.id:
                sim = compute_template_similarity(template, t)
                if sim >= threshold:
                    similar.append((t, sim))
        
        # Sort by similarity (descending)
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def record_outcome(
        self,
        template_id: str,
        success: bool,
    ) -> None:
        """
        Record the outcome of using a template.
        
        Args:
            template_id: ID of the template used.
            success: Whether the commitment succeeded.
        """
        if template_id in self.templates:
            self.templates[template_id].record_outcome(success)
    
    def apply_decay(self) -> None:
        """Apply confidence decay to unused templates."""
        now = datetime.now(timezone.utc)
        to_remove = []
        
        for template_id, template in self.templates.items():
            last_used = self._usage_timestamps.get(template_id, template.metadata.created_at)
            days_unused = (now - last_used).days
            
            if days_unused > 0:
                # Apply decay
                decay_factor = self.config.confidence_decay ** days_unused
                template.default_confidence *= decay_factor
                
                # Mark for removal if below threshold
                if template.default_confidence < self.config.min_confidence:
                    to_remove.append(template_id)
        
        # Remove low-confidence templates
        for template_id in to_remove:
            self.remove(template_id)
    
    def _prune_lowest_confidence(self) -> None:
        """Remove the template with lowest confidence."""
        if not self.templates:
            return
        
        lowest_id = min(
            self.templates.keys(),
            key=lambda tid: self.templates[tid].default_confidence,
        )
        self.remove(lowest_id)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get library statistics."""
        if not self.templates:
            return {
                "count": 0,
                "mean_confidence": 0.0,
                "mean_success_rate": 0.0,
                "total_usage": 0,
            }
        
        confidences = [t.default_confidence for t in self.templates.values()]
        success_rates = [t.metadata.success_rate for t in self.templates.values()]
        usages = [t.metadata.usage_count for t in self.templates.values()]
        
        return {
            "count": len(self.templates),
            "mean_confidence": np.mean(confidences),
            "mean_success_rate": np.mean(success_rates),
            "total_usage": sum(usages),
            "sources": self._count_sources(),
        }
    
    def _count_sources(self) -> dict[str, int]:
        """Count templates by source."""
        sources: dict[str, int] = {}
        for template in self.templates.values():
            source = template.metadata.source
            sources[source] = sources.get(source, 0) + 1
        return sources
    
    def __len__(self) -> int:
        return len(self.templates)
    
    def __iter__(self):
        return iter(self.templates.values())


class TemplateInducer:
    """
    Induces templates from commitment experience.
    
    The inducer:
    1. Groups similar commitments
    2. Extracts common patterns
    3. Generalizes to templates
    4. Validates templates before adding to library
    
    Example:
        >>> inducer = TemplateInducer(library)
        >>> inducer.observe(commitment, result, context)
        >>> # After enough observations, templates are induced
        >>> new_templates = inducer.induce()
    """
    
    def __init__(
        self,
        library: TemplateLibrary | None = None,
        config: InductionConfig | None = None,
    ) -> None:
        """
        Initialize the template inducer.
        
        Args:
            library: Template library to add induced templates to.
            config: Induction configuration.
        """
        self.library = library or TemplateLibrary()
        self.config = config or InductionConfig()
        
        # Experience buffer
        self.experiences: list[CommitmentExperience] = []
        
        # Candidate templates being built
        self.candidates: dict[str, TemplateCandidate] = {}
    
    def observe(
        self,
        commitment: GroundedCommitment,
        result: VerificationResult,
        context: dict[str, Any],
    ) -> None:
        """
        Observe a commitment outcome.
        
        Args:
            commitment: The commitment that was made.
            result: Verification result.
            context: Context in which commitment was made.
        """
        exp = CommitmentExperience(
            commitment=commitment,
            result=result,
            context=context,
        )
        self.experiences.append(exp)
        
        # Add to appropriate candidate
        candidate_key = self._get_candidate_key(commitment)
        if candidate_key not in self.candidates:
            self.candidates[candidate_key] = TemplateCandidate()
        self.candidates[candidate_key].add_experience(exp)
    
    def _get_candidate_key(self, commitment: GroundedCommitment) -> str:
        """Get the key for grouping commitments into candidates."""
        # Group by action type and success condition
        action_type = commitment.promised_behavior.action_type
        success_expr = commitment.success_condition.expression
        return f"{action_type}::{success_expr}"
    
    def induce(self) -> list[CommitmentTemplate]:
        """
        Induce templates from accumulated experience.
        
        Returns:
            List of newly induced templates.
        """
        induced = []
        
        for key, candidate in list(self.candidates.items()):
            # Check if candidate meets criteria
            if candidate.count < self.config.min_commitments:
                continue
            if candidate.success_rate < self.config.min_success_rate:
                continue
            
            # Convert to template
            template = candidate.to_template()
            
            # Check if similar template already exists
            similar = self.library.find_similar(template, self.config.similarity_threshold)
            if similar:
                # Merge with existing template
                existing, sim = similar[0]
                self._merge_template(existing, candidate)
            else:
                # Add new template
                self.library.add(template)
                induced.append(template)
            
            # Clear candidate
            del self.candidates[key]
        
        return induced
    
    def _merge_template(
        self,
        existing: CommitmentTemplate,
        candidate: TemplateCandidate,
    ) -> None:
        """Merge candidate experience into existing template."""
        # Update statistics
        for exp in candidate.experiences:
            existing.record_outcome(exp.success)
        
        # Expand context bounds
        for key, (low, high) in candidate.context_bounds.items():
            if key in existing.context_region.bounds:
                old_low, old_high = existing.context_region.bounds[key]
                existing.context_region.bounds[key] = (
                    min(old_low, low),
                    max(old_high, high),
                )
            else:
                existing.context_region.bounds[key] = (low, high)
        
        # Expand allowed values
        for key, values in candidate.context_values.items():
            if key in existing.context_region.allowed_values:
                existing.context_region.allowed_values[key] |= values
            else:
                existing.context_region.allowed_values[key] = values
    
    def get_statistics(self) -> dict[str, Any]:
        """Get inducer statistics."""
        return {
            "total_experiences": len(self.experiences),
            "active_candidates": len(self.candidates),
            "candidate_sizes": {k: c.count for k, c in self.candidates.items()},
            "library_size": len(self.library),
        }


def induce_template_from_commitments(
    commitments: list[tuple[GroundedCommitment, VerificationResult, dict[str, Any]]],
    name: str | None = None,
    min_success_rate: float = 0.5,
) -> CommitmentTemplate | None:
    """
    Induce a single template from a list of commitments.
    
    Convenience function for one-shot template induction.
    
    Args:
        commitments: List of (commitment, result, context) tuples.
        name: Optional template name.
        min_success_rate: Minimum success rate to create template.
        
    Returns:
        Induced template, or None if criteria not met.
    """
    if not commitments:
        return None
    
    candidate = TemplateCandidate()
    for commitment, result, context in commitments:
        exp = CommitmentExperience(
            commitment=commitment,
            result=result,
            context=context,
        )
        candidate.add_experience(exp)
    
    if candidate.success_rate < min_success_rate:
        return None
    
    return candidate.to_template(name)
