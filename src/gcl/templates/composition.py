"""
Template composition operators.

This module implements Definition 11 from the theoretical framework:
- Sequential composition: T₁ ; T₂
- Parallel composition: T₁ ‖ T₂
- Conditional composition: T₁ ◁p▷ T₂

Theorem 4 (Template Closure): The set of templates is closed under
sequential, parallel, and conditional composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from gcl.templates.template import (
    ActionSchema,
    CommitmentTemplate,
    ContextRegionSpec,
    TemplateMetadata,
    TriggerSchema,
    VerificationSchema,
)


class CompositionType(str, Enum):
    """Types of template composition."""
    
    SEQUENTIAL = "sequential"  # T₁ ; T₂
    PARALLEL = "parallel"      # T₁ ‖ T₂
    CONDITIONAL = "conditional"  # T₁ ◁p▷ T₂


@dataclass
class ComposedTemplate:
    """
    A template composed from other templates.
    
    Tracks the composition structure for analysis and debugging.
    """
    
    template: CommitmentTemplate
    composition_type: CompositionType
    components: list[CommitmentTemplate]
    predicate: Callable[[dict[str, Any]], bool] | None = None  # For conditional


class TemplateComposer:
    """
    Implements template algebra from Definition 11.
    
    Provides operators for composing templates:
    - sequential: Execute T₁, then T₂ on result
    - parallel: Execute T₁ and T₂ simultaneously
    - conditional: Choose T₁ or T₂ based on predicate
    
    Theorem 4 guarantees closure: composed templates are valid templates.
    
    Example:
        >>> composer = TemplateComposer()
        >>> t_combined = composer.sequential(t_fetch, t_process)
        >>> t_choice = composer.conditional(t_fast, t_accurate, lambda s: s["time_critical"])
    """
    
    @staticmethod
    def sequential(
        t1: CommitmentTemplate,
        t2: CommitmentTemplate,
        name: str | None = None,
    ) -> CommitmentTemplate:
        """
        Sequential composition: T₁ ; T₂
        
        Execute t1, then t2 on the result. Both must succeed for
        the composed commitment to succeed.
        
        From Definition 11:
        - τ_{T₁;T₂}(s) = τ₁(s)
        - α_{T₁;T₂}(s) = α₂(result(α₁(s)))
        - ψ_{T₁;T₂}(s, s', a) = ψ₁(s, s_mid, a₁) ∧ ψ₂(s_mid, s', a₂)
        
        Args:
            t1: First template to execute.
            t2: Second template to execute on t1's result.
            name: Optional name for composed template.
            
        Returns:
            Composed template.
        """
        composed_name = name or f"{t1.metadata.name}_then_{t2.metadata.name}"
        
        # Trigger: use t1's trigger (t2 triggers on t1's completion)
        trigger = TriggerSchema(
            name=f"{composed_name}_trigger",
            description=f"Sequential: {t1.trigger_schema.name} then {t2.trigger_schema.name}",
            expression=t1.trigger_schema.expression,
            required_keys=list(set(t1.trigger_schema.required_keys + t2.trigger_schema.required_keys)),
            defaults={**t1.trigger_schema.defaults, **t2.trigger_schema.defaults},
        )
        
        # Action: combined action type
        action = ActionSchema(
            action_type=f"sequential({t1.action_schema.action_type}, {t2.action_schema.action_type})",
            description=f"Execute {t1.action_schema.action_type} then {t2.action_schema.action_type}",
            parameter_templates={
                **t1.action_schema.parameter_templates,
                **{f"phase2_{k}": v for k, v in t2.action_schema.parameter_templates.items()},
            },
            static_parameters={
                "phase1": t1.action_schema.static_parameters,
                "phase2": t2.action_schema.static_parameters,
            },
        )
        
        # Verification: both must succeed
        # Combined expression: phase1_success and phase2_success
        verification = VerificationSchema(
            name=f"{composed_name}_verify",
            description=f"Both {t1.verification_schema.name} and {t2.verification_schema.name} must succeed",
            success_expression=f"({t1.verification_schema.success_expression}) and ({t2.verification_schema.success_expression})",
            failure_modes=[
                {
                    "name": f"phase1_failure_{fm.get('name', i)}",
                    "condition": fm.get("condition", "False"),
                    "consequence_type": fm.get("consequence_type", "penalty"),
                    "magnitude": fm.get("magnitude", 0.5),
                    "severity": fm.get("severity", 0.5),
                    "description": f"Phase 1: {fm.get('description', '')}",
                }
                for i, fm in enumerate(t1.verification_schema.failure_modes)
            ] + [
                {
                    "name": f"phase2_failure_{fm.get('name', i)}",
                    "condition": fm.get("condition", "False"),
                    "consequence_type": fm.get("consequence_type", "penalty"),
                    "magnitude": fm.get("magnitude", 0.5),
                    "severity": fm.get("severity", 0.5),
                    "description": f"Phase 2: {fm.get('description', '')}",
                }
                for i, fm in enumerate(t2.verification_schema.failure_modes)
            ],
        )
        
        # Context: intersection of validated regions
        context = ContextRegionSpec(
            description=f"Intersection of {t1.context_region.description} and {t2.context_region.description}",
            bounds={**t1.context_region.bounds, **t2.context_region.bounds},
            allowed_values={**t1.context_region.allowed_values, **t2.context_region.allowed_values},
            tags=t1.context_region.tags & t2.context_region.tags,
        )
        
        # Metadata
        metadata = TemplateMetadata(
            name=composed_name,
            source="composed",
            tags=t1.metadata.tags | t2.metadata.tags | {"sequential"},
        )
        
        return CommitmentTemplate(
            trigger_schema=trigger,
            action_schema=action,
            verification_schema=verification,
            context_region=context,
            metadata=metadata,
            default_stake=t1.default_stake + t2.default_stake,
            default_confidence=t1.default_confidence * t2.default_confidence,  # Both must succeed
        )
    
    @staticmethod
    def parallel(
        t1: CommitmentTemplate,
        t2: CommitmentTemplate,
        name: str | None = None,
    ) -> CommitmentTemplate:
        """
        Parallel composition: T₁ ‖ T₂
        
        Execute t1 and t2 simultaneously. Both must succeed for
        the composed commitment to succeed.
        
        From Definition 11:
        - τ_{T₁‖T₂}(s) = τ₁(s) ∧ τ₂(s)
        - α_{T₁‖T₂}(s) = (α₁(s), α₂(s))
        - ψ_{T₁‖T₂}(s, s', a) = ψ₁(s, s', a₁) ∧ ψ₂(s, s', a₂)
        
        Args:
            t1: First template.
            t2: Second template.
            name: Optional name for composed template.
            
        Returns:
            Composed template.
        """
        composed_name = name or f"{t1.metadata.name}_parallel_{t2.metadata.name}"
        
        # Trigger: both must trigger
        trigger = TriggerSchema(
            name=f"{composed_name}_trigger",
            description=f"Parallel: {t1.trigger_schema.name} and {t2.trigger_schema.name}",
            expression=f"({t1.trigger_schema.expression}) and ({t2.trigger_schema.expression})",
            required_keys=list(set(t1.trigger_schema.required_keys + t2.trigger_schema.required_keys)),
            defaults={**t1.trigger_schema.defaults, **t2.trigger_schema.defaults},
        )
        
        # Action: parallel action type
        action = ActionSchema(
            action_type=f"parallel({t1.action_schema.action_type}, {t2.action_schema.action_type})",
            description=f"Execute {t1.action_schema.action_type} and {t2.action_schema.action_type} in parallel",
            parameter_templates={
                **{f"branch1_{k}": v for k, v in t1.action_schema.parameter_templates.items()},
                **{f"branch2_{k}": v for k, v in t2.action_schema.parameter_templates.items()},
            },
            static_parameters={
                "branch1": t1.action_schema.static_parameters,
                "branch2": t2.action_schema.static_parameters,
            },
        )
        
        # Verification: both must succeed
        verification = VerificationSchema(
            name=f"{composed_name}_verify",
            description=f"Both {t1.verification_schema.name} and {t2.verification_schema.name} must succeed",
            success_expression=f"({t1.verification_schema.success_expression}) and ({t2.verification_schema.success_expression})",
            failure_modes=[
                {
                    "name": f"branch1_failure_{fm.get('name', i)}",
                    "condition": fm.get("condition", "False"),
                    "consequence_type": fm.get("consequence_type", "penalty"),
                    "magnitude": fm.get("magnitude", 0.5),
                    "severity": fm.get("severity", 0.5),
                    "description": f"Branch 1: {fm.get('description', '')}",
                }
                for i, fm in enumerate(t1.verification_schema.failure_modes)
            ] + [
                {
                    "name": f"branch2_failure_{fm.get('name', i)}",
                    "condition": fm.get("condition", "False"),
                    "consequence_type": fm.get("consequence_type", "penalty"),
                    "magnitude": fm.get("magnitude", 0.5),
                    "severity": fm.get("severity", 0.5),
                    "description": f"Branch 2: {fm.get('description', '')}",
                }
                for i, fm in enumerate(t2.verification_schema.failure_modes)
            ],
        )
        
        # Context: intersection of validated regions
        context = ContextRegionSpec(
            description=f"Intersection of {t1.context_region.description} and {t2.context_region.description}",
            bounds={**t1.context_region.bounds, **t2.context_region.bounds},
            allowed_values={**t1.context_region.allowed_values, **t2.context_region.allowed_values},
            tags=t1.context_region.tags & t2.context_region.tags,
        )
        
        # Metadata
        metadata = TemplateMetadata(
            name=composed_name,
            source="composed",
            tags=t1.metadata.tags | t2.metadata.tags | {"parallel"},
        )
        
        return CommitmentTemplate(
            trigger_schema=trigger,
            action_schema=action,
            verification_schema=verification,
            context_region=context,
            metadata=metadata,
            default_stake=max(t1.default_stake, t2.default_stake),  # Risk is max of both
            default_confidence=t1.default_confidence * t2.default_confidence,  # Both must succeed
        )
    
    @staticmethod
    def conditional(
        t1: CommitmentTemplate,
        t2: CommitmentTemplate,
        predicate: Callable[[dict[str, Any]], bool],
        predicate_expression: str = "condition",
        name: str | None = None,
    ) -> CommitmentTemplate:
        """
        Conditional composition: T₁ ◁p▷ T₂
        
        Choose t1 if predicate is true, otherwise t2.
        
        From Definition 11:
        - τ_{T₁◁p▷T₂}(s) = τ₁(s) ∨ τ₂(s)
        - α_{T₁◁p▷T₂}(s) = α₁(s) if p(s) else α₂(s)
        
        Args:
            t1: Template to use if predicate is true.
            t2: Template to use if predicate is false.
            predicate: Function to decide which template to use.
            predicate_expression: String representation of predicate.
            name: Optional name for composed template.
            
        Returns:
            Composed template.
        """
        composed_name = name or f"{t1.metadata.name}_or_{t2.metadata.name}"
        
        # Trigger: either can trigger
        trigger = TriggerSchema(
            name=f"{composed_name}_trigger",
            description=f"Conditional: {t1.trigger_schema.name} or {t2.trigger_schema.name}",
            expression=f"({t1.trigger_schema.expression}) or ({t2.trigger_schema.expression})",
            required_keys=list(set(t1.trigger_schema.required_keys + t2.trigger_schema.required_keys)),
            defaults={**t1.trigger_schema.defaults, **t2.trigger_schema.defaults},
        )
        
        # Action: conditional action type
        action = ActionSchema(
            action_type=f"conditional({t1.action_schema.action_type}, {t2.action_schema.action_type})",
            description=f"If {predicate_expression}: {t1.action_schema.action_type}, else: {t2.action_schema.action_type}",
            parameter_templates={
                **{f"if_true_{k}": v for k, v in t1.action_schema.parameter_templates.items()},
                **{f"if_false_{k}": v for k, v in t2.action_schema.parameter_templates.items()},
            },
            static_parameters={
                "if_true": t1.action_schema.static_parameters,
                "if_false": t2.action_schema.static_parameters,
                "predicate": predicate_expression,
            },
        )
        
        # Verification: depends on which branch was taken
        # We use a combined expression that checks the appropriate condition
        verification = VerificationSchema(
            name=f"{composed_name}_verify",
            description=f"Verify based on branch: {t1.verification_schema.name} or {t2.verification_schema.name}",
            success_expression=f"(branch_taken == 'true' and ({t1.verification_schema.success_expression})) or (branch_taken == 'false' and ({t2.verification_schema.success_expression}))",
            failure_modes=t1.verification_schema.failure_modes + t2.verification_schema.failure_modes,
        )
        
        # Context: union of validated regions
        context = ContextRegionSpec(
            description=f"Union of {t1.context_region.description} and {t2.context_region.description}",
            bounds={},  # Union means we accept either region
            allowed_values={},
            tags=t1.context_region.tags | t2.context_region.tags,
        )
        
        # Metadata
        metadata = TemplateMetadata(
            name=composed_name,
            source="composed",
            tags=t1.metadata.tags | t2.metadata.tags | {"conditional"},
        )
        
        # Create template with custom confidence function
        template = CommitmentTemplate(
            trigger_schema=trigger,
            action_schema=action,
            verification_schema=verification,
            context_region=context,
            metadata=metadata,
            default_stake=max(t1.default_stake, t2.default_stake),
            default_confidence=max(t1.default_confidence, t2.default_confidence),
        )
        
        # Store predicate for runtime evaluation
        template._conditional_predicate = predicate
        template._conditional_templates = (t1, t2)
        
        return template
    
    @staticmethod
    def repeat(
        template: CommitmentTemplate,
        n: int,
        name: str | None = None,
    ) -> CommitmentTemplate:
        """
        Repeat a template n times sequentially.
        
        Args:
            template: Template to repeat.
            n: Number of repetitions.
            name: Optional name for composed template.
            
        Returns:
            Composed template.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return template
        
        result = template
        for _ in range(n - 1):
            result = TemplateComposer.sequential(result, template)
        
        if name:
            result.metadata.name = name
        else:
            result.metadata.name = f"{template.metadata.name}_x{n}"
        
        return result
    
    @staticmethod
    def fallback(
        primary: CommitmentTemplate,
        fallback: CommitmentTemplate,
        name: str | None = None,
    ) -> CommitmentTemplate:
        """
        Create a template that falls back to another on failure.
        
        This is a special case of conditional composition where
        the predicate is "primary succeeded".
        
        Args:
            primary: Primary template to try first.
            fallback: Fallback template if primary fails.
            name: Optional name for composed template.
            
        Returns:
            Composed template with fallback behavior.
        """
        composed_name = name or f"{primary.metadata.name}_with_fallback"
        
        # Trigger: primary's trigger
        trigger = TriggerSchema(
            name=f"{composed_name}_trigger",
            description=f"Fallback: try {primary.trigger_schema.name}, fallback to {fallback.trigger_schema.name}",
            expression=primary.trigger_schema.expression,
            required_keys=list(set(primary.trigger_schema.required_keys + fallback.trigger_schema.required_keys)),
            defaults={**primary.trigger_schema.defaults, **fallback.trigger_schema.defaults},
        )
        
        # Action: fallback action type
        action = ActionSchema(
            action_type=f"fallback({primary.action_schema.action_type}, {fallback.action_schema.action_type})",
            description=f"Try {primary.action_schema.action_type}, fallback to {fallback.action_schema.action_type}",
            parameter_templates={
                **{f"primary_{k}": v for k, v in primary.action_schema.parameter_templates.items()},
                **{f"fallback_{k}": v for k, v in fallback.action_schema.parameter_templates.items()},
            },
            static_parameters={
                "primary": primary.action_schema.static_parameters,
                "fallback": fallback.action_schema.static_parameters,
            },
        )
        
        # Verification: either primary succeeds or fallback succeeds
        verification = VerificationSchema(
            name=f"{composed_name}_verify",
            description=f"Either {primary.verification_schema.name} or {fallback.verification_schema.name} succeeds",
            success_expression=f"({primary.verification_schema.success_expression}) or ({fallback.verification_schema.success_expression})",
            failure_modes=[
                {
                    "name": "both_failed",
                    "condition": f"not ({primary.verification_schema.success_expression}) and not ({fallback.verification_schema.success_expression})",
                    "consequence_type": "penalty",
                    "magnitude": 0.7,
                    "severity": 0.7,
                    "description": "Both primary and fallback failed",
                }
            ],
        )
        
        # Context: union of validated regions
        context = ContextRegionSpec(
            description=f"Union of {primary.context_region.description} and {fallback.context_region.description}",
            bounds={},
            allowed_values={},
            tags=primary.context_region.tags | fallback.context_region.tags,
        )
        
        # Metadata
        metadata = TemplateMetadata(
            name=composed_name,
            source="composed",
            tags=primary.metadata.tags | fallback.metadata.tags | {"fallback"},
        )
        
        # Confidence: P(primary succeeds) + P(primary fails) * P(fallback succeeds)
        # Approximated as: max of both (conservative)
        combined_confidence = 1 - (1 - primary.default_confidence) * (1 - fallback.default_confidence)
        
        return CommitmentTemplate(
            trigger_schema=trigger,
            action_schema=action,
            verification_schema=verification,
            context_region=context,
            metadata=metadata,
            default_stake=max(primary.default_stake, fallback.default_stake),
            default_confidence=combined_confidence,
        )


def compute_template_similarity(
    t1: CommitmentTemplate,
    t2: CommitmentTemplate,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute structural similarity between two templates.
    
    Implements Definition 10:
    Sim(T₁, T₂) = λ_τ · Sim_τ + λ_α · Sim_α + λ_ψ · Sim_ψ
    
    Args:
        t1: First template.
        t2: Second template.
        weights: Optional weights for components (default: equal).
        
    Returns:
        Similarity score in [0, 1].
    """
    if weights is None:
        weights = {"trigger": 1/3, "action": 1/3, "verification": 1/3}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Trigger similarity (based on expression overlap)
    trigger_sim = _expression_similarity(
        t1.trigger_schema.expression,
        t2.trigger_schema.expression,
    )
    
    # Action similarity (based on action type)
    action_sim = 1.0 if t1.action_schema.action_type == t2.action_schema.action_type else 0.0
    
    # Verification similarity (based on success expression)
    verify_sim = _expression_similarity(
        t1.verification_schema.success_expression,
        t2.verification_schema.success_expression,
    )
    
    return (
        weights.get("trigger", 0) * trigger_sim +
        weights.get("action", 0) * action_sim +
        weights.get("verification", 0) * verify_sim
    )


def _expression_similarity(expr1: str, expr2: str) -> float:
    """Compute similarity between two expressions using token overlap."""
    # Simple token-based similarity
    tokens1 = set(expr1.lower().split())
    tokens2 = set(expr2.lower().split())
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    return len(intersection) / len(union)
