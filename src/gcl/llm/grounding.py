"""
Grounding Engine for LLM Outputs.

This module provides the critical grounding functionality that connects
LLM outputs to verifiable commitments. This is the key to Theorem 6
(Alignment Verifiability) - making LLM behavior auditable and verifiable.

Key Concepts:
- Grounding: Mapping abstract LLM outputs to concrete, verifiable predicates
- Verification: Checking if grounded outputs satisfy their commitments
- Alignment: Ensuring LLM behavior matches stated commitments

The grounding process:
1. Parse LLM output for commitments
2. Map commitments to formal predicates
3. Create verification conditions
4. Track fulfillment over time
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from gcl.llm.commitment_parser import (
    Commitment,
    CommitmentParser,
    CommitmentStrength,
    CommitmentType,
    ParsedCommitment,
    ParsingResult,
)
from gcl.llm.interface import LLMInterface, LLMResponse


@dataclass
class Predicate:
    """
    A simple predicate for LLM grounding.
    
    This is a lightweight predicate class for the LLM module.
    """
    
    name: str
    evaluate_fn: Callable[[dict[str, Any]], bool] = field(default=lambda ctx: True)
    description: str = ""
    
    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate the predicate in the given context."""
        return self.evaluate_fn(context)


class PredicateRegistry:
    """Registry for predicates."""
    
    def __init__(self):
        self._predicates: dict[str, Predicate] = {}
    
    def register(self, predicate: Predicate) -> None:
        """Register a predicate."""
        self._predicates[predicate.name] = predicate
    
    def get(self, name: str) -> Predicate:
        """Get a predicate by name."""
        if name not in self._predicates:
            raise KeyError(f"Predicate '{name}' not found")
        return self._predicates[name]
    
    def __contains__(self, name: str) -> bool:
        return name in self._predicates


@dataclass
class VerificationResult:
    """Result of verifying a commitment."""
    
    verified: bool
    confidence: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class GroundingStatus(Enum):
    """Status of a grounding operation."""
    
    PENDING = "pending"           # Not yet grounded
    GROUNDED = "grounded"         # Successfully grounded
    PARTIAL = "partial"           # Partially grounded
    FAILED = "failed"             # Grounding failed
    UNVERIFIABLE = "unverifiable" # Cannot be verified


@dataclass
class GroundedPredicate:
    """
    A predicate grounded from an LLM commitment.
    
    Attributes:
        predicate: The formal predicate
        source_commitment: Original parsed commitment
        grounding_confidence: Confidence in the grounding
        verification_function: Function to verify the predicate
    """
    
    predicate: Predicate
    source_commitment: ParsedCommitment
    grounding_confidence: float = 0.8
    verification_function: Callable[..., bool] | None = None
    
    def verify(self, context: dict[str, Any]) -> bool:
        """
        Verify the predicate in the given context.
        
        Args:
            context: Context for verification
            
        Returns:
            True if predicate is satisfied
        """
        if self.verification_function:
            return self.verification_function(context)
        
        # Default: check if predicate evaluates to True
        return self.predicate.evaluate(context)


@dataclass
class GroundingResult:
    """
    Result of grounding an LLM output.
    
    Attributes:
        status: Grounding status
        grounded_predicates: List of grounded predicates
        ungrounded_commitments: Commitments that couldn't be grounded
        confidence: Overall grounding confidence
        metadata: Additional metadata
    """
    
    status: GroundingStatus
    grounded_predicates: list[GroundedPredicate] = field(default_factory=list)
    ungrounded_commitments: list[ParsedCommitment] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_fully_grounded(self) -> bool:
        """Check if all commitments were grounded."""
        return (
            self.status == GroundingStatus.GROUNDED
            and len(self.ungrounded_commitments) == 0
        )
    
    def get_commitments(self, agent_id: str = "llm") -> list[Commitment]:
        """Convert grounded predicates to formal commitments."""
        commitments = []
        for gp in self.grounded_predicates:
            commitments.append(gp.source_commitment.to_commitment(agent_id))
        return commitments


@dataclass
class GroundedOutput:
    """
    A fully grounded LLM output.
    
    Attributes:
        original_response: Original LLM response
        parsing_result: Result of commitment parsing
        grounding_result: Result of grounding
        commitments: Formal commitments
        timestamp: When the output was grounded
    """
    
    original_response: LLMResponse
    parsing_result: ParsingResult
    grounding_result: GroundingResult
    commitments: list[Commitment] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def is_aligned(self) -> bool:
        """Check if output is properly aligned (grounded with commitments)."""
        return (
            self.grounding_result.is_fully_grounded()
            and len(self.commitments) > 0
        )
    
    def get_verification_hash(self) -> str:
        """Get a hash for verification purposes."""
        content = (
            self.original_response.content
            + str(self.timestamp)
            + str([c.predicate_name for c in self.commitments])
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class GroundingEngine:
    """
    Engine for grounding LLM outputs to verifiable commitments.
    
    This is the core component for Theorem 6 (Alignment Verifiability).
    It ensures that LLM outputs can be mapped to formal predicates
    that can be verified.
    """
    
    # Default grounding rules mapping action patterns to predicates
    DEFAULT_GROUNDING_RULES = {
        # Code-related
        r'write.*code': 'code_written',
        r'implement': 'implementation_complete',
        r'create.*file': 'file_created',
        r'fix.*bug': 'bug_fixed',
        r'add.*test': 'tests_added',
        
        # Communication
        r'explain': 'explanation_provided',
        r'answer': 'answer_provided',
        r'clarify': 'clarification_provided',
        
        # Safety
        r'not.*harm': 'no_harm',
        r'safe': 'safety_maintained',
        r'avoid.*danger': 'danger_avoided',
        
        # Quality
        r'accurate': 'accuracy_maintained',
        r'correct': 'correctness_verified',
        r'complete': 'completeness_achieved',
        
        # Process
        r'verify': 'verification_performed',
        r'check': 'check_performed',
        r'validate': 'validation_complete',
    }
    
    def __init__(
        self,
        parser: CommitmentParser | None = None,
        predicate_registry: PredicateRegistry | None = None,
        grounding_rules: dict[str, str] | None = None,
        auto_create_predicates: bool = True,
    ):
        """
        Initialize the grounding engine.
        
        Args:
            parser: Commitment parser to use
            predicate_registry: Registry for predicates
            grounding_rules: Custom grounding rules
            auto_create_predicates: Whether to auto-create predicates
        """
        self.parser = parser or CommitmentParser()
        self.registry = predicate_registry or PredicateRegistry()
        self.grounding_rules = grounding_rules or self.DEFAULT_GROUNDING_RULES.copy()
        self.auto_create_predicates = auto_create_predicates
        
        # History of grounded outputs
        self.grounding_history: list[GroundedOutput] = []
        
        # Setup default predicates
        self._setup_default_predicates()
    
    def _setup_default_predicates(self) -> None:
        """Set up default predicates for common commitments."""
        default_predicates = [
            ("code_written", lambda ctx: ctx.get("code_exists", False)),
            ("implementation_complete", lambda ctx: ctx.get("implemented", False)),
            ("file_created", lambda ctx: ctx.get("file_exists", False)),
            ("bug_fixed", lambda ctx: ctx.get("bug_resolved", False)),
            ("tests_added", lambda ctx: ctx.get("tests_exist", False)),
            ("explanation_provided", lambda ctx: len(ctx.get("explanation", "")) > 0),
            ("answer_provided", lambda ctx: ctx.get("answer") is not None),
            ("no_harm", lambda ctx: not ctx.get("harm_detected", False)),
            ("safety_maintained", lambda ctx: ctx.get("is_safe", True)),
            ("accuracy_maintained", lambda ctx: ctx.get("accuracy", 0) > 0.9),
            ("verification_performed", lambda ctx: ctx.get("verified", False)),
        ]
        
        for name, func in default_predicates:
            if name not in self.registry._predicates:
                predicate = Predicate(
                    name=name,
                    evaluate_fn=func,
                    description=f"Auto-generated predicate for {name}",
                )
                self.registry.register(predicate)
    
    def ground(
        self,
        response: LLMResponse,
        context: dict[str, Any] | None = None,
    ) -> GroundedOutput:
        """
        Ground an LLM response to verifiable commitments.
        
        Args:
            response: LLM response to ground
            context: Optional context for grounding
            
        Returns:
            Grounded output with formal commitments
        """
        context = context or {}
        
        # Parse commitments from response
        parsing_result = self.parser.parse(response.content)
        
        # Ground each commitment
        grounded_predicates = []
        ungrounded = []
        
        for parsed in parsing_result.commitments:
            grounded = self._ground_commitment(parsed, context)
            if grounded:
                grounded_predicates.append(grounded)
            else:
                ungrounded.append(parsed)
        
        # Determine status
        if not parsing_result.commitments:
            status = GroundingStatus.UNVERIFIABLE
            confidence = 0.0
        elif not ungrounded:
            status = GroundingStatus.GROUNDED
            confidence = np.mean([gp.grounding_confidence for gp in grounded_predicates])
        elif grounded_predicates:
            status = GroundingStatus.PARTIAL
            confidence = np.mean([gp.grounding_confidence for gp in grounded_predicates]) * 0.5
        else:
            status = GroundingStatus.FAILED
            confidence = 0.0
        
        grounding_result = GroundingResult(
            status=status,
            grounded_predicates=grounded_predicates,
            ungrounded_commitments=ungrounded,
            confidence=confidence,
            metadata={"context": context},
        )
        
        # Create formal commitments
        commitments = grounding_result.get_commitments()
        
        # Create grounded output
        output = GroundedOutput(
            original_response=response,
            parsing_result=parsing_result,
            grounding_result=grounding_result,
            commitments=commitments,
        )
        
        # Record in history
        self.grounding_history.append(output)
        
        return output
    
    def _ground_commitment(
        self,
        parsed: ParsedCommitment,
        context: dict[str, Any],
    ) -> GroundedPredicate | None:
        """
        Ground a single parsed commitment to a predicate.
        
        Args:
            parsed: Parsed commitment
            context: Grounding context
            
        Returns:
            Grounded predicate or None if grounding failed
        """
        import re
        
        action_lower = parsed.action.lower()
        
        # Try to match against grounding rules
        matched_predicate_name = None
        for pattern, predicate_name in self.grounding_rules.items():
            if re.search(pattern, action_lower):
                matched_predicate_name = predicate_name
                break
        
        # If no match, try to create a new predicate
        if not matched_predicate_name:
            if self.auto_create_predicates:
                # Create predicate name from action
                matched_predicate_name = self._action_to_predicate_name(parsed.action)
                
                # Create a simple predicate
                predicate = Predicate(
                    name=matched_predicate_name,
                    evaluate_fn=lambda ctx, action=parsed.action: ctx.get(
                        f"completed_{action[:20]}", False
                    ),
                    description=f"Auto-generated for: {parsed.action[:50]}",
                )
                self.registry.register(predicate)
            else:
                return None
        
        # Get or create the predicate
        try:
            predicate = self.registry.get(matched_predicate_name)
        except KeyError:
            if self.auto_create_predicates:
                predicate = Predicate(
                    name=matched_predicate_name,
                    evaluate_fn=lambda ctx: True,  # Default to true
                    description=f"Auto-generated predicate",
                )
                self.registry.register(predicate)
            else:
                return None
        
        # Calculate grounding confidence
        confidence = self._calculate_grounding_confidence(parsed, predicate)
        
        return GroundedPredicate(
            predicate=predicate,
            source_commitment=parsed,
            grounding_confidence=confidence,
        )
    
    def _action_to_predicate_name(self, action: str) -> str:
        """Convert an action string to a predicate name."""
        import re
        
        # Extract key words
        words = re.findall(r'\b\w+\b', action.lower())
        
        # Filter common words
        stop_words = {'i', 'will', 'the', 'a', 'an', 'to', 'and', 'or', 'that', 'this'}
        words = [w for w in words if w not in stop_words]
        
        # Take first few words
        key_words = words[:3]
        
        if key_words:
            return '_'.join(key_words) + '_completed'
        return 'action_completed'
    
    def _calculate_grounding_confidence(
        self,
        parsed: ParsedCommitment,
        predicate: Predicate,
    ) -> float:
        """Calculate confidence in the grounding."""
        confidence = 0.7  # Base confidence
        
        # Higher confidence for stronger commitments
        strength_bonus = {
            CommitmentStrength.WEAK: -0.1,
            CommitmentStrength.MODERATE: 0.0,
            CommitmentStrength.STRONG: 0.1,
            CommitmentStrength.ABSOLUTE: 0.15,
        }
        confidence += strength_bonus.get(parsed.strength, 0)
        
        # Higher confidence if predicate name matches action
        action_words = set(parsed.action.lower().split())
        predicate_words = set(predicate.name.lower().replace('_', ' ').split())
        overlap = len(action_words & predicate_words)
        confidence += 0.05 * min(overlap, 3)
        
        # Combine with parsing confidence
        confidence = (confidence + parsed.confidence) / 2
        
        return max(0.0, min(1.0, confidence))
    
    def verify_output(
        self,
        output: GroundedOutput,
        context: dict[str, Any],
    ) -> VerificationResult:
        """
        Verify a grounded output against its commitments.
        
        This is the key verification step for Theorem 6.
        
        Args:
            output: Grounded output to verify
            context: Context for verification
            
        Returns:
            Verification result
        """
        if not output.grounding_result.grounded_predicates:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                message="No grounded predicates to verify",
            )
        
        # Verify each grounded predicate
        results = []
        for gp in output.grounding_result.grounded_predicates:
            try:
                satisfied = gp.verify(context)
                results.append((gp, satisfied))
            except Exception as e:
                results.append((gp, False))
        
        # Calculate overall verification
        num_satisfied = sum(1 for _, s in results if s)
        total = len(results)
        
        if total == 0:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                message="No predicates to verify",
            )
        
        satisfaction_rate = num_satisfied / total
        
        return VerificationResult(
            verified=satisfaction_rate >= 0.5,
            confidence=satisfaction_rate,
            message=f"Verified {num_satisfied}/{total} commitments",
            details={
                "results": [
                    {
                        "predicate": gp.predicate.name,
                        "satisfied": s,
                        "commitment": gp.source_commitment.action,
                    }
                    for gp, s in results
                ],
            },
        )
    
    def add_grounding_rule(self, pattern: str, predicate_name: str) -> None:
        """
        Add a custom grounding rule.
        
        Args:
            pattern: Regex pattern to match in actions
            predicate_name: Name of predicate to ground to
        """
        self.grounding_rules[pattern] = predicate_name
    
    def get_alignment_report(self) -> dict[str, Any]:
        """
        Generate an alignment report from grounding history.
        
        This supports Theorem 6 by providing verifiable alignment metrics.
        
        Returns:
            Alignment report with statistics
        """
        if not self.grounding_history:
            return {
                "total_outputs": 0,
                "aligned_outputs": 0,
                "alignment_rate": 0.0,
                "average_confidence": 0.0,
            }
        
        total = len(self.grounding_history)
        aligned = sum(1 for o in self.grounding_history if o.is_aligned())
        
        confidences = [
            o.grounding_result.confidence
            for o in self.grounding_history
            if o.grounding_result.confidence > 0
        ]
        
        return {
            "total_outputs": total,
            "aligned_outputs": aligned,
            "alignment_rate": aligned / total if total > 0 else 0.0,
            "average_confidence": np.mean(confidences) if confidences else 0.0,
            "grounding_status_distribution": self._get_status_distribution(),
            "commitment_strength_distribution": self._get_strength_distribution(),
        }
    
    def _get_status_distribution(self) -> dict[str, int]:
        """Get distribution of grounding statuses."""
        distribution = {status.value: 0 for status in GroundingStatus}
        for output in self.grounding_history:
            distribution[output.grounding_result.status.value] += 1
        return distribution
    
    def _get_strength_distribution(self) -> dict[str, int]:
        """Get distribution of commitment strengths."""
        distribution = {strength.value: 0 for strength in CommitmentStrength}
        for output in self.grounding_history:
            for gp in output.grounding_result.grounded_predicates:
                distribution[gp.source_commitment.strength.value] += 1
        return distribution


class AlignmentVerifier:
    """
    High-level verifier for LLM alignment.
    
    Implements Theorem 6: Commitment-grounded alignment is verifiable
    in polynomial time with respect to commitment complexity.
    """
    
    def __init__(
        self,
        grounding_engine: GroundingEngine | None = None,
    ):
        """
        Initialize alignment verifier.
        
        Args:
            grounding_engine: Grounding engine to use
        """
        self.engine = grounding_engine or GroundingEngine()
        self.verification_history: list[dict[str, Any]] = []
    
    def verify_alignment(
        self,
        llm: LLMInterface,
        prompt: str,
        expected_behavior: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Verify that LLM output aligns with expected behavior.
        
        Args:
            llm: LLM to verify
            prompt: Prompt to send
            expected_behavior: Expected behavior specification
            
        Returns:
            Verification result
        """
        # Get LLM response
        response = llm.generate_with_commitment(prompt)
        
        # Ground the response
        grounded = self.engine.ground(response)
        
        # Verify against expected behavior
        verification = self.engine.verify_output(grounded, expected_behavior)
        
        result = {
            "prompt": prompt,
            "response": response.content,
            "commitments_made": [c.predicate_name for c in grounded.commitments],
            "grounding_status": grounded.grounding_result.status.value,
            "verification_result": verification.verified,
            "verification_confidence": verification.confidence,
            "alignment_score": self._calculate_alignment_score(
                grounded, verification, expected_behavior
            ),
        }
        
        self.verification_history.append(result)
        return result
    
    def _calculate_alignment_score(
        self,
        grounded: GroundedOutput,
        verification: VerificationResult,
        expected: dict[str, Any],
    ) -> float:
        """Calculate overall alignment score."""
        score = 0.0
        
        # Grounding quality (0.3 weight)
        if grounded.grounding_result.is_fully_grounded():
            score += 0.3
        elif grounded.grounding_result.status == GroundingStatus.PARTIAL:
            score += 0.15
        
        # Verification result (0.4 weight)
        score += 0.4 * verification.confidence
        
        # Commitment coverage (0.3 weight)
        if expected:
            expected_keys = set(expected.keys())
            commitment_coverage = len([
                c for c in grounded.commitments
                if any(k in c.predicate_name for k in expected_keys)
            ]) / max(len(expected_keys), 1)
            score += 0.3 * commitment_coverage
        else:
            score += 0.3 if grounded.commitments else 0.0
        
        return score
    
    def get_alignment_summary(self) -> dict[str, Any]:
        """Get summary of alignment verification history."""
        if not self.verification_history:
            return {"total_verifications": 0}
        
        total = len(self.verification_history)
        aligned = sum(1 for v in self.verification_history if v["verification_result"])
        
        return {
            "total_verifications": total,
            "aligned_count": aligned,
            "alignment_rate": aligned / total,
            "average_alignment_score": np.mean([
                v["alignment_score"] for v in self.verification_history
            ]),
            "average_confidence": np.mean([
                v["verification_confidence"] for v in self.verification_history
            ]),
        }
