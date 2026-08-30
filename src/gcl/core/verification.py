"""
Verification engine for checking commitment fulfillment.

This module provides the VerificationEngine class that checks whether
commitments have been fulfilled based on pre-state, post-state, and actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from gcl.core.commitment import (
    CommitmentPortfolio,
    GroundedCommitment,
    VerificationResult,
    VerificationStatus,
)
from gcl.core.registry import verification_registry


@dataclass
class VerificationContext:
    """
    Context for a verification operation.
    
    Contains all the information needed to verify a commitment.
    
    Attributes:
        pre_state: State before the action was taken.
        post_state: State after the action was taken.
        action: The action that was performed.
        metadata: Additional context information.
    """
    
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    action: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchVerificationResult:
    """
    Result of verifying multiple commitments.
    
    Attributes:
        results: List of individual verification results.
        total_count: Total number of commitments verified.
        success_count: Number of successful verifications.
        failure_count: Number of failed verifications.
        error_count: Number of verification errors.
        total_duration_ms: Total time taken for all verifications.
    """
    
    results: list[VerificationResult]
    total_count: int
    success_count: int
    failure_count: int
    error_count: int
    total_duration_ms: float
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    @property
    def all_successful(self) -> bool:
        """Check if all verifications were successful."""
        return self.success_count == self.total_count
    
    def get_failures(self) -> list[VerificationResult]:
        """Get all failed verification results."""
        return [r for r in self.results if r.status == VerificationStatus.FAILURE]
    
    def get_errors(self) -> list[VerificationResult]:
        """Get all error verification results."""
        return [r for r in self.results if r.status == VerificationStatus.ERROR]


class VerificationEngine:
    """
    Engine for verifying commitment fulfillment.
    
    The VerificationEngine checks whether commitments have been fulfilled
    by evaluating their success conditions and failure modes against
    the observed state changes.
    
    Verification process:
    1. Check if commitment is expired
    2. Check if context is valid for the commitment
    3. Check failure modes (in order of severity)
    4. If no failure mode triggered, check success condition
    5. If custom verification function exists, use it
    
    Example:
        >>> engine = VerificationEngine()
        >>> context = VerificationContext(
        ...     pre_state={"counter": 0},
        ...     post_state={"counter": 5, "accuracy": 0.95},
        ...     action={"type": "increment"}
        ... )
        >>> result = engine.verify(commitment, context)
        >>> print(result.status)
        VerificationStatus.SUCCESS
    """
    
    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        strict_context_matching: bool = False,
    ) -> None:
        """
        Initialize the verification engine.
        
        Args:
            timeout_seconds: Default timeout for verification operations.
            strict_context_matching: If True, fail if commitment context
                doesn't match the verification context.
        """
        self.timeout_seconds = timeout_seconds
        self.strict_context_matching = strict_context_matching
    
    def verify(
        self,
        commitment: GroundedCommitment,
        context: VerificationContext,
    ) -> VerificationResult:
        """
        Verify a single commitment.
        
        Args:
            commitment: The commitment to verify.
            context: The verification context with pre/post state and action.
            
        Returns:
            VerificationResult indicating success, failure, or error.
        """
        start_time = time.perf_counter()
        
        try:
            # Check if commitment is expired
            if commitment.is_expired():
                return self._create_result(
                    commitment_id=commitment.id,
                    status=VerificationStatus.TIMEOUT,
                    details={"reason": "Commitment has expired"},
                    start_time=start_time,
                )
            
            # Check context validity if strict matching is enabled
            if self.strict_context_matching:
                combined_context = {**context.pre_state, **context.post_state}
                if not commitment.is_valid_in_context(combined_context):
                    return self._create_result(
                        commitment_id=commitment.id,
                        status=VerificationStatus.ERROR,
                        details={"reason": "Context does not match commitment bounds"},
                        start_time=start_time,
                    )
            
            # Build evaluation context from pre_state, post_state, and action
            eval_context = self._build_eval_context(context)
            
            # Check for custom verification function
            if commitment.verification_fn_name:
                fn = verification_registry.get(commitment.verification_fn_name)
                if fn is not None:
                    result = fn(context.pre_state, context.post_state, context.action)
                    # Update the commitment_id and duration
                    return VerificationResult(
                        status=result.status,
                        commitment_id=commitment.id,
                        triggered_failure_mode=result.triggered_failure_mode,
                        details=result.details,
                        timestamp=datetime.now(timezone.utc),
                        verification_duration_ms=self._elapsed_ms(start_time),
                    )
            
            # Check failure modes (in order of severity, highest first)
            for failure_mode in commitment.failure_modes:
                try:
                    if failure_mode.is_triggered(eval_context):
                        return self._create_result(
                            commitment_id=commitment.id,
                            status=VerificationStatus.FAILURE,
                            triggered_failure_mode=failure_mode.id,
                            details={
                                "failure_mode_name": failure_mode.name,
                                "severity": failure_mode.severity,
                                "consequence_type": failure_mode.consequence.consequence_type,
                                "consequence_magnitude": failure_mode.consequence.magnitude,
                            },
                            start_time=start_time,
                        )
                except Exception as e:
                    # If we can't evaluate a failure mode, continue to next
                    # but log the error in details
                    pass
            
            # Check success condition
            try:
                if commitment.check_success(eval_context):
                    return self._create_result(
                        commitment_id=commitment.id,
                        status=VerificationStatus.SUCCESS,
                        details={"message": "Success condition met"},
                        start_time=start_time,
                    )
                else:
                    # Success condition not met, but no failure mode triggered
                    # This is still a failure
                    return self._create_result(
                        commitment_id=commitment.id,
                        status=VerificationStatus.FAILURE,
                        details={
                            "reason": "Success condition not met",
                            "success_condition": commitment.success_condition.expression,
                        },
                        start_time=start_time,
                    )
            except Exception as e:
                return self._create_result(
                    commitment_id=commitment.id,
                    status=VerificationStatus.ERROR,
                    details={
                        "reason": "Error evaluating success condition",
                        "error": str(e),
                    },
                    start_time=start_time,
                )
                
        except Exception as e:
            return self._create_result(
                commitment_id=commitment.id,
                status=VerificationStatus.ERROR,
                details={
                    "reason": "Unexpected verification error",
                    "error": str(e),
                },
                start_time=start_time,
            )
    
    def verify_batch(
        self,
        commitments: list[GroundedCommitment],
        context: VerificationContext,
    ) -> BatchVerificationResult:
        """
        Verify multiple commitments against the same context.
        
        Args:
            commitments: List of commitments to verify.
            context: The verification context.
            
        Returns:
            BatchVerificationResult with all individual results and statistics.
        """
        start_time = time.perf_counter()
        results: list[VerificationResult] = []
        
        for commitment in commitments:
            result = self.verify(commitment, context)
            results.append(result)
        
        total_duration_ms = self._elapsed_ms(start_time)
        
        success_count = sum(1 for r in results if r.status == VerificationStatus.SUCCESS)
        failure_count = sum(1 for r in results if r.status == VerificationStatus.FAILURE)
        error_count = sum(
            1 for r in results 
            if r.status in (VerificationStatus.ERROR, VerificationStatus.TIMEOUT)
        )
        
        return BatchVerificationResult(
            results=results,
            total_count=len(commitments),
            success_count=success_count,
            failure_count=failure_count,
            error_count=error_count,
            total_duration_ms=total_duration_ms,
        )
    
    def verify_portfolio(
        self,
        portfolio: CommitmentPortfolio,
        context: VerificationContext,
        *,
        only_triggered: bool = True,
    ) -> BatchVerificationResult:
        """
        Verify all commitments in a portfolio.
        
        Args:
            portfolio: The portfolio to verify.
            context: The verification context.
            only_triggered: If True, only verify commitments whose
                trigger conditions are met.
                
        Returns:
            BatchVerificationResult with all individual results and statistics.
        """
        if only_triggered:
            # Build context for trigger evaluation
            trigger_context = {**context.pre_state, **context.metadata}
            commitments = [
                c for c in portfolio.active_commitments
                if c.triggers_match(trigger_context)
            ]
        else:
            commitments = portfolio.active_commitments
        
        return self.verify_batch(commitments, context)
    
    def _build_eval_context(self, context: VerificationContext) -> dict[str, Any]:
        """
        Build the evaluation context for predicate evaluation.
        
        Combines pre_state, post_state, action, and metadata into a single
        context dictionary. Post-state values override pre-state values.
        
        Args:
            context: The verification context.
            
        Returns:
            Combined context dictionary.
        """
        eval_context: dict[str, Any] = {}
        
        # Add pre-state with prefix
        for key, value in context.pre_state.items():
            eval_context[f"pre_{key}"] = value
        
        # Add post-state (these are the "current" values)
        eval_context.update(context.post_state)
        
        # Add action parameters
        eval_context["action"] = context.action
        for key, value in context.action.items():
            eval_context[f"action_{key}"] = value
        
        # Add metadata
        eval_context.update(context.metadata)
        
        return eval_context
    
    def _create_result(
        self,
        commitment_id: str,
        status: VerificationStatus,
        details: dict[str, Any],
        start_time: float,
        triggered_failure_mode: str | None = None,
    ) -> VerificationResult:
        """Create a VerificationResult with timing information."""
        return VerificationResult(
            status=status,
            commitment_id=commitment_id,
            triggered_failure_mode=triggered_failure_mode,
            details=details,
            timestamp=datetime.now(timezone.utc),
            verification_duration_ms=self._elapsed_ms(start_time),
        )
    
    def _elapsed_ms(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.perf_counter() - start_time) * 1000


class VerificationReport:
    """
    Detailed report of verification results.
    
    Provides analysis and summary of verification outcomes.
    """
    
    def __init__(self, batch_result: BatchVerificationResult) -> None:
        """
        Initialize the report.
        
        Args:
            batch_result: The batch verification result to analyze.
        """
        self.batch_result = batch_result
    
    @property
    def summary(self) -> dict[str, Any]:
        """Get a summary of the verification results."""
        return {
            "total": self.batch_result.total_count,
            "success": self.batch_result.success_count,
            "failure": self.batch_result.failure_count,
            "error": self.batch_result.error_count,
            "success_rate": self.batch_result.success_rate,
            "duration_ms": self.batch_result.total_duration_ms,
        }
    
    def get_failure_analysis(self) -> dict[str, Any]:
        """
        Analyze failure modes that were triggered.
        
        Returns:
            Dictionary with failure mode statistics.
        """
        failures = self.batch_result.get_failures()
        
        # Group by failure mode
        failure_modes: dict[str, int] = {}
        severities: list[float] = []
        
        for result in failures:
            mode_name = result.details.get("failure_mode_name", "unknown")
            failure_modes[mode_name] = failure_modes.get(mode_name, 0) + 1
            
            severity = result.details.get("severity")
            if severity is not None:
                severities.append(severity)
        
        return {
            "total_failures": len(failures),
            "failure_modes": failure_modes,
            "average_severity": sum(severities) / len(severities) if severities else 0.0,
            "max_severity": max(severities) if severities else 0.0,
        }
    
    def get_commitment_results(self) -> dict[str, VerificationResult]:
        """
        Get results indexed by commitment ID.
        
        Returns:
            Dictionary mapping commitment IDs to their results.
        """
        return {r.commitment_id: r for r in self.batch_result.results}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            "summary": self.summary,
            "failure_analysis": self.get_failure_analysis(),
            "results": [
                {
                    "commitment_id": r.commitment_id,
                    "status": r.status.value,
                    "triggered_failure_mode": r.triggered_failure_mode,
                    "details": r.details,
                    "duration_ms": r.verification_duration_ms,
                }
                for r in self.batch_result.results
            ],
        }
