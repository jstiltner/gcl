"""
LLM Integration Module for GCL.

This module provides abstractions for integrating Large Language Models
with the Grounded Commitment Learning framework. It enables:

1. LLM Interface - Abstract interface for different LLM providers
2. Commitment Parsing - Extract commitments from LLM outputs
3. Grounding - Ground LLM outputs to verifiable commitments
4. Alignment Verification - Verify LLM behavior against commitments

Key Components:
- LLMInterface: Abstract base class for LLM providers
- MockLLM: Testing implementation
- CommitmentParser: Parse commitments from natural language
- GroundingEngine: Ground LLM outputs to formal commitments
"""

from gcl.llm.interface import (
    LLMInterface,
    LLMResponse,
    MockLLM,
    OpenAILLM,
    ResponseType,
    CommitmentAwareLLM,
)
from gcl.llm.commitment_parser import (
    Commitment,
    CommitmentParser,
    CommitmentStrength,
    CommitmentType,
    CommitmentExtractor,
    ParsedCommitment,
    ParsingResult,
)
from gcl.llm.grounding import (
    AlignmentVerifier,
    GroundingEngine,
    GroundingResult,
    GroundingStatus,
    GroundedOutput,
    GroundedPredicate,
    Predicate,
    PredicateRegistry,
    VerificationResult,
)

__all__ = [
    # Interface
    "LLMInterface",
    "LLMResponse",
    "MockLLM",
    "OpenAILLM",
    "ResponseType",
    "CommitmentAwareLLM",
    # Parser
    "Commitment",
    "CommitmentParser",
    "CommitmentStrength",
    "CommitmentType",
    "CommitmentExtractor",
    "ParsedCommitment",
    "ParsingResult",
    # Grounding
    "AlignmentVerifier",
    "GroundingEngine",
    "GroundingResult",
    "GroundingStatus",
    "GroundedOutput",
    "GroundedPredicate",
    "Predicate",
    "PredicateRegistry",
    "VerificationResult",
]
