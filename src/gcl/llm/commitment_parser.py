"""
Commitment Parser for LLM Outputs.

This module provides sophisticated parsing of natural language to extract
structured commitments that can be verified by the GCL framework.

Key Features:
- Pattern-based extraction of commitment statements
- Semantic analysis of commitment strength
- Temporal parsing for deadlines
- Condition extraction for conditional commitments
- Confidence scoring for parsed commitments
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CommitmentType(Enum):
    """Type of commitment."""
    
    BEHAVIORAL = "behavioral"
    INFORMATIONAL = "informational"
    RESOURCE = "resource"


@dataclass
class Commitment:
    """
    A simple commitment representation for LLM outputs.
    
    This is a lightweight commitment class used for LLM integration,
    separate from the more complex GroundedCommitment in the core module.
    """
    
    agent_id: str
    predicate_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    stake: float = 0.5
    commitment_type: CommitmentType = CommitmentType.BEHAVIORAL
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if "id" not in self.metadata:
            import uuid
            self.metadata["id"] = str(uuid.uuid4())
    
    @property
    def id(self) -> str:
        """Get commitment ID."""
        return self.metadata.get("id", "")


class CommitmentStrength(Enum):
    """Strength/certainty of a commitment."""
    
    WEAK = "weak"           # "I might...", "I could..."
    MODERATE = "moderate"   # "I will try...", "I should..."
    STRONG = "strong"       # "I will...", "I commit to..."
    ABSOLUTE = "absolute"   # "I guarantee...", "I promise..."


@dataclass
class ParsedCommitment:
    """
    A commitment parsed from natural language.
    
    Attributes:
        raw_text: Original text of the commitment
        action: The action being committed to
        conditions: Conditions under which commitment applies
        deadline: Temporal deadline if specified
        strength: Strength of the commitment
        confidence: Parser's confidence in the extraction (0-1)
        metadata: Additional parsed metadata
    """
    
    raw_text: str
    action: str
    conditions: list[str] = field(default_factory=list)
    deadline: str | None = None
    strength: CommitmentStrength = CommitmentStrength.MODERATE
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_commitment(
        self,
        agent_id: str,
        predicate_name: str = "llm_commitment",
    ) -> Commitment:
        """
        Convert to a formal GCL Commitment.
        
        Args:
            agent_id: ID of the agent making the commitment
            predicate_name: Name for the predicate
            
        Returns:
            Formal Commitment object
        """
        # Map strength to stake
        stake_map = {
            CommitmentStrength.WEAK: 0.1,
            CommitmentStrength.MODERATE: 0.3,
            CommitmentStrength.STRONG: 0.6,
            CommitmentStrength.ABSOLUTE: 0.9,
        }
        
        return Commitment(
            agent_id=agent_id,
            predicate_name=predicate_name,
            parameters={
                "action": self.action,
                "conditions": self.conditions,
                "deadline": self.deadline,
                "raw_text": self.raw_text,
            },
            stake=stake_map.get(self.strength, 0.3),
            commitment_type=CommitmentType.BEHAVIORAL,
            metadata={
                "strength": self.strength.value,
                "confidence": self.confidence,
                **self.metadata,
            },
        )
    
    def is_conditional(self) -> bool:
        """Check if this is a conditional commitment."""
        return len(self.conditions) > 0
    
    def has_deadline(self) -> bool:
        """Check if commitment has a deadline."""
        return self.deadline is not None


@dataclass
class ParsingResult:
    """
    Result of parsing text for commitments.
    
    Attributes:
        commitments: List of parsed commitments
        original_text: Original text that was parsed
        parse_confidence: Overall confidence in parsing
        warnings: Any warnings generated during parsing
    """
    
    commitments: list[ParsedCommitment]
    original_text: str
    parse_confidence: float = 0.8
    warnings: list[str] = field(default_factory=list)
    
    def has_commitments(self) -> bool:
        """Check if any commitments were found."""
        return len(self.commitments) > 0
    
    def get_strong_commitments(self) -> list[ParsedCommitment]:
        """Get only strong or absolute commitments."""
        return [
            c for c in self.commitments
            if c.strength in (CommitmentStrength.STRONG, CommitmentStrength.ABSOLUTE)
        ]
    
    def to_commitments(self, agent_id: str) -> list[Commitment]:
        """Convert all parsed commitments to formal Commitments."""
        return [c.to_commitment(agent_id) for c in self.commitments]


class CommitmentParser:
    """
    Parser for extracting commitments from natural language.
    
    Uses pattern matching and heuristics to identify commitment
    statements and extract their components.
    """
    
    # Commitment indicator patterns
    COMMITMENT_PATTERNS = [
        # Explicit markers
        (r'\[COMMITMENT:\s*([^\]]+)\]', CommitmentStrength.STRONG),
        (r'\[PROMISE:\s*([^\]]+)\]', CommitmentStrength.ABSOLUTE),
        
        # Strong commitments
        (r'I guarantee\s+(?:that\s+)?(.+?)(?:\.|$)', CommitmentStrength.ABSOLUTE),
        (r'I promise\s+(?:that\s+)?(.+?)(?:\.|$)', CommitmentStrength.ABSOLUTE),
        (r'I commit to\s+(.+?)(?:\.|$)', CommitmentStrength.STRONG),
        (r'I am committed to\s+(.+?)(?:\.|$)', CommitmentStrength.STRONG),
        
        # Moderate commitments
        (r'I will\s+(.+?)(?:\.|$)', CommitmentStrength.MODERATE),
        (r'I shall\s+(.+?)(?:\.|$)', CommitmentStrength.MODERATE),
        (r'I am going to\s+(.+?)(?:\.|$)', CommitmentStrength.MODERATE),
        
        # Weak commitments
        (r'I will try to\s+(.+?)(?:\.|$)', CommitmentStrength.WEAK),
        (r'I should\s+(.+?)(?:\.|$)', CommitmentStrength.WEAK),
        (r'I might\s+(.+?)(?:\.|$)', CommitmentStrength.WEAK),
        (r'I could\s+(.+?)(?:\.|$)', CommitmentStrength.WEAK),
    ]
    
    # Condition patterns
    CONDITION_PATTERNS = [
        r'if\s+(.+?)(?:,|then)',
        r'when\s+(.+?)(?:,|then)',
        r'provided\s+(?:that\s+)?(.+?)(?:,|$)',
        r'assuming\s+(?:that\s+)?(.+?)(?:,|$)',
        r'as long as\s+(.+?)(?:,|$)',
        r'unless\s+(.+?)(?:,|$)',
    ]
    
    # Deadline patterns
    DEADLINE_PATTERNS = [
        r'by\s+(tomorrow|today|next week|next month|\d{1,2}/\d{1,2}(?:/\d{2,4})?)',
        r'within\s+(\d+\s+(?:minutes?|hours?|days?|weeks?|months?))',
        r'before\s+(.+?)(?:\.|,|$)',
        r'no later than\s+(.+?)(?:\.|,|$)',
        r'deadline:\s*(.+?)(?:\.|,|$)',
    ]
    
    # Strengthening words
    STRENGTHENING_WORDS = {
        'definitely', 'certainly', 'absolutely', 'always', 'never',
        'ensure', 'guarantee', 'must', 'shall'
    }
    
    # Weakening words
    WEAKENING_WORDS = {
        'maybe', 'perhaps', 'possibly', 'might', 'could',
        'try', 'attempt', 'hope', 'if possible'
    }
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        extract_conditions: bool = True,
        extract_deadlines: bool = True,
    ):
        """
        Initialize the parser.
        
        Args:
            min_confidence: Minimum confidence to include a commitment
            extract_conditions: Whether to extract conditions
            extract_deadlines: Whether to extract deadlines
        """
        self.min_confidence = min_confidence
        self.extract_conditions = extract_conditions
        self.extract_deadlines = extract_deadlines
    
    def parse(self, text: str) -> ParsingResult:
        """
        Parse text to extract commitments.
        
        Args:
            text: Text to parse
            
        Returns:
            ParsingResult with extracted commitments
        """
        commitments = []
        warnings = []
        
        # Try each commitment pattern
        for pattern, base_strength in self.COMMITMENT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                raw_text = match.group(0).strip()
                action = match.group(1).strip()
                
                # Skip if too short
                if len(action) < 5:
                    continue
                
                # Adjust strength based on modifiers
                strength = self._adjust_strength(raw_text, base_strength)
                
                # Extract conditions
                conditions = []
                if self.extract_conditions:
                    conditions = self._extract_conditions(raw_text)
                
                # Extract deadline
                deadline = None
                if self.extract_deadlines:
                    deadline = self._extract_deadline(raw_text)
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    raw_text, action, strength, conditions
                )
                
                if confidence >= self.min_confidence:
                    commitments.append(ParsedCommitment(
                        raw_text=raw_text,
                        action=action,
                        conditions=conditions,
                        deadline=deadline,
                        strength=strength,
                        confidence=confidence,
                    ))
        
        # Remove duplicates (same action)
        commitments = self._deduplicate(commitments)
        
        # Calculate overall confidence
        if commitments:
            parse_confidence = np.mean([c.confidence for c in commitments])
        else:
            parse_confidence = 0.0
            warnings.append("No commitments found in text")
        
        return ParsingResult(
            commitments=commitments,
            original_text=text,
            parse_confidence=parse_confidence,
            warnings=warnings,
        )
    
    def parse_structured(
        self,
        text: str,
        expected_format: dict[str, Any] | None = None,
    ) -> ParsingResult:
        """
        Parse text expecting a structured format.
        
        Args:
            text: Text to parse
            expected_format: Expected structure (e.g., JSON schema)
            
        Returns:
            ParsingResult with extracted commitments
        """
        # Try to parse as JSON first
        import json
        
        try:
            # Look for JSON blocks
            json_pattern = r'```json\s*(\{.+?\})\s*```'
            json_matches = re.findall(json_pattern, text, re.DOTALL)
            
            if json_matches:
                for json_str in json_matches:
                    data = json.loads(json_str)
                    if 'commitments' in data:
                        return self._parse_json_commitments(data, text)
        except json.JSONDecodeError:
            pass
        
        # Fall back to pattern-based parsing
        return self.parse(text)
    
    def _parse_json_commitments(
        self,
        data: dict[str, Any],
        original_text: str,
    ) -> ParsingResult:
        """Parse commitments from JSON structure."""
        commitments = []
        
        for item in data.get('commitments', []):
            if isinstance(item, str):
                # Simple string commitment
                commitments.append(ParsedCommitment(
                    raw_text=item,
                    action=item,
                    strength=CommitmentStrength.MODERATE,
                    confidence=0.9,
                ))
            elif isinstance(item, dict):
                # Structured commitment
                commitments.append(ParsedCommitment(
                    raw_text=item.get('text', str(item)),
                    action=item.get('action', item.get('text', '')),
                    conditions=item.get('conditions', []),
                    deadline=item.get('deadline'),
                    strength=CommitmentStrength[
                        item.get('strength', 'MODERATE').upper()
                    ],
                    confidence=item.get('confidence', 0.9),
                    metadata=item.get('metadata', {}),
                ))
        
        return ParsingResult(
            commitments=commitments,
            original_text=original_text,
            parse_confidence=0.95 if commitments else 0.0,
        )
    
    def _adjust_strength(
        self,
        text: str,
        base_strength: CommitmentStrength,
    ) -> CommitmentStrength:
        """Adjust commitment strength based on modifiers."""
        text_lower = text.lower()
        
        # Check for strengthening words
        has_strengthening = any(
            word in text_lower for word in self.STRENGTHENING_WORDS
        )
        
        # Check for weakening words
        has_weakening = any(
            word in text_lower for word in self.WEAKENING_WORDS
        )
        
        if has_strengthening and not has_weakening:
            # Upgrade strength
            if base_strength == CommitmentStrength.WEAK:
                return CommitmentStrength.MODERATE
            elif base_strength == CommitmentStrength.MODERATE:
                return CommitmentStrength.STRONG
            elif base_strength == CommitmentStrength.STRONG:
                return CommitmentStrength.ABSOLUTE
        
        elif has_weakening and not has_strengthening:
            # Downgrade strength
            if base_strength == CommitmentStrength.ABSOLUTE:
                return CommitmentStrength.STRONG
            elif base_strength == CommitmentStrength.STRONG:
                return CommitmentStrength.MODERATE
            elif base_strength == CommitmentStrength.MODERATE:
                return CommitmentStrength.WEAK
        
        return base_strength
    
    def _extract_conditions(self, text: str) -> list[str]:
        """Extract conditions from commitment text."""
        conditions = []
        
        for pattern in self.CONDITION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend(matches)
        
        return [c.strip() for c in conditions if c.strip()]
    
    def _extract_deadline(self, text: str) -> str | None:
        """Extract deadline from commitment text."""
        for pattern in self.DEADLINE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _calculate_confidence(
        self,
        raw_text: str,
        action: str,
        strength: CommitmentStrength,
        conditions: list[str],
    ) -> float:
        """Calculate confidence score for parsed commitment."""
        confidence = 0.7  # Base confidence
        
        # Explicit markers increase confidence
        if '[COMMITMENT:' in raw_text.upper() or '[PROMISE:' in raw_text.upper():
            confidence += 0.2
        
        # Stronger commitments are more confident
        strength_bonus = {
            CommitmentStrength.WEAK: -0.1,
            CommitmentStrength.MODERATE: 0.0,
            CommitmentStrength.STRONG: 0.1,
            CommitmentStrength.ABSOLUTE: 0.15,
        }
        confidence += strength_bonus.get(strength, 0)
        
        # Longer, more specific actions are more confident
        if len(action) > 20:
            confidence += 0.05
        if len(action) > 50:
            confidence += 0.05
        
        # Conditions slightly reduce confidence (more complex)
        confidence -= 0.02 * len(conditions)
        
        return max(0.0, min(1.0, confidence))
    
    def _deduplicate(
        self,
        commitments: list[ParsedCommitment],
    ) -> list[ParsedCommitment]:
        """Remove duplicate commitments, keeping highest confidence."""
        seen_actions = {}
        
        for c in commitments:
            # Normalize action for comparison
            normalized = c.action.lower().strip()
            
            if normalized not in seen_actions:
                seen_actions[normalized] = c
            elif c.confidence > seen_actions[normalized].confidence:
                seen_actions[normalized] = c
        
        return list(seen_actions.values())
    
    def validate_commitment(
        self,
        commitment: ParsedCommitment,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate a parsed commitment.
        
        Args:
            commitment: Commitment to validate
            context: Optional context for validation
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check action is not empty
        if not commitment.action or len(commitment.action) < 3:
            issues.append("Action is too short or empty")
        
        # Check for vague language
        vague_words = ['something', 'stuff', 'things', 'etc', 'whatever']
        if any(word in commitment.action.lower() for word in vague_words):
            issues.append("Action contains vague language")
        
        # Check confidence threshold
        if commitment.confidence < 0.5:
            issues.append(f"Low confidence: {commitment.confidence:.2f}")
        
        # Check for contradictory conditions
        if commitment.conditions:
            for cond in commitment.conditions:
                if 'unless' in cond.lower() and 'if' in cond.lower():
                    issues.append("Potentially contradictory conditions")
        
        return len(issues) == 0, issues


class CommitmentExtractor:
    """
    High-level extractor that combines parsing with validation.
    """
    
    def __init__(
        self,
        parser: CommitmentParser | None = None,
        require_validation: bool = True,
    ):
        """
        Initialize extractor.
        
        Args:
            parser: Parser to use (creates default if None)
            require_validation: Whether to validate extracted commitments
        """
        self.parser = parser or CommitmentParser()
        self.require_validation = require_validation
    
    def extract(
        self,
        text: str,
        agent_id: str = "llm",
    ) -> list[Commitment]:
        """
        Extract and convert commitments from text.
        
        Args:
            text: Text to extract from
            agent_id: Agent ID for commitments
            
        Returns:
            List of formal Commitment objects
        """
        result = self.parser.parse(text)
        
        commitments = []
        for parsed in result.commitments:
            if self.require_validation:
                is_valid, issues = self.parser.validate_commitment(parsed)
                if not is_valid:
                    continue
            
            commitments.append(parsed.to_commitment(agent_id))
        
        return commitments
    
    def extract_with_metadata(
        self,
        text: str,
        agent_id: str = "llm",
    ) -> tuple[list[Commitment], ParsingResult]:
        """
        Extract commitments with full parsing metadata.
        
        Args:
            text: Text to extract from
            agent_id: Agent ID for commitments
            
        Returns:
            Tuple of (commitments, parsing result)
        """
        result = self.parser.parse(text)
        commitments = result.to_commitments(agent_id)
        return commitments, result
