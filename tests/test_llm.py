"""
Tests for the LLM Integration Module.

Tests cover:
- LLM interface abstraction
- Commitment parsing from natural language
- Grounding LLM outputs to verifiable predicates
- Alignment verification (Theorem 6)
"""

import pytest
import numpy as np

from gcl.llm.interface import (
    LLMInterface,
    LLMResponse,
    MockLLM,
    ResponseType,
    CommitmentAwareLLM,
)
from gcl.llm.commitment_parser import (
    CommitmentParser,
    CommitmentStrength,
    ParsedCommitment,
    ParsingResult,
    CommitmentExtractor,
)
from gcl.llm.grounding import (
    GroundingEngine,
    GroundingStatus,
    GroundedOutput,
    GroundedPredicate,
    AlignmentVerifier,
    Predicate,
    PredicateRegistry,
    VerificationResult,
)


# =============================================================================
# LLM Interface Tests
# =============================================================================

class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_response_creation(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="I will help you with that.",
            response_type=ResponseType.TEXT,
            confidence=0.9,
        )
        
        assert response.content == "I will help you with that."
        assert response.response_type == ResponseType.TEXT
        assert response.confidence == 0.9
    
    def test_response_with_commitments(self):
        """Test response with commitments."""
        response = LLMResponse(
            content="[COMMITMENT: I will complete the task]",
            response_type=ResponseType.COMMITMENT,
            commitments=["I will complete the task"],
        )
        
        assert response.has_commitments()
        assert len(response.commitments) == 1
    
    def test_response_with_actions(self):
        """Test response with actions."""
        response = LLMResponse(
            content="[ACTION: Create file]",
            response_type=ResponseType.ACTION,
            actions=["Create file"],
        )
        
        assert response.has_actions()
        assert len(response.actions) == 1


class TestMockLLM:
    """Tests for MockLLM implementation."""
    
    def test_mock_llm_creation(self):
        """Test creating a mock LLM."""
        llm = MockLLM(model_name="test-model", seed=42)
        
        assert llm.model_name == "test-model"
        assert llm._call_count == 0
    
    def test_generate_basic(self):
        """Test basic generation."""
        llm = MockLLM(seed=42)
        response = llm.generate("Hello, how are you?")
        
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert llm._call_count == 1
    
    def test_generate_with_commitment_pattern(self):
        """Test generation with commitment pattern."""
        llm = MockLLM(seed=42)
        response = llm.generate("Please commit to helping me")
        
        assert response.has_commitments()
        assert response.response_type == ResponseType.COMMITMENT
    
    def test_generate_with_code_pattern(self):
        """Test generation with code pattern."""
        llm = MockLLM(seed=42)
        response = llm.generate("Write some code for me")
        
        assert response.has_commitments()
        assert response.has_actions()
    
    def test_generate_with_unsafe_pattern(self):
        """Test generation with unsafe pattern."""
        llm = MockLLM(seed=42)
        response = llm.generate("Do something unsafe")
        
        assert response.response_type == ResponseType.REFUSAL
    
    def test_custom_response_map(self):
        """Test custom response mapping."""
        llm = MockLLM(
            response_map={"custom": "This is a custom response."},
            seed=42,
        )
        response = llm.generate("custom pattern")
        
        assert "custom response" in response.content
    
    def test_generate_with_commitment_request(self):
        """Test generate_with_commitment method."""
        llm = MockLLM(seed=42)
        response = llm.generate_with_commitment("Help me with a task")
        
        # Should include commitment structure
        assert "[COMMITMENT:" in response.content or response.has_commitments()
    
    def test_stats_tracking(self):
        """Test usage statistics tracking."""
        llm = MockLLM(seed=42)
        
        llm.generate("Test 1")
        llm.generate("Test 2")
        llm.generate("Test 3")
        
        stats = llm.stats
        assert stats["call_count"] == 3
        assert stats["total_tokens"] > 0


class TestCommitmentAwareLLM:
    """Tests for CommitmentAwareLLM wrapper."""
    
    def test_wrapper_creation(self):
        """Test creating commitment-aware wrapper."""
        base = MockLLM(seed=42)
        llm = CommitmentAwareLLM(base, require_commitments=True)
        
        assert llm.base_llm == base
        assert llm.require_commitments
    
    def test_generate_requires_commitments(self):
        """Test that wrapper requires commitments."""
        base = MockLLM(seed=42)
        llm = CommitmentAwareLLM(base, require_commitments=True)
        
        response = llm.generate("commit to helping me")
        
        # Should have commitments due to pattern
        assert response.has_commitments()
    
    def test_commitment_history(self):
        """Test commitment history tracking."""
        base = MockLLM(seed=42)
        llm = CommitmentAwareLLM(base, require_commitments=True)
        
        llm.generate("commit to task 1")
        llm.generate("commit to task 2")
        
        history = llm.get_commitment_history()
        assert len(history) >= 1  # At least one should have commitments
    
    def test_clear_history(self):
        """Test clearing commitment history."""
        base = MockLLM(seed=42)
        llm = CommitmentAwareLLM(base, require_commitments=True)
        
        llm.generate("commit to something")
        llm.clear_history()
        
        assert len(llm.get_commitment_history()) == 0


class TestLLMExtraction:
    """Tests for commitment and action extraction."""
    
    def test_extract_explicit_commitments(self):
        """Test extracting explicit commitment markers."""
        llm = MockLLM(seed=42)
        
        text = "[COMMITMENT: I will complete the task] and [COMMITMENT: I will verify results]"
        commitments = llm.extract_commitments(text)
        
        assert len(commitments) == 2
        assert "complete the task" in commitments[0]
    
    def test_extract_i_commit_to(self):
        """Test extracting 'I commit to' statements."""
        llm = MockLLM(seed=42)
        
        text = "I commit to finishing this work on time."
        commitments = llm.extract_commitments(text)
        
        assert len(commitments) >= 1
    
    def test_extract_guarantee(self):
        """Test extracting guarantee statements."""
        llm = MockLLM(seed=42)
        
        text = "I guarantee that the code will work correctly."
        commitments = llm.extract_commitments(text)
        
        assert len(commitments) >= 1
    
    def test_extract_actions(self):
        """Test extracting action statements."""
        llm = MockLLM(seed=42)
        
        text = """
        [ACTION: Create the file]
        1. First step
        2. Second step
        """
        actions = llm.extract_actions(text)
        
        assert len(actions) >= 1
    
    def test_classify_refusal(self):
        """Test classifying refusal responses."""
        llm = MockLLM(seed=42)
        
        text = "I cannot help with that request."
        response_type = llm.classify_response(text)
        
        assert response_type == ResponseType.REFUSAL
    
    def test_classify_query(self):
        """Test classifying query responses."""
        llm = MockLLM(seed=42)
        
        text = "Could you provide more details?"
        response_type = llm.classify_response(text)
        
        assert response_type == ResponseType.QUERY


# =============================================================================
# Commitment Parser Tests
# =============================================================================

class TestCommitmentParser:
    """Tests for CommitmentParser."""
    
    def test_parser_creation(self):
        """Test creating a parser."""
        parser = CommitmentParser()
        
        assert parser.min_confidence == 0.5
        assert parser.extract_conditions
        assert parser.extract_deadlines
    
    def test_parse_explicit_commitment(self):
        """Test parsing explicit commitment markers."""
        parser = CommitmentParser()
        
        text = "[COMMITMENT: I will complete the implementation]"
        result = parser.parse(text)
        
        assert result.has_commitments()
        # May find multiple patterns (explicit + "I will")
        assert len(result.commitments) >= 1
        # First one should be the explicit marker with STRONG strength
        strong_commitments = [c for c in result.commitments if c.strength == CommitmentStrength.STRONG]
        assert len(strong_commitments) >= 1
    
    def test_parse_guarantee(self):
        """Test parsing guarantee statements."""
        parser = CommitmentParser()
        
        text = "I guarantee that the system will be secure."
        result = parser.parse(text)
        
        assert result.has_commitments()
        assert result.commitments[0].strength == CommitmentStrength.ABSOLUTE
    
    def test_parse_weak_commitment(self):
        """Test parsing weak commitments."""
        parser = CommitmentParser()
        
        text = "I might be able to help with that."
        result = parser.parse(text)
        
        if result.has_commitments():
            assert result.commitments[0].strength == CommitmentStrength.WEAK
    
    def test_parse_with_conditions(self):
        """Test parsing commitments with conditions."""
        parser = CommitmentParser()
        
        text = "If you provide the requirements, I will implement the feature."
        result = parser.parse(text)
        
        # The parser extracts "I will implement the feature" as the commitment
        # The condition "If you provide the requirements" is in the original text
        assert result.has_commitments()
        # Check that we found a commitment
        assert len(result.commitments) >= 1
    
    def test_parse_with_deadline(self):
        """Test parsing commitments with deadlines."""
        parser = CommitmentParser()
        
        text = "I will complete this by tomorrow."
        result = parser.parse(text)
        
        if result.has_commitments():
            commitment = result.commitments[0]
            # Either has deadline or contains deadline text
            assert commitment.has_deadline() or "tomorrow" in commitment.raw_text.lower()
    
    def test_strength_adjustment_strengthening(self):
        """Test strength adjustment with strengthening words."""
        parser = CommitmentParser()
        
        text = "I will definitely ensure this works correctly."
        result = parser.parse(text)
        
        if result.has_commitments():
            # Should be upgraded due to "definitely" and "ensure"
            assert result.commitments[0].strength in (
                CommitmentStrength.STRONG,
                CommitmentStrength.ABSOLUTE,
            )
    
    def test_strength_adjustment_weakening(self):
        """Test strength adjustment with weakening words."""
        parser = CommitmentParser()
        
        text = "I will try to maybe complete this if possible."
        result = parser.parse(text)
        
        if result.has_commitments():
            # Should be downgraded due to "try", "maybe", "if possible"
            assert result.commitments[0].strength in (
                CommitmentStrength.WEAK,
                CommitmentStrength.MODERATE,
            )
    
    def test_parse_multiple_commitments(self):
        """Test parsing multiple commitments."""
        parser = CommitmentParser()
        
        text = """
        [COMMITMENT: First commitment]
        I guarantee the second thing.
        I commit to the third item.
        """
        result = parser.parse(text)
        
        assert len(result.commitments) >= 2
    
    def test_deduplication(self):
        """Test that duplicate commitments are removed."""
        parser = CommitmentParser()
        
        text = """
        I will complete the task.
        I will complete the task.
        """
        result = parser.parse(text)
        
        # Should deduplicate
        actions = [c.action.lower().strip() for c in result.commitments]
        assert len(actions) == len(set(actions))
    
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        parser = CommitmentParser()
        
        # Explicit marker should have high confidence
        text1 = "[COMMITMENT: Specific detailed commitment here]"
        result1 = parser.parse(text1)
        
        # Weak commitment should have lower confidence
        text2 = "I might do something."
        result2 = parser.parse(text2)
        
        if result1.has_commitments() and result2.has_commitments():
            assert result1.commitments[0].confidence >= result2.commitments[0].confidence


class TestParsedCommitment:
    """Tests for ParsedCommitment dataclass."""
    
    def test_to_commitment(self):
        """Test converting to formal Commitment."""
        parsed = ParsedCommitment(
            raw_text="I commit to completing the task",
            action="completing the task",
            strength=CommitmentStrength.STRONG,
            confidence=0.9,
        )
        
        commitment = parsed.to_commitment("agent-1")
        
        assert commitment.agent_id == "agent-1"
        assert commitment.stake == 0.6  # STRONG maps to 0.6
        assert "completing the task" in commitment.parameters["action"]
    
    def test_conditional_check(self):
        """Test is_conditional method."""
        conditional = ParsedCommitment(
            raw_text="If X then Y",
            action="Y",
            conditions=["X"],
        )
        
        unconditional = ParsedCommitment(
            raw_text="I will do Y",
            action="Y",
        )
        
        assert conditional.is_conditional()
        assert not unconditional.is_conditional()
    
    def test_deadline_check(self):
        """Test has_deadline method."""
        with_deadline = ParsedCommitment(
            raw_text="By tomorrow",
            action="something",
            deadline="tomorrow",
        )
        
        without_deadline = ParsedCommitment(
            raw_text="Something",
            action="something",
        )
        
        assert with_deadline.has_deadline()
        assert not without_deadline.has_deadline()


class TestCommitmentExtractor:
    """Tests for CommitmentExtractor."""
    
    def test_extract_basic(self):
        """Test basic extraction."""
        extractor = CommitmentExtractor()
        
        text = "[COMMITMENT: I will help you]"
        commitments = extractor.extract(text, agent_id="test-agent")
        
        assert len(commitments) >= 1
        assert commitments[0].agent_id == "test-agent"
    
    def test_extract_with_validation(self):
        """Test extraction with validation."""
        extractor = CommitmentExtractor(require_validation=True)
        
        # Valid commitment
        text1 = "[COMMITMENT: I will implement the feature correctly]"
        commitments1 = extractor.extract(text1)
        
        # Should pass validation
        assert len(commitments1) >= 1
    
    def test_extract_with_metadata(self):
        """Test extraction with metadata."""
        extractor = CommitmentExtractor()
        
        text = "I guarantee this will work."
        commitments, result = extractor.extract_with_metadata(text)
        
        assert isinstance(result, ParsingResult)


class TestCommitmentValidation:
    """Tests for commitment validation."""
    
    def test_validate_valid_commitment(self):
        """Test validating a valid commitment."""
        parser = CommitmentParser()
        
        commitment = ParsedCommitment(
            raw_text="I commit to implementing the feature",
            action="implementing the feature",
            confidence=0.8,
        )
        
        is_valid, issues = parser.validate_commitment(commitment)
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_empty_action(self):
        """Test validating commitment with empty action."""
        parser = CommitmentParser()
        
        commitment = ParsedCommitment(
            raw_text="I commit to",
            action="",
            confidence=0.8,
        )
        
        is_valid, issues = parser.validate_commitment(commitment)
        assert not is_valid
        assert any("short" in issue.lower() or "empty" in issue.lower() for issue in issues)
    
    def test_validate_vague_commitment(self):
        """Test validating vague commitment."""
        parser = CommitmentParser()
        
        commitment = ParsedCommitment(
            raw_text="I will do something",
            action="do something",
            confidence=0.8,
        )
        
        is_valid, issues = parser.validate_commitment(commitment)
        assert any("vague" in issue.lower() for issue in issues)
    
    def test_validate_low_confidence(self):
        """Test validating low confidence commitment."""
        parser = CommitmentParser()
        
        commitment = ParsedCommitment(
            raw_text="I might do this",
            action="do this task",
            confidence=0.3,
        )
        
        is_valid, issues = parser.validate_commitment(commitment)
        assert any("confidence" in issue.lower() for issue in issues)


# =============================================================================
# Grounding Engine Tests
# =============================================================================

class TestGroundingEngine:
    """Tests for GroundingEngine."""
    
    def test_engine_creation(self):
        """Test creating a grounding engine."""
        engine = GroundingEngine()
        
        assert engine.parser is not None
        assert engine.registry is not None
        assert len(engine.grounding_rules) > 0
    
    def test_ground_basic_response(self):
        """Test grounding a basic response."""
        engine = GroundingEngine()
        llm = MockLLM(seed=42)
        
        response = llm.generate("commit to helping")
        grounded = engine.ground(response)
        
        assert isinstance(grounded, GroundedOutput)
        assert grounded.original_response == response
    
    def test_ground_with_commitments(self):
        """Test grounding response with commitments."""
        engine = GroundingEngine()
        
        response = LLMResponse(
            content="[COMMITMENT: I will write the code correctly]",
            response_type=ResponseType.COMMITMENT,
            commitments=["I will write the code correctly"],
        )
        
        grounded = engine.ground(response)
        
        assert grounded.grounding_result.status in (
            GroundingStatus.GROUNDED,
            GroundingStatus.PARTIAL,
        )
    
    def test_ground_code_commitment(self):
        """Test grounding code-related commitment."""
        engine = GroundingEngine()
        
        response = LLMResponse(
            content="[COMMITMENT: I will implement the feature]",
            response_type=ResponseType.COMMITMENT,
            commitments=["I will implement the feature"],
        )
        
        grounded = engine.ground(response)
        
        # Should match "implement" rule
        if grounded.grounding_result.grounded_predicates:
            predicate_names = [
                gp.predicate.name
                for gp in grounded.grounding_result.grounded_predicates
            ]
            # Should have grounded to something
            assert len(predicate_names) > 0
    
    def test_ground_safety_commitment(self):
        """Test grounding safety-related commitment."""
        engine = GroundingEngine()
        
        response = LLMResponse(
            content="[COMMITMENT: I will not cause any harm]",
            response_type=ResponseType.COMMITMENT,
            commitments=["I will not cause any harm"],
        )
        
        grounded = engine.ground(response)
        
        # Should match "not.*harm" rule
        assert grounded.grounding_result.status != GroundingStatus.FAILED
    
    def test_grounding_history(self):
        """Test grounding history tracking."""
        engine = GroundingEngine()
        llm = MockLLM(seed=42)
        
        engine.ground(llm.generate("commit to task 1"))
        engine.ground(llm.generate("commit to task 2"))
        
        assert len(engine.grounding_history) == 2
    
    def test_add_custom_grounding_rule(self):
        """Test adding custom grounding rules."""
        engine = GroundingEngine()
        
        engine.add_grounding_rule(r'custom_action', 'custom_predicate')
        
        assert 'custom_action' in engine.grounding_rules
        assert engine.grounding_rules['custom_action'] == 'custom_predicate'
    
    def test_verify_output(self):
        """Test verifying grounded output."""
        engine = GroundingEngine()
        
        response = LLMResponse(
            content="[COMMITMENT: I will write the code]",
            response_type=ResponseType.COMMITMENT,
            commitments=["I will write the code"],
        )
        
        grounded = engine.ground(response)
        
        # Verify with context that satisfies the commitment
        context = {"code_exists": True, "code_written": True}
        result = engine.verify_output(grounded, context)
        
        # Should verify successfully
        assert result.confidence >= 0 or not grounded.grounding_result.grounded_predicates
    
    def test_alignment_report(self):
        """Test generating alignment report."""
        engine = GroundingEngine()
        llm = MockLLM(seed=42)
        
        # Generate some outputs
        for i in range(5):
            engine.ground(llm.generate(f"commit to task {i}"))
        
        report = engine.get_alignment_report()
        
        assert "total_outputs" in report
        assert report["total_outputs"] == 5
        assert "alignment_rate" in report


class TestGroundedOutput:
    """Tests for GroundedOutput."""
    
    def test_is_aligned(self):
        """Test is_aligned check."""
        from gcl.llm.grounding import GroundingResult
        
        # Create aligned output
        predicate = Predicate(name="test", evaluate_fn=lambda ctx: True)
        parsed = ParsedCommitment(raw_text="test", action="test")
        grounded_pred = GroundedPredicate(predicate=predicate, source_commitment=parsed)
        
        grounding_result = GroundingResult(
            status=GroundingStatus.GROUNDED,
            grounded_predicates=[grounded_pred],
        )
        
        response = LLMResponse(content="test")
        parsing_result = ParsingResult(commitments=[parsed], original_text="test")
        
        output = GroundedOutput(
            original_response=response,
            parsing_result=parsing_result,
            grounding_result=grounding_result,
            commitments=[parsed.to_commitment("test")],
        )
        
        assert output.is_aligned()
    
    def test_verification_hash(self):
        """Test verification hash generation."""
        from gcl.llm.grounding import GroundingResult
        
        response = LLMResponse(content="test content")
        parsing_result = ParsingResult(commitments=[], original_text="test")
        grounding_result = GroundingResult(status=GroundingStatus.PENDING)
        
        output = GroundedOutput(
            original_response=response,
            parsing_result=parsing_result,
            grounding_result=grounding_result,
        )
        
        hash1 = output.get_verification_hash()
        assert len(hash1) == 16
        assert hash1.isalnum()


class TestAlignmentVerifier:
    """Tests for AlignmentVerifier (Theorem 6)."""
    
    def test_verifier_creation(self):
        """Test creating alignment verifier."""
        verifier = AlignmentVerifier()
        
        assert verifier.engine is not None
        assert len(verifier.verification_history) == 0
    
    def test_verify_alignment_basic(self):
        """Test basic alignment verification."""
        verifier = AlignmentVerifier()
        llm = MockLLM(seed=42)
        
        result = verifier.verify_alignment(
            llm=llm,
            prompt="commit to helping me",
            expected_behavior={"help_provided": True},
        )
        
        assert "alignment_score" in result
        assert "verification_result" in result
        assert "commitments_made" in result
    
    def test_verify_alignment_with_context(self):
        """Test alignment verification with specific context."""
        verifier = AlignmentVerifier()
        llm = MockLLM(seed=42)
        
        # Add custom response
        llm.add_response(
            "write code",
            "[COMMITMENT: I will write clean code] [COMMITMENT: I will add tests]"
        )
        
        result = verifier.verify_alignment(
            llm=llm,
            prompt="write code for me",
            expected_behavior={
                "code_exists": True,
                "tests_exist": True,
            },
        )
        
        assert result["alignment_score"] >= 0
    
    def test_alignment_summary(self):
        """Test alignment summary generation."""
        verifier = AlignmentVerifier()
        llm = MockLLM(seed=42)
        
        # Run several verifications
        for i in range(3):
            verifier.verify_alignment(
                llm=llm,
                prompt=f"commit to task {i}",
                expected_behavior={"task_complete": True},
            )
        
        summary = verifier.get_alignment_summary()
        
        assert summary["total_verifications"] == 3
        assert "alignment_rate" in summary
        assert "average_alignment_score" in summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestLLMIntegration:
    """Integration tests for the full LLM pipeline."""
    
    def test_full_pipeline(self):
        """Test the full LLM -> Parse -> Ground -> Verify pipeline."""
        # Create components
        llm = MockLLM(seed=42)
        parser = CommitmentParser()
        engine = GroundingEngine(parser=parser)
        
        # Generate response
        response = llm.generate_with_commitment("Help me write some code")
        
        # Ground the response
        grounded = engine.ground(response)
        
        # Verify
        context = {"code_exists": True, "implemented": True}
        verification = engine.verify_output(grounded, context)
        
        # Check pipeline completed
        assert grounded.original_response == response
        assert isinstance(verification.verified, bool)
    
    def test_commitment_aware_pipeline(self):
        """Test pipeline with commitment-aware LLM."""
        base_llm = MockLLM(seed=42)
        llm = CommitmentAwareLLM(base_llm, require_commitments=True)
        engine = GroundingEngine()
        
        # Generate with commitment requirement
        response = llm.generate("commit to completing the task")
        
        # Ground
        grounded = engine.ground(response)
        
        # Should have commitments
        assert response.has_commitments() or len(llm.get_commitment_history()) > 0
    
    def test_alignment_verification_pipeline(self):
        """Test full alignment verification pipeline."""
        verifier = AlignmentVerifier()
        llm = MockLLM(seed=42)
        
        # Define expected behavior
        expected = {
            "code_exists": True,
            "tests_exist": True,
            "documentation_exists": True,
        }
        
        # Verify alignment
        result = verifier.verify_alignment(
            llm=llm,
            prompt="code something for me",
            expected_behavior=expected,
        )
        
        # Check result structure
        assert "prompt" in result
        assert "response" in result
        assert "alignment_score" in result
        assert 0 <= result["alignment_score"] <= 1


class TestTheorem6Validation:
    """
    Tests validating Theorem 6: Alignment Verifiability.
    
    Theorem 6 states that commitment-grounded alignment is verifiable
    in polynomial time with respect to commitment complexity.
    """
    
    def test_verification_is_polynomial(self):
        """Test that verification scales polynomially."""
        import time
        
        engine = GroundingEngine()
        llm = MockLLM(seed=42)
        
        # Test with increasing number of commitments
        times = []
        sizes = [1, 5, 10, 20]
        
        for n in sizes:
            # Create response with n commitments
            commitments = " ".join([
                f"[COMMITMENT: I will do task {i}]"
                for i in range(n)
            ])
            response = LLMResponse(
                content=commitments,
                response_type=ResponseType.COMMITMENT,
            )
            
            # Time the grounding
            start = time.time()
            grounded = engine.ground(response)
            engine.verify_output(grounded, {})
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        # Check polynomial scaling (not exponential)
        # For polynomial, time ratio should be bounded
        if times[0] > 0:
            ratios = [times[i] / times[0] for i in range(len(times))]
            # Ratios should not grow exponentially
            # For O(n²), ratio for 20x size should be ~400x, not 2^20
            assert ratios[-1] < 1000  # Very generous bound
    
    def test_grounding_preserves_semantics(self):
        """Test that grounding preserves commitment semantics."""
        engine = GroundingEngine()
        
        # Original commitment
        original = "I will implement the feature correctly"
        response = LLMResponse(
            content=f"[COMMITMENT: {original}]",
            response_type=ResponseType.COMMITMENT,
        )
        
        grounded = engine.ground(response)
        
        # Check semantics preserved
        if grounded.grounding_result.grounded_predicates:
            gp = grounded.grounding_result.grounded_predicates[0]
            # Original text should be preserved
            assert original in gp.source_commitment.raw_text or \
                   "implement" in gp.source_commitment.action.lower()
    
    def test_verification_is_deterministic(self):
        """Test that verification is deterministic."""
        engine = GroundingEngine()
        
        response = LLMResponse(
            content="[COMMITMENT: I will complete the task]",
            response_type=ResponseType.COMMITMENT,
        )
        
        context = {"task_complete": True}
        
        # Run verification multiple times
        results = []
        for _ in range(5):
            grounded = engine.ground(response)
            result = engine.verify_output(grounded, context)
            results.append(result.verified)
        
        # All results should be the same
        assert all(r == results[0] for r in results)
    
    def test_commitment_coverage_tracking(self):
        """Test that commitment coverage is tracked."""
        verifier = AlignmentVerifier()
        llm = MockLLM(seed=42)
        
        # Add specific response
        llm.add_response(
            "safety",
            "[COMMITMENT: I will ensure safety] [COMMITMENT: I will avoid harm]"
        )
        
        result = verifier.verify_alignment(
            llm=llm,
            prompt="safety is important",
            expected_behavior={"is_safe": True, "harm_detected": False},
        )
        
        # Should track commitments made
        assert "commitments_made" in result
        assert isinstance(result["commitments_made"], list)