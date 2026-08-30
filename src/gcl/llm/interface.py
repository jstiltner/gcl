"""
LLM Interface Abstraction.

This module provides abstract interfaces for integrating different LLM providers
with the GCL framework. The key insight is that LLMs can be viewed as
commitment-generating systems that need grounding.

Key Classes:
- LLMInterface: Abstract base class for LLM providers
- LLMResponse: Structured response from LLM
- MockLLM: Testing implementation with deterministic behavior
- OpenAILLM: OpenAI API integration (optional)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np


class ResponseType(Enum):
    """Type of LLM response."""
    
    TEXT = "text"
    COMMITMENT = "commitment"
    ACTION = "action"
    QUERY = "query"
    REFUSAL = "refusal"


@dataclass
class LLMResponse:
    """
    Structured response from an LLM.
    
    Attributes:
        content: Raw text content of the response
        response_type: Classified type of response
        confidence: Model's confidence in the response (0-1)
        metadata: Additional metadata (tokens, latency, etc.)
        commitments: Extracted commitment statements
        actions: Proposed actions
    """
    
    content: str
    response_type: ResponseType = ResponseType.TEXT
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    commitments: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    
    def has_commitments(self) -> bool:
        """Check if response contains commitments."""
        return len(self.commitments) > 0
    
    def has_actions(self) -> bool:
        """Check if response proposes actions."""
        return len(self.actions) > 0


class LLMInterface(ABC):
    """
    Abstract interface for LLM providers.
    
    This interface allows the GCL framework to work with different LLM
    backends while maintaining consistent commitment extraction and grounding.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize LLM interface.
        
        Args:
            model_name: Name/identifier of the model
        """
        self.model_name = model_name
        self._call_count = 0
        self._total_tokens = 0
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Structured LLM response
        """
        pass
    
    @abstractmethod
    def generate_with_commitment(
        self,
        prompt: str,
        commitment_template: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response that includes explicit commitments.
        
        This method prompts the LLM to structure its response with
        explicit commitment statements that can be verified.
        
        Args:
            prompt: User prompt/query
            commitment_template: Optional template for commitment format
            **kwargs: Additional arguments
            
        Returns:
            Response with extracted commitments
        """
        pass
    
    def extract_commitments(self, text: str) -> list[str]:
        """
        Extract commitment statements from text.
        
        Looks for patterns like:
        - "I commit to..."
        - "I will..."
        - "I guarantee..."
        - "[COMMITMENT: ...]"
        
        Args:
            text: Text to extract commitments from
            
        Returns:
            List of commitment statements
        """
        commitments = []
        
        # Pattern 1: Explicit commitment markers
        explicit_pattern = r'\[COMMITMENT:\s*([^\]]+)\]'
        commitments.extend(re.findall(explicit_pattern, text, re.IGNORECASE))
        
        # Pattern 2: "I commit to" statements
        commit_pattern = r'I commit to\s+([^.!?\n]+[.!?]?)'
        commitments.extend(re.findall(commit_pattern, text, re.IGNORECASE))
        
        # Pattern 3: "I will" statements (strong form)
        will_pattern = r'I will\s+([^.!?\n]+[.!?]?)'
        will_matches = re.findall(will_pattern, text, re.IGNORECASE)
        # Filter to keep only strong commitments
        for match in will_matches:
            if any(word in match.lower() for word in ['ensure', 'guarantee', 'always', 'never']):
                commitments.append(match)
        
        # Pattern 4: "I guarantee" statements
        guarantee_pattern = r'I guarantee\s+([^.!?\n]+[.!?]?)'
        commitments.extend(re.findall(guarantee_pattern, text, re.IGNORECASE))
        
        return [c.strip() for c in commitments if c.strip()]
    
    def extract_actions(self, text: str) -> list[str]:
        """
        Extract proposed actions from text.
        
        Args:
            text: Text to extract actions from
            
        Returns:
            List of action statements
        """
        actions = []
        
        # Pattern 1: Explicit action markers
        explicit_pattern = r'\[ACTION:\s*([^\]]+)\]'
        actions.extend(re.findall(explicit_pattern, text, re.IGNORECASE))
        
        # Pattern 2: Numbered steps
        step_pattern = r'(?:^|\n)\s*\d+[.)]\s*([^\n]+)'
        actions.extend(re.findall(step_pattern, text))
        
        # Pattern 3: Bullet points with action verbs
        bullet_pattern = r'(?:^|\n)\s*[-•*]\s*([A-Z][^\n]+)'
        bullet_matches = re.findall(bullet_pattern, text)
        for match in bullet_matches:
            # Check if starts with action verb
            action_verbs = ['create', 'update', 'delete', 'send', 'check', 'verify', 
                          'run', 'execute', 'call', 'fetch', 'save', 'load']
            if any(match.lower().startswith(verb) for verb in action_verbs):
                actions.append(match)
        
        return [a.strip() for a in actions if a.strip()]
    
    def classify_response(self, text: str) -> ResponseType:
        """
        Classify the type of response.
        
        Args:
            text: Response text to classify
            
        Returns:
            Response type classification
        """
        text_lower = text.lower()
        
        # Check for refusal
        refusal_phrases = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i won't", "i will not", "i refuse", "not allowed",
            "against my", "violates my"
        ]
        if any(phrase in text_lower for phrase in refusal_phrases):
            return ResponseType.REFUSAL
        
        # Check for commitments
        if self.extract_commitments(text):
            return ResponseType.COMMITMENT
        
        # Check for actions
        if self.extract_actions(text):
            return ResponseType.ACTION
        
        # Check for queries
        if text.strip().endswith('?'):
            return ResponseType.QUERY
        
        return ResponseType.TEXT
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "model": self.model_name,
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
        }


class MockLLM(LLMInterface):
    """
    Mock LLM for testing.
    
    Provides deterministic responses based on patterns in the prompt,
    useful for testing commitment extraction and grounding without
    requiring actual LLM API calls.
    """
    
    def __init__(
        self,
        model_name: str = "mock-llm",
        response_map: dict[str, str] | None = None,
        default_response: str = "I understand your request.",
        seed: int | None = None,
    ):
        """
        Initialize mock LLM.
        
        Args:
            model_name: Name for the mock model
            response_map: Mapping from prompt patterns to responses
            default_response: Default response when no pattern matches
            seed: Random seed for reproducibility
        """
        super().__init__(model_name)
        self.response_map = response_map or {}
        self.default_response = default_response
        self.rng = np.random.default_rng(seed)
        
        # Add default commitment-aware responses
        self._setup_default_responses()
    
    def _setup_default_responses(self) -> None:
        """Set up default response patterns."""
        defaults = {
            "commit": (
                "I understand your request. "
                "[COMMITMENT: I will complete the task as specified] "
                "[COMMITMENT: I will verify my work before submission]"
            ),
            "help": (
                "I can help you with that. Here's what I'll do:\n"
                "1. Analyze your requirements\n"
                "2. Propose a solution\n"
                "3. Implement the changes\n"
                "[COMMITMENT: I will explain each step clearly]"
            ),
            "code": (
                "I'll write the code for you.\n"
                "[COMMITMENT: The code will follow best practices]\n"
                "[COMMITMENT: I will include error handling]\n"
                "[ACTION: Create the implementation]\n"
                "[ACTION: Add unit tests]"
            ),
            "unsafe": (
                "I cannot help with that request as it could cause harm. "
                "I'm designed to be helpful while avoiding potentially "
                "dangerous or unethical actions."
            ),
            "question": (
                "That's a great question. Let me think about it.\n"
                "Could you provide more context about your specific use case?"
            ),
        }
        
        # Only add defaults if not already specified
        for key, value in defaults.items():
            if key not in self.response_map:
                self.response_map[key] = value
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock response."""
        self._call_count += 1
        
        # Find matching response
        response_text = self.default_response
        prompt_lower = prompt.lower()
        
        for pattern, response in self.response_map.items():
            if pattern.lower() in prompt_lower:
                response_text = response
                break
        
        # Add some randomness based on temperature
        if temperature > 0.5 and self.rng.random() < 0.1:
            response_text += "\n\nIs there anything else I can help with?"
        
        # Extract commitments and actions
        commitments = self.extract_commitments(response_text)
        actions = self.extract_actions(response_text)
        response_type = self.classify_response(response_text)
        
        # Estimate tokens
        tokens = len(response_text.split())
        self._total_tokens += tokens
        
        return LLMResponse(
            content=response_text,
            response_type=response_type,
            confidence=0.9 if commitments else 0.7,
            metadata={
                "tokens": tokens,
                "temperature": temperature,
                "model": self.model_name,
            },
            commitments=commitments,
            actions=actions,
        )
    
    def generate_with_commitment(
        self,
        prompt: str,
        commitment_template: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with explicit commitment structure."""
        # Modify prompt to request commitments
        enhanced_prompt = (
            f"{prompt}\n\n"
            "Please structure your response with explicit commitments using "
            "the format [COMMITMENT: your commitment here]."
        )
        
        if commitment_template:
            enhanced_prompt += f"\n\nCommitment template: {commitment_template}"
        
        return self.generate(enhanced_prompt, **kwargs)
    
    def add_response(self, pattern: str, response: str) -> None:
        """
        Add a custom response pattern.
        
        Args:
            pattern: Pattern to match in prompts
            response: Response to return when pattern matches
        """
        self.response_map[pattern] = response


class OpenAILLM(LLMInterface):
    """
    OpenAI API integration.
    
    Requires openai package and API key to be configured.
    Falls back to MockLLM if openai is not available.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: str | None = None,
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (uses env var if not provided)
        """
        super().__init__(model_name)
        self.api_key = api_key
        self._client = None
        self._available = False
        
        try:
            import openai
            if api_key:
                self._client = openai.OpenAI(api_key=api_key)
            else:
                self._client = openai.OpenAI()
            self._available = True
        except (ImportError, Exception):
            # OpenAI not available, will use mock fallback
            self._mock = MockLLM(model_name=f"mock-{model_name}")
    
    @property
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return self._available
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        self._call_count += 1
        
        if not self._available:
            return self._mock.generate(
                prompt, system_prompt, temperature, max_tokens, **kwargs
            )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            self._total_tokens += tokens
            
            commitments = self.extract_commitments(content)
            actions = self.extract_actions(content)
            response_type = self.classify_response(content)
            
            return LLMResponse(
                content=content,
                response_type=response_type,
                confidence=0.9,
                metadata={
                    "tokens": tokens,
                    "model": self.model_name,
                    "finish_reason": response.choices[0].finish_reason,
                },
                commitments=commitments,
                actions=actions,
            )
            
        except Exception as e:
            # Fall back to mock on error
            return LLMResponse(
                content=f"Error calling OpenAI API: {e}",
                response_type=ResponseType.TEXT,
                confidence=0.0,
                metadata={"error": str(e)},
            )
    
    def generate_with_commitment(
        self,
        prompt: str,
        commitment_template: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with commitment structure."""
        system_prompt = (
            "You are a helpful AI assistant that makes explicit commitments. "
            "When responding, clearly state your commitments using the format: "
            "[COMMITMENT: your specific commitment]. "
            "Be precise and verifiable in your commitments."
        )
        
        if commitment_template:
            system_prompt += f"\n\nUse this commitment template: {commitment_template}"
        
        return self.generate(
            prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


class CommitmentAwareLLM(LLMInterface):
    """
    LLM wrapper that enforces commitment-based responses.
    
    This wrapper ensures all LLM outputs include verifiable commitments
    and tracks commitment fulfillment over time.
    """
    
    def __init__(
        self,
        base_llm: LLMInterface,
        require_commitments: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize commitment-aware LLM.
        
        Args:
            base_llm: Underlying LLM to wrap
            require_commitments: Whether to require commitments in responses
            max_retries: Max retries to get commitment-containing response
        """
        super().__init__(f"commitment-aware-{base_llm.model_name}")
        self.base_llm = base_llm
        self.require_commitments = require_commitments
        self.max_retries = max_retries
        self.commitment_history: list[dict[str, Any]] = []
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response, optionally requiring commitments."""
        if not self.require_commitments:
            return self.base_llm.generate(
                prompt, system_prompt, temperature, max_tokens, **kwargs
            )
        
        # Try to get response with commitments
        for attempt in range(self.max_retries):
            response = self.base_llm.generate_with_commitment(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            if response.has_commitments():
                # Record commitments
                self.commitment_history.append({
                    "prompt": prompt,
                    "commitments": response.commitments,
                    "timestamp": np.datetime64('now'),
                })
                return response
            
            # Increase temperature slightly for retry
            temperature = min(1.0, temperature + 0.1)
        
        # Return last response even without commitments
        return response
    
    def generate_with_commitment(
        self,
        prompt: str,
        commitment_template: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with explicit commitment request."""
        return self.base_llm.generate_with_commitment(
            prompt, commitment_template, **kwargs
        )
    
    def get_commitment_history(self) -> list[dict[str, Any]]:
        """Get history of commitments made."""
        return self.commitment_history.copy()
    
    def clear_history(self) -> None:
        """Clear commitment history."""
        self.commitment_history.clear()
