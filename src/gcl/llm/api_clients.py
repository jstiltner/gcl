"""
Unified API clients for LLM experiments.
Supports: Claude (Anthropic), GPT-4 (OpenAI)

This module provides a consistent interface for interacting with different
LLM providers, with caching to minimize API costs during experiments.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class LLMResponse:
    """Structured response from an LLM API call."""
    content: str
    model: str
    usage: Dict[str, int]  # tokens
    raw_response: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "raw_response": None  # Don't serialize raw response
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMResponse":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            model=data["model"],
            usage=data["usage"],
            raw_response=data.get("raw_response")
        )


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    @abstractmethod
    def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None, 
        **kwargs
    ) -> LLMResponse:
        """Generate a completion for the given prompt."""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for semantic comparison."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model to use (default: claude-sonnet-4-6)
        """
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = model
            self._available = True
        except ImportError:
            self._available = False
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            self._available = False
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None, 
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Claude."""
        messages = [{"role": "user", "content": prompt}]
        
        request = dict(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            system=system or "You are a helpful assistant.",
            messages=messages,
        )
        if "temperature" in kwargs:
            request["temperature"] = kwargs["temperature"]
        try:
            response = self.client.messages.create(**request)
        except TypeError:
            # Newer anthropic SDK versions dropped the temperature parameter.
            request.pop("temperature", None)
            response = self.client.messages.create(**request)
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "input": response.usage.input_tokens, 
                "output": response.usage.output_tokens
            },
            raw_response=response
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.
        
        Note: Anthropic doesn't have a native embeddings API.
        Use OpenAI or Voyage AI for embeddings.
        """
        raise NotImplementedError(
            "Anthropic doesn't have embeddings API. Use OpenAI or Voyage for embeddings."
        )


class OpenAIClient(LLMClient):
    """OpenAI GPT API client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",  # Latest GPT-4o model
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
            model: Chat model to use (default: gpt-4o)
            embedding_model: Embedding model (default: text-embedding-3-small)
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
            self.embedding_model = embedding_model
            self._available = True
        except ImportError:
            self._available = False
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            self._available = False
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None, 
        **kwargs
    ) -> LLMResponse:
        """Generate completion using GPT."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "input": response.usage.prompt_tokens, 
                "output": response.usage.completion_tokens
            },
            raw_response=response
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding


class ResponseCache:
    """
    Cache LLM responses to avoid redundant API costs.
    
    Caches are stored as JSON files keyed by hash of (model, system, prompt).
    """
    
    def __init__(self, cache_dir: Path = Path(".cache/llm")):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cached responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
    
    def _cache_key(self, model: str, prompt: str, system: Optional[str]) -> str:
        """Generate cache key from request parameters."""
        content = f"{model}:{system or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        Get cached response if available.
        
        Args:
            model: Model name
            prompt: User prompt
            system: System prompt
            
        Returns:
            Cached LLMResponse or None if not cached
        """
        key = self._cache_key(model, prompt, system)
        cache_path = self.cache_dir / f"{key}.json"
        
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                self._hits += 1
                return LLMResponse.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                # Invalid cache entry, remove it
                cache_path.unlink()
        
        self._misses += 1
        return None
    
    def set(
        self, 
        model: str, 
        prompt: str, 
        response: LLMResponse, 
        system: Optional[str] = None
    ) -> None:
        """
        Cache a response.
        
        Args:
            model: Model name
            prompt: User prompt
            response: Response to cache
            system: System prompt
        """
        key = self._cache_key(model, prompt, system)
        cache_path = self.cache_dir / f"{key}.json"
        cache_path.write_text(json.dumps(response.to_dict(), indent=2))
    
    def clear(self) -> int:
        """
        Clear all cached responses.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        }


class CachedClient(LLMClient):
    """
    Wrapper that adds caching to any LLM client.
    """
    
    def __init__(self, client: LLMClient, cache: Optional[ResponseCache] = None):
        """
        Initialize cached client.
        
        Args:
            client: Underlying LLM client
            cache: Cache to use (creates new one if not provided)
        """
        self._client = client
        self._cache = cache or ResponseCache()
    
    @property
    def model_name(self) -> str:
        return self._client.model_name
    
    def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None, 
        **kwargs
    ) -> LLMResponse:
        """Generate completion, using cache if available."""
        # Check cache first
        cached = self._cache.get(self._client.model_name, prompt, system)
        if cached:
            return cached
        
        # Call API
        response = self._client.complete(prompt, system, **kwargs)
        
        # Cache response
        self._cache.set(self._client.model_name, prompt, response, system)
        
        return response
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding (not cached - embeddings are cheap)."""
        return self._client.get_embedding(text)
    
    @property
    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._cache.stats


def get_client(
    provider: str, 
    api_key: Optional[str] = None,
    use_cache: bool = True
) -> LLMClient:
    """
    Factory function for LLM clients.
    
    Args:
        provider: Provider name ("anthropic", "claude", "openai", "gpt")
        api_key: Optional API key (uses env var if not provided)
        use_cache: Whether to wrap client with caching
        
    Returns:
        Configured LLM client
    """
    provider_lower = provider.lower()
    
    if provider_lower in ["anthropic", "claude"]:
        client = AnthropicClient(api_key)
    elif provider_lower in ["openai", "gpt", "gpt-4", "gpt4"]:
        client = OpenAIClient(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")
    
    if use_cache:
        return CachedClient(client)
    
    return client


def check_api_keys() -> Dict[str, bool]:
    """
    Check which API keys are available.
    
    Returns:
        Dictionary mapping provider to availability
    """
    return {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
    }


def get_available_clients(use_cache: bool = True) -> Dict[str, LLMClient]:
    """
    Get all available LLM clients based on configured API keys.
    
    Args:
        use_cache: Whether to use response caching
        
    Returns:
        Dictionary mapping provider name to client
    """
    clients = {}
    keys = check_api_keys()
    
    if keys["anthropic"]:
        try:
            clients["claude"] = get_client("anthropic", use_cache=use_cache)
        except Exception as e:
            print(f"Warning: Could not initialize Anthropic client: {e}")
    
    if keys["openai"]:
        try:
            clients["openai"] = get_client("openai", use_cache=use_cache)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
    
    return clients
