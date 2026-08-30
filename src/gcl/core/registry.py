"""
Verification function registry.

This module provides a registry for verification functions that can be
referenced by name in commitments, enabling serialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from gcl.core.commitment import VerificationResult

# Type for verification functions
# Signature: (pre_state, post_state, action) -> VerificationResult
VerificationFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], "VerificationResult"]

F = TypeVar("F", bound=VerificationFn)


class VerificationRegistry:
    """
    Registry for verification functions.
    
    This singleton class maintains a mapping from string names to verification
    functions, enabling commitments to reference verification logic by name
    for serialization purposes.
    
    Usage:
        >>> from gcl.core.registry import verification_registry
        >>> 
        >>> @verification_registry.register("accuracy_check")
        ... def check_accuracy(pre_state, post_state, action):
        ...     # Verification logic here
        ...     return VerificationResult(...)
        >>> 
        >>> # Later, retrieve the function
        >>> fn = verification_registry.get("accuracy_check")
        >>> result = fn(pre_state, post_state, action)
    """

    _instance: VerificationRegistry | None = None
    _functions: dict[str, VerificationFn]
    _metadata: dict[str, dict[str, Any]]

    def __new__(cls) -> VerificationRegistry:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._functions = {}
            cls._instance._metadata = {}
        return cls._instance

    def register(
        self,
        name: str,
        *,
        description: str | None = None,
        version: str = "1.0.0",
        tags: list[str] | None = None,
    ) -> Callable[[F], F]:
        """
        Decorator to register a verification function.
        
        Args:
            name: Unique name for the function.
            description: Optional description of what the function verifies.
            version: Version string for the function.
            tags: Optional list of tags for categorization.
            
        Returns:
            Decorator function that registers the wrapped function.
            
        Raises:
            ValueError: If a function with this name is already registered.
            
        Example:
            >>> @verification_registry.register(
            ...     "accuracy_check",
            ...     description="Checks if prediction accuracy meets threshold",
            ...     tags=["accuracy", "classification"]
            ... )
            ... def check_accuracy(pre_state, post_state, action):
            ...     ...
        """
        if name in self._functions:
            raise ValueError(
                f"Verification function '{name}' is already registered. "
                "Use force_register() to override."
            )

        def decorator(fn: F) -> F:
            self._functions[name] = fn
            self._metadata[name] = {
                "description": description or fn.__doc__,
                "version": version,
                "tags": tags or [],
                "module": fn.__module__,
                "qualname": fn.__qualname__,
            }
            return fn

        return decorator

    def force_register(
        self,
        name: str,
        fn: VerificationFn,
        *,
        description: str | None = None,
        version: str = "1.0.0",
        tags: list[str] | None = None,
    ) -> None:
        """
        Register a function, overwriting any existing registration.
        
        Args:
            name: Unique name for the function.
            fn: The verification function to register.
            description: Optional description of what the function verifies.
            version: Version string for the function.
            tags: Optional list of tags for categorization.
        """
        self._functions[name] = fn
        self._metadata[name] = {
            "description": description or fn.__doc__,
            "version": version,
            "tags": tags or [],
            "module": fn.__module__,
            "qualname": fn.__qualname__,
        }

    def get(self, name: str) -> VerificationFn | None:
        """
        Get a verification function by name.
        
        Args:
            name: The registered name of the function.
            
        Returns:
            The verification function, or None if not found.
        """
        return self._functions.get(name)

    def get_or_raise(self, name: str) -> VerificationFn:
        """
        Get a verification function by name, raising if not found.
        
        Args:
            name: The registered name of the function.
            
        Returns:
            The verification function.
            
        Raises:
            KeyError: If no function is registered with this name.
        """
        fn = self._functions.get(name)
        if fn is None:
            available = ", ".join(sorted(self._functions.keys())) or "(none)"
            raise KeyError(
                f"No verification function registered with name '{name}'. "
                f"Available functions: {available}"
            )
        return fn

    def get_metadata(self, name: str) -> dict[str, Any] | None:
        """
        Get metadata for a registered function.
        
        Args:
            name: The registered name of the function.
            
        Returns:
            Dictionary of metadata, or None if not found.
        """
        return self._metadata.get(name)

    def list_functions(self) -> list[str]:
        """
        List all registered function names.
        
        Returns:
            Sorted list of registered function names.
        """
        return sorted(self._functions.keys())

    def list_by_tag(self, tag: str) -> list[str]:
        """
        List functions that have a specific tag.
        
        Args:
            tag: The tag to filter by.
            
        Returns:
            List of function names with the given tag.
        """
        return [
            name
            for name, meta in self._metadata.items()
            if tag in meta.get("tags", [])
        ]

    def unregister(self, name: str) -> bool:
        """
        Remove a function from the registry.
        
        Args:
            name: The name of the function to remove.
            
        Returns:
            True if the function was removed, False if it wasn't registered.
        """
        if name in self._functions:
            del self._functions[name]
            del self._metadata[name]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered functions. Use with caution."""
        self._functions.clear()
        self._metadata.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a function is registered: 'name' in registry."""
        return name in self._functions

    def __len__(self) -> int:
        """Return the number of registered functions."""
        return len(self._functions)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        return f"VerificationRegistry({len(self._functions)} functions)"


# Global registry instance
verification_registry = VerificationRegistry()


# Built-in verification functions


@verification_registry.register(
    "always_success",
    description="Always returns success. Useful for testing.",
    tags=["testing", "builtin"],
)
def always_success(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    action: dict[str, Any],
) -> "VerificationResult":
    """Verification function that always returns success."""
    # Import here to avoid circular imports
    from gcl.core.commitment import VerificationResult, VerificationStatus

    return VerificationResult(
        status=VerificationStatus.SUCCESS,
        commitment_id="",  # Will be set by caller
        details={"message": "Always succeeds"},
    )


@verification_registry.register(
    "always_failure",
    description="Always returns failure. Useful for testing.",
    tags=["testing", "builtin"],
)
def always_failure(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    action: dict[str, Any],
) -> "VerificationResult":
    """Verification function that always returns failure."""
    from gcl.core.commitment import VerificationResult, VerificationStatus

    return VerificationResult(
        status=VerificationStatus.FAILURE,
        commitment_id="",
        details={"message": "Always fails"},
    )


@verification_registry.register(
    "state_change_check",
    description="Verifies that a specific state field changed as expected.",
    tags=["state", "builtin"],
)
def state_change_check(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    action: dict[str, Any],
) -> "VerificationResult":
    """
    Verify that state changed as specified in action parameters.
    
    Expected action parameters:
        - field: The state field to check
        - expected_value: The expected value after action (optional)
        - expected_change: The expected delta (optional)
    """
    from gcl.core.commitment import VerificationResult, VerificationStatus

    field = action.get("field")
    if not field:
        return VerificationResult(
            status=VerificationStatus.ERROR,
            commitment_id="",
            details={"error": "No 'field' specified in action"},
        )

    pre_value = pre_state.get(field)
    post_value = post_state.get(field)

    # Check expected value
    if "expected_value" in action:
        if post_value == action["expected_value"]:
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                commitment_id="",
                details={
                    "field": field,
                    "pre_value": pre_value,
                    "post_value": post_value,
                },
            )
        return VerificationResult(
            status=VerificationStatus.FAILURE,
            commitment_id="",
            details={
                "field": field,
                "expected": action["expected_value"],
                "actual": post_value,
            },
        )

    # Check expected change
    if "expected_change" in action:
        try:
            actual_change = post_value - pre_value
            if actual_change == action["expected_change"]:
                return VerificationResult(
                    status=VerificationStatus.SUCCESS,
                    commitment_id="",
                    details={
                        "field": field,
                        "change": actual_change,
                    },
                )
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                commitment_id="",
                details={
                    "field": field,
                    "expected_change": action["expected_change"],
                    "actual_change": actual_change,
                },
            )
        except TypeError:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                commitment_id="",
                details={"error": f"Cannot compute change for field '{field}'"},
            )

    # Just check that something changed
    if pre_value != post_value:
        return VerificationResult(
            status=VerificationStatus.SUCCESS,
            commitment_id="",
            details={
                "field": field,
                "pre_value": pre_value,
                "post_value": post_value,
            },
        )

    return VerificationResult(
        status=VerificationStatus.FAILURE,
        commitment_id="",
        details={
            "field": field,
            "message": "No change detected",
        },
    )
