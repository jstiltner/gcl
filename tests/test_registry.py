"""Tests for the verification registry module."""

import pytest

from gcl.core.commitment import VerificationResult, VerificationStatus
from gcl.core.registry import VerificationRegistry, verification_registry


class TestVerificationRegistry:
    """Tests for the VerificationRegistry class."""

    def setup_method(self):
        """Create a fresh registry for each test."""
        # Create a new registry instance for testing (bypass singleton)
        self.registry = VerificationRegistry.__new__(VerificationRegistry)
        self.registry._functions = {}
        self.registry._metadata = {}

    def test_register_function(self):
        """Test registering a verification function."""

        @self.registry.register("test_fn", description="A test function")
        def test_fn(pre_state, post_state, action):
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                commitment_id="test",
            )

        assert "test_fn" in self.registry
        assert self.registry.get("test_fn") is test_fn

    def test_register_with_metadata(self):
        """Test that metadata is stored correctly."""

        @self.registry.register(
            "test_fn",
            description="Test description",
            version="2.0.0",
            tags=["test", "example"],
        )
        def test_fn(pre_state, post_state, action):
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                commitment_id="test",
            )

        metadata = self.registry.get_metadata("test_fn")
        assert metadata is not None
        assert metadata["description"] == "Test description"
        assert metadata["version"] == "2.0.0"
        assert metadata["tags"] == ["test", "example"]

    def test_register_duplicate_raises(self):
        """Test that registering a duplicate name raises an error."""

        @self.registry.register("duplicate")
        def fn1(pre_state, post_state, action):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @self.registry.register("duplicate")
            def fn2(pre_state, post_state, action):
                pass

    def test_force_register_overwrites(self):
        """Test that force_register overwrites existing functions."""

        def fn1(pre_state, post_state, action):
            return "fn1"

        def fn2(pre_state, post_state, action):
            return "fn2"

        self.registry.force_register("test", fn1)
        assert self.registry.get("test")({}, {}, {}) == "fn1"

        self.registry.force_register("test", fn2)
        assert self.registry.get("test")({}, {}, {}) == "fn2"

    def test_get_nonexistent_returns_none(self):
        """Test that getting a nonexistent function returns None."""
        assert self.registry.get("nonexistent") is None

    def test_get_or_raise_raises_keyerror(self):
        """Test that get_or_raise raises KeyError for nonexistent functions."""
        with pytest.raises(KeyError, match="No verification function"):
            self.registry.get_or_raise("nonexistent")

    def test_list_functions(self):
        """Test listing all registered functions."""

        @self.registry.register("fn_a")
        def fn_a(pre_state, post_state, action):
            pass

        @self.registry.register("fn_b")
        def fn_b(pre_state, post_state, action):
            pass

        functions = self.registry.list_functions()
        assert functions == ["fn_a", "fn_b"]  # Sorted

    def test_list_by_tag(self):
        """Test listing functions by tag."""

        @self.registry.register("fn_a", tags=["accuracy"])
        def fn_a(pre_state, post_state, action):
            pass

        @self.registry.register("fn_b", tags=["accuracy", "classification"])
        def fn_b(pre_state, post_state, action):
            pass

        @self.registry.register("fn_c", tags=["other"])
        def fn_c(pre_state, post_state, action):
            pass

        accuracy_fns = self.registry.list_by_tag("accuracy")
        assert set(accuracy_fns) == {"fn_a", "fn_b"}

    def test_unregister(self):
        """Test unregistering a function."""

        @self.registry.register("to_remove")
        def fn(pre_state, post_state, action):
            pass

        assert "to_remove" in self.registry
        result = self.registry.unregister("to_remove")
        assert result is True
        assert "to_remove" not in self.registry

    def test_unregister_nonexistent(self):
        """Test unregistering a nonexistent function returns False."""
        result = self.registry.unregister("nonexistent")
        assert result is False

    def test_clear(self):
        """Test clearing all functions."""

        @self.registry.register("fn1")
        def fn1(pre_state, post_state, action):
            pass

        @self.registry.register("fn2")
        def fn2(pre_state, post_state, action):
            pass

        assert len(self.registry) == 2
        self.registry.clear()
        assert len(self.registry) == 0

    def test_len(self):
        """Test __len__ method."""
        assert len(self.registry) == 0

        @self.registry.register("fn1")
        def fn1(pre_state, post_state, action):
            pass

        assert len(self.registry) == 1

    def test_contains(self):
        """Test __contains__ method."""
        assert "test" not in self.registry

        @self.registry.register("test")
        def fn(pre_state, post_state, action):
            pass

        assert "test" in self.registry


class TestBuiltinVerificationFunctions:
    """Tests for the built-in verification functions."""

    @pytest.fixture(autouse=True)
    def ensure_builtins_registered(self):
        """Ensure built-in functions are registered by importing the module."""
        # Force re-import to ensure built-in functions are registered
        import importlib
        import gcl.core.registry as reg_module
        importlib.reload(reg_module)
        # Get the fresh registry
        from gcl.core.registry import verification_registry as fresh_registry
        self.registry = fresh_registry

    def test_always_success(self):
        """Test the always_success function."""
        fn = self.registry.get("always_success")
        assert fn is not None, f"Available functions: {self.registry.list_functions()}"

        result = fn({}, {}, {})
        assert result.status == VerificationStatus.SUCCESS

    def test_always_failure(self):
        """Test the always_failure function."""
        fn = self.registry.get("always_failure")
        assert fn is not None, f"Available functions: {self.registry.list_functions()}"

        result = fn({}, {}, {})
        assert result.status == VerificationStatus.FAILURE

    def test_state_change_check_expected_value(self):
        """Test state_change_check with expected_value."""
        fn = self.registry.get("state_change_check")
        assert fn is not None, f"Available functions: {self.registry.list_functions()}"

        pre_state = {"counter": 0}
        post_state = {"counter": 5}
        action = {"field": "counter", "expected_value": 5}

        result = fn(pre_state, post_state, action)
        assert result.status == VerificationStatus.SUCCESS

        # Test failure case
        action = {"field": "counter", "expected_value": 10}
        result = fn(pre_state, post_state, action)
        assert result.status == VerificationStatus.FAILURE

    def test_state_change_check_expected_change(self):
        """Test state_change_check with expected_change."""
        fn = self.registry.get("state_change_check")
        assert fn is not None

        pre_state = {"counter": 5}
        post_state = {"counter": 8}
        action = {"field": "counter", "expected_change": 3}

        result = fn(pre_state, post_state, action)
        assert result.status == VerificationStatus.SUCCESS

        # Test failure case
        action = {"field": "counter", "expected_change": 5}
        result = fn(pre_state, post_state, action)
        assert result.status == VerificationStatus.FAILURE

    def test_state_change_check_any_change(self):
        """Test state_change_check for any change."""
        fn = self.registry.get("state_change_check")
        assert fn is not None

        pre_state = {"value": "old"}
        post_state = {"value": "new"}
        action = {"field": "value"}

        result = fn(pre_state, post_state, action)
        assert result.status == VerificationStatus.SUCCESS

        # Test no change
        post_state = {"value": "old"}
        result = fn(pre_state, post_state, action)
        assert result.status == VerificationStatus.FAILURE

    def test_state_change_check_missing_field(self):
        """Test state_change_check with missing field parameter."""
        fn = self.registry.get("state_change_check")
        assert fn is not None

        result = fn({}, {}, {})
        assert result.status == VerificationStatus.ERROR
        assert "field" in result.details.get("error", "")

    def test_builtin_functions_have_tags(self):
        """Test that built-in functions have appropriate tags."""
        testing_fns = self.registry.list_by_tag("testing")
        assert "always_success" in testing_fns, f"Available: {self.registry.list_functions()}"
        assert "always_failure" in testing_fns

        builtin_fns = self.registry.list_by_tag("builtin")
        assert "always_success" in builtin_fns
        assert "always_failure" in builtin_fns
        assert "state_change_check" in builtin_fns
