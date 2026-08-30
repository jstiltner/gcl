"""Tests for the predicate module."""

import pytest

from gcl.core.predicates import (
    Predicate,
    SafeExpressionError,
    SafeExpressionParser,
    safe_eval,
)


class TestSafeExpressionParser:
    """Tests for the SafeExpressionParser class."""

    def test_simple_comparison(self):
        """Test simple comparison expressions."""
        assert safe_eval("x > 5", {"x": 10}) is True
        assert safe_eval("x > 5", {"x": 3}) is False
        assert safe_eval("x == 5", {"x": 5}) is True
        assert safe_eval("x != 5", {"x": 3}) is True

    def test_all_comparison_operators(self):
        """Test all comparison operators."""
        ctx = {"x": 5}
        assert safe_eval("x == 5", ctx) is True
        assert safe_eval("x != 4", ctx) is True
        assert safe_eval("x < 10", ctx) is True
        assert safe_eval("x <= 5", ctx) is True
        assert safe_eval("x > 3", ctx) is True
        assert safe_eval("x >= 5", ctx) is True

    def test_chained_comparison(self):
        """Test chained comparisons like 1 < x < 10."""
        assert safe_eval("1 < x < 10", {"x": 5}) is True
        assert safe_eval("1 < x < 10", {"x": 0}) is False
        assert safe_eval("1 < x < 10", {"x": 15}) is False

    def test_logical_operators(self):
        """Test logical operators (and, or, not)."""
        ctx = {"x": 5, "y": 10}
        assert safe_eval("x > 3 and y > 5", ctx) is True
        assert safe_eval("x > 10 and y > 5", ctx) is False
        assert safe_eval("x > 10 or y > 5", ctx) is True
        assert safe_eval("not x > 10", ctx) is True

    def test_arithmetic_operators(self):
        """Test arithmetic operators."""
        ctx = {"x": 10, "y": 3}
        assert safe_eval("x + y", ctx) == 13
        assert safe_eval("x - y", ctx) == 7
        assert safe_eval("x * y", ctx) == 30
        assert safe_eval("x / y", ctx) == pytest.approx(3.333, rel=0.01)
        assert safe_eval("x // y", ctx) == 3
        assert safe_eval("x % y", ctx) == 1
        assert safe_eval("y ** 2", ctx) == 9

    def test_unary_operators(self):
        """Test unary operators."""
        assert safe_eval("-x", {"x": 5}) == -5
        assert safe_eval("+x", {"x": 5}) == 5
        assert safe_eval("not x", {"x": False}) is True

    def test_membership_operators(self):
        """Test 'in' and 'not in' operators."""
        ctx = {"items": [1, 2, 3], "x": 2}
        assert safe_eval("x in items", ctx) is True
        assert safe_eval("5 in items", ctx) is False
        assert safe_eval("5 not in items", ctx) is True

    def test_attribute_access(self):
        """Test attribute access on objects."""

        class Obj:
            value = 42

        ctx = {"obj": Obj()}
        assert safe_eval("obj.value", ctx) == 42

    def test_dict_attribute_access(self):
        """Test attribute-style access on dicts."""
        ctx = {"data": {"value": 42}}
        assert safe_eval("data.value", ctx) == 42

    def test_subscript_access(self):
        """Test subscript access."""
        ctx = {"data": {"key": "value"}, "items": [1, 2, 3]}
        assert safe_eval("data['key']", ctx) == "value"
        assert safe_eval("items[0]", ctx) == 1
        assert safe_eval("items[-1]", ctx) == 3

    def test_literals(self):
        """Test literal values."""
        assert safe_eval("42", {}) == 42
        assert safe_eval("3.14", {}) == pytest.approx(3.14)
        assert safe_eval("'hello'", {}) == "hello"
        assert safe_eval("True", {}) is True
        assert safe_eval("False", {}) is False
        assert safe_eval("None", {}) is None

    def test_list_literal(self):
        """Test list literals."""
        assert safe_eval("[1, 2, 3]", {}) == [1, 2, 3]
        assert safe_eval("[x, y]", {"x": 1, "y": 2}) == [1, 2]

    def test_dict_literal(self):
        """Test dict literals."""
        assert safe_eval("{'a': 1, 'b': 2}", {}) == {"a": 1, "b": 2}

    def test_tuple_literal(self):
        """Test tuple literals."""
        assert safe_eval("(1, 2, 3)", {}) == (1, 2, 3)

    def test_unknown_variable_raises(self):
        """Test that unknown variables raise an error."""
        with pytest.raises(SafeExpressionError, match="Unknown variable"):
            safe_eval("unknown_var > 5", {})

    def test_function_call_rejected(self):
        """Test that function calls are rejected."""
        with pytest.raises(SafeExpressionError, match="not allowed"):
            safe_eval("len(items)", {"items": [1, 2, 3]})

    def test_import_rejected(self):
        """Test that import statements are rejected."""
        with pytest.raises(SafeExpressionError):
            safe_eval("__import__('os')", {})

    def test_lambda_rejected(self):
        """Test that lambda expressions are rejected."""
        with pytest.raises(SafeExpressionError, match="not allowed"):
            safe_eval("(lambda x: x + 1)(5)", {})

    def test_comprehension_rejected(self):
        """Test that comprehensions are rejected."""
        with pytest.raises(SafeExpressionError, match="not allowed"):
            safe_eval("[x for x in items]", {"items": [1, 2, 3]})

    def test_syntax_error(self):
        """Test that syntax errors are caught."""
        with pytest.raises(SafeExpressionError, match="Invalid expression syntax"):
            safe_eval("x >", {"x": 5})

    def test_max_depth_exceeded(self):
        """Test that deeply nested expressions are rejected."""
        parser = SafeExpressionParser(max_depth=5)
        # Create a deeply nested expression with actual operations
        # Each binary operation adds depth: a + (b + (c + (d + (e + f))))
        deep_expr = "a + (b + (c + (d + (e + (f + g)))))"
        ctx = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}
        with pytest.raises(SafeExpressionError, match="too deeply nested"):
            parser.evaluate(deep_expr, ctx)

    def test_complex_expression(self):
        """Test a complex real-world expression."""
        ctx = {
            "confidence": 0.85,
            "accuracy": 0.92,
            "response_time": 150,
            "max_time": 200,
        }
        expr = "confidence > 0.8 and accuracy > 0.9 and response_time < max_time"
        assert safe_eval(expr, ctx) is True


class TestPredicate:
    """Tests for the Predicate class."""

    def test_create_predicate(self):
        """Test creating a predicate."""
        pred = Predicate(
            name="high_confidence",
            expression="confidence > 0.8",
            description="Check if confidence is high",
        )
        assert pred.name == "high_confidence"
        assert pred.expression == "confidence > 0.8"
        assert pred.description == "Check if confidence is high"

    def test_evaluate_predicate(self):
        """Test evaluating a predicate."""
        pred = Predicate(name="test", expression="x > 5")
        assert pred.evaluate({"x": 10}) is True
        assert pred.evaluate({"x": 3}) is False

    def test_call_predicate(self):
        """Test calling predicate directly."""
        pred = Predicate(name="test", expression="x > 5")
        assert pred({"x": 10}) is True
        assert pred({"x": 3}) is False

    def test_invalid_expression_rejected(self):
        """Test that invalid expressions are rejected at creation."""
        with pytest.raises(ValueError, match="Invalid predicate expression"):
            Predicate(name="bad", expression="len(x)")

    def test_predicate_and(self):
        """Test combining predicates with AND."""
        pred1 = Predicate(name="p1", expression="x > 5")
        pred2 = Predicate(name="p2", expression="y < 10")
        combined = pred1 & pred2

        assert combined({"x": 10, "y": 5}) is True
        assert combined({"x": 3, "y": 5}) is False
        assert combined({"x": 10, "y": 15}) is False

    def test_predicate_or(self):
        """Test combining predicates with OR."""
        pred1 = Predicate(name="p1", expression="x > 5")
        pred2 = Predicate(name="p2", expression="y < 10")
        combined = pred1 | pred2

        assert combined({"x": 10, "y": 15}) is True
        assert combined({"x": 3, "y": 5}) is True
        assert combined({"x": 3, "y": 15}) is False

    def test_predicate_not(self):
        """Test negating a predicate."""
        pred = Predicate(name="test", expression="x > 5")
        negated = ~pred

        assert negated({"x": 3}) is True
        assert negated({"x": 10}) is False

    def test_predicate_serialization(self):
        """Test that predicates can be serialized to JSON."""
        pred = Predicate(
            name="test",
            expression="x > 5",
            description="Test predicate",
        )
        json_str = pred.model_dump_json()
        restored = Predicate.model_validate_json(json_str)

        assert restored.name == pred.name
        assert restored.expression == pred.expression
        assert restored.description == pred.description
        assert restored({"x": 10}) is True

    def test_predicate_name_validation(self):
        """Test that predicate name is validated."""
        with pytest.raises(ValueError):
            Predicate(name="", expression="x > 5")

    def test_predicate_expression_length_validation(self):
        """Test that expression length is validated."""
        with pytest.raises(ValueError):
            Predicate(name="test", expression="")
