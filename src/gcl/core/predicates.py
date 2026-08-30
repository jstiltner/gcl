"""
Safe expression parser for predicates.

This module provides a secure way to evaluate predicate expressions
without using Python's eval() function.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SafeExpressionError(Exception):
    """Raised when an expression cannot be safely evaluated."""

    pass


class SafeExpressionParser:
    """
    A safe expression parser that evaluates simple expressions without eval().
    
    Supported operations:
    - Comparison: ==, !=, <, >, <=, >=
    - Logical: and, or, not
    - Arithmetic: +, -, *, /, //, %, **
    - Membership: in, not in
    - Identity: is, is not
    - Attribute access: context.field
    - Subscript access: context['field']
    - Literals: numbers, strings, booleans, None, lists, dicts
    
    NOT supported (for security):
    - Function calls
    - Lambda expressions
    - Comprehensions
    - Import statements
    - Exec/eval
    - Assignment
    """

    # Allowed binary operators
    BINARY_OPS: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    # Allowed unary operators
    UNARY_OPS: dict[type, Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }

    def __init__(self, max_depth: int = 10) -> None:
        """
        Initialize the parser.
        
        Args:
            max_depth: Maximum AST depth to prevent stack overflow attacks.
        """
        self.max_depth = max_depth

    def parse(self, expression: str) -> ast.Expression:
        """
        Parse an expression string into an AST.
        
        Args:
            expression: The expression string to parse.
            
        Returns:
            The parsed AST expression.
            
        Raises:
            SafeExpressionError: If the expression is invalid or unsafe.
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise SafeExpressionError(f"Invalid expression syntax: {e}") from e

        self._validate_ast(tree, depth=0)
        return tree

    def _validate_ast(self, node: ast.AST, depth: int) -> None:
        """
        Validate that an AST node is safe to evaluate.
        
        Args:
            node: The AST node to validate.
            depth: Current recursion depth.
            
        Raises:
            SafeExpressionError: If the node is unsafe.
        """
        if depth > self.max_depth:
            raise SafeExpressionError(
                f"Expression too deeply nested (max depth: {self.max_depth})"
            )

        # Allowed node types
        allowed_types = (
            ast.Expression,
            ast.BoolOp,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.Constant,
            ast.Name,
            ast.Attribute,
            ast.Subscript,
            ast.Index,  # Python 3.8 compatibility
            ast.Slice,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.And,
            ast.Or,
            ast.Load,
        )

        # Also allow operator types
        operator_types = tuple(self.BINARY_OPS.keys()) + tuple(self.UNARY_OPS.keys())

        if not isinstance(node, allowed_types + operator_types):
            raise SafeExpressionError(
                f"Unsafe expression: {type(node).__name__} not allowed"
            )

        # Recursively validate children
        for child in ast.iter_child_nodes(node):
            self._validate_ast(child, depth + 1)

    def evaluate(self, expression: str, context: dict[str, Any]) -> Any:
        """
        Safely evaluate an expression with the given context.
        
        Args:
            expression: The expression string to evaluate.
            context: Dictionary of variables available in the expression.
            
        Returns:
            The result of evaluating the expression.
            
        Raises:
            SafeExpressionError: If the expression is invalid or unsafe.
        """
        tree = self.parse(expression)
        return self._eval_node(tree.body, context)

    def _eval_node(self, node: ast.AST, context: dict[str, Any]) -> Any:
        """
        Evaluate a single AST node.
        
        Args:
            node: The AST node to evaluate.
            context: Dictionary of variables available in the expression.
            
        Returns:
            The result of evaluating the node.
        """
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id not in context:
                raise SafeExpressionError(f"Unknown variable: {node.id}")
            return context[node.id]

        if isinstance(node, ast.Attribute):
            value = self._eval_node(node.value, context)
            if hasattr(value, node.attr):
                return getattr(value, node.attr)
            # Also try dict-style access for flexibility
            if isinstance(value, dict) and node.attr in value:
                return value[node.attr]
            raise SafeExpressionError(
                f"Attribute '{node.attr}' not found on {type(value).__name__}"
            )

        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value, context)
            # Handle different Python versions
            if isinstance(node.slice, ast.Index):
                # Python 3.8
                index = self._eval_node(node.slice.value, context)  # type: ignore
            else:
                # Python 3.9+
                index = self._eval_node(node.slice, context)
            return value[index]

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            op_func = self.BINARY_OPS.get(type(node.op))
            if op_func is None:
                raise SafeExpressionError(f"Unknown operator: {type(node.op).__name__}")
            return op_func(left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            op_func = self.UNARY_OPS.get(type(node.op))
            if op_func is None:
                raise SafeExpressionError(f"Unknown operator: {type(node.op).__name__}")
            return op_func(operand)

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                op_func = self.BINARY_OPS.get(type(op))
                if op_func is None:
                    raise SafeExpressionError(f"Unknown operator: {type(op).__name__}")
                if not op_func(left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._eval_node(v, context) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(self._eval_node(v, context) for v in node.values)
            raise SafeExpressionError(f"Unknown boolean operator: {type(node.op).__name__}")

        if isinstance(node, ast.List):
            return [self._eval_node(elt, context) for elt in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, context) for elt in node.elts)

        if isinstance(node, ast.Dict):
            return {
                self._eval_node(k, context): self._eval_node(v, context)
                for k, v in zip(node.keys, node.values)
                if k is not None  # Handle **kwargs expansion (not allowed but be safe)
            }

        raise SafeExpressionError(f"Cannot evaluate node type: {type(node).__name__}")


# Global parser instance
_parser = SafeExpressionParser()


def safe_eval(expression: str, context: dict[str, Any]) -> Any:
    """
    Safely evaluate an expression with the given context.
    
    This is a convenience function that uses the global parser instance.
    
    Args:
        expression: The expression string to evaluate.
        context: Dictionary of variables available in the expression.
        
    Returns:
        The result of evaluating the expression.
        
    Raises:
        SafeExpressionError: If the expression is invalid or unsafe.
        
    Examples:
        >>> safe_eval("x > 5", {"x": 10})
        True
        >>> safe_eval("name == 'test'", {"name": "test"})
        True
        >>> safe_eval("data['key'] + 1", {"data": {"key": 5}})
        6
    """
    return _parser.evaluate(expression, context)


class Predicate(BaseModel):
    """
    A condition that can be evaluated against a context.
    
    Predicates use a safe expression language that supports:
    - Comparison operators: ==, !=, <, >, <=, >=
    - Logical operators: and, or, not
    - Arithmetic: +, -, *, /, //, %, **
    - Variable access: variable_name
    - Attribute access: object.attribute
    - Subscript access: dict['key'] or list[0]
    
    Examples:
        >>> pred = Predicate(name="high_confidence", expression="confidence > 0.8")
        >>> pred.evaluate({"confidence": 0.9})
        True
        >>> pred.evaluate({"confidence": 0.5})
        False
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name for this predicate",
    )
    expression: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Safe expression string to evaluate",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Optional description of what this predicate checks",
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate that the expression can be parsed safely."""
        try:
            _parser.parse(v)
        except SafeExpressionError as e:
            raise ValueError(f"Invalid predicate expression: {e}") from e
        return v

    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate this predicate against the given context.
        
        Args:
            context: Dictionary of variables available for evaluation.
            
        Returns:
            True if the predicate condition is met, False otherwise.
            
        Raises:
            SafeExpressionError: If evaluation fails (e.g., missing variable).
        """
        result = safe_eval(self.expression, context)
        return bool(result)

    def __call__(self, context: dict[str, Any]) -> bool:
        """Allow calling predicate directly: pred(context)."""
        return self.evaluate(context)

    def __and__(self, other: "Predicate") -> "Predicate":
        """Combine predicates with AND: pred1 & pred2."""
        return Predicate(
            name=f"({self.name} AND {other.name})",
            expression=f"({self.expression}) and ({other.expression})",
            description=f"Combined: {self.description or self.name} AND {other.description or other.name}",
        )

    def __or__(self, other: "Predicate") -> "Predicate":
        """Combine predicates with OR: pred1 | pred2."""
        return Predicate(
            name=f"({self.name} OR {other.name})",
            expression=f"({self.expression}) or ({other.expression})",
            description=f"Combined: {self.description or self.name} OR {other.description or other.name}",
        )

    def __invert__(self) -> "Predicate":
        """Negate predicate: ~pred."""
        return Predicate(
            name=f"NOT({self.name})",
            expression=f"not ({self.expression})",
            description=f"Negated: {self.description or self.name}",
        )
