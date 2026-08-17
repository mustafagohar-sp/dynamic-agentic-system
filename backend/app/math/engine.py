import ast
import operator
from dataclasses import dataclass


class MathEvaluationError(ValueError):
    """Raised when a mathematical expression cannot be safely evaluated."""


@dataclass(frozen=True)
class MathResult:
    expression: str
    value: int | float


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class MathEngine:
    def calculate(self, expression: str) -> MathResult:
        if not expression.strip():
            raise MathEvaluationError(
                "Mathematical expression cannot be empty"
            )

        try:
            tree = ast.parse(
                expression,
                mode="eval",
            )
        except SyntaxError as exc:
            raise MathEvaluationError(
                "Invalid mathematical expression"
            ) from exc

        value = self._evaluate(tree.body)

        if isinstance(value, float) and (
            value != value
            or value in (float("inf"), float("-inf"))
        ):
            raise MathEvaluationError(
                "Mathematical result is not finite"
            )

        return MathResult(
            expression=expression,
            value=value,
        )

    def _evaluate(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise MathEvaluationError(
                    "Boolean values are not allowed"
                )

            if not isinstance(node.value, (int, float)):
                raise MathEvaluationError(
                    "Only numeric values are allowed"
                )

            return node.value

        if isinstance(node, ast.BinOp):
            operator_function = _ALLOWED_OPERATORS.get(
                type(node.op)
            )

            if operator_function is None:
                raise MathEvaluationError(
                    "Operator is not allowed"
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            try:
                return operator_function(left, right)
            except ZeroDivisionError as exc:
                raise MathEvaluationError(
                    "Division by zero is not allowed"
                ) from exc
            except (OverflowError, ValueError) as exc:
                raise MathEvaluationError(
                    "Mathematical operation failed"
                ) from exc

        if isinstance(node, ast.UnaryOp):
            operator_function = _ALLOWED_OPERATORS.get(
                type(node.op)
            )

            if operator_function is None:
                raise MathEvaluationError(
                    "Operator is not allowed"
                )

            return operator_function(
                self._evaluate(node.operand)
            )

        raise MathEvaluationError(
            "Expression contains unsupported syntax"
        )