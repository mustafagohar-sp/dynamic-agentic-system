import pytest

from app.math.engine import MathEngine, MathEvaluationError


def test_calculates_addition():
    result = MathEngine().calculate("2 + 3")

    assert result.expression == "2 + 3"
    assert result.value == 5


def test_calculates_complex_expression():
    result = MathEngine().calculate(
        "(100 - 20) * 2 / 4"
    )

    assert result.value == 40


def test_calculates_power():
    result = MathEngine().calculate("2 ** 3")

    assert result.value == 8


def test_calculates_modulo():
    result = MathEngine().calculate("17 % 5")

    assert result.value == 2


def test_calculates_negative_numbers():
    result = MathEngine().calculate("-10 + 4")

    assert result.value == -6


def test_rejects_empty_expression():
    with pytest.raises(
        MathEvaluationError,
        match="cannot be empty",
    ):
        MathEngine().calculate("")


def test_rejects_invalid_expression():
    with pytest.raises(
        MathEvaluationError,
        match="Invalid mathematical expression",
    ):
        MathEngine().calculate("2 +")


def test_rejects_division_by_zero():
    with pytest.raises(
        MathEvaluationError,
        match="Division by zero",
    ):
        MathEngine().calculate("10 / 0")


def test_rejects_function_calls():
    with pytest.raises(
        MathEvaluationError,
        match="unsupported syntax",
    ):
        MathEngine().calculate("abs(-10)")


def test_rejects_python_code():
    with pytest.raises(
        MathEvaluationError,
        match="unsupported syntax",
    ):
        MathEngine().calculate(
            "__import__('os').system('whoami')"
        )


def test_rejects_non_numeric_values():
    with pytest.raises(
        MathEvaluationError,
        match="Only numeric values",
    ):
        MathEngine().calculate("'hello'")


def test_rejects_boolean_values():
    with pytest.raises(
        MathEvaluationError,
        match="Boolean values",
    ):
        MathEngine().calculate("True")