from app.math.engine import MathResult
from app.math.result import MathResponse, MathResultHandler


def test_math_result_handler_formats_integer_result():
    math_result = MathResult(
        expression="17 * 24",
        value=408,
    )

    response = MathResultHandler().format(math_result)

    assert response == MathResponse(
        expression="17 * 24",
        result=408,
        text="The result is 408.",
    )


def test_math_result_handler_formats_decimal_result():
    math_result = MathResult(
        expression="10 / 4",
        value=2.5,
    )

    response = MathResultHandler().format(math_result)

    assert response.expression == "10 / 4"
    assert response.result == 2.5
    assert response.text == "The result is 2.5."