from dataclasses import dataclass

from app.math.engine import MathResult


@dataclass(frozen=True)
class MathResponse:
    expression: str
    result: int | float
    text: str


class MathResultHandler:
    def format(self, math_result: MathResult) -> MathResponse:
        return MathResponse(
            expression=math_result.expression,
            result=math_result.value,
            text=f"The result is {math_result.value}.",
        )