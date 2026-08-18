import pytest

from app.llm.fallback import ModelFallback


class FakeLLMClient:
    def __init__(
        self,
        response=None,
        error=None,
    ):
        self.response = response
        self.error = error
        self.calls = []

    def generate(
        self,
        messages,
        temperature=0.0,
    ):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
            }
        )

        if self.error:
            raise self.error

        return self.response


def test_fallback_returns_primary_response():
    primary = FakeLLMClient(
        response="Primary response"
    )
    fallback = FakeLLMClient(
        response="Fallback response"
    )

    service = ModelFallback(
        primary_client=primary,
        fallback_client=fallback,
    )

    result = service.generate(
        messages=[
            {
                "role": "user",
                "content": "Test question",
            }
        ]
    )

    assert result == "Primary response"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0


def test_fallback_uses_fallback_when_primary_fails():
    primary = FakeLLMClient(
        error=RuntimeError("Primary model failed")
    )
    fallback = FakeLLMClient(
        response="Fallback response"
    )

    service = ModelFallback(
        primary_client=primary,
        fallback_client=fallback,
    )

    result = service.generate(
        messages=[
            {
                "role": "user",
                "content": "Test question",
            }
        ]
    )

    assert result == "Fallback response"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_fallback_propagates_fallback_failure():
    primary = FakeLLMClient(
        error=RuntimeError("Primary model failed")
    )
    fallback = FakeLLMClient(
        error=RuntimeError("Fallback model failed")
    )

    service = ModelFallback(
        primary_client=primary,
        fallback_client=fallback,
    )

    with pytest.raises(
        RuntimeError,
        match="Fallback model failed",
    ):
        service.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Test question",
                }
            ]
        )