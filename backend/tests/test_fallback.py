import pytest

from app.llm.fallback import ModelFallback


class FakeLLMClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, messages, temperature=0.0):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
            }
        )

        if self.error:
            raise self.error

        return self.response


def test_fallback_uses_primary_client():
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
        ],
        temperature=0.2,
    )

    assert result == "Primary response"

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0

    assert primary.calls[0]["temperature"] == 0.2


def test_fallback_uses_fallback_client_when_primary_fails():
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
        ],
        temperature=0.0,
    )

    assert result == "Fallback response"

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1

    assert fallback.calls[0]["temperature"] == 0.0


def test_fallback_passes_same_messages_to_fallback():
    primary = FakeLLMClient(
        error=RuntimeError("Primary model failed")
    )
    fallback = FakeLLMClient(
        response="Fallback response"
    )

    messages = [
        {
            "role": "system",
            "content": "System prompt",
        },
        {
            "role": "user",
            "content": "Test question",
        },
    ]

    service = ModelFallback(
        primary_client=primary,
        fallback_client=fallback,
    )

    service.generate(
        messages=messages,
        temperature=0.1,
    )

    assert fallback.calls[0]["messages"] == messages