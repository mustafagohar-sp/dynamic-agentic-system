from types import SimpleNamespace

from app.llm.client import LLMClient
from app.config import settings


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Test LLM response"
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


def test_llm_client_returns_response(monkeypatch):
    client = LLMClient()

    client.client = FakeClient()

    result = client.generate(
        messages=[
            {
                "role": "user",
                "content": "Test question",
            }
        ]
    )

    assert result == "Test LLM response"


def test_llm_client_rejects_empty_response():
    client = LLMClient()

    class EmptyCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=""
                        )
                    )
                ]
            )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=EmptyCompletions()
        )
    )

    try:
        client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Test question",
                }
            ]
        )
    except ValueError as exc:
        assert str(exc) == "LLM returned an empty response"
    else:
        raise AssertionError("Expected ValueError")

def test_llm_client_uses_default_model():
    client = LLMClient()

    assert client.model == settings.openrouter_model


def test_llm_client_accepts_custom_model():
    client = LLMClient(
        model="test/custom-model"
    )

    assert client.model == "test/custom-model"