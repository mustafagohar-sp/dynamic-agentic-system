from app.llm.service import LLMService
from app.personas.config import PersonaConfig


class FakeModelSelector:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def select(self, persona):
        self.calls.append(persona)
        return self.model


class FakeLLMClient:
    def __init__(self, model, response=None, error=None):
        self.model = model
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

        return self.response or f"Response from {self.model}"


def test_llm_service_uses_persona_configuration(monkeypatch):
    selector = FakeModelSelector(
        model="primary-model"
    )

    clients = {}

    def fake_llm_client(model=None):
        client = FakeLLMClient(model)
        clients[model] = client
        return client

    monkeypatch.setattr(
        "app.llm.service.LLMClient",
        fake_llm_client,
    )

    service = LLMService(
        model_selector=selector,
        fallback_model="fallback-model",
    )

    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Use only the provided context.",
        temperature=0.0,
        preferred_model="primary-model",
    )

    result = service.generate(
        persona=persona,
        user_message="What was the revenue?",
    )

    assert result == "Response from primary-model"

    assert selector.calls == [persona]

    call = clients["primary-model"].calls[0]

    assert call["temperature"] == 0.0
    assert call["messages"][0] == {
        "role": "system",
        "content": "Use only the provided context.",
    }
    assert call["messages"][1] == {
        "role": "user",
        "content": "What was the revenue?",
    }


def test_llm_service_creates_fallback_model(monkeypatch):
    selector = FakeModelSelector(
        model="primary-model"
    )

    clients = {}

    def fake_llm_client(model=None):
        client = FakeLLMClient(model)
        clients[model] = client
        return client

    monkeypatch.setattr(
        "app.llm.service.LLMClient",
        fake_llm_client,
    )

    service = LLMService(
        model_selector=selector,
        fallback_model="fallback-model",
    )

    persona = PersonaConfig(
        name="general_assistant",
        system_prompt="Be helpful.",
        temperature=0.2,
        preferred_model="primary-model",
    )

    service.generate(
        persona=persona,
        user_message="Explain this.",
    )

    assert "primary-model" in clients
    assert "fallback-model" in clients


def test_llm_service_allows_system_prompt_override(monkeypatch):
    selector = FakeModelSelector(
        model="primary-model"
    )

    clients = {}

    def fake_llm_client(model=None):
        client = FakeLLMClient(model)
        clients[model] = client
        return client

    monkeypatch.setattr(
        "app.llm.service.LLMClient",
        fake_llm_client,
    )

    service = LLMService(
        model_selector=selector,
        fallback_model="fallback-model",
    )

    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Generic persona prompt.",
        temperature=0.0,
        preferred_model="primary-model",
    )

    service.generate(
        persona=persona,
        user_message="What was the revenue?",
        system_prompt="Use only the supplied context.",
    )

    call = clients["primary-model"].calls[0]

    assert call["messages"][0] == {
        "role": "system",
        "content": "Use only the supplied context.",
    }


def test_llm_service_uses_fallback_when_primary_fails(monkeypatch):
    selector = FakeModelSelector(
        model="primary-model"
    )

    clients = {}

    def fake_llm_client(model=None):
        if model == "primary-model":
            client = FakeLLMClient(
                model=model,
                error=RuntimeError("Primary model failed"),
            )
        else:
            client = FakeLLMClient(
                model=model,
                response="Fallback response",
            )

        clients[model] = client
        return client

    monkeypatch.setattr(
        "app.llm.service.LLMClient",
        fake_llm_client,
    )

    service = LLMService(
        model_selector=selector,
        fallback_model="fallback-model",
    )

    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Use only the provided context.",
        temperature=0.0,
        preferred_model="primary-model",
    )

    result = service.generate(
        persona=persona,
        user_message="What was the revenue?",
    )

    assert result == "Fallback response"

    assert len(clients["primary-model"].calls) == 1
    assert len(clients["fallback-model"].calls) == 1

    assert clients["fallback-model"].calls[0]["messages"] == [
        {
            "role": "system",
            "content": "Use only the provided context.",
        },
        {
            "role": "user",
            "content": "What was the revenue?",
        },
    ]

    assert (
        clients["fallback-model"].calls[0]["temperature"]
        == 0.0
    )