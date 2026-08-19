from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaConfig:
    name: str
    system_prompt: str
    temperature: float
    preferred_model: str
    description: str
    document_keywords: tuple[str, ...]