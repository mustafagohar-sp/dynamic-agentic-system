from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaConfig:
    name: str
    system_prompt: str
    temperature: float
    preferred_model: str