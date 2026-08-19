import json

from app.llm.client import LLMClient
from app.personas.config import PersonaConfig


class PersonaScopeResult:
    def __init__(
        self,
        allowed: bool,
        suggested_persona: str | None = None,
    ):
        self.allowed = allowed
        self.suggested_persona = suggested_persona


SCOPE_PROMPT = """
You are a persona routing checker.

Determine whether the user's question belongs to the selected persona.

Available personas:

financial:
- revenue
- profit
- expenses
- wages
- transfers
- commercial performance
- financial reports

legal:
- contracts
- regulations
- compliance
- data protection
- safeguarding
- legal agreements

general:
- players
- coaches
- squad
- football operations
- club operations
- stadium
- supporters
- general club information

Return ONLY JSON:

{
  "allowed": true or false,
  "suggested_persona": "financial" or "legal" or "general" or null
}

Rules:
- If the question matches the selected persona, allowed=true.
- If it belongs to another persona, allowed=false and suggest the correct persona.
"""


class PersonaScopeChecker:

    def __init__(
        self,
        llm_client: LLMClient,
    ):
        self.llm_client = llm_client


    def check(
        self,
        query: str,
        persona: PersonaConfig,
    ) -> PersonaScopeResult:

        messages = [
            {
                "role": "system",
                "content": SCOPE_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Selected persona: {persona.name}\n"
                    f"Question: {query}"
                ),
            },
        ]

        response = self.llm_client.generate(
            messages=messages,
            temperature=0.0,
        )

        data = json.loads(response)

        return PersonaScopeResult(
            allowed=data["allowed"],
            suggested_persona=data.get(
                "suggested_persona"
            ),
        )