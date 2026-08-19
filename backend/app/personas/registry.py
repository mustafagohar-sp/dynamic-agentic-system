from app.config import settings
from app.personas.config import PersonaConfig


PERSONAS = {
    "financial": PersonaConfig(
        name="financial",
        system_prompt=(
            "You are a financial analyst persona. "
            "Answer only questions related to financial information. "
            "Use only the provided context. "
            "Do not answer questions outside the financial domain."
        ),
        temperature=0.0,
        preferred_model=settings.openrouter_model,
        description=(
            "Financial information including revenue, expenses, "
            "wages, transfers, commercial performance, contracts, "
            "and financial agreements."
        ),
        document_keywords=(
            "Northbridge_FC_Annual_Financial_Report_2024_25.pdf",
            "Northbridge_FC_Commercial_Performance_Report_2024_25.pdf",
            "Northbridge_FC_Transfer_and_Wage_Report_2024_25.pdf",
            "Northbridge_FC_Contracts_Commercial_Agreements_Register_2024_25.pdf",
            "northbridge_v2_update.txt",
        ),
    ),

    "legal": PersonaConfig(
        name="legal",
        system_prompt=(
            "You are a legal and regulatory analyst persona. "
            "Answer only questions related to legal, regulatory, "
            "compliance, contracts, safeguarding, data protection, "
            "and player legal matters. "
            "Use only the provided context. "
            "Do not answer questions outside the legal domain."
        ),
        temperature=0.0,
        preferred_model=settings.openrouter_model,
        description=(
            "Legal, regulatory, compliance, contracts, safeguarding, "
            "data protection, and player legal information."
        ),
        document_keywords=(
            "Northbridge_FC_Contracts_Commercial_Agreements_Register_2024_25.pdf",
            "Northbridge_FC_Football_Operations_Player_Legal_Report_2024_25.pdf",
            "Northbridge_FC_Governance_Safeguarding_Data_Protection_Report_2024_25.pdf",
            "Northbridge_FC_Legal_Regulatory_Compliance_Report_2024_25.pdf",
        ),
    ),

    "general": PersonaConfig(
        name="general",
        system_prompt=(
            "You are a general Northbridge FC information assistant. "
            "Answer questions using the provided general football, "
            "club operations, stadium, supporter, community, and "
            "organizational context. "
            "Do not answer specialized financial or legal questions "
            "unless they are appropriate for this persona."
        ),
        temperature=0.0,
        preferred_model=settings.openrouter_model,
        description=(
            "General club information including football operations, "
            "players, coaching, club operations, stadium operations, "
            "supporters, community, and infrastructure."
        ),
        document_keywords=(
            "Northbridge_FC_Stadium_Infrastructure_Investment_Report_2024_25.pdf",
            "Northbridge_FC_Club_Operations_Organizational_Report_2024_25.pdf",
            "Northbridge_FC_Commercial_Supporter_Community_Operations_Report_2024_25.pdf",
            "Northbridge_FC_Football_Operations_Performance_Report_2024_25.pdf",
            "Northbridge_FC_Matchday_Stadium_Operations_Report_2024_25.pdf",
        ),
    ),
}


def get_persona(name: str) -> PersonaConfig:
    try:
        return PERSONAS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown persona: {name}"
        ) from exc