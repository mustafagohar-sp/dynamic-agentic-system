from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.config import settings
from app.database.service import DatabaseService
from app.graph.graph import build_graph
from app.llm.client import LLMClient
from app.llm.model_selector import ModelSelector
from app.llm.service import LLMService
from app.math.engine import MathEngine
from app.math.result import MathResultHandler
from app.personas.registry import get_persona
from app.rag.grounded_response import GroundedResponseService
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SourceResponse,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):

    llm_service = LLMService(
        model_selector=ModelSelector(),
        fallback_model=settings.openrouter_model,
    )

    grounded_response = GroundedResponseService(
        llm_service=llm_service,
        persona=get_persona(request.persona),
    )

    graph = build_graph(
        db=db,
        grounded_response_service=grounded_response,
        database_service=DatabaseService(),
        llm_client=LLMClient(),
        math_engine=MathEngine(),
        math_result_handler=MathResultHandler(),
    )

    result = graph.invoke(
        {
            "user_message": request.message,
            "knowledge_base_id": str(
                request.knowledge_base_id
            ),
            "persona": request.persona,
        }
    )

    response = result["result"]

    sources = []

    if hasattr(response, "sources"):
        sources = [
            SourceResponse(
                filename=source.filename,
                chunk_index=source.chunk_index,
            )
            for source in response.sources
        ]

    if hasattr(response, "answer"):
        answer = response.answer

    elif hasattr(response, "text"):
        answer = response.text

    elif hasattr(response, "allowed"):
        if not response.allowed:
            if response.suggested_persona:
                answer = (
                    "This question is outside the selected "
                    f"{request.persona} persona. Please switch "
                    f"to the {response.suggested_persona} persona "
                    "to get an appropriate answer."
                )
            else:
                answer = (
                    "This question is outside the selected "
                    f"{request.persona} persona."
                )
        else:
            answer = str(response)

    else:
        answer = str(response)

    return ChatResponse(
        route=result["routing_decision"].target.value,
        answer=answer,
        sources=sources,
    )