from uuid import UUID

from sqlalchemy.orm import Session

from app.database.service import DatabaseService
from app.llm.client import LLMClient
from app.math.engine import MathEngine
from app.rag.context import assemble_context
from app.rag.grounded_response import GroundedResponseService
from app.rag.retrieval import retrieve
from app.router.router import RouteTarget, RoutingDecision
from app.math.result import MathResultHandler
from app.personas.scope_checker import PersonaScopeChecker
from app.personas.registry import get_persona


MATH_EXTRACTION_SYSTEM_PROMPT = """You extract mathematical expressions
from user questions.

Return ONLY the mathematical expression that should be calculated.

Examples:
"What is 17 multiplied by 24?" -> 17 * 24
"Calculate 100 divided by 4" -> 100 / 4
"What is 15% of 200?" -> 200 * 0.15

Do not include explanations or markdown.
"""


class QueryExecutor:
    def __init__(
        self,
        grounded_response_service: GroundedResponseService,
        database_service: DatabaseService,
        llm_client: LLMClient,
        math_engine: MathEngine,
        math_result_handler: MathResultHandler,
    ):
        self.grounded_response_service = grounded_response_service
        self.database_service = database_service
        self.llm_client = llm_client
        self.math_engine = math_engine
        self.math_result_handler = math_result_handler
        self.scope_checker = PersonaScopeChecker(llm_client=llm_client)

    def execute(
        self,
        db: Session,
        knowledge_base_id: UUID,
        query: str,
        decision: RoutingDecision,
        persona: str,
    ):
        if decision.target == RouteTarget.RAG:
            return self._execute_rag(
                db=db,
                knowledge_base_id=knowledge_base_id,
                query=query,
                persona=persona,
            )

        if decision.target == RouteTarget.DATABASE:
            return self._execute_database(
                db=db,
                knowledge_base_id=knowledge_base_id,
                query=query,
                decision=decision,
            )

        if decision.target == RouteTarget.MATH:
            return self._execute_math(query)

        raise ValueError(
            f"Unsupported route target: {decision.target}"
        )

    def _execute_rag(
        self,
        db: Session,
        knowledge_base_id: UUID,
        query: str,
        persona: str,
    ):
        persona_config = get_persona(persona)

        scope_result = self.scope_checker.check(
            query=query,
            persona=persona_config,
        )

        if not scope_result.allowed:
            return scope_result

        results = retrieve(
            db=db,
            knowledge_base_id=knowledge_base_id,
            query=query,
            top_k=10,
            persona=persona,
        )

        if not results:
            return self.scope_checker.out_of_scope_response(
                persona=persona,
            )

        context = assemble_context(results)

        return self.grounded_response_service.answer(
            question=query,
            context=context,
        )

    def _execute_database(
        self,
        db: Session,
        knowledge_base_id: UUID,
        query: str,
        decision: RoutingDecision,
    ):
        if decision.classification.intent.value == "system_metadata":
            query_lower = query.lower()

            if "document" in query_lower:
                return self.database_service.get_document_count(
                    db=db,
                    knowledge_base_id=knowledge_base_id,
                )

            if "chunk" in query_lower:
                return self.database_service.get_chunk_count(
                    db=db,
                    knowledge_base_id=knowledge_base_id,
                )

            if "active" in query_lower and "version" in query_lower:
                return self.database_service.get_active_version(
                    db=db,
                    knowledge_base_id=knowledge_base_id,
                )

            if "status" in query_lower and "version" in query_lower:
                version = self.database_service.get_active_version(
                    db=db,
                    knowledge_base_id=knowledge_base_id,
                )

                if version is None:
                    return None

                return self.database_service.get_version_status(
                    db=db,
                    knowledge_base_id=knowledge_base_id,
                    version_id=version.id,
                )

        raise ValueError(
            "Unsupported database query"
        )

    def _execute_math(
        self,
        query: str,
    ):
        messages = [
            {
                "role": "system",
                "content": MATH_EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        expression = self.llm_client.generate(
            messages=messages,
            temperature=0.0,
        ).strip()

        math_result = self.math_engine.calculate(expression)

        return self.math_result_handler.format(
            math_result
        )