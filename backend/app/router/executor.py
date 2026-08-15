from uuid import UUID

from sqlalchemy.orm import Session

from app.database.service import DatabaseService
from app.rag.context import assemble_context
from app.rag.grounded_response import GroundedResponseService
from app.rag.retrieval import retrieve
from app.router.router import RouteTarget, RoutingDecision


class QueryExecutor:
    def __init__(
        self,
        grounded_response_service: GroundedResponseService,
        database_service: DatabaseService,
    ):
        self.grounded_response_service = grounded_response_service
        self.database_service = database_service

    def execute(
        self,
        db: Session,
        knowledge_base_id: UUID,
        query: str,
        decision: RoutingDecision,
    ):
        if decision.target == RouteTarget.RAG:
            return self._execute_rag(
                db=db,
                knowledge_base_id=knowledge_base_id,
                query=query,
            )

        if decision.target == RouteTarget.DATABASE:
            return self._execute_database(
                db=db,
                knowledge_base_id=knowledge_base_id,
                query=query,
                decision=decision,
            )

        raise ValueError(
            f"Unsupported route target: {decision.target}"
        )

    def _execute_rag(
        self,
        db: Session,
        knowledge_base_id: UUID,
        query: str,
    ):
        results = retrieve(
            db=db,
            knowledge_base_id=knowledge_base_id,
            query=query,
            top_k=5,
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
    