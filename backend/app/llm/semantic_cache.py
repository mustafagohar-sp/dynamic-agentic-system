from uuid import UUID

from app.rag.embeddings import generate_embedding
from app.rag.vector_store import get_index


CACHE_NAMESPACE = "llm-cache"
DEFAULT_SIMILARITY_THRESHOLD = 0.90


class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.similarity_threshold = similarity_threshold

    def get(
        self,
        query: str,
        knowledge_base_id: UUID,
        version_id: UUID,
    ) -> str | None:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = generate_embedding(query)

        index = get_index()

        result = index.query(
            vector=query_embedding,
            top_k=1,
            namespace=CACHE_NAMESPACE,
            include_metadata=True,
            filter={
                "knowledge_base_id": str(knowledge_base_id),
                "version_id": str(version_id),
            },
        )

        matches = result.get("matches", [])

        if not matches:
            return None

        match = matches[0]

        if match.get("score", 0.0) < self.similarity_threshold:
            return None

        metadata = match.get("metadata", {})

        return metadata.get("response")

    def set(
        self,
        query: str,
        response: str,
        knowledge_base_id: UUID,
        version_id: UUID,
    ) -> None:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not response:
            raise ValueError("Response cannot be empty")

        query_embedding = generate_embedding(query)

        index = get_index()

        cache_id = (
            f"{knowledge_base_id}:"
            f"{version_id}:"
            f"{hash(query)}"
        )

        index.upsert(
            namespace=CACHE_NAMESPACE,
            vectors=[
                {
                    "id": cache_id,
                    "values": query_embedding,
                    "metadata": {
                        "knowledge_base_id": str(
                            knowledge_base_id
                        ),
                        "version_id": str(version_id),
                        "query": query,
                        "response": response,
                    },
                }
            ],
        )