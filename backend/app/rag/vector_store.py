from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.rag.embeddings import generate_embedding


pc = Pinecone(api_key=settings.pinecone_api_key)


def create_index() -> None:
    existing_indexes = [index.name for index in pc.list_indexes()]

    if settings.pinecone_index_name in existing_indexes:
        return

    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )


def get_index():
    return pc.Index(settings.pinecone_index_name)


def upsert_chunk(
    chunk_id,
    content: str,
    knowledge_base_id,
    version_id,
    document_id,
    persona: str = "general",
    embedding: list[float] | None = None,
) -> None:
    if embedding is None:
        embedding = generate_embedding(content)

    vector = {
        "id": str(chunk_id),
        "values": embedding,
        "metadata": {
            "knowledge_base_id": str(knowledge_base_id),
            "version_id": str(version_id),
            "document_id": str(document_id),
            "chunk_id": str(chunk_id),
            "persona": persona,
        },
    }

    index = get_index()

    index.upsert(
        vectors=[vector],
    )


def upsert_chunks(
    vectors: list[dict],
) -> None:
    if not vectors:
        return

    index = get_index()

    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]

        index.upsert(
            vectors=batch,
        )


def delete_chunks(chunk_ids: list) -> None:
    if not chunk_ids:
        return

    index = get_index()

    index.delete(
        ids=[str(chunk_id) for chunk_id in chunk_ids],
    )


def search_similar(
    query: str,
    top_k: int = 5,
    version_id=None,
    persona: str | None = None,
):
    if not query.strip():
        raise ValueError("Query cannot be empty")

    query_embedding = generate_embedding(query)

    index = get_index()

    query_kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }

    filters = {}

    if version_id is not None:
        filters["version_id"] = str(version_id)

    if persona is not None:
        filters["persona"] = persona

    if filters:
        query_kwargs["filter"] = filters

    return index.query(**query_kwargs)