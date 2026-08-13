from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

_model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Cannot generate an embedding for empty text")

    embedding = _model.encode(
        text,
        normalize_embeddings=True,
    )

    return embedding.tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Cannot generate embeddings for empty text")

    embeddings = _model.encode(
        texts,
        normalize_embeddings=True,
    )

    return embeddings.tolist()