from langchain_openai import OpenAIEmbeddings

from playlist_rag.config import settings

_embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    dimensions=settings.embedding_dim,
    api_key=settings.openai_api_key or None,
)


def _check_dim(vector: list[float]) -> None:
    if len(vector) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dim mismatch: got {len(vector)}, "
            f"expected {settings.embedding_dim}"
        )


def embed_text(text: str) -> list[float]:
    if not text or not isinstance(text, str):
        raise ValueError("embed_text requires a non-empty string")
    vector = _embeddings.embed_query(text)
    _check_dim(vector)
    return vector


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _embeddings.embed_documents(texts)
    for v in vectors:
        _check_dim(v)
    return vectors
