from functools import lru_cache
import os

MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, device="cpu", cache_folder=os.getenv("EMBEDDING_CACHE", "data/models"))


def embed(texts: list[str], query: bool = False) -> list[list[float]]:
    inputs = [QUERY_PREFIX + text for text in texts] if query else texts
    return embedding_model().encode(inputs, normalize_embeddings=True, show_progress_bar=False).tolist()
