import os
from dotenv import load_dotenv
from reasoning.models import Evidence
from retrieval.embed import MODEL_ID, embed


class Retriever:
    def __init__(self, collection_id: str = "default"):
        import chromadb
        load_dotenv()
        from library.store import get_collection
        get_collection(collection_id)
        name = (os.getenv("CHROMA_COLLECTION", "research_bge_small_v1") if collection_id == "default"
                else "research_user_" + collection_id)
        self.client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "data/chroma"))
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": MODEL_ID})
        if (self.collection.metadata or {}).get("embedding_model") != MODEL_ID:
            raise ValueError("Collection embedding model mismatch. Use a new collection.")

    def replace_paper(self, paper_id: str, chunks: list[Evidence]):
        if not chunks:
            raise ValueError(f"No chunks extracted for {paper_id}; existing index was preserved.")
        if any(c.paper_id != paper_id for c in chunks):
            raise ValueError("Chunks must belong to the specified paper")
        # Complete embedding before updating persisted data. Upsert before deleting stale IDs.
        vectors = embed([c.text for c in chunks])
        old = self.collection.get(where={"paper_id": paper_id})["ids"]
        for start in range(0, len(chunks), 100):
            batch = chunks[start:start + 100]
            self.collection.upsert(ids=[c.chunk_id for c in batch], embeddings=vectors[start:start + 100],
                                   documents=[c.text for c in batch],
                                   metadatas=[c.model_dump(exclude={"text", "score", "chunk_id"}) for c in batch])
        stale = sorted(set(old) - {c.chunk_id for c in chunks})
        if stale:
            self.collection.delete(ids=stale)

    def fingerprint(self):
        """Invalidate cached answers when indexed text or metadata changes."""
        import hashlib
        import json
        raw = self.collection.get(include=["documents", "metadatas"])
        rows = sorted(zip(raw["ids"], raw["documents"], raw["metadatas"]), key=lambda row: row[0])
        payload = {"rows": rows, "metadata": self.collection.metadata}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def search(self, query: str, top_k: int) -> list[Evidence]:
        count = self.collection.count()
        if not count:
            raise ValueError("The corpus is empty. Run python -m ingestion.fetch_papers first.")
        raw = self.collection.query(query_embeddings=embed([query], query=True), n_results=min(top_k, count),
                                    include=["documents", "metadatas", "distances"])
        return [Evidence(chunk_id=cid, text=text, score=float(1 - distance), **metadata)
                for cid, text, metadata, distance in zip(raw["ids"][0], raw["documents"][0],
                                                        raw["metadatas"][0], raw["distances"][0])]
