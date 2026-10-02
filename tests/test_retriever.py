from retrieval import retriever as module


def test_persistence_reindex_and_search(tmp_path, monkeypatch, evidence):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(module, "embed", lambda texts, query=False: [[1.0, 0.0, 0.0] for _ in texts])
    retriever = module.Retriever()
    retriever.replace_paper("p1", [evidence])
    assert retriever.search("test", 4)[0].chunk_id == "c1"
    updated = evidence.model_copy(update={"chunk_id": "c2", "text": "Changed content"})
    retriever.replace_paper("p1", [updated])
    reopened = module.Retriever()
    assert reopened.collection.count() == 1
    assert reopened.search("test", 1)[0].text == "Changed content"
