from fastapi.testclient import TestClient
from api import main
from llm.watsonx_client import LLMError
from reasoning.models import AskRequest
from reasoning.pipeline import run_pipeline


def test_api_rejects_invalid_limits():
    client = TestClient(main.app)
    assert client.post("/ask", json={"question": "valid question", "max_hops": 99}).status_code == 422
    assert client.post("/ask", json={"question": "     "}).status_code == 422


def test_api_provider_failure(monkeypatch):
    def fail(request):
        raise LLMError("Missing configuration: WATSONX_APIKEY")
    monkeypatch.setattr(main, "run_pipeline", fail)
    response = TestClient(main.app).post("/ask", json={"question": "valid question"})
    assert response.status_code == 502 and "Missing configuration" in response.json()["detail"]


def test_api_full_response(monkeypatch, scripted_llm, evidence):
    class Retriever:
        def search(self, query, top_k):
            return [evidence]
    scripted_llm([{"subquestions": ["first", "second"]},
                  {"sufficient": True, "reason": "Enough", "query": None, "missing_fact": None},
                  {"claims": [], "abstained": True, "limitations": ["Missing relevant evidence"]}])
    monkeypatch.setattr(main, "run_pipeline", lambda request: run_pipeline(request, Retriever()))
    response = TestClient(main.app).post("/ask", json={"question": "valid question"})
    assert response.status_code == 200
    assert response.json()["final"]["abstained"]
