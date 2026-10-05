from streamlit.testing.v1 import AppTest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def test_frontend_loads_and_validates_question():
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.title[0].value == "Research Synthesis"
    app.button[0].click().run()
    assert "at least five" in app.error[0].value


def test_frontend_renders_api_result(monkeypatch, evidence):
    from unittest.mock import Mock
    import requests
    result = {"question": "How does it work?", "warnings": [], "final_answer": "An answer [c1]",
              "draft_answer": "A draft [c1]", "final": {"limitations": []}, "citations": [evidence.model_dump()],
              "critique_log": [{"verdicts": [{"claim_id": "a", "status": "supported"}]}],
              "subquestions": ["one", "two"], "hop_log": []}
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Mock(ok=True, json=lambda: result))
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("How does it work?")
    app.button[0].click().run()
    assert not app.exception
    assert any("An answer" in text.value for text in app.markdown)


def test_frontend_connection_failure_is_actionable(monkeypatch):
    import requests
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(requests, "post", unavailable)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("How does retrieval work?")
    app.button[0].click().run()
    assert not app.exception
    assert "FastAPI server" in app.error[0].value


def test_frontend_numbered_sources_and_abstention(evidence):
    app = AppTest.from_file(APP)
    app.session_state["result"] = {
        "question": "When does it retrieve?", "warnings": [], "final_answer": "",
        "draft_answer": "", "draft": {"claims": [], "abstained": True},
        "final": {"claims": [{"text": "Retrieves on demand.", "citation_ids": ["c1"]}],
                  "abstained": False, "limitations": []},
        "citations": [evidence.model_dump()], "evidence": [evidence.model_dump()],
        "critique_log": [], "subquestions": ["When?", "Which trigger?"],
        "hop_log": [{"hop": 1, "queries": ["trigger"], "added_chunk_ids": ["c1"],
                     "decision": {"reason": "Enough", "missing_fact": None}, "stop_reason": "sufficient"}],
        "revised": False}
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Answer", "Sources", "Review", "Retrieval"]
    assert any("Retrieves on demand. **[1]**" in m.value for m in app.markdown)
    assert any("insufficient" in i.value for i in app.info)
