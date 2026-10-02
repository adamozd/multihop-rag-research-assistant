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
