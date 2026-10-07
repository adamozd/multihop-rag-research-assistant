from streamlit.testing.v1 import AppTest
from pathlib import Path
import pytest
from unittest.mock import Mock


@pytest.fixture(autouse=True)
def collection_api(monkeypatch):
    import requests
    def get(url, **kwargs):
        value = [{"id": "default", "name": "RAG & multi-hop QA", "read_only": True}] if url.endswith("/collections") else []
        return Mock(json=lambda: value, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", get)

APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def test_frontend_loads_and_validates_question():
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.title[0].value == "Research Synthesis"
    next(b for b in app.button if b.label == "Synthesize").click().run()
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
    next(b for b in app.button if b.label == "Synthesize").click().run()
    assert not app.exception
    assert any("An answer" in text.value for text in app.markdown)


def test_frontend_connection_failure_is_actionable(monkeypatch):
    import requests
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(requests, "post", unavailable)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("How does retrieval work?")
    next(b for b in app.button if b.label == "Synthesize").click().run()
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


def test_collection_switch_clears_previous_result(monkeypatch):
    import requests
    cid = 'a' * 32
    def get(url, **kwargs):
        data = [{'id': 'default', 'name': 'Starter', 'read_only': True},
                {'id': cid, 'name': 'Climate', 'read_only': False}] if url.endswith('/collections') else []
        return Mock(json=lambda: data, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'get', get)
    app = AppTest.from_file(APP).run()
    app.selectbox[0].set_value(cid).run()
    assert not app.exception
    assert 'result' not in app.session_state
    assert app.session_state['previous_collection'] == cid
    assert any('Add PDFs before' in i.value for i in app.info)
    captured = []
    def post(url, **kwargs):
        captured.append(kwargs['json'])
        return Mock(ok=False, status_code=503, text='Empty collection')
    monkeypatch.setattr(requests, 'post', post)
    app.text_area[0].set_value('What climate evidence exists?')
    next(b for b in app.button if b.label == 'Synthesize').click().run()
    assert captured[-1]['collection_id'] == cid


def test_evidence_explorer_loads_page_context(monkeypatch, evidence):
    import requests
    import pymupdf
    with pymupdf.open() as doc:
        page = doc.new_page()
        picture = page.get_pixmap().tobytes('png')
    def get(url, **kwargs):
        if url.endswith('/collections'):
            data = [{'id': 'default', 'name': 'Starter', 'read_only': True}]
        elif '/pages/' in url:
            data = {'text': 'Additional page context.', 'page': 2}
        else:
            data = []
        return Mock(json=lambda: data, content=picture, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'get', get)
    app = AppTest.from_file(APP)
    app.session_state['result'] = {
        'question': 'How many passages?', 'collection_id': 'default', 'warnings': [],
        'final': {'claims': [{'text': 'Two passages.', 'citation_ids': ['c1']}], 'limitations': []},
        'draft_answer': '', 'final_answer': 'Two passages.', 'citations': [evidence.model_dump()],
        'evidence': [evidence.model_dump()], 'critique_log': [], 'subquestions': [], 'hop_log': []}
    app.run()
    next(b for b in app.button if b.label == 'Show source page').click().run()
    assert not app.exception
    assert any(t.value == evidence.text for t in app.text)
    assert any('does not retroactively' in i.value for i in app.info)
    assert any(t.value == 'Additional page context.' for t in app.text)


def test_efficiency_settings_are_sent_to_api(monkeypatch):
    import requests
    captured = []
    def post(url, **kwargs):
        captured.append(kwargs['json'])
        return Mock(ok=False, status_code=503, text='Quota exhausted')
    monkeypatch.setattr(requests, 'post', post)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value('How does retrieval work?')
    next(b for b in app.button if b.label == 'Synthesize').click().run()
    assert captured[-1]['mode'] == 'efficient' and captured[-1]['use_cache']
    next(s for s in app.selectbox if s.label == 'Reasoning mode').set_value('Research baseline').run()
    next(b for b in app.button if b.label == 'Synthesize').click().run()
    assert not app.exception and captured[-1]['mode'] == 'baseline'
