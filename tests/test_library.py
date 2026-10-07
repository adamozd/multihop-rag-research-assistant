import re
from types import SimpleNamespace

import pymupdf
import pytest
from fastapi.testclient import TestClient
from api.main import app
from library import store
from retrieval import retriever as retrieval_module
from retrieval import embed as embedding_module


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LIBRARY_PATH', str(tmp_path / 'library'))
    monkeypatch.setenv('CHROMA_PATH', str(tmp_path / 'chroma'))
    monkeypatch.setattr(retrieval_module, 'embed', lambda texts, query=False: [[1.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(embedding_module, 'embedding_model', lambda: SimpleNamespace(tokenizer=Tokenizer()))
    return TestClient(app)


def pdf_bytes(text='Evidence about adaptation appears on this page.'):
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), 'Abstract', fontname='hebo')
        page.insert_text((72, 110), text)
        return doc.tobytes()


def test_upload_duplicate_persistence_and_collection_isolation(library, scripted_llm):
    a = library.post('/collections', json={'name': 'Climate'}).json()['id']
    b = library.post('/collections', json={'name': 'Medicine'}).json()['id']
    content = pdf_bytes()
    first = library.post(f'/collections/{a}/papers?filename=climate.pdf', content=content)
    assert first.status_code == 201
    paper = first.json()
    assert paper['chunks'] and not paper['already_indexed']
    assert library.post(f'/collections/{a}/papers', content=content).json()['already_indexed']
    assert len(store.papers(a)) == 1 and store.papers(b) == []
    assert retrieval_module.Retriever(a).search('adaptation', 4)[0].paper_id == paper['paper_id']
    assert retrieval_module.Retriever(b).collection.count() == 0
    assert retrieval_module.Retriever().collection.count() == 0
    assert library.get(f'/collections/{b}/papers/{paper["paper_id"]}/pdf').status_code == 404
    assert library.get(f'/collections/{a}/papers/{paper["paper_id"]}/pdf').content == content
    # The same pipeline routes its retrieval to the selected collection.
    scripted_llm([{'subquestions': ['What adaptation?', 'What evidence?']},
                  {'sufficient': True, 'reason': 'Enough', 'query': None, 'missing_fact': None},
                  {'claims': [], 'abstained': True, 'limitations': ['No measured result.']}])
    result = library.post('/ask', json={'question': 'What adaptation evidence exists?', 'collection_id': a, 'critique': False, 'mode': 'baseline'})
    assert result.status_code == 200
    assert result.json()['collection_id'] == a
    assert {e['paper_id'] for e in result.json()['evidence']} == {paper['paper_id']}


def test_pdf_context_and_render(library):
    cid = library.post('/collections', json={'name': 'Sources'}).json()['id']
    paper = library.post(f'/collections/{cid}/papers', content=pdf_bytes()).json()['paper_id']
    base = f'/collections/{cid}/papers/{paper}/pages/'
    page = library.get(base+'1')
    assert page.json()['context_only'] and 'adaptation' in page.json()['text']
    image = library.get(base+'1/image')
    assert image.status_code == 200 and image.content.startswith(b'\x89PNG')
    assert library.get(base+'0').status_code == 404
    assert library.get(base+'2/image').status_code == 404


def test_invalid_uploads_do_not_change_collection(library, monkeypatch):
    cid = library.post('/collections', json={'name': 'Validation'}).json()['id']
    url = f'/collections/{cid}/papers'
    assert library.post(url, content=b'not a PDF').status_code == 422
    with pymupdf.open() as doc:
        doc.new_page()
        blank = doc.tobytes()
    assert library.post(url, content=blank).status_code == 422
    assert library.post('/collections/default/papers', content=pdf_bytes()).status_code == 409
    assert library.post('/collections/'+'a'*32+'/papers', content=pdf_bytes()).status_code == 404
    monkeypatch.setattr(store, 'MAX_PDF_BYTES', 10)
    assert library.post(url, content=b'x'*11).status_code == 413
    assert store.papers(cid) == []
    assert library.post('/collections', json={'name': '   '}).status_code == 422


def test_empty_collection_does_not_call_llm(library, monkeypatch):
    from llm import watsonx_client
    monkeypatch.setattr(watsonx_client, 'llm_call', lambda prompt: pytest.fail('Empty collections must not spend quota'))
    cid = library.post('/collections', json={'name': 'Empty'}).json()['id']
    assert library.post('/ask', json={'question': 'Any evidence?', 'collection_id': cid}).status_code == 503
    assert library.post('/ask', json={'question': 'Any evidence?', 'collection_id': '../escape'}).status_code == 422
    assert library.post('/ask', json={'question': 'Any evidence?', 'collection_id': 'a'*32}).status_code == 404


def test_failed_index_rolls_back_and_allows_retry(library, monkeypatch):
    cid = library.post('/collections', json={'name': 'Retry'}).json()['id']
    original = retrieval_module.Retriever.replace_paper
    def fail(self, paper_id, chunks):
        original(self, paper_id, chunks)
        raise RuntimeError('Simulated write failure')
    monkeypatch.setattr(retrieval_module.Retriever, 'replace_paper', fail)
    response = library.post(f'/collections/{cid}/papers', content=pdf_bytes())
    assert response.status_code == 503
    assert store.papers(cid) == []
    assert retrieval_module.Retriever(cid).collection.count() == 0
    monkeypatch.setattr(retrieval_module.Retriever, 'replace_paper', original)
    assert library.post(f'/collections/{cid}/papers?filename=../../safe.pdf', content=pdf_bytes()).json()['title'] == 'safe'


def test_encrypted_pdf_rejected(library):
    cid = library.post('/collections', json={'name': 'Encrypted'}).json()['id']
    with pymupdf.open() as doc:
        doc.new_page()
        encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='secret')
    result = library.post(f'/collections/{cid}/papers', content=encrypted)
    assert result.status_code == 422 and 'Password' in result.json()['detail']
