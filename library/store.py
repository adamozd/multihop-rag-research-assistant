"""Local collection catalog. IDs, never user filenames, determine storage paths."""
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from contextlib import contextmanager

from dotenv import load_dotenv

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200
_ingest_lock = Lock()


def root():
    load_dotenv()
    path = Path(os.getenv('LIBRARY_PATH', 'data/library'))
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _db():
    db = sqlite3.connect(root() / 'catalog.sqlite3')
    db.row_factory = sqlite3.Row
    db.executescript('''CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, name TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS papers (collection_id TEXT, paper_id TEXT, title TEXT, pages INTEGER,
    chunks INTEGER, PRIMARY KEY(collection_id, paper_id));''')
    try:
        with db:
            yield db
    finally:
        db.close()


def get_collection(collection_id):
    if collection_id == 'default':
        return {'id': 'default', 'name': 'RAG & multi-hop QA', 'read_only': True}
    if not re.fullmatch(r'[0-9a-f]{32}', collection_id):
        raise KeyError('Collection not found')
    with _db() as db:
        row = db.execute('SELECT * FROM collections WHERE id=?', (collection_id,)).fetchone()
    if row is None:
        raise KeyError('Collection not found')
    return {**dict(row), 'read_only': False}


def list_collections():
    with _db() as db:
        rows = db.execute('SELECT * FROM collections ORDER BY rowid').fetchall()
    return [get_collection('default')] + [{**dict(r), 'read_only': False} for r in rows]


def create_collection(name):
    name = name.strip()
    if not 1 <= len(name) <= 80:
        raise ValueError('Collection name must contain 1–80 characters.')
    cid = uuid.uuid4().hex
    with _db() as db:
        db.execute('INSERT INTO collections VALUES (?, ?)', (cid, name))
    return get_collection(cid)


def papers(collection_id):
    get_collection(collection_id)
    if collection_id == 'default':
        path = Path('data/papers/manifest.json')
        if not path.exists():
            return []
        return [{'paper_id': p['paper_id'], 'title': p['title'], 'chunks': p['chunks'],
                 'url': p['url']} for p in json.loads(path.read_text())['papers']]
    with _db() as db:
        return [dict(r) for r in db.execute('SELECT paper_id,title,pages,chunks FROM papers WHERE collection_id=? ORDER BY title', (collection_id,))]


def paper_path(collection_id, paper_id):
    get_collection(collection_id)
    if collection_id == 'default':
        manifest = Path('data/papers/manifest.json')
        rows = json.loads(manifest.read_text())['papers'] if manifest.exists() else []
        item = next((p for p in rows if p['paper_id'] == paper_id), None)
        if item:
            path = Path(item['pdf']).resolve()
            # Only serve downloaded corpus files, never arbitrary manifest paths.
            if path.is_relative_to(Path('data/papers').resolve()) and path.is_file():
                return path
    elif re.fullmatch(r'pdf_[0-9a-f]{64}', paper_id):
        with _db() as db:
            item = db.execute('SELECT 1 FROM papers WHERE collection_id=? AND paper_id=?', (collection_id, paper_id)).fetchone()
        path = root() / collection_id / (paper_id + '.pdf')
        if item and path.is_file():
            return path
    raise KeyError('Paper unavailable in this collection')


def ingest_pdf(collection_id, content, filename):
    """Validate and index before advertising a paper as ready; repeated PDFs are idempotent."""
    import pymupdf
    from ingestion.parse_pdf import parse_pdf
    from ingestion.chunk import chunk_paper
    from retrieval.embed import embedding_model
    from retrieval.retriever import Retriever

    if get_collection(collection_id)['read_only']:
        raise ValueError('Create a personal collection to upload PDFs; the benchmark corpus is read-only.')
    if not content or len(content) > MAX_PDF_BYTES:
        raise ValueError('PDF must be nonempty and at most 20 MB.')
    if not content.startswith(b'%PDF-'):
        raise ValueError('The uploaded file is not a PDF.')
    try:
        with pymupdf.open(stream=content, filetype='pdf') as doc:
            if doc.needs_pass:
                raise ValueError('Password-protected PDFs are not supported.')
            if not 1 <= len(doc) <= MAX_PDF_PAGES:
                raise ValueError('PDF must contain 1–200 pages.')
            page_count = len(doc)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError('This PDF could not be opened.') from exc
    paper_id = 'pdf_' + hashlib.sha256(content).hexdigest()
    title = Path(filename.replace('\\', '/')).stem.strip()[:160] or 'Uploaded paper'
    directory = root() / collection_id
    directory.mkdir(parents=True, exist_ok=True)
    with _ingest_lock:
        previous = next((p for p in papers(collection_id) if p['paper_id'] == paper_id), None)
        if previous:
            return {**previous, 'already_indexed': True}
        temporary = None
        retriever = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.pdf', delete=False) as f:
                f.write(content)
                temporary = Path(f.name)
            pages = parse_pdf(temporary)
            paper = {'paper_id': paper_id, 'title': title,
                     'url': f'/collections/{collection_id}/papers/{paper_id}/pdf'}
            chunks = chunk_paper(paper, pages, embedding_model().tokenizer)
            retriever = Retriever(collection_id)
            retriever.replace_paper(paper_id, chunks)
            temporary.replace(directory / (paper_id + '.pdf'))
            with _db() as db:
                db.execute('INSERT INTO papers VALUES (?,?,?,?,?)', (collection_id, paper_id, title, page_count, len(chunks)))
            return {'paper_id': paper_id, 'title': title, 'pages': page_count, 'chunks': len(chunks), 'already_indexed': False}
        except Exception:
            # Do not leave partially indexed evidence from a failed upload searchable.
            if retriever is not None:
                retriever.collection.delete(where={'paper_id': paper_id})
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
