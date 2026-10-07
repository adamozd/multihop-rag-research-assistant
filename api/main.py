import logging
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ConfigDict
from library import store
from llm.json_call import StructuredOutputError
from llm.watsonx_client import LLMError, QuotaExceeded, BudgetExceeded
from reasoning.models import AskRequest, Result
from reasoning.pipeline import run_pipeline

app = FastAPI(title="Multi-Hop Research Synthesis", version="0.1.0")
logger = logging.getLogger(__name__)


@app.post("/ask", response_model=Result)
def ask(request: AskRequest):
    try:
        return run_pipeline(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Collection not found") from exc
    except QuotaExceeded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BudgetExceeded as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LLMError, StructuredOutputError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("Corpus or configuration error")
        raise HTTPException(status_code=503, detail="Corpus unavailable or configuration invalid. Check server logs.") from exc
    except Exception as exc:
        logger.exception("Research request failed")
        raise HTTPException(status_code=500, detail="Research request failed; check server logs.") from exc

# Collection and source endpoints use local storage and embeddings, not watsonx.


class CollectionCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    name: str = Field(min_length=1, max_length=80)


@app.get('/collections')
def collections():
    return store.list_collections()


@app.post('/collections', status_code=201)
def create_collection(request: CollectionCreate):
    return store.create_collection(request.name)


@app.get('/collections/{collection_id}/papers')
def collection_papers(collection_id: str):
    try:
        return store.papers(collection_id)
    except KeyError as exc:
        raise HTTPException(404, detail='Collection not found') from exc


@app.post('/collections/{collection_id}/papers', status_code=201)
async def upload_paper(collection_id: str, request: Request, filename: str = Query(default='paper.pdf', max_length=255)):
    try:
        collection = store.get_collection(collection_id)
        if collection['read_only']:
            raise HTTPException(409, detail='Create a personal collection before uploading PDFs.')
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > store.MAX_PDF_BYTES:
                raise HTTPException(413, detail='PDF exceeds the 20 MB limit.')
            data.extend(chunk)
        return await run_in_threadpool(store.ingest_pdf, collection_id, bytes(data), filename)
    except KeyError as exc:
        raise HTTPException(404, detail='Collection not found') from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('PDF indexing failed')
        raise HTTPException(503, detail='PDF indexing failed. Check the local embedding model and server logs; then retry.') from exc


def _paper_path(collection_id, paper_id):
    try:
        return store.paper_path(collection_id, paper_id)
    except KeyError as exc:
        raise HTTPException(404, detail='Paper unavailable in this collection') from exc


@app.get('/collections/{collection_id}/papers/{paper_id}/pdf')
def source_pdf(collection_id: str, paper_id: str):
    return FileResponse(_paper_path(collection_id, paper_id), media_type='application/pdf')


@app.get('/collections/{collection_id}/papers/{paper_id}/pages/{page}')
def source_page(collection_id: str, paper_id: str, page: int):
    import pymupdf
    from ingestion.parse_pdf import parse_pdf
    path = _paper_path(collection_id, paper_id)
    with pymupdf.open(path) as doc:
        if not 1 <= page <= len(doc):
            raise HTTPException(404, detail='Page not found')
        count = len(doc)
    return {'page': page, 'page_count': count, 'text': parse_pdf(path)[page - 1][1],
            'context_only': True}


@app.get('/collections/{collection_id}/papers/{paper_id}/pages/{page}/image')
def source_page_image(collection_id: str, paper_id: str, page: int):
    import pymupdf
    with pymupdf.open(_paper_path(collection_id, paper_id)) as doc:
        if not 1 <= page <= len(doc):
            raise HTTPException(404, detail='Page not found')
        source = doc[page - 1]
        scale = min(2, 1500 / max(source.rect.width, source.rect.height))
        image = source.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes('png')
    return Response(image, media_type='image/png')
