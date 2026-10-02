import logging
from fastapi import FastAPI, HTTPException
from llm.json_call import StructuredOutputError
from llm.watsonx_client import LLMError
from reasoning.models import AskRequest, Result
from reasoning.pipeline import run_pipeline

app = FastAPI(title="Multi-Hop Research Synthesis", version="0.1.0")
logger = logging.getLogger(__name__)


@app.post("/ask", response_model=Result)
def ask(request: AskRequest):
    try:
        return run_pipeline(request)
    except (LLMError, StructuredOutputError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("Corpus or configuration error")
        raise HTTPException(status_code=503, detail="Corpus unavailable or configuration invalid. Check server logs.") from exc
    except Exception as exc:
        logger.exception("Research request failed")
        raise HTTPException(status_code=500, detail="Research request failed; check server logs.") from exc
