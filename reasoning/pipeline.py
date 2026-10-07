"""Dispatch between efficient application requests and the original research flow."""
import hashlib
import json
import os
import time
from collections import OrderedDict
from threading import Lock

from dotenv import load_dotenv
from llm.usage import track_usage
from reasoning.baseline import run_pipeline as run_baseline
from reasoning.efficient import run_efficient
from reasoning.models import AskRequest, Result

# A bounded, process-local cache. Restarting the API clears cached paper content.
_cache = OrderedDict()
_run_lock = Lock()
CACHE_SECONDS = 3600
CACHE_ENTRIES = 32


def _cache_key(request, retriever):
    if request.mode != 'efficient' or not request.use_cache or not hasattr(retriever, 'fingerprint'):
        return None
    settings = {key: os.getenv(key, '') for key in
                ('WATSONX_MODEL_ID', 'WATSONX_URL', 'WATSONX_PROJECT_ID', 'WATSONX_MAX_NEW_TOKENS')}
    payload = {'request': request.model_dump(), 'corpus': retriever.fingerprint(), 'settings': settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_pipeline(request: AskRequest, retriever=None) -> Result:
    load_dotenv()
    if retriever is None:
        from retrieval.retriever import Retriever
        retriever = Retriever(request.collection_id)
        if not retriever.collection.count():
            raise ValueError('The selected collection is empty. Index papers before asking a question.')
    # Also prevents simultaneous identical requests from both spending tokens.
    with _run_lock, track_usage(request.mode) as usage:
        key = _cache_key(request, retriever)
        if key in _cache:
            created, saved = _cache[key]
            if time.monotonic() - created < CACHE_SECONDS:
                _cache.move_to_end(key)
                result = saved.model_copy(deep=True)
                result.cache_hit = True
                result.usage = usage.report()
                return result
            del _cache[key]
        result = run_efficient(request, retriever) if request.mode == 'efficient' else run_baseline(request, retriever)
        result.mode = request.mode
        result.usage = usage.report()
        if key is not None:
            _cache[key] = (time.monotonic(), result.model_copy(deep=True))
            while len(_cache) > CACHE_ENTRIES:
                _cache.popitem(last=False)
        return result
