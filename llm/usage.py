"""Request-scoped limits and provider-reported usage; never log prompt contents."""
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

current_usage = ContextVar('llm_usage', default=None)
_log_lock = Lock()


@dataclass
class Usage:
    max_calls: int
    max_prompt_chars: int | None = None
    max_output_tokens: int | None = None
    calls: list = field(default_factory=list)

    def reserve(self, prompt, output_limit):
        from llm.watsonx_client import BudgetExceeded
        if len(self.calls) >= self.max_calls:
            raise BudgetExceeded('Request stopped at its model-call limit, including JSON repairs.')
        if self.max_prompt_chars is not None and len(prompt) > self.max_prompt_chars:
            raise BudgetExceeded('Request stopped because the model prompt exceeds its context budget.')
        record = {'prompt_chars': len(prompt), 'output_limit': output_limit,
                  'prompt_tokens': None, 'completion_tokens': None, 'status': 'started'}
        self.calls.append(record)
        return record

    def report(self):
        complete = all(c['prompt_tokens'] is not None and c['completion_tokens'] is not None for c in self.calls)
        return {'model_calls': len(self.calls), 'usage_complete': complete,
                'prompt_tokens': sum(c['prompt_tokens'] or 0 for c in self.calls) if complete else None,
                'completion_tokens': sum(c['completion_tokens'] or 0 for c in self.calls) if complete else None,
                'calls': self.calls}


@contextmanager
def track_usage(mode):
    usage = Usage(max_calls=6 if mode == 'efficient' else 16,
                  max_prompt_chars=32000 if mode == 'efficient' else None,
                  max_output_tokens=1800 if mode == 'efficient' else None)
    token = current_usage.set(usage)
    started = time.monotonic()
    try:
        yield usage
    finally:
        current_usage.reset(token)
        # Includes failed requests. Missing provider counts stay unknown, never guessed.
        entry = {'request_id': uuid.uuid4().hex, 'mode': mode,
                 'recorded_at': datetime.now(timezone.utc).isoformat(),
                 'model_id': os.getenv('WATSONX_MODEL_ID'),
                 'elapsed_seconds': round(time.monotonic() - started, 3), **usage.report()}
        try:
            path = Path(os.getenv('LLM_USAGE_LOG', 'data/usage.jsonl'))
            with _log_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as stream:
                    stream.write(json.dumps(entry) + '\n')
        except OSError:
            logging.getLogger(__name__).warning('Could not persist usage totals; check LLM_USAGE_LOG permissions.')
