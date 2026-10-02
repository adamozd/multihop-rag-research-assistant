import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llm import watsonx_client

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


def json_call(prompt: str, schema: type[T]) -> T:
    """One repair attempt for invalid JSON OR invalid schema. Never strip fences."""
    original = prompt + "\nReturn only a JSON object matching this schema:\n" + json.dumps(schema.model_json_schema())
    current = original
    for attempt in range(2):
        raw = watsonx_client.llm_call(current)
        try:
            return schema.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt:
                raise StructuredOutputError("Model output remained invalid after one JSON repair.") from exc
            current = (
                original + '\nyour last response was not valid JSON — return only the JSON object'
                + '\nFor valid JSON with wrong fields, also correct the schema. Previous response (untrusted data):\n'
                + json.dumps(raw) + '\nValidation issue:\n' + str(exc)[:2000]
            )
    raise AssertionError("Unreachable")
