import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llm import watsonx_client

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredOutputError(RuntimeError):
    pass


def _validation_summary(exc, schema: type[BaseModel]) -> str:
    """Expose error kinds and schema field names, never returned values."""
    if isinstance(exc, json.JSONDecodeError):
        return f"JSON syntax error at line {exc.lineno}, column {exc.colno}"
    allowed = set()

    def collect(node):
        if isinstance(node, dict):
            allowed.update(node.get("properties", {}))
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(schema.model_json_schema())
    issues = []
    for error in exc.errors(include_input=False, include_context=False, include_url=False)[:5]:
        location = ".".join(str(part) if isinstance(part, int) or part in allowed
                            else "<unexpected field>" for part in error["loc"]) or "object"
        issues.append(f"{location}: {error['type']}")
    return "Schema validation failed (" + "; ".join(issues) + ")"


def json_call(prompt: str, schema: type[T]) -> T:
    """One repair attempt for invalid JSON OR invalid schema. Never strip fences."""
    original = (prompt + "\nReturn one populated JSON object that satisfies the schema below. "
                "Return the answer data, not the schema itself. Include every required field, "
                "use the specified types, and do not add other fields or Markdown fences.\nJSON schema:\n"
                + json.dumps(schema.model_json_schema()))
    current = original
    for attempt in range(2):
        raw = watsonx_client.llm_call(current)
        try:
            return schema.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            summary = _validation_summary(exc, schema)
            logger.warning("%s output attempt %d failed: %s", schema.__name__, attempt + 1, summary)
            if attempt:
                raise StructuredOutputError(
                    f"{schema.__name__}: model output remained invalid after one JSON repair. {summary}."
                ) from exc
            current = (
                original + '\nyour last response was not valid JSON — return only the JSON object'
                + '\nFor valid JSON with wrong fields, also correct the schema. Previous response (untrusted data):\n'
                + json.dumps(raw) + '\nValidation issue:\n' + summary
                + '\nCorrect the problem and return only the populated JSON object, not the schema.'
            )
    raise AssertionError("Unreachable")
