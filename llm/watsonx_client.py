"""The only module that knows about the LLM provider."""
import os
from functools import lru_cache
from threading import Lock

from dotenv import load_dotenv


class LLMError(RuntimeError):
    pass


_lock = Lock()


@lru_cache(maxsize=1)
def _model():
    load_dotenv()
    required = ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL_ID")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise LLMError("Missing configuration: " + ", ".join(missing))
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    return ModelInference(
        model_id=os.environ["WATSONX_MODEL_ID"],
        credentials=Credentials(api_key=os.environ["WATSONX_APIKEY"], url=os.environ["WATSONX_URL"]),
        project_id=os.environ["WATSONX_PROJECT_ID"],
        validate=True,
        max_retries=2,
    )


def _chat_text(response: dict) -> str:
    """Extract text without exposing raw provider payloads in API errors."""
    if not isinstance(response, dict):
        raise LLMError("watsonx returned an invalid chat response.")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMError("watsonx returned no chat choices.")
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        # Only expose known status labels, never arbitrary response contents.
        reason = choice.get("finish_reason")
        if reason not in ("stop", "length", "content_filter", "tool_calls", "function_call"):
            reason = "unknown"
        raise LLMError(f"watsonx returned no chat text (finish_reason={reason}).")
    return content


def llm_call(prompt: str) -> str:
    """Generate text; all reasoning calls, including JSON repairs, pass here."""
    try:
        # Shared SDK sessions are serialized; request pipeline state stays local.
        with _lock:
            model = _model()
            max_tokens = int(os.getenv("WATSONX_MAX_NEW_TOKENS", "3000"))
            if max_tokens < 1:
                raise LLMError("WATSONX_MAX_NEW_TOKENS must be a positive integer.")
            # Chat applies the selected model's conversation template.
            response = model.chat(
                messages=[
                    {"role": "system", "content": "Return exactly one valid JSON object. Follow the requested output schema. Do not use Markdown fences or explanatory text outside the JSON."},
                    {"role": "user", "content": prompt},
                ],
                params={"temperature": 0, "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"}},
            )
        return _chat_text(response)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError("watsonx inference failed. Check credentials, region and model availability.") from exc


if __name__ == "__main__":
    # SDK validation checks the configured model against the authenticated region.
    _model()
    print("Configured model validated against the regional watsonx catalog.")
