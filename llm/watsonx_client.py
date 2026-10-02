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
        params={"decoding_method": "greedy", "max_new_tokens": int(os.getenv("WATSONX_MAX_NEW_TOKENS", "3000"))},
        validate=True,
        max_retries=2,
    )


def llm_call(prompt: str) -> str:
    """Generate text; all reasoning calls, including JSON repairs, pass here."""
    try:
        # Shared SDK sessions are serialized; request pipeline state stays local.
        with _lock:
            response = _model().generate_text(prompt=prompt)
        if not isinstance(response, str) or not response.strip():
            raise LLMError("The model returned no text.")
        return response
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError("watsonx inference failed. Check credentials, region and model availability.") from exc


if __name__ == "__main__":
    # SDK validation checks the configured model against the authenticated region.
    _model()
    print("Configured model validated against the regional watsonx catalog.")
