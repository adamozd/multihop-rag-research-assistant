import json
import pytest
from llm import watsonx_client
from reasoning.models import Evidence


@pytest.fixture
def evidence():
    return Evidence(chunk_id="c1", paper_id="p1", title="Paper One", url="https://arxiv.org/abs/p1",
                    section="Methods", page_start=2, page_end=2, text="The method retrieves two passages.")


@pytest.fixture
def scripted_llm(monkeypatch):
    def install(responses):
        responses = iter(responses)
        prompts = []
        def call(prompt):
            prompts.append(prompt)
            result = next(responses)
            return result if isinstance(result, str) else json.dumps(result)
        monkeypatch.setattr(watsonx_client, "llm_call", call)
        return prompts
    return install
