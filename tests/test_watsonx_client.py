from unittest.mock import Mock

import pytest

from llm import watsonx_client as client
from llm.json_call import json_call
from reasoning.models import Decomposition


def response(text, reason="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": reason}]}


def test_instruction_call_uses_chat_parameters(monkeypatch):
    model = Mock()
    model.chat.return_value = response('{"ok": true}')
    monkeypatch.setattr(client, "_model", lambda: model)
    monkeypatch.setenv("WATSONX_MAX_NEW_TOKENS", "120")
    assert client.llm_call("Return JSON") == '{"ok": true}'
    model.chat.assert_called_once()
    sent = model.chat.call_args.kwargs
    assert sent["messages"][0]["role"] == "system"
    assert "JSON" in sent["messages"][0]["content"]
    assert sent["messages"][-1] == {"role": "user", "content": "Return JSON"}
    assert sent["params"] == {"temperature": 0, "max_tokens": 120,
                              "response_format": {"type": "json_object"}}
    model.generate_text.assert_not_called()


@pytest.mark.parametrize("payload", [None, {}, {"choices": []}, {"choices": [None]}])
def test_malformed_response_is_explicit(payload):
    with pytest.raises(client.LLMError, match="chat"):
        client._chat_text(payload)


@pytest.mark.parametrize("text", [None, "", "   ", []])
def test_empty_text_reports_finish_reason(text):
    with pytest.raises(client.LLMError, match="finish_reason=length"):
        client._chat_text(response(text, "length"))


def test_unknown_provider_status_is_not_exposed():
    with pytest.raises(client.LLMError, match="finish_reason=unknown") as error:
        client._chat_text(response("", "sensitive-provider-details"))
    assert "sensitive" not in str(error.value)


def test_provider_errors_do_not_expose_payloads(monkeypatch):
    model = Mock()
    model.chat.side_effect = RuntimeError("secret-test-token")
    monkeypatch.setattr(client, "_model", lambda: model)
    with pytest.raises(client.LLMError, match="watsonx inference failed") as error:
        client.llm_call("test")
    assert "secret-test-token" not in str(error.value)


def test_json_repair_still_uses_shared_chat_client(monkeypatch):
    model = Mock()
    model.chat.side_effect = [response("not JSON"), response('{"subquestions": ["first", "second"]}')]
    monkeypatch.setattr(client, "_model", lambda: model)
    result = json_call("Decompose", Decomposition)
    assert result.subquestions == ["first", "second"]
    assert model.chat.call_count == 2
    repair = model.chat.call_args.kwargs["messages"][-1]["content"]
    assert "your last response was not valid JSON — return only the JSON object" in repair
