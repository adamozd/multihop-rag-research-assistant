import pytest

from llm.json_call import StructuredOutputError, json_call
from reasoning.models import Answer, Decomposition


def test_parse_failure_names_stage_without_disclosing_output(scripted_llm, caplog):
    scripted_llm(["private-source-text", "private-source-text"])
    with pytest.raises(StructuredOutputError) as error:
        json_call("decompose", Decomposition)
    assert "Decomposition" in str(error.value)
    assert "JSON syntax error at line 1, column 1" in str(error.value)
    assert "private-source-text" not in str(error.value) + caplog.text


def test_wrong_field_types_have_actionable_safe_diagnostics(scripted_llm):
    scripted_llm([{"claims": [], "abstained": "secret-input", "limitations": []}] * 2)
    with pytest.raises(StructuredOutputError) as error:
        json_call("answer", Answer)
    assert "Answer" in str(error.value)
    assert "abstained: bool_type" in str(error.value)
    assert "secret-input" not in str(error.value)


def test_unknown_field_names_are_redacted(scripted_llm, caplog):
    scripted_llm([{"subquestions": ["first", "second"], "private-key-name": "value"}] * 2)
    with pytest.raises(StructuredOutputError) as error:
        json_call("decompose", Decomposition)
    assert "<unexpected field>: extra_forbidden" in str(error.value)
    assert "private-key-name" not in str(error.value) + caplog.text


def test_repair_addresses_schema_without_relaxing_validation(scripted_llm):
    prompts = scripted_llm([{"subquestions": "wrong-type"}, {"subquestions": ["first", "second"]}])
    assert len(json_call("decompose", Decomposition).subquestions) == 2
    assert "subquestions: list_type" in prompts[1]
    assert "your last response was not valid JSON — return only the JSON object" in prompts[1]
