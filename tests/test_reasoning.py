import pytest
from llm.json_call import StructuredOutputError, json_call
from reasoning.critic import critique
from reasoning.models import Answer, AskRequest, Decomposition
from reasoning.pipeline import run_pipeline

DECOMPOSE = {"subquestions": ["What is retrieved?", "What evidence supports it?"]}
DONE = {"sufficient": True, "reason": "Evidence covers both questions", "query": None}
DRAFT = {"claims": [{"claim_id": "a", "text": "The method retrieves two passages.", "citation_ids": ["c1"]}],
         "abstained": False, "limitations": []}


def audit(status="supported", ids=None):
    return {"verdicts": [{"claim_id": "a", "status": status, "explanation": "Checked the quoted method.",
                          "evidence_ids": ["c1"] if ids is None else ids}]}


class FakeRetriever:
    def __init__(self, evidence):
        self.evidence, self.queries = evidence, []

    def search(self, query, top_k):
        self.queries.append(query)
        return [self.evidence]


def test_json_repair_once(scripted_llm):
    prompts = scripted_llm(["```json\n{}\n```", DECOMPOSE])
    assert len(json_call("decompose", Decomposition).subquestions) == 2
    assert len(prompts) == 2
    assert "your last response was not valid JSON — return only the JSON object" in prompts[1]


def test_json_failure_is_bounded(scripted_llm):
    prompts = scripted_llm(["not JSON", "still not JSON"])
    with pytest.raises(StructuredOutputError):
        json_call("decompose", Decomposition)
    assert len(prompts) == 2


def test_schema_repair(scripted_llm):
    scripted_llm([{"subquestions": ["only one"]}, DECOMPOSE])
    assert len(json_call("decompose", Decomposition).subquestions) == 2


def test_shared_client_full_revision_flow(scripted_llm, evidence):
    prompts = scripted_llm([DECOMPOSE, DONE, DRAFT, audit("unsupported"), DRAFT, audit()])
    result = run_pipeline(AskRequest(question="How does this work?"), FakeRetriever(evidence))
    assert result.revised and len(result.critique_log) == 2
    assert len(prompts) == 6  # Every component and revision used the one patched llm_call.
    assert len(result.evidence) == 1  # Dedup across subquestions.
    assert result.citations[0].chunk_id == "c1"


def test_revision_cap_retains_unresolved_flags(scripted_llm, evidence):
    scripted_llm([DECOMPOSE, DONE, DRAFT, audit("unsupported"), DRAFT, audit("contradicted")])
    result = run_pipeline(AskRequest(question="How does this work?"), FakeRetriever(evidence))
    assert len(result.critique_log) == 2
    assert any("remain unsupported" in warning for warning in result.warnings)


def test_supported_draft_is_not_revised(scripted_llm, evidence):
    prompts = scripted_llm([DECOMPOSE, DONE, DRAFT, audit()])
    result = run_pipeline(AskRequest(question="How does this work?"), FakeRetriever(evidence))
    assert not result.revised and len(prompts) == 4


def test_ablation_disables_critic(scripted_llm, evidence):
    prompts = scripted_llm([DECOMPOSE, DONE, DRAFT])
    result = run_pipeline(AskRequest(question="How does this work?", critique=False), FakeRetriever(evidence))
    assert result.critique_log == [] and len(prompts) == 3


def test_hop_limit_and_targeted_query(scripted_llm, evidence):
    missing = {"sufficient": False, "reason": "Missing bridge", "query": "targeted bridge"}
    scripted_llm([DECOMPOSE, missing, missing, DRAFT])
    retriever = FakeRetriever(evidence)
    result = run_pipeline(AskRequest(question="How does this work?", critique=False, max_hops=2), retriever)
    assert len(result.hop_log) == 2 and retriever.queries[-1] == "targeted bridge"
    assert len(retriever.queries) == 3
    assert any("hop limit" in w for w in result.warnings)


def test_repeated_query_stops(scripted_llm, evidence):
    scripted_llm([DECOMPOSE, {"sufficient": False, "reason": "Missing", "query": "WHAT IS RETRIEVED?"}, DRAFT])
    result = run_pipeline(AskRequest(question="How does this work?", critique=False), FakeRetriever(evidence))
    assert len(result.hop_log) == 1
    assert any("repeated" in w for w in result.warnings)


def test_critic_cannot_omit_a_claim(scripted_llm, evidence):
    scripted_llm([{"verdicts": []}])
    with pytest.raises(StructuredOutputError, match="every claim"):
        critique(Answer.model_validate(DRAFT), [evidence])


@pytest.mark.parametrize("citation_ids,verdict_ids", [(["fake"], ["c1"]), (["c1"], ["fake"]), (["c1"], [])])
def test_citation_integrity_overrides_model(scripted_llm, evidence, citation_ids, verdict_ids):
    answer = Answer.model_validate(DRAFT)
    answer.claims[0].citation_ids = citation_ids
    scripted_llm([audit(ids=verdict_ids)])
    assert critique(answer, [evidence]).verdicts[0].status == "unsupported"


def test_abstention_needs_no_critic_call(scripted_llm, evidence):
    prompts = scripted_llm([DECOMPOSE, DONE, {"claims": [], "abstained": True, "limitations": ["No evidence"]}])
    result = run_pipeline(AskRequest(question="Unknown result?"), FakeRetriever(evidence))
    assert result.final.abstained and len(prompts) == 3
    assert result.critique_log[0].verdicts == []


def test_evidence_budget_preserves_later_hops(scripted_llm, evidence):
    class ManyRetriever:
        def search(self, query, top_k):
            return [evidence.model_copy(update={"chunk_id": f"{query}-{i}"}) for i in range(top_k)]
    scripted_llm([{"subquestions": ["q1", "q2", "q3", "q4"]},
                  {"sufficient": False, "reason": "Missing first link", "query": "bridge1"},
                  {"sufficient": False, "reason": "Missing second link", "query": "bridge2"}, DONE,
                  {"claims": [], "abstained": True, "limitations": []}])
    result = run_pipeline(AskRequest(question="Complex research question", max_hops=3, top_k=8), ManyRetriever())
    assert len(result.evidence) == 24
    assert [len(h["added_chunk_ids"]) for h in result.hop_log] == [8, 8, 8]
    assert len({e.chunk_id.split("-")[0] for e in result.evidence[:8]}) == 4
