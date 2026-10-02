import json
import pytest
from eval.metrics import paper_recall, summarize, support_counts
from eval.run_eval import score_human
from reasoning.models import Answer, Critique


def test_empty_support_is_not_perfect():
    assert support_counts(Critique(verdicts=[]))["rate"] is None


def test_hop_recall_uses_citations_not_retrieval(evidence):
    answer = Answer(claims=[], abstained=True, limitations=[])
    assert paper_recall(answer, [evidence], ["p1"]) == 0
    assert paper_recall(answer, [evidence], []) is None


def test_pooled_support_and_abstention_are_separate():
    value = {"supported": 1, "claims": 2, "hop_recall": 0.5, "abstained": False}
    empty = {"supported": 0, "claims": 0, "hop_recall": 0, "abstained": True}
    rows = [{"id": "q", "kind": "multi_hop", "should_abstain": False,
             "without_critique": value, "with_critique": empty}]
    report = summarize(rows)
    assert report["without_critique"]["claim_support_rate"] == 0.5
    assert report["with_critique"]["claim_support_rate"] is None
    assert report["with_critique"]["answerable_response_rate"] == 0
    assert report["support_rate_delta"] is None


def test_human_scoring_requires_complete_labels(tmp_path):
    run_path = tmp_path / "run.json"
    annotations = tmp_path / "review.json"
    run_path.write_text(json.dumps({"run_id": "test", "review_map": {"x": {}}, "rows": []}))
    annotations.write_text("[]")
    with pytest.raises(ValueError, match="exactly"):
        score_human(run_path, annotations)


def test_eval_corpus_alignment():
    from pathlib import Path
    questions = json.loads(Path("eval/eval_questions.json").read_text())
    papers = {p["id"] for p in json.loads(Path("ingestion/corpus.json").read_text())["papers"]}
    assert 15 <= len(questions) <= 20
    assert len({q["id"] for q in questions}) == len(questions)
    assert all(set(q["expected_papers"]) <= papers for q in questions)
    assert all(len(q["expected_papers"]) >= 2 for q in questions if q["kind"] == "multi_hop")


def test_human_scoring_overrides_internal_judgment(tmp_path):
    run_path = tmp_path / "run.json"
    annotations = tmp_path / "review.json"
    value = {"supported": 1, "claims": 1, "hop_recall": 1.0, "abstained": False}
    run_path.write_text(json.dumps({"run_id": "r", "review_map": {
        "first": {"question_id": "q", "arm": "without_critique"},
        "second": {"question_id": "q", "arm": "with_critique"}},
        "rows": [{"id": "q", "kind": "multi_hop", "should_abstain": False,
                  "without_critique": value, "with_critique": value}]}))
    annotations.write_text(json.dumps([{"review_id": "first", "label": "unsupported"},
                                       {"review_id": "second", "label": "supported"}]))
    score_human(run_path, annotations)
    report = json.loads((tmp_path / "human-report.json").read_text())
    assert report["metrics"]["support_rate_delta"] == 1.0
