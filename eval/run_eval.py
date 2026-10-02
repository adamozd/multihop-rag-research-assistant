"""Paired draft/final ablation. Internal critic scores are NOT independent ground truth."""
import argparse
import hashlib
import json
import os
import platform
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import paper_recall, summarize, support_counts
from reasoning.models import AskRequest
from reasoning.pipeline import run_pipeline


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def score_human(run_path: Path, annotations: Path):
    run = json.loads(run_path.read_text())
    labels = json.loads(annotations.read_text())
    if len({item["review_id"] for item in labels}) != len(labels):
        raise ValueError("Duplicate review IDs")
    by_id = {item["review_id"]: item for item in labels}
    if set(by_id) != set(run["review_map"]):
        raise ValueError("Annotations must contain exactly the review IDs for this run")
    rows = json.loads(json.dumps(run["rows"]))
    by_question = {row["id"]: row for row in rows}
    for row in rows:
        for arm in ("without_critique", "with_critique"):
            row[arm]["supported"] = 0
    for review_id, target in run["review_map"].items():
        status = by_id[review_id]["label"]
        if status not in ("supported", "unsupported", "contradicted"):
            raise ValueError("Every claim needs a supported/unsupported/contradicted label")
        by_question[target["question_id"]][target["arm"]]["supported"] += status == "supported"
    report = {"measurement": "human_claim_support", "run_id": run["run_id"],
              "failures": run.get("failures", []), "metrics": summarize(rows)}
    write_json(run_path.with_name("human-report.json"), report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("eval_questions.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/results"))
    parser.add_argument("--score-run", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    if args.score_run:
        if not args.annotations:
            parser.error("--score-run requires --annotations")
        score_human(args.score_run, args.annotations)
        return
    from dotenv import load_dotenv
    from retrieval.retriever import Retriever
    load_dotenv()
    questions = json.loads(args.questions.read_text())
    if not 15 <= len(questions) <= 20 or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Expected 15–20 questions with unique IDs")
    retriever = Retriever()
    indexed = {m["paper_id"] for m in retriever.collection.get(include=["metadatas"])["metadatas"]}
    required = {pid for q in questions for pid in q["expected_papers"]}
    if required - indexed:
        raise ValueError("Missing expected corpus papers: " + ", ".join(sorted(required - indexed)))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output / run_id
    output.mkdir(parents=True)
    manifest = Path("data/papers/manifest.json")
    run = {"run_id": run_id, "measurement": "internal_critic_proxy_not_independent_ground_truth",
           "model_id": os.getenv("WATSONX_MODEL_ID"), "region_url": os.getenv("WATSONX_URL"),
           "python": platform.python_version(), "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
           "corpus_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
           "settings": {"max_hops": 2, "top_k": 4, "max_revisions": 1},
           "rows": [], "failures": [], "review_map": {}}
    review = []
    for question in questions:
        try:
            result = run_pipeline(AskRequest(question=question["question"]), retriever)
            write_json(output / f"{question['id']}.json", result.model_dump())
            row = {"id": question["id"], "kind": question["kind"], "should_abstain": question["should_abstain"]}
            for arm, answer, audit in (("without_critique", result.draft, result.critique_log[0]),
                                       ("with_critique", result.final, result.critique_log[-1])):
                row[arm] = {**support_counts(audit), "abstained": answer.abstained,
                            "hop_recall": paper_recall(answer, result.evidence, question["expected_papers"])}
                for claim in answer.claims:
                    review_id = uuid.uuid4().hex
                    run["review_map"][review_id] = {"question_id": question["id"], "arm": arm, "claim_id": claim.claim_id}
                    review.append({"review_id": review_id, "question": question["question"], "claim": claim.model_dump(),
                                   "cited_evidence": [e.model_dump() for e in result.evidence if e.chunk_id in claim.citation_ids],
                                   "label": None, "notes": ""})
            run["rows"].append(row)
        except Exception as exc:
            run["failures"].append({"id": question["id"], "error": str(exc)})
        run["metrics"] = summarize(run["rows"])
        write_json(output / "run.json", run)
        print(f"Completed {question['id']}; failures: {len(run['failures'])}", flush=True)
    random.Random(42).shuffle(review)
    write_json(output / "human-review.json", review)
    print(f"Results: {output}")
    if run["failures"]:
        raise SystemExit("Evaluation incomplete. Failures are recorded and excluded, not silently scored.")


if __name__ == "__main__":
    main()
