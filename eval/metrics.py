from reasoning.models import Answer, Critique, Evidence


def support_counts(critique: Critique) -> dict:
    total = len(critique.verdicts)
    supported = sum(v.status == "supported" for v in critique.verdicts)
    return {"supported": supported, "claims": total, "rate": supported / total if total else None}


def paper_recall(answer: Answer, evidence: list[Evidence], expected: list[str]) -> float | None:
    if not expected:
        return None
    cited_ids = {cid for claim in answer.claims for cid in claim.citation_ids}
    cited_papers = {e.paper_id for e in evidence if e.chunk_id in cited_ids}
    return len(cited_papers & set(expected)) / len(set(expected))


def summarize(rows: list[dict]) -> dict:
    report = {}
    for arm in ("without_critique", "with_critique"):
        values = [row[arm] for row in rows]
        claims = sum(v["claims"] for v in values)
        supported = sum(v["supported"] for v in values)
        recalls = [row[arm]["hop_recall"] for row in rows
                   if row["kind"] == "multi_hop" and row[arm]["hop_recall"] is not None]
        answerable = [row for row in rows if not row["should_abstain"]]
        no_answer = [row for row in rows if row["should_abstain"]]
        report[arm] = {
            "supported_claims": supported, "total_claims": claims,
            "claim_support_rate": supported / claims if claims else None,
            "mean_multi_hop_paper_recall": sum(recalls) / len(recalls) if recalls else None,
            "answerable_response_rate": sum(not r[arm]["abstained"] for r in answerable) / len(answerable) if answerable else None,
            "no_answer_abstention_rate": sum(r[arm]["abstained"] for r in no_answer) / len(no_answer) if no_answer else None,
        }
    before, after = [report[arm]["claim_support_rate"] for arm in ("without_critique", "with_critique")]
    report["support_rate_delta"] = after - before if before is not None and after is not None else None
    report["questions_scored"] = len(rows)
    return report
