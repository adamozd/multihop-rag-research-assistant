import re
from reasoning.critic import critique
from reasoning.decomposer import decompose
from reasoning.hop_controller import decide
from reasoning.models import AskRequest, Result, render_answer
from reasoning.synthesizer import synthesize

MAX_EVIDENCE = 24


def _query_key(query: str) -> str:
    return " ".join(re.findall(r"\w+", query.casefold()))


def run_pipeline(request: AskRequest, retriever=None) -> Result:
    if retriever is None:
        from retrieval.retriever import Retriever
        retriever = Retriever()
    subquestions = decompose(request.question)
    evidence, queries, hop_log = {}, [], []
    next_queries = subquestions
    warnings = []
    for hop in range(1, request.max_hops + 1):
        added = []
        # Reserve room for later targeted hops instead of filling all context at hop one.
        hop_budget = MAX_EVIDENCE - (request.max_hops - hop) * request.top_k
        batches = [retriever.search(query, request.top_k) for query in next_queries]
        # Round-robin preserves coverage across subquestions under the evidence cap.
        for rank in range(request.top_k):
            for batch in batches:
                if rank < len(batch):
                    item = batch[rank]
                    if item.chunk_id not in evidence and len(evidence) < hop_budget:
                        evidence[item.chunk_id] = item
                        added.append(item.chunk_id)
        queries.extend(next_queries)
        decision = decide(request.question, subquestions, list(evidence.values()), queries)
        hop_log.append({"hop": hop, "queries": next_queries, "added_chunk_ids": added,
                        "decision": decision.model_dump()})
        if decision.sufficient:
            hop_log[-1]["stop_reason"] = "sufficient"
            break
        if hop > 1 and not added:
            hop_log[-1]["stop_reason"] = "no_new_evidence"
            warnings.append(f"Retrieval found no new evidence. Unresolved: {decision.missing_fact}")
            break
        if hop == request.max_hops:
            hop_log[-1]["stop_reason"] = "hop_limit"
            warnings.append(f"Retrieval hop limit reached. Unresolved: {decision.missing_fact}")
            break
        if _query_key(decision.query) in {_query_key(q) for q in queries}:
            hop_log[-1]["stop_reason"] = "repeated_query"
            warnings.append(f"Hop controller repeated a query; retrieval stopped. Unresolved: {decision.missing_fact}")
            break
        next_queries = [decision.query]
    evidence_list = list(evidence.values())
    draft = synthesize(request.question, evidence_list)
    final, log, revised = draft, [], False
    if request.critique:
        log.append(critique(draft, evidence_list))
        if any(v.status != "supported" for v in log[-1].verdicts):
            final = synthesize(request.question, evidence_list, draft, log[-1])
            revised = True
            log.append(critique(final, evidence_list))
        if any(v.status != "supported" for v in log[-1].verdicts):
            warnings.append("Some claims remain unsupported or contradicted after revision. Consult critique log.")
    else:
        warnings.append("Critique disabled: this answer has not been audited.")
    citation_ids = {ref for c in final.claims for ref in c.citation_ids}
    if citation_ids - evidence.keys():
        warnings.append("Answer contains invalid citation IDs; those citations are unresolved.")
    return Result(question=request.question, subquestions=subquestions, draft=draft, final=final,
                  draft_answer=render_answer(draft), final_answer=render_answer(final),
                  evidence=evidence_list, citations=[e for e in evidence_list if e.chunk_id in citation_ids],
                  critique_log=log, hop_log=hop_log, revised=revised, warnings=warnings)
