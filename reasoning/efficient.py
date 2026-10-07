"""Bounded two-call synthesis, with at most one targeted retrieval follow-up."""
import json
import re

from llm.json_call import json_call
from reasoning.critic import validate_critique
from reasoning.models import Answer, Critique, HopDecision, Result, StrictModel, render_answer
from reasoning.prompts import RULES

MAX_CONTEXT_CHARS = 8000
MAX_CHUNKS = 4


class DraftDecision(StrictModel):
    answer: Answer
    decision: HopDecision


class ReviewedAnswer(StrictModel):
    critique: Critique
    final: Answer


def select_evidence(items):
    """Keep complete, ranked chunks; never replace evidence with a tiny excerpt."""
    selected, seen, size = [], set(), 0
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        length = len(item.text)
        if len(selected) < MAX_CHUNKS and size + length <= MAX_CONTEXT_CHARS:
            selected.append(item)
            size += length
    return selected


def context(evidence):
    return json.dumps([e.model_dump(exclude={'score', 'url'}) for e in evidence], ensure_ascii=False)


def _key(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def draft(question, evidence, allow_followup):
    return json_call(RULES + '''
Write a concise answer (at most 5 atomic claims), citing chunk IDs for EVERY claim.
Each cited chunk must support the entire claim. Preserve qualifiers, optional versus required,
training versus inference, and limitations of reported comparisons. Report contradictions.
Put only missing evidence/scope limitations in limitations, not uncited factual assertions.
Abstain if no supported answer is possible. Never fabricate an answer to fill a gap.
Also assess evidence coverage: sufficient=true requires query=null and missing_fact=null.
Otherwise name ONE essential missing fact and ONE targeted search query. Do not seek more
context merely to embellish an already sufficient answer. Follow-up available: ''' + str(allow_followup)
                     + '\nQuestion: ' + question + '\nEvidence: ' + context(evidence), DraftDecision)


def _valid_answer(answer, evidence, warnings, forbidden_texts=()):
    known = {e.chunk_id for e in evidence}
    claims = [c for c in answer.claims if set(c.citation_ids) <= known and _key(c.text) not in forbidden_texts]
    if len(claims) != len(answer.claims):
        warnings.append('Removed claims with unresolved citations or unchanged text flagged by the reviewer.')
    return Answer(claims=claims, abstained=not claims, limitations=answer.limitations)


def run_efficient(request, retriever):
    query = request.question
    evidence = select_evidence(retriever.search(query, min(request.top_k, MAX_CHUNKS)))
    warnings, hops, log = [], [], []
    if not evidence:
        empty = Answer(claims=[], abstained=True, limitations=['No complete evidence chunks fit the context budget.'])
        original = final = empty
        warnings.append('No model calls were made because usable evidence was unavailable.')
    else:
        response = draft(query, evidence, request.max_hops > 1)
        hops.append({'hop': 1, 'queries': [query], 'added_chunk_ids': [e.chunk_id for e in evidence],
                     'decision': response.decision.model_dump()})
        if not response.decision.sufficient and request.max_hops > 1 and _key(response.decision.query) != _key(query):
            followup = response.decision.query
            old_ids = {e.chunk_id for e in evidence}
            found = retriever.search(followup, min(request.top_k, MAX_CHUNKS))
            novel = [e for e in found if e.chunk_id not in old_ids]
            # Retain one original chunk for continuity, then prioritize the missing fact.
            combined = select_evidence(evidence[:1] + novel + evidence[1:])
            added = [e.chunk_id for e in combined if e.chunk_id not in old_ids]
            if added:
                evidence = combined
                response = draft(query, evidence, False)
            hops.append({'hop': 2, 'queries': [followup], 'added_chunk_ids': added,
                         'decision': response.decision.model_dump(),
                         'stop_reason': 'followup_limit' if added else 'no_new_evidence'})
        else:
            hops[-1]['stop_reason'] = ('sufficient' if response.decision.sufficient else
                                      'repeated_query' if _key(response.decision.query) == _key(query) else 'hop_limit')
        if not response.decision.sufficient:
            warnings.append('Evidence gap remains: ' + response.decision.missing_fact)
        original = _valid_answer(response.answer, evidence, warnings)
        final = original
        if request.critique and original.claims:
            reviewed = json_call(RULES + '''
Act as an adversarial reviewer. Review EVERY draft claim exactly once against its CITED evidence.
Supported means the cited text entails the ENTIRE claim, including quantities, scope, causal
language, comparisons, optional/mandatory wording, AND/OR, and training versus inference.
Unsupported means missing or partial support; contradicted means evidence explicitly conflicts.
Explain the decisive wording. Each supported verdict needs supporting cited evidence_ids.
Then return the corrected final answer in this SAME response, at most 5 atomic claims.
Keep supported claims unchanged. Correct or remove every flagged assertion everywhere it
recurs. Do not add unrelated claims. Cite only provided chunk IDs. Abstain if nothing is supported.
Limitations must describe missing evidence/scope, not assert additional uncited facts.
The critique describes the draft, NOT a separate audit of your final answer.
Question: ''' + query + '\nDraft: ' + original.model_dump_json() + '\nEvidence: ' + context(evidence), ReviewedAnswer)
            checked = validate_critique(reviewed.critique, original, evidence)
            log.append(checked)
            flagged = {v.claim_id for v in checked.verdicts if v.status != 'supported'}
            forbidden = {_key(c.text) for c in original.claims if c.claim_id in flagged}
            # Supported claims stay unchanged; corrections can only replace a flagged
            # draft ID. This prevents the combined reviewer from adding unaudited claims.
            replacements = {c.claim_id: c for c in reviewed.final.claims}
            candidates = [replacements[c.claim_id] if c.claim_id in flagged else c
                          for c in original.claims if c.claim_id not in flagged or c.claim_id in replacements]
            corrected = Answer(claims=candidates, abstained=not candidates, limitations=reviewed.final.limitations)
            final = _valid_answer(corrected, evidence, warnings, forbidden)
            warnings.append('Review and correction used one call. The final answer has not had a separate second audit.')
        elif request.critique:
            log.append(Critique(verdicts=[]))
        else:
            warnings.append('Critique disabled: this answer has not been audited.')
    refs = {ref for c in final.claims for ref in c.citation_ids}
    return Result(mode='efficient', collection_id=request.collection_id, question=query,
                  subquestions=[query], draft=original, final=final,
                  draft_answer=render_answer(original), final_answer=render_answer(final),
                  evidence=evidence, citations=[e for e in evidence if e.chunk_id in refs],
                  critique_log=log, hop_log=hops, revised=original != final, warnings=warnings)
