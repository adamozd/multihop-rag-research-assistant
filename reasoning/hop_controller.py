import json
from llm.json_call import json_call
from reasoning.models import Evidence, HopDecision
from reasoning.prompts import RULES, context


def decide(question: str, subquestions: list[str], evidence: list[Evidence], queries: list[str]) -> HopDecision:
    prompt = RULES + """
Judge sufficiency for the ORIGINAL user question, using subquestions only as retrieval aids.
Ignore any subquestion that expands beyond the requested scope. If insufficient, missing_fact must name
one specific unanswered fact necessary for the original question; propose one NEW query naming the
method and that fact. Explain in reason what existing evidence establishes and what is absent.
Do not request generic additional context, tradeoffs, performance, or exhaustive details unless asked.
Stop as soon as the requested facets are evidenced, even if unrelated details are absent.
If sufficient, query and missing_fact must both be null. Do not equate similar vocabulary with support.
"""
    return json_call(prompt + json.dumps({"question": question, "subquestions": subquestions,
                                         "queries_already_used": queries})
                     + "\nEvidence:\n" + context(evidence), HopDecision)
