import json
from llm.json_call import json_call
from reasoning.models import Answer, Critique, Evidence
from reasoning.prompts import RULES, context


def synthesize(question: str, evidence: list[Evidence], draft: Answer | None = None,
               critique: Critique | None = None) -> Answer:
    prompt = RULES + """\nAnswer using atomic, individually verifiable claims, each with chunk citation_ids.
Every factual statement including contradictions must be a cited claim. Explicitly describe conflicting
findings as separate cited claims; avoid unsupported cross-paper numerical comparisons.
Use stable unique claim_ids. Limitations may describe ONLY missing evidence or scope, not new facts.
If no answer is supported, return claims=[], abstained=true and explain the missing evidence in limitations.
For partial answers, retain supported claims and identify unanswered parts in limitations.\nQuestion: """
    prompt += json.dumps(question) + "\nEvidence:\n" + context(evidence)
    if draft is not None:
        prompt += "\nRevise the draft ONCE. Correct or remove every flagged claim; never replace it with speculation."
        prompt += "\nDraft: " + draft.model_dump_json() + "\nCritique: " + critique.model_dump_json()
    return json_call(prompt, Answer)
