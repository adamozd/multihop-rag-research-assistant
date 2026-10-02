import json
from llm.json_call import json_call
from reasoning.models import Evidence, HopDecision
from reasoning.prompts import RULES, context


def decide(question: str, subquestions: list[str], evidence: list[Evidence], queries: list[str]) -> HopDecision:
    return json_call(RULES + "\nAssess whether every subquestion has sufficient evidence. "
                     "If not, propose one NEW targeted query addressing the most important missing link. "
                     "If sufficient, query must be null. Do not equate similar vocabulary with support.\n"
                     + json.dumps({"question": question, "subquestions": subquestions, "queries_already_used": queries})
                     + "\nEvidence:\n" + context(evidence), HopDecision)
