import json
from llm.json_call import json_call
from reasoning.models import Decomposition
from reasoning.prompts import RULES


def decompose(question: str) -> list[str]:
    return json_call(RULES + "\nDecompose into 2–4 distinct atomic retrieval questions. "
                     "Preserve named methods, dependencies and comparisons. Question: " + json.dumps(question),
                     Decomposition).subquestions
