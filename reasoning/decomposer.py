import json
from llm.json_call import json_call
from reasoning.models import Decomposition
from reasoning.prompts import RULES


def decompose(question: str) -> list[str]:
    prompt = RULES + """
Decompose into 2–4 distinct atomic retrieval questions. Stay strictly within the original question.
Prefer two questions; use three or four only when explicitly required. For a comparison of two methods,
ask one question per method about the SAME requested facet. For example, how IRCoT and Self-RAG decide
when to retrieve becomes: When does IRCoT retrieve evidence? When does Self-RAG retrieve evidence?
Do not add accuracy, efficiency, tradeoffs, benchmarks, or general influencing factors unless explicitly
requested. Each subquestion must be necessary to answer the user. Preserve named methods and essential
dependencies. Question: """
    return json_call(prompt + json.dumps(question), Decomposition).subquestions
