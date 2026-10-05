from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class Evidence(StrictModel):
    chunk_id: str
    paper_id: str
    title: str
    url: str
    section: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    score: float = 0.0


class Decomposition(StrictModel):
    subquestions: list[str] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def nonempty(self):
        if any(not q.strip() for q in self.subquestions) or len(set(self.subquestions)) != len(self.subquestions):
            raise ValueError("Subquestions must be nonempty and unique")
        return self


class HopDecision(StrictModel):
    sufficient: bool
    reason: str = Field(min_length=1)
    query: str | None
    missing_fact: str | None

    @model_validator(mode="after")
    def targeted_query(self):
        if not self.sufficient and (not self.query or not self.query.strip()):
            raise ValueError("Insufficient evidence requires a targeted query")
        if self.sufficient and (self.query is not None or self.missing_fact is not None):
            raise ValueError("Sufficient evidence requires null query and missing_fact")
        if not self.sufficient and not (self.missing_fact and self.missing_fact.strip()):
            raise ValueError("Another hop requires a concrete missing fact")
        return self


class Claim(StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class Answer(StrictModel):
    claims: list[Claim] = Field(max_length=12)
    abstained: bool
    limitations: list[str]

    @model_validator(mode="after")
    def consistency(self):
        ids = [c.claim_id for c in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("Claim IDs must be unique")
        if self.abstained != (len(self.claims) == 0):
            raise ValueError("Abstention requires zero claims; an answer requires claims")
        return self


class Verdict(StrictModel):
    claim_id: str
    status: Literal["supported", "unsupported", "contradicted"]
    explanation: str = Field(min_length=1)
    evidence_ids: list[str]


class Critique(StrictModel):
    verdicts: list[Verdict]


class AskRequest(StrictModel):
    question: str = Field(min_length=5, max_length=2000)
    critique: bool = True
    top_k: int = Field(default=4, ge=1, le=8)
    max_hops: int = Field(default=2, ge=1, le=3)


class Result(StrictModel):
    question: str
    subquestions: list[str]
    draft: Answer
    final: Answer
    draft_answer: str
    final_answer: str
    citations: list[Evidence]
    evidence: list[Evidence]
    critique_log: list[Critique]
    hop_log: list[dict]
    revised: bool
    warnings: list[str]


def render_answer(answer: Answer) -> str:
    if answer.abstained:
        return "Insufficient evidence in the retrieved corpus to answer this question."
    return "\n\n".join(c.text + " " + " ".join(f"[{ref}]" for ref in c.citation_ids) for c in answer.claims)
