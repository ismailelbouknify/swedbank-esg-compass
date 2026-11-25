"""Answering one questionnaire question: retrieved passages -> one model call -> verification -> at most one
corrective retry. No passage means no call and no answer."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.services.llm.base import LLMProvider, LLMUnavailableError
from app.services.rag.prompt import INSUFFICIENT_EVIDENCE, SYSTEM_PROMPT, response_schema, user_prompt
from app.services.rag.questions import Question
from app.services.rag.retrieval import Passage
from app.services.rag.verify import Quote, Verification, verify_answer

log = logging.getLogger(__name__)

# Outcomes
VERIFIED = "verified"  # a level with verified quotations
ABSTAINED = "abstained"  # the model said the evidence is insufficient
NO_PASSAGES = "no_passages"  # nothing relevant was retrieved: the model was not asked
UNVERIFIED = "unverified"  # the answer failed verification twice: level withheld, flagged for review
MODEL_ERROR = "model_error"  # the model could not be used for this question
BUDGET = "budget"  # the assessment's model-call ceiling was reached first


@dataclass
class Answer:
    question: Question
    outcome: str
    passages: list[Passage]
    level: int | None = None  # set only for VERIFIED
    proposed_level: int | None = None  # what the model proposed when verification failed
    explanation: str = ""
    quotes: list[Quote] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0
    candidates: int = 0  # retrieval diagnostic: passages sharing any query term


class Answerer:
    def __init__(self, llm: LLMProvider, company: str, year: int, *, max_output_tokens: int, call_budget: int | None,
                 names_company: Callable[[str], bool] | None = None):
        self.llm, self.company, self.year = llm, company, year
        self.names_company = names_company
        self.max_output_tokens = max_output_tokens
        self.call_budget = call_budget  # None: no ceiling
        self.calls = 0
        self.seconds = 0.0

    def _budget_left(self) -> bool:
        return self.call_budget is None or self.calls < self.call_budget

    def _call(self, q: Question, passages: list[Passage], correction: list[str] | None) -> str:
        self.calls += 1
        started = time.perf_counter()
        try:
            return self.llm.complete_json(SYSTEM_PROMPT, user_prompt(q, self.company, self.year, passages, correction),
                                          schema=response_schema(q, passages), max_tokens=self.max_output_tokens)
        finally:
            self.seconds += time.perf_counter() - started

    def answer(self, q: Question, passages: list[Passage], accepted_source_ids: set[int], candidates: int = 0) -> Answer:
        result = Answer(q, NO_PASSAGES, passages, candidates=candidates)
        if not passages:
            return result
        calls_before, seconds_before = self.calls, self.seconds
        v: Verification | None = None
        correction: list[str] | None = None
        try:
            for attempt in (1, 2):
                if not self._budget_left():
                    result.outcome = BUDGET
                    break
                raw = self._call(q, passages, correction)
                v = verify_answer(raw, max_level=q.max_level, passages=passages, accepted_source_ids=accepted_source_ids,
                                  names_company=self.names_company)
                if v.ok:
                    break
                log.info("rag_verification_failed: question=%s:%s attempt=%d codes=%s", q.factor, q.dimension, attempt, v.codes)
                correction = v.errors
        except LLMUnavailableError as exc:
            result.outcome, result.errors = MODEL_ERROR, [str(exc)[:300]]
            v = None
        result.calls, result.seconds = self.calls - calls_before, self.seconds - seconds_before
        if v is None:
            return result
        result.explanation, result.errors, result.codes = v.explanation, v.errors, v.codes
        if not v.ok:
            result.outcome, result.proposed_level, result.quotes = UNVERIFIED, v.level, v.quotes
        elif v.status == INSUFFICIENT_EVIDENCE:
            result.outcome = ABSTAINED
        else:
            result.outcome, result.level, result.quotes = VERIFIED, v.level, v.quotes
        return result
