"""Cited generation with abstention and citation validation.

Backends share one interface, `generate(task, question, sources) -> str`:
  * AnthropicGenerator  - grounded LLM output (needs ANTHROPIC_API_KEY)
  * ExtractiveGenerator - offline baseline built only from source text (cannot hallucinate)

`task` is "qa" (answer a question) or "investigate" (draft an investigation summary).
`answer()` wraps retrieval, abstention, generation and citation validation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from . import config
from .chunk import split_sentences
from .models import Reranker
from .retrieve import Hit, Retriever


ABSTAIN_TEXT = "I couldn't find sufficient support for that in the indexed ASRS reports."


INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


_CITE = re.compile(r"\[(\d+)\]")


_RULES = f"""Rules:
1. Every factual sentence must end with one or more citations like [1] or [2][3] referring to the numbered sources.
2. Use only information stated in the sources. Do not add outside knowledge or invent details.
3. Sources are voluntary pilot/controller reports. Describe what reporters said happened; never state a definitive cause.
4. If the sources do not contain enough information, reply with exactly: {INSUFFICIENT}"""


SYSTEM_PROMPTS = {
    "qa": "You answer questions for aviation safety analysts using ONLY the numbered ASRS report excerpts provided.\n\n"
    + _RULES
    + "\n5. Be concise: at most 4 sentences.",
    "investigate": "You help an aviation safety investigator draft a preliminary summary of a new event, using ONLY the "
    "numbered precedent reports provided.\n\n"
    + _RULES
    + "\n5. Write three short labelled parts: 'Similar precedents' (what earlier reports describe), "
    "'Candidate contributing factors' (hypotheses worth checking, phrased as possibilities), and "
    "'Information still needed' (questions the investigator should answer; these need no citation). "
    "Keep the whole draft under 200 words.",
}



@dataclass
class Source:
    n: int  # 1-based citation number shown to the model
    report_id: str
    title: str  # short context line, e.g. "B737-700 | Initial Approach | Part 121 | 2018"
    text: str



@dataclass
class Answer:
    text: str
    abstained: bool
    sources: list[Source]
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    top_score: float | None = None



    @property
    def cited_report_ids(self) -> list[str]:
        by_n = {s.n: s.report_id for s in self.sources}
        out: list[str] = []
        for n in self.cited:
            if by_n[n] not in out:
                out.append(by_n[n])
        return out



class Generator(Protocol):



    def generate(self, task: str, question: str, sources: list[Source]) -> str: ...



def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.n}] (ASRS {s.report_id}; {s.title})\n{s.text}" for s in sources)



def build_user_prompt(task: str, question: str, sources: list[Source]) -> str:
    label = "New event description" if task == "investigate" else "Question"
    return f"Sources:\n\n{format_sources(sources)}\n\n{label}: {question}"



class AnthropicGenerator:



    def __init__(self, client=None, model: str | None = None, max_tokens: int = 600):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("AVIATION_RAG_MODEL", config.LLM_MODEL)
        self.max_tokens = max_tokens


    def generate(self, task: str, question: str, sources: list[Source]) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPTS[task],
            messages=[{"role": "user", "content": build_user_prompt(task, question, sources)}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()



class ExtractiveGenerator:
    """Offline baseline. qa: best-matching source sentences. investigate: one line per precedent."""



    def __init__(self, reranker: Reranker | None = None, max_sentences: int = 3):
        self.reranker = reranker or Reranker()
        self.max_sentences = max_sentences


    def generate(self, task: str, question: str, sources: list[Source]) -> str:
        if task == "investigate":
            return " ".join(f"{self._first_sentence(s.text)} [{s.n}]" for s in sources[:3]) or INSUFFICIENT
        cands = [(s.n, i, sent) for s in sources for i, sent in enumerate(split_sentences(s.text))]
        if not cands:
            return INSUFFICIENT
        scores = self.reranker.score(question, [c[2] for c in cands])
        best = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)[: self.max_sentences]
        return " ".join(f"{cands[i][2]} [{cands[i][0]}]" for i in sorted(best, key=lambda i: cands[i][:2]))


    @staticmethod
    def _first_sentence(text: str) -> str:
        s = split_sentences(text)
        return s[0] if s else text



def validate_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []
    for m in _CITE.finditer(text):
        n = int(m.group(1))
        bucket = valid if 1 <= n <= n_sources else invalid
        if n not in bucket:
            bucket.append(n)
    return valid, invalid



def strip_invalid(text: str, n_sources: int) -> str:
    return _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)



def make_sources(retriever: Retriever, hits: list[Hit], with_synopsis: bool = False) -> list[Source]:
    out = []
    for n, h in enumerate(hits, start=1):
        c = retriever.index.chunks[h.chunk_idx]
        text = c.text
        if with_synopsis:
            text = f"{retriever.index.reports[c.report_id]['synopsis']} {text}"
        out.append(Source(n=n, report_id=c.report_id, title=c.header, text=text))
    return out



def finalize(raw: str, sources: list[Source], top_score: float | None) -> Answer:
    """Turn raw generator output into an Answer (abstention token + citation validation)."""
    if INSUFFICIENT in raw:
        return Answer(ABSTAIN_TEXT, True, sources, top_score=top_score)
    valid, invalid = validate_citations(raw, len(sources))
    return Answer(strip_invalid(raw, len(sources)), False, sources, valid, invalid, top_score)
