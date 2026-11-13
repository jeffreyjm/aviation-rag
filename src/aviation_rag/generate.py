"""Cited generation with abstention and citation validation.

Backends share one interface, `generate(task, question, sources) -> str`:
  * AnthropicGenerator  - grounded LLM output (needs ANTHROPIC_API_KEY)
  * ExtractiveGenerator - offline baseline built only from source text (cannot hallucinate)

`task` is "qa" (answer a question) or "investigate" (draft an investigation summary).
`answer()` wraps retrieval, abstention, generation and citation validation.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


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
