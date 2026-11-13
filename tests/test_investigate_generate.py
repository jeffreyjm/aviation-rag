from types import SimpleNamespace

import pytest

from aviation_rag.generate import (
    ABSTAIN_TEXT, INSUFFICIENT, SYSTEM_PROMPTS, AnthropicGenerator, ExtractiveGenerator, Source,
    answer, strip_invalid, validate_citations,
)
from aviation_rag.investigate import aggregate_factors, base_rates
from conftest import FakeReranker


class FakeGen:



    def __init__(self, text):
        self.text, self.calls = text, []


    def generate(self, task, question, sources):
        self.calls.append((task, sources))
        return self.text



def test_validate_and_strip_citations():
    assert validate_citations("A [1]. B [2][7].", 3) == ([1, 2], [7])
    assert strip_invalid("A [1]. B [2][7].", 3) == "A [1]. B [2]."



def test_answer_validates_and_abstains(retriever):
    res = answer("thrust lever stuck landing", retriever, FakeGen("Stuck lever [1][9]."), k=2)
    assert res.invalid_citations == [9] and "[9]" not in res.text and res.cited_report_ids[0] == "1"
    assert answer("x", retriever, FakeGen(INSUFFICIENT), k=2).abstained
    gen = FakeGen("never called [1]")
    res = answer("zzz qqq", retriever, gen, k=2, abstain_below=99.0)
    assert res.abstained and res.text == ABSTAIN_TEXT and gen.calls == []



def test_anthropic_generator_uses_task_specific_prompt():
    captured = {}

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                captured.update(kw)
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=" Draft [1]. ")])

    gen = AnthropicGenerator(client=Client(), model="m")
    out = gen.generate("investigate", "A thing happened.", [Source(1, "42", "B737 | Cruise", "Body.")])
    assert out == "Draft [1]."
    assert captured["system"] == SYSTEM_PROMPTS["investigate"] and "never state a definitive cause" in captured["system"]
    user = captured["messages"][0]["content"]
    assert "[1] (ASRS 42; B737 | Cruise)" in user and "New event description: A thing happened." in user



def test_extractive_investigate_lists_precedent_synopses():
    src = [Source(1, "a", "t", "Crew had a stuck lever. More detail."), Source(2, "b", "t", "Altitude bust. Other.")]
    out = ExtractiveGenerator(FakeReranker()).generate("investigate", "q", src)
    assert out == "Crew had a stuck lever. [1] Altitude bust. [2]"



def test_aggregate_factors_counts_and_lift(index):
    rates = base_rates(index)
    recs = [index.reports["1"], index.reports["5"]]  # both primary problem "Aircraft"
    rows = aggregate_factors(recs, rates)["primary_problem"]
    assert rows[0].label == "Aircraft" and rows[0].count == 2 and rows[0].share == 1.0
    assert rows[0].base_rate == pytest.approx(2 / 5) and rows[0].lift == pytest.approx(1.0 / 0.4)
