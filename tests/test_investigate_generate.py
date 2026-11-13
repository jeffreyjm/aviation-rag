from types import SimpleNamespace

from aviation_rag.generate import SYSTEM_PROMPTS, AnthropicGenerator, ExtractiveGenerator, Source
from conftest import FakeReranker


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
