from retrieval.query_transform.condense import QueryCondenser


class _FakeLLM:
    def __init__(self, out): self.out = out
    def generate(self, **kw): return self.out


def _condenser(out: str) -> QueryCondenser:
    c = QueryCondenser.__new__(QueryCondenser)
    c.llm = _FakeLLM(out)
    return c


def test_no_history_returns_original():
    c = _condenser("SHOULD NOT BE USED")
    assert c.condense("What is X?", []) == "What is X?"


def test_resolves_pronoun():
    c = _condenser("Can employees carry over unused annual leave?")
    out = c.condense("Can I carry them over?",
                     [("How many days of annual leave?", "15 days per year.")])
    assert "annual leave" in out


def test_llm_failure_falls_back_to_original():
    class _Boom:
        def generate(self, **kw): raise RuntimeError("rate limit")
    c = QueryCondenser.__new__(QueryCondenser)
    c.llm = _Boom()
    assert c.condense("Can I carry them over?", [("q", "a")]) == "Can I carry them over?"


def test_absurdly_long_output_rejected():
    c = _condenser("x" * 600)
    assert c.condense("short q", [("q", "a")]) == "short q"
