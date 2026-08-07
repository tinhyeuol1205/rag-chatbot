from retrieval.query_transform.multi_query import MultiQueryExpander


def test_parse_variants_strips_preamble_and_numbering():
    """★ Bug P3-5: bỏ preamble và numbering khỏi variant."""
    exp = MultiQueryExpander.__new__(MultiQueryExpander)   # không cần LLM
    raw = ("Here are 3 alternative versions:\n"
           "1. What is the laptop policy?\n"
           "2. How do employees get a work computer?\n"
           "- Which devices does the company issue?\n")
    out = exp._parse_variants(raw, exclude="laptop policy")
    assert out == [
        "What is the laptop policy?",
        "How do employees get a work computer?",
        "Which devices does the company issue?",
    ]


def test_parse_variants_excludes_original():
    exp = MultiQueryExpander.__new__(MultiQueryExpander)
    out = exp._parse_variants("What is the laptop policy?\nSomething else entirely",
                              exclude="What is the laptop policy?")
    assert out == ["Something else entirely"]


def test_parse_variants_skips_short_and_title_lines():
    exp = MultiQueryExpander.__new__(MultiQueryExpander)
    raw = "Here is:\nNo\nA longer valid question here?\n"
    out = exp._parse_variants(raw, exclude="q")
    assert out == ["A longer valid question here?"]
