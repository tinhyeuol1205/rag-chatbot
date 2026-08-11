from retrieval.search.sparse import tokenize


def test_keeps_product_codes_with_punctuation():
    """★ Bug P2-9: mã hiệu dính dấu câu vẫn phải khớp."""
    assert "tc-456" in tokenize("See TC-456.")
    assert "tc-456" in tokenize("(TC-456), closed")
    assert tokenize("See TC-456.") == tokenize("see tc-456")


def test_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("...!?") == []


def test_handles_normal_text():
    assert "laptop" in tokenize("What is the laptop policy?")
    assert tokenize("Annual Leave") == ["annual", "leave"]


def test_preserves_vietnamese_words():
    assert tokenize("Chính sách nghỉ phép") == ["chính", "sách", "nghỉ", "phép"]


def test_preserves_unicode_and_product_codes_together():
    assert tokenize("Quy định TC-456 áp dụng.") == [
        "quy", "định", "tc-456", "áp", "dụng"
    ]


def test_nfkc_normalization():
    assert tokenize("ＡＢＣ-１２３") == ["abc-123"]
