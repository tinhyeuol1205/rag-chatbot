from retrieval.search.hybrid import rrf_fusion


def _doc(cid: str) -> dict:
    return {"chunk_id": cid, "content": f"content-{cid}"}


def test_rrf_sums_across_lists():
    """★ Bug P1-3: chunk ở NHIỀU list phải thắng chunk chỉ ở 1 list."""
    list_a = [_doc("A"), _doc("B")]
    list_b = [_doc("B"), _doc("C")]
    merged = rrf_fusion(list_a, list_b)
    assert merged[0]["chunk_id"] == "B"          # B ở cả 2 list
    assert merged[0]["n_hits"] == 2


def test_rrf_rank_starts_at_one():
    """★ Bug P2-8: công thức phải là 1/(k+1) cho hit đầu tiên."""
    merged = rrf_fusion([_doc("A")], k=60)
    assert merged[0]["rrf_score"] == 1.0 / 61


def test_rrf_empty_input():
    assert rrf_fusion() == []
    assert rrf_fusion([], []) == []


def test_rrf_does_not_mutate_input():
    list_a = [_doc("A")]
    rrf_fusion(list_a)
    assert "rrf_score" not in list_a[0]


def test_rrf_weights_prefer_hyde_list():
    """weights: list HyDE nặng hơn → doc chỉ ở list HyDE thắng."""
    list_a = [_doc("A")]   # HyDE (weight 2.0)
    list_b = [_doc("B")]   # thường (weight 1.0)
    merged = rrf_fusion(list_a, list_b, weights=[2.0, 1.0], k=60)
    assert merged[0]["chunk_id"] == "A"
