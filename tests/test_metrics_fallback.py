from evaluation.metrics import EvalResult, _simple_evaluate


def test_simple_evaluate_marks_itself_as_fallback():
    """★ P2-2: fallback KHÔNG được trông giống RAG Triad thật."""
    results = [EvalResult(question="q", answer="a b c", ground_truth="a b",
                          contexts=["ctx"])]
    out = _simple_evaluate(results)
    assert "_mode" in out
    assert "FALLBACK" in out["_mode"]


def test_simple_evaluate_empty():
    assert _simple_evaluate([]) == {}


def test_keyword_overlap_computed():
    results = [EvalResult(question="q", answer="alpha beta", ground_truth="alpha",
                          contexts=[])]
    out = _simple_evaluate(results)
    assert out["avg_keyword_overlap"] == 1.0      # "alpha" khớp hết ground_truth
