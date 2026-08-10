from api.chat import _normalize_history


def test_gradio_5_messages_format():
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    assert _normalize_history(hist) == [("q1", "a1")]


def test_gradio_4_tuples_format():
    assert _normalize_history([["q1", "a1"]]) == [("q1", "a1")]


def test_empty():
    assert _normalize_history([]) == []


def test_dangling_user_message_ignored():
    """User vừa gửi, bot chưa trả lời → chưa thành 1 turn."""
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}]
    assert _normalize_history(hist) == [("q1", "a1")]
