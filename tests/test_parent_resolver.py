from retrieval.context.parent_resolver import ParentResolver


class _FakeQdrant:
    """Giả lập Qdrant trả về subset ID (hành vi thật của retrieve)."""

    def __init__(self, store):
        self.store = store

    def get_by_ids(self, collection_name, ids):
        return [type("P", (), {"id": i, "payload": self.store[i]})()
                for i in ids if i in self.store]


def test_missing_parent_falls_back_to_child():
    """★ Bug P0-1: parent mất trong DB thì phải giữ child, KHÔNG được xoá."""
    resolver = ParentResolver()
    resolver.qdrant = _FakeQdrant({})            # parent collection rỗng
    children = [{"chunk_id": "c1", "content": "child text",
                 "parent_id": "p1", "file_name": "a.md"}]
    out = resolver.resolve(children)
    assert len(out) == 1                          # trước fix: len == 0
    assert out[0]["content"] == "child text"


def test_two_children_same_parent_dedupe():
    resolver = ParentResolver()
    resolver.qdrant = _FakeQdrant({"p1": {"content": "parent text", "file_name": "a.md"}})
    children = [
        {"chunk_id": "c1", "content": "x", "parent_id": "p1", "file_name": "a.md"},
        {"chunk_id": "c2", "content": "y", "parent_id": "p1", "file_name": "a.md"},
    ]
    out = resolver.resolve(children)
    assert len(out) == 1
    assert out[0]["content"] == "parent text"


def test_child_without_parent_id_kept():
    resolver = ParentResolver()
    resolver.qdrant = _FakeQdrant({})
    children = [{"chunk_id": "c1", "content": "orphan", "parent_id": None}]
    out = resolver.resolve(children)
    assert len(out) == 1
    assert out[0]["content"] == "orphan"
