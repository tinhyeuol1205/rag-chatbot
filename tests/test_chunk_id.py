import uuid

from ingestion.models import Chunk, DocumentMetadata


def _md() -> DocumentMetadata:
    return DocumentMetadata(file_name="a.md", file_type="md", source_path="/x/a.md")


def test_same_prefix_different_content_gets_different_id():
    """★ Bug P2-5: 200 ký tự đầu giống nhau nhưng nội dung khác → ID phải khác."""
    prefix = "A" * 250
    c1 = Chunk(content=prefix + "ENDING-ONE", position="0:0", metadata=_md())
    c2 = Chunk(content=prefix + "ENDING-TWO", position="0:1", metadata=_md())
    assert c1.chunk_id != c2.chunk_id


def test_same_content_different_position_gets_different_id():
    """Cùng text ở 2 vị trí → 2 chunk riêng, không collapse."""
    a = Chunk(content="repeated boilerplate", position="0:0", metadata=_md())
    b = Chunk(content="repeated boilerplate", position="3:1", metadata=_md())
    assert a.chunk_id != b.chunk_id


def test_chunk_id_is_valid_uuid():
    """★ Qdrant chỉ nhận uint64 hoặc UUID."""
    c = Chunk(content="hello", position="0:0", metadata=_md())
    uuid.UUID(c.chunk_id)          # raise nếu không phải UUID hợp lệ


def test_chunk_id_deterministic():
    """Re-ingest content không đổi → cùng ID (idempotent)."""
    a = Chunk(content="same", position="0:0", metadata=_md())
    b = Chunk(content="same", position="0:0", metadata=_md())
    assert a.chunk_id == b.chunk_id
