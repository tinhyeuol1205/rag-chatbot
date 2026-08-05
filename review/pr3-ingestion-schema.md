# PR 3 — Ingestion, metadata, ID scheme

> ## ⚠️ PR NÀY BẮT BUỘC RE-INGEST TỪ ĐẦU
>
> Đổi ID scheme (MD5 hex → UUIDv5) làm **mọi ID cũ vô hiệu**. Phải:
> ```bash
> make clean          # xoá volume Qdrant (docker compose down -v)
> make local-start
> make ingest
> ```
> **Nếu bỏ qua:** child mới (UUID có dấu gạch) + parent cũ (hex không dấu gạch) cùng tồn
> tại → `parent_map` không match → [P0-1](pr1-correctness.md#p0-1--parentresolver-âm-thầm-xoá-documents--context-rỗng)
> kích hoạt trên diện rộng, bot trả lời "I don't have enough information" cho mọi câu hỏi.

> **Mục tiêu:** Làm citation có ý nghĩa (tên section thật thay vì "Section 7"), làm
> re-ingest idempotent thật sự, và loại bỏ collision ID.
>
> **Phụ thuộc:** [PR 1](pr1-correctness.md) (P0-1 phải xong trước — PR này xoá hack
> `.replace("-","")` trong cùng hàm đó).
> **Cần re-ingest:** ⚠️ **CÓ**.
> **Issues:** P1-2, P1-6, P2-5
>
> **File bị sửa:**
> - `src/ingestion/models.py` (UUIDv5 + `position`)
> - `src/ingestion/chunking/parent_child.py` (chunk theo từng document)
> - `src/ingestion/pipeline.py` (delete-by-file, relative path key, `is_file()`, try/except)
> - `src/core/db/qdrant.py` (`delete_by_file_name`, `create_payload_index`, `scroll_all`)
> - `src/retrieval/context/parent_resolver.py` (xoá hack normalize)
> - `src/ingestion/parsers/markdown_parser.py` (`###` + code fence)
> - `src/retrieval/search/dense.py`, `src/retrieval/search/sparse.py` (đọc `page_number`)
> - `tests/test_chunk_id.py`, `tests/test_markdown_parser.py` (mới)
>
> **Definition of Done:**
> - [ ] Sau `make clean && make ingest`, citation ra tên section thật ("Password Policy")
>       thay vì "Section 7"
> - [ ] Chạy `make ingest` **2 lần liên tiếp** → số point trong Qdrant **không tăng**
> - [ ] `test_chunk_id.py` xanh (UUID hợp lệ, không collision theo prefix)
> - [ ] Log **không** có `missing_parents`

---

## P2-5 🟡 `chunk_id` = MD5(200 ký tự đầu) → collision làm mất chunk

**Làm mục này TRƯỚC** vì P1-2 phụ thuộc field `position` được thêm ở đây.

**File:** `src/ingestion/models.py:58-63`

```python
def _generate_id(self) -> str:
    key = f"{self.content[:200]}:{self.metadata.file_name}"   # ❌ chỉ 200 ký tự đầu
    return hashlib.md5(key.encode()).hexdigest()
```

**Vấn đề:** Hai chunk khác nhau nhưng **giống nhau ở 200 ký tự đầu** → cùng `chunk_id` →
`upsert` khiến chunk sau **ghi đè** chunk trước → mất dữ liệu, không log, không lỗi.

Kịch bản thực tế: tài liệu có nhiều mục cùng mở đầu giống nhau (bảng biểu, boilerplate,
"Điều khoản áp dụng: ...", template form, hoặc PDF có header/footer lặp trên mỗi trang —
mà `PDFParser` gộp header vào đầu mỗi page-document). Với `CHILD_CHUNK_SIZE=400`, 200 ký
tự là **một nửa** chunk → xác suất collision không hề nhỏ.

Ngoài ra `chunk_id` **không mã hoá vị trí**, nên cùng một đoạn text xuất hiện 2 lần trong
1 file sẽ collapse thành 1 chunk (đôi khi mong muốn, nhưng đang xảy ra ngoài ý thức).

**Fix — hash toàn bộ content + vị trí, và sinh UUID thật:**

```python
# src/ingestion/models.py
import hashlib
import uuid

from pydantic import BaseModel

# Namespace cố định cho project — đảm bảo ID deterministic giữa các lần chạy
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class Chunk(BaseModel):
    """
    Sau khi chunking → Chunk.
    - chunk_id: UUIDv5 deterministic — dùng làm Qdrant point ID
    - parent_id: Link đến parent chunk (cho Parent-Child Retrieval)
    - position: vị trí trong file ("docIdx:parentIdx[:childIdx]") — chống collision
    - is_parent: True nếu đây là parent chunk (chunk lớn, chỉ lưu text)
    """

    chunk_id: str = ""
    content: str
    parent_id: str | None = None
    is_parent: bool = False
    position: str = ""          # ★ MỚI
    metadata: DocumentMetadata

    def model_post_init(self, __context) -> None:
        """Tự động tạo chunk_id sau khi init nếu chưa có."""
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """UUIDv5 deterministic từ (file, vị trí, TOÀN BỘ content).

        - Dùng full content → không collision do prefix giống nhau
        - Kèm position → 2 đoạn text trùng nhau ở 2 vị trí vẫn là 2 chunk
        - Trả UUID chuẩn → Qdrant nhận trực tiếp, không cần normalize dấu '-'
        - Deterministic: cùng input → cùng ID (idempotent khi re-ingest content không đổi)
        """
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        key = f"{self.metadata.source_path or self.metadata.file_name}|{self.position}|{digest}"
        return str(uuid.uuid5(_NAMESPACE, key))
```

**Sau khi đổi sang UUID chuẩn, XOÁ hack normalize ở `parent_resolver.py`:**

```python
# TRƯỚC (hack, dòng 59-65 và 72):
# ★ Normalize: bỏ dấu '-' vì Qdrant tự convert MD5 hex → UUID format
parent_map = {str(point.id).replace("-", ""): point.payload for point in parent_points}
pid = str(doc.get("parent_id", "")).replace("-", "")

# SAU (không cần hack — cả 2 phía đều là UUID canonical):
parent_map = {str(point.id): point.payload for point in parent_points}
pid = str(doc.get("parent_id") or "")
```

Xoá luôn block comment 3 dòng giải thích hack (dòng 59-61) — nó không còn đúng.

---

## P1-2 🟠 Metadata thật bị ghi đè → citation vô dụng

**File:** `src/ingestion/chunking/parent_child.py:44-71`

```python
full_text = "\n\n".join(doc.content for doc in documents)   # ❌ gộp hết → mất ranh giới
base_metadata = documents[0].metadata if documents else ...  # ❌ lấy metadata của doc ĐẦU TIÊN
...
metadata=DocumentMetadata(
    file_name=base_metadata.file_name,
    file_type=base_metadata.file_type,
    source_path=base_metadata.source_path,
    section_title=f"Section {i + 1}",     # ❌ GHI ĐÈ section title thật
    # page_number bị DROP hoàn toàn
),
```

**Triệu chứng:** `MarkdownParser` đã trích được `section_title` thật ("Annual Leave
Policy", "Password Policy" — `markdown_parser.py:32-33`), `PDFParser` đã trích
`page_number` (`pdf_parser.py:26-27`). Cả hai bị **xoá sạch** ở bước chunking. Citation
cuối cùng người dùng thấy là:

```
- company_policy.md → Section 7
```

Vô nghĩa — không ai biết "Section 7" là mục nào. Đúng ra phải là:

```
[1] company_policy.md → Password Policy
[2] handbook.pdf → p.12
```

Đây làm hỏng feature "source citation" mà README quảng cáo (dòng 16) và `SYSTEM_PROMPT`
rule 3 yêu cầu ("Always cite the source document").

**Nguyên nhân gốc:** Gộp toàn bộ file thành 1 string trước khi chunk. Comment ở dòng 45
biện minh "vì 1 PDF có nhiều pages, ta muốn chunk xuyên suốt pages" — nhưng đánh đổi này
không cần thiết: chunk **theo từng RawDocument** vẫn giữ được ngữ cảnh (mỗi RawDocument
là 1 section markdown / 1 trang PDF — đã là đơn vị ngữ nghĩa hoàn chỉnh) mà **không mất**
metadata.

**Fix — chunk theo từng document, giữ metadata gốc:**

```python
def parent_child_chunk(
    documents: list[RawDocument],
) -> tuple[list[Chunk], list[Chunk]]:
    """Tạo parent chunks và child chunks từ danh sách documents.

    Chunk theo TỪNG document (1 section MD / 1 trang PDF) để giữ nguyên
    section_title và page_number cho citation.
    """
    if not documents:
        return [], []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.PARENT_CHUNK_SIZE,
        chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHILD_CHUNK_SIZE,
        chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks: list[Chunk] = []
    child_chunks: list[Chunk] = []

    # ★ Chunk theo TỪNG document → giữ nguyên metadata của document đó
    for doc_idx, doc in enumerate(documents):
        md = doc.metadata
        for p_idx, parent_text in enumerate(parent_splitter.split_text(doc.content)):
            # section_title: ưu tiên cái parser trích được; chỉ fallback khi thiếu
            if md.section_title:
                title = md.section_title
            elif md.page_number is not None:
                title = f"Page {md.page_number}"
            else:
                title = f"Section {doc_idx + 1}"

            parent = Chunk(
                content=parent_text,
                is_parent=True,
                parent_id=None,
                position=f"{doc_idx}:{p_idx}",       # ★ chống collision (P2-5)
                metadata=DocumentMetadata(
                    file_name=md.file_name,
                    file_type=md.file_type,
                    page_number=md.page_number,      # ★ GIỮ
                    section_title=title,             # ★ GIỮ
                    source_path=md.source_path,
                ),
            )
            parent_chunks.append(parent)

            for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):
                child_chunks.append(
                    Chunk(
                        content=child_text,
                        is_parent=False,
                        parent_id=parent.chunk_id,   # ★ Link đến parent
                        position=f"{doc_idx}:{p_idx}:{c_idx}",
                        metadata=parent.metadata.model_copy(),   # thừa hưởng metadata thật
                    )
                )

    logger.info(
        "Parent-Child chunking done",
        source_docs=len(documents),
        parents=len(parent_chunks),
        children=len(child_chunks),
        avg_children_per_parent=round(len(child_chunks) / max(len(parent_chunks), 1), 1),
    )
    return parent_chunks, child_chunks
```
**Phải sửa kèm — `page_number` chưa được ghi vào payload Qdrant.**
`pipeline.py:143-179`, thêm vào **cả** `_store_children` và `_store_parents`:

```python
payload={
    "content": chunk.content,
    "parent_id": chunk.parent_id,          # ← chỉ có ở _store_children
    "file_name": chunk.metadata.file_name,
    "file_type": chunk.metadata.file_type,
    "source_path": chunk.metadata.source_path,
    "section_title": chunk.metadata.section_title,
    "page_number": chunk.metadata.page_number,   # ★ THÊM
},
```

**Và đọc nó ra ở retrieval layer:**

```python
# src/retrieval/search/dense.py — _format_results (dòng 74-87)
formatted.append({
    "chunk_id": r.id,
    "content": r.payload.get("content", ""),
    "score": r.score,
    "parent_id": r.payload.get("parent_id"),
    "file_name": r.payload.get("file_name", ""),
    "section_title": r.payload.get("section_title", ""),
    "page_number": r.payload.get("page_number"),     # ★ THÊM
    "source": "dense",
})
```

```python
# src/retrieval/search/sparse.py — _build_index (dòng 103-109)
self._documents.append({
    "chunk_id": point.id,
    "content": content,
    "parent_id": point.payload.get("parent_id"),
    "file_name": point.payload.get("file_name", ""),
    "section_title": point.payload.get("section_title", ""),
    "page_number": point.payload.get("page_number"),   # ★ THÊM
})
```

`parent_resolver.py` (PR 1) và `assembler.py` (PR 2) đã sẵn sàng đọc `page_number` —
sau PR này field đó mới có giá trị thật thay vì `None`.

**Sửa thêm `markdown_parser.py`:** regex `r"\n(?=#{1,2}\s)"` (dòng 23) chỉ split ở `#` và
`##`. File có `###` sẽ nằm chung một RawDocument rất lớn. Ngoài ra `#` trong code block
(```` ``` ````) sẽ bị split sai:

```python
def parse(self, file_path: Path) -> list[RawDocument]:
    logger.info("Parsing Markdown", file=file_path.name)
    text = file_path.read_text(encoding="utf-8")

    # Split theo header cấp 1-3, BỎ QUA các dòng # nằm trong fenced code block
    in_fence = False
    sections: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        is_header = (not in_fence) and re.match(r"#{1,3}\s", line)
        if is_header and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))

    documents = []
    for section in sections:
        content = section.strip()
        if not content:
            continue
        first_line = content.split("\n")[0]
        section_title = first_line.lstrip("#").strip() if first_line.startswith("#") else None
        documents.append(
            RawDocument(
                content=content,
                metadata=DocumentMetadata(
                    file_name=file_path.name,
                    file_type=file_path.suffix.lstrip(".").lower() or "md",  # ★ .txt ≠ md
                    section_title=section_title,
                    source_path=str(file_path),
                ),
            )
        )

    logger.info("Markdown parsed", file=file_path.name, sections=len(documents))
    return documents
```

> Lưu ý nhỏ: dispatcher map `.txt` → `MarkdownParser` (`dispatcher.py:26`) nhưng parser
> hardcode `file_type="md"`. Đoạn trên sửa luôn — `.txt` giờ có `file_type="txt"`.

---

## P1-6 🟠 Re-ingest để lại chunk rác vĩnh viễn

**File:** `src/ingestion/pipeline.py:43-80`

**Triệu chứng:** Sửa nội dung `company_policy.md` rồi chạy lại `make ingest` → chunk của
**bản cũ** vẫn còn nguyên trong Qdrant. Bot trả lời trộn lẫn thông tin cũ và mới. Kết hợp
với P0-1, các child chunk cũ trỏ tới parent đã mất → bị xoá âm thầm khỏi kết quả.

**Nguyên nhân gốc:** `chunk_id` = hash(content) → nội dung đổi thì ID đổi → `upsert` tạo
**point mới** thay vì cập nhật point cũ. Point cũ không ai xoá. Docstring `models.py:60`
viết "Deterministic: cùng content + file → cùng ID (idempotent khi re-ingest)" — đúng
với content *không đổi*, nhưng idempotency đó **không** dọn được chunk của content đã đổi.

**Fix — xoá theo `file_name` trước khi ghi lại file đó.** Thêm vào `QdrantConnector`:

```python
# src/core/db/qdrant.py
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue


def delete_by_file_name(self, collection_name: str, file_name: str) -> None:
    """Xoá mọi point của 1 file — gọi TRƯỚC khi ingest lại file đó."""
    if not self._collection_exists(collection_name):
        return
    self.client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(must=[
                FieldCondition(key="file_name", match=MatchValue(value=file_name))
            ])
        ),
        wait=True,
    )
    logger.info("Deleted existing points for file",
                collection=collection_name, file=file_name)


def create_payload_index(self, collection_name: str, field_name: str) -> None:
    """Index cho payload field — bắt buộc để filter/delete-by-filter chạy nhanh."""
    from qdrant_client.models import PayloadSchemaType
    try:
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created payload index", collection=collection_name, field=field_name)
    except Exception as e:      # index đã tồn tại → bỏ qua
        logger.debug("Payload index skipped", field=field_name, error=str(e))
```

Trong `pipeline.py`:

```python
def _init_collections(self) -> None:
    """Tạo 2 collections trong Qdrant nếu chưa tồn tại."""
    self.qdrant.create_vector_collection(settings.CHILD_COLLECTION)
    self.qdrant.create_payload_collection(settings.PARENT_COLLECTION)
    # index để delete-by-filter và metadata filtering hoạt động
    for coll in (settings.CHILD_COLLECTION, settings.PARENT_COLLECTION):
        self.qdrant.create_payload_index(coll, "file_name")


def run(self, data_dir: str) -> None:
    ...
    for file_name, documents in all_documents.items():
        logger.info("Processing file", file=file_name, raw_docs=len(documents))

        # ★ Bước 1.5: dọn chunk cũ của file này (idempotent thật sự)
        self.qdrant.delete_by_file_name(settings.CHILD_COLLECTION, file_name)
        self.qdrant.delete_by_file_name(settings.PARENT_COLLECTION, file_name)

        parent_chunks, child_chunks = parent_child_chunk(documents)
        ...
```

### ⚠️ Bug kèm theo: 2 file cùng tên ở 2 thư mục con sẽ đè nhau

`file_name` hiện chỉ là **basename** (`pipeline.py:111` dùng `file_path.name`), nhưng
`rglob("*")` quét **đệ quy** (dòng 98). Nếu có `docs/a/policy.md` và `docs/b/policy.md`,
chúng cùng key `"policy.md"` trong dict `result` → file thứ 2 **ghi đè** file thứ 1,
**mất dữ liệu ngay từ bước parse**, trước cả khi vào Qdrant.

Sửa luôn — key theo đường dẫn tương đối, và dùng nó làm `file_name`:

```python
def _parse_all_files(self, data_path: Path) -> dict[str, list[RawDocument]]:
    """Scan thư mục, parse tất cả files hỗ trợ.

    Returns:
        Dict mapping: relative_path → list[RawDocument]
    """
    supported = ParserDispatcher.supported_extensions()
    files = [f for f in data_path.rglob("*")
             if f.is_file() and f.suffix.lower() in supported]     # ★ thêm is_file()
    if not files:
        logger.warning("No supported files found", path=str(data_path), supported=supported)
        return {}

    result: dict[str, list[RawDocument]] = {}
    for file_path in sorted(files):
        rel = file_path.relative_to(data_path).as_posix()   # ★ unique key
        try:
            parser = ParserDispatcher.get_parser(file_path)
            documents = parser.parse(file_path)
        except Exception:
            # ★ 1 file hỏng KHÔNG được giết cả pipeline
            logger.exception("Failed to parse file, skipping", file=rel)
            continue
        if documents:
            result[rel] = documents
        else:
            logger.warning("Parser returned no documents", file=rel)

    logger.info("Parsed files", found=len(files), parsed=len(result))
    return result
```
**Về `is_file()`:** `rglob("*")` cũng trả về **thư mục**; một thư mục tên `notes.md/` sẽ
được đưa vào danh sách và làm parser crash. Hiện chưa bị vì sample docs phẳng, nhưng đây
là bug chờ.

**Về try/except quanh parser:** hiện tại 1 PDF hỏng làm `make ingest` chết giữa đường,
những file sau không được xử lý. Với `IngestionError`/`ParsingError` có sẵn trong
`errors.py` (đang không dùng — xem [PR 5 / P3-3](pr5-eval-deps-docs.md#p3-3--dead-code)),
đây là chỗ nên dùng chúng.

**Lưu ý:** vì `file_name` giờ là relative path, `delete_by_file_name` phải nhận cùng giá
trị. Kiểm tra `parser.parse()` set `metadata.file_name = file_path.name` (basename) —
**không khớp** với key `rel`. Chọn một trong hai:

- **(a)** Truyền `rel` vào parser để set `metadata.file_name = rel` — nhất quán nhưng phải
  sửa signature cả 3 parser.
- **(b)** Sau khi parse, ghi đè trong `pipeline.py` (đơn giản hơn, khuyến nghị):

```python
        if documents:
            for d in documents:
                d.metadata.file_name = rel      # ★ đồng bộ với key dùng để delete
            result[rel] = documents
```

---

## Test cần thêm

```python
# tests/test_chunk_id.py
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
```

```python
# tests/test_parent_child_chunking.py
from ingestion.chunking.parent_child import parent_child_chunk
from ingestion.models import DocumentMetadata, RawDocument


def test_preserves_section_title():
    """★ Bug P1-2: section_title thật KHÔNG được ghi đè thành 'Section N'."""
    docs = [
        RawDocument(content="## Password Policy\n" + "x" * 500,
                    metadata=DocumentMetadata(file_name="p.md", file_type="md",
                                              section_title="Password Policy")),
    ]
    parents, children = parent_child_chunk(docs)
    assert all(p.metadata.section_title == "Password Policy" for p in parents)
    assert all(c.metadata.section_title == "Password Policy" for c in children)


def test_preserves_page_number():
    docs = [
        RawDocument(content="y" * 500,
                    metadata=DocumentMetadata(file_name="d.pdf", file_type="pdf",
                                              page_number=12)),
    ]
    parents, children = parent_child_chunk(docs)
    assert all(p.metadata.page_number == 12 for p in parents)
    assert all(c.metadata.page_number == 12 for c in children)


def test_children_link_to_correct_parent():
    docs = [
        RawDocument(content="a" * 3000,
                    metadata=DocumentMetadata(file_name="a.md", file_type="md")),
    ]
    parents, children = parent_child_chunk(docs)
    parent_ids = {p.chunk_id for p in parents}
    assert len(parents) > 1                      # đủ dài để tách nhiều parent
    assert all(c.parent_id in parent_ids for c in children)


def test_empty_documents():
    assert parent_child_chunk([]) == ([], [])
```

```python
# tests/test_markdown_parser.py
from pathlib import Path

from ingestion.parsers.markdown_parser import MarkdownParser


def test_extracts_section_titles(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\nintro\n\n## Password Policy\nmin 12 chars\n", encoding="utf-8")
    docs = MarkdownParser().parse(f)
    titles = [d.metadata.section_title for d in docs]
    assert "Password Policy" in titles


def test_splits_h3_headers(tmp_path: Path):
    """### cũng phải tách section, không dồn vào 1 doc khổng lồ."""
    f = tmp_path / "doc.md"
    f.write_text("## A\ntext a\n\n### B\ntext b\n", encoding="utf-8")
    docs = MarkdownParser().parse(f)
    assert len(docs) == 2


def test_ignores_hash_inside_code_fence(tmp_path: Path):
    """Dòng '# comment' trong code block KHÔNG được coi là header."""
    f = tmp_path / "doc.md"
    f.write_text("## Setup\n```bash\n# install deps\npip install x\n```\ndone\n",
                 encoding="utf-8")
    docs = MarkdownParser().parse(f)
    assert len(docs) == 1
    assert "# install deps" in docs[0].content
```

---

## Kiểm chứng

```bash
make test                       # 3 file test mới xanh
make clean                      # ⚠️ BẮT BUỘC — xoá volume Qdrant
make local-start
make ingest 2>&1 | tail -20
```

**1. Kiểm tra idempotency** — chạy `make ingest` **lần thứ 2** và so số point:

```bash
curl -s localhost:6333/collections/child_chunks | python -c "import json,sys; print('child:', json.load(sys.stdin)['result']['points_count'])"
curl -s localhost:6333/collections/parent_chunks | python -c "import json,sys; print('parent:', json.load(sys.stdin)['result']['points_count'])"
make ingest > /dev/null 2>&1
# chạy lại 2 câu curl trên — số phải GIỐNG HỆT
```

**2. Kiểm tra citation có ý nghĩa** — `make run-ui`, hỏi "What is the password policy?".
Answer phải cite tên section thật:

- ✅ `[1] company_policy.md → Password Policy`
- ❌ `- company_policy.md → Section 7` (chưa fix)

**3. Kiểm tra không còn chunk rác** — sửa 1 dòng trong `data/sample_docs/company_policy.md`,
chạy lại `make ingest`, rồi kiểm tra log **không** có `missing_parents` khi query.

**4. Kiểm tra ID là UUID** — đọc trực tiếp từ Qdrant:

```bash
curl -s -X POST localhost:6333/collections/child_chunks/points/scroll \
  -H 'Content-Type: application/json' -d '{"limit":1}' \
  | python -c "import json,sys; print(json.load(sys.stdin)['result']['points'][0]['id'])"
# phải ra dạng có dấu gạch: 3f2a1b4c-5d6e-5f70-8901-2a3b4c5d6e7f
```
