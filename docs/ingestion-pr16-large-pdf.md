# Ingest một PDF lớn (PR16)

PR16 nhận trực tiếp một file PDF 1000+ trang. Operator không cần cắt PDF thành
nhiều file. Parser tự tạo page-window tạm thời, rồi xử lý lần lượt:

```text
PDF gốc → window 16 trang → fast parse → OCR trang thiếu (tối đa 4 trang)
        → chunk → embedding → Qdrant → checkpoint → window kế tiếp
```

Tên file, `source_uri`, hash của PDF gốc và số trang toàn cục vẫn được lưu trong
manifest/Qdrant. Tên file tạm không xuất hiện trong citation.

## Cấu hình khuyến nghị

```dotenv
INGEST_VERSIONED=true
INGEST_MAX_MEMORY_MB=2048
INGEST_PDF_PAGE_WINDOW=16
INGEST_PDF_OCR_PAGE_WINDOW=4
INGEST_PDF_OCR_MODE=missing_pages
INGEST_PDF_MAX_PAGES=5000
INGEST_PDF_SPOOL_MAX_MB=64
INGEST_PDF_WINDOW_TIMEOUT_SECONDS=300
```

`INGEST_MAX_MEMORY_MB` là ngưỡng fail-closed theo RSS tại boundary mỗi window;
nó không thể bảo vệ khỏi spike bên trong thư viện OCR. Nếu image/table lớn làm
RSS vượt ngưỡng, source bị đánh dấu failed và generation đang phục vụ không bị
đổi alias.

## Dev test

Đặt đúng một PDF vào thư mục fixture và chạy:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/large-pdf \
  --sync \
  --dry-run \
  --job-id large-pdf-dev-plan \
  --generation-id large-pdf-dev-plan
```

Dry-run không cần Qdrant đang chạy, nhưng vẫn chạy parser/OCR để đo coverage.
Vì vậy PDF scan vẫn cần cài Tesseract dù chưa upsert dữ liệu.

Sau khi kiểm tra summary, chạy ingest thật với Qdrant local/dev:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/large-pdf \
  --sync \
  --job-id large-pdf-dev \
  --generation-id large-pdf-dev \
  --no-resume
```

Không đặt `INGEST_PDF_PAGE_WINDOW` quá lớn để cố tăng tốc. Hãy đo RSS trên PDF
thật; tăng window chỉ khi còn headroom và OCR image size đã được kiểm thử.

## Product rollout

1. Copy nguyên file PDF vào source mount của ingestion worker; không đổi tên part.
2. Chạy dry-run và kiểm tra page limit, disk tạm, RSS limit và dung lượng GPU
   embedding/OCR.
3. Chạy full `--sync` vào generation staging mới.
4. Kiểm tra `pages_seen`, `pages_empty`, `ocr_pages`, số parent/child và sample
   citation ở trang 1, trang biên window và trang cuối.
5. Chỉ alias-switch sau khi toàn generation committed và native dense/BM25
   coverage đã được validate.
6. Giữ generation cũ trong rollback window.

Ví dụ product:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/corpus/legal-vn \
  --sync \
  --job-id legal-vn-pdf1000-2026-08 \
  --generation-id legal-vn-pdf1000-2026-08 \
  --no-resume \
  --summary-path /srv/ingest-runs/legal-vn-pdf1000.json
```

## Resume và lỗi

Nếu worker chết sau window 600, chạy lại cùng `--job-id`/`--generation-id` và
không thay đổi hash PDF hoặc fingerprint. Manifest sẽ bắt đầu từ window chưa
hoàn tất; các window đã commit chỉ được skip sau khi kiểm tra checkpoint và point
IDs trong Qdrant.

Đổi PDF, model, parser strategy hoặc page-window khi resume là không an toàn. Hãy
tạo job/generation mới. PDF mã hóa, hỏng hoặc vượt `INGEST_PDF_MAX_PAGES` phải
được xử lý ở generation staging; không ingest in-place vào alias active.

## Giới hạn hiện tại

- Mỗi PDF vẫn được xử lý tuần tự trong một worker; PR16 chưa phân tán một PDF
  trên nhiều host.
- `parse()`/`parse_with_quality()` vẫn là API compatibility và có thể
  materialize toàn bộ output. Production pipeline dùng `iter_windows()`.
- OCR chỉ fill trang thiếu theo metadata trang; element PDF nhiều trang nhưng
  thiếu page metadata sẽ fail closed để tránh citation sai.
