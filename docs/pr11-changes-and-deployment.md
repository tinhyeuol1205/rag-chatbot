# PR11 — bảng thay đổi source và hướng dẫn vận hành

Tài liệu này mô tả trạng thái source tại mốc PR11, bao gồm phần tương
thích với luồng PR10. Mục tiêu của PR11 là ingest tài liệu lớn theo từng
document/window/batch, có manifest để resume, và chuyển generation vào alias
đang được retrieval sử dụng chỉ sau khi đã kiểm tra tính toàn vẹn.

## 1. Luồng xử lý mới

~~~text
source directory
    -> discover + SHA-256 + pipeline fingerprint
    -> SQLite manifest (source/job/checkpoint)
    -> staging collections theo generation
    -> parse + quality gate
    -> chunk từng document/window
    -> embed theo batch + Qdrant upsert theo item/byte budget
    -> verify point IDs + child -> parent sample
    -> atomic switch hai alias
    -> activate manifest + dọn generation quá retention
~~~

Alias ổn định cho retrieval là "child_chunks_active" và
"parent_chunks_active". Collection vật lý của một generation có dạng
"child_chunks_active__<generation-id>" và "parent_chunks_active__<generation-id>".
Generation đang ghi không được dùng trực tiếp cho retrieval.

Sau PR12/PR14, child schema có thêm native BM25 và dense 1024d; cần full
backfill generation mới. Phần rollout queue, GPU services và migration hiện tại
được cập nhật tại [docs/pr14-deployment.md](pr14-deployment.md).

## 2. Bảng liệt kê thay đổi source

| File/module | Loại | Thay đổi chính | Tác động khi chạy |
|---|---|---|---|
| src/core/config.py | Sửa | Thêm alias active và alias legacy; thêm version pipeline/schema/parser/chunker, model revision, manifest, batch/byte/retry/retention/memory và parser-quality settings; validate giới hạn dương, retry delay và ratio. | Cấu hình ingestion phải được pin và kiểm tra ngay lúc khởi động; cấu hình sai dừng job sớm. |
| src/core/db/qdrant.py | Sửa lớn | Tạo/validate collection theo generation với schema fingerprint, dimension/distance và payload indexes; upsert giới hạn item + byte; scroll/chunk ID; delete theo filter/ID; retry chỉ lỗi transient; copy source giữa generation; switch alias atomically và idempotent; validate integrity; xóa generation. | Request lớn không dồn vào RAM/Qdrant; crash có thể resume; alias cũ vẫn phục vụ nếu staging thất bại. |
| src/ingestion/batching.py | Mới | approximate_size ước lượng JSON UTF-8 và iter_batches giới hạn đồng thời số item và byte; item vượt ngưỡng vẫn được gửi riêng. | Giữ kích thước request ổn định, tránh 413/timeout do payload lớn. |
| src/ingestion/manifest.py | Mới | SQLite WAL lưu source hash/fingerprint/status, job, batch checkpoint, active/working generation; checkpoint có point IDs; activation, removed-source và generation retention; rollback reset source để buộc revalidate. | Resume cùng job ID không embed/upsert lại batch đã xác minh; có audit trail và rollback an toàn hơn. |
| src/ingestion/pipeline.py | Sửa lớn | Giữ _run_legacy cho compatibility; mặc định chạy versioned pipeline. Discover dạng iterator, hash/fingerprint, skip file không đổi bằng copy point, quality gate, metadata normalization, document/window chunking, embed/upsert checkpoint, verify IDs, carry-forward khi non-sync, stale prune chỉ sau activation, atomic alias switch, retention và rollback. | Không switch alias khi có lỗi; lần chạy sau chỉ xử lý phần đổi; file xóa chỉ bị prune khi dùng sync và activation thành công. |
| src/ingestion/embeddings.py | Sửa | Model singleton thread-safe; tùy chọn revision bất biến; kiểm tra dimension trước khi publish model; output float32/numpy; embed_batches; retry lỗi encoder transient. | Tránh load model lặp/không đồng nhất; batch embedding có giới hạn và retry có kiểm soát. |
| src/ingestion/models.py | Sửa | Metadata thêm source_uri, document version, structural anchor, element/offset/bbox/checksum, generation và fingerprint; thêm ParseQuality; chunk ID UUIDv5 dựa trên dataset/source/position/content hash đầy đủ. | Citation/traceability tốt hơn và ID ổn định khi offset bị dịch chuyển. |
| src/ingestion/parsers/base.py | Sửa | Parser có iter_documents và parse_with_quality, giữ API parse cũ. | Pipeline có thể xử lý theo cửa sổ và áp quality gate thống nhất. |
| src/ingestion/parsers/markdown_parser.py | Sửa | Đọc theo dòng; nhận heading H1–H6; nhận fenced code bằng backtick/tilde; lưu anchor/offset; đếm quality. | Markdown lớn không cần ghép toàn bộ file trước khi chia section; không nhầm heading trong code fence. |
| src/ingestion/parsers/pdf_parser.py | Sửa lớn | Fast partition trước, OCR fallback cho page thiếu; tránh OCR trùng page; thống kê page rỗng/OCR/table/image/caption/unsupported; lưu page/bbox/anchor; kiểm tra page count tùy chọn. | PDF scan vẫn có đường fallback; chất lượng parse được ghi lại và có thể chặn activation. |
| src/ingestion/parsers/docx_parser.py | Sửa | Chia element theo INGEST_DOCUMENT_WINDOW; lưu heading/anchor/offset; đếm quality; giữ metadata cấu trúc. | DOCX dài được đưa qua pipeline theo window, giảm peak memory ở phần downstream. |
| src/ingestion/chunking/parent_child.py | Sửa | Thêm iterator parent/child theo document/window; stable structural-anchor position; sao chép metadata đầy đủ. | Chunk liên kết được với parent và giữ ID ổn định khi tài liệu chèn nội dung trước đó. |
| src/ingestion/main.py | Sửa | CLI thêm --job-id, --generation-id, --no-resume; summary có job/generation/skipped/quality; failure summary chỉ ghi loại exception đã redact. | Scheduler có thể resume và thu được artifact vận hành mà không leak lỗi nội bộ. |
| .env.example | Sửa | Khai báo toàn bộ biến PR11: alias/version, manifest, batch/byte/retry, retention/memory, quality threshold và PDF strategy. | Có template cấu hình đồng nhất giữa dev, staging và production. |
| pyproject.toml | Sửa | Khai báo trực tiếp dependency numpy cho embedding/batching. | Dependency không còn phụ thuộc transitively vào package khác. |
| uv.lock | Sửa | Lock lại dependency sau khi thêm numpy trực tiếp. | CI/deploy tái lập đúng dependency; phải chạy uv lock --check. |

## 3. File hỗ trợ, kiểm thử và tài liệu

| File | Nội dung |
|---|---|
| tests/test_ingestion_pr11.py | 7 test cho byte batching, manifest/fingerprint/checkpoint, Markdown fence/anchor, stable chunk ID, skip + rollback, carry-forward non-sync và transient retry. |
| scripts/load_ingestion_pr11.py | Harness chạy parser/embedder/Qdrant thật, ghi elapsed time, peak RSS, số file discover/process/skip/fail và throughput vào JSON. |
| docs/ingestion-pr11-runbook.md | Runbook backfill, resume, activation, rollback và failure handling. |
| docs/ingestion-pr11-load-test.md | Mẫu báo cáo load test và tiêu chí thử nghiệm corpus tương đương production. |
| Makefile | Thêm target ingest-load; install-dev, local-start, local-stop, check dùng cho vòng đời dev. |
| README.md | Cập nhật alias/generation/manifest, dry-run, resume, rollback, migration và limitation BM25/scale. |

## 4. Hướng dẫn dev/test

### Chuẩn bị môi trường

1. Tạo môi trường và dependency lock:

   ~~~bash
   cp .env.example .env
   make install-dev
   ~~~

2. Trong .env, dùng namespace riêng cho developer; không dùng manifest hoặc
   dataset của production:

   ~~~dotenv
   INGEST_DATASET_ID=dev_<your-name>
   INGEST_MANIFEST_PATH=/tmp/rag-chatbot-<your-name>-manifest.sqlite3
   INGEST_VERSIONED=true
   INGEST_FAIL_ON_QUALITY=true
   ~~~

   Cần điền credential LLM hợp lệ (hoặc OPENAI_BASE_URL của server nội bộ)
   vì settings validate provider lúc import. Không commit .env hay secret.

3. Khởi động Qdrant local:

   ~~~bash
   make local-start
   curl --fail http://127.0.0.1:6333/healthz
   ~~~

### Quality gate và unit test

Chạy toàn bộ gate trước khi mở PR hoặc đóng image:

~~~bash
make check
UV_CACHE_DIR=/tmp/rag-chatbot-uv-review-cache uv lock --check
UV_CACHE_DIR=/tmp/rag-chatbot-uv-review-cache uv pip check --python rag/bin/python
PYTHONPATH=src rag/bin/python -m compileall -q src tests scripts
git diff --check
~~~

make check hiện bao gồm test suite và Ruff. Warning từ Starlette/httpx hoặc
payload index của Qdrant in-memory không phải test failure; payload index phải
được kiểm tra lại trên Qdrant server thật.

### Chạy ingestion local

Dry-run để xem kế hoạch mà không mutate Qdrant:

~~~bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/sample_docs --sync --dry-run \
  --job-id dev-plan --summary-path /tmp/dev-plan.json
~~~

Chạy thật:

~~~bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/sample_docs --sync --job-id dev-run \
  --summary-path /tmp/dev-run.json
~~~

Nếu process bị dừng, chạy lại đúng --job-id dev-run để resume checkpoint.
Dùng --no-resume chỉ khi muốn từ chối job cũ và tạo một job/generation khác.
Kiểm tra /tmp/dev-run.json, alias trong Qdrant và thử truy vấn API trước khi
coi test thành công.

Sau khi ingest ở process khác với API, restart API để rebuild BM25 index. Dừng
Qdrant khi kết thúc phiên:

~~~bash
make local-stop
~~~

Không dùng make clean trừ khi cố ý xóa volume Qdrant local và cache. Lệnh đó
không thể dùng để "reset nhẹ" một test đang cần giữ generation rollback.

## 5. Hướng dẫn deploy/backfill production

### Điều kiện bắt buộc trước khi chạy

- Dùng Qdrant server/durable volume, không dùng client in-memory. Qdrant phải
  có đủ disk, RAM và payload indexes cho dataset_id, file_name, source_uri và
  generation_id.
- Đặt INGEST_MANIFEST_PATH trên volume bền vững và dùng cùng file khi resume.
  SQLite manifest không cung cấp distributed lock; chỉ nên có một ingestion
  writer cho mỗi dataset tại một thời điểm.
- Pin EMBEDDING_MODEL_ID, INGEST_EMBEDDING_MODEL_REVISION, EMBEDDING_SIZE,
  parser/chunker/pipeline/schema versions. Đổi model, dimension hoặc format
  phải tạo backfill có chủ đích.
- Dùng INGEST_DATASET_ID riêng cho từng corpus. Publish file nguồn bằng temp
  file rồi rename atomically; không ingest trong lúc file đang được ghi.
- Chạy load test trên phần cứng/corpus đại diện. INGEST_MAX_MEMORY_MB là
  target/acceptance budget, không phải kill switch của process.
- Legacy child_chunks/parent_chunks không tự migrate. Cần full backfill vào
  generation mới trước khi traffic chuyển sang alias active.

### Backfill có kiểm soát

1. Đặt .env production, backup manifest và lưu config đã dùng cùng artifact
   deploy.
2. Chạy dry-run và review danh sách source lỗi/stale:

   ~~~bash
   PYTHONPATH=src rag/bin/python -m ingestion.main \
     --data-dir /srv/corpus --sync --dry-run \
     --job-id backfill-<yyyymmdd> \
     --summary-path /srv/ingest-runs/backfill-plan.json
   ~~~

3. Chạy backfill vào staging generation:

   ~~~bash
   PYTHONPATH=src rag/bin/python -m ingestion.main \
     --data-dir /srv/corpus --sync \
     --job-id backfill-<yyyymmdd> \
     --summary-path /srv/ingest-runs/backfill-<yyyymmdd>.json
   ~~~

   Không tự thao tác xóa hoặc đổi alias bằng tay. Pipeline chỉ switch alias
   sau khi schema, dimension/distance, payload indexes, sampled
   child-to-parent references và quality gate đều đạt; có source fail thì
   generation không được activate.

4. Nếu job fail giữa chừng, giữ nguyên generation để điều tra và chạy lại cùng
   job ID. Checkpoint chỉ được bỏ qua sau khi point IDs đã được đọc lại từ
   Qdrant, nên crash ngay sau write không gây embed trùng.
5. Sau activation, kiểm tra:

   - summary JSON không có failed_files;
   - hai alias active cùng trỏ về cùng generation;
   - /health và truy vấn đại diện trả citation đúng dataset;
   - log không có retry kéo dài hoặc quality ratio bất thường;
   - restart worker/API để BM25 nạp generation mới nếu ingestion chạy ngoài
     process API.

--allow-empty-source chỉ dùng sau khi xác minh mount. Nếu không, sync an toàn
sẽ từ chối source rỗng thay vì prune toàn bộ dataset.

### Rollback

Chọn generation đã commit còn trong retention, sau đó:

~~~bash
PYTHONPATH=src rag/bin/python - <<'PY'
from ingestion.pipeline import IngestionPipeline
IngestionPipeline().rollback("<generation-id>")
PY
~~~

Kiểm tra alias, /health và một mẫu retrieval sau rollback. Rollback cũng reset
trạng thái/hash của source trong manifest để lần ingest kế tiếp revalidate và
không tin mù vào kết quả của generation vừa bị loại. Không xóa collection đang
được một trong hai alias trỏ tới.

Generation cũ đã commit sẽ được dọn tự động khi vượt
INGEST_GENERATION_RETENTION; generation lỗi được giữ lại để điều tra. Trước
khi giảm retention hoặc xóa thủ công, lưu summary và xác nhận không còn
rollback window cần thiết.

## 6. Load test và tiêu chí chấp nhận

Chạy trên Qdrant thật, dataset/manifest riêng:

~~~bash
PYTHONPATH=src rag/bin/python scripts/load_ingestion_pr11.py \
  /srv/corpus-representative \
  --job-id load-<yyyymmdd> \
  --output /srv/ingest-runs/load-<yyyymmdd>.json
~~~

Hoặc:

~~~bash
make ingest-load DATA_DIR=/srv/corpus-representative
~~~

Lưu cùng artifact: số file/byte/chunk parent-child, elapsed và throughput,
peak RSS, request byte lớn nhất, retry/error rate, thời gian switch alias và
kết quả failure injection sau một write batch. Tối thiểu phải thử corpus mục
tiêu khoảng 1M chunk hoặc kích thước tương đương production, dừng process sau
một write thành công, rồi resume cùng job ID và xác nhận không embed lại batch
đã commit. Harness không tự sinh corpus 1M và không đo mọi trường nêu trên;
các trường còn lại phải lấy từ Qdrant/manifest/monitoring.

## 7. Lưu ý giới hạn và xử lý sự cố

| Tình huống | Cách xử lý |
|---|---|
| Alias vẫn trỏ generation cũ | Xem failed_files, quality và schema trong summary/manifest; generation fail không được activate là hành vi an toàn. |
| Resume báo checkpoint không hợp lệ | Không xóa manifest ngay; kiểm tra collection/generation có bị xóa hoặc đổi schema không, sau đó dùng job ID mới cho backfill có chủ đích. |
| Retry tăng liên tục | Kiểm tra Qdrant timeout, HTTP 408/425/429/5xx, network và request byte; lỗi validation/4xx không transient sẽ không được retry. |
| Quality gate fail | Mở counters PDF/Markdown/DOCX, sửa nguồn hoặc điều chỉnh threshold có review; không tắt INGEST_FAIL_ON_QUALITY trong production để che lỗi. |
| Peak memory vẫn cao | Downstream đã bounded theo file/window/batch nhưng thư viện unstructured có thể materialize element list cho một PDF/DOCX rất lớn; tách file hoặc đặt giới hạn worker. |
| Bảng/lists phức tạp bị chia chưa đẹp | Chunker hiện vẫn chủ yếu theo ký tự; cần PR tiếp theo cho token/structure-aware chunking nếu corpus phụ thuộc mạnh vào bảng. |
| Retrieval không thấy dữ liệu mới | Kiểm tra alias rồi restart API để rebuild BM25; dense Qdrant và BM25 có lifecycle cache khác nhau. |
| Chạy test local làm mất dữ liệu | make clean xóa Qdrant volume; dùng dataset/manifest riêng và make local-stop để chỉ dừng service. |

PR11 chưa có distributed job lock và chưa chứng minh tuyệt đối trên corpus
1M trong môi trường này. Hai điểm đó phải được kiểm tra ở staging trước khi
cho phép scheduler chạy song song hoặc công bố SLO production.
