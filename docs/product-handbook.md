# Product Handbook — RAG Chatbot

Tài liệu này hướng dẫn vận hành hệ thống từ lúc checkout source, chạy local,
đưa lên production, ingest dữ liệu mới và xử lý các tình huống thường gặp trong
sản phẩm thật.

## 1. Phạm vi và kiến trúc

```text
Client
  -> FastAPI
  -> Redis admission queue
  -> RAG worker
  -> BGE-M3 embedding GPU
  -> Qdrant dense + native BM25 hybrid
  -> BGE-Reranker-v2-m3 GPU
  -> LLM provider
```

- FastAPI production chỉ enqueue và đọc kết quả; không chạy RAG trực tiếp.
- Redis Streams giới hạn số job outstanding và pacing quota LLM.
- Worker chỉ bắt đầu embedding/search/reranking sau khi đã claim job và reserve
  quota.
- Qdrant lưu dense vector 1024 chiều và sparse vector tên `bm25`.
- Dense và sparse prefetch luôn dùng cùng `RetrievalScope`.
- Ingest tạo generation mới rồi switch child/parent aliases atomically.

Các model production mặc định:

| Thành phần | Model | Chạy production |
|---|---|---|
| Embedding | `BAAI/bge-m3`, 1024d | GPU HTTP service |
| Reranker | `BAAI/bge-reranker-v2-m3` | GPU HTTP service |
| Sparse | Qdrant native `Qdrant/bm25` | Qdrant |

## 2. Chọn đúng runbook

| Mục tiêu | Dùng cấu hình | Đọc các phần |
|---|---|---|
| Dev/local, test tính năng, sample documents | `APP_ENV=development`, `RAG_EXECUTION_MODE=inline` | 3 và 4 |
| Product/staging/production, nhiều replica | `APP_ENV=production`, `RAG_EXECUTION_MODE=redis_worker` | 5 đến 13 |

Không dùng `inline` hoặc model local để phục vụ product. Không dùng lệnh clean
volume local trên Redis/Qdrant production.

## 3. DEV — chạy từ đầu

### 3.1 Yêu cầu

- Python `>=3.10,<3.13`.
- `uv`.
- Docker/Compose cho local Redis và Qdrant.
- API key của OpenAI-compatible provider hoặc Gemini.
- Production: Redis có auth/TLS, Qdrant private network, embedding và reranker
  GPU service.

### 3.2 Checkout và cài dependency

```bash
git clone <repo-url>
cd rag-chatbot
cp .env.example .env
make install-dev       # local/dev
# make install         # production image/host
```

`uv.lock` là nguồn dependency chuẩn. CI và production phải dùng
`uv sync --locked`, không tự ý chạy `uv add` trên server.

### 3.3 Cấu hình local tối thiểu

Trong `.env`:

```dotenv
APP_ENV=development
RAG_EXECUTION_MODE=inline

LLM_PROVIDER=openai
OPENAI_API_KEY=<dev-key>
OPENAI_MODEL_ID=gpt-4o-mini

EMBEDDING_RUNTIME=local
EMBEDDING_MODEL_ID=BAAI/bge-m3
EMBEDDING_SIZE=1024
EMBEDDING_DEVICE=cpu       # dùng cuda nếu máy có GPU

RERANKER_RUNTIME=local
RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3

INGEST_DATASET_ID=sample_docs
INGEST_VERSIONED=true
```

`API_KEY` có thể để trống khi chỉ bind localhost. Không expose API ra Internet
khi `API_KEY` trống.

### 3.4 Khởi động dependency local

```bash
make local-start
docker compose ps
```

Kiểm tra:

```bash
curl -f http://127.0.0.1:6333/collections
redis-cli -h 127.0.0.1 ping
```

Nếu đổi volume hoặc xoá dữ liệu local, nhớ rằng Qdrant và Redis được lưu trong
Docker volumes. Không chạy lệnh xoá volume trên môi trường có dữ liệu thật.

### 3.5 Tiếp tục sang ingest

Sau `make local-start`, chưa cần mở API. Hãy chạy mục 4 để tạo generation và
switch aliases trước; API/UI chỉ nên khởi động sau khi ingest thành công.

## 4. DEV — ingest lần đầu

### 4.1 Chuẩn bị thư mục dữ liệu

Pipeline quét đệ quy các file được hỗ trợ, dùng đường dẫn tương đối làm
`source_uri` và `file_name`.

Ví dụ:

```text
data/real-docs/
├── policies/security/password-policy.pdf
├── policies/hr/leave.md
└── products/tc-456/manual.docx
```

Đặt `INGEST_DATASET_ID` là namespace server-side. Không lấy dataset ID trực tiếp
từ request của người dùng.

### 4.2 Xem trước kế hoạch

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/real-docs \
  --sync \
  --dry-run \
  --summary-path /tmp/ingest-plan.json
```

Kiểm tra số file discovered, file lỗi và file sẽ bị prune trong summary JSON.

### 4.3 Chạy ingest

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/real-docs \
  --sync \
  --job-id initial-real-docs-2026-08 \
  --generation-id pr14-real-docs-2026-08 \
  --no-resume \
  --summary-path data/ingest_runs/initial-real-docs.json
```

Pipeline sẽ:

1. Parse PDF/DOCX/Markdown.
2. Tạo parent/child chunks.
3. Embed child bằng BGE-M3.
4. Tạo native sparse BM25 cho child.
5. Upsert vào collection staging của generation.
6. Validate schema, sparse coverage và child-parent references.
7. Switch `child_chunks_active` và `parent_chunks_active`.

Không ghi trực tiếp vào collection đang được alias active.

### 4.4 Kiểm tra sau ingest

```bash
cat data/ingest_runs/initial-real-docs.json
curl http://127.0.0.1:8080/ready
```

Cần xác nhận:

- `failed_files` rỗng.
- Dense dimension là `1024`.
- Child points có sparse vector `bm25`.
- Parent references tồn tại.
- Alias child và parent cùng trỏ tới generation mới.
- `dataset_id` đúng namespace mong muốn.

### 4.5 Chạy API/UI sau ingest

```bash
make run-api       # FastAPI: http://127.0.0.1:8080
make run-ui UI_DEMO_MODE=dev  # thin Gradio UI → FastAPI
```

Trong inline mode, request chạy trực tiếp trong API process. Chỉ dùng cho dev
và test đơn máy.

### 4.6 DEV — clean toàn bộ dữ liệu cũ

Phần này chỉ dành cho local/dev. Lệnh `make clean` chạy
`docker compose down -v`, tức xóa toàn bộ Docker volumes của Compose project,
bao gồm Qdrant và Redis. Đây là thao tác không thể khôi phục nếu chưa backup.

Trước khi clean:

```bash
git status --short
grep -E '^(APP_ENV|RAG_EXECUTION_MODE|QDRANT_HOST|REDIS_URL)=' .env
docker compose ps
```

Chỉ tiếp tục khi chắc chắn đang ở repository và Docker project local:

```bash
make clean
rm -f data/ingest_runs/manifest.sqlite3
rm -f data/ingest_runs/*.json
```

Nếu gặp lỗi `failed to connect to the docker API` hoặc `permission denied` thì
`make clean` đã dừng ở bước Docker; Qdrant/Redis volumes chưa được xóa và hai
lệnh `rm` phía sau cũng chưa chạy. Khởi động Docker Desktop, chọn đúng context,
rồi xác nhận trước khi chạy lại:

```bash
open -a Docker
docker context use desktop-linux
docker info
make clean
```

Nếu chỉ cần xóa manifest và summary trên host trong lúc Docker đang tắt, có thể
chạy riêng:

```bash
rm -f data/ingest_runs/manifest.sqlite3
rm -f data/ingest_runs/*.json
```

Lệnh này không xóa dữ liệu Qdrant/Redis trong Docker volumes; phải chạy lại
`make clean` sau khi Docker daemon hoạt động.

`make clean` không xóa source mẫu trong `data/sample_docs/`. Nếu muốn giữ source
để ingest lại, không xóa thư mục này. Sau khi clean, khởi tạo lại từ đầu:

```bash
make local-start
make ingest
```

Không chạy `make clean`, `docker compose down -v` hoặc `redis-cli FLUSHALL` trên
staging/production.

## 5. PRODUCT — production runbook

### 5.1 Điều kiện trước khi nhận traffic

- Đã cài dependency bằng `make install`/`uv sync --locked`.
- Redis dùng auth/TLS và key namespace riêng.
- Qdrant, embedding GPU và reranker GPU nằm trong private network.
- Đã full backfill generation mới và validate trước alias switch.
- `APP_ENV=production` và model revisions là immutable.

### 5.2 Cấu hình production

Production `.env` tối thiểu:

```dotenv
APP_ENV=production
RAG_EXECUTION_MODE=redis_worker

REDIS_URL=rediss://:<password>@redis.example.com:6380/0
REDIS_KEY_PREFIX=rag:production
REDIS_QUEUE_GROUP=rag-workers
RAG_QUEUE_MAX_OUTSTANDING=32
RAG_WORKER_CONCURRENCY=2

EMBEDDING_RUNTIME=remote
EMBEDDING_MODEL_ID=BAAI/bge-m3
EMBEDDING_SIZE=1024
EMBEDDING_MODEL_REVISION=<immutable-bge-m3-revision>
EMBEDDING_BASE_URL=http://embedding-gpu:8080

RERANKER_RUNTIME=remote
RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3
RERANKER_MODEL_REVISION=<immutable-reranker-revision>
RERANKER_BASE_URL=http://reranker-gpu:8081

LLM_RATE_LIMIT_CALLS=15
LLM_RATE_LIMIT_WINDOW_SECONDS=60
```

Khởi động theo thứ tự:

```bash
RAG_EXECUTION_MODE=redis_worker make run-worker
make run-api API_HOST=0.0.0.0 API_RELOAD=false
```

Chạy nhiều worker replicas với cùng `REDIS_QUEUE_GROUP`. Mỗi replica vẫn phải
dùng cùng model revision, Qdrant aliases, Redis namespace và LLM quota.

### 5.3 Khởi động worker, API và readiness

- `/health`: process còn sống, không đảm bảo dependency đã sẵn sàng.
- `/ready`: kiểm tra Redis, Qdrant alias/schema và remote GPU services.

```bash
curl -i http://127.0.0.1:8080/health
curl -i http://127.0.0.1:8080/ready
```

Load balancer chỉ nên route traffic vào instance trả `200` từ `/ready`.

### 5.4 PRODUCT — chạy thin UI cho internal demo

UI product không đọc Redis/Qdrant và không import retriever. Nó gọi `/ready` trước
khi listen; nếu API không báo `redis_worker`, remote GPU runtimes và heartbeat
worker thì UI dừng ngay.

```bash
make run-ui \
  UI_DEMO_MODE=product \
  UI_API_BASE_URL=http://127.0.0.1:8080
```

Đặt `UI_API_KEY` bằng service key được phép gọi `API_KEY` của FastAPI. Nếu cần
đăng nhập cơ bản cho trusted network, đặt `UI_AUTH_USERNAME` và
`UI_AUTH_PASSWORD`. `UI_PUBLIC_SHARE` mặc định `false`; bật public share không
biến demo thành hệ thống có user identity, tenant ACL hoặc quota theo user.

Profile product dùng Redis per-job event stream nên token được relay incremental:
`status* → token* → sources → end`. Khi queue/quota đầy, UI nhận 429/503 cùng
`Retry-After`; request bị từ chối trước embedding/search/reranking/LLM.

## 6. PRODUCT — API smoke test

### 6.1 Non-streaming chat

```bash
curl -X POST http://127.0.0.1:8080/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <api-key>' \
  -H 'Idempotency-Key: ticket-2026-001' \
  -d '{"query":"Chính sách hoàn tiền là gì?"}'
```

Gửi lại cùng `Idempotency-Key` sẽ không tạo job RAG trùng.

### 6.2 SSE

```bash
curl -N -X POST http://127.0.0.1:8080/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"Tôi cần nghỉ phép bao nhiêu ngày?"}'
```

Trong product mode, worker ghi từng token vào Redis event stream và FastAPI relay
ngay qua SSE. Event kết thúc theo thứ tự `status* → token* → sources → end`.
Client reconnect có thể dùng event `id`; stream có TTL hữu hạn. TTFT vẫn bao gồm
thời gian retrieval/reranking trước token đầu tiên, không nên coi là cam kết dưới
mọi tải.

## 7. PRODUCT — ingest dữ liệu mới

### 7.1 Thêm file mới

Đặt file vào đúng dataset directory, sau đó chạy non-sync để giữ các file cũ:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/datasets/company-a \
  --job-id company-a-add-2026-08-17 \
  --generation-id company-a-2026-08-17 \
  --summary-path /var/lib/rag/ingest/add-2026-08-17.json
```

Không có `--sync` nghĩa là file không xuất hiện trong lần chạy này được carry
forward từ generation active. Đây là lựa chọn phù hợp cho add-only ingestion.

### 7.2 Cập nhật file

Giữ nguyên `source_uri`, thay nội dung file rồi chạy ingest. Content hash thay
đổi sẽ khiến file được parse/chunk/embed lại. Không resume một generation đã
commit; dùng `job-id` và `generation-id` mới.

### 7.3 Xóa file

Dùng `--sync` với toàn bộ thư mục nguồn đầy đủ:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/datasets/company-a \
  --sync \
  --job-id company-a-sync-2026-08-17 \
  --generation-id company-a-2026-08-17-sync
```

File đã biến mất khỏi source sẽ được prune khỏi generation mới. Nếu có file lỗi,
pipeline không xóa stale files và không switch alias để tránh phục vụ snapshot
không đầy đủ.

### 7.4 Xóa toàn bộ dataset có chủ ý

`--sync` từ source directory rỗng bị từ chối mặc định:

```text
Refusing destructive sync from an empty source
```

Chỉ dùng khi đã xác minh mount và thực sự muốn dataset rỗng:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/datasets/company-a \
  --sync \
  --allow-empty-source \
  --job-id company-a-empty-2026-08-17 \
  --generation-id company-a-empty-2026-08-17
```

### 7.5 File lỗi hoặc job bị ngắt

Summary JSON là nguồn điều tra đầu tiên:

```bash
jq '{job_id,generation_id,failed_files,processed_files,skipped_files}' \
  /var/lib/rag/ingest/latest.json
```

Nếu job chưa commit và muốn resume đúng generation:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/datasets/company-a \
  --sync \
  --job-id company-a-sync-2026-08-17
```

Không resume nếu source content, model revision hoặc schema đã đổi. Khi đó tạo
job/generation mới và chạy full backfill.

### 7.6 PRODUCT — clean toàn bộ dữ liệu cũ

Có hai mức độ cần phân biệt:

1. **Logical removal:** tài liệu không còn được phục vụ qua alias.
2. **Physical erasure:** xóa concrete Qdrant collections, manifest và object
   storage theo chính sách retention/legal hold.

Không dùng `make clean`, `docker compose down -v`, `redis-cli FLUSHALL` hoặc
`DELETE /collections/*` trên production/shared environment.

Quy trình xóa logical một dataset:

1. Tạm dừng upload và scheduler của dataset.
2. Chờ hoặc cancel job Redis đang pending; không để ingest khác chạy song song.
3. Backup manifest, Qdrant snapshot và source object metadata.
4. Chạy `--sync --allow-empty-source` để tạo empty generation cho dataset.
5. Kiểm tra alias, `/ready` và xác nhận query trong scope trả về không có context.
6. Giữ generation cũ trong rollback window trước khi physical erase.

Lệnh mẫu:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/datasets/company-a-empty \
  --sync \
  --allow-empty-source \
  --job-id company-a-delete-2026-08-17 \
  --generation-id company-a-empty-2026-08-17 \
  --summary-path /var/lib/rag/ingest/company-a-delete.json
```

Với multi-tenant, phải thực hiện theo từng `dataset_id`. Xóa một dataset không
đồng nghĩa với xóa dữ liệu tenant khác trong cùng collection.

Physical erasure chỉ thực hiện bằng runbook/admin script có allowlist concrete
generation, sau khi kiểm tra generation đó không còn là child hoặc parent alias
active và hết thời gian retention. Không xóa collection theo wildcard. Redis
queue cũng chỉ được purge theo namespace sau khi worker đã dừng và không còn job
đang chạy; không dùng `FLUSHALL`.

Nếu cần reset toàn bộ một product environment, phải coi đó là thao tác
disaster-recovery: freeze traffic, backup/verify restore, scale API/worker về 0,
destroy đúng Qdrant/Redis volumes của environment đó, rồi provision lại. Thao
tác này cần approval riêng và không được thực hiện trên shared production.

## 8. PRODUCT — lịch ingest và workflow sản phẩm

Một workflow upload tài liệu nên là:

```text
Upload -> malware/type check -> lưu immutable object
       -> gán dataset_id từ tenant/server policy
       -> dry-run ingest
       -> staging generation
       -> validate/evaluate
       -> switch alias
       -> cập nhật trạng thái tài liệu là searchable
```

### Gợi ý job scheduler

- Dùng một lock cho mỗi `dataset_id` để tránh hai ingest cùng lúc.
- Dùng `--job-id` có mã dataset + timestamp.
- Lưu summary JSON và log vào persistent volume.
- Không cho phép upload request tự ý chọn collection hoặc `dataset_id`.
- Chỉ trả trạng thái `searchable` sau khi alias switch thành công.

## 9. PRODUCT — các trường hợp thực tế

### 9.1 Multi-tenant

Mỗi tenant nên có namespace riêng:

```text
tenant-a -> dataset_id=tenant_a
tenant-b -> dataset_id=tenant_b
```

`RetrievalScope` phải được tạo ở server boundary và truyền vào dense search,
sparse search và parent lookup. Không lấy `dataset_id` từ query body của client.

Qdrant native BM25 tính IDF theo collection. Nếu tenant cần chất lượng BM25
hoàn toàn độc lập hoặc isolation mạnh hơn, cân nhắc collection/shard riêng theo
tenant thay vì chỉ filter payload.

### 9.2 Chính sách nội bộ cập nhật hàng ngày

- Dùng `--sync` từ một snapshot directory đầy đủ.
- Không switch alias nếu có file parse lỗi.
- Giữ tối thiểu `INGEST_GENERATION_RETENTION=2` generation.
- Smoke test các câu hỏi quan trọng sau switch.

### 9.3 Tài liệu lớn hoặc PDF OCR

- Kiểm tra `INGEST_MAX_MEMORY_MB`, disk tạm và RSS trước khi chạy batch lớn.
- Tách file quá lớn theo page/section ở upstream nếu parser/OCR vượt memory.
- Chạy ingest ngoài giờ cao điểm và giới hạn số job đồng thời.
- Không dùng cùng một generation cho hai lần ingest có nội dung khác nhau.

### 9.4 Mã sản phẩm, acronym và bảng

Hybrid search phù hợp vì:

- Dense/BGE-M3 xử lý ngữ nghĩa và paraphrase.
- Native BM25 giữ tín hiệu exact keyword như `TC-456`, acronym và tên cột bảng.
- Reranker chọn lại các child chunks trước khi lấy parent context.

Các loại query này phải nằm trong bộ evaluation trước mỗi model/index rollout.

### 9.5 LLM quota và overload

Với:

```dotenv
RAG_QUEUE_MAX_OUTSTANDING=32
LLM_RATE_LIMIT_CALLS=15
LLM_RATE_LIMIT_WINDOW_SECONDS=60
```

- Job thứ 33 khi queue đầy nhận `429` và `Retry-After`.
- Redis lỗi nhận `503`; production không fallback inline.
- Job chờ quá lâu nhận `504` và được cancel/request cancel.
- Không retry mù request generation; client nên backoff theo `Retry-After`.

## 10. PRODUCT — thay đổi model hoặc schema

Không đổi model/dimension tại chỗ.

Quy trình bắt buộc:

1. Pin model revision mới.
2. Tăng pipeline/schema version nếu schema thay đổi.
3. Tạo generation staging mới.
4. Full re-ingest toàn bộ corpus.
5. Chạy evaluation Việt/Anh và shadow ranking.
6. Switch alias.
7. Theo dõi error/latency/quality trong rollback window.

Đổi embedding revision làm thay đổi toàn bộ dense vectors và bắt buộc backfill.
Đổi reranker revision không cần tạo lại dense vectors nhưng vẫn phải benchmark
và deploy đồng bộ GPU service.

Rollback phải dùng application config/model tương thích với generation cũ. Không
được trỏ API đang dùng BGE-M3 1024 chiều vào generation cũ có dimension khác.

## 11. Chẩn đoán lỗi nhanh

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `/health` 200 nhưng `/ready` 503 | Redis/Qdrant/GPU chưa sẵn sàng | Kiểm tra từng dependency và alias |
| `queue_full` | Redis outstanding đạt giới hạn | Chờ `Retry-After`, scale worker hoặc tăng queue có kiểm soát |
| `QueueUnavailableError` | Redis timeout/auth/TLS | Kiểm tra URL, network, credentials |
| `Sparse vector bm25 is not found` | Alias trỏ generation cũ hoặc ingest thiếu sparse | Không switch traffic; kiểm tra schema và coverage |
| `Embedding output has unexpected shape` | GPU service/model dimension sai | Đảm bảo BGE-M3 trả đúng 1024 phần tử |
| `Reranker service unavailable` | GPU endpoint/health lỗi | Kiểm tra `/health`, DNS và timeout |
| `Refusing destructive sync from an empty source` | Sai mount/path hoặc source rỗng | Xác minh path; chỉ thêm `--allow-empty-source` khi có chủ ý |
| `Cannot carry a source across ingestion schemas` | Đang copy dữ liệu từ schema/model cũ | Chạy full backfill với toàn bộ corpus |
| generation đã committed | Generation immutable | Tạo `job-id`/`generation-id` mới |
| Kết quả chứa tài liệu tenant khác | Scope tạo sai ở server | Kiểm tra `RetrievalScope`, không nhận scope từ client |

## 12. Bảo mật và dữ liệu nhạy cảm

- Dùng Redis TLS/auth trong production và namespace key riêng cho từng môi trường.
- Qdrant, Redis và GPU services không public Internet.
- Dùng API key hoặc gateway authentication.
- Không ghi raw query, tài liệu hoặc prompt nhạy cảm vào log/evaluation artifact.
- Backup manifest SQLite, Qdrant snapshots và object storage theo chính sách dữ liệu.
- Khi xóa tenant, xóa/retire toàn bộ generation tương ứng và xác nhận alias không
  còn trỏ vào generation đó.

## 13. Checklist release

```text
[ ] `make install` với uv.lock
[ ] Model revision đã pin và GPU `/health` 2xx
[ ] Redis TLS/auth và key namespace đúng môi trường
[ ] Qdrant schema dense 1024 + sparse bm25/IDF
[ ] Full backfill staging hoàn tất
[ ] failed_files rỗng
[ ] sparse coverage và parent references đạt
[ ] Evaluation Việt/Anh đạt ngưỡng
[ ] Alias switch thành công
[ ] Worker chạy cùng REDIS_QUEUE_GROUP
[ ] `/ready` 200
[ ] `/chat` smoke test trong đúng dataset scope
[ ] Queue/LLM/GPU/Qdrant metrics đang được theo dõi
[ ] Generation cũ còn trong rollback window
```

Các lệnh kiểm tra local:

```bash
make check
RUN_QDRANT_INTEGRATION=1 \
  PYTHONPATH=src rag/bin/python -m pytest \
  tests/test_qdrant_pr12_integration.py -q
```
