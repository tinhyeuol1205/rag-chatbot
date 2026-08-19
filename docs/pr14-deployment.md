# PR14 — triển khai Redis admission và multilingual GPU inference

PR14 giữ đúng hai hợp đồng retrieval của PR12: Qdrant native BM25 + dense
hybrid (RRF) và `RetrievalScope` được gắn vào query chính lẫn cả hai prefetch.
Các `RequestDeadline`, stage semaphore, fan-out cap và token-aware context
budget của PR12 đã được hoàn nguyên. Context vẫn dùng `MAX_CONTEXT_CHARS` của
baseline; `LLM_MAX_OUTPUT_TOKENS` chỉ là giới hạn output/cost của provider.

Runbook tách riêng luồng dev/product, ingest dữ liệu mới và clean dữ liệu cũ xem
tại [docs/product-handbook.md](product-handbook.md).

## Kiến trúc production

```text
Client -> FastAPI (auth/validate) -> Redis Stream (bounded outstanding jobs)
                                      |
                               RAG worker(s)
                                      |
                  atomic LLM reservation: 15 calls / 60 seconds
                                      |
                  BGE-M3 GPU -> Qdrant dense+native BM25 ->
                  BGE-Reranker-v2-m3 GPU -> LLM provider
```

API không chạy embedding, search, reranking hoặc LLM trước khi job được worker
claim và reservation bắt đầu. Redis là admission boundary dùng chung giữa các
API replicas; production không fallback sang inline khi Redis lỗi.

## Cấu hình

1. Copy `.env.example` thành `.env` và đặt `APP_ENV=production`,
   `RAG_EXECUTION_MODE=redis_worker`.
2. Đặt `EMBEDDING_RUNTIME=remote`, `EMBEDDING_MODEL_ID=BAAI/bge-m3`,
   `EMBEDDING_SIZE=1024`, `EMBEDDING_MODEL_REVISION` là revision bất biến và
   `EMBEDDING_BASE_URL` tới service GPU.
3. Đặt `RERANKER_RUNTIME=remote`, model
   `BAAI/bge-reranker-v2-m3`, revision bất biến và `RERANKER_BASE_URL`.
4. `REDIS_URL` phải dùng endpoint có auth/TLS trong production; đổi toàn bộ
   Redis key names khi chạy nhiều môi trường. `LLM_RATE_LIMIT_CALLS=15` và
   `LLM_RATE_LIMIT_WINDOW_SECONDS=60` là quota của cùng provider/model account.
5. `QDRANT_SPARSE_VECTOR_NAME=bm25`, `QDRANT_SPARSE_MODEL=Qdrant/bm25` và
   `QDRANT_SPARSE_LANGUAGE=none` phải giống cấu hình lúc ingest. Không đổi dense
   dimension/model mà không tạo generation mới.

Các HTTP inference service phải hỗ trợ:

- `POST /embed` với `{inputs, model, revision, normalize}` và trả list vector
  hoặc `{embeddings: [...]}`; mọi vector phải có đúng 1024 phần tử.
- `POST /rerank` với `{query, documents, model, revision}` và trả list score
  hoặc `{scores: [...]}`.
- `GET /health` trả HTTP 2xx.

Pin image/model revision và không tải model weights vào API/worker process.

## Khởi động

Dev/local:

```bash
make install-dev
make local-start                 # Redis + Qdrant
make ingest                      # tạo generation 14 và switch aliases
make run-api                     # RAG_EXECUTION_MODE=inline
```

Production:

```bash
make install
make run-api API_HOST=0.0.0.0
RAG_EXECUTION_MODE=redis_worker make run-worker
```

Chạy nhiều worker replicas với cùng `REDIS_QUEUE_GROUP`; mỗi worker dùng
`RAG_WORKER_CONCURRENCY` consumers. Đặt `RAG_QUEUE_MAX_OUTSTANDING` theo RAM và
thời gian chờ chấp nhận được. Queue đầy trả `429` và `Retry-After`; Redis lỗi
trả `503`; job quá TTL/wait trả `504`.

`Idempotency-Key` (tối đa 128 ký tự) có thể gửi cùng `/chat` hoặc
`/chat/stream`; Redis trả lại job đang tồn tại thay vì enqueue trùng.

## Migration và backfill

PR14 thay cả model dense và child schema, vì vậy phải re-ingest toàn bộ corpus:

1. Tạo generation staging mới với `.env` đã pin BGE-M3 revision và pipeline/schema
   version `14.0`/`3`.
2. Chạy ingest trên dataset thật; không ghi vào collection đang là alias active.
3. Kiểm tra dimension 1024, sparse modifier `IDF`, native sparse coverage 100%,
   child-parent references và dataset payload indexes.
4. Replay bộ đánh giá Việt/Anh (acronym, mã sản phẩm, bảng, tài liệu dài) và
   shadow rank trước alias switch. Không ghi raw query nhạy cảm vào log.
5. Switch hai alias atomically, giữ generation cũ ít nhất
   `INGEST_GENERATION_RETENTION` và rollback window.

Native BM25 được query trực tiếp tại Qdrant; không có BM25 cache/scroll trong API
và không cần restart sau alias switch. Dense và sparse prefetch bắt buộc nhận
cùng `RetrievalScope`; filter sau top-K không được coi là đủ.

## Dev/test checklist

- `make check` phải xanh; unit test không cần Redis/Qdrant HTTP.
- Có thể dùng `RAG_EXECUTION_MODE=inline`, `EMBEDDING_RUNTIME=local` để test
  adapter; không dùng cấu hình này cho production.
- Chạy smoke native Qdrant khi có Docker:
  `RUN_QDRANT_INTEGRATION=1 PYTHONPATH=src rag/bin/python -m pytest
  tests/test_qdrant_pr12_integration.py -q`.
- Kiểm tra `/ready` sau khi Qdrant aliases, Redis và hai GPU service đã healthy.
- Theo dõi queue outstanding, reservation schedule, worker failures, embedding /
  rerank latency và LLM 429/timeout. Không bật SDK blind retry; OpenAI đã đặt
  `max_retries=0`, Gemini attempts=1.

## Rollback

Nếu quality hoặc latency không đạt, dừng ingest mới, giữ worker/API hiện tại,
switch alias về generation cũ bằng `IngestionPipeline().rollback(<generation>)`,
rồi xác nhận `/ready` và một truy vấn smoke trong đúng scope. Không trộn child
generation mới với parent generation cũ và không xóa generation cũ trước khi
hết thời gian có request đang chạy.
