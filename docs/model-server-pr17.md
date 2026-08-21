# PR17 — model server BGE-M3 trên macOS Apple Silicon

PR17 thêm một FastAPI process tự host `BAAI/bge-m3` và
`BAAI/bge-reranker-v2-m3`. API/worker gọi process này qua HTTP; process chỉ
chạy một Uvicorn worker và giữ một instance mỗi model. Nhiều HTTP request vẫn
được nhận đồng thời, nhưng dynamic batcher gom chúng thành batch ngắn (mặc
định 8 ms) và `MPSExecutionArbiter` chỉ cho phép một forward batch tại một
thời điểm để tránh tranh chấp unified memory.

## Chạy dev trên Mac

```bash
make install-dev
cp .env.example .env
make run-model-server MODEL_SERVER_DEVICE=cpu
```

Sau khi đã cài PyTorch có MPS và đang chạy trên Apple Silicon:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 make run-model-server MODEL_SERVER_DEVICE=mps
curl -s http://127.0.0.1:8082/ready | jq
curl -s http://127.0.0.1:8082/metadata | jq
```

Không dùng `--reload` hoặc `--workers > 1`: mỗi worker sẽ load thêm hàng GB
trọng số và tạo thêm queue/MPS context. Chỉ bật `MODEL_SERVER_ALLOW_CPU_FALLBACK`
trong dev khi muốn kiểm thử trên máy không có MPS.

## Hợp đồng HTTP

`POST /embed` nhận:

```json
{
  "inputs": ["quy định về nghỉ phép"],
  "model": "BAAI/bge-m3",
  "revision": "<git-or-hub-revision>",
  "normalize": true,
  "priority": "online"
}
```

Response gồm `embeddings`, `model`, `revision` và `dimension` (BGE-M3 là
1024). `POST /rerank` nhận `query`, `documents`, `model`, `revision` và
`priority`; response gồm một `scores` cho mỗi document. `priority=batch` dành
cho ingest/backfill, còn request chat dùng `online`. Nếu cấu hình
`MODEL_SERVER_API_KEY`, gửi key qua `X-API-Key`.

Các lỗi chính: `409` sai model/revision/normalize, `413` vượt giới hạn một
request/body, `429` queue đầy (có `Retry-After`), `504` chờ queue quá hạn và
`503` model chưa sẵn sàng hoặc forward thất bại. `/health` chỉ kiểm tra process;
`/ready` kiểm tra model và queue; `/metadata` là gate để API xác nhận đúng
revision/dimension trước khi phục vụ.

## Kết nối RAG API

Đặt `EMBEDDING_RUNTIME=remote`, `RERANKER_RUNTIME=remote` và trỏ cả hai
`*_BASE_URL` tới model server, ví dụ `http://127.0.0.1:8082` cho cả hai
(hoặc hai reverse-proxy cùng process). API chỉ
đòi `/ready` + `/metadata` khi `MODEL_SERVER_ENABLED=true`; khi tắt, tương
thích với service PR14 cũ và chỉ gọi `/health`. Khi bật trong production,
đặt revision bất biến, `MODEL_SERVER_DEVICE=mps`,
`MODEL_SERVER_ALLOW_CPU_FALLBACK=false`, `MODEL_SERVER_WORKERS=1`.

## Tuning và vận hành

- `*_DYNAMIC_BATCH_MAX_ITEMS` và `*_DYNAMIC_BATCH_MAX_TOKENS` là trần batch;
  `MODEL_SERVER_MAX_PENDING_*` là trần toàn queue. Tăng từng bước sau khi đo
  RSS/MPS memory, p95 latency và throughput.
- `MODEL_SERVER_BATCH_WAIT_MS` đánh đổi TTFT với khả năng coalescing; 5–10 ms
  phù hợp chat interactive, ingest có thể dùng `priority=batch`.
- `MODEL_SERVER_MAX_ITEMS_PER_REQUEST_PER_BATCH` ngăn một request ingest lớn
  chiếm trọn batch; phần còn lại được requeue sau khi scatter.
- `MODEL_SERVER_MPS_MAX_INFLIGHT_BATCHES=1` là baseline an toàn. Chỉ tăng nếu
  đã load-test trên đúng máy và không có MPS OOM.
- `/metrics` trả JSON queue depth, số batch/item và lỗi inference; scrape qua
  reverse-proxy có auth trong production.
- Model revision hoặc dimension đổi thì phải backfill/re-index generation mới,
  replay quality test rồi mới switch alias; không đổi in-place collection.

## Smoke/load gate

```bash
curl -sf http://127.0.0.1:8082/ready
PYTHONPATH=src rag/bin/python -m pytest tests/test_model_server_pr17.py -q
```

Load gate tối thiểu phải chạy đồng thời `/embed`, `/rerank` và `/chat` với
corpus đại diện, ghi p50/p95/p99, queue depth, batch size, RSS/MPS memory,
timeout/429 rate và kết quả Recall@K/MRR/nDCG. Không coi việc nhiều request
được HTTP 200 là bằng chứng GPU chạy song song: PR17 cố ý serialize forward
trên một MPS context và batch chúng ở biên.
