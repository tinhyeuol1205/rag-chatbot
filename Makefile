PYTHONPATH := $(abspath src)
PYTHON := rag/bin/python
UV := $(shell command -v uv 2>/dev/null || echo /Users/binh.dv/.local/bin/uv)
# Bắt uv sync đồng bộ vào đúng môi trường rag/ (thay vì tạo .venv mặc định)
export UV_PROJECT_ENVIRONMENT := rag

.PHONY: help install install-dev local-start local-stop ingest ingest-sync ingest-load run-api run-worker run-model-server load-model-server run-ui evaluate benchmark-latency test lint check clean

help: ## Hiển thị danh sách lệnh
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ======================================
# ------- Setup & Infrastructure -------
# ======================================

install: ## Cài đặt dependencies (uv sync --locked — đọc pyproject.toml + uv.lock, đồng bộ vào rag/)
	$(UV) sync --locked

install-dev: ## Cài đặt dependencies + dev deps (uv sync --locked — đọc pyproject.toml + uv.lock, đồng bộ vào rag/)
	$(UV) sync --locked --extra dev

local-start: ## Khởi động Redis + Qdrant (Docker)
	docker compose up -d

local-stop: ## Dừng Docker
	docker compose down --remove-orphans

# ======================================
# ---------- Data Ingestion ------------
# ======================================

ingest: ## Ingest tất cả sample documents vào Qdrant
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ingestion.main

ingest-sync: ## Đồng bộ Qdrant với source directory, gồm xóa file không còn trên disk
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ingestion.main --sync

ingest-load: ## Chạy load benchmark PR11 (DATA_DIR=/srv/corpus)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/load_ingestion_pr11.py $(DATA_DIR)

# ======================================
# --------- Run Application -----------
# ======================================

API_HOST ?= 127.0.0.1
API_PORT ?= 8080          # ★ 8000 đang bị vLLM dùng — xem review PR 4 / P2-6
API_RELOAD ?= true
API_RELOAD_FLAG := $(if $(filter true 1 yes,$(API_RELOAD)),--reload,)

run-api: ## Chạy FastAPI backend (mặc định 127.0.0.1:8080)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) $(API_RELOAD_FLAG)

run-worker: ## Chạy Redis Streams RAG worker (production mode)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m workers.rag_worker

MODEL_SERVER_HOST ?= 127.0.0.1
MODEL_SERVER_PORT ?= 8082
MODEL_SERVER_DEVICE ?= cpu

run-model-server: ## Chạy BGE-M3 + reranker dynamic batcher (một worker, CPU hoặc MPS)
	MODEL_SERVER_ENABLED=true MODEL_SERVER_DEVICE=$(MODEL_SERVER_DEVICE) MODEL_SERVER_WORKERS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn model_server.main:app --host $(MODEL_SERVER_HOST) --port $(MODEL_SERVER_PORT) --workers 1

MODEL_LOAD_REQUESTS ?= 100
MODEL_LOAD_CONCURRENCY ?= 32

load-model-server: ## Load smoke PR17 (BASE_URL=http://127.0.0.1:8082 ENDPOINT=embed|rerank)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/load_model_server_pr17.py --base-url $(or $(BASE_URL),http://127.0.0.1:8082) --endpoint $(or $(ENDPOINT),embed) --requests $(MODEL_LOAD_REQUESTS) --concurrency $(MODEL_LOAD_CONCURRENCY)

UI_DEMO_MODE ?= dev
UI_API_BASE_URL ?= http://127.0.0.1:8080
UI_HOST ?= 127.0.0.1
UI_PORT ?= 7860
UI_PUBLIC_SHARE ?= false

run-ui: ## Chạy thin UI qua FastAPI: UI_DEMO_MODE=dev|product
	@if [ "$(UI_DEMO_MODE)" != "dev" ] && [ "$(UI_DEMO_MODE)" != "product" ]; then \
		echo "UI_DEMO_MODE phải là dev hoặc product (giá trị: $(UI_DEMO_MODE))" >&2; \
		exit 2; \
	fi
	UI_DEMO_MODE=$(UI_DEMO_MODE) UI_API_BASE_URL=$(UI_API_BASE_URL) UI_HOST=$(UI_HOST) UI_PORT=$(UI_PORT) UI_PUBLIC_SHARE=$(UI_PUBLIC_SHARE) PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m api.ui

# ======================================
# ----------- Evaluation ---------------
# ======================================

evaluate: ## Chạy RAG evaluation (RAGAS metrics)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m evaluation.evaluate

BENCHMARK_QUERY ?= Chính sách nghỉ phép của công ty là gì?
BENCHMARK_MODE ?= in-process
BENCHMARK_RUNS ?= 3
BENCHMARK_OUTPUT ?= artifacts/rag-latency.json

benchmark-latency: ## Đo latency từng module và RAG TTFT (BENCHMARK_MODE=in-process|http|both)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_rag_latency.py --mode $(BENCHMARK_MODE) --query "$(BENCHMARK_QUERY)" --runs $(BENCHMARK_RUNS) --output $(BENCHMARK_OUTPUT)

# ======================================
# ------------- Testing ----------------
# ======================================

test: ## Chạy unit tests
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

lint: ## Static checks (ruff)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check src tests

check: lint test ## Chạy toàn bộ quality gate local

# ======================================
# ------------- Cleanup ----------------
# ======================================

clean: ## Xóa Qdrant data và cache
	docker compose down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
