PYTHONPATH := $(abspath src)
PYTHON := rag/bin/python
UV := $(shell command -v uv 2>/dev/null || echo /Users/binh.dv/.local/bin/uv)
# Bắt uv sync đồng bộ vào đúng môi trường rag/ (thay vì tạo .venv mặc định)
export UV_PROJECT_ENVIRONMENT := rag

.PHONY: help install install-dev local-start local-stop ingest ingest-sync ingest-load run-api run-ui evaluate test lint check clean

help: ## Hiển thị danh sách lệnh
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ======================================
# ------- Setup & Infrastructure -------
# ======================================

install: ## Cài đặt dependencies (uv sync --locked — đọc pyproject.toml + uv.lock, đồng bộ vào rag/)
	$(UV) sync --locked

install-dev: ## Cài đặt dependencies + dev deps (uv sync --locked — đọc pyproject.toml + uv.lock, đồng bộ vào rag/)
	$(UV) sync --locked --extra dev

local-start: ## Khởi động Qdrant (Docker)
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

run-api: ## Chạy FastAPI backend (mặc định 127.0.0.1:8080)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --reload

run-ui: ## Chạy Gradio chat UI (port 7860)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m api.ui

# ======================================
# ----------- Evaluation ---------------
# ======================================

evaluate: ## Chạy RAG evaluation (RAGAS metrics)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m evaluation.evaluate

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
