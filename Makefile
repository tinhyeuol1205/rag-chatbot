PYTHONPATH := $(abspath src)
PYTHON := rag/bin/python
UV := $(shell command -v uv 2>/dev/null || echo /Users/binh.dv/.local/bin/uv)
# Bắt uv sync đồng bộ vào đúng môi trường rag/ (thay vì tạo .venv mặc định)
export UV_PROJECT_ENVIRONMENT := rag

.PHONY: help install install-dev local-start local-stop ingest run-api run-ui evaluate test clean

help: ## Hiển thị danh sách lệnh
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ======================================
# ------- Setup & Infrastructure -------
# ======================================

install: ## Cài đặt dependencies (uv sync — đọc pyproject.toml + uv.lock, đồng bộ vào rag/)
	$(UV) sync

install-dev: ## Cài đặt dependencies + dev deps (pytest, ruff)
	$(UV) sync --extra dev

local-start: ## Khởi động Qdrant (Docker)
	docker compose up -d

local-stop: ## Dừng Docker
	docker compose down --remove-orphans

# ======================================
# ---------- Data Ingestion ------------
# ======================================

ingest: ## Ingest tất cả sample documents vào Qdrant
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ingestion.main

# ======================================
# --------- Run Application -----------
# ======================================

run-api: ## Chạy FastAPI backend (port 8000)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

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

# ======================================
# ------------- Cleanup ----------------
# ======================================

clean: ## Xóa Qdrant data và cache
	docker compose down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
