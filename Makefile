.PHONY: setup start-server stop-server logs run-worker run-pipeline run-all

setup:
	@echo "Syncing python dependencies..."
	uv sync
	@echo "Pulling latest docker images..."
	docker compose pull

start-server:
	@echo "Starting Temporal dev server..."
	docker compose up -d
	@echo "Waiting a few seconds for Temporal to initialize..."
	@sleep 3

stop-server:
	@echo "Stopping Temporal server..."
	docker compose down

logs:
	docker compose logs -f

run-worker:
	@echo "Starting Temporal worker (Ctrl-C to stop)..."
	uv run python worker.py

run-pipeline:
	@echo "Dispatching workflows..."
	uv run python run_workflow.py

run-all: start-server
	@echo "Starting worker in the background..."
	@uv run python worker.py > worker.log 2>&1 & echo $$! > worker.pid
	@sleep 2
	@echo "Running the pipeline..."
	@uv run python run_workflow.py || (echo "Pipeline failed!"; kill `cat worker.pid` 2>/dev/null; rm worker.pid; exit 1)
	@echo "Pipeline finished successfully."
	@kill `cat worker.pid` 2>/dev/null || true
	@rm worker.pid
	@echo "Worker stopped. Temporal is still running, stop it with 'make stop-server'."
