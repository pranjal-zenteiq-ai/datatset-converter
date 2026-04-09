"""
worker.py — Start the Temporal worker process.

Usage:
    uv run python worker.py
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

from think import (
    DatasetSFTWorkflow,
    kimi_generate_plan,
    kimi_generate_transformer_script,
    profile_dataset,
    run_transform_script,
    validate_output_file,
)

load_dotenv()

TASK_QUEUE = "dataset-converter"


async def main() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    print(f"[worker] Connecting to Temporal at {address} …")
    client = await Client.connect(address)

    import concurrent.futures

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DatasetSFTWorkflow],
        activities=[
            profile_dataset,
            kimi_generate_plan,
            kimi_generate_transformer_script,
            run_transform_script,
            validate_output_file,
        ],
        activity_executor=concurrent.futures.ThreadPoolExecutor(max_workers=20),
    )

    print(f"[worker] Listening on task queue '{TASK_QUEUE}' — Ctrl-C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
