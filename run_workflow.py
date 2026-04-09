"""
run_workflow.py — Submit one DatasetSFTWorkflow per dataset and wait for results.

Usage:
    uv run python run_workflow.py
"""
from __future__ import annotations

import asyncio
import os
from typing import List, Tuple

from dotenv import load_dotenv
from temporalio.client import Client

from think import DatasetSFTWorkflow

load_dotenv()

TASK_QUEUE = "dataset-converter"

# (hf_dataset_name, local_input_path, output_path)
DATASETS: List[Tuple[str, str, str]] = [
    (
        "meta-math/MetaMathQA",
        "data/metamath.parquet",
        "output/metamath.jsonl",
    ),
    (
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "data/magicoder.parquet",
        "output/magicoder.jsonl",
    ),
    (
        "allenai/tulu-v2-sft-mixture",
        "data/tulu.parquet",
        "output/tulu.jsonl",
    ),
    (
        "teknium/OpenHermes-2.5",
        "data/openhermes.parquet",
        "output/openhermes.jsonl",
    ),
    (
        "LDJnr/Puffin",
        "data/puffin.parquet",
        "output/puffin.jsonl",
    ),
]


def _workflow_id(dataset_name: str) -> str:
    return "convert-" + dataset_name.replace("/", "-").replace(".", "-").lower()


async def main() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    print(f"[runner] Connecting to Temporal at {address} …\n")
    client = await Client.connect(address)

    for dataset_name, input_path, output_path in DATASETS:
        wf_id = _workflow_id(dataset_name)
        print(f"[runner] ▶  {dataset_name}")
        print(f"          input  → {input_path}")
        print(f"          output → {output_path}")
        print(f"          id     → {wf_id}")

        result = await client.execute_workflow(
            DatasetSFTWorkflow.run,
            args=[dataset_name, input_path, output_path, 2000],
            id=wf_id,
            task_queue=TASK_QUEUE,
        )

        print(f"[runner] ✓  Done: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
