"""
run_workflow.py — Launch dataset conversion workflows in parallel.

Usage:
    uv run python run_workflow.py
"""
from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from typing import List, Tuple

from dotenv import load_dotenv
from temporalio.client import Client, WorkflowHandle

from think import DatasetSFTWorkflow

load_dotenv()

TASK_QUEUE = "dataset-converter"

# (hf_dataset_name, local_input_path, output_path)
DATASETS: List[Tuple[str, str, str]] = [
    ("totally-not-an-llm/EverythingLM-data-V3", "data/everythinglm.parquet", "output/everythinglm.jsonl"),
    ("microsoft/orca-math-word-problems-200k", "data/orca.parquet", "output/orca.jsonl"),
    ("qintongli/GSM-Plus-v0", "data/gsm.parquet", "output/gsm.jsonl"),
    ("Vezora/Tested-143k-Python-Alpaca", "data/python_alpaca.parquet", "output/python_alpaca.jsonl"),
    ("LDJnr/Puffin", "data/puffin.parquet", "output/puffin.jsonl"),
    ("roneneldan/TinyStories", "data/tinystories.parquet", "output/tinystories.jsonl"),
    ("nvidia/HelpSteer2", "data/nemotron.parquet", "output/nemotron.jsonl"),
    ("allenai/tulu-3-sft-mixture", "data/tulu3.parquet", "output/tulu3.jsonl"),
]


def _workflow_id(dataset_name: str) -> str:
    return "convert-" + dataset_name.replace("/", "-").replace(".", "-").lower()


async def main() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    print(f"[runner] Connecting to Temporal at {address} …\n")
    client = await Client.connect(address)

    handles = []
    for dataset_name, input_path, output_path in DATASETS:
        wf_id = _workflow_id(dataset_name)
        
        if not os.path.exists(input_path):
            print(f"[runner] ⚠  Skipping {dataset_name}: Input file {input_path} missing.")
            continue
            
        # Terminate existing workflow if it's already running
        try:
            handle = client.get_workflow_handle(wf_id)
            await handle.terminate(reason="Restarting runner")
            print(f"[runner] ⟲  Terminated existing workflow: {wf_id}")
        except Exception:
            pass

        print(f"[runner] ▶  Launching: {wf_id}")
        handle = await client.start_workflow(
            DatasetSFTWorkflow.run,
            args=[dataset_name, input_path, output_path],
            id=wf_id,
            task_queue=TASK_QUEUE,
        )
        handles.append((wf_id, handle))

    print(f"\n[runner] All {len(handles)} workflows launched. Waiting for results...\n")
    
    for wf_id, handle in handles:
        try:
            result = await handle.result()
            print(f"[runner] ✓  {wf_id}: {result}")
        except Exception as e:
            print(f"[runner] ✘  {wf_id} failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
