# dataset-converter

Temporal-orchestrated pipeline that converts HuggingFace datasets into a strict
**SFT JSONL** format using **Kimi K2 Thinking** (via NVIDIA NIM) to generate the
per-dataset converter code.

---

## Output format

Every source row produces exactly **4 consecutive JSONL lines** — one JSON object
per message role — followed by a blank separator line:

```jsonl
{"role":"system",   "content":"You are a math reasoning assistant."}
{"role":"user",     "content":"What is 2+2?"}
{"role":"assistant","content":"4"}
{"role":"tool",     "content":""}

{"role":"system",   "content":"..."}
...
```

Rules enforced at write-time **and** at validation-time:

| # | Role | Policy |
|---|------|--------|
| 1 | `system` | Static prompt chosen by Kimi, or empty when absent in source |
| 2 | `user` | Question / input / last-user-turn extracted from the row |
| 3 | `assistant` | Answer / response / last-assistant-turn |
| 4 | `tool` | Any tool-call content tied to that exchange, otherwise `""` |

Order is **fixed** — the schema validator rejects anything out of sequence.

---

## Project layout

```
dataset_converter/
├── think.py             ← Temporal activities + DatasetSFTWorkflow
├── worker.py            ← Start the Temporal worker
├── run_workflow.py      ← Submit workflows for every dataset
├── download_dataset.py  ← (unchanged) Download raw parquet files
├── pyproject.toml
├── .env                 ← NVIDIA_API_KEY + TEMPORAL_ADDRESS
├── data/                ← Raw .parquet inputs  (git-ignored)
├── output/              ← Converted .jsonl files (git-ignored)
└── generated/           ← Optional: saved generated scripts (git-ignored)
```

---

## Architecture

```
run_workflow.py
      │  submit DatasetSFTWorkflow per dataset
      ▼
 Temporal Server
      │  schedule activities on task queue "dataset-converter"
      ▼
 worker.py  (always-on process)
      │
      ├─ profile_dataset            read schema + 5 rows, no full load
      ├─ kimi_generate_plan         Kimi → JSON mapping plan
      ├─ kimi_generate_transformer_script  Kimi → Python convert_record()
      ├─ run_transform_script       AST-validate → sandboxed exec → JSONL
      └─ validate_output_file       spot-check 25 sample groups
```

**LLM scope**: Kimi only does inference (planning) and code generation.
File I/O, chunking, writing, and schema enforcement all happen inside
Temporal activities — never inside LLM-generated code.

**Sandbox**: the generated `convert_record(record)` runs inside a restricted
`exec` namespace — no file I/O, no network, no subprocess, no `eval`/`open`.
An AST walk rejects dangerous imports and calls before the script ever executes.

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Edit `.env`:

```env
NVIDIA_API_KEY="nvapi-..."
TEMPORAL_ADDRESS="localhost:7233"
```

### 3. Download datasets  *(skip if `data/` already populated)*

```bash
uv run python download_dataset.py
```

### 4. Start Temporal  *(skip if already running)*

```bash
# via Docker — easiest
docker run --rm -p 7233:7233 -p 8080:8080 \
  temporalio/auto-setup:latest
```

Or use the [Temporal CLI](https://docs.temporal.io/cli):

```bash
temporal server start-dev
```

---

## Run

**Terminal A** — keep this running:

```bash
uv run python worker.py
```

**Terminal B** — submit all workflows and wait for results:

```bash
uv run python run_workflow.py
```

Converted files appear in `output/` as they finish.

---

## Datasets

| HuggingFace name | Input | Output |
|---|---|---|
| `meta-math/MetaMathQA` | `data/metamath.parquet` | `output/metamath.jsonl` |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | `data/magicoder.parquet` | `output/magicoder.jsonl` |
| `allenai/tulu-v2-sft-mixture` | `data/tulu.parquet` | `output/tulu.jsonl` |
| `teknium/OpenHermes-2.5` | `data/openhermes.parquet` | `output/openhermes.jsonl` |
| `LDJnr/Puffin` | `data/puffin.parquet` | `output/puffin.jsonl` |

---

## Reading the output

```python
import json
from itertools import islice

def iter_samples(path, max_samples=None):
    """Yield one sample (list of 4 message dicts) at a time."""
    with open(path, encoding="utf-8") as fh:
        lines = iter(fh)
        count = 0
        while True:
            if max_samples is not None and count >= max_samples:
                break
            block = []
            for line in lines:
                line = line.strip()
                if not line:
                    break          # blank separator = end of this sample
                block.append(json.loads(line))
            if not block:
                break              # EOF
            yield block
            count += 1

for sample in islice(iter_samples("output/metamath.jsonl"), 3):
    for msg in sample:
        print(msg)
    print()
```
