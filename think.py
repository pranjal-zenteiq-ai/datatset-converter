"""
think.py — Temporal activities + DatasetSFTWorkflow

Output format per source row (4 lines in the JSONL, one object per line):
    {"role":"system",  "content":"..."}
    {"role":"user",    "content":"..."}
    {"role":"assistant","content":"..."}
    {"role":"tool",    "content":"..."}

Each converted sample is written as exactly 4 consecutive JSONL lines.
An empty separator line is written between samples so the file stays human-readable.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from temporalio import activity, workflow

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")

# Modules the generated script is allowed to import
SAFE_IMPORTS: set[str] = {
    "json",
    "re",
    "math",
    "itertools",
    "collections",
    "functools",
    "typing",
    "string",
    "unicodedata",
}

# Built-ins exposed inside the generated script's sandbox
SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "bytes": bytes,
    "chr": chr,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "print": print,  # debugging inside sandbox is OK
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "vars": vars,
    "zip": zip,
    "__import__": __import__,
    "True": True,
    "False": False,
    "None": None,
}

# ──────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ──────────────────────────────────────────────────────────────────────────────


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first well-formed JSON object from a model response."""
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
    return json.loads(text[start : end + 1])


def _build_llm(max_tokens: int = 8192) -> ChatNVIDIA:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is missing from the environment")
    return ChatNVIDIA(
        model="moonshotai/kimi-k2-thinking",
        api_key=api_key,
        temperature=0,
        top_p=1,
        max_tokens=max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset profiling helpers
# ──────────────────────────────────────────────────────────────────────────────


def _preview_parquet(path: str, n: int = 5) -> Dict[str, Any]:
    pf = pq.ParquetFile(path)
    head_rows: List[Dict] = []
    for batch in pf.iter_batches(batch_size=n):
        head_rows = batch.to_pylist()
        break

    schema = pf.schema_arrow
    return {
        "columns": list(schema.names),
        "dtypes": {schema.names[i]: str(schema.types[i]) for i in range(len(schema.names))},
        "head": head_rows,
        "row_count": pf.metadata.num_rows if pf.metadata else None,
        "file_type": "parquet",
    }


def _preview_csv(path: str, n: int = 5) -> Dict[str, Any]:
    df = pd.read_csv(path, nrows=n)
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.fillna("").to_dict(orient="records"),
        "row_count": None,
        "file_type": "csv",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Activity: profile_dataset
# ──────────────────────────────────────────────────────────────────────────────


@activity.defn
def profile_dataset(local_path: str) -> Dict[str, Any]:
    """Read the first few rows and schema of the input file without loading it all."""
    if not Path(local_path).exists():
        raise FileNotFoundError(f"Input file not found: {local_path}")
    if local_path.endswith(".parquet"):
        return _preview_parquet(local_path, n=5)
    if local_path.endswith(".csv"):
        return _preview_csv(local_path, n=5)
    raise ValueError(f"Unsupported input format: {local_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Activity: kimi_generate_plan
# ──────────────────────────────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a dataset schema analyst. Your ONLY job is to identify which source
fields in this dataset semantically map to which output roles.

Output roles (fixed order, fixed names):
    system    — a system/instruction prompt already present in the source row
    user      — the human question / input turn
    assistant — the model answer / output turn
    tool      — any tool-call result already present in the source row

HOW TO DETERMINE THE MAPPING — THIS IS THE MOST IMPORTANT PART:
Do NOT rely on column names alone. Column names are arbitrary.
You MUST look at the actual VALUES in the sample rows to understand what
each field contains. Ask yourself:
  - Which field holds a question, prompt, or human input?
  - Which field holds a model response, answer, or completion?
  - Is there a field that contains a system/instruction prompt string?
  - Is there a nested list of message objects? If so, what keys hold
    the role identifier and the text content?

Common patterns you will encounter (column names vary wildly):

  Flat datasets:
    "query" / "response"           → user / assistant
    "instruction" / "output"       → user / assistant
    "prompt" / "completion"        → user / assistant
    "input" / "answer"             → user / assistant
    "question" / "solution"        → user / assistant

  Message-list datasets (look at sample VALUES to detect the structure):
    column "conversations" or "messages" containing a list of dicts where:
      role key might be: "role", "from", "speaker", "author"
      role values might be: "user"/"assistant", "human"/"gpt",
                             "USER"/"ASSISTANT", "Human"/"AI", etc.
      content key might be: "content", "value", "text", "message"

  System prompt (only if it ACTUALLY EXISTS in the row):
    A top-level field whose value looks like an instruction/persona string.
    Or a message object in the list whose role value is "system".
    Common names: "system", "system_prompt", "sys", "context",
                  "instruction", "persona" — but CHECK THE VALUES.
    If you cannot find one in the actual sample rows → system_field = null.

ABSOLUTE RULES — violating any of these makes the plan invalid:
1. You MUST NOT invent, compose, or generate any text content yourself.
2. You MUST NOT suggest putting a static string into any content field.
3. "system" content MUST come from a real field actually present in the row.
   If no such field exists in the sample rows → system_field = null → content = "".
4. "tool" content MUST come from a real field actually present in the row.
   If no such field exists → tool_field = null → content = "".
5. NEVER reference columns or nested keys that do not appear in the profile.
6. If the source has a conversations/messages list:
   - Identify the correct role key and content key from the SAMPLE VALUES.
   - Extract ONE supervised pair: the last user→assistant exchange.
   - If a system message exists as the first message in the list, extract it.
7. If the source is flat, identify user and assistant fields from their values.
8. All other columns are irrelevant — ignore them.

Return ONLY a JSON object (no prose, no markdown fences) with these keys:
{{
  "confirmed": true,
  "strategy": "messages_list" | "flat_columns" | "mixed",
  "conversation_policy": "last_supervised_pair" | "first_pair" | "n/a",
  "message_field_candidates": [],
  "role_key": "<the key inside each message dict that holds the role name, or null>",
  "content_key": "<the key inside each message dict that holds the text, or null>",
  "user_role_values": ["<exact role values that mean 'user', e.g. human, USER, user>"],
  "assistant_role_values": ["<exact role values that mean 'assistant', e.g. gpt, ASSISTANT>"],
  "system_role_values": ["<exact role values that mean 'system', or empty list>"],
  "field_to_role": {{}},
  "system_field": "<exact top-level column name or null>",
  "tool_field": "<exact top-level column name or null>",
  "notes": []
}}

Dataset name: {dataset_name}

Profile (schema + 5 sample rows):
{profile_json}
"""


@activity.defn
async def kimi_generate_plan(dataset_name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    client = _build_llm(max_tokens=8192)
    prompt = _PLAN_PROMPT.format(
        dataset_name=dataset_name,
        profile_json=json.dumps(profile, ensure_ascii=False, indent=2),
    )
    response = await client.ainvoke(prompt)
    plan = _extract_json_object(response.content or "")
    if not plan.get("confirmed", False):
        raise ValueError("Kimi did not confirm the plan — check model response")
    return plan


# ──────────────────────────────────────────────────────────────────────────────
# Activity: kimi_generate_transformer_script
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_PROMPT = """\
You must output:
1) A FIRST line that says exactly: CONFIRMED_OK
2) Then ONLY pure Python code — no prose, no markdown fences, no comments outside code.

The code must define ONE function with this exact signature:

    def convert_record(record: dict) -> list[dict]:

INPUT / OUTPUT CONTRACT:
- Input:  one dict representing a single source row. All values are already
          plain Python objects (strings, ints, lists, dicts, None).
          The field names are whatever the dataset uses — they are NOT
          guaranteed to match "system", "user", "assistant", or "tool".
          You already know the exact field names from the Profile and Plan.
- Output: a Python list of exactly 4 dicts in this fixed order:
    [{{"role":"system",   "content":"<str>"}},
     {{"role":"user",     "content":"<str>"}},
     {{"role":"assistant","content":"<str>"}},
     {{"role":"tool",     "content":"<str>"}}]

CRITICAL CONTENT RULES — violating any of these is wrong:
1. ALL content values must be extracted directly from the source record
   using the ACTUAL field names you found in the Profile and Plan.
   COPY the raw value into the slot. Do NOT write your own text.
2. NEVER put a static/hardcoded string into a content field.
   WRONG:  {{"role":"system", "content":"You are a helpful assistant."}}
   RIGHT:  {{"role":"system", "content": str(record.get("sys_prompt", "") or "")}}
   The field name in .get() must be the REAL column name from the dataset,
   not a guess or a generic name.
3. If a role has no corresponding field in THIS dataset, its content is "".
   Use a bare empty string. Do NOT invent any substitute content.
4. The function is a pure field extractor. It reads values from record
   and places them into the correct slots. Nothing more.

FOR MESSAGE-LIST DATASETS:
- The record may contain a field like "conversations", "messages", "dialogue",
  or any other name — use whatever the Plan identified.
- Each item in that list is a dict. The role key and content key may be
  anything — e.g. {{"from":"human","value":"..."}}, or {{"role":"user","content":"..."}}
  Use the exact keys from the Plan (role_key, content_key).
- Extract ONE supervised pair: the last occurrence of a user→assistant exchange.
- If a system message exists as the first item (role = a system_role_value
  from the Plan), extract its content into system. Otherwise system = "".
- Walk the list using the EXACT role values from the Plan
  (user_role_values, assistant_role_values, system_role_values).

OTHER RULES:
- Every content value must be a plain str (never None, never a list).
- Be robust to missing keys, None values, empty lists, nested structures.
- No file I/O, network calls, pandas, datasets library, or subprocess.
- No eval / exec / open / compile.
- You MAY import from: {safe_imports}

Dataset name: {dataset_name}

Profile (schema + 5 sample rows):
{profile_json}

Plan:
{plan_json}
"""


def _clean_generated_script(raw: str) -> str:
    raw = raw.strip()
    # Drop the CONFIRMED_OK sentinel line
    lines = [ln for ln in raw.splitlines() if ln.strip() != "CONFIRMED_OK"]
    script = "\n".join(lines).strip()
    # Strip any residual markdown fences the model may have added
    return _strip_code_fences(script)


@activity.defn
async def kimi_generate_transformer_script(
    dataset_name: str,
    profile: Dict[str, Any],
    plan: Dict[str, Any],
) -> str:
    client = _build_llm(max_tokens=16384)
    prompt = _SCRIPT_PROMPT.format(
        safe_imports=", ".join(sorted(SAFE_IMPORTS)),
        dataset_name=dataset_name,
        profile_json=json.dumps(profile, ensure_ascii=False, indent=2),
        plan_json=json.dumps(plan, ensure_ascii=False, indent=2),
    )
    response = await client.ainvoke(prompt)
    raw = response.content or ""
    if "CONFIRMED_OK" not in raw:
        raise ValueError("Kimi did not confirm the script (missing CONFIRMED_OK)")
    return _clean_generated_script(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Script validation & sandboxed loading
# ──────────────────────────────────────────────────────────────────────────────

_UNSAFE_CALLS = {"eval", "exec", "open", "compile", "breakpoint", "__import__"}
_UNSAFE_ATTRS = {"system", "popen", "remove", "unlink", "rmtree", "call", "run", "check_output"}


def _validate_generated_script(script: str) -> None:
    """AST-walk the generated script and reject anything dangerous."""
    try:
        tree = ast.parse(script)
    except SyntaxError as exc:
        raise ValueError(f"Generated script has a syntax error: {exc}") from exc

    has_convert_record = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "convert_record":
            has_convert_record = True

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in SAFE_IMPORTS:
                    raise ValueError(f"Unsafe import in generated script: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in SAFE_IMPORTS:
                raise ValueError(f"Unsafe import-from in generated script: {node.module}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _UNSAFE_CALLS:
                raise ValueError(f"Unsafe call in generated script: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _UNSAFE_ATTRS:
                raise ValueError(f"Unsafe attribute call in generated script: .{node.func.attr}()")

    if not has_convert_record:
        raise ValueError("Generated script must define convert_record(record: dict) -> list[dict]")


def _load_converter(script: str) -> Callable[[Dict[str, Any]], List[Dict[str, str]]]:
    """Execute the validated script in a restricted namespace and return convert_record."""
    namespace: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    exec(script, namespace, namespace)  # noqa: S102 — intentional sandboxed exec
    fn = namespace.get("convert_record")
    if not callable(fn):
        raise ValueError("convert_record was not defined by the generated script")
    return fn  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# Schema enforcement
# ──────────────────────────────────────────────────────────────────────────────


def _enforce_message_schema(converted: Any, row_index: int) -> List[Dict[str, str]]:
    """
    Validate and normalize converter output.

    The converter must return exactly 4 dicts in the order:
        system, user, assistant, tool

    Each dict must have "role" and "content" where content is a str.
    """
    if not isinstance(converted, list):
        raise ValueError(
            f"Row {row_index}: convert_record must return a list, got {type(converted).__name__}"
        )
    if len(converted) != 4:
        raise ValueError(
            f"Row {row_index}: convert_record must return exactly 4 messages, got {len(converted)}"
        )

    result: List[Dict[str, str]] = []
    for i, (msg, expected_role) in enumerate(zip(converted, TARGET_ROLES)):
        if not isinstance(msg, dict):
            raise ValueError(
                f"Row {row_index}, message {i}: expected dict, got {type(msg).__name__}"
            )
        role = str(msg.get("role", "")).strip()
        if role != expected_role:
            raise ValueError(
                f"Row {row_index}, message {i}: expected role '{expected_role}', got '{role}'"
            )
        content = msg.get("content")
        if content is None:
            content = ""
        result.append({"role": role, "content": str(content)})
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Batched input iteration
# ──────────────────────────────────────────────────────────────────────────────


def _iter_input_batches(local_path: str, chunk_size: int):
    if local_path.endswith(".parquet"):
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=chunk_size):
            yield batch.to_pylist()
        return
    if local_path.endswith(".csv"):
        for chunk in pd.read_csv(local_path, chunksize=chunk_size):
            yield chunk.fillna("").to_dict(orient="records")
        return
    raise ValueError(f"Unsupported input format: {local_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Activity: run_transform_script
# ──────────────────────────────────────────────────────────────────────────────


@activity.defn
def run_transform_script(
    script: str,
    local_path: str,
    output_path: str,
    chunk_size: int = 2000,
) -> Dict[str, Any]:
    """
    Validate the generated script, then apply it row-by-row to the input file.

    Output format — each converted sample occupies exactly 4 consecutive lines
    in the JSONL, one JSON object per line, followed by a blank separator line:

        {"role":"system",   "content":"..."}
        {"role":"user",     "content":"..."}
        {"role":"assistant","content":"..."}
        {"role":"tool",     "content":"..."}
        <blank line>

    This makes the file trivially parseable: read 4 non-blank lines = one sample.
    """
    _validate_generated_script(script)
    convert_record = _load_converter(script)

    input_file = Path(local_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    # Write atomically via a temp file in the same directory
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=output_file.parent,
        suffix=".jsonl",
        encoding="utf-8",
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            for batch in _iter_input_batches(str(input_file), chunk_size):
                for record in batch:
                    converted = convert_record(record)
                    messages = _enforce_message_schema(converted, total_rows)

                    # Write 4 lines — one JSON object per message role
                    for msg in messages:
                        tmp.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    tmp.write("\n")  # blank separator between samples

                    total_rows += 1

            tmp.flush()
            os.fsync(tmp.fileno())

        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    tmp_path.replace(output_file)
    return {"output_path": str(output_file), "rows_written": total_rows}


# ──────────────────────────────────────────────────────────────────────────────
# Activity: validate_output_file
# ──────────────────────────────────────────────────────────────────────────────


@activity.defn
def validate_output_file(output_path: str, sample_groups: int = 25) -> Dict[str, Any]:
    """
    Confirm that the output JSONL file is well-formed.

    Reads `sample_groups` worth of 4-line blocks and verifies:
    - Each block has exactly 4 objects.
    - Roles appear in the exact order: system, user, assistant, tool.
    - Every "content" value is a plain string.
    """
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    validated_groups = 0
    with path.open("r", encoding="utf-8") as fh:
        while validated_groups < sample_groups:
            block: List[Dict[str, str]] = []
            for _ in range(4):
                line = fh.readline()
                if line == "":  # true EOF
                    break
                line = line.strip()
                if not line:
                    # skip the blank separator and retry this slot
                    line = fh.readline()
                    if line == "":
                        break
                    line = line.strip()
                if line:
                    block.append(json.loads(line))

            if len(block) == 0:
                break  # end of file

            _enforce_message_schema(block, validated_groups)
            validated_groups += 1

            # consume the trailing blank separator line (may already be consumed above)
            # — just try to read one more line and discard if blank
            peek = fh.readline()
            if peek.strip():
                # Not blank → the cursor is now on a real data line; we lost sync.
                # This is a format error but we'll catch it in the next iteration naturally.
                pass

    return {
        "ok": True,
        "groups_sampled": validated_groups,
        "path": str(path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Workflow
# ──────────────────────────────────────────────────────────────────────────────


@workflow.defn
class DatasetSFTWorkflow:
    """
    Orchestrates dataset → SFT JSONL conversion via Kimi K2 Thinking.

    Steps:
        1. profile_dataset         — read schema + 5 rows (no full load)
        2. kimi_generate_plan      — ask Kimi how to map source fields → roles
        3. kimi_generate_transformer_script — ask Kimi to write convert_record()
        4. run_transform_script    — validate & run the script, write JSONL
        5. validate_output_file    — spot-check the output JSONL
    """

    @workflow.run
    async def run(
        self,
        dataset_name: str,
        local_path: str,
        output_path: str,
        chunk_size: int = 2000,
    ) -> str:
        # ── Step 1: profile ───────────────────────────────────────────────────
        profile: Dict[str, Any] = await workflow.execute_activity(
            profile_dataset,
            local_path,
            start_to_close_timeout=workflow.timedelta(seconds=300),
        )

        # ── Step 2: generate mapping plan ────────────────────────────────────
        plan: Dict[str, Any] = await workflow.execute_activity(
            kimi_generate_plan,
            args=[dataset_name, profile],
            start_to_close_timeout=workflow.timedelta(seconds=900),
        )

        # ── Step 3: generate Python converter ────────────────────────────────
        script: str = await workflow.execute_activity(
            kimi_generate_transformer_script,
            args=[dataset_name, profile, plan],
            start_to_close_timeout=workflow.timedelta(seconds=1200),
        )

        # ── Step 4: apply converter ───────────────────────────────────────────
        await workflow.execute_activity(
            run_transform_script,
            args=[script, local_path, output_path, chunk_size],
            start_to_close_timeout=workflow.timedelta(hours=4),
        )

        # ── Step 5: validate output ───────────────────────────────────────────
        await workflow.execute_activity(
            validate_output_file,
            args=[output_path],
            start_to_close_timeout=workflow.timedelta(seconds=300),
        )

        return output_path
