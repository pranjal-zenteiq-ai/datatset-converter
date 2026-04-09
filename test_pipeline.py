"""
test_pipeline.py — Rigorous pipeline tests for all 4 datasets.

Tests are written against the REAL schemas observed in the actual parquet files:

  metamath   : flat  — query/response (no system field)
  magicoder  : flat  — instruction/response (no system field)
  openhermes : list  — conversations[].{from, value}
                       from = "human"|"gpt"|"system"
                       top-level system_prompt may or may not be present
  tulu       : list  — messages[].{role, content}
                       role = "user"|"assistant"|"system"

Run:
    uv run python test_pipeline.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq

from think import (
    TARGET_ROLES,
    _enforce_message_schema,
    _iter_input_batches,
    _load_converter,
    _validate_generated_script,
    profile_dataset,
    run_transform_script,
    validate_output_file,
)
import think

# Mock iter_input_batches to only process the first batch, slicing the size for speed.
_original_iter = _iter_input_batches
def _mock_iter_input_batches(local_path: str, chunk_size: int):
    # Only yield ONE batch of chunk_size to simulate file processing quickly during tests
    for batch in _original_iter(local_path, chunk_size):
        yield batch
        break
think._iter_input_batches = _mock_iter_input_batches

PASS = "✓"
FAIL = "✗"
results: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    icon = PASS if condition else FAIL
    msg = f"  {icon}  {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append(f"{icon} {label}")
    if not condition:
        raise AssertionError(f"FAILED: {label}  {detail}")


def section(title: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def run_converter(script: str, rows: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    """Validate + load + run converter on a list of rows. Return normalised messages."""
    _validate_generated_script(script)
    fn = _load_converter(script)
    output = []
    for i, row in enumerate(rows):
        raw = fn(row)
        msgs = _enforce_message_schema(raw, i)
        output.append(msgs)
    return output


def msgs_to_str(msgs: List[Dict[str, str]]) -> str:
    return "\n".join(f"    {json.dumps(m, ensure_ascii=False)}" for m in msgs)


# ══════════════════════════════════════════════════════════════════════
# DATASET 1 — MetaMathQA
# Schema: flat  |  columns: type, query, original_question, response
# ══════════════════════════════════════════════════════════════════════

section("METAMATH — flat dataset, no system field")

METAMATH_SCRIPT = textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    user_text      = str(record.get("query", "") or "")
    assistant_text = str(record.get("response", "") or "")
    return [
        {"role": "system",    "content": ""},
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": assistant_text},
        {"role": "tool",      "content": ""},
    ]
""").strip()

METAMATH_ROWS = [
    {
        "type": "MATH_AnsAug",
        "query": "What is 2+2?",
        "original_question": "What is 2+2?",
        "response": "4",
    },
    {
        "type": "GSM_SV",
        "query": "Find x if x^2 = 9",
        "original_question": "Find x",
        "response": "x = 3",
    },
    # Edge: None values
    {
        "type": "MATH_AnsAug",
        "query": None,
        "original_question": None,
        "response": None,
    },
]

mm_output = run_converter(METAMATH_SCRIPT, METAMATH_ROWS)

for i, msgs in enumerate(mm_output):
    check(f"metamath row {i}: returns exactly 4 messages", len(msgs) == 4)
    check(f"metamath row {i}: roles in correct order",
          [m["role"] for m in msgs] == list(TARGET_ROLES))
    check(f"metamath row {i}: system is empty (no system field in dataset)",
          msgs[0]["content"] == "",
          f"got: {msgs[0]['content']!r}")
    check(f"metamath row {i}: tool is empty",
          msgs[3]["content"] == "")
    check(f"metamath row {i}: all content is str",
          all(isinstance(m["content"], str) for m in msgs))

# Row 0 specific
check("metamath: user content copied verbatim", mm_output[0][1]["content"] == "What is 2+2?")
check("metamath: assistant content copied verbatim", mm_output[0][2]["content"] == "4")
# Row 2 — None values → empty strings
check("metamath: None query → empty string", mm_output[2][1]["content"] == "")
check("metamath: None response → empty string", mm_output[2][2]["content"] == "")

print(f"\n  Sample output row 0:")
print(msgs_to_str(mm_output[0]))

# Full file transform + validate
section("METAMATH — full file transform (200 rows)")
stats = run_transform_script(METAMATH_SCRIPT, "data/metamath.parquet", "output/_test_metamath.jsonl", chunk_size=200)
check("metamath: rows_written matches parquet row_count sample", stats["rows_written"] > 0)
check("metamath: output file exists", Path(stats["output_path"]).exists())
vr = validate_output_file("output/_test_metamath.jsonl", sample_groups=50)
check("metamath: validate_output_file passes", vr["ok"] is True)
check("metamath: sampled 50 groups", vr["groups_sampled"] == 50)
Path("output/_test_metamath.jsonl").unlink()


# ══════════════════════════════════════════════════════════════════════
# DATASET 2 — Magicoder
# Schema: flat  |  columns: instruction, response
# ══════════════════════════════════════════════════════════════════════

section("MAGICODER — flat dataset, no system field")

MAGICODER_SCRIPT = textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    user_text      = str(record.get("instruction", "") or "")
    assistant_text = str(record.get("response", "") or "")
    return [
        {"role": "system",    "content": ""},
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": assistant_text},
        {"role": "tool",      "content": ""},
    ]
""").strip()

MAGICODER_ROWS = [
    {
        "instruction": "Write a Python function that reverses a list.",
        "response": "def reverse_list(lst): return lst[::-1]",
    },
    {
        "instruction": "Explain recursion.",
        "response": "Recursion is when a function calls itself.",
    },
    # Edge: missing keys
    {},
]

mc_output = run_converter(MAGICODER_SCRIPT, MAGICODER_ROWS)

for i, msgs in enumerate(mc_output):
    check(f"magicoder row {i}: 4 messages", len(msgs) == 4)
    check(f"magicoder row {i}: roles correct", [m["role"] for m in msgs] == list(TARGET_ROLES))
    check(f"magicoder row {i}: system empty", msgs[0]["content"] == "")
    check(f"magicoder row {i}: tool empty", msgs[3]["content"] == "")

check("magicoder: instruction → user verbatim",
      mc_output[0][1]["content"] == "Write a Python function that reverses a list.")
check("magicoder: response → assistant verbatim",
      mc_output[0][2]["content"] == "def reverse_list(lst): return lst[::-1]")
check("magicoder: missing keys → empty strings",
      mc_output[2][1]["content"] == "" and mc_output[2][2]["content"] == "")

print(f"\n  Sample output row 0:")
print(msgs_to_str(mc_output[0]))

section("MAGICODER — full file transform (200 rows)")
stats = run_transform_script(MAGICODER_SCRIPT, "data/magicoder.parquet", "output/_test_magicoder.jsonl", chunk_size=200)
check("magicoder: rows_written > 0", stats["rows_written"] > 0)
vr = validate_output_file("output/_test_magicoder.jsonl", sample_groups=50)
check("magicoder: validate passes", vr["ok"] is True)
Path("output/_test_magicoder.jsonl").unlink()


# ══════════════════════════════════════════════════════════════════════
# DATASET 3 — OpenHermes  (most complex)
# Schema: list  |  conversations[].{from, value}
#   from = "human" | "gpt" | "system"
#   top-level system_prompt: str | null (checked FIRST, falls back to list)
# ══════════════════════════════════════════════════════════════════════

section("OPENHERMES — message list, 'from'/'value' keys, human/gpt/system roles")

OPENHERMES_SCRIPT = textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    USER_ROLES      = {"human", "user", "Human", "USER"}
    ASSISTANT_ROLES = {"gpt",   "assistant", "GPT", "ASSISTANT"}
    SYSTEM_ROLES    = {"system", "System", "SYSTEM"}

    # System: prefer top-level system_prompt, else look in list for from=system
    system_text = str(record.get("system_prompt", "") or "")

    conversations = record.get("conversations") or []
    if not isinstance(conversations, list):
        conversations = []

    # Extract system from list if not already found at top-level
    if not system_text:
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            frm = str(msg.get("from", "") or "").strip()
            if frm in SYSTEM_ROLES:
                system_text = str(msg.get("value", "") or "")
                break

    # Extract last user->assistant pair
    last_user_text      = ""
    last_assistant_text = ""
    pending_user        = ""

    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        frm  = str(msg.get("from", "") or "").strip()
        val  = str(msg.get("value", "") or "")
        if frm in USER_ROLES:
            pending_user = val
        elif frm in ASSISTANT_ROLES and pending_user:
            last_user_text      = pending_user
            last_assistant_text = val
            pending_user        = ""

    return [
        {"role": "system",    "content": system_text},
        {"role": "user",      "content": last_user_text},
        {"role": "assistant", "content": last_assistant_text},
        {"role": "tool",      "content": ""},
    ]
""").strip()

# Case A: minimal 2-turn, no system_prompt, no system in list
OH_ROW_A = {
    "system_prompt": None,
    "conversations": [
        {"from": "human", "value": "How many days in February?",  "weight": None},
        {"from": "gpt",   "value": "28 days (29 in a leap year)", "weight": None},
    ],
    "skip_prompt_formatting": False,
    "category": "orca",
    "source": "airoboros2.2",
}

# Case B: top-level system_prompt present
OH_ROW_B = {
    "system_prompt": "Engaging",
    "conversations": [
        {"from": "human", "value": "Discuss income redistribution.", "weight": None},
        {"from": "gpt",   "value": "Income redistribution affects Pareto efficiency…", "weight": None},
    ],
}

# Case C: system message INSIDE the list, no top-level system_prompt
OH_ROW_C = {
    "system_prompt": None,
    "conversations": [
        {"from": "system", "value": "You are to take on the role of: Donovan"},
        {"from": "human",  "value": "What is Madagascar known for?"},
        {"from": "gpt",    "value": "Ah, you're talking about lemurs and baobabs!"},
    ],
}

# Case D: multi-turn — should extract LAST user->gpt pair
OH_ROW_D = {
    "system_prompt": None,
    "conversations": [
        {"from": "human", "value": "First question"},
        {"from": "gpt",   "value": "First answer"},
        {"from": "human", "value": "Follow-up question"},
        {"from": "gpt",   "value": "Follow-up answer"},
    ],
}

# Case E: edge — empty conversations
OH_ROW_E = {
    "system_prompt": None,
    "conversations": [],
}

# Case F: edge — None conversations
OH_ROW_F = {
    "system_prompt": None,
    "conversations": None,
}

oh_output = run_converter(OPENHERMES_SCRIPT, [OH_ROW_A, OH_ROW_B, OH_ROW_C, OH_ROW_D, OH_ROW_E, OH_ROW_F])

for i, msgs in enumerate(oh_output):
    check(f"openhermes row {i}: 4 messages", len(msgs) == 4)
    check(f"openhermes row {i}: roles correct", [m["role"] for m in msgs] == list(TARGET_ROLES))
    check(f"openhermes row {i}: all content is str", all(isinstance(m["content"], str) for m in msgs))

# Case A: no system anywhere
check("openhermes A: system empty (no system_prompt, no system turn)",
      oh_output[0][0]["content"] == "", f"got: {oh_output[0][0]['content']!r}")
check("openhermes A: user extracted from 'human' turn",
      oh_output[0][1]["content"] == "How many days in February?")
check("openhermes A: assistant extracted from 'gpt' turn",
      oh_output[0][2]["content"] == "28 days (29 in a leap year)")

# Case B: top-level system_prompt
check("openhermes B: top-level system_prompt copied into system",
      oh_output[1][0]["content"] == "Engaging")
check("openhermes B: user correct", oh_output[1][1]["content"] == "Discuss income redistribution.")

# Case C: system from inside list
check("openhermes C: system extracted from list (from=system)",
      oh_output[2][0]["content"] == "You are to take on the role of: Donovan")
check("openhermes C: user correct", oh_output[2][1]["content"] == "What is Madagascar known for?")
check("openhermes C: assistant correct",
      oh_output[2][2]["content"] == "Ah, you're talking about lemurs and baobabs!")

# Case D: multi-turn → LAST pair
check("openhermes D: last user turn extracted (not first)",
      oh_output[3][1]["content"] == "Follow-up question",
      f"got: {oh_output[3][1]['content']!r}")
check("openhermes D: last assistant turn extracted",
      oh_output[3][2]["content"] == "Follow-up answer")

# Case E/F: empty/None → all empty strings
check("openhermes E: empty list → user empty", oh_output[4][1]["content"] == "")
check("openhermes F: None conversations → user empty", oh_output[5][1]["content"] == "")

print(f"\n  Sample output row A (no system):")
print(msgs_to_str(oh_output[0]))
print(f"\n  Sample output row B (top-level system_prompt):")
print(msgs_to_str(oh_output[1]))
print(f"\n  Sample output row C (system in list):")
print(msgs_to_str(oh_output[2]))
print(f"\n  Sample output row D (multi-turn → last pair):")
print(msgs_to_str(oh_output[3]))

section("OPENHERMES — full file transform (200 rows)")
stats = run_transform_script(OPENHERMES_SCRIPT, "data/openhermes.parquet", "output/_test_openhermes.jsonl", chunk_size=200)
check("openhermes: rows_written > 0", stats["rows_written"] > 0)
vr = validate_output_file("output/_test_openhermes.jsonl", sample_groups=50)
check("openhermes: validate passes", vr["ok"] is True)
Path("output/_test_openhermes.jsonl").unlink()


# ══════════════════════════════════════════════════════════════════════
# DATASET 4 — Tulu
# Schema: list  |  messages[].{role, content}
#   role = "user" | "assistant" | "system"
# ══════════════════════════════════════════════════════════════════════

section("TULU — message list, 'role'/'content' keys, user/assistant/system")

TULU_SCRIPT = textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    USER_ROLES      = {"user",      "USER",      "human"}
    ASSISTANT_ROLES = {"assistant", "ASSISTANT", "gpt"}
    SYSTEM_ROLES    = {"system",    "SYSTEM"}

    messages = record.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    system_text         = ""
    last_user_text      = ""
    last_assistant_text = ""
    pending_user        = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role    = str(msg.get("role", "") or "").strip()
        content = str(msg.get("content", "") or "")
        if role in SYSTEM_ROLES:
            system_text = content
        elif role in USER_ROLES:
            pending_user = content
        elif role in ASSISTANT_ROLES and pending_user:
            last_user_text      = pending_user
            last_assistant_text = content
            pending_user        = ""

    return [
        {"role": "system",    "content": system_text},
        {"role": "user",      "content": last_user_text},
        {"role": "assistant", "content": last_assistant_text},
        {"role": "tool",      "content": ""},
    ]
""").strip()

# Case A: single user→assistant, no system (flan_v2 style)
TU_ROW_A = {
    "dataset": "flan_v2",
    "id": "flan_v2_0",
    "messages": [
        {"role": "user",      "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ],
}

# Case B: system + user + assistant (open_orca style)
TU_ROW_B = {
    "dataset": "open_orca",
    "id": "open_orca_1",
    "messages": [
        {"role": "system",    "content": "You are an AI assistant that follows instruction extremely well."},
        {"role": "user",      "content": "Summarise the recycling process."},
        {"role": "assistant", "content": "Cans are collected, melted, and reformed."},
    ],
}

# Case C: multi-turn without system → last pair
TU_ROW_C = {
    "dataset": "sharegpt",
    "id": "sg_1",
    "messages": [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user",      "content": "What is recursion?"},
        {"role": "assistant", "content": "A function calling itself."},
    ],
}

# Case D: multi-turn WITH system → last pair + system
TU_ROW_D = {
    "dataset": "sharegpt",
    "id": "sg_2",
    "messages": [
        {"role": "system",    "content": "You are a coding assistant."},
        {"role": "user",      "content": "Write hello world in Python."},
        {"role": "assistant", "content": "print('Hello, world!')"},
        {"role": "user",      "content": "Now in Java."},
        {"role": "assistant", "content": "System.out.println(\"Hello, world!\");"},
    ],
}

# Case E: edge — empty messages list
TU_ROW_E = {"dataset": "x", "id": "x_0", "messages": []}

# Case F: edge — None messages
TU_ROW_F = {"dataset": "x", "id": "x_1", "messages": None}

tulu_output = run_converter(TULU_SCRIPT, [TU_ROW_A, TU_ROW_B, TU_ROW_C, TU_ROW_D, TU_ROW_E, TU_ROW_F])

for i, msgs in enumerate(tulu_output):
    check(f"tulu row {i}: 4 messages", len(msgs) == 4)
    check(f"tulu row {i}: roles correct", [m["role"] for m in msgs] == list(TARGET_ROLES))
    check(f"tulu row {i}: all content is str", all(isinstance(m["content"], str) for m in msgs))

# Case A: no system
check("tulu A: system empty (no system role)", tulu_output[0][0]["content"] == "")
check("tulu A: user extracted", tulu_output[0][1]["content"] == "What is the capital of France?")
check("tulu A: assistant extracted", tulu_output[0][2]["content"] == "Paris.")
check("tulu A: tool empty", tulu_output[0][3]["content"] == "")

# Case B: system role exists
check("tulu B: system content copied verbatim",
      tulu_output[1][0]["content"] == "You are an AI assistant that follows instruction extremely well.")
check("tulu B: user correct", tulu_output[1][1]["content"] == "Summarise the recycling process.")
check("tulu B: assistant correct", tulu_output[1][2]["content"] == "Cans are collected, melted, and reformed.")

# Case C: multi-turn, last pair
check("tulu C: last user turn (not first)",
      tulu_output[2][1]["content"] == "What is recursion?",
      f"got: {tulu_output[2][1]['content']!r}")
check("tulu C: last assistant turn",
      tulu_output[2][2]["content"] == "A function calling itself.")
check("tulu C: system still empty", tulu_output[2][0]["content"] == "")

# Case D: multi-turn + system
check("tulu D: system from system role",
      tulu_output[3][0]["content"] == "You are a coding assistant.")
check("tulu D: last user turn",
      tulu_output[3][1]["content"] == "Now in Java.",
      f"got: {tulu_output[3][1]['content']!r}")
check("tulu D: last assistant turn",
      tulu_output[3][2]["content"] == "System.out.println(\"Hello, world!\");")

# Case E/F: edge
check("tulu E: empty messages → all empty", tulu_output[4][0]["content"] == "" and tulu_output[4][1]["content"] == "")
check("tulu F: None messages → all empty",  tulu_output[5][0]["content"] == "" and tulu_output[5][1]["content"] == "")

print(f"\n  Sample output row A (no system):")
print(msgs_to_str(tulu_output[0]))
print(f"\n  Sample output row B (system in list):")
print(msgs_to_str(tulu_output[1]))
print(f"\n  Sample output row D (multi-turn + system):")
print(msgs_to_str(tulu_output[3]))

section("TULU — full file transform (200 rows)")
stats = run_transform_script(TULU_SCRIPT, "data/tulu.parquet", "output/_test_tulu.jsonl", chunk_size=200)
check("tulu: rows_written > 0", stats["rows_written"] > 0)
vr = validate_output_file("output/_test_tulu.jsonl", sample_groups=50)
check("tulu: validate passes", vr["ok"] is True)
Path("output/_test_tulu.jsonl").unlink()


# ══════════════════════════════════════════════════════════════════════
# CROSS-CUTTING: schema enforcement tests
# (These test that the enforcer correctly REJECTS bad converter output)
# ══════════════════════════════════════════════════════════════════════

section("SCHEMA ENFORCEMENT — rejects invalid converter output")

def expect_error(label: str, script: str, row: Dict) -> None:
    try:
        _validate_generated_script(script)
        fn = _load_converter(script)
        raw = fn(row)
        _enforce_message_schema(raw, 0)
        check(f"[should have raised] {label}", False, "No error was raised!")
    except (ValueError, AssertionError) as exc:
        check(f"correctly rejected: {label}", True, f"Caught: {type(exc).__name__}: {exc}")

SAMPLE_ROW = {"query": "test", "response": "answer"}

# Wrong number of messages
expect_error("3 messages instead of 4", textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    return [
        {"role": "system",    "content": ""},
        {"role": "user",      "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
""").strip(), SAMPLE_ROW)

# Wrong role order (assistant before user)
expect_error("wrong role order", textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    return [
        {"role": "system",    "content": ""},
        {"role": "assistant", "content": "a"},
        {"role": "user",      "content": "q"},
        {"role": "tool",      "content": ""},
    ]
""").strip(), SAMPLE_ROW)

# Wrong role name
expect_error("unknown role name", textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    return [
        {"role": "sys",       "content": ""},
        {"role": "user",      "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "tool",      "content": ""},
    ]
""").strip(), SAMPLE_ROW)

# Not a list
expect_error("returns dict instead of list", textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    return {"role": "user", "content": "oops"}
""").strip(), SAMPLE_ROW)

# ══════════════════════════════════════════════════════════════════════
# AST SAFETY — rejects dangerous scripts
# ══════════════════════════════════════════════════════════════════════

section("AST SAFETY — rejects unsafe imports and calls")

def expect_validation_error(label: str, script: str) -> None:
    try:
        _validate_generated_script(script)
        check(f"[should have raised] {label}", False, "No error was raised!")
    except ValueError as exc:
        check(f"correctly blocked: {label}", True, f"{exc}")

expect_validation_error("import os", textwrap.dedent("""
import os
def convert_record(record: dict) -> list[dict]:
    return []
""").strip())

expect_validation_error("import subprocess", textwrap.dedent("""
import subprocess
def convert_record(record: dict) -> list[dict]:
    return []
""").strip())

expect_validation_error("eval() call", textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    return eval("[]")
""").strip())

expect_validation_error("open() call", textwrap.dedent("""
def convert_record(record: dict) -> list[dict]:
    open("/etc/passwd")
    return []
""").strip())

expect_validation_error("os.remove via attribute", textwrap.dedent("""
import os
def convert_record(record: dict) -> list[dict]:
    os.remove("/tmp/x")
    return []
""").strip())


# ══════════════════════════════════════════════════════════════════════
# OUTPUT FORMAT — verify JSONL file structure on disk
# ══════════════════════════════════════════════════════════════════════

section("OUTPUT FORMAT — 4 lines per sample, blank separator, correct JSON")

run_transform_script(METAMATH_SCRIPT, "data/metamath.parquet", "output/_fmt_test.jsonl", chunk_size=10)

path = Path("output/_fmt_test.jsonl")
with path.open(encoding="utf-8") as fh:
    raw_lines = fh.readlines()

# Group into blocks separated by blank lines
blocks = []
current: list = []
for line in raw_lines:
    stripped = line.strip()
    if stripped:
        current.append(stripped)
    else:
        if current:
            blocks.append(current)
            current = []
if current:
    blocks.append(current)

check("output: every block has exactly 4 lines",
      all(len(b) == 4 for b in blocks[:50]),
      f"block sizes: {[len(b) for b in blocks[:10]]}")

check("output: each line is valid JSON",
      all(json.loads(line) is not None for block in blocks[:50] for line in block))

check("output: first line of each block is system",
      all(json.loads(b[0])["role"] == "system" for b in blocks[:50]))

check("output: second line is user",
      all(json.loads(b[1])["role"] == "user" for b in blocks[:50]))

check("output: third line is assistant",
      all(json.loads(b[2])["role"] == "assistant" for b in blocks[:50]))

check("output: fourth line is tool",
      all(json.loads(b[3])["role"] == "tool" for b in blocks[:50]))

check("output: all content values are strings",
      all(isinstance(json.loads(line)["content"], str)
          for block in blocks[:50] for line in block))

check("output: metamath system is always empty (no system field in source)",
      all(json.loads(b[0])["content"] == "" for b in blocks[:50]))

print(f"\n  First 2 raw blocks from file:")
for b in blocks[:2]:
    for line in b:
        print(f"    {line}")
    print()

path.unlink()


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

section("SUMMARY")
passed = [r for r in results if r.startswith(PASS)]
failed = [r for r in results if r.startswith(FAIL)]
print(f"\n  Total : {len(results)}")
print(f"  Passed: {len(passed)}")
print(f"  Failed: {len(failed)}")
if failed:
    print("\n  FAILURES:")
    for f in failed:
        print(f"    {f}")
    raise SystemExit(1)
else:
    print(f"\n  {'='*55}")
    print(f"  ALL {len(passed)} TESTS PASSED ✓")
    print(f"  {'='*55}")
