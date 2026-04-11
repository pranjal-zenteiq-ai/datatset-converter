from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    import pandas as pd
    import pyarrow.parquet as pq
    from dotenv import load_dotenv
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

TARGET_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")

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

MESSAGE_FIELD_CANDIDATES: tuple[str, ...] = (
    "messages",
    "message",
    "conversations",
    "conversation",
    "dialogue",
    "dialog",
    "turns",
    "history",
)

ROLE_KEY_CANDIDATES: tuple[str, ...] = (
    "role",
    "from",
    "speaker",
    "author",
    "sender",
    "type",
)

CONTENT_KEY_CANDIDATES: tuple[str, ...] = (
    "content",
    "value",
    "text",
    "message",
    "body",
    "parts",
)

ROLE_ALIASES: dict[str, set[str]] = {
    "system": {"system", "sys", "instruction", "prompt", "context"},
    "user": {"user", "human", "client", "customer", "question", "input"},
    "assistant": {"assistant", "gpt", "bot", "ai", "model", "completion", "answer", "response"},
    "tool": {"tool", "function", "tool_call", "tool_result", "observation", "plugin"},
}

_JSON_CONTAINER_KEYS: tuple[str, ...] = MESSAGE_FIELD_CANDIDATES


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python|json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
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


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _serialize_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(_to_jsonable(value), ensure_ascii=False)
    except Exception:
        return str(value)


def _maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[:1] not in {"[", "{"}:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _normalize_role(value: Any, role_aliases: Optional[Dict[str, Iterable[str]]] = None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    aliases = role_aliases or ROLE_ALIASES
    for canonical, names in aliases.items():
        norm_names = {str(name).strip().lower() for name in names}
        if text == canonical or text in norm_names:
            return canonical
    return None


def _first_present_key(keys: Iterable[str], mapping: Dict[str, Any]) -> Optional[str]:
    for key in keys:
        if key in mapping:
            return key
    return None


def _lookup_key_case_insensitive(mapping: Dict[str, Any], target: str) -> Optional[str]:
    target_norm = str(target).strip().lower()
    for key in mapping.keys():
        if str(key).strip().lower() == target_norm:
            return key
    return None


def _is_message_like_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    lowered = {str(k).strip().lower() for k in obj.keys()}
    if lowered.intersection(set(ROLE_KEY_CANDIDATES)) and lowered.intersection(set(CONTENT_KEY_CANDIDATES)):
        return True
    if {"role", "content"}.issubset(lowered):
        return True
    return False


def _unwrap_message_container(value: Any) -> Optional[List[Any]]:
    value = _maybe_parse_json_string(value)

    if isinstance(value, list):
        if any(isinstance(item, dict) for item in value):
            return value
        return None

    if isinstance(value, dict):
        for key in _JSON_CONTAINER_KEYS:
            real_key = _lookup_key_case_insensitive(value, key)
            if real_key is not None and isinstance(value[real_key], list):
                if any(isinstance(item, dict) for item in value[real_key]):
                    return value[real_key]
        if _is_message_like_dict(value):
            return [value]

    return None


def _infer_message_keys(items: List[Dict[str, Any]], spec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    role_key = spec.get("role_key")
    content_key = spec.get("content_key")

    if isinstance(role_key, str) and role_key.strip():
        role_key = role_key.strip()
    else:
        role_key = None

    if isinstance(content_key, str) and content_key.strip():
        content_key = content_key.strip()
    else:
        content_key = None

    if role_key is None:
        for candidate in ROLE_KEY_CANDIDATES:
            if any(_lookup_key_case_insensitive(item, candidate) is not None for item in items):
                role_key = candidate
                break

    if content_key is None:
        for candidate in CONTENT_KEY_CANDIDATES:
            if any(_lookup_key_case_insensitive(item, candidate) is not None for item in items):
                content_key = candidate
                break

    return role_key, content_key


def _normalize_aliases(spec: Dict[str, Any]) -> Dict[str, set[str]]:
    aliases: Dict[str, set[str]] = {role: set(values) for role, values in ROLE_ALIASES.items()}
    raw_aliases = spec.get("role_aliases")
    if isinstance(raw_aliases, dict):
        for canonical, values in raw_aliases.items():
            if canonical is None:
                continue
            canonical_name = str(canonical).strip().lower()
            if canonical_name not in TARGET_ROLES:
                continue
            if isinstance(values, (list, tuple, set)):
                aliases.setdefault(canonical_name, set()).update(
                    str(v).strip().lower() for v in values if str(v).strip()
                )
            elif values is not None:
                aliases.setdefault(canonical_name, set()).add(str(values).strip().lower())
    return aliases


def _ordered_record_keys(record: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    preferred = spec.get("ordered_top_level_fields")
    if isinstance(preferred, list) and preferred:
        seen = set()
        ordered: List[str] = []
        for key in preferred:
            key = str(key)
            if key in record and key not in seen:
                ordered.append(key)
                seen.add(key)
        for key in record.keys():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
        return ordered
    return list(record.keys())


def _extract_messages_from_list(
    items: List[Any],
    spec: Dict[str, Any],
    row_index: int,
    source_field: str,
) -> List[Dict[str, str]]:
    alias_map = _normalize_aliases(spec)
    items_dict = [item for item in items if isinstance(item, dict)]
    if not items_dict:
        raise ValueError(f"Row {row_index}: field '{source_field}' did not contain message dictionaries")

    role_key, content_key = _infer_message_keys(items_dict, spec)
    if role_key is None or content_key is None:
        raise ValueError(
            f"Row {row_index}: could not infer role/content keys for message list in field '{source_field}'"
        )

    messages: List[Dict[str, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Row {row_index}: item {idx} in '{source_field}' is not a dict"
            )

        real_role_key = _lookup_key_case_insensitive(item, role_key)
        real_content_key = _lookup_key_case_insensitive(item, content_key)
        if real_role_key is None:
            raise ValueError(
                f"Row {row_index}: item {idx} in '{source_field}' is missing role key '{role_key}'"
            )
        if real_content_key is None:
            raise ValueError(
                f"Row {row_index}: item {idx} in '{source_field}' is missing content key '{content_key}'"
            )

        role = _normalize_role(item.get(real_role_key), alias_map)
        if role is None:
            raise ValueError(
                f"Row {row_index}: item {idx} in '{source_field}' has unknown role value {item.get(real_role_key)!r}"
            )
        if role not in TARGET_ROLES:
            raise ValueError(
                f"Row {row_index}: item {idx} in '{source_field}' normalized to unsupported role '{role}'"
            )

        content_value = item.get(real_content_key)
        if content_value is None:
            raise ValueError(
                f"Row {row_index}: item {idx} in '{source_field}' has null content"
            )

        messages.append({"role": role, "content": _serialize_content(content_value)})

    return messages


def _extract_messages_from_record(record: Dict[str, Any], spec: Dict[str, Any], row_index: int) -> List[Dict[str, str]]:
    field_to_role = spec.get("field_to_role")
    if not isinstance(field_to_role, dict):
        field_to_role = {}

    normalized_field_to_role: Dict[str, str] = {}
    for field, role in field_to_role.items():
        if field is None or role is None:
            continue
        normalized_field_to_role[str(field)] = str(role).strip().lower()

    conversation_candidates = spec.get("conversation_field_candidates")
    if isinstance(conversation_candidates, list):
        conversation_candidates = [str(x) for x in conversation_candidates if str(x).strip()]
    else:
        conversation_candidates = []

    ordered_keys = _ordered_record_keys(record, spec)
    candidate_keys = list(dict.fromkeys([*conversation_candidates, *ordered_keys]))

    messages: List[Dict[str, str]] = []
    used_any = False

    for field in candidate_keys:
        if field not in record:
            continue

        raw_value = record[field]
        if raw_value is None:
            if field in normalized_field_to_role or field in conversation_candidates:
                raise ValueError(f"Row {row_index}: field '{field}' is null")
            continue

        parsed_value = _maybe_parse_json_string(raw_value)
        nested_messages = _unwrap_message_container(parsed_value)
        if nested_messages is not None:
            used_any = True
            messages.extend(_extract_messages_from_list(nested_messages, spec, row_index, field))
            continue

        mapped_role = normalized_field_to_role.get(field)
        if mapped_role is not None:
            used_any = True
            if mapped_role not in TARGET_ROLES:
                raise ValueError(
                    f"Row {row_index}: field '{field}' mapped to unsupported role '{mapped_role}'"
                )
            messages.append({"role": mapped_role, "content": _serialize_content(raw_value)})
            continue

    if not used_any or not messages:
        raise ValueError(f"Row {row_index}: no convertible conversation data found")

    return messages


def _preview_parquet(path: str, n: int = 5) -> Dict[str, Any]:
    pf = pq.ParquetFile(path)
    head_rows: List[Dict[str, Any]] = []
    for batch in pf.iter_batches(batch_size=n):
        head_rows = batch.to_pylist()
        break

    schema = pf.schema_arrow
    return {
        "columns": list(schema.names),
        "dtypes": {schema.names[i]: str(schema.types[i]) for i in range(len(schema.names))},
        "head": [_to_jsonable(row) for row in head_rows],
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


@activity.defn
def profile_dataset(local_path: str) -> Dict[str, Any]:
    if not Path(local_path).exists():
        raise FileNotFoundError(f"Input file not found: {local_path}")
    if local_path.endswith(".parquet"):
        return _preview_parquet(local_path, n=5)
    if local_path.endswith(".csv"):
        return _preview_csv(local_path, n=5)
    raise ValueError(f"Unsupported input format: {local_path}")


_PLAN_PROMPT = """You are a dataset schema analyst.

Your job is to infer how to convert each source row into one normalized conversation array.
Do not invent any text. Do not write code. Return only valid JSON.

Normalized output shape per row:
[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"..."},
  {"role":"tool","content":"..."}
]

Important rules:
- The output array for a row may contain any subset of those roles.
- Only include roles whose data is actually present in the source row.
- Preserve the original turn order inside any message list.
- If a content field is not a plain string, runtime will JSON-serialize the existing value.
- Extra non-conversation columns should be ignored.
- If a conversation is stored as a list of message dicts, identify the field name, the role key, and the content key from the sample values.
- If conversation data is spread across top-level columns, identify exact field-to-role mappings from the sample values, not from column names alone.

Return exactly one JSON object with these keys:
{
  "confirmed": true,
  "strategy": "messages_list" | "flat_columns" | "mixed",
  "conversation_field_candidates": [],
  "role_key": null,
  "content_key": null,
  "role_aliases": {
    "system": [],
    "user": [],
    "assistant": [],
    "tool": []
  },
  "field_to_role": {},
  "ordered_top_level_fields": [],
  "notes": []
}

Dataset name: {dataset_name}

Profile (schema + 5 sample rows):
{profile_json}
"""


@activity.defn
def kimi_generate_plan(dataset_name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    client = _build_llm(max_tokens=8192)
    prompt = _PLAN_PROMPT.format(
        dataset_name=dataset_name,
        profile_json=json.dumps(profile, ensure_ascii=False, indent=2),
    )
    response = client.invoke(prompt)
    plan = _extract_json_object(response.content or "")
    if not plan.get("confirmed", False):
        raise ValueError("Kimi did not confirm the plan — check model response")
    return plan


def _derive_error_path(output_path: str) -> str:
    path = Path(output_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}.errors.jsonl"))
    return str(path.with_name(f"{path.name}.errors.jsonl"))


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


def _write_error_record(
    fh,
    row_index: int,
    error: Exception,
    record: Dict[str, Any],
) -> None:
    payload = {
        "row_index": row_index,
        "error": str(error),
        "record": _to_jsonable(record),
    }
    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


@activity.defn
def run_transform_script(
    plan: Dict[str, Any],
    local_path: str,
    output_path: str,
    error_path: Optional[str] = None,
    chunk_size: int = 2000,
) -> Dict[str, Any]:
    if error_path is None or not str(error_path).strip():
        error_path = _derive_error_path(output_path)

    input_file = Path(local_path)
    output_file = Path(output_path)
    error_file = Path(error_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written_rows = 0
    errored_rows = 0

    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=output_file.parent,
        suffix=".jsonl",
        encoding="utf-8",
    ) as tmp_out, tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=error_file.parent,
        suffix=".jsonl",
        encoding="utf-8",
    ) as tmp_err:
        tmp_out_path = Path(tmp_out.name)
        tmp_err_path = Path(tmp_err.name)

        try:
            for batch in _iter_input_batches(str(input_file), chunk_size):
                for record in batch:
                    try:
                        converted = _extract_messages_from_record(record, plan, total_rows)
                        if not converted:
                            raise ValueError("no messages extracted")

                        tmp_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        written_rows += 1
                    except Exception as exc:
                        errored_rows += 1
                        _write_error_record(tmp_err, total_rows, exc, record)

                    total_rows += 1

            tmp_out.flush()
            tmp_err.flush()
            os.fsync(tmp_out.fileno())
            os.fsync(tmp_err.fileno())

        except Exception:
            try:
                tmp_out_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                tmp_err_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    tmp_out_path.replace(output_file)
    tmp_err_path.replace(error_file)

    return {
        "output_path": str(output_file),
        "error_path": str(error_file),
        "rows_seen": total_rows,
        "rows_written": written_rows,
        "rows_errored": errored_rows,
    }


def _validate_message_array(item: Any, row_index: int) -> None:
    if not isinstance(item, list):
        raise ValueError(f"Row {row_index}: each line must be a JSON array")
    if not item:
        raise ValueError(f"Row {row_index}: empty arrays are not allowed")
    for msg_index, msg in enumerate(item):
        if not isinstance(msg, dict):
            raise ValueError(
                f"Row {row_index}, message {msg_index}: expected dict, got {type(msg).__name__}"
            )
        role = str(msg.get("role", "")).strip()
        if role not in TARGET_ROLES:
            raise ValueError(
                f"Row {row_index}, message {msg_index}: invalid role '{role}'"
            )
        if "content" not in msg:
            raise ValueError(
                f"Row {row_index}, message {msg_index}: missing content field"
            )
        content = msg.get("content")
        if not isinstance(content, str):
            raise ValueError(
                f"Row {row_index}, message {msg_index}: content must be a string"
            )


@activity.defn
def validate_output_file(output_path: str, sample_rows: int = 25) -> Dict[str, Any]:
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    validated = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if validated >= sample_rows:
                break
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            _validate_message_array(parsed, validated)
            validated += 1

    return {
        "ok": True,
        "rows_sampled": validated,
        "path": str(path),
    }


@workflow.defn
class DatasetSFTWorkflow:
    @workflow.run
    async def run(
        self,
        dataset_name: str,
        local_path: str,
        output_path: str,
        error_path: Optional[str] = None,
        chunk_size: int = 2000,
    ) -> str:
        profile: Dict[str, Any] = await workflow.execute_activity(
            profile_dataset,
            local_path,
            start_to_close_timeout=workflow.timedelta(seconds=300),
        )

        plan: Dict[str, Any] = await workflow.execute_activity(
            kimi_generate_plan,
            args=[dataset_name, profile],
            start_to_close_timeout=workflow.timedelta(seconds=900),
        )

        await workflow.execute_activity(
            run_transform_script,
            args=[plan, local_path, output_path, error_path, chunk_size],
            start_to_close_timeout=workflow.timedelta(hours=4),
        )

        await workflow.execute_activity(
            validate_output_file,
            args=[output_path],
            start_to_close_timeout=workflow.timedelta(seconds=300),
        )

        return output_path
