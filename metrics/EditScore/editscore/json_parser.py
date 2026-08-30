import json
import re
from typing import Any, Optional


def _normalize_result(data: Any) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None

    reasoning = next(
        (
            data[key]
            for key in ("reasoning", "reason", "rationale")
            if isinstance(data.get(key), str)
        ),
        "",
    )
    value = data.get("score", [])
    if not isinstance(value, list):
        value = [value]
    scores = [
        float(score)
        for score in value
        if isinstance(score, (int, float, str))
    ]
    if not reasoning and not scores:
        return None
    return {"score": scores, "reasoning": reasoning}


def _fix_json(text: str) -> str:
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    single_quoted = text.replace("'", '"')
    try:
        json.loads(single_quoted)
        return single_quoted
    except json.JSONDecodeError:
        return re.sub(r'([\{\s,])(\w+)\s*:', r'\1"\2":', text)


def _repair_reasoning(text: str) -> str:
    pattern = re.compile(
        r'("reasoning"\s*:\s*")(.*?)(?="\s*[,}])',
        re.DOTALL,
    )
    return pattern.sub(
        lambda match: match.group(1) + match.group(2).replace('"', '\\"'),
        text,
    )


def _fallback(text: str) -> dict[str, Any]:
    score_match = re.search(
        r'["\']score["\']\s*:\s*(.*)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    score_text = score_match.group(1) if score_match else text
    scores = [
        float(value)
        for value in re.findall(r"[-+]?\d*\.?\d+", score_text)
    ]

    reason_match = re.search(
        r'["\']reasoning["\']\s*:\s*["\']?(.*?)["\']?\s*,\s*["\']score',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    reasoning = reason_match.group(1).strip() if reason_match else ""
    return {"score": scores, "reasoning": reasoning.replace('\\"', '"')}


def parse_vlm_output_to_dict(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {"score": [], "reasoning": ""}

    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    for transform in (lambda value: value, _fix_json, _repair_reasoning):
        try:
            result = _normalize_result(json.loads(transform(candidate)))
            if result is not None:
                return result
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return _fallback(candidate)
