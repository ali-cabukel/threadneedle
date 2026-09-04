import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def extract_citations(tool_output: str) -> list[dict]:
    """Pull citation objects out of a tool's JSON payload, if present."""
    try:
        parsed = json.loads(tool_output)
    except (TypeError, json.JSONDecodeError):
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get("citations"), list):
        return [c for c in parsed["citations"] if isinstance(c, dict)]
    return []


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {None, "text"}:
                parts.append(str(block.get("text") or ""))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content)
