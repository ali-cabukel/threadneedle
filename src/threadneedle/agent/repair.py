"""Repair checkpointed threads that were cut off mid tool-call.

OpenAI requires every `AIMessage.tool_calls` id to be followed *immediately*
by matching `ToolMessage`s. If a tool crashes, LangGraph can persist the
assistant tool_calls and then append later HumanMessages (retries) in front of
any filler ToolMessages. Appending fillers at the end does not satisfy the API.
We rewrite the list so replies sit directly after the tool_calls message.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

log = logging.getLogger(__name__)

_INTERRUPT_REPLY = (
    "Tool call was interrupted before it finished. Retry the tool if the answer still needs it."
)


def _tool_call_ids(message: Any) -> list[str]:
    calls = getattr(message, "tool_calls", None) or []
    if not calls and isinstance(message, dict):
        calls = message.get("tool_calls") or []
    ids: list[str] = []
    for call in calls:
        if isinstance(call, dict):
            ids.append(str(call.get("id") or ""))
        else:
            ids.append(str(getattr(call, "id", "") or ""))
    return [item for item in ids if item]


def _is_tool_message(message: Any) -> bool:
    if isinstance(message, ToolMessage):
        return True
    if isinstance(message, dict):
        return message.get("role") == "tool" or message.get("type") == "tool"
    return getattr(message, "type", None) == "tool"


def _tool_call_id(message: Any) -> str | None:
    tool_call_id = getattr(message, "tool_call_id", None)
    if isinstance(message, dict):
        tool_call_id = tool_call_id or message.get("tool_call_id")
    return str(tool_call_id) if tool_call_id else None


def _placeholder(call_id: str) -> ToolMessage:
    return ToolMessage(content=_INTERRUPT_REPLY, tool_call_id=call_id)


def dangling_tool_messages(messages: list[Any]) -> list[ToolMessage]:
    """Placeholder ToolMessages for unanswered ids, ignoring order.

    Prefer `repaired_history` when rewriting a thread: OpenAI cares about
    position, not only that the ids exist somewhere later in the list.
    """
    open_ids: list[str] = []
    for message in messages:
        if _is_tool_message(message):
            tool_call_id = _tool_call_id(message)
            if tool_call_id in open_ids:
                open_ids.remove(tool_call_id)
            continue
        ids = _tool_call_ids(message)
        if ids:
            open_ids.extend(ids)
    return [_placeholder(call_id) for call_id in open_ids]


def repaired_history(messages: list[Any]) -> list[Any]:
    """Return a copy of `messages` valid for OpenAI tool-calling.

    Existing ToolMessages are pulled forward to sit immediately after the
    assistant message that requested them. Missing ids get a placeholder.
    Duplicate ToolMessages left behind by an earlier append-at-end repair
    are dropped.
    """
    first_reply: dict[str, Any] = {}
    for message in messages:
        if not _is_tool_message(message):
            continue
        call_id = _tool_call_id(message)
        if call_id and call_id not in first_reply:
            first_reply[call_id] = message

    result: list[Any] = []
    emitted: set[str] = set()
    pending: list[str] = []

    def close_pending() -> None:
        nonlocal pending
        for call_id in pending:
            if call_id in emitted:
                continue
            existing = first_reply.get(call_id)
            result.append(existing if existing is not None else _placeholder(call_id))
            emitted.add(call_id)
        pending = []

    for message in messages:
        if _is_tool_message(message):
            call_id = _tool_call_id(message)
            if call_id and call_id in emitted:
                continue
            if call_id and call_id in pending:
                result.append(message)
                emitted.add(call_id)
                pending = [item for item in pending if item != call_id]
                continue
            continue

        ids = _tool_call_ids(message)
        if ids:
            close_pending()
            result.append(message)
            pending = list(ids)
            continue

        close_pending()
        result.append(message)

    close_pending()
    return result


def _history_sig(messages: list[Any]) -> list[tuple]:
    sig: list[tuple] = []
    for message in messages:
        kind = getattr(message, "type", None) or (
            message.get("type") if isinstance(message, dict) else type(message).__name__
        )
        sig.append(
            (
                kind,
                getattr(message, "id", None) if not isinstance(message, dict) else message.get("id"),
                _tool_call_id(message),
                tuple(_tool_call_ids(message)),
            )
        )
    return sig


def history_needs_repair(messages: list[Any]) -> bool:
    return _history_sig(messages) != _history_sig(repaired_history(messages))


def replace_messages_update(messages: list[Any]) -> dict[str, list]:
    """LangGraph state update that overwrites the messages channel."""
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]}


async def repair_thread(agent: Any, config: dict) -> int:
    """Rewrite the checkpoint so tool replies follow tool_calls. Returns 1 if rewritten."""
    try:
        state = await agent.aget_state(config)
    except Exception:
        log.exception("could not load checkpoint for repair")
        return 0
    values = getattr(state, "values", None) or {}
    messages: list[BaseMessage] = list(values.get("messages") or [])
    if not messages or not history_needs_repair(messages):
        return 0
    fixed = repaired_history(messages)
    log.warning(
        "rewriting poisoned tool history on thread %s (%d -> %d messages)",
        (config.get("configurable") or {}).get("thread_id"),
        len(messages),
        len(fixed),
    )
    payload = replace_messages_update(fixed)
    try:
        await agent.aupdate_state(config, payload, as_node="tools")
    except Exception:
        log.exception("aupdate_state(as_node='tools') failed; retrying without as_node")
        await agent.aupdate_state(config, payload)
    return 1
