"""FastAPI routes for chat (sync + SSE) and index stats."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)

from threadneedle.agent.repair import repair_thread
from threadneedle.api.schemas import ChatRequest, ChatResponse
from threadneedle.api.sse import extract_citations, message_text, sse_event
from threadneedle.config import settings
from threadneedle.indexing.build import show_stats

log = logging.getLogger(__name__)

router = APIRouter()


def _agent_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def last_human_turn_ids(messages: list) -> tuple[str | None, list[str]]:
    """Ids from the last human message through the end of the thread."""
    last_idx = None
    for i, message in enumerate(messages):
        kind = getattr(message, "type", None)
        if kind is None and isinstance(message, dict):
            kind = message.get("type") or message.get("role")
        if kind in {"human", "user"}:
            last_idx = i
    if last_idx is None:
        return None, []
    human = messages[last_idx]
    if isinstance(human, dict):
        text = message_text(human.get("content"))
    else:
        text = message_text(getattr(human, "content", None))
    ids: list[str] = []
    for message in messages[last_idx:]:
        mid = message.get("id") if isinstance(message, dict) else getattr(message, "id", None)
        if mid:
            ids.append(str(mid))
    return text, ids


async def rewind_last_turn(agent, config: dict) -> None:
    """Drop the last user turn and the assistant/tool messages that followed it."""
    try:
        state = await agent.aget_state(config)
    except Exception:
        log.exception("could not load checkpoint to regenerate")
        return
    values = getattr(state, "values", None) or {}
    _, ids = last_human_turn_ids(list(values.get("messages") or []))
    if not ids:
        return
    await agent.aupdate_state(
        config, {"messages": [RemoveMessage(id=item) for item in ids]}
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/index/stats")
async def index_stats() -> dict:
    try:
        return show_stats(settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    agent = request.app.state.agent
    citations: list[dict] = []
    config = _agent_config(req.thread_id)
    try:
        if req.regenerate:
            await rewind_last_turn(agent, config)
        await repair_thread(agent, config)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as exc:
        log.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    messages = result.get("messages") or []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            citations.extend(extract_citations(message_text(msg.content)))

    final = messages[-1] if messages else None
    text = message_text(getattr(final, "content", "")) if final else ""
    return ChatResponse(thread_id=req.thread_id, response=text, citations=citations)


async def _stream_agent(agent, req: ChatRequest) -> AsyncIterator[str]:
    config = _agent_config(req.thread_id)
    citations: list[dict] = []
    final_text_parts: list[str] = []
    seen_citation_keys: set[tuple] = set()

    try:
        if req.regenerate:
            await rewind_last_turn(agent, config)
        await repair_thread(agent, config)
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            name = event.get("name") or ""
            data = event.get("data") or {}

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if isinstance(chunk, AIMessageChunk):
                    piece = message_text(chunk.content)
                    if piece:
                        final_text_parts.append(piece)
                        yield sse_event("token", {"text": piece})
                elif chunk is not None:
                    piece = message_text(getattr(chunk, "content", chunk))
                    if piece:
                        final_text_parts.append(piece)
                        yield sse_event("token", {"text": piece})

            elif kind == "on_tool_start":
                yield sse_event(
                    "tool",
                    {
                        "status": "start",
                        "name": name,
                        "args": data.get("input") or {},
                    },
                )

            elif kind == "on_tool_end":
                output = data.get("output")
                if isinstance(output, ToolMessage):
                    raw = message_text(output.content)
                else:
                    raw = message_text(output)
                found = extract_citations(raw)
                for cite in found:
                    key = (
                        cite.get("source"),
                        cite.get("page"),
                        cite.get("section_path"),
                    )
                    if key in seen_citation_keys:
                        continue
                    seen_citation_keys.add(key)
                    citations.append(cite)
                    yield sse_event("citation", cite)
                preview = raw if len(raw) < 800 else raw[:800] + "…"
                yield sse_event(
                    "tool",
                    {
                        "status": "end",
                        "name": name,
                        "result": preview,
                    },
                )

            elif kind == "on_chat_model_end":
                # Some models only emit the full message at end (no token stream).
                output = data.get("output")
                if isinstance(output, AIMessage) and not final_text_parts:
                    text = message_text(output.content)
                    if text and not output.tool_calls:
                        final_text_parts.append(text)
                        yield sse_event("token", {"text": text})

        yield sse_event(
            "done",
            {
                "text": "".join(final_text_parts),
                "citations": citations,
                "thread_id": req.thread_id,
            },
        )
    except Exception as exc:
        log.exception("SSE stream failed")
        yield sse_event("error", {"detail": str(exc)})


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    agent = request.app.state.agent

    async def events() -> AsyncIterator[str]:
        async for chunk in _stream_agent(agent, req):
            yield chunk

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
