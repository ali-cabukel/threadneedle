"""LangGraph tool-calling agent over the Chroma policy index."""

from __future__ import annotations

from typing import Any

from threadneedle.agent.prompts import SYSTEM_PROMPT
from threadneedle.agent.repair import (
    history_needs_repair,
    repaired_history,
    replace_messages_update,
)
from threadneedle.agent.tools import TOOLS
from threadneedle.config import settings


def _history_middleware() -> list[Any]:
    """Fix tool_call / ToolMessage pairing before the model sees the thread."""
    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError:
        return []

    class RepairToolHistoryMiddleware(AgentMiddleware):
        """Rewrite poisoned tool history for both sync and async invoke paths."""

        def _persist(self, state):
            messages = list(state.get("messages") or [])
            if not history_needs_repair(messages):
                return None
            return replace_messages_update(repaired_history(messages))

        def _prepare(self, request):
            messages = list(request.messages or [])
            if history_needs_repair(messages):
                return request.override(messages=repaired_history(messages))
            return request

        def before_model(self, state, runtime):
            return self._persist(state)

        async def abefore_model(self, state, runtime):
            return self._persist(state)

        def wrap_model_call(self, request, handler):
            return handler(self._prepare(request))

        async def awrap_model_call(self, request, handler):
            return await handler(self._prepare(request))

    return [RepairToolHistoryMiddleware()]


def build_agent(checkpointer: Any | None = None):
    """Build the Threadneedle agent.

    Uses LangChain 1.x `create_agent` when available, and falls back to
    LangGraph's prebuilt ReAct agent on older installs.
    """
    model = f"openai:{settings.openai_chat_model}"
    try:
        from langchain.agents import create_agent

        kwargs: dict[str, Any] = {
            "model": model,
            "tools": TOOLS,
            "system_prompt": SYSTEM_PROMPT,
            "middleware": _history_middleware(),
        }
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer
        return create_agent(**kwargs)
    except (ImportError, TypeError):
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0)
        kwargs = {"model": llm, "tools": TOOLS, "prompt": SYSTEM_PROMPT}
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer
        return create_react_agent(**kwargs)
