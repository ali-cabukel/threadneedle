from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from threadneedle.agent.repair import dangling_tool_messages, repaired_history
from threadneedle.agent.tools import _chroma_filter, _never_crash
from threadneedle.api.routes import last_human_turn_ids
from threadneedle.api.sse import extract_citations, sse_event
from threadneedle.indexing.ingest import _suffix_for


def test_suffix_for_html_latest_urls():
    assert _suffix_for("https://www.ons.gov.uk/economy/foo/latest") == ".html"
    assert _suffix_for(
        "https://www.bankofengland.co.uk/-/media/boe/files/x.pdf"
    ) == ".pdf"
    assert _suffix_for("https://example.com/doc", declared="html") == ".html"


def test_chroma_filter_single_and_and():
    assert _chroma_filter(edition="2025-11") == {"edition": "2025-11"}
    filt = _chroma_filter(edition="2025-11", publisher="bank_of_england")
    assert filt == {
        "$and": [{"edition": "2025-11"}, {"publisher": "bank_of_england"}]
    }
    assert _chroma_filter() is None


def test_sse_event_and_citation_extraction():
    raw = sse_event("token", {"text": "hi"})
    assert raw.startswith("event: token\n")
    assert '"text": "hi"' in raw
    citations = extract_citations(
        '{"citations": [{"title": "MPR", "edition": "2025-11"}]}'
    )
    assert citations[0]["edition"] == "2025-11"
    assert extract_citations("not json") == []


def test_dangling_tool_messages_fills_unanswered_ids():
    messages = [
        HumanMessage(content="what is inflation?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_policy_docs", "id": "call_a", "args": {"query": "cpih"}},
                {"name": "ons_observation", "id": "call_b", "args": {"series": "cpih"}},
            ],
        ),
    ]
    fillers = dangling_tool_messages(messages)
    assert [m.tool_call_id for m in fillers] == ["call_a", "call_b"]


def test_dangling_tool_messages_ignores_answered_ids():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "list_corpus", "id": "call_a", "args": {}},
                {"name": "list_corpus", "id": "call_b", "args": {}},
            ],
        ),
        ToolMessage(content="ok", tool_call_id="call_a"),
    ]
    fillers = dangling_tool_messages(messages)
    assert [m.tool_call_id for m in fillers] == ["call_b"]
    assert dangling_tool_messages(messages + [ToolMessage(content="ok", tool_call_id="call_b")]) == []


def test_repaired_history_inserts_replies_before_later_humans():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_policy_docs", "id": "call_a", "args": {"query": "rates"}},
            {"name": "ons_observation", "id": "call_b", "args": {"series": "cpih"}},
        ],
    )
    late_a = ToolMessage(content="interrupted a", tool_call_id="call_a")
    late_b = ToolMessage(content="interrupted b", tool_call_id="call_b")
    messages = [
        HumanMessage(content="original"),
        ai,
        HumanMessage(content="retry 1"),
        HumanMessage(content="retry 2"),
        late_a,
        late_b,
        HumanMessage(content="retry 3"),
    ]
    fixed = repaired_history(messages)
    kinds = [m.type for m in fixed]
    assert kinds == ["human", "ai", "tool", "tool", "human", "human", "human"]
    assert [m.tool_call_id for m in fixed[2:4]] == ["call_a", "call_b"]
    assert fixed[2] is late_a and fixed[3] is late_b


def test_never_crash_returns_json_error():
    @_never_crash
    def boom():
        raise RuntimeError("chroma down")

    payload = boom()
    assert '"error": "RuntimeError"' in payload
    assert "chroma down" in payload


def test_last_human_turn_ids_covers_trailing_assistant_messages():
    messages = [
        HumanMessage(content="first", id="h1"),
        AIMessage(content="ok", id="a1"),
        HumanMessage(content="second", id="h2"),
        AIMessage(content="", id="a2", tool_calls=[{"name": "t", "id": "c1", "args": {}}]),
        ToolMessage(content="hit", tool_call_id="c1", id="t1"),
        AIMessage(content="answer", id="a3"),
    ]
    text, ids = last_human_turn_ids(messages)
    assert text == "second"
    assert ids == ["h2", "a2", "t1", "a3"]
    assert last_human_turn_ids([]) == (None, [])

