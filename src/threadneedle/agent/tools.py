"""Agent tools: Chroma retrieval, edition comparison, live ONS figures."""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.tools import tool

from threadneedle.config import settings
from threadneedle.indexing.store import get_vector_store

log = logging.getLogger(__name__)


def _never_crash(fn):
    """Return a JSON error string instead of raising, so the tool loop can finish."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.exception("tool %s failed", fn.__name__)
            return json.dumps({"error": type(exc).__name__, "detail": str(exc)})

    return wrapped

CITATION_KEYS = (
    "title",
    "edition",
    "page",
    "section_path",
    "source",
    "publisher",
    "doc_type",
)

# Curated ONS Beta API series. Numbers are fetched live, not from the index.
ONS_SERIES: dict[str, dict[str, Any]] = {
    "cpih": {
        "dataset_id": "cpih01",
        "label": "CPIH (Consumer Prices Index including owner occupiers' housing costs)",
        "dimensions": {"geography": "K02000001", "aggregate": "cpih1dim1A0"},
    },
    "cpi": {
        "dataset_id": "cpih01",
        "label": "CPIH headline (use this for UK inflation)",
        "dimensions": {"geography": "K02000001", "aggregate": "cpih1dim1A0"},
    },
}


def _chroma_filter(
    edition: str | None = None,
    publisher: str | None = None,
    doc_type: str | None = None,
) -> dict[str, Any] | None:
    pairs = {
        key: value
        for key, value in (
            ("edition", edition),
            ("publisher", publisher),
            ("doc_type", doc_type),
        )
        if value
    }
    if not pairs:
        return None
    if len(pairs) == 1:
        key, value = next(iter(pairs.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in pairs.items()]}


def citation_from_doc(doc: Document) -> dict[str, Any]:
    meta = doc.metadata or {}
    return {key: meta.get(key) for key in CITATION_KEYS}


def format_hits(docs: list[Document]) -> str:
    results = []
    for i, doc in enumerate(docs, start=1):
        citation = citation_from_doc(doc)
        results.append(
            {
                "rank": i,
                "text": doc.page_content,
                **citation,
            }
        )
    return json.dumps(
        {"results": results, "citations": [citation_from_doc(d) for d in docs]},
        indent=2,
    )


def search_docs(
    query: str,
    k: int = 6,
    edition: str | None = None,
    publisher: str | None = None,
    doc_type: str | None = None,
) -> list[Document]:
    filt = _chroma_filter(edition=edition, publisher=publisher, doc_type=doc_type)
    kwargs: dict[str, Any] = {"k": k}
    if filt is not None:
        kwargs["filter"] = filt
    return get_vector_store().similarity_search(query, **kwargs)


@tool
@_never_crash
def search_policy_docs(
    query: str,
    k: int = 6,
    edition: str | None = None,
    publisher: str | None = None,
    doc_type: str | None = None,
) -> str:
    """Search indexed UK macro policy documents (BoE MPRs, MPC minutes, ONS bulletins, HMT fiscal statements).

    Use edition like "2025-11" to restrict to one release. publisher is one of
    bank_of_england, ons, hm_treasury. doc_type is one of monetary_policy_report,
    mpc_minutes, statistical_bulletin, fiscal_policy.
    """
    docs = search_docs(query, k=k, edition=edition, publisher=publisher, doc_type=doc_type)
    if not docs:
        return json.dumps({"results": [], "citations": [], "note": "No matching chunks."})
    return format_hits(docs)


@tool
@_never_crash
def compare_editions(query: str, editions: list[str], k: int = 3) -> str:
    """Retrieve the same topic from multiple editions so you can compare how the official view changed.

    editions should be keys like ["2025-02", "2025-05", "2025-08", "2025-11"].
    """
    if not editions:
        return json.dumps({"error": "Pass at least one edition, e.g. ['2025-08', '2025-11']."})

    by_edition: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    for edition in editions:
        docs = search_docs(query, k=k, edition=edition)
        by_edition[edition] = [
            {"text": d.page_content, **citation_from_doc(d)} for d in docs
        ]
        citations.extend(citation_from_doc(d) for d in docs)
    return json.dumps({"query": query, "by_edition": by_edition, "citations": citations}, indent=2)


def _ons_client() -> httpx.Client:
    return httpx.Client(
        base_url=settings.ons_api_base.rstrip("/"),
        timeout=30.0,
        headers={"Accept": "application/json", "User-Agent": "threadneedle/0.1"},
        follow_redirects=True,
    )


def _latest_version_href(dataset_id: str) -> tuple[str, dict[str, Any]]:
    with _ons_client() as client:
        response = client.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        payload = response.json()
    links = payload.get("links") or {}
    latest = (links.get("latest_version") or {}).get("href")
    if not latest:
        raise ValueError(f"ONS dataset {dataset_id!r} has no latest_version link.")
    return latest, payload


def _observations(
    version_href: str, dimensions: dict[str, str], n_latest: int
) -> list[dict[str, Any]]:
    params = {**dimensions, "time": "*"}
    path = version_href
    if path.startswith("http"):
        # Absolute href from the API — call it directly.
        with httpx.Client(
            timeout=30.0,
            headers={"Accept": "application/json", "User-Agent": "threadneedle/0.1"},
            follow_redirects=True,
        ) as client:
            response = client.get(path.rstrip("/") + "/observations", params=params)
            response.raise_for_status()
            payload = response.json()
    else:
        with _ons_client() as client:
            response = client.get(path.rstrip("/") + "/observations", params=params)
            response.raise_for_status()
            payload = response.json()

    observations = payload.get("observations") or []
    cleaned = []
    for item in observations:
        dims = item.get("dimensions") or {}
        time_id = ((dims.get("time") or {}).get("id")) or item.get("time")
        cleaned.append(
            {
                "time": time_id,
                "value": item.get("observation"),
                "unit": payload.get("unit_of_measure"),
            }
        )
    cleaned = [row for row in cleaned if row.get("value") not in (None, "")]
    return cleaned[-n_latest:]


@tool
@_never_crash
def ons_observation(series: str = "cpih", n_latest: int = 12) -> str:
    """Fetch the latest official ONS figures (live API, not the document index).

    series is a short alias: "cpih" or "cpi" for UK CPIH. Use this for current
    inflation numbers rather than quoting a figure from a PDF.
    """
    alias = series.strip().lower()
    spec = ONS_SERIES.get(alias)
    if spec is None:
        return json.dumps(
            {
                "error": f"Unknown series {series!r}.",
                "available": {k: v["label"] for k, v in ONS_SERIES.items()},
            }
        )
    try:
        href, meta = _latest_version_href(spec["dataset_id"])
        points = _observations(href, spec["dimensions"], n_latest=max(1, min(n_latest, 48)))
    except httpx.HTTPError as exc:
        log.warning("ONS API error: %s", exc)
        return json.dumps({"error": f"ONS API request failed: {exc}"})
    except Exception as exc:
        log.warning("ONS parse error: %s", exc)
        return json.dumps({"error": str(exc)})

    title = (meta.get("title") or spec["label"]) if isinstance(meta, dict) else spec["label"]
    return json.dumps(
        {
            "series": alias,
            "label": spec["label"],
            "dataset_id": spec["dataset_id"],
            "title": title,
            "observations": points,
            "source": settings.ons_api_base,
        },
        indent=2,
    )


@tool
@_never_crash
def list_corpus() -> str:
    """List what is currently indexed: publishers, document types, editions, titles.

    Call this before inventing an edition or publisher that may not be in the store.
    """
    store = get_vector_store()
    try:
        data = store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
    except AttributeError:
        return json.dumps({"error": "list_corpus requires the Chroma backend."})

    documents: dict[str, dict[str, Any]] = {}
    for meta in metadatas:
        source = meta.get("source") or "unknown"
        documents.setdefault(
            source,
            {
                "title": meta.get("title"),
                "publisher": meta.get("publisher"),
                "doc_type": meta.get("doc_type"),
                "edition": meta.get("edition"),
                "publication_date": meta.get("publication_date"),
                "chunks": 0,
            },
        )
        documents[source]["chunks"] += 1

    return json.dumps(
        {"document_count": len(documents), "documents": list(documents.values())},
        indent=2,
    )


TOOLS = [search_policy_docs, compare_editions, ons_observation, list_corpus]
