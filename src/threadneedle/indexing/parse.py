"""Stage 2 — parsing.

Turns each source file into a list of LangChain Documents holding *markdown*,
one per page (PDFs) or one per document/page (HTML).

Why markdown and not plain text: the header structure it preserves is what the
chunker uses in stage 3 to split on real section boundaries instead of
arbitrary character offsets.

Backends are pluggable via PDF_PARSER:
  docling       — default; better tables, HTML, and multi-column layout
  pymupdf4llm   — faster PDF-only fallback
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from threadneedle.config import Settings, settings
from threadneedle.indexing.ingest import SourceDoc

log = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    re.compile(r"^Bank of England\s+Page \d+\s*$", re.MULTILINE),
    re.compile(r"^Monetary Policy Report\s+\w+ \d{4}\s*$", re.MULTILINE),
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.MULTILINE),
]

CHART_LINE = re.compile(r"^\s*(Chart|Figure|Table)\s+[\dA-Z][\w.\-]*[:.]?\s", re.MULTILINE)

_HEADER_LABELS = {"title", "section_header", "subtitle"}


def _clean(text: str) -> str:
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf_pymupdf4llm(path: Path) -> list[tuple[str, dict[str, Any]]]:
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    out: list[tuple[str, dict[str, Any]]] = []
    for i, page in enumerate(pages, start=1):
        text = page.get("text", "") or ""
        page_meta = page.get("metadata") or {}
        out.append(
            (
                text,
                {
                    "page": page_meta.get("page", i),
                    "n_tables": len(page.get("tables") or []),
                    "n_images": len(page.get("images") or []),
                },
            )
        )
    return out


def _try_export(item: Any, method: str, doc: Any | None) -> str:
    exporter = getattr(item, method, None)
    if exporter is None:
        return ""
    attempts: list[dict] = [{}]
    if doc is not None:
        attempts.append({"doc": doc})
    for kwargs in attempts:
        try:
            exported = exporter(**kwargs)
        except TypeError:
            continue
        except Exception:
            return ""
        if exported and str(exported).strip():
            return str(exported).strip()
    return ""


def _item_markdown(item: Any, doc: Any | None = None) -> str:
    """Turn a Docling document item into markdown, preserving headings/tables."""
    label = str(getattr(item, "label", "") or "").lower()
    text = (getattr(item, "text", None) or "").strip()
    name = type(item).__name__

    if name in {"TitleItem", "SectionHeaderItem"} or label in _HEADER_LABELS:
        level = int(getattr(item, "level", 1) or 1)
        if name == "TitleItem" or label == "title":
            level = 1
        if text:
            return f"{'#' * min(max(level, 1), 6)} {text}"

    if name == "TableItem" or label == "table":
        exported = _try_export(item, "export_to_markdown", doc) or _try_export(
            item, "export_to_html", doc
        )
        if exported:
            return exported

    if text:
        return text

    return _try_export(item, "export_to_markdown", doc)


def _page_no(item: Any) -> int:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return 1
    return int(getattr(prov[0], "page_no", 1) or 1)


def _docling_converter(do_ocr: bool):
    """Reuse one converter so layout models are not reloaded per file."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_options = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=True)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )


_CONVERTERS: dict[bool, Any] = {}


def _get_docling_converter(do_ocr: bool):
    converter = _CONVERTERS.get(do_ocr)
    if converter is None:
        if not do_ocr:
            log.info(
                "Docling OCR disabled (born-digital PDFs). "
                "Set DOCLING_DO_OCR=1 if you add scanned documents."
            )
        converter = _docling_converter(do_ocr)
        _CONVERTERS[do_ocr] = converter
    return converter


def _parse_with_docling(
    path: Path, cfg: Settings = settings
) -> list[tuple[str, dict[str, Any]]]:
    """Parse PDF or HTML with Docling, grouped by provenance page number."""
    result = _get_docling_converter(cfg.docling_do_ocr).convert(str(path))
    doc = result.document

    pages: dict[int, list[str]] = defaultdict(list)
    for item, _level in doc.iterate_items():
        markdown = _item_markdown(item, doc)
        if markdown:
            pages[_page_no(item)].append(markdown)

    if pages:
        return [
            ("\n\n".join(blocks), {"page": page})
            for page, blocks in sorted(pages.items())
        ]

    fallback = doc.export_to_markdown()
    if fallback and fallback.strip():
        return [(fallback, {"page": 0})]
    return []


PDF_BACKENDS: dict[str, Callable[[Path], list[tuple[str, dict[str, Any]]]]] = {
    "pymupdf4llm": _parse_pdf_pymupdf4llm,
    "docling": _parse_with_docling,
}


def parse_document(doc: SourceDoc, cfg: Settings = settings) -> list[Document]:
    """Parse one SourceDoc into per-page markdown Documents."""
    suffix = doc.suffix

    if suffix == ".pdf":
        backend_name = cfg.pdf_parser
        backend = PDF_BACKENDS.get(backend_name)
        if backend is None:
            raise ValueError(
                f"Unknown PDF_PARSER={backend_name!r}. "
                f"Options: {sorted(PDF_BACKENDS)}"
            )
    elif suffix in {".html", ".htm"}:
        backend_name, backend = "docling", _parse_with_docling
    elif suffix in {".txt", ".md"}:
        backend_name = "raw"
        backend = lambda p: [(p.read_text(errors="ignore"), {"page": 0})]  # noqa: E731
    else:
        raise ValueError(f"No parser for {suffix} ({doc.local_path.name})")

    log.info("parsing %s with %s", doc.local_path.name, backend_name)
    raw_pages = backend(doc.local_path)

    documents: list[Document] = []
    for text, page_meta in raw_pages:
        cleaned = _clean(text)
        if not cleaned:
            continue
        documents.append(
            Document(
                page_content=cleaned,
                metadata={
                    **doc.metadata,
                    **page_meta,
                    "source": doc.source,
                    "parser": backend_name,
                    "has_chart_caption": bool(CHART_LINE.search(cleaned)),
                },
            )
        )

    log.info("  -> %d non-empty pages", len(documents))
    return documents


def parse_all(docs: list[SourceDoc], cfg: Settings = settings) -> list[Document]:
    parsed: list[Document] = []
    for doc in docs:
        try:
            parsed.extend(parse_document(doc, cfg))
        except Exception as exc:
            log.error("failed to parse %s — %s", doc.local_path.name, exc)
    return parsed


def cache_parsed(documents: list[Document], cfg: Settings = settings) -> Path:
    """Write parsed output to disk so you can eyeball it before embedding."""
    cfg.ensure_dirs()
    out = cfg.parsed_dir / "parsed.jsonl"
    with out.open("w") as fh:
        for doc in documents:
            fh.write(
                json.dumps({"text": doc.page_content, "metadata": doc.metadata}) + "\n"
            )
    log.info("wrote parsed cache to %s", out)
    return out


if __name__ == "__main__":
    from threadneedle.indexing.ingest import ingest

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parsed = parse_all(ingest())
    cache_parsed(parsed)
    print(f"{len(parsed)} pages parsed")
