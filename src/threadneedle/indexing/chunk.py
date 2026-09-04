"""Stage 3 — splitting.

Two-pass strategy:

  1. MarkdownHeaderTextSplitter cuts on real section boundaries, so a chunk
     never straddles "Costs and prices" and "The labour market".
  2. RecursiveCharacterTextSplitter cuts the oversized sections that remain,
     preferring paragraph then sentence boundaries.

Each chunk carries a `section_path` breadcrumb ("Monetary policy summary >
Costs and prices"). That string is prepended to the embedded text so short
chunks still encode what they are about.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from threadneedle.config import Settings, settings

log = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

HEADER_KEYS = ["h1", "h2", "h3"]

NUMERIC_NOISE = re.compile(r"^[\s\d.,%()\-–—/|]+$")


def _section_path(metadata: dict[str, Any]) -> str:
    parts = [str(metadata[k]).strip() for k in HEADER_KEYS if metadata.get(k)]
    return " > ".join(parts)


def _is_noise(text: str, min_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    if NUMERIC_NOISE.match(stripped):
        return True
    digits = sum(ch.isdigit() for ch in stripped)
    if digits / max(len(stripped), 1) > 0.4:
        return True
    return False


def _chunk_id(source: str, section: str, index: int, text: str) -> str:
    """Deterministic ID: same input text always yields the same ID."""
    payload = f"{source}|{section}|{index}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def split_documents(
    documents: list[Document],
    cfg: Settings = settings,
    prepend_section_path: bool = True,
) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=prepend_section_path,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )

    chunks: list[Document] = []
    dropped = 0

    for doc in documents:
        try:
            sections = header_splitter.split_text(doc.page_content)
        except Exception:
            sections = [Document(page_content=doc.page_content, metadata={})]

        if not sections:
            sections = [Document(page_content=doc.page_content, metadata={})]

        for section in sections:
            section_meta = {**doc.metadata, **section.metadata}
            section_path = _section_path(section.metadata)

            pieces = recursive_splitter.split_text(section.page_content)

            for i, piece in enumerate(pieces):
                if _is_noise(piece, cfg.min_chunk_chars):
                    dropped += 1
                    continue

                content = piece
                if prepend_section_path and section_path:
                    content = f"{section_path}\n\n{piece}"

                metadata = {
                    **section_meta,
                    "section_path": section_path,
                    "chunk_index": i,
                    "chunk_chars": len(piece),
                }
                for key in HEADER_KEYS:
                    metadata.pop(key, None)

                metadata["chunk_id"] = _chunk_id(
                    metadata.get("source", ""), section_path, i, piece
                )

                chunks.append(Document(page_content=content, metadata=metadata))

    log.info(
        "split %d pages into %d chunks (dropped %d noise fragments)",
        len(documents),
        len(chunks),
        dropped,
    )
    return chunks


def sanitize_metadata(chunks: list[Document]) -> list[Document]:
    """Vector stores only accept scalar metadata."""
    for chunk in chunks:
        clean: dict[str, Any] = {}
        for key, value in chunk.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, tuple)):
                clean[key] = ", ".join(str(v) for v in value)
            else:
                clean[key] = str(value)
        chunk.metadata = clean
    return chunks


if __name__ == "__main__":
    from threadneedle.indexing.ingest import ingest
    from threadneedle.indexing.parse import parse_all

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parsed = parse_all(ingest())
    result = sanitize_metadata(split_documents(parsed))

    print(f"\n{len(result)} chunks\n")
    for chunk in result[:3]:
        print("-" * 70)
        print(
            "meta:",
            {
                k: chunk.metadata[k]
                for k in ("edition", "page", "section_path")
                if k in chunk.metadata
            },
        )
        print(chunk.page_content[:400])
