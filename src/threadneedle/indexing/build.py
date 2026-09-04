"""Stage 5 — indexing.

Runs the full pipeline and writes to the vector store through LangChain's
record manager, which gives you real upsert semantics:

  * re-running is idempotent — unchanged chunks are skipped, not duplicated
  * when a document is re-published, its old chunks are deleted automatically
  * dropping a document from sources.yaml removes it from the index

Usage:
    uv run threadneedle-index
    uv run threadneedle-index --cleanup scoped_full
    uv run threadneedle-index --rebuild
    uv run threadneedle-index --dry-run
    uv run threadneedle-index --stats
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.indexing import index

try:
    from langchain_classic.indexes import SQLRecordManager
except ImportError:  # pragma: no cover
    try:
        from langchain.indexes import SQLRecordManager
    except ImportError:
        from langchain_community.indexes._sql_record_manager import (  # type: ignore
            SQLRecordManager,
        )

from threadneedle.config import Settings, settings
from threadneedle.indexing.chunk import sanitize_metadata, split_documents
from threadneedle.indexing.ingest import ingest
from threadneedle.indexing.parse import cache_parsed, parse_all
from threadneedle.indexing.store import get_vector_store

log = logging.getLogger(__name__)


def build_chunks(cfg: Settings, include_unlisted: bool = False) -> list[Document]:
    sources = ingest(cfg, include_unlisted=include_unlisted)
    if not sources:
        raise SystemExit("Nothing to index — check sources.yaml.")

    pages = parse_all(sources, cfg)
    cache_parsed(pages, cfg)

    chunks = split_documents(pages, cfg)
    return sanitize_metadata(chunks)


def report(chunks: list[Document]) -> None:
    if not chunks:
        print("No chunks produced.")
        return

    lengths = sorted(len(c.page_content) for c in chunks)
    by_edition = Counter(c.metadata.get("edition", "unknown") for c in chunks)
    with_section = sum(1 for c in chunks if c.metadata.get("section_path"))

    print(f"\nchunks:            {len(chunks)}")
    print(f"median length:     {lengths[len(lengths) // 2]} chars")
    print(f"p95 length:        {lengths[int(len(lengths) * 0.95)]} chars")
    print(f"with section path: {with_section} ({with_section / len(chunks):.0%})")
    print("\nby edition:")
    for edition, count in sorted(by_edition.items()):
        print(f"  {edition:>10}  {count}")


def run(cfg: Settings, args: argparse.Namespace) -> None:
    if args.rebuild:
        log.warning("rebuilding from scratch — clearing vector store and record db")
        if cfg.vector_dir.exists():
            shutil.rmtree(cfg.vector_dir)
        if cfg.record_db_url.startswith("sqlite:///"):
            Path(cfg.record_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)
        cfg.ensure_dirs()

    chunks = build_chunks(cfg, include_unlisted=args.include_unlisted)
    report(chunks)

    if args.dry_run:
        print("\n--dry-run: stopping before embedding.")
        print("Inspect data/parsed/parsed.jsonl to sanity-check extraction quality.")
        return

    vector_store = get_vector_store(cfg)

    record_manager = SQLRecordManager(cfg.namespace, db_url=cfg.record_db_url)
    record_manager.create_schema()

    cleanup = args.cleanup
    log.info("indexing %d chunks (cleanup=%s)", len(chunks), cleanup)

    result = index(
        chunks,
        record_manager,
        vector_store,
        cleanup=cleanup,
        source_id_key="source",
        batch_size=args.batch_size,
        force_update=args.force_update,
    )

    print("\nindex result:")
    for key, value in result.items():
        print(f"  {key:<14} {value}")
    print(f"\nnamespace: {cfg.namespace}")
    print(f"store:     {cfg.vector_dir}")


def show_stats(cfg: Settings) -> dict:
    """Peek at the live index — used by the CLI and the /index/stats API."""
    vector_store = get_vector_store(cfg)
    try:
        data = vector_store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
    except AttributeError:
        return {"error": "Stats are only implemented for the Chroma backend."}

    summary: dict = {
        "vectors": len(metadatas),
        "namespace": cfg.namespace,
        "collection": cfg.collection_name,
    }
    for field in ("publisher", "doc_type", "edition"):
        counts = Counter(m.get(field, "unknown") for m in metadatas)
        summary[field] = dict(sorted(counts.items()))

    sample = []
    for doc in vector_store.similarity_search("services inflation", k=3):
        meta = doc.metadata
        sample.append(
            {
                "edition": meta.get("edition"),
                "page": meta.get("page"),
                "section_path": meta.get("section_path"),
                "title": meta.get("title"),
            }
        )
    summary["sample_retrieval"] = sample
    return summary


def print_stats(cfg: Settings) -> None:
    summary = show_stats(cfg)
    if "error" in summary:
        print(summary["error"])
        return

    print(f"vectors: {summary['vectors']}")
    for field in ("publisher", "doc_type", "edition"):
        print(f"\n{field}:")
        for value, count in (summary.get(field) or {}).items():
            print(f"  {str(value):<28} {count}")

    print("\nsample retrieval — 'services inflation':")
    for hit in summary.get("sample_retrieval") or []:
        print(
            f"  [{hit.get('edition')} p{hit.get('page')}] "
            f"{str(hit.get('section_path') or '')[:60]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleanup",
        choices=["incremental", "scoped_full", "full"],
        default="incremental",
        help=(
            "incremental: delete a source's old chunks as its new ones arrive. "
            "scoped_full: also prune chunks for sources in this run that no "
            "longer produce them (best for a full re-parse). "
            "full: prune everything not in this run — only safe when the run "
            "covers the whole corpus."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="wipe the vector store and record db first",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and chunk only; skip embedding",
    )
    parser.add_argument(
        "--include-unlisted",
        action="store_true",
        help="also index files sitting in data/raw/",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="re-embed even unchanged chunks (after a model swap)",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--stats",
        action="store_true",
        help="inspect the existing index and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.stats:
        print_stats(settings)
        return

    run(settings, args)


if __name__ == "__main__":
    main()
