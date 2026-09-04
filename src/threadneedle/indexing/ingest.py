"""Stage 1 — ingestion.

Pulls everything named in sources.yaml down to data/raw/ and returns a list of
SourceDoc records carrying the local path plus the manifest metadata.

Downloads are cached, so re-running is cheap and the record manager downstream
sees a stable `source` value.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from threadneedle.config import Settings, settings

log = logging.getLogger(__name__)

USER_AGENT = "threadneedle-indexer/0.1 (+research use)"

SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".md"}


@dataclass
class SourceDoc:
    """One document on its way into the index."""

    source: str  # stable identity — URL if remote, else relative path
    local_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def suffix(self) -> str:
        return self.local_path.suffix.lower()


def _url_tail(url: str) -> str:
    return url.rstrip("/").split("/")[-1] or "document"


def _suffix_for(url: str, content_type: str = "", declared: str | None = None) -> str:
    if declared:
        value = declared.lower().lstrip(".")
        return f".{value}" if not declared.startswith(".") else declared.lower()

    tail = _url_tail(url)
    suffix = Path(tail).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return suffix

    ct = content_type.lower()
    if "pdf" in ct:
        return ".pdf"
    if "html" in ct or "text/" in ct:
        return ".html"
    if "ons.gov.uk" in url or "gov.uk" in url:
        return ".html"
    return ".pdf"


def _safe_filename(url: str, suffix: str) -> str:
    """Readable name plus a short hash so two URLs never collide."""
    tail = _url_tail(url)
    digest = hashlib.sha256(url.encode()).hexdigest()[:8]
    stem = Path(tail).stem[:80] or "document"
    return f"{stem}-{digest}{suffix}"


def _download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cached: %s", dest.name)
        return dest

    log.info("downloading: %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)

    log.info("saved %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
    return dest


def load_manifest(cfg: Settings = settings) -> list[dict[str, Any]]:
    with cfg.manifest_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    entries = raw.get("documents") or []
    if not entries:
        raise ValueError(f"No documents found in {cfg.manifest_path}")
    return entries


def ingest(
    cfg: Settings = settings,
    include_unlisted: bool = False,
) -> list[SourceDoc]:
    """Resolve the manifest into local files ready for parsing."""
    cfg.ensure_dirs()
    docs: list[SourceDoc] = []
    seen_paths: set[Path] = set()

    for entry in load_manifest(cfg):
        metadata = {
            k: v
            for k, v in entry.items()
            if k not in {"url", "path", "format"} and v is not None
        }

        if "url" in entry:
            url = entry["url"]
            suffix = _suffix_for(url, declared=entry.get("format"))
            local_path = cfg.raw_dir / _safe_filename(url, suffix)
            try:
                _download(url, local_path)
            except httpx.HTTPError as exc:
                log.error("skipping %s — %s", url, exc)
                continue
            source = url
        elif "path" in entry:
            local_path = Path(entry["path"])
            if not local_path.is_absolute():
                local_path = cfg.manifest_path.parent / local_path
            if not local_path.exists():
                log.error("skipping missing file: %s", local_path)
                continue
            source = str(local_path.resolve())
        else:
            log.error("manifest entry needs either `url` or `path`: %r", entry)
            continue

        metadata.setdefault("title", local_path.stem)
        docs.append(SourceDoc(source=source, local_path=local_path, metadata=metadata))
        seen_paths.add(local_path.resolve())

    if include_unlisted:
        for path in sorted(cfg.raw_dir.iterdir()):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if path.resolve() in seen_paths:
                continue
            log.info("picking up unlisted file: %s", path.name)
            docs.append(
                SourceDoc(
                    source=str(path.resolve()),
                    local_path=path,
                    metadata={
                        "title": path.stem,
                        "publisher": "unknown",
                        "doc_type": "unknown",
                    },
                )
            )

    log.info("ingested %d documents", len(docs))
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for doc in ingest():
        print(f"{doc.metadata.get('edition', '-'):>8}  {doc.local_path.name}")
