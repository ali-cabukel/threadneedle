"""Central configuration for indexing and the chatbot.

Everything is overridable by environment variable so the same settings drive
the CLI indexer and the FastAPI service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _discover_root() -> Path:
    """Repo root: directory that contains pyproject.toml and sources.yaml."""
    env_root = os.environ.get("THREADNEEDLE_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    if here.parent.name == "threadneedle" and here.parents[1].name == "src":
        root = here.parents[2]
        if (root / "pyproject.toml").exists():
            return root

    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (base / "pyproject.toml").exists() and (base / "sources.yaml").exists():
            return base

    return Path.cwd().resolve()


PROJECT_ROOT = _discover_root()

load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- paths -------------------------------------------------------------
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    parsed_dir: Path = PROJECT_ROOT / "data" / "parsed"
    vector_dir: Path = PROJECT_ROOT / "data" / "chroma"
    manifest_path: Path = PROJECT_ROOT / "sources.yaml"
    checkpoint_db: Path = PROJECT_ROOT / "data" / "checkpoints.sqlite"
    static_dir: Path = PROJECT_ROOT / "static"

    # --- parsing -----------------------------------------------------------
    # Docling is slower but materially better on MPR tables and HTML bulletins.
    # Swap with PDF_PARSER=pymupdf4llm for a faster PDF-only pass.
    pdf_parser: str = field(default_factory=lambda: _env("PDF_PARSER", "docling"))
    # BoE / ONS / gov.uk files are born-digital. OCR on chart images just emits
    # RapidOCR empty-result warnings and slows the run. Opt in for scans.
    docling_do_ocr: bool = field(
        default_factory=lambda: _env_bool("DOCLING_DO_OCR", False)
    )

    # --- chunking ----------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 200))

    # --- embeddings --------------------------------------------------------
    embedding_backend: str = field(
        default_factory=lambda: _env("EMBEDDING_BACKEND", "huggingface")
    )
    hf_embedding_model: str = field(
        default_factory=lambda: _env("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- vector store ------------------------------------------------------
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "chroma"))
    collection_name: str = field(
        default_factory=lambda: _env("COLLECTION_NAME", "uk_macro_policy")
    )
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))

    # --- record manager (incremental indexing) -----------------------------
    record_db_url: str = field(
        default_factory=lambda: _env(
            "RECORD_DB_URL",
            f"sqlite:///{PROJECT_ROOT / 'data' / 'record_manager.sqlite'}",
        )
    )

    # --- chatbot -----------------------------------------------------------
    openai_chat_model: str = field(
        default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    )
    ons_api_base: str = field(
        default_factory=lambda: _env("ONS_API_BASE", "https://api.beta.ons.gov.uk/v1")
    )
    retrieval_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_K", 6))

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.parsed_dir, self.vector_dir, self.checkpoint_db.parent):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def namespace(self) -> str:
        """Record manager namespace. Must change if the embedding space changes."""
        model = (
            self.hf_embedding_model
            if self.embedding_backend == "huggingface"
            else self.openai_embedding_model
        )
        return f"{self.vector_backend}/{self.collection_name}/{model}"


settings = Settings()
