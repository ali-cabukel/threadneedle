"""Stage 4 — embeddings and vector store.

Factories only. The chatbot imports `get_vector_store()` and gets exactly the
store the indexer wrote to, with no duplicated configuration.

Chroma's Rust client is a process-wide singleton per persist path. Creating a
second PersistentClient (easy when the agent runs tools concurrently) fails
with a tenant error, then Chroma 1.x crashes in `stop()` with
`'RustBindingsAPI' object has no attribute 'bindings'`. We keep one client
and one LangChain wrapper behind a lock.
"""

from __future__ import annotations

import logging
import threading

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from threadneedle.config import Settings, settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_embeddings: Embeddings | None = None
_chroma_client = None
_vector_store: VectorStore | None = None


def get_embeddings(cfg: Settings = settings) -> Embeddings:
    global _embeddings
    with _lock:
        if _embeddings is None:
            _embeddings = _build_embeddings(cfg)
        return _embeddings


def _build_embeddings(cfg: Settings) -> Embeddings:
    backend = cfg.embedding_backend.lower()

    if backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=cfg.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=cfg.openai_embedding_model)

    raise ValueError(
        f"Unknown EMBEDDING_BACKEND={backend!r}. Options: huggingface, openai"
    )


def get_chroma_client(cfg: Settings = settings):
    """One PersistentClient per process — Chroma 1.x cannot open the same path twice."""
    global _chroma_client
    with _lock:
        if _chroma_client is None:
            import chromadb

            cfg.ensure_dirs()
            log.info("opening Chroma at %s", cfg.vector_dir)
            _chroma_client = chromadb.PersistentClient(path=str(cfg.vector_dir))
        return _chroma_client


def get_vector_store(cfg: Settings = settings) -> VectorStore:
    global _chroma_client, _embeddings, _vector_store
    backend = cfg.vector_backend.lower()

    with _lock:
        if _vector_store is not None:
            return _vector_store

        embeddings = _embeddings or _build_embeddings(cfg)
        if _embeddings is None:
            _embeddings = embeddings

        if backend == "chroma":
            from langchain_chroma import Chroma

            cfg.ensure_dirs()
            if _chroma_client is None:
                import chromadb

                log.info("opening Chroma at %s", cfg.vector_dir)
                _chroma_client = chromadb.PersistentClient(path=str(cfg.vector_dir))

            _vector_store = Chroma(
                client=_chroma_client,
                collection_name=cfg.collection_name,
                embedding_function=embeddings,
            )
            return _vector_store

        if backend == "qdrant":
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

            client = QdrantClient(url=cfg.qdrant_url)
            _vector_store = QdrantVectorStore(
                client=client,
                collection_name=cfg.collection_name,
                embedding=embeddings,
            )
            return _vector_store

    raise ValueError(f"Unknown VECTOR_BACKEND={backend!r}. Options: chroma, qdrant")
