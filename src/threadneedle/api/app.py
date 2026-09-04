"""FastAPI entrypoint for the Threadneedle chatbot."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from threadneedle.agent.graph import build_agent
from threadneedle.api.routes import router
from threadneedle.config import settings

STATIC_DIR = settings.static_dir


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    db_path = str(settings.checkpoint_db)
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.agent = build_agent(checkpointer)
        yield


app = FastAPI(
    title="Threadneedle",
    description="UK macro policy RAG over Bank of England, ONS and HM Treasury sources.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Chat UI not found.")
    return FileResponse(index_path)
