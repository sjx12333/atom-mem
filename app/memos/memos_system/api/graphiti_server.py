"""Small Graphiti-only local server.

The legacy MemOS v2 server remains available as source for migration and
tests, but this is the deployment entry point.  It does not import Qdrant,
Neo4j, the old memory extractors, or PAD code.
"""

from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    from .graphiti_routes import install_graphiti_routes
    from ..graphiti_memory.paths import GRAPHITI_ROOT, HOME_ROOT
    from ..graphiti_memory.runtime import configure_logging
except ImportError:  # direct execution from memos_system
    from api.graphiti_routes import install_graphiti_routes
    from graphiti_memory.paths import GRAPHITI_ROOT, HOME_ROOT
    from graphiti_memory.runtime import configure_logging


MANAGEMENT_SECRET = os.getenv("GRAPHITI_MANAGEMENT_TOKEN") or "local-graphiti-session"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Match the reference WebUI/backend lifecycle: if a local Neo4j install is
    # configured, bring it up before the first Graphiti request.  A failed
    # start is reported through the status endpoint and must not prevent the
    # HTTP gateway from serving diagnostics or JSON-backed memory operations.
    service = getattr(app.state, "graphiti_memory_service", None)
    if service:
        try:
            await asyncio.to_thread(service.start_neo4j)
        except Exception:
            pass
    yield
    if service:
        service.on_process_exit()


app = FastAPI(
    title="Graphiti Memory Gateway",
    version="1.0.0",
    description="Local role-scoped Graphiti memory and formation inspection gateway",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8003", "http://localhost:8003",
        "http://127.0.0.1:8004", "http://localhost:8004",
        "http://127.0.0.1:5173", "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "X-Graphiti-Management-Session"],
)
service = install_graphiti_routes(app, GRAPHITI_ROOT, management_token=MANAGEMENT_SECRET)
service.log_info = configure_logging(GRAPHITI_ROOT)

WEBUI_DIST = HOME_ROOT / "webui"
if (WEBUI_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(WEBUI_DIST / "assets")), name="webui-assets")


@app.get("/")
async def root():
    index = WEBUI_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"service": "atom-mem", "status": "running", "webui": "not built"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "atom-mem",
        "data_root": "data/graphiti",
        "absolute_paths": False,
        "maintenance": service.maintenance_status(),
    }


@app.get("/{path:path}", include_in_schema=False)
async def webui_spa(path: str):
    """SPA fallback: deep links such as /graph must return index.html.

    API prefixes are registered before this catch-all, so they keep priority.
    """
    if path.startswith(("graphiti/", "v1/", "docs", "redoc", "openapi.json")):
        raise HTTPException(status_code=404, detail="Not Found")
    requested = WEBUI_DIST / path
    if requested.is_file():
        return FileResponse(requested)
    index = WEBUI_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
