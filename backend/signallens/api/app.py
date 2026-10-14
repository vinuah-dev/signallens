"""FastAPI application factory. Serves the API under /api and, in production,
the built frontend from STATIC_DIR as a single-page app."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..core.errors import SignalLensError
from .routes import router

log = logging.getLogger("signallens")


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details or {}}})


def create_app() -> FastAPI:
    app = FastAPI(title="SignalLens API", version=__version__)

    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(SignalLensError)
    async def domain_error(_: Request, exc: SignalLensError):
        return _error(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def request_error(_: Request, exc: RequestValidationError):
        problems = [f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors()]
        return _error(422, "bad_request", "The request was malformed: " + "; ".join(problems))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return _error(500, "internal_error", "Something went wrong on the server. Please try again.")

    app.include_router(router)

    static_dir = Path(os.getenv("STATIC_DIR", Path(__file__).resolve().parents[3] / "frontend" / "dist"))
    if static_dir.is_dir() and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return _error(404, "not_found", "Unknown API route.")
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
