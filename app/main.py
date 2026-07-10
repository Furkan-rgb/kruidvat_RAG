"""FastAPI entry point for the local Kruidvat Ingredient Advisor."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import AskRequest, AskResponse, HealthResponse, StoredProductResponse
from app.service import InputError, RAGService, ServiceError

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kruidvat Ingredient Advisor",
    description="Local semantic retrieval and grounded cosmetics advice.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_service() -> RAGService:
    """Create a lightweight service; connections remain lazy and per request."""
    return RAGService()


@app.exception_handler(ServiceError)
async def service_error_handler(_request: Request, exc: ServiceError):
    return JSONResponse(status_code=503, content={"detail": exc.detail()})


@app.exception_handler(InputError)
async def input_error_handler(_request: Request, exc: InputError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_input",
                "message": str(exc),
                "remediation": "Correct the request values and try again.",
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, _exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "code": "unexpected_error",
                "message": "An unexpected server error occurred.",
                "remediation": "Check the server log for details and try again.",
            }
        },
    )


@app.get("/api/health", response_model=HealthResponse)
def health():
    return get_service().health()


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    return get_service().ask(
        payload.question,
        mode=payload.mode,
        top_k=payload.top_k,
    ).to_dict()


def _stream_events(payload: AskRequest):
    """Encode service events as newline-delimited JSON, including late errors."""
    try:
        yield from (
            json.dumps(event, ensure_ascii=False) + "\n"
            for event in get_service().stream_ask(
                payload.question,
                mode=payload.mode,
                top_k=payload.top_k,
            )
        )
    except ServiceError as exc:
        yield json.dumps({"type": "error", **exc.detail()}, ensure_ascii=False) + "\n"
    except Exception:
        logger.exception("Unexpected error while streaming an answer")
        yield json.dumps(
            {
                "type": "error",
                "code": "unexpected_error",
                "message": "An unexpected server error occurred.",
                "remediation": "Check the server log for details and try again.",
            }
        ) + "\n"


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):
    return StreamingResponse(
        _stream_events(payload),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/products/{product_id}", response_model=StoredProductResponse)
def product(product_id: int):
    found = get_service().get_product(product_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Product not found.")
    return asdict(found)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
