"""FastAPI factory with private inputs, bounded requests, and explicit readiness."""

import asyncio
import json
import logging
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from starlette.exceptions import HTTPException

from triage.service.runtime import Runtime, ServiceError, load_runtime
from triage.service.schemas import (
    ErrorResponse,
    ModelInfo,
    ServiceSettings,
    TriageRequest,
    TriageResponse,
)

logger = logging.getLogger("triage.service")


def error_response(request_id, code, message, status):
    return JSONResponse(
        {"request_id": request_id, "error": {"code": code, "message": message}}, status_code=status
    )


class RequestBoundary:
    """ASGI body cap before JSON parsing, including chunked bodies with no Content-Length."""

    def __init__(self, app, *, max_body_bytes, api_key):
        self.app, self.max_body_bytes, self.api_key = app, max_body_bytes, api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.perf_counter()
        scope["state"]["request_started"] = start
        status = 500

        async def tracked_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        async def reject(code, message, http_status):
            await error_response(request_id, code, message, http_status)(
                scope, receive, tracked_send
            )

        try:
            headers = dict(scope["headers"])
            if self.api_key and scope["path"] in ("/v1/triage", "/v1/model"):
                expected = ("Bearer " + self.api_key).encode("utf-8")
                if not secrets.compare_digest(headers.get(b"authorization", b""), expected):
                    return await reject("unauthorized", "Authentication failed", 401)
            raw_length = headers.get(b"content-length")
            if raw_length is not None:
                try:
                    length = int(raw_length)
                except ValueError:
                    return await reject("invalid_request", "Invalid Content-Length", 422)
                if length < 0 or length > self.max_body_bytes:
                    return await reject("request_too_large", "Request body exceeds the limit", 422)
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.max_body_bytes:
                    return await reject("request_too_large", "Request body exceeds the limit", 422)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, tracked_send)
        finally:
            # Do not log text, client IDs, raw paths/query strings, headers, or exception messages.
            logger.info(
                json.dumps(
                    {
                        "event": "request",
                        "status": status,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                    }
                )
            )


def create_app(
    bundle: Path | None = None,
    policy_path: Path | None = None,
    *,
    settings: ServiceSettings | None = None,
    api_key: str | None = None,
    runtime: Runtime | None = None,
):
    settings = settings or ServiceSettings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime
        if runtime is None:
            try:
                if bundle is None or policy_path is None:
                    raise ValueError("Missing bundle or policy")
                app.state.runtime = await asyncio.to_thread(load_runtime, bundle, policy_path)
            except Exception as exc:
                logger.error(
                    json.dumps({"event": "bundle_unavailable", "type": type(exc).__name__})
                )
        yield
        if app.state.runtime is not None:
            app.state.runtime.close()
        app.state.runtime = None

    app = FastAPI(title="Customer request triage", version="0.2.0", lifespan=lifespan)
    app.state.runtime = None
    app.add_middleware(RequestBoundary, max_body_bytes=settings.max_body_bytes, api_key=api_key)
    admitted = deque()
    bearer = HTTPBearer(auto_error=False)

    def require_runtime():
        active = app.state.runtime
        if active is None or not active.ready:
            raise ServiceError("model_unavailable", "Model or policy bundle is unavailable")
        return active

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        response = error_response(request.state.request_id, exc.code, exc.message, exc.status)
        if exc.status == 429:
            response.headers["Retry-After"] = str(int(settings.rate_limit_window_seconds) + 1)
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return error_response(
            request.state.request_id, "invalid_request", "Request must match the input schema", 422
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error_response(
            request.state.request_id, "http_error", "Request not supported", exc.status_code
        )

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready", responses={503: {"model": ErrorResponse}})
    async def ready():
        require_runtime()
        return {"status": "ready"}

    @app.get(
        "/v1/model",
        dependencies=[Depends(bearer)],
        response_model=ModelInfo,
        responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def model():
        active = require_runtime()
        return ModelInfo(
            model_version=active.adapter.model_version,
            policy_version=active.policy.policy_version,
            automatic_routing=active.policy.automatic_routing,
            supported_intent_count=len(active.adapter.catalog),
            fixture=active.adapter.fixture,
        )

    @app.post(
        "/v1/triage",
        dependencies=[Depends(bearer)],
        response_model=TriageResponse,
        responses={code: {"model": ErrorResponse} for code in (401, 422, 429, 503)},
    )
    async def triage(body: TriageRequest, request: Request):
        start = request.state.request_started
        now = time.monotonic()
        while admitted and admitted[0] <= now - settings.rate_limit_window_seconds:
            admitted.popleft()
        if settings.rate_limit_requests:
            if len(admitted) >= settings.rate_limit_requests:
                raise ServiceError("rate_limited", "Request rate limit exceeded", 429)
            admitted.append(now)
        active = require_runtime()
        decision = await active.infer(
            body.text, settings.max_input_tokens, settings.inference_timeout_seconds
        )
        return TriageResponse(
            request_id=request.state.request_id,
            intent=decision.intent,
            decision=decision.decision,
            reason=decision.reason,
            model_version=active.adapter.model_version,
            policy_version=active.policy.policy_version,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    # /metrics is intentionally absent until the private telemetry endpoint is implemented.
    return app
