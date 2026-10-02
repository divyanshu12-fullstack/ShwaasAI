"""Deployment settings and bounded per-process HTTP controls."""

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
import re
import time
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from backend.app.core.config import ConfigurationError


UUID_SEGMENT = re.compile(r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?=/|$)")
MAX_BUCKETS = 10_000


def _positive_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DeploymentSettings:
    rate_limit_per_minute: int = 120
    analyze_rate_limit_per_minute: int = 12

    @classmethod
    def from_env(cls) -> "DeploymentSettings":
        return cls(
            rate_limit_per_minute=_positive_env("RATE_LIMIT_PER_MINUTE", 120),
            analyze_rate_limit_per_minute=_positive_env("ANALYZE_RATE_LIMIT_PER_MINUTE", 12),
        )


class RateLimiter:
    """Sliding 60-second limits, keyed by transport peer IP, per worker."""

    def __init__(self, settings: DeploymentSettings) -> None:
        self.settings = settings
        self.buckets: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self.lock = asyncio.Lock()

    async def retry_after(self, peer: str, analysis_identity: str | None) -> int | None:
        now = time.monotonic()
        limits = [(peer, "api", self.settings.rate_limit_per_minute)]
        if analysis_identity is not None:
            limits.append((analysis_identity, "analysis", self.settings.analyze_rate_limit_per_minute))
        async with self.lock:
            retry_after = None
            for identity, kind, maximum in limits:
                bucket = self.buckets.setdefault((identity, kind), deque())
                while bucket and now - bucket[0] >= 60:
                    bucket.popleft()
                self.buckets.move_to_end((identity, kind))
                if len(bucket) >= maximum:
                    retry_after = max(1, math.ceil(60 - (now - bucket[0])))
                    break
            if retry_after is None:
                for identity, kind, _ in limits:
                    self.buckets[(identity, kind)].append(now)
            while len(self.buckets) > MAX_BUCKETS:
                self.buckets.popitem(last=False)
        return retry_after


def _request_logger() -> logging.Logger:
    logger = logging.getLogger("shwaasai.http")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class DeploymentMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: DeploymentSettings) -> None:
        super().__init__(app)
        self.limiter = RateLimiter(settings)
        self.logger = _request_logger()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.monotonic()
        request_id = uuid4().hex
        request.state.request_id = request_id
        path = request.url.path
        status_code = 500
        error_type = None
        try:
            if path.startswith("/api/v1/") and path != "/api/v1/health":
                peer = request.client.host if request.client else "unknown"
                is_analysis = request.method == "POST" and path in (
                    "/api/v1/analyze", "/api/v1/analyze/audio", "/api/v1/analyze/upload"
                )
                analysis_identity = None
                if is_analysis:
                    bearer = request.headers.get("authorization", "")
                    if bearer.lower().startswith("bearer "):
                        digest = hashlib.sha256(bearer.encode()).hexdigest()[:16]
                        analysis_identity = f"{peer}:{digest}"
                    else:
                        analysis_identity = peer
                retry_after = await self.limiter.retry_after(peer, analysis_identity)
                if retry_after is not None:
                    result = JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded", "request_id": request_id},
                        headers={"Retry-After": str(retry_after)},
                    )
                else:
                    result = await call_next(request)
            else:
                result = await call_next(request)
            status_code = result.status_code
        except Exception as exc:
            error_type = type(exc).__name__
            result = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        result.headers["X-Request-ID"] = request_id
        event = {
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": UUID_SEGMENT.sub("/{id}", path),
            "status": status_code,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
        }
        if error_type is not None:
            event["error_type"] = error_type
        self.logger.info(json.dumps(event, separators=(",", ":")))
        return result
