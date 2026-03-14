"""
FastAPI health-check endpoint (E5-S5).

Exposes:
  GET /healthz  — liveness + RPC connectivity check
  GET /readyz   — readiness check (model trained, contract configured)

Run as a standalone ASGI server alongside the main orchestrator::

    import asyncio
    from src.utils.health import start_health_server
    asyncio.create_task(start_health_server(port=8080))

Or run directly::

    uvicorn src.utils.health:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False
    log.warning("fastapi/uvicorn not installed — health endpoints disabled")

# Module-level app instance (used by uvicorn and create_task)
app = FastAPI(title="Arbisy Health", docs_url=None, redoc_url=None) if _HAS_FASTAPI else None  # type: ignore[assignment]

# Runtime state injected by the orchestrator
_rpc_url: str = os.getenv("POLYGON_ZKEVM_RPC_URL", "")
_contract_address: str = os.getenv("FLASH_LOAN_ARBITRAGE_CONTRACT", "")
_model_trained: bool = False
_nn_model_trained: bool = False
_start_time: float = time.time()


def configure(
    rpc_url: str,
    contract_address: str,
    model_trained: bool,
    nn_model_trained: bool,
) -> None:
    """Called by the orchestrator to inject runtime state."""
    global _rpc_url, _contract_address, _model_trained, _nn_model_trained
    _rpc_url           = rpc_url
    _contract_address  = contract_address
    _model_trained     = model_trained
    _nn_model_trained  = nn_model_trained


async def _check_rpc(rpc_url: str) -> bool:
    """Non-blocking RPC connectivity check."""
    if not rpc_url:
        return False
    try:
        from web3 import AsyncWeb3
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        return await asyncio.wait_for(w3.is_connected(), timeout=3.0)
    except Exception:
        return False


if _HAS_FASTAPI and app is not None:

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """
        Liveness probe.  Returns 200 when the process is running and can reach
        the Polygon zkEVM RPC, or 503 if the RPC is unreachable.
        """
        rpc_ok = await _check_rpc(_rpc_url)
        body: Dict[str, Any] = {
            "status":  "ok" if rpc_ok else "degraded",
            "rpc":     "connected" if rpc_ok else "unreachable",
            "uptime_s": round(time.time() - _start_time, 1),
        }
        code = 200 if rpc_ok else 503
        return JSONResponse(content=body, status_code=code)

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """
        Readiness probe.  Returns 200 only when the ML models are trained and
        the flash-loan contract address is configured.
        """
        ready = _model_trained and bool(_contract_address)
        body: Dict[str, Any] = {
            "status":           "ready" if ready else "not_ready",
            "model_trained":    _model_trained,
            "nn_model_trained": _nn_model_trained,
            "contract":         "configured" if _contract_address else "missing",
        }
        code = 200 if ready else 503
        return JSONResponse(content=body, status_code=code)


async def start_health_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the FastAPI health server as an asyncio task."""
    if not _HAS_FASTAPI:
        log.warning("fastapi/uvicorn not installed — health server not started")
        return
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    log.info("Health server starting on http://%s:%d", host, port)
    await server.serve()
