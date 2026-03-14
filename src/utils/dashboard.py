"""
FastAPI + WebSocket live monitoring dashboard (E5-S4).

Provides a minimal real-time dashboard for the Arbisy system:

Endpoints
---------
GET  /            HTML dashboard page
GET  /api/status  Current system status (JSON)
WS   /ws/feed     WebSocket — streams OpportunityPayload and
                  ExecutionResultPayload events as JSON
GET  /api/metrics Current ML model metrics and execution stats
GET  /api/log     Last N execution log entries (JSON)

Usage
-----
Start the dashboard server alongside the main orchestrator:

    # Standalone (for debugging):
    uvicorn src.utils.dashboard:app --host 0.0.0.0 --port 8080 --reload

    # Or integrate into ArbisyOrchestrator:
    from src.utils.dashboard import dashboard_manager
    dashboard_manager.start(port=8080)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = None  # type: ignore

# ── In-memory event ring buffers ──────────────────────────────────────────────
_MAX_LOG_ENTRIES = 200

_opportunity_log: Deque[Dict[str, Any]] = deque(maxlen=_MAX_LOG_ENTRIES)
_execution_log:   Deque[Dict[str, Any]] = deque(maxlen=_MAX_LOG_ENTRIES)
_model_metrics:   Dict[str, Any] = {}
_system_status:   Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "total_opportunities": 0,
    "total_executions": 0,
    "successful_executions": 0,
    "total_profit_usd": 0.0,
    "consecutive_losses": 0,
    "circuit_breaker_tripped": False,
}

# Active WebSocket connections
_connections: Set[WebSocket] = set()


# ── Public API used by the orchestrator ───────────────────────────────────────

def record_opportunity(payload_dict: Dict[str, Any]) -> None:
    """Record a new opportunity for the dashboard feed."""
    payload_dict["_event"] = "opportunity"
    payload_dict["_ts"] = time.time()
    _opportunity_log.append(payload_dict)
    _system_status["total_opportunities"] += 1
    _schedule_broadcast(payload_dict)


def record_execution(payload_dict: Dict[str, Any]) -> None:
    """Record an execution result for the dashboard feed."""
    payload_dict["_event"] = "execution"
    payload_dict["_ts"] = time.time()
    _execution_log.append(payload_dict)
    _system_status["total_executions"] += 1
    if payload_dict.get("success"):
        _system_status["successful_executions"] += 1
        _system_status["total_profit_usd"] += float(
            payload_dict.get("actual_profit_usd", 0)
        )
        _system_status["consecutive_losses"] = 0
    else:
        _system_status["consecutive_losses"] += 1
        if _system_status["consecutive_losses"] >= 3:
            _system_status["circuit_breaker_tripped"] = True
    _schedule_broadcast(payload_dict)


def update_model_metrics(metrics: Dict[str, Any]) -> None:
    """Update the cached model metrics (called after each retrain)."""
    _model_metrics.update(metrics)
    _model_metrics["updated_at"] = time.time()


def set_running(running: bool) -> None:
    _system_status["running"] = running
    if running and _system_status["started_at"] is None:
        _system_status["started_at"] = time.time()


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def _broadcast(message: Dict[str, Any]) -> None:
    """Broadcast a JSON event to all connected WebSocket clients."""
    if not _connections:
        return
    text = json.dumps(message, default=str)
    dead: Set[WebSocket] = set()
    for ws in list(_connections):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


def _schedule_broadcast(message: Dict[str, Any]) -> None:
    """Schedule a broadcast on the running event loop, if one exists."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast(message))
    except RuntimeError:
        # No running loop (e.g. called from a non-async context in tests)
        pass


# ── HTML dashboard ────────────────────────────────────────────────────────────
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arbisy — Live Dashboard</title>
<style>
  body { font-family: monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 16px; }
  h1 { color: #58a6ff; }
  h2 { color: #79c0ff; margin-top: 1em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
  .stat { font-size: 2em; color: #3fb950; }
  .stat-label { color: #8b949e; font-size: 0.85em; }
  .green { color: #3fb950; }
  .red   { color: #f85149; }
  .yellow{ color: #d29922; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85em; }
  th { background: #21262d; color: #8b949e; text-align: left; padding: 6px 8px; }
  td { padding: 4px 8px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #161b22; }
  #status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                background: #3fb950; margin-right: 6px; }
  #status-dot.offline { background: #f85149; }
</style>
</head>
<body>
<h1><span id="status-dot"></span> Arbisy Live Dashboard</h1>

<div class="grid">
  <div class="card">
    <div class="stat-label">Total Profit (USD)</div>
    <div class="stat green" id="total-profit">$0.00</div>
  </div>
  <div class="card">
    <div class="stat-label">Executions (success / total)</div>
    <div class="stat" id="exec-ratio">0 / 0</div>
  </div>
  <div class="card">
    <div class="stat-label">Opportunities Found</div>
    <div class="stat" id="opp-count">0</div>
  </div>
  <div class="card">
    <div class="stat-label">Circuit Breaker</div>
    <div class="stat green" id="cb-status">OK</div>
  </div>
</div>

<h2>Live Event Feed</h2>
<table id="feed-table">
  <thead><tr><th>Time</th><th>Type</th><th>Details</th><th>Result</th></tr></thead>
  <tbody id="feed-body"></tbody>
</table>

<script>
const ws = new WebSocket("ws://" + location.host + "/ws/feed");

ws.onopen = () => {
  document.getElementById("status-dot").classList.remove("offline");
  fetch("/api/status").then(r => r.json()).then(updateStats);
};
ws.onclose = () => document.getElementById("status-dot").classList.add("offline");

ws.onmessage = (evt) => {
  const data = JSON.parse(evt.data);
  addFeedRow(data);
  fetch("/api/status").then(r => r.json()).then(updateStats);
};

function updateStats(s) {
  document.getElementById("total-profit").textContent = "$" + (s.total_profit_usd || 0).toFixed(2);
  document.getElementById("exec-ratio").textContent =
    (s.successful_executions || 0) + " / " + (s.total_executions || 0);
  document.getElementById("opp-count").textContent = s.total_opportunities || 0;
  const cb = document.getElementById("cb-status");
  if (s.circuit_breaker_tripped) {
    cb.textContent = "TRIPPED";
    cb.className = "stat red";
  } else {
    cb.textContent = "OK";
    cb.className = "stat green";
  }
}

function addFeedRow(data) {
  const tbody = document.getElementById("feed-body");
  const tr = document.createElement("tr");
  const ts = new Date((data._ts || Date.now() / 1000) * 1000).toLocaleTimeString();
  const type = data._event || "?";
  let details = "";
  let result = "";
  if (type === "opportunity") {
    details = (data.arb_type || "") + " | score=" + (data.ml_score || 0).toFixed(3) +
              " | $" + (data.expected_profit_usd || 0).toFixed(2);
    result = '<span class="yellow">PENDING</span>';
  } else if (type === "execution") {
    details = data.tx_hash ? data.tx_hash.slice(0, 12) + "..." : "(no tx)";
    result = data.success
      ? '<span class="green">✓ $' + (data.actual_profit_usd || 0).toFixed(2) + '</span>'
      : '<span class="red">✗ ' + (data.error_message || "failed") + '</span>';
  }
  tr.innerHTML = "<td>" + ts + "</td><td>" + type + "</td><td>" + details + "</td><td>" + result + "</td>";
  tbody.insertBefore(tr, tbody.firstChild);
  // Keep only 50 rows in the DOM
  while (tbody.rows.length > 50) tbody.deleteRow(tbody.rows.length - 1);
}
</script>
</body>
</html>
"""


# ── FastAPI application ───────────────────────────────────────────────────────

if _HAS_FASTAPI:
    app = FastAPI(title="Arbisy Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page():
        return _DASHBOARD_HTML

    @app.get("/api/status")
    async def api_status():
        return dict(_system_status)

    @app.get("/api/metrics")
    async def api_metrics():
        return dict(_model_metrics)

    @app.get("/api/log")
    async def api_log(n: int = 50):
        opp_list  = list(_opportunity_log)[-n:]
        exec_list = list(_execution_log)[-n:]
        return {"opportunities": opp_list, "executions": exec_list}

    @app.websocket("/ws/feed")
    async def ws_feed(websocket: WebSocket):
        await websocket.accept()
        _connections.add(websocket)
        try:
            while True:
                # Keep alive — client drives the connection
                await asyncio.sleep(30)
                await websocket.send_text(json.dumps({"_event": "ping"}))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            _connections.discard(websocket)

else:
    # Stub so imports don't break when fastapi is absent
    app = None  # type: ignore


# ── Background server manager ─────────────────────────────────────────────────

class DashboardManager:
    """
    Manages the uvicorn server as a background asyncio task.

    Usage (inside an async context):
        dm = DashboardManager()
        await dm.start(port=8080)
        ...
        dm.stop()
    """

    def __init__(self) -> None:
        self._server: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        if not _HAS_FASTAPI:
            log.warning("Dashboard disabled — fastapi/uvicorn not installed")
            return
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            loop="none",   # Use existing asyncio loop
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(
            self._server.serve(), name="dashboard_server"
        )
        log.info("Dashboard server started at http://%s:%d", host, port)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            self._task.cancel()


dashboard_manager = DashboardManager()
