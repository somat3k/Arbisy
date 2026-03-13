"""
Base AI agent class.

Supports multiple LLM providers (Groq, OpenRouter, Gemini, OpenAI) with:
- Automatic provider rotation on rate-limit (HTTP 429)
- Retry logic with exponential back-off
- Structured JSON output parsing
- Async HTTP via httpx
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class LLMProvider(str, Enum):
    GROQ        = "groq"
    OPENROUTER  = "openrouter"
    GEMINI      = "gemini"
    OPENAI      = "openai"


# Provider base URLs and default models
PROVIDER_CONFIG: Dict[LLMProvider, Dict[str, str]] = {
    LLMProvider.GROQ: {
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama3-70b-8192",
        "chat_path": "/chat/completions",
    },
    LLMProvider.OPENROUTER: {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "meta-llama/llama-3-70b-instruct",
        "chat_path": "/chat/completions",
    },
    LLMProvider.GEMINI: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model":    "gemini-1.5-flash",
        "chat_path": "/models/{model}:generateContent",
    },
    LLMProvider.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "model":    "gpt-4o-mini",
        "chat_path": "/chat/completions",
    },
}

PROVIDER_ORDER = [
    LLMProvider.GROQ,
    LLMProvider.OPENROUTER,
    LLMProvider.GEMINI,
    LLMProvider.OPENAI,
]


@dataclass
class LLMResponse:
    content: str
    provider: LLMProvider
    model: str
    latency_ms: float
    parsed: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Abstract base for all AI agents.

    Subclasses implement `system_prompt` and `process`.
    """

    def __init__(
        self,
        name: str,
        preferred_provider: Optional[LLMProvider] = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self._cfg = get_config()
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._provider_index = PROVIDER_ORDER.index(
            preferred_provider or LLMProvider.GROQ
        )
        self._rate_limited_until: Dict[LLMProvider, float] = {}
        self._http: Optional[httpx.AsyncClient] = None
        log.info("Agent '%s' initialised (preferred=%s)", name, self._current_provider)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _current_provider(self) -> LLMProvider:
        return PROVIDER_ORDER[self._provider_index % len(PROVIDER_ORDER)]

    def _next_provider(self) -> LLMProvider:
        self._provider_index += 1
        return self._current_provider

    def _api_key(self, provider: LLMProvider) -> str:
        key_map = {
            LLMProvider.GROQ:       self._cfg.groq_api_key,
            LLMProvider.OPENROUTER: self._cfg.openrouter_api_key,
            LLMProvider.GEMINI:     self._cfg.gemini_api_key,
            LLMProvider.OPENAI:     self._cfg.openai_api_key,
        }
        return key_map[provider]

    # ── HTTP client ───────────────────────────────────────────────────────────

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── LLM call (OpenAI-compatible) ──────────────────────────────────────────

    async def _call_openai_compat(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
    ) -> LLMResponse:
        cfg = PROVIDER_CONFIG[provider]
        url  = cfg["base_url"] + cfg["chat_path"]
        key  = self._api_key(provider)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":    cfg["model"],
            "messages": messages,
        }

        client  = await self._get_http()
        t0      = time.monotonic()
        resp    = await client.post(url, json=payload, headers=headers)
        latency = (time.monotonic() - t0) * 1000

        if resp.status_code == 429:
            raise httpx.HTTPStatusError("Rate limited", request=resp.request, response=resp)

        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"]

        return LLMResponse(
            content=content,
            provider=provider,
            model=cfg["model"],
            latency_ms=latency,
        )

    async def _call_gemini(
        self,
        messages: List[Dict[str, str]],
    ) -> LLMResponse:
        cfg   = PROVIDER_CONFIG[LLMProvider.GEMINI]
        model = cfg["model"]
        key   = self._api_key(LLMProvider.GEMINI)
        url   = f"{cfg['base_url']}/models/{model}:generateContent?key={key}"

        # Convert OpenAI-style messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        client  = await self._get_http()
        t0      = time.monotonic()
        resp    = await client.post(url, json={"contents": contents})
        latency = (time.monotonic() - t0) * 1000

        if resp.status_code == 429:
            raise httpx.HTTPStatusError("Rate limited", request=resp.request, response=resp)

        resp.raise_for_status()
        data    = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]

        return LLMResponse(
            content=content,
            provider=LLMProvider.GEMINI,
            model=model,
            latency_ms=latency,
        )

    # ── Main call with retry + provider rotation ──────────────────────────────

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        expect_json: bool = False,
    ) -> LLMResponse:
        """
        Call the LLM with automatic retry and provider rotation.

        Parameters
        ----------
        messages:    OpenAI-style message list.
        expect_json: If True, attempt to parse the response as JSON.
        """
        last_error: Optional[Exception] = None
        now = time.time()

        for attempt in range(self._max_retries * len(PROVIDER_ORDER)):
            provider = self._current_provider

            # Skip rate-limited providers
            if self._rate_limited_until.get(provider, 0) > now:
                self._next_provider()
                continue

            api_key = self._api_key(provider)
            if not api_key:
                log.debug("No API key for %s — skipping", provider)
                self._next_provider()
                continue

            try:
                if provider == LLMProvider.GEMINI:
                    response = await self._call_gemini(messages)
                else:
                    response = await self._call_openai_compat(provider, messages)

                if expect_json:
                    response.parsed = self._parse_json(response.content)

                log.debug(
                    "[%s] %s responded in %.0fms",
                    self.name, provider, response.latency_ms
                )
                return response

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    log.warning("[%s] %s rate-limited — rotating", self.name, provider)
                    self._rate_limited_until[provider] = time.time() + 60
                    self._next_provider()
                else:
                    log.error("[%s] HTTP %d from %s", self.name, exc.response.status_code, provider)
                last_error = exc

            except Exception as exc:
                log.error("[%s] %s call failed: %s", self.name, provider, exc)
                last_error = exc
                await asyncio.sleep(min(2 ** (attempt % self._max_retries), 30))

        raise RuntimeError(
            f"All LLM providers exhausted for agent '{self.name}'"
        ) from last_error

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, tolerating markdown code fences."""
        import re
        # Strip ```json ... ``` fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find first {...} block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""

    @abstractmethod
    async def process(self, *args: Any, **kwargs: Any) -> Any:
        """Main processing logic — implemented by each subclass."""
