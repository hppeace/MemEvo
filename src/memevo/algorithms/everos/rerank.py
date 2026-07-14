"""Rerank providers needed by the MemEvo EverOS adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
from everos.component.rerank import RerankResult, RerankServiceError


class Qwen3DashScopeRerankProvider:
    """DashScope's OpenAI-compatible endpoint for ``qwen3-rerank``."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> None:
        if model != "qwen3-rerank":
            raise ValueError(f"unsupported Qwen3 rerank model: {model!r}")
        if not api_key:
            raise ValueError("DashScope rerank API key is empty")
        if timeout <= 0 or batch_size < 1 or max_concurrent < 1 or max_retries < 0:
            raise ValueError("invalid DashScope rerank request limits")

        self._model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._slots = asyncio.Semaphore(max_concurrent)

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        chunks = [
            (offset, list(documents[offset : offset + self._batch_size]))
            for offset in range(0, len(documents), self._batch_size)
        ]
        partials = await asyncio.gather(
            *(self._score_chunk(query, chunk, instruction) for _, chunk in chunks)
        )
        results = [
            RerankResult(index=offset + result.index, score=result.score)
            for (offset, _), partial in zip(chunks, partials, strict=True)
            for result in partial
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results

    async def _score_chunk(
        self,
        query: str,
        documents: list[str],
        instruction: str | None,
    ) -> list[RerankResult]:
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        if instruction is not None:
            payload["instruct"] = instruction

        body = await self._request(payload)
        items = body.get("results")
        if not isinstance(items, list):
            raise RerankServiceError(
                f"DashScope qwen3-rerank response missing results: {body!r}"
            )

        results: list[RerankResult] = []
        try:
            for item in items:
                results.append(
                    RerankResult(
                        index=int(item["index"]),
                        score=float(item["relevance_score"]),
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankServiceError(
                f"malformed qwen3-rerank result entry: {item!r}"
            ) from exc
        return results

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with self._slots:
            for attempt in range(self._max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.post(
                            self._url, json=payload, headers=headers
                        )
                except httpx.HTTPError as exc:
                    if attempt == self._max_retries:
                        raise RerankServiceError(
                            f"DashScope qwen3-rerank transport failure: {exc}"
                        ) from exc
                    continue

                if response.status_code == 200:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise RerankServiceError(
                            "DashScope qwen3-rerank returned invalid JSON"
                        ) from exc
                    if not isinstance(body, dict):
                        raise RerankServiceError(
                            "DashScope qwen3-rerank returned a non-object response"
                        )
                    return body

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self._max_retries:
                        continue
                raise RerankServiceError(
                    f"DashScope qwen3-rerank HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

        raise AssertionError("unreachable")
