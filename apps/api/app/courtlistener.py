from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CourtListenerError(RuntimeError):
    """Raised for CourtListener client errors."""


class CourtListenerClient:
    def __init__(
        self,
        base_url: str = "https://www.courtlistener.com",
        token_env: str = "COURTLISTENER_TOKEN",
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)

    def _get_token(self) -> str:
        token = os.getenv(self.token_env, "").strip()
        if not token:
            try:
                from app.settings import settings

                token = (settings.courtlistener_token or "").strip()
            except Exception:
                token = ""
        if not token:
            raise CourtListenerError(
                f"Missing CourtListener token in environment variable: {self.token_env}"
            )
        return token

    def lookup_citations(self, text: str) -> dict:
        cleaned = text.strip()
        if not cleaned:
            raise CourtListenerError("Citation lookup text must be non-empty")

        token = self._get_token()
        url = f"{self.base_url}/api/rest/v4/citation-lookup/"
        payload = json.dumps({"text": cleaned}).encode("utf-8")
        request = Request(url=url, data=payload, method="POST")
        request.add_header("Authorization", f"Token {token}")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")

        body = ""
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if retryable and attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise CourtListenerError(
                    f"CourtListener request failed with status {exc.code}: {detail[:300]}"
                ) from exc
            except URLError as exc:
                raise CourtListenerError(
                    f"CourtListener request failed: {exc.reason}"
                ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise CourtListenerError(
                "CourtListener response was not valid JSON"
            ) from exc

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"results": parsed}
        return {"results": [parsed]}

    def lookup_citation_list(self, citations: list[str]) -> dict:
        cleaned_citations = [
            citation.strip() for citation in citations if citation.strip()
        ]
        if not cleaned_citations:
            raise CourtListenerError("Citation list must be non-empty")
        return self.lookup_citations("\n".join(cleaned_citations))
