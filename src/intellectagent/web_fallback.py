from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from .grounding import validate_grounding
from .models import Answer, SearchHit


class WebSearchError(RuntimeError):
    """Raised when a configured web-search provider cannot return usable evidence."""


class WebSearchProvider(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[SearchHit]: ...


@dataclass(frozen=True)
class WebSearchSettings:
    enabled: bool = False
    max_results: int = 5
    region: str = "kr-kr"
    safesearch: str = "moderate"
    timeout_seconds: int = 8


class DDGSWebSearchProvider:
    """Search-only provider using the current ``ddgs`` package.

    Only search-result metadata/snippets are passed to the LLM. The application does
    not automatically fetch arbitrary result URLs, reducing SSRF and hostile-page
    ingestion risk. URLs are retained as source metadata for the UI.
    """

    def __init__(self, *, region: str = "kr-kr", safesearch: str = "moderate", timeout_seconds: int = 8):
        self.region = region
        self.safesearch = safesearch
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _chunk_id(url: str, title: str, body: str) -> str:
        raw = f"{url}\n{title}\n{body}".encode("utf-8", errors="replace")
        return hashlib.sha256(raw).hexdigest()[:20]

    def search(self, query: str, *, max_results: int = 5) -> list[SearchHit]:
        query = str(query).strip()
        if not query:
            return []
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise WebSearchError("ddgs is not installed. Run: pip install -r requirements.txt") from exc

        try:
            rows = DDGS(timeout=self.timeout_seconds).text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max(1, int(max_results)),
            )
        except Exception as exc:
            raise WebSearchError(f"web search failed: {type(exc).__name__}: {exc}") from exc

        hits: list[SearchHit] = []
        for rank, row in enumerate(rows or [], start=1):
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            url = str(row.get("href") or row.get("url") or "").strip()
            body = str(row.get("body") or row.get("snippet") or row.get("description") or "").strip()
            if not (title or body):
                continue
            cid = self._chunk_id(url, title, body)
            text = "\n".join(part for part in (title, body) if part)
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    doc_id=url or f"web-result-{rank}",
                    text=text,
                    score=1.0 / rank,
                    metadata={
                        "source_type": "web",
                        "title": title,
                        "url": url,
                        "rank": rank,
                        "provider": "ddgs",
                    },
                )
            )
        return hits


class WebFallbackEngine:
    """Generate a grounded answer only from web-search result snippets."""

    def __init__(self, provider: WebSearchProvider, generator: Any, *, max_results: int = 5):
        self.provider = provider
        self.generator = generator
        self.max_results = max(1, int(max_results))

    def ask(self, question: str) -> Answer:
        hits = self.provider.search(question, max_results=self.max_results)
        if not hits:
            return Answer(
                question,
                "RAG 문서가 없고 인터넷 검색에서도 근거를 찾지 못했습니다.",
                (),
                (),
                True,
                policy="web-fallback",
            )
        text = self.generator.generate(question, hits)
        grounding = validate_grounding(text, hits)
        if not grounding.ok:
            return Answer(
                question,
                "인터넷 검색 결과는 찾았지만 근거 검증을 통과하지 못해 답변을 제공하지 않습니다.",
                (),
                tuple(hits),
                True,
                policy="web-fallback",
            )
        return Answer(
            question,
            text,
            grounding.cited_ids,
            tuple(hits),
            False,
            policy="web-fallback",
        )


def settings_from_feature_flags(flags: dict[str, Any]) -> WebSearchSettings:
    return WebSearchSettings(
        enabled=bool(flags.get("use_empty_rag_web_fallback", False)),
        max_results=max(1, int(flags.get("web_search_max_results", 5))),
        region=str(flags.get("web_search_region", "kr-kr")),
        safesearch=str(flags.get("web_search_safesearch", "moderate")),
        timeout_seconds=max(1, int(flags.get("web_search_timeout_seconds", 8))),
    )
