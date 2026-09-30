from __future__ import annotations

import hashlib
from typing import Any

from .models import Answer, SearchHit


class OllamaKnowledgeError(RuntimeError):
    """Raised when the local Ollama model cannot answer from its internal knowledge."""


def _response_text(response: Any) -> str:
    if isinstance(response, dict):
        return str((response.get("message") or {}).get("content", "")).strip()
    message = getattr(response, "message", None)
    return str(getattr(message, "content", "") if message is not None else "").strip()


class OllamaKnowledgeEngine:
    """Last-resort answer path using only the local model's pretrained knowledge.

    This mode intentionally does not invent external citations.  The UI receives a
    synthetic provenance card that clearly says the answer is model-internal and has
    not been externally verified.
    """

    def __init__(self, model: str, client: Any | None = None):
        self.model = str(model)
        self._client = client

    def _client_or_module(self):
        if self._client is not None:
            return self._client
        import ollama
        return ollama

    def _source_hit(self) -> SearchHit:
        raw = f"ollama-internal:{self.model}".encode("utf-8")
        source_id = hashlib.sha256(raw).hexdigest()[:20]
        return SearchHit(
            chunk_id=source_id,
            doc_id=f"ollama://{self.model}",
            text=(
                f"이 답변은 로컬 Ollama 모델 '{self.model}'의 사전학습된 내부 지식을 사용했습니다. "
                "RAG 문서나 인터넷 검색 결과로 검증된 출처가 아닙니다."
            ),
            score=0.0,
            metadata={
                "source_type": "model_internal",
                "title": f"Ollama 내부지식 · {self.model}",
                "url": "",
                "provider": "ollama",
                "model": self.model,
                "externally_verified": False,
            },
        )

    def ask(self, question: str, *, reason: str = "web_unavailable") -> Answer:
        question = str(question).strip()
        if not question:
            raise ValueError("question is empty")

        system = (
            "사용자의 질문 자체에 직접 답하세요. "
            "답변 본문에서 시스템의 운영 방식이나 데이터 처리 경로를 설명하지 마세요. "
            "모르는 내용은 모른다고 말하고, 확실하지 않은 최신 정보는 불확실하다고 표시하세요. "
            "확인하지 못한 URL·논문·책·기관·통계·날짜를 출처처럼 만들어내지 마세요. "
            "답변은 한국어로 간결하고 명확하게 작성하세요."
        )
        user = f"질문: {question}"
        try:
            response = self._client_or_module().chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0.2},
            )
        except Exception as exc:
            raise OllamaKnowledgeError(f"ollama internal-knowledge query failed: {type(exc).__name__}: {exc}") from exc

        text = _response_text(response)
        if not text:
            raise OllamaKnowledgeError("ollama returned an empty response")

        hit = self._source_hit()
        return Answer(
            question=question,
            text=text,
            citations=(),
            hits=(hit,),
            fallback=False,
            policy=f"ollama-internal:{reason}",
        )
