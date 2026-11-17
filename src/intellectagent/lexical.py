"""어휘(BM25) 검색. 외부 의존성 없이 동작하고, 형태소 분석기(Kiwi)는 선택 사항이다.

설계 메모
- 한국어는 조사가 붙어 ``서버는`` / ``서버가`` / ``서버를`` 이 서로 다른 토큰이 된다.
  조사를 정규식으로 *떼어 내면* ``지도`` -> ``지`` 처럼 2글자 명사가 사라진다(IntellectAgent_AI 의 결함).
  그래서 기본 토크나이저는 **어절 내부 문자 bigram** 으로 한다: ``지도는`` -> ``지도``, ``도는``.
  조사만 다른 어절이 공통 bigram(``지도``)으로 만나므로 사전 없이도 조사 변이를 흡수한다.
- BM25 수식과 idf 하한 처리는 rank_bm25.BM25Okapi 와 같게 맞췄다(교차검증 테스트 존재).
  주의: Okapi idf 는 "문서의 절반에 나오는 낱말"에서 0 이 된다. 코퍼스가 2~3건이면 거의 모든 낱말이 이에 해당해
  점수가 전부 0 이 될 수 있다. 소규모 코퍼스는 ``idf_variant="lucene"`` 을 쓴다(기본값은 AutoRAG 와의 대조를 위해 okapi).
- 토크나이저 이름은 AutoRAG 설정 이름(``ko_kiwi``)과 맞춰 결과 표를 서로 대조하기 쉽게 했다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Sequence

from .models import Chunk, SearchHit, index_text

_WORD_RE = re.compile(r"[가-힣a-zA-Z0-9]+")

Tokenizer = Callable[[str], list[str]]


def tok_word(text: str) -> list[str]:
    """어절 단위(조사 변이를 흡수하지 못하는 기준선)."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def tok_char_bigram(text: str) -> list[str]:
    """어절 내부 문자 bigram. 1글자 어절은 그대로 둔다(영문/숫자 어절은 어절 전체도 함께 넣는다)."""
    out: list[str] = []
    for w in _WORD_RE.findall(text):
        w = w.lower()
        if re.fullmatch(r"[a-z0-9]+", w):
            out.append(w)  # 영문·숫자 토큰은 쪼개지 않는다 (예: "bm25", "api")
            continue
        if len(w) == 1:
            out.append(w)
        else:
            out.extend(w[i : i + 2] for i in range(len(w) - 1))
    return out


def tok_ko_kiwi(text: str) -> list[str]:  # pragma: no cover - 선택 의존성
    """Kiwi 형태소 분석 (``pip install kiwipiepy``). 명사·어근·동사 어간 등 의미 형태소만 남긴다."""
    try:
        from kiwipiepy import Kiwi
    except ImportError as exc:
        raise ImportError("ko_kiwi 토크나이저는 kiwipiepy 가 필요합니다: pip install kiwipiepy") from exc
    global _KIWI
    try:
        _KIWI
    except NameError:
        _KIWI = Kiwi()
    keep = ("NN", "NR", "NP", "VV", "VA", "XR", "SL", "SN", "MAG")
    return [t.form.lower() for t in _KIWI.tokenize(text) if t.tag.startswith(keep)]


TOKENIZERS: dict[str, Tokenizer] = {
    "word": tok_word,
    "char_bigram": tok_char_bigram,
    "ko_kiwi": tok_ko_kiwi,
}


class BM25Index:
    """BM25Okapi (k1=1.5, b=0.75, epsilon=0.25 — rank_bm25 기본값과 동일)."""

    def __init__(self, chunks: Sequence[Chunk], tokenizer: Tokenizer = tok_char_bigram,
                 k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25, idf_variant: str = "okapi"):
        if idf_variant not in ("okapi", "lucene"):
            raise ValueError("idf_variant must be 'okapi' or 'lucene'")
        self.idf_variant = idf_variant
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self._docs = [Counter(tokenizer(index_text(c))) for c in self.chunks]
        self._len = [sum(d.values()) for d in self._docs]
        self.corpus_size = len(self.chunks)
        self.avgdl = (sum(self._len) / self.corpus_size) if self.corpus_size else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        idf: dict[str, float] = {}
        idf_sum = 0.0
        negative: list[str] = []
        for w, n in df.items():
            if idf_variant == "lucene":  # 항상 양수: 문서가 아주 적은 코퍼스(예: 샘플 2건)에서도 점수가 0 이 되지 않는다
                idf[w] = math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5))
                continue
            v = math.log(self.corpus_size - n + 0.5) - math.log(n + 0.5)
            idf[w] = v
            idf_sum += v
            if v < 0:
                negative.append(w)
        self.average_idf = idf_sum / len(idf) if idf else 0.0
        eps = self.epsilon * self.average_idf
        for w in negative:
            idf[w] = eps
        self.idf = idf

    def scores(self, query: str) -> list[float]:
        q = self.tokenizer(query)
        out = [0.0] * self.corpus_size
        if not self.corpus_size or self.avgdl == 0:
            return out
        for i, d in enumerate(self._docs):
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self.avgdl)
            s = 0.0
            for w in q:
                f = d.get(w, 0)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:top_k]
        # 점수 0 은 "일치 없음" — 근거로 내보내지 않는다(우연한 순서가 정답처럼 보이는 것을 막는다).
        return [
            SearchHit(self.chunks[i].chunk_id, self.chunks[i].doc_id, self.chunks[i].text,
                      float(sc[i]), dict(self.chunks[i].metadata))
            for i in order if sc[i] > 0
        ]
