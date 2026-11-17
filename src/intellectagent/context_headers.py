"""문맥 머리말: 청크가 '어느 문서의 어느 장·절' 인지 색인에만 덧붙인다 (Anthropic 의 contextual retrieval 과 같은 발상이지만 LLM 호출이 없다).

예) 청크 "30일간 보관한다" -> 색인 글: "[서버운영지침.md > 3장 백업 > 3.2 보관 기간]\\n30일간 보관한다"
- 머리말은 ``chunk.metadata["context"]`` 에만 두고 ``chunk.text`` 는 그대로 둔다. 그래서 근거 표시·인용 검증(grounding)·토큰 지표(정답 근거와 청크 텍스트 비교)는 영향을 받지 않는다.
  임베딩/BM25 입력만 ``models.index_text`` 를 거친다.
- 제목 감지는 보수적이다(오탐이 색인을 오염시키므로): 마크다운 '#', '제N장/절/편', '1.' '1.2' 같은 번호 + 짧은 제목,
  그리고 파서가 넣는 표지 '[슬라이드 n]' '[시트 이름]' '[발표자 노트]'. 제목이 없는 문서(PDF 등)는 파일 이름만 붙는다.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

from .models import Chunk, Document

_MD = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")
_KO = re.compile(r"^(제\s*\d+\s*[장절편부]\b.*)$")
_NUM = re.compile(r"^((?:\d+\.)+\d*|\d+[.)])\s+(\S.*)$")
_MARK = re.compile(r"^\[((?:슬라이드|시트|발표자 노트|각주/미주)[^\]]*)\]\s*$")
_SENTENCE_END = re.compile(r"[.!?。]$|(?:다|요|함|음)\.?$")
MAX_TITLE = 60


def find_headings(text: str) -> list[tuple[int, int, str]]:
    """(문자 위치, 깊이, 제목) 목록. 깊이 1 이 가장 상위."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.split("\n"):
        s = line.strip()
        start = pos
        pos += len(line) + 1
        if not s or len(s) > 120:
            continue
        if (m := _MD.match(s)) and len(m.group(2)) <= 100:
            out.append((start, len(m.group(1)), m.group(2)))
        elif (m := _MARK.match(s)):
            out.append((start, 1, m.group(1)))
        elif (m := _KO.match(s)) and len(s) <= MAX_TITLE:
            out.append((start, 1, re.sub(r"\s+", " ", m.group(1))))
        elif (m := _NUM.match(s)) and len(s) <= MAX_TITLE and not _SENTENCE_END.search(s):
            depth = m.group(1).rstrip(".)").count(".") + 1
            out.append((start, depth, s))
    return out


def _path_at(headings: Sequence[tuple[int, int, str]], offset: int) -> list[str]:
    stack: list[tuple[int, str]] = []
    for pos, depth, title in headings:
        if pos > offset:
            break
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, title))
    return [t for _, t in stack]


def add_context_headers(doc: Document, chunks: Sequence[Chunk], max_len: int = 160) -> list[Chunk]:
    """각 청크에 ``metadata["context"]`` 를 단다. 청크가 원문의 어디인지 찾지 못하면 직전 위치의 제목을 쓴다."""
    name = str(doc.metadata.get("name") or doc.path.name)
    headings = find_headings(doc.text)
    out: list[Chunk] = []
    cursor = 0
    for c in chunks:
        probe = c.text[:60]
        at = doc.text.find(probe, cursor) if probe else -1
        if at < 0:
            at = doc.text.find(probe) if probe else -1
        if at >= 0:
            cursor = at + 1          # 겹치는 청킹(overlap)에서도 다음 청크가 앞쪽에서 시작할 수 있다
        else:
            at = cursor
        parts = [name, *_path_at(headings, at)]
        header = "[" + " > ".join(parts) + "]"
        if len(header) > max_len:
            header = header[: max_len - 2] + "…]"
        out.append(replace(c, metadata={**c.metadata, "context": header}))
    return out
