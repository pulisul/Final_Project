"""질문과 문서 본문을 같은 방식으로 정리하는 한 곳.

질문과 색인된 본문이 서로 다른 표기(자모 분리, 눈에 안 보이는 글자)를 가지면 같은 글자인데도 검색이 어긋난다.
로더(본문)와 CoreRAG·워크플로(질문)가 모두 이 함수를 쓰게 해서 두 쪽이 갈라지지 않게 한다.
"""
from __future__ import annotations

import unicodedata

# 눈에 보이지 않는 글자: 제로폭 공백/비접합/접합, 단어 결합자, BOM, 소프트 하이픈
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def clean_text(text: str) -> str:
    """NFC(완성형) 로 맞추고 눈에 안 보이는 글자를 지운다. 이미 깨끗한 글은 그대로 돌려준다."""
    return unicodedata.normalize("NFC", text).translate(_INVISIBLE)
