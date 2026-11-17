from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from .models import SearchHit

_ID_RE = re.compile(r"\[([0-9a-f]{20})\]")
_NUM_RE = re.compile(r"\[(\d{1,3}(?:\s*[,，]\s*\d{1,3})*)\]")   # [1] [1,2] — 프롬프트에 번호 인용(citation_style=index)을 쓸 때
_LABEL_MAX = 20
_HEADING_MAX = 40                       # 제목줄·굵은 글씨 한 줄 머리말로 인정하는 최대 길이
REFUSAL_PHRASE = "관련 근거 문서를 찾지 못했습니다"            # 프롬프트가 모델에게 시키는 '근거 없음' 답
_REFUSAL_TAIL = "질문을 더 구체적으로 바꿔 주세요"
_BRACKETS = str.maketrans({"［": "[", "］": "]", "【": "[", "】": "]", "〔": "[", "〕": "]"})
_LEAD_MD = re.compile(r"^\s*(?:#{1,6}\s+|>+\s*|[-*•·]\s+|\d{1,2}[.)]\s+)")   # 제목·인용·글머리·번호 목록의 머리
_SENTENCE_END = ("다", "요", "음", "임", "함", ".", "!", "?", "。")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SOURCE_LABEL = re.compile(r"^(?:출처|근거|참고|참조|인용|sources?|references?)\s*[:：]?$", re.I)   # '출처: [1]' 꼬리줄


@dataclass(frozen=True)
class GroundingResult:
    ok: bool
    cited_ids: tuple[str,...]
    unsupported_citations: tuple[str,...]
    citation_coverage: float
    unsupported_numbers: tuple[str,...] = ()   # check_numbers=True 일 때만 채워진다
    model_refused: bool = False                # 모델이 스스로 '근거 없음'이라고 답함(인용 형식 위반이 아님)


def _core(line: str) -> str:
    """인용 표기와 마크다운 장식(제목 #, 굵게 **, 글머리, 번호)을 뺀 본문."""
    t = _NUM_RE.sub(" ", _ID_RE.sub(" ", line))
    t = _LEAD_MD.sub("", t)
    return re.sub(r"[*_`]+", "", t).strip()


def _is_label(line: str, core: str | None = None) -> bool:
    """'답변:' 같은 머리말 줄. 주장이 아니므로 인용을 요구하지 않는다.
    - 짧고(20자 이하) 콜론으로 끝나는 줄 ('**답변:**' 처럼 굵은 글씨 장식이 있어도 인정)
    - 제목줄(#)이나 줄 전체가 굵은 글씨인 짧은 줄. 단, 문장처럼 끝나는 줄(…다/…요/마침표)은 주장으로 본다."""
    core = _core(line) if core is None else core
    if len(core) <= _LABEL_MAX and core.endswith((":", "：")):
        return True
    t = line.strip()
    decorated = t.startswith("#") or bool(re.match(r"^(\*\*|__).+(\*\*|__)\s*[:：]?\s*$", t))
    return decorated and 0 < len(core) <= _HEADING_MAX and not core.rstrip(":：").endswith(_SENTENCE_END)


def _line_refs(line: str, hits: list[SearchHit]) -> tuple[list[str], list[str]]:
    """한 줄의 인용 → (유효한 chunk_id 목록, 잘못된 인용 표기 목록). 번호는 hits 순서(1부터)로 chunk_id 에 대응한다."""
    valid = {h.chunk_id for h in hits}
    good: list[str] = []; bad: list[str] = []
    for cid in _ID_RE.findall(line):
        (good if cid in valid else bad).append(cid)
    for grp in _NUM_RE.findall(line):
        for tok in re.split(r"[,，]", grp):
            n = int(tok)
            if 1 <= n <= len(hits):
                good.append(hits[n - 1].chunk_id)
            else:
                bad.append(str(n))
    return good, bad


def _numbers(text: str) -> set[str]:
    return {m.group().replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}


def validate_grounding(answer_text:str,hits:list[SearchHit], *, check_numbers: bool = False) -> GroundingResult:
    """인용 *형식* 검증: 인용이 있고, 모두 이번 검색 결과를 가리키며, 모든 주장 줄에 인용이 있어야 한다.

    - 인용 표시만 적힌 줄(또는 '출처: [1]' 꼬리줄)은 바로 앞 주장 줄의 인용으로 본다. 앞선 주장이 없으면 아무것도 주장하지 않은 것이라 통과시키지 않는다.
    - 전각 괄호(［１］ 【1】)도 인용으로 읽는다.
    - 모델이 '근거 없음'이라고만 답했으면 model_refused=True (형식 위반이 아니라 모델의 거절).
    - check_numbers=True: 주장 줄의 숫자가 그 줄이 인용한 청크 본문에도 있어야 한다(기본 꺼짐). 지어낸 숫자만 잡으며,
      한글 수사('열흘')·계산·단위 환산은 알지 못해 정상 답을 거절할 수 있다.

    ⚠ 인용된 청크가 그 *내용을* 실제로 뒷받침하는지는 확인하지 않는다(의미 검증은 grounding_verifier 확장 또는 사람 검토).
    """
    by_id = {h.chunk_id: h for h in hits}
    cited: list[str] = []; bad: list[str] = []
    claim_lines = 0; covered = 0; substantive = 0
    prev_open = False                       # 직전 주장 줄에 아직 인용이 없는가
    records: list[dict] = []                # check_numbers 용: 주장 줄별 (숫자, 인용 id)
    refusal_seen = False; other_content = False
    for raw in answer_text.splitlines():
        ln = unicodedata.normalize("NFKC", raw.strip().translate(_BRACKETS))
        if not ln:
            continue
        good, wrong = _line_refs(ln, hits)
        cited += good; bad += wrong
        core = _core(ln)
        if not core.strip(" \t:：-–—.,;·|") or (good and _SOURCE_LABEL.match(core)):   # 인용 표시·장식(표 구분선 포함)·'출처:' 꼬리줄만 있는 줄
            if good and prev_open:
                covered += 1; prev_open = False
                if records: records[-1]["ids"] += good
            continue
        rest = core.replace(REFUSAL_PHRASE, "").replace(_REFUSAL_TAIL, "").strip(" \t:：-–—.,;·!?")
        if REFUSAL_PHRASE in core: refusal_seen = True
        if rest: other_content = True
        if not (good or wrong) and _is_label(ln, core):
            continue
        claim_lines += 1
        if len(re.findall(r"[가-힣A-Za-z0-9]", core)) >= 2:
            substantive += 1
        if good:
            covered += 1; prev_open = False
        else:
            prev_open = True
        records.append({"nums": _numbers(core), "ids": list(good)})
    coverage = covered / claim_lines if claim_lines else 1.0
    cited_t = tuple(dict.fromkeys(cited)); bad_t = tuple(dict.fromkeys(bad))
    refused = refusal_seen and not other_content
    unsupported: list[str] = []
    if check_numbers:
        for rec in records:
            if not rec["ids"]:
                continue
            ev = set().union(*(_numbers(by_id[i].text) for i in rec["ids"] if i in by_id)) if rec["ids"] else set()
            unsupported += sorted(n for n in rec["nums"] if n not in ev)
    unsupported_t = tuple(dict.fromkeys(unsupported))
    ok = (bool(cited_t) and not bad_t and coverage >= 1.0 and substantive >= 1 and not refused and not unsupported_t)
    return GroundingResult(ok, cited_t, bad_t, coverage, unsupported_t, refused)
