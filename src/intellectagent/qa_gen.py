"""평가용 질문·정답 자동 생성 (AutoRAG ``data/qa`` 의 흐름: 샘플 -> 질문 -> 정답 -> 필터 -> 근거 추출).

목적: 손으로 만든 평가셋이 너무 적을 때 *초안* 을 빠르게 만들고, 사람이 검수해서 확정한다.
이 모듈이 만든 질문은 검수 전에는 "초안(draft)" 이며 점수 주장의 근거로 쓰면 안 된다.

설계
- LLM 은 ``Callable[[str], str]`` 로 주입한다(로컬 Ollama 어댑터 ``ollama_llm`` 제공). 외부 API 를 쓰지 않는다.
- AutoRAG 필터 대응: ``dontknow``(모른다는 답 제거) / ``passage_dependency``(문맥 없이는 뜻이 없는 질문 제거).
  원본은 LLM 판정이고, 여기서는 *규칙 기반* 이 기본이며 ``dependency_llm`` 을 주면 LLM 판정을 추가한다.
- 근거 추출은 LLM 대신 결정적 규칙(정답과 가장 겹치는 문장)으로 한다. 결과가 항상 청크의 부분 문자열이라 검증할 수 있다.
- 편향 경고: 청크를 보고 만든 질문은 청크의 낱말을 그대로 쓰기 쉬워 어휘 검색(BM25)에 유리하다.
  각 질문에 ``lexical_overlap``(질문 bigram 중 청크에 있는 비율)을 기록하니, 보고할 때 이 값으로 층을 나눠 보라.
- ``relevant_doc_ids`` 는 문서 단위, ``gt_contents`` 는 청킹과 무관한 근거 텍스트 -> 청킹 설정이 달라도 같은 질문셋으로 비교된다.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .hashing import sha256_text
from .lexical import BM25Index, tok_char_bigram
from .models import Chunk, EvalCase

LLM = Callable[[str], str]

# ---- 프롬프트 -------------------------------------------------------------------------------------------------------
QUERY_PROMPT = """당신은 문서 검색 시스템의 평가용 질문을 만드는 도우미입니다.
아래 TEXT를 읽고, TEXT만 있으면 답할 수 있는 사실 질문 1개를 한국어로 작성하세요.

규칙:
1. 질문 한 줄만 출력하세요. 설명, 번호, 따옴표를 붙이지 마세요.
2. 질문하는 사람은 이 TEXT를 모릅니다. "위 글", "이 문서", "주어진 텍스트" 같은 표현을 쓰지 마세요.
3. 질문만 읽어도 무엇에 대한 것인지 알 수 있게, 구체적인 대상(제도·시스템·항목의 이름)을 포함하세요.
4. TEXT의 문장을 그대로 베끼지 말고 다른 표현으로 바꿔 쓰세요.
5. 파일 이름이나 제목 자체를 묻지 마세요.

TEXT:
{text}

질문:"""

ANSWER_PROMPT = """아래 TEXT만 근거로 질문에 한두 문장으로 답하세요.
TEXT에 답이 없으면 정확히 '알 수 없음'이라고만 답하세요.

TEXT:
{text}

질문: {question}
답:"""

DEPENDENCY_PROMPT = """다음 질문이 '단락 의존' 질문인지 판단하세요.
단락 의존이란 어떤 문서·표·단락을 보고 있는지 모르면 의미가 없거나 답이 달라지는 질문입니다.
예) "주어진 표에서 가장 높은 점수는?" -> True / "프랑스의 수도는?" -> False
True 또는 False 한 단어만 출력하세요.

질문: {question}
답:"""

UNANSWERABLE_PROMPT = """사내 문서 검색 시스템이 있습니다. 문서 이름 목록은 다음과 같습니다.
{names}

이 문서들에 *답이 없을 것 같지만* 사용자가 물어볼 법한 질문을 {n}개 만드세요.
(예: 다루지 않는 주제, 문서에 없는 구체적 수치·인물·제품)
한 줄에 질문 하나씩, 번호 없이 출력하세요.
"""

# ---- 규칙 ----------------------------------------------------------------------------------------------------------
_DEICTIC = re.compile(r"(이|그|해당|위|아래|다음|본|주어진|제공된|상기)\s*(문서|글|문단|단락|본문|텍스트|지문|표|그림|도표|자료|파일|페이지|내용)")
_ACCORDING = re.compile(r"(문서|본문|텍스트|지문|글|표|그림)\s*(에\s*따르면|에\s*의하면|에서\s*(언급|설명|말한|제시))")
# 앞 두 묶음은 AutoRAG dontknow_filter_rule_based 의 한국어 구문, 뒤쪽은 추가한 것(출처 구분)
DONT_KNOW_AUTORAG = ["몰라요", "모르겠습니다", "모르겠어요", "몰라", "내가 어떻게 알아?", "모르겠소", "몰라유", "모르것는디",
                     "모르겠어유", "모르겠네유", "모르겠네요"]
DONT_KNOW_EXTRA = ["알 수 없", "알수 없", "확인할 수 없", "찾을 수 없", "언급되어 있지 않", "정보가 없", "제공되지 않"]
_THINK = re.compile(r"<think>.*?</think>", re.S)
_SENT = re.compile(r"(?<=[.!?다요])\s+|\n+")


def is_passage_dependent(question: str) -> bool:
    return bool(_DEICTIC.search(question) or _ACCORDING.search(question))


def is_dont_know(answer: str) -> bool:
    return any(p in answer for p in DONT_KNOW_AUTORAG + DONT_KNOW_EXTRA)


# ---- 질문 유형 (Agentic-AutoRAG 의 유형별 시험 구성 아이디어; 여기서는 *규칙 기반 추정* 이다) ----------------------------------------
_QTYPE_RULES = (
    ("comparison", re.compile(r"차이|비교|보다|각각|둘\s*중|어느\s*쪽|어느\s*것")),
    ("numeric", re.compile(r"몇|얼마|며칠|금액|비용|비율|퍼센트|%|횟수|한도")),
    ("procedure", re.compile(r"어떻게|절차|방법|순서|단계|하려면|신청")),
    ("definition", re.compile(r"무엇|뭔가|뭐|뜻|의미|정의|이란|란\s*\?")),
)
QTYPES = tuple(k for k, _ in _QTYPE_RULES) + ("factoid", "unanswerable")


def classify_question(question: str) -> str:
    """질문 유형 추정: 먼저 맞는 규칙이 이긴다(비교 > 수치 > 절차 > 정의). 어디에도 안 맞으면 factoid. 정확한 분류가 아니라 층별 보고용 꼬리표다."""
    for name, rx in _QTYPE_RULES:
        if rx.search(question):
            return name
    return "factoid"


def answer_overlap(a: str, b: str) -> float:
    """두 답의 문자 bigram dice 계수(0~1). 인용 표기 [..] 와 공백은 무시."""
    clean = lambda t: re.sub(r"\s+", " ", re.sub(r"\[[^\]]{1,40}\]", " ", t or "")).strip()  # noqa: E731
    return _dice(clean(a), clean(b))


def answers_agree(a: str, b: str, threshold: float = 0.5) -> bool:
    """한쪽이 다른 쪽을 포함하거나(공백 무시, 2자 이상) bigram dice ≥ threshold 면 같은 답으로 본다. 어휘 기반 근사다."""
    na, nb = re.sub(r"\s+", "", a or ""), re.sub(r"\s+", "", b or "")
    if len(na) >= 2 and len(nb) >= 2 and (na in nb or nb in na):
        return True
    return answer_overlap(a, b) >= threshold


def lexical_overlap(question: str, text: str) -> float:
    q = set(tok_char_bigram(question))
    return len(q & set(tok_char_bigram(text))) / len(q) if q else 0.0


def _dice(a: str, b: str) -> float:
    ca, cb = Counter(tok_char_bigram(a)), Counter(tok_char_bigram(b))
    n = sum(ca.values()) + sum(cb.values())
    return 2 * sum((ca & cb).values()) / n if n else 0.0


def extract_evidence(chunk_text: str, answer: str, question: str = "") -> str:
    """정답과 가장 많이 겹치는 문장(동점이면 질문과 더 겹치는 문장). 반환값은 항상 chunk_text 의 부분 문자열이거나 빈 문자열."""
    best, best_score = "", 0.0
    for sent in (s.strip() for s in _SENT.split(chunk_text)):
        if len(sent) < 8:
            continue
        score = _dice(sent, answer) + 0.25 * _dice(sent, question)
        if score > best_score:
            best, best_score = sent, score
    return best if best_score > 0 and best in chunk_text else ""


def parse_question(raw: str) -> str | None:
    raw = _THINK.sub("", raw or "").strip()
    cands = [re.sub(r"^\s*(?:[-*•]|\d+[.)]|질문\s*[:：])\s*", "", ln).strip().strip("\"'“”‘’`") for ln in raw.splitlines() if ln.strip()]
    cands = [c for c in cands if c]
    if not cands:
        return None
    q = next((c for c in cands if c.endswith("?")), cands[0])
    if not (6 <= len(q) <= 200) or len(q.split()) < 2:
        return None
    return q


def parse_bool(raw: str) -> bool | None:
    t = _THINK.sub("", raw or "").strip().lower()
    if t.startswith("true") or t.startswith("참"):
        return True
    if t.startswith("false") or t.startswith("거짓"):
        return False
    return None


# ---- 데이터 --------------------------------------------------------------------------------------------------------
@dataclass
class QARecord:
    """EvalCase 와 같은 필드(case_id, question, relevant_doc_ids, should_fallback)에 근거 정보를 더한 레코드."""
    case_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    should_fallback: bool = False
    answer: str = ""
    evidence: str = ""
    gt_contents: tuple[str, ...] = ()
    gold_chunk_id: str = ""
    lexical_overlap: float = 0.0
    kind: str = "factoid"
    status: str = "draft"  # draft | approved | rejected
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    qtype: str = ""  # classify_question() 결과(답 없는 질문은 "unanswerable"). 옛 파일에는 없어서 기본값을 둔다.

    def to_eval_case(self) -> EvalCase:
        return EvalCase(self.case_id, self.question, tuple(self.relevant_doc_ids), self.should_fallback)

    def to_json(self) -> str:
        d = asdict(self)
        d["relevant_doc_ids"] = list(self.relevant_doc_ids)
        d["gt_contents"] = list(self.gt_contents)
        return json.dumps(d, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "QARecord":
        d = json.loads(line)
        d["relevant_doc_ids"] = tuple(d.get("relevant_doc_ids", ()))
        d["gt_contents"] = tuple(d.get("gt_contents", ()))
        return cls(**d)


@dataclass
class GenerationResult:
    drafts: list[QARecord]
    rejected: list[QARecord]
    stats: dict


def write_jsonl(records: Sequence[QARecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[QARecord]:
    return [QARecord.from_json(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_approved(path: Path) -> list[QARecord]:
    """검수에서 status 를 'approved' 로 바꾼 것만. 초안 그대로는 평가에 쓰지 않는다."""
    return [r for r in read_jsonl(path) if r.status == "approved"]


# ---- 생성 ----------------------------------------------------------------------------------------------------------
def sample_chunks(chunks: Sequence[Chunk], n: int, seed: int = 42, min_chars: int = 80) -> list[Chunk]:
    """문서마다 고르게(라운드 로빈) 결정적으로 뽑는다. AutoRAG ``random_single_hop`` 은 문서 편중이 생길 수 있어 분산시켰다."""
    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for c in sorted(chunks, key=lambda c: (c.doc_id, c.ordinal, c.chunk_id)):
        if len(c.text.strip()) >= min_chars:
            by_doc.setdefault(c.doc_id, []).append(c)
    for v in by_doc.values():
        rng.shuffle(v)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    out: list[Chunk] = []
    while len(out) < n and any(by_doc[d] for d in docs):
        for d in docs:
            if by_doc[d] and len(out) < n:
                out.append(by_doc[d].pop())
    return out


def generate_qa(llm: LLM, chunks: Sequence[Chunk], n: int, *, seed: int = 42, min_chars: int = 80, attempts_factor: int = 3,
                dependency_llm: LLM | None = None, flag_overlap: float = 0.8) -> GenerationResult:
    candidates = sample_chunks(chunks, n * max(1, attempts_factor), seed, min_chars)
    drafts: list[QARecord] = []
    rejected: list[QARecord] = []
    seen: set[str] = set()
    stats: Counter = Counter(candidates=len(candidates))

    def reject(chunk: Chunk, question: str, reason: str, answer: str = "") -> None:
        stats[f"rejected_{reason}"] += 1
        rejected.append(QARecord(f"rej-{sha256_text(chunk.chunk_id + question)[:10]}", question, (chunk.doc_id,), answer=answer,
                                 gold_chunk_id=chunk.chunk_id, status="rejected", reason=reason))

    for chunk in candidates:
        if len(drafts) >= n:
            break
        try:
            q = parse_question(llm(QUERY_PROMPT.format(text=chunk.text)))
        except Exception:
            stats["llm_error"] += 1
            continue
        if q is None:
            reject(chunk, "", "bad_question")
            continue
        if is_passage_dependent(q):
            reject(chunk, q, "passage_dependent")
            continue
        if dependency_llm is not None:
            try:
                if parse_bool(dependency_llm(DEPENDENCY_PROMPT.format(question=q))) is True:
                    reject(chunk, q, "passage_dependent_llm")
                    continue
            except Exception:
                stats["dependency_llm_error"] += 1
        key = re.sub(r"\s+", "", q)
        if key in seen:
            reject(chunk, q, "duplicate")
            continue
        try:
            ans = _THINK.sub("", llm(ANSWER_PROMPT.format(text=chunk.text, question=q))).strip()
        except Exception:
            stats["llm_error"] += 1
            continue
        if not ans or is_dont_know(ans):
            reject(chunk, q, "dont_know", ans)
            continue
        seen.add(key)
        ev = extract_evidence(chunk.text, ans, q)
        flags: list[str] = []
        if not ev:
            flags.append("no_evidence_sentence")  # 근거 문장을 못 찾음 -> 청크 전체를 근거로 쓰되 검수 필요
        ov = lexical_overlap(q, chunk.text)
        if ov >= flag_overlap:
            flags.append("high_lexical_overlap")
        drafts.append(QARecord(f"qa-{sha256_text(chunk.chunk_id + q)[:10]}", q, (chunk.doc_id,), False, ans, ev,
                               (ev,) if ev else (chunk.text,), chunk.chunk_id, round(ov, 4), "factoid", "draft", "", flags, classify_question(q)))
    stats["drafts"] = len(drafts)
    stats["rejected"] = len(rejected)
    return GenerationResult(drafts, rejected, dict(stats))


def oracle_filter(llm: LLM, drafts: Sequence[QARecord], *, threshold: float = 0.5) -> tuple[list[QARecord], list[QARecord], dict]:
    """정답 복원 검사(oracle answerability; Agentic-AutoRAG 의 '정답 구간만 주고 답이 복원되는가' 게이트의 재구현).

    질문과 *근거 텍스트(gt_contents)만* 주고 LLM 에게 다시 답하게 한다. '알 수 없음'이거나 원래 답과 어긋나면 그 질문은
    근거가 정답을 뒷받침하지 못하는 것이므로 거른다 — 이런 질문은 검색이 완벽해도 맞출 수 없어 점수를 깎는다.
    답 없는 질문(should_fallback)은 대상이 아니다. LLM 오류가 나면 거르지 않고 ``oracle_error`` 플래그만 단다(검수 때 보라).
    한계: 같은 모델이 만든 답과 비교하므로 모델의 같은 오해는 못 잡는다. 판정은 어휘 기반(``answers_agree``)이라 의역된 정답은 잘못 거를 수 있다.
    """
    kept: list[QARecord] = []; rejected: list[QARecord] = []; stats: Counter = Counter()
    for r in drafts:
        if r.should_fallback or not r.answer or not r.gt_contents:
            kept.append(r); stats["skipped"] += 1
            continue
        try:
            again = _THINK.sub("", llm(ANSWER_PROMPT.format(text="\n".join(r.gt_contents), question=r.question))).strip()
        except Exception:
            r.flags.append("oracle_error"); kept.append(r); stats["oracle_error"] += 1
            continue
        if not again or is_dont_know(again):
            r.status, r.reason = "rejected", "oracle_unanswerable"
            rejected.append(r); stats["oracle_unanswerable"] += 1
        elif not answers_agree(again, r.answer, threshold):
            r.status, r.reason = "rejected", "oracle_mismatch"
            rejected.append(r); stats["oracle_mismatch"] += 1
        else:
            kept.append(r); stats["passed"] += 1
    return kept, rejected, dict(stats)


def generate_unanswerable(llm: LLM, doc_names: Sequence[str], n: int, bm25: BM25Index, *, max_coverage: float = 0.6,
                          max_names: int = 30) -> GenerationResult:
    """말뭉치에 답이 없을 법한 질문 초안. ``coverage`` = 질문 토큰 중 말뭉치 어휘에 있는 비율(휴리스틱) — max_coverage 이상이면 버린다.
    이 필터는 "답이 없다"를 보장하지 못한다. 반드시 사람이 검수한다(플래그 unanswerable_unverified)."""
    names = "\n".join(f"- {x}" for x in list(doc_names)[:max_names])
    raw = _THINK.sub("", llm(UNANSWERABLE_PROMPT.format(names=names, n=n)))
    drafts: list[QARecord] = []
    rejected: list[QARecord] = []
    vocab = set(bm25.idf)
    for ln in raw.splitlines():
        q = parse_question(ln)
        if q is None or not q.endswith("?"):  # 안내 문구("다음은 질문입니다")·잡담을 질문으로 오인하지 않게 물음표로 끝나는 줄만 인정
            continue
        toks = bm25.tokenizer(q)
        cov = sum(t in vocab for t in toks) / len(toks) if toks else 1.0
        rec = QARecord(f"un-{sha256_text(q)[:10]}", q, (), True, kind="unanswerable", lexical_overlap=round(cov, 4),
                       flags=["unanswerable_unverified"], qtype="unanswerable")
        if cov >= max_coverage:
            rec.status, rec.reason = "rejected", "corpus_coverage_high"
            rejected.append(rec)
        elif len(drafts) < n:
            drafts.append(rec)
    return GenerationResult(drafts, rejected, {"drafts": len(drafts), "rejected": len(rejected)})


# ---- 로컬 LLM 어댑터 -------------------------------------------------------------------------------------------------
def ollama_llm(model: str, client=None, temperature: float = 0.0, seed: int = 42) -> LLM:
    """Ollama 채팅 호출을 ``Callable[[str], str]`` 로 감싼다. temperature 0 + seed 고정으로 재현성을 높인다(완전한 보장은 아님)."""
    def call(prompt: str) -> str:
        c = client
        if c is None:
            import ollama as c  # type: ignore
        r = c.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": temperature, "seed": seed})
        if isinstance(r, dict):
            return str((r.get("message") or {}).get("content", ""))
        msg = getattr(r, "message", None)
        return str(getattr(msg, "content", "") if msg is not None else "")
    return call
