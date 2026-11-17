"""실환경 평가: 실제 임베딩(BGE-M3) + 실문서 + 검수된 질문 + (선택) Ollama 로 '근거 있는 질문은 찾고, 없는 질문은 막는가' 를 잰다.

합성 테스트 228건은 가짜 모델로 코드의 자기 일관성을 보증할 뿐이다. 이 스크립트가 만드는 수치가 성능 증거다.

사용 (팀 컴퓨터, Ollama 실행 중):
  PYTHONPATH=src python scripts/autorag/real_env_eval.py --root <문서폴더> --qa evidence/qa/qa_draft.jsonl --out evidence/real_eval \
      --retrievers vector,hybrid_rrf,bge_m3 --chunk 500:50 --e2e 20
  # 배관만 확인(성능 증거 아님, 판정은 항상 INVALID):  --embedder hash --skip-ollama

무엇을 재나
 1. 사전 점검: 파이썬/토치 장치/임베더 로드·의미 점검/Ollama 서버·모델
 2. 검색: hit@k, MRR@k 와 95% 신뢰구간(부트스트랩) — 후보(검색 방식)별
 3. gate(fallback) 보정: 근거 있는 질문과 없는 질문의 gate 값 분포 → 현재 임계값의 오차단율/정거절률, 추천 임계값, AUC
 4. (선택) 종단 실행: 실제 Ollama 로 N 문항을 답하게 해 fallback 동작·근거 검증 통과·지연시간 측정

전제: qa_draft.jsonl 에서 검수한 줄만 status=approved 로 바꿔 둔다. 답 없는 질문(should_fallback=true)을 5개 이상 넣어야 거절 성능을 믿을 수 있다.
판정 기준(--min-hit 등)은 *제안값(추정)*이다. 팀에서 정해 인자로 넘긴다.
종료 코드: 0 = PASS, 1 = FAIL, 2 = 입력/환경 오류, 3 = INVALID(해시 임베더).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_grid import EXTS, parse_chunks, parse_specs  # noqa: E402

from intellectagent.chunk_grid import build_candidates  # noqa: E402
from intellectagent.embeddings import HashEmbedder, SentenceTransformerEmbedder  # noqa: E402
from intellectagent.loaders import SafeDocumentLoader  # noqa: E402
from intellectagent.pipeline import CoreRAG  # noqa: E402
from intellectagent.qa_gen import answer_overlap, load_approved  # noqa: E402
from intellectagent.real_eval import (Criteria, bootstrap_ci, classify_outcome, first_relevant_rank, gate_point, gate_sweep,  # noqa: E402
                                      never_found, overlap_band, pareto_frontier, percentile, rank_metrics, stratify, verdict)


# ---- 1. 사전 점검 -------------------------------------------------------------------------------------------------------
def _check(name: str, ok: bool, detail: str = "", level: str = "error") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "level": level}


def preflight_python() -> dict:
    v = sys.version_info
    return _check("python", (v.major, v.minor) == (3, 11), f"{v.major}.{v.minor}.{v.micro} (프로젝트 요구: 3.11)", "warn")


def preflight_device() -> dict:
    try:
        import torch
    except ImportError:
        return _check("torch", False, "torch 미설치", "warn")
    dev = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    return _check("torch", True, f"torch {torch.__version__}, 장치 {dev}", "warn")


def preflight_embedder(embedder, expect_dim: int | None = None) -> dict:
    """임베더가 '의미' 를 아는지 본다: 관련 문장 쌍의 유사도가 무관한 쌍보다 높아야 한다."""
    a, b, c = "검색 증강 생성(RAG)은 문서를 찾아 답변의 근거로 쓰는 방법이다.", "RAG 는 외부 문서를 검색해 답변을 만든다.", "오늘 저녁 메뉴로 김치찌개를 먹었다."
    try:
        va, vb, vc = embedder.embed([a, b, c])
    except Exception as exc:  # noqa: BLE001
        return _check("embedder", False, f"임베딩 실패: {type(exc).__name__}: {exc}")
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))  # noqa: E731 - 정규화된 벡터 가정
    rel, unrel = dot(va, vb), dot(va, vc)
    ok = rel > unrel and (expect_dim is None or len(va) == expect_dim)
    return _check("embedder", ok, f"차원 {len(va)}, 관련쌍 {rel:.3f} vs 무관쌍 {unrel:.3f}" + ("" if rel > unrel else " — 의미 구분 실패(해시 임베더이거나 모델 이상)"))


def preflight_ollama(model: str, host: str = "http://127.0.0.1:11434", timeout: float = 3.0) -> dict:
    """로컬 Ollama 서버(localhost 전용)에 모델이 받아져 있는지."""
    if not host.startswith(("http://127.0.0.1", "http://localhost")):
        return _check("ollama", False, f"로컬 주소가 아님: {host} (이 점검은 localhost 만 허용)")
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=timeout) as r:  # noqa: S310 - localhost 고정
            tags = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return _check("ollama", False, f"서버 응답 없음({host}): {exc}. `ollama serve` 후 `ollama pull {model}`")
    names = [m.get("name", "") for m in tags.get("models", [])]
    ok = any(n == model or n.split(":")[0] == model.split(":")[0] and (":" not in model or n == model) for n in names)
    return _check("ollama", ok, f"설치된 모델 {len(names)}개" + ("" if ok else f", '{model}' 없음 → `ollama pull {model}`"))


# ---- 2~3. 검색·gate 평가 --------------------------------------------------------------------------------------------------
def gate_of(hits) -> float:
    return CoreRAG._gate(hits) if hits else 0.0     # 결과가 없으면 막힌 것과 같다


def evaluate_candidate(cand, answerable, unanswerable, top_k: int, threshold: float, crit: Criteria, boot: int, seed: int,
                       rows_out: list | None = None) -> dict:
    ranks, gates_a, gates_u, lat, by_case = [], [], [], [], {}
    for rec in answerable:
        t = time.perf_counter(); hits = cand.search(rec.question, top_k); lat.append((time.perf_counter() - t) * 1000)
        rk = first_relevant_rank([h.doc_id for h in hits], rec.relevant_doc_ids)
        g = gate_of(hits); ranks.append(rk); gates_a.append(g); by_case[rec.case_id] = rk
        if rows_out is not None:
            rows_out.append({"candidate": cand.name, "case_id": rec.case_id, "kind": "answerable", "rank": rk or "", "gate": round(g, 6)})
    for rec in unanswerable:
        hits = cand.search(rec.question, top_k); g = gate_of(hits); gates_u.append(g)
        if rows_out is not None:
            rows_out.append({"candidate": cand.name, "case_id": rec.case_id, "kind": "unanswerable", "rank": "", "gate": round(g, 6)})
    m = rank_metrics(ranks, top_k)
    hit_vals = [1.0 if r is not None and r <= top_k else 0.0 for r in ranks]
    mrr_vals = [1.0 / r if r is not None and r <= top_k else 0.0 for r in ranks]
    sweep = gate_sweep(gates_a, gates_u, crit.max_false_fallback)
    cur = gate_point(gates_a, gates_u, threshold)
    rec_p = sweep["recommended"]
    qt = stratify([(getattr(r, "qtype", "") or "unknown", rk) for r, rk in zip(answerable, ranks)], top_k)
    ov = stratify([(overlap_band(r.lexical_overlap), rk) for r, rk in zip(answerable, ranks)], top_k)
    return {"by_qtype": qt, "by_overlap": ov, "_ranks": by_case, "candidate": cand.name, "judge_errors": getattr(cand, "errors", None), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
            "hit": m["hit"], "hit_ci95": bootstrap_ci(hit_vals, boot, seed), "mrr": m["mrr"], "mrr_ci95": bootstrap_ci(mrr_vals, boot, seed),
            "latency_ms_p50": percentile(lat, 50), "latency_ms_p95": percentile(lat, 95),
            "gate": {"current": cur.__dict__, "auc": sweep["auc"],
                     "recommended": rec_p.__dict__ if rec_p else None,
                     "answerable_gate_p10_p50_p90": [percentile(gates_a, p) for p in (10, 50, 90)],
                     "unanswerable_gate_p10_p50_p90": [percentile(gates_u, p) for p in (10, 50, 90)] if gates_u else None},
            "_current_point": cur}


class JudgedCandidate:
    """후보 + 관련성 판정. 판정이 다 버리면 빈 결과(gate 0.0 = 거절)가 된다. 판정기 장애는 원본 결과로 대체(fail-open)되므로 last_error 를 센다."""
    def __init__(self, cand, judge):
        self.cand, self.judge, self.name, self.errors = cand, judge, cand.name + "+judge", 0
    def search(self, q, k):
        out = self.judge.filter(q, self.cand.search(q, k))[:k]
        self.errors += 1 if self.judge.last_error else 0
        return out


# ---- 4. 종단 실행 -------------------------------------------------------------------------------------------------------
class _Adapter:
    """grid 후보의 search(q, k) 를 CoreRAG 가 기대하는 search(q, k, candidate_k=) 로 맞춘다."""
    def __init__(self, cand): self.cand = cand
    def search(self, question, top_k, candidate_k=None): return self.cand.search(question, top_k)


def run_e2e(cand, records, model: str, top_k: int, threshold: float, limit: int, out_dir: Path, generator=None, citation_style: str = "id", gen_options: dict | None = None) -> dict:
    from intellectagent.generation import OllamaGenerator
    from intellectagent.grounding import validate_grounding
    base_gen = generator or OllamaGenerator(model, citation_style=citation_style, options=gen_options)

    class _Rec:  # 생성기 원문을 남겨 '왜 막혔는지'를 사후에 구분한다
        raw = ""
        def generate(self, question, hits):
            self.raw = base_gen.generate(question, hits)
            return self.raw
    rec_gen = _Rec()
    pipe = CoreRAG(_Adapter(cand), rec_gen, top_k, threshold)
    rows, errors, lat = [], 0, []
    for rec in records[:limit]:
        t = time.perf_counter()
        rec_gen.raw = ""
        try:
            ans = pipe.ask(rec.question)
        except Exception as exc:  # noqa: BLE001 - 서버 끊김 등은 기록하고 3번 연속이면 중단
            errors += 1
            rows.append({"case_id": rec.case_id, "expected_fallback": rec.should_fallback, "error": f"{type(exc).__name__}: {exc}"})
            if errors >= 3 and all("error" in r for r in rows[-3:]):
                break
            continue
        dt = (time.perf_counter() - t) * 1000; lat.append(dt)
        rel = set(rec.relevant_doc_ids)
        cited_chunks = set(ans.citations)
        outcome = classify_outcome(
            expected_fallback=rec.should_fallback, fallback=ans.fallback,
            gold_in_hits=any(h.doc_id in rel for h in ans.hits),
            gold_cited=any(h.doc_id in rel for h in ans.hits if h.chunk_id in cited_chunks),
            answer_overlap=answer_overlap(ans.text, rec.answer) if rec.answer and not ans.fallback else None)
        reason, gdetail = "", None
        if ans.fallback:
            if not rec_gen.raw:
                reason = "score_gate"  # 생성기까지 가지 못함(검색 결과 없음 또는 점수 < 임계값)
            else:
                g = validate_grounding(rec_gen.raw, list(ans.hits))
                reason = "model_refused" if g.model_refused else "grounding_failed"   # 모델이 스스로 '근거 없음'이라 답한 경우를 형식 위반과 구분
                gdetail = {"cited_ok": len(g.cited_ids), "bad_citations": len(g.unsupported_citations), "coverage": round(g.citation_coverage, 3)}
        rows.append({"case_id": rec.case_id, "qtype": getattr(rec, "qtype", "") or "unknown", "expected_fallback": rec.should_fallback,
                     "fallback": ans.fallback, "outcome": outcome, "n_citations": len(ans.citations), "latency_ms": round(dt, 1), "answer": ans.text[:600],
                     "refusal_reason": reason, "grounding": gdetail, "raw_answer": rec_gen.raw[:800], "citation_style": citation_style})
    done = [r for r in rows if "fallback" in r]
    a = [r for r in done if not r["expected_fallback"]]; u = [r for r in done if r["expected_fallback"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2e_samples.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    outcomes: dict[str, int] = {}
    by_type: dict[str, dict[str, int]] = {}
    for r in done:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        d = by_type.setdefault(r["qtype"], {}); d[r["outcome"]] = d.get(r["outcome"], 0) + 1
    reasons: dict[str, int] = {}
    for r in done:
        if r.get("refusal_reason"):
            reasons[r["refusal_reason"]] = reasons.get(r["refusal_reason"], 0) + 1
    return {"asked": len(rows), "answered_or_fallback": len(done), "errors": errors, "outcomes": outcomes, "outcomes_by_qtype": by_type,
            "refusal_reasons": reasons, "citation_style": citation_style, "generation_options": gen_options,
            "answerable_answered_rate": (sum(1 for r in a if not r["fallback"]) / len(a)) if a else None,
            "answerable_with_citation_rate": (sum(1 for r in a if not r["fallback"] and r["n_citations"] > 0) / len(a)) if a else None,
            "unanswerable_refused_rate": (sum(1 for r in u if r["fallback"]) / len(u)) if u else None,
            "latency_ms_p50": percentile(lat, 50), "latency_ms_p95": percentile(lat, 95)}


# ---- 보고서 -------------------------------------------------------------------------------------------------------------
def fmt_ci(v, ci) -> str:
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def render(info: dict) -> str:
    L = [f"# 실환경 평가 보고서", "", f"- 종합 판정: **{info['verdict']['overall']}**  (대상 후보: `{info['best']}`)",
         f"- 문서 {info['documents']}개 / 답 있는 질문 {info['n_answerable']} / 답 없는 질문 {info['n_unanswerable']} / top_k={info['top_k']} / 임베더={info['embedder']} / 현재 fallback_min_score={info['threshold']}", ""]
    L += [f"> ⚠ {w}" for w in info["verdict"]["warnings"] + info["warnings"]] + [""]
    L += ["## 사전 점검", "", "| 항목 | 결과 | 내용 |", "|---|---|---|"]
    L += [f"| {c['name']} | {'OK' if c['ok'] else ('경고' if c['level'] == 'warn' else '실패')} | {c['detail']} |" for c in info["preflight"]]
    L += ["", "## 검색 성능 (95% 신뢰구간은 질문을 다시 뽑는 부트스트랩)", "",
          "| 후보 | hit@k [95%] | MRR [95%] | 현재 임계값 오차단율 | 현재 정거절률 | 추천 임계값 | AUC | p50 ms |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in info["candidates"]:
        g = r["gate"]; rc = g["recommended"]
        L.append(f"| {r['candidate']}{' ★' if r['candidate'] == info['best'] else ''} | {fmt_ci(r['hit'], r['hit_ci95'])} | {fmt_ci(r['mrr'], r['mrr_ci95'])} | "
                 f"{g['current']['false_fallback_rate']:.2f} | {g['current']['correct_refusal_rate']:.2f} | "
                 f"{('%.3f' % rc['threshold']) if rc else '없음'} | {('%.2f' % g['auc']) if g['auc'] is not None else '-'} | {r['latency_ms_p50']:.0f} |")
    L += ["", "- 추천 임계값 = 오차단율이 허용치 이하인 점 중 정거절률이 가장 높은 값. **'없음'이면 점수만으로는 근거 유무를 가를 수 없다**(질문·문서·검색 방식을 고쳐야 한다).",
          "- 검색 방식마다 gate 값의 척도가 다르다(vector/bge_m3=코사인, bm25=BM25 점수). `fallback_min_score` 는 선택한 방식의 추천값으로 따로 정한다."]
    b = next(r for r in info["candidates"] if r["candidate"] == info["best"])
    if info.get("pareto"):
        L += ["", "- 정확도(MRR)·지연시간(p50) Pareto 후보: " + ", ".join(f"`{n}`" for n in info["pareto"]) + " (둘 다 더 나은 후보가 없는 것들. 느려도 정확한 쪽과 빠른 쪽을 함께 보여 준다)"]
    for title, key in (("질문 유형별 (대상 후보)", "by_qtype"), ("질문-청크 어휘 겹침 층별 (대상 후보)", "by_overlap")):
        if b.get(key):
            L += ["", f"## {title}", "", "| 층 | 질문 수 | hit@k | MRR |", "|---|---:|---:|---:|"]
            L += [f"| {k} | {int(m['n'])} | {m['hit']:.2f} | {m['mrr']:.2f} |" for k, m in b[key].items()]
    L += ["", "- 유형은 규칙 기반 추정이다. 층에 질문이 몇 개 안 되면(n<5) 수치를 믿지 말 것. 겹침 high 층만 점수가 높다면 질문이 청크 낱말을 베낀 탓일 수 있다(어휘 검색에 유리)."]
    if info.get("never_found"):
        nf = info["never_found"]
        L += ["", f"## 모든 후보가 못 찾은 질문 {len(nf)}개 — 먼저 사람이 점검", "", "라벨(정답 문서) 오류, 문서 파싱 누락, 지나치게 모호한 질문일 가능성이 높다: " + ", ".join(f"`{i}`" for i in nf[:30]) + (" …" if len(nf) > 30 else "")]
    ck = info["verdict"]["checks"]
    L += ["", "## 판정 기준 (대상 후보, 현재 임계값 기준)", "", "| 항목 | 값 | 기준 | 통과 |", "|---|---:|---:|---|"]
    for k, c in ck.items():
        bound = f"≥ {c['min']}" if "min" in c else f"≤ {c['max']}"
        L.append(f"| {k} | {c['value']:.3f} | {bound} | {'O' if c['pass'] else 'X'} |")
    if info.get("e2e"):
        e = info["e2e"]
        f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        L += ["", f"## 종단 실행 (Ollama `{info['ollama_model']}`, {e['asked']}문항, 오류 {e['errors']})", "",
              f"- 근거 있는 질문을 답한 비율 {f(e['answerable_answered_rate'])}, 그중 출처 포함 {f(e['answerable_with_citation_rate'])}",
              f"- 근거 없는 질문을 거절한 비율 {f(e['unanswerable_refused_rate'])}",
              f"- 지연시간 p50 {e['latency_ms_p50']:.0f} ms / p95 {e['latency_ms_p95']:.0f} ms",
              "- 결과 분류: " + ", ".join(f"{k} {v}" for k, v in sorted(e.get("outcomes", {}).items())) + "  (hallucinated_answer = 답 없는 질문에 답함, refused = 정답 문서를 가져왔는데도 막음, wrong_source/answer_mismatch 는 어휘 기반 추정이라 사람이 확인)",
              "- 거절 원인: " + (", ".join(f"{k} {v}" for k, v in sorted(e.get("refusal_reasons", {}).items())) or "없음") + f" / 인용 형식 `{e.get('citation_style', 'id')}`  (score_gate = 점수가 임계값 미만, grounding_failed = 답은 만들었으나 인용 형식 검증 탈락)",
              "- 답변 내용의 정확성은 이 수치로 알 수 없다. `e2e_samples.jsonl` 을 사람이 직접 읽어 확인한다."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--qa", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--embedder", default="BAAI/bge-m3", help="모델 이름 또는 'hash'(배관 확인, 판정 INVALID)")
    ap.add_argument("--chunk", "--chunks", dest="chunk", default="500:50", help="size:overlap[:ctx] (쉼표로 여러 개)")
    ap.add_argument("--retrievers", default="vector,hybrid_rrf,bge_m3")
    ap.add_argument("--top-k", type=int, default=5); ap.add_argument("--fallback-min-score", type=float, default=0.12)
    ap.add_argument("--e2e", type=int, default=0, help="Ollama 로 답하게 할 질문 수(0=건너뜀)")
    ap.add_argument("--temperature", type=float, default=0.0, help="종단 평가의 생성 온도(기본 0 = 재현용). 음수면 Ollama 기본값을 쓴다")
    ap.add_argument("--gen-seed", type=int, default=42, help="종단 평가의 생성 시드(부트스트랩용 --seed 와 별개)")
    ap.add_argument("--citation-style", choices=["id", "index"], default="id", help="종단 평가의 인용 형식(id=청크 ID, index=[1]번호). 같은 질문셋으로 id/index 를 비교하라")
    ap.add_argument("--ollama-model", default="qwen2.5:14b"); ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument("--judge", action="store_true", help="청크별 관련성 판정(Ollama)을 각 후보에 얹어 함께 평가한다(후보 이름에 +judge). 질문 수 x top_k 번 LLM 호출")
    ap.add_argument("--judge-std", type=float, default=1.0, help="적응형 임계값 평균 − n·표준편차 의 n")
    ap.add_argument("--bootstrap", type=int, default=2000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-hit", type=float, default=0.80); ap.add_argument("--min-mrr", type=float, default=0.60)
    ap.add_argument("--max-false-fallback", type=float, default=0.10); ap.add_argument("--min-correct-refusal", type=float, default=0.80)
    ap.add_argument("--max-file-mb", type=int, default=60)
    a = ap.parse_args(argv)
    crit = Criteria(a.min_hit, a.min_mrr, a.max_false_fallback, a.min_correct_refusal)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hash_mode = a.embedder == "hash"
    if a.judge and a.skip_ollama:
        print("--judge 는 Ollama 가 필요합니다(--skip-ollama 와 함께 쓸 수 없음).", file=sys.stderr)
        return 2

    pre = [preflight_python(), preflight_device()]
    try:
        embedder = HashEmbedder() if hash_mode else SentenceTransformerEmbedder(a.embedder)
    except Exception as exc:  # noqa: BLE001
        pre.append(_check("embedder", False, f"모델 로드 실패: {type(exc).__name__}: {exc}"))
        embedder = None
    if embedder is not None:
        pre.append(preflight_embedder(embedder, None if hash_mode else 1024))
    if not a.skip_ollama and (a.e2e or a.judge):
        pre.append(preflight_ollama(a.ollama_model, a.ollama_host))
    if any(not c["ok"] and c["level"] == "error" for c in pre):
        (out / "real_eval.json").write_text(json.dumps({"preflight": pre, "verdict": {"overall": "ERROR"}}, ensure_ascii=False, indent=2), encoding="utf-8")
        for c in pre:
            print(("OK  " if c["ok"] else "FAIL") + f" {c['name']}: {c['detail']}", file=sys.stderr)
        print("사전 점검 실패로 중단합니다.", file=sys.stderr)
        return 2

    root = Path(a.root).resolve()
    loader = SafeDocumentLoader(root, EXTS, a.max_file_mb * 1024 * 1024)
    docs, failed = [], []
    for p in loader.discover():
        try:
            docs.append(loader.load(p))
        except Exception as exc:  # noqa: BLE001
            failed.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    if not docs:
        print("문서를 하나도 읽지 못했습니다. scripts/autorag/check_real_docs.py 로 먼저 점검하세요.", failed, file=sys.stderr)
        return 2
    approved = load_approved(Path(a.qa))
    answerable = [r for r in approved if not r.should_fallback]
    unanswerable = [r for r in approved if r.should_fallback]
    if not answerable:
        print("approved 상태의 답 있는 질문이 없습니다.", file=sys.stderr)
        return 2
    orphan = sorted({i for r in answerable for i in r.relevant_doc_ids} - {d.doc_id for d in docs})
    if orphan:
        print(f"질문의 정답 문서 ID 가 --root 문서와 맞지 않습니다({len(orphan)}개): {orphan[:5]}", file=sys.stderr)
        return 2

    specs = parse_specs(a.retrievers, 0.5)
    m3_encoder = None
    if any(s.kind == "bge_m3" for s in specs):
        from intellectagent.bge_m3 import FlagM3Encoder, HashM3Encoder
        try:
            m3_encoder = HashM3Encoder() if hash_mode else FlagM3Encoder(a.embedder)
        except Exception as exc:  # noqa: BLE001
            print(f"bge_m3 인코더 로드 실패(FlagEmbedding 필요): {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    cands = build_candidates(docs, parse_chunks(a.chunk), specs, embedder=embedder, m3_encoder=m3_encoder, gate=True)
    if a.judge:
        from intellectagent.relevance_judge import OllamaJudge, RelevanceJudge
        cands = cands + [JudgedCandidate(c, RelevanceJudge(OllamaJudge(a.ollama_model), n_std=a.judge_std, max_candidates=a.top_k)) for c in cands]

    rows_out: list[dict] = []
    results = [evaluate_candidate(c, answerable, unanswerable, a.top_k, a.fallback_min_score, crit, a.bootstrap, a.seed, rows_out) for c in cands]
    best = max(results, key=lambda r: (r["mrr"], r["hit"]))
    v = verdict({"hit": best["hit"], "mrr": best["mrr"]}, best["_current_point"], len(answerable), len(unanswerable), crit, valid_embedder=not hash_mode)
    for r in results:
        r.pop("_current_point")
    ranks_by_cand = {r["candidate"]: r.pop("_ranks") for r in results}
    never = never_found(ranks_by_cand, a.top_k)
    pareto = sorted(pareto_frontier([(r["candidate"], r["mrr"], r["latency_ms_p50"]) for r in results]))

    warnings = []
    bad_judge = [r["candidate"] for r in results if r.get("judge_errors")]
    if bad_judge:
        warnings.append("관련성 판정기 오류로 원본 결과를 쓴 질문이 있다(fail-open): " + ", ".join(bad_judge) + " — 해당 +judge 수치는 판정 효과를 반영하지 않는다.")
    if failed:
        warnings.append(f"읽지 못한 문서 {len(failed)}개: " + ", ".join(Path(f["path"]).name for f in failed))
    info = {"verdict": v, "best": best["candidate"], "documents": len(docs), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
            "top_k": a.top_k, "embedder": a.embedder, "threshold": a.fallback_min_score, "criteria": crit.__dict__, "preflight": pre,
            "candidates": results, "never_found": never, "pareto": pareto, "warnings": warnings, "failed_docs": failed, "ollama_model": a.ollama_model, "args": vars(a)}
    if a.e2e and not a.skip_ollama:
        best_cand = next(c for c in cands if c.name == best["candidate"])
        mixed = sorted(approved, key=lambda r: (r.should_fallback, r.case_id))
        half = max(a.e2e // 2, 1)
        pick = ([r for r in mixed if not r.should_fallback][: a.e2e - min(half, len(unanswerable))] + [r for r in mixed if r.should_fallback][:half])[: a.e2e]
        info["e2e"] = run_e2e(best_cand, pick, a.ollama_model, a.top_k, a.fallback_min_score, a.e2e, out, citation_style=a.citation_style,
                                gen_options=({"temperature": a.temperature, "seed": a.gen_seed} if a.temperature >= 0 else None))
    (out / "real_eval.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    with (out / "per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "case_id", "kind", "rank", "gate"]); w.writeheader(); w.writerows(rows_out)
    md = render(info)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"저장: {out}/report.md, real_eval.json, per_question.csv")
    return {"PASS": 0, "FAIL": 1, "INVALID": 3}[v["overall"]]


if __name__ == "__main__":
    sys.exit(main())
