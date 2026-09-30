from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import Answer


@dataclass(frozen=True)
class Week6EvalCase:
    case_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    reference_answer: str
    should_fallback: bool = False
    question_type: str = "fact"


@dataclass(frozen=True)
class Week6RunConfig:
    run_id: str
    seed: int
    system_name: str
    top_k: int
    dataset_version: str
    notes: str = ""


def load_week6_cases(path: Path) -> list[Week6EvalCase]:
    cases: list[Week6EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cases.append(Week6EvalCase(
            case_id=str(d["case_id"]),
            question=str(d["question"]),
            relevant_doc_ids=tuple(d.get("relevant_doc_ids", [])),
            reference_answer=str(d.get("reference_answer", "")),
            should_fallback=bool(d.get("should_fallback", False)),
            question_type=str(d.get("question_type", "fact")),
        ))
    return cases


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[max(0, int(len(ys) * 0.95) - 1)]


def _rr(relevant_doc_ids: set[str], hit_doc_ids: list[str]) -> float:
    for idx, doc_id in enumerate(hit_doc_ids, 1):
        if doc_id in relevant_doc_ids:
            return 1.0 / idx
    return 0.0


def _retrieval_precision(relevant_doc_ids: set[str], hit_doc_ids: list[str]) -> float:
    if not hit_doc_ids:
        return 1.0 if not relevant_doc_ids else 0.0
    return sum(1 for x in hit_doc_ids if x in relevant_doc_ids) / len(hit_doc_ids)


def _retrieval_recall(relevant_doc_ids: set[str], hit_doc_ids: list[str]) -> float:
    if not relevant_doc_ids:
        return 1.0
    return len(relevant_doc_ids.intersection(hit_doc_ids)) / len(relevant_doc_ids)


def _citation_validity(answer: Answer) -> float:
    if answer.fallback:
        return 1.0 if not answer.citations else 0.0
    if not answer.citations:
        return 0.0
    available = {h.chunk_id for h in answer.hits}
    return 1.0 if set(answer.citations).issubset(available) else 0.0


def evaluate_pipeline_week6(pipeline: Any, cases: list[Week6EvalCase], cfg: Week6RunConfig) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    hit_flags: list[float] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    fallback_ok: list[float] = []
    citation_present: list[float] = []
    citation_valid: list[float] = []
    answer_nonempty: list[float] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        ans: Answer = pipeline.ask(case.question)
        latency = (time.perf_counter() - started) * 1000.0
        relevant = set(case.relevant_doc_ids)
        hit_doc_ids = [h.doc_id for h in ans.hits]
        hit_chunk_ids = [h.chunk_id for h in ans.hits]
        hit = float(bool(relevant.intersection(hit_doc_ids))) if relevant else float(ans.fallback)
        rr = _rr(relevant, hit_doc_ids) if relevant else float(ans.fallback)
        precision = _retrieval_precision(relevant, hit_doc_ids)
        recall = _retrieval_recall(relevant, hit_doc_ids)
        f_ok = float(ans.fallback == case.should_fallback)
        c_present = float(ans.fallback or bool(ans.citations))
        c_valid = _citation_validity(ans)
        nonempty = float(bool(ans.text.strip()))

        hit_flags.append(hit)
        reciprocal_ranks.append(rr)
        precisions.append(precision)
        recalls.append(recall)
        fallback_ok.append(f_ok)
        citation_present.append(c_present)
        citation_valid.append(c_valid)
        answer_nonempty.append(nonempty)
        latencies.append(latency)

        raw_rows.append({
            "case_id": case.case_id,
            "question_type": case.question_type,
            "question": case.question,
            "reference_answer": case.reference_answer,
            "relevant_doc_ids": list(case.relevant_doc_ids),
            "should_fallback": case.should_fallback,
            "answer": ans.text,
            "fallback": ans.fallback,
            "citations": list(ans.citations),
            "hit_doc_ids": hit_doc_ids,
            "hit_chunk_ids": hit_chunk_ids,
            "retrieved_contexts": [h.text for h in ans.hits],
            "latency_ms": latency,
            "retrieval_hit": hit,
            "reciprocal_rank": rr,
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "fallback_correct": f_ok,
            "citation_present": c_present,
            "citation_valid": c_valid,
        })

    return {
        "run_config": asdict(cfg),
        "case_count": len(cases),
        "metrics": {
            "retrieval": {
                "hit_rate_at_k": _mean(hit_flags),
                "mrr": _mean(reciprocal_ranks),
                "doc_precision_at_k": _mean(precisions),
                "id_context_recall": _mean(recalls),
            },
            "generation": {
                "citation_success": _mean(citation_present),
                "citation_validity": _mean(citation_valid),
                "answer_nonempty_rate": _mean(answer_nonempty),
            },
            "fallback": {
                "fallback_accuracy": _mean(fallback_ok),
            },
            "latency": {
                "latency_ms_mean": _mean(latencies),
                "latency_ms_p95": _p95(latencies),
            },
        },
        "raw_rows": raw_rows,
    }


def save_run(result: dict[str, Any], *, summary_path: Path, raw_log_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "raw_rows"}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with raw_log_path.open("w", encoding="utf-8") as f:
        for row in result["raw_rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare_week6_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ma = a["metrics"]
    mb = b["metrics"]
    def delta(group: str, key: str) -> float:
        return float(mb[group][key]) - float(ma[group][key])
    return {
        "baseline": a["run_config"]["system_name"],
        "candidate": b["run_config"]["system_name"],
        "same_case_count": a["case_count"] == b["case_count"],
        "delta": {
            "hit_rate_at_k": delta("retrieval", "hit_rate_at_k"),
            "mrr": delta("retrieval", "mrr"),
            "citation_validity": delta("generation", "citation_validity"),
            "fallback_accuracy": delta("fallback", "fallback_accuracy"),
            "latency_ms_mean": delta("latency", "latency_ms_mean"),
            "latency_ms_p95": delta("latency", "latency_ms_p95"),
        },
    }
