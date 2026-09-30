from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from intellectagent.embeddings import SentenceTransformerEmbedder

SAMPLES = [
    "RAG는 검색된 근거를 바탕으로 답변을 생성합니다.",
    "문서 검색과 생성 모델을 결합합니다.",
    "이 문장은 임베딩 배치 검사용 샘플입니다.",
    "폴백은 근거가 부족할 때 답변을 중단합니다.",
    "재현성은 동일 환경과 설정에서 결과를 다시 얻는 능력입니다.",
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()
    embedder = SentenceTransformerEmbedder(args.model)
    vectors = embedder.embed(SAMPLES)
    invalid = []
    dims = set()
    for i, v in enumerate(vectors):
        dims.add(len(v))
        if not v or any(not math.isfinite(x) for x in v) or sum(abs(x) for x in v) == 0:
            invalid.append(i)
    payload = {"model": args.model, "count": len(vectors), "dims": sorted(dims), "invalid_rows": invalid}
    out = ROOT / "evidence" / "week1" / "embedding_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if len(vectors) == len(SAMPLES) and len(dims) == 1 and not invalid else 2

if __name__ == "__main__":
    raise SystemExit(main())
