"""AutoRAG 방식 격자 탐색 시연 — 손으로 만든 소규모 한국어 코퍼스(14문서/20질문).

⚠ 성능 주장이 아니다. 이 코퍼스와 질문은 같은 사람이 만들었고 HashEmbedder 는 의미 임베딩이 아니다.
  목적: (1) 격자 탐색·지표·선택 파이프라인이 끝까지 돈다 (2) 조사 변이/짧은 명사에서 어떤 차이가 생기는지 기전을 보인다.
  실제 주장은 실문서 + BGE-M3 + 라벨 검수된 질문셋으로 같은 스크립트를 다시 돌려 얻어야 한다.
사용: PYTHONPATH=src python scripts/autorag/demo_grid.py [출력폴더]
"""
from __future__ import annotations

import sys
from pathlib import Path

from intellectagent.chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from intellectagent.embeddings import HashEmbedder
from intellectagent.grid_search import Candidate, run_grid
from intellectagent.hashing import sha256_text
from intellectagent.hybrid import HybridRetriever
from intellectagent.lexical import BM25Index, tok_char_bigram, tok_word
from intellectagent.models import Document, EvalCase
from intellectagent.retrieval import Retriever
from intellectagent.vectorstore import InMemoryVectorStore

DOCS = {
    "d01": "연차 휴가는 입사 1년 후부터 15일이 부여되며, 신청은 팀장 승인을 거쳐 인사시스템에서 한다.",
    "d02": "서버 백업은 매일 새벽 2시에 자동으로 실행되고 사본은 다른 지역의 저장소에 30일 동안 보관된다.",
    "d03": "배송 지도는 고객 주소를 기준으로 경로를 계산하며 매주 월요일에 최신 도로 정보로 갱신된다.",
    "d04": "출장비는 교통비와 숙박비로 나뉘며 영수증을 출장 후 7일 이내에 정산 담당자에게 제출해야 한다.",
    "d05": "외부 반출 문서는 암호화가 의무이며 비밀번호는 별도 메신저로 전달해야 한다.",
    "d06": "서비스 장애가 발생하면 당직자가 15분 안에 상황을 공지하고 원인 분석 보고서를 3일 안에 작성한다.",
    "d07": "고객 불만은 접수 24시간 이내에 담당 부서로 배정되고 처리 결과는 고객에게 문자로 안내한다.",
    "d08": "복지몰 이용 대상은 만 나이가 19세 이상인 정규직 직원이며 계약직은 입사 6개월 후부터 이용할 수 있다.",
    "d09": "로그인 시도가 5회 연속 실패하면 계정이 잠기며 잠금 해제는 헬프데스크에서 본인 확인 후 처리한다.",
    "d10": "회의실은 사내 캘린더에서 예약하며 사용하지 않으면 시작 10분 후 예약이 자동으로 취소된다.",
    "d11": "신규 입사자 교육은 입사 첫 주에 진행되며 보안 교육과 업무 도구 교육이 포함된다.",
    "d12": "재택근무는 주 2회까지 가능하며 전날 오후 6시까지 팀장에게 신청해야 한다.",
    "d13": "법인카드는 업무 목적 사용만 허용되고 월말에 사용 내역을 증빙과 함께 제출한다.",
    "d14": "채용 공고는 인사팀이 작성하고 면접은 서류 합격 후 2주 이내에 진행한다.",
}
LEXICAL = [  # 어휘는 겹치지만 조사/어미가 달라진 질문 (16)
    ("지도를 언제 갱신하나요?", "d03"), ("휴가를 신청하려면?", "d01"), ("백업이 몇 시에 되나요?", "d02"),
    ("출장비를 정산할 때 영수증은?", "d04"), ("나이 제한이 있나요", "d08"), ("로그인을 시도하면 계정이 잠기나요", "d09"),
    ("고객의 불만은 얼마나 빨리 처리되나요", "d07"), ("장애가 나면 누가 공지하나요", "d06"), ("회의실을 예약하는 곳은", "d10"),
    ("재택근무를 하려면 언제까지 신청해야 하나", "d12"), ("법인카드를 개인적으로 써도 되나요", "d13"),
    ("신규 입사자는 어떤 교육을 받나요", "d11"), ("채용 면접은 언제 하나요", "d14"), ("암호화는 누가 하나요", "d05"),
    ("문서를 외부로 보낼 때 비밀번호는", "d05"), ("서버를 복사본으로 보관하는 기간", "d02"),
]
SEMANTIC = [  # 어휘 겹침이 없는 의미적 질문 (4) — 어휘 검색으로는 원리상 못 찾는다
    ("쉬는 날은 며칠인가요", "d01"), ("컴퓨터가 멈췄을 때 대응", "d06"), ("집에서 일해도 되나요", "d12"),
    ("길 안내 데이터는 어떻게 업데이트", "d03"),
]


def build():
    docs = [Document(i, Path(f"{i}.md"), t, sha256_text(t)) for i, t in DOCS.items()]
    chunks = [c for d in docs for c in chunk_with_strategy(d, ChunkStrategyConfig("length", 700, 80))]
    emb = HashEmbedder()
    store = InMemoryVectorStore(); store.upsert(chunks, emb.embed([c.text for c in chunks]))
    vec = Retriever(emb, store)
    bm_word = BM25Index(chunks, tok_word, idf_variant="lucene")
    bm_bi = BM25Index(chunks, tok_char_bigram, idf_variant="lucene")
    return [
        Candidate("vector(hash)", lambda q, k: vec.search(q, k), {"kind": "vector"}),
        Candidate("bm25(어절)", bm_word.search, {"tokenizer": "word"}),
        Candidate("bm25(bigram)", bm_bi.search, {"tokenizer": "char_bigram"}),
        Candidate("hybrid(vector+어절)", HybridRetriever(vec, bm_word).search, {"rrf_k": 60}),
        Candidate("hybrid(vector+bigram)", HybridRetriever(vec, bm_bi).search, {"rrf_k": 60}),
    ]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    cands = build()
    for title, qs in (("어휘 변이 질문 16개", LEXICAL), ("의미 질문 4개(어휘 겹침 없음)", SEMANTIC), ("전체 20개", LEXICAL + SEMANTIC)):
        cases = [EvalCase(f"q{i}", q, (d,)) for i, (q, d) in enumerate(qs)]
        res = run_grid(cands, cases, top_k=3, strategy="mean", out_dir=(out / title.split()[0] if out and title.startswith("전체") else None))
        print(f"\n=== {title}  (top_k=3, 선택 전략=mean) → 선택: {res['best']}")
        print(f"{'후보':24s} {'recall':>7s} {'prec':>6s} {'f1':>6s} {'ndcg':>6s} {'map':>6s} {'mrr':>6s}")
        for r in res["rows"]:
            print(f"{r['candidate']:24s} {r['retrieval_recall']:7.3f} {r['retrieval_precision']:6.3f} {r['retrieval_f1']:6.3f} "
                  f"{r['retrieval_ndcg']:6.3f} {r['retrieval_map']:6.3f} {r['retrieval_mrr']:6.3f}")


if __name__ == "__main__":
    main()
