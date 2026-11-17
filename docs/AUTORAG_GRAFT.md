# AutoRAG 접목 현황

원본: AutoRAG `legacy/`(Python, Apache-2.0)의 정의와 2.0의 아이디어. 코드는 복사하지 않고 *정의를 따라 다시 구현*했고, 아래 항목은 원본 함수와 직접 대조했다.

| 기능 | 우리 모듈 | 원본 대조 |
|---|---|---|
| 검색 지표 6종 (F1/recall/precision/nDCG/MAP/MRR) | `retrieval_metrics.py` | `scripts/autorag/crosscheck.py` — 최대 오차 0 |
| 내용 기반 토큰 지표 3종 (청킹 무관 비교) | `retrieval_contents.py` | `crosscheck_more.py` — 9,000건 오차 2e-16 |
| BM25 (okapi/lucene idf), 문자 bigram, kiwi(선택) | `lexical.py` | rank_bm25 와 오차 1.8e-15 (kiwi 는 미실행) |
| RRF 융합 | `hybrid.py::rrf_fuse` | 원본 `rrf_pure` 와 불일치 0 |
| CC 융합 (mm/tmm/z/dbsf) | `hybrid.py::cc_fuse` | 3,000건 점수열 불일치 0 (동점 순서만 252건 다름) |
| 격자 탐색 + 최적 선택 (mean/rank/normalize_mean, 속도 임계값) | `grid_search.py` | `crosscheck_strategy.py` — 6,000건 불일치 0 (수학적 동점 4건 제외) |
| 청킹 x 검색 x (리랭커) 격자 | `chunk_grid.py` | 원본에 대응 함수 없음(구성은 자체) — 테스트로 검증 |
| 질문·정답 자동 생성, dontknow/passage_dependency 필터, 근거 추출 | `qa_gen.py` | 원본은 LLM 판정, 여기서는 규칙 기반 + 선택적 LLM — 동작이 다르다 |
| HyDE | `query_expansion.py` | 프롬프트 형식 동일, LLM 은 주입 |
| 문서 파서 (docx/xlsx/pptx/hwpx/hwp) | `doc_parsers.py` | 원본 legacy 는 클라우드 파서, 2.0 은 Node(kordoc) — 대응 아님 |
| 파싱 캐시 (2.0 의 parsed mirror) | `parse_cache.py` | 아이디어만 차용 |

## 아직 없는 것
- 한국어 리랭커 ko-reranker: 코드는 필요 없을 가능성이 높다 — `config.reranker_model` 을 `Dongjin-kr/ko-reranker` 로 바꿔 `CrossEncoderReranker` 로 로드되는지 확인하면 된다(추정, 이 환경에서는 모델을 내려받지 못해 미확인).
- 생성 지표(bleu, meteor, rouge, sem_score, g_eval, bert_score), multi_query_expansion, query_decompose, passage_filter, passage_augmenter, 프롬프트 메이커, 압축기.
- Kiwi/Okt/Kkma 형태소 토크나이저 실행 검증(kiwi 코드는 있으나 미실행, okt/kkma 없음).
- 파이프라인(`pipeline.py`)·Streamlit 앱에 `HybridRetriever`/`HydeRetriever` 연결. 현재는 격자 탐색·스크립트에서만 쓴다.
- 실제 문서 + 의미 임베딩 + 검수된 질문셋에서의 성능 수치. 데모 숫자는 합성 데이터이며 주장이 아니다.

## 주의
- 청크를 보고 LLM 이 만든 질문은 청크 낱말을 쓰기 쉬워 BM25 에 유리하다. `QARecord.lexical_overlap` 으로 층을 나눠 보고하라.
- `qa_gen` 산출물은 `status=draft`; `load_approved()` 만 평가에 쓴다.
- 파싱 캐시는 `feature_flags.use_parse_cache=true` 일 때만 켜지며 기본은 꺼짐. 캐시 폴더는 `allowed_root` 밖에 둔다.

## 실문서 점검에서 추가된 것 (문서 파서)

실제 파일(PDF·docx·pptx·doc·ppt·odt·ods)을 LibreOffice / pdftotext 결과와 대조해 찾은 결함과 조치.

| 결함 | 조치 |
|---|---|
| PDF 추출에 단어당 1줄(공백·빈 줄 ~33%), 합자(ﬁ ﬂ 등) 때문에 "official" 같은 검색어가 맞지 않음 | `doc_parsers.clean_pdf_page`: 합자 풀기, NUL 제거, 단어당 1줄 페이지 재결합 |
| `.doc` `.ppt` `.odt` `.ods` `.odp` 미지원 | 순수 파이썬 파서 추가 (OLE 조각표 / PPT 레코드 트리 / ODF XML). 암호화·손상 파일은 `ParseError` |
| `.ppt` 슬라이드 순서·노트 연결, 필드 자리표시자(`*`, `<number>`) 혼입 | 영속 디렉터리 기반 순서, `notesIdRef` 연결, 자리표시자 제거 |

한계: `.ppt` 표는 셀이 줄 단위로 나옴, `.xls`/RTF 미지원, 스캔 PDF OCR 없음, 실제 한컴 `.hwp/.hwpx` 파일은 미검증.
기본 파일 크기 제한(10MB)을 넘는 문서는 거부된다(설정값이며 결함 아님).

## 파이프라인 연결 (설정만으로 켠다)

`config/core*.json` 의 `feature_flags` 에 추가한다. 없으면 기존 동작(벡터 검색) 그대로다.

| 키 | 값 | 설명 |
|---|---|---|
| `retrieval_mode` | `vector`(기본) / `bm25` / `hybrid_rrf` / `hybrid_cc` | 검색 방식 |
| `bm25_tokenizer` | `char_bigram`(기본) 등 `lexical.TOKENIZERS` | BM25·하이브리드의 토크나이저 |
| `hybrid_cc_weight`, `hybrid_cc_normalize` | 0~1(기본 0.5), `mm/tmm/z/dbsf` | `hybrid_cc` 에서만 사용 |

- BM25 는 벡터 저장소(InMemory/Chroma)의 청크로 만들고, 문서가 추가·삭제되면 다음 검색 때 자동으로 다시 만든다.
- **판정 기준 주의:** RRF 점수는 최대 ≈0.03, `mm` 융합은 1순위가 항상 1.0 이라 `fallback_min_score`(0.12)를 그대로 쓸 수 없다. 하이브리드는 결과마다 `gate_score`(벡터 코사인 원점수)를 달고, CoreRAG 는 그 최댓값으로 "근거 없음"을 판정한다. 리랭커를 켠 경우 기존처럼 리랭커 점수가 기준이다.
- `bm25` 단독 모드는 리랭커를 쓰지 않고 BM25 점수가 판정 기준이다(0.12 의 적절성은 코퍼스별로 확인 필요 — 추정).
- 어떤 방식이 좋은지는 `scripts/autorag/demo_chunk_grid.py` 와 BGE-M3 로 직접 비교해 정한다. 기본값을 바꾸기 전에는 반드시 그 비교를 거친다.

## 실문서 비교 실행 순서 (팀 컴퓨터)

```bash
export PYTHONPATH=src
# 1) 질문 초안 (먼저 --stub 으로 배관 확인)
python scripts/autorag/gen_qa.py --root <문서폴더> --out evidence/qa --model qwen2.5:14b --n 60 --unanswerable 10 --parse-cache <허용 루트 밖>
# 2) evidence/qa/qa_draft.jsonl 을 사람이 검수: 맞는 줄만 "status": "approved"
# 3) 비교 (BGE-M3). 배관만 볼 때는 --embedder hash
python scripts/autorag/run_real_grid.py --root <같은 문서폴더> --qa evidence/qa/qa_draft.jsonl --out evidence/grid
#    선택: --chunks 300:30,500:50,700:80,1000:100  --retrievers vector,bm25,hybrid_rrf,hybrid_cc  --reranker-model Dongjin-kr/ko-reranker
```
산출물: `report.md`(★=tok_f1 최고, overlap 높음/낮음 분리표), `grid_result.csv/json`, `run_info.json`.
approved 질문이 없거나 정답 문서 ID 가 --root 와 맞지 않으면 오류로 멈춘다(다른 폴더로 만든 질문셋을 잘못 쓰는 것을 막기 위함).

## 여러 폴더(큰 말뭉치) 색인

AutoRAG 도 PC 를 무작정 전부 읽지 않는다. 사용자가 *승인한 문서 루트*만 색인하고, 시스템·캐시·자격증명 폴더는 제외하며, 정확히 같은 파일은 가장 최근본 하나만 남긴다(dupey). 같은 원칙을 `corpus_scan.py` 로 구현했다.

1. 후보 보기(색인 안 함): `PYTHONPATH=src python scripts/autorag/scan_corpus.py --suggest`
2. 점검(읽기 전용): `... scan_corpus.py --roots ~/Documents,~/Downloads --manifest evidence/corpus_manifest.json`
   → 색인 대상 수·용량, 확장자별 수, 제외된 것과 **이유별 개수·예시**, 중복 제외, 지난번 대비 추가/수정/삭제
3. 켜기: `feature_flags` 에 `"search_roots": ["/승인한/폴더", ...]` (기존 `allowed_root` 와 합쳐진다). 선택: `"exclude_exact_duplicates": false`, `"allow_home_root": true`
4. 평소처럼 색인(CLI `index` / Streamlit). 문서가 추가·변경되면 해시로 감지해 그 문서만 다시 색인한다. BM25 는 자동 반영.

안전장치(코드로 강제): 파일시스템 루트·시스템 폴더·홈 폴더 전체 거부(홈은 `allow_home_root`), 숨김/심볼릭 링크/`.ssh` `.aws` `node_modules` `.git` 등 제외, `비밀번호*` `credentials*` `*.pem` `.env*` 같은 파일명 제외, 색인·증거·파싱캐시 폴더는 스캔에서 제외. 원본은 읽기만 한다.
새 형식: `.eml`(제목·보낸이·받는이·날짜·본문·첨부 이름. 첨부 내용은 읽지 않음).

한계: 삭제된 파일의 청크를 색인에서 지우는 기능은 기존 `Indexer` 에도 없다(`removed` 는 보고만 한다). 중복으로 빠진 옛 사본이 이미 색인돼 있었다면 남는다. `.xls`·`.mbox`·카카오톡/노션 같은 데이터소스, MinSync/Jikji 는 구현하지 않았다.
색인에는 PC 문서 내용이 들어간다. 인덱스 폴더와 Streamlit 은 본인 PC 안에서만 쓰고 외부에 노출하지 말 것.

## 더 크게 색인하기 ① 문맥 머리말 ② BGE-M3 다중 표현

### ① 문맥 머리말 (`context_headers.py`, LLM 호출 없음)
각 청크 앞에 `[파일명 > 장 > 절]` 을 **색인용 글에만** 붙인다. `chunk.text`(화면·인용·평가)는 그대로다.
- 켜기: `feature_flags.context_headers: true` (색인 시), 또는 비교 때 청킹 설정 뒤에 `:ctx` (`--chunks 500:50,500:50:ctx`).
- ⚠ **이미 색인한 문서는 다시 색인되지 않는다**(해시가 같으면 건너뜀). 이 값을 바꾸면 `index_dir`, 메타데이터 DB, `evidence/index_checkpoint.json` 을 지우고 다시 색인할 것.
- 제목 감지는 보수적이다: `#`, `제N장/절`, `1.` `1.2` + 짧은 제목, 파서 표지(`[슬라이드 n]` `[시트 x]`). 제목이 없는 문서는 파일 이름만 붙는다. 청크가 여러 절에 걸치면 *시작 위치*의 절 경로를 쓴다.
- 파일 이름이 모든 청크에 들어가므로 파일 이름이 의미 없는 해시이면 잡음이다. 효과는 반드시 `run_real_grid.py` 로 확인한다(좋아질지 나빠질지 이 문서들에서는 아직 측정하지 못했다).

### ② BGE-M3 다중 표현 (`bge_m3.py`)
한 모델의 dense + sparse(어휘 가중치) + ColBERT(토큰 벡터)를 함께 쓴다. 점수 공식은 FlagEmbedding 1.3.5 소스와 같다(합산은 0.4:0.2:0.4, 가중치 합으로 나눔).
- 준비: `pip install FlagEmbedding`. 켜기: `feature_flags`: `"retrieval_mode": "bge_m3"`, `"use_sentence_transformer": true`(실모델; 끄면 해시 가짜 인코더). 선택: `m3_weights: [0.4,0.2,0.4]`, `m3_colbert_rerank: false`.
- 동작: 전체를 dense+sparse 로 훑어 후보 `candidate_k`(설정값)개를 고른 뒤 **후보만** ColBERT 로 재채점한다. ColBERT 를 청크마다 저장하면 용량이 폭증(청크 3천 개 ≈ 1GB 이상)해서 질의 때 계산하고 LRU(512)로 재사용한다.
- dense/sparse 는 `index_dir` 옆 `m3_cache.sqlite` 에 저장된다(키 = 모델+색인 글). 문서가 추가되면 새 청크만 인코딩한다. `cli index` 는 끝에 미리 인코딩(`warm`)하므로 첫 질문이 느리지 않다. Streamlit 은 첫 질문에서 인코딩한다.
- "근거 없음" 판정은 dense 코사인 원점수(`gate_score`)로 한다. BGE-M3 의 무관한 문서 코사인은 높게 나오는 편이라 `fallback_min_score` 를 실측으로 다시 정해야 할 수 있다(추정).
- 한계: 이 모드는 `use_reranker` 를 쓰지 않는다. 색인 때 기존 SentenceTransformer 임베더도 dense 를 한 번 더 계산한다(중복 비용). **실제 BGE-M3 모델로는 실행하지 못했다**(이 환경에서 모델 다운로드 불가) — 래퍼는 FlagEmbedding 인터페이스를 흉내 낸 가짜 모델로, 점수 계산은 공식 정의를 따로 구현한 기준값과 대조해 검증했다.
- 비교: `--retrievers vector,bm25,hybrid_rrf,bge_m3 --m3-ablation` → dense / sparse / dense+sparse / 전부를 나란히 본다. `--m3-cache m3.sqlite` 로 재실행 비용을 줄인다.

## 실환경 검증 (팀 컴퓨터에서)
합성 테스트로는 모델·한컴 파일의 실제 동작을 알 수 없다. `scripts/autorag/check_real_docs.py`(실문서 파서 점검, 사람이 고른 기대 문장으로 대조)와 `scripts/autorag/real_env_eval.py`(hit/MRR 신뢰구간, fallback 임계값 보정, Ollama 종단 실행)의 절차는 [`docs/REAL_ENV_CHECKLIST.md`](REAL_ENV_CHECKLIST.md) 를 따른다. 두 도구는 이 문서를 쓴 환경에서 실모델·실문서로 실행해 보지 못했다.

## 참고 저장소 대조에서 고친 것 (2026-10)
| 항목 | 내용 |
|---|---|
| fallback 판정 일치 | `gating.gate_score` 한 곳으로 통일. 예전에는 워크플로(LangGraph 노드)가 `hits[0].score` 만 봐서 하이브리드(RRF≈0.03)에서는 코사인이 충분해도 항상 거절했다. CoreRAG 와 워크플로가 같은 입력에서 같은 결정을 내리는지 테스트로 고정. |
| 인용 검증 완화 | `답변:` 처럼 20자 이하 + 콜론으로 끝나는 *머리말 줄*은 인용을 요구하지 않는다. 그 밖의 모든 줄에는 여전히 인용이 필요하다. |
| 번호 인용 (옵션) | `feature_flags.citation_style: "index"` 이면 프롬프트에 `[1]..[k]` 를 쓰고 검증기가 hits 순서로 chunk_id 에 대응시킨다. 기본 `"id"`(기존 동작). 20자 ID 를 베끼다 틀려 정답이 거절되는 빈도는 **측정하지 못했다** — 팀 PC 에서 `real_env_eval.py --e2e` 를 두 설정으로 돌려 "근거 있는 질문을 답한 비율"을 비교한 뒤 기본값을 정한다. |
| 알려진 한계 | `validate_grounding` 은 인용의 *형식*만 본다. 유효한 인용을 붙이면 내용이 틀려도(`연차 30일 [id]`, 근거는 15일) 통과한다. 의미 검증(청크별 관련성 판정, 문장 단위 지지 확인)은 아직 없다. |

## 청크별 관련성 판정 (`relevance_judge.py`, 기본 꺼짐)
고정 `fallback_min_score` 는 검색 방식·임베더마다 다시 보정해야 한다. 이 옵션은 로컬 LLM 에게 청크마다 "이 문서가 질문에 답할 정보를 담고 있나? Yes/No" 를 물어, 답이 못 되는 청크를 버린다. 모두 버려지면 CoreRAG 가 기존 방식대로 거절한다.
- 켜기: `feature_flags`: `"use_relevance_judge": true` (+ `use_ollama`). 선택: `judge_model`(기본 `ollama_model`), `judge_std_multiplier`(기본 1.0), `judge_max_candidates`(기본 top_k), `judge_logprobs`(기본 true).
- 규칙: 점수 = log P(Yes) − log P(No) (로그확률이 없으면 텍스트로 +1/−1). ① 점수 > 0 인 청크만 남기고 ② 남은 청크 중 `평균 − n·표준편차` 미만을 더 버린다. ①이 절대 기준이라 "전부 관련 없는데 그중 나은 것"이 통과하지 않는다. n 기본 1.0 은 필요한 근거를 버리지 않으려는 보수적 값(MAIN-RAG 원문 기본은 0).
- **fail-open**: LLM 오류·판독 불가면 원래 검색 결과를 돌려준다(기존 gate·grounding 은 그대로 적용). `RelevanceJudge.last_error` 에 남고, `real_env_eval.py` 보고서에 경고로 나온다.
- 비용: 질문당 최대 `judge_max_candidates` 회 LLM 호출. 지연시간은 `real_env_eval.py --judge` 의 p50/p95 로 확인한다.
- **미검증**: 실제 Ollama/Qwen 으로 돌려 보지 못했다. Ollama 로그확률은 v0.12.11 부터 지원(공식 릴리스 노트)하지만 응답 구조를 확인하지 못해 파서가 추정에 기반한다(실패하면 텍스트로 대체). 한국어에서 14B 판정이 얼마나 정확한지, "Yesterday" 같은 접두어 오판 위험, 효과 유무 모두 측정 전이다.
- 측정: `python scripts/autorag/real_env_eval.py ... --judge` — 각 후보 옆에 `+judge` 후보가 추가되어 hit/MRR·오차단율·정거절률·지연시간을 나란히 본다. 기본값을 켜기 전에 정거절률이 오르고 오차단율이 허용치(10%) 이내인지 확인한다.

## 시험 문제 품질 관리 (Agentic-AutoRAG 아이디어 적용)

평가 점수는 문제집이 나쁘면 의미가 없다. 다음 장치를 추가했다. 전부 휴리스틱이며 사람 검수를 대체하지 않는다(추정).

- **오라클 답변 가능성 검사** `gen_qa.py --oracle [--oracle-threshold 0.5]`: 정답 근거 텍스트만 LLM 에 주고 다시 답하게 해서, "알 수 없음"이거나 기존 답과 어긋나면 초안에서 제외(`oracle_unanswerable` / `oracle_mismatch`). LLM 오류는 제외하지 않고 `oracle_error` 표시만 한다. 답 없는 질문(should_fallback)은 호출하지 않는다.
- **질문 유형** `qtype`(비교/수치/절차/정의/사실, 규칙 기반): JSONL 마지막 필드라 예전 파일과 호환. real_env_eval 이 유형별 hit/MRR 을 따로 보고한다.
- **정답 문서 어휘 겹침 구간**: high/mid/low 로 나눠 층화 보고(겹침이 높은 문제만 잘 풀면 점수가 부풀려진다).
- **실패 유형 분류**(`--e2e`): correct_refusal / hallucinated_answer / retrieval_miss / refused / wrong_source / answer_mismatch / ok. wrong_source·answer_mismatch 는 어휘 기반 추정이라 사람 확인이 필요하다.
- **Pareto 프런티어**: MRR↑ · p50 지연↓ 기준으로 지배되지 않는 후보만 표시.
- **never-found 목록**: 모든 후보가 못 찾은 문제 ID. 검색기 문제가 아니라 문제/문서 문제일 가능성을 먼저 의심한다.
