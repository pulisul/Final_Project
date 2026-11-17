# 제3자 고지 (Third-Party Notices)

IntellectAgent는 아래 오픈소스 프로젝트와 논문의 **정의와 아이디어를 참고해 직접 다시 구현**했다.
라이선스 파일에서 확인한 저작권자·라이선스를 그대로 적는다. 이 문서는 법률 자문이 아니다.

## 0. 코드 복사 여부

- 참조 저장소 16곳과 이 프로젝트의 `src/` 파이썬 코드를 비교해 **공백을 무시한 연속 4줄(25자 이상) 일치 0건**을 확인했다.
- 이 검사의 한계: 파이썬 파일의 정확히 같은 줄만 찾는다. 변수명을 바꾼 복사, 프롬프트 문구, 문서 텍스트는 잡지 못한다.
- 일부 모듈 머리말에도 "재구현, 코드 복사 아님"과 출처를 적어 두었다(예: `relevance_judge.py`, `query_expansion.py`).
- 아래 라이선스의 고지 조건은 코드를 복사·재배포할 때 적용된다. 이 프로젝트는 복사하지 않았지만 출처를 밝힌다.

## 1. 실제로 반영한 것

| 프로젝트 | 저작권자 / 라이선스 (LICENSE 파일 기준) | 반영한 내용 | 우리 코드 |
|---|---|---|---|
| [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) | NomaDamas / Marker Inc. — 저장소 루트 **MIT**(2025), `legacy/` **Apache-2.0** | 검색 지표 6종, BM25, RRF·CC 융합, 격자 탐색과 최적 선택, 질문 생성 필터, HyDE, 파싱 캐시(아이디어) | `retrieval_metrics.py`, `retrieval_contents.py`, `lexical.py`, `hybrid.py`, `grid_search.py`, `chunk_grid.py`, `qa_gen.py`, `query_expansion.py`, `parse_cache.py` |
| [Agentic-AutoRAG](https://github.com/Agentic-Systems-Lab/Agentic-AutoRAG) | Lasse Bærland Strand, **MIT**(2026) | 질문 유형별 시험 구성, 정답 복원 가능성 검사, 실패 유형 분류(아이디어) | `qa_gen.py`, `real_eval.py` |
| [Haystack](https://github.com/deepset-ai/haystack) | deepset GmbH, **Apache-2.0** | 번호 인용(`[1]`) 방식 — `AnswerBuilder`의 방식을 참고 | `generation.py` (`citation_style="index"`) |
| Self-RAG ([저장소](https://github.com/AkariAsai/self-rag)) | Akari Asai, **MIT**(2023) | `IsRel`(관련성 판정) 개념 | `relevance_judge.py` |
| CRAG ([저장소](https://github.com/HuskyInSalt/CRAG)) | **LICENSE 파일 없음** → 사용 허락 없음. 코드 복사 없음 | 맞음/모호/틀림 구분 개념(논문) | `relevance_judge.py`, LangGraph 분기 이름 |
| HyDE (저장소 주소는 README에 없음) | **LICENSE 파일 없음** → 사용 허락 없음. 코드 복사 없음 | 가상 문서로 검색하는 개념(논문). 프롬프트는 한국어로 새로 씀 | `query_expansion.py` |
| MAIN-RAG (ACL 2025, 논문) | 저장소를 받지 않았고 라이선스를 확인하지 못함. 코드 복사 없음 | 판정 에이전트와 적응형 임계값(평균 − n·표준편차) 개념 | `relevance_judge.py` |

논문 출처: HyDE (arXiv:2212.10496), CRAG (arXiv:2401.15884), Self-RAG (arXiv:2310.11511), MAIN-RAG (ACL 2025).
저장소 링크는 각 저장소 README에 적힌 주소다. HyDE는 README에 저장소 주소가 없어 적지 않았다.

## 2. 분석만 하고 반영하지 않은 것

| 프로젝트 | 저작권자 / 라이선스 |
|---|---|
| LightRAG | LightRAG Team, MIT(2025) |
| GraphRAG | Microsoft Corporation, MIT |
| RAPTOR | Parth Sarthi, MIT |
| FLARE | Zhengbao Jiang, MIT(2023) |
| IRCoT | Apache-2.0 (LICENSE에 저작권자 이름란이 비어 있음. 논문 저자: Trivedi 외) |
| Adaptive-RAG | Apache-2.0 (LICENSE에 저작권자 이름란이 비어 있음. 논문 저자: Soyeong Jeong 외) |
| AutoRAG-Research | NomaDamas, Apache-2.0 (`pyproject.toml` 기준) |
| Open-RAG | LICENSE 파일이 Creative Commons BY 4.0 (코드에 CC BY를 쓴 점이 특이하므로 사용 전 확인 필요) |
| [kordoc](https://github.com/chrisryugj/kordoc) | chrisryugj, **MIT** (v4.15.7, AutoRAG `NOTICE` 기준) | AutoRAG 2.0 이 HWP·HWPX 등을 읽는 데 쓰는 라이브러리. 지원 형식 범위만 착안했고 코드는 쓰지 않음(우리는 `doc_parsers.py` 자체 구현) |
| RQ-RAG, RAG-query-rewriting | LICENSE 파일 없음 → 사용 허락 없음 |

### 시험용 파일

| 파일 | 출처 / 라이선스 |
|---|---|
| `tests/fixtures/hwp5/minimal-body-table.hwp` | [edwardkim/rhwp](https://github.com/edwardkim/rhwp) 커밋 e666360, **MIT** (전문: `tests/fixtures/hwp5/LICENSE.rhwp`). 실제 HWP 읽기 시험에만 사용 |

## 3. 실행에 쓰는 소프트웨어와 모델

| 이름 | 라이선스 | 확인 근거 |
|---|---|---|
| BGE-M3 (BAAI/bge-m3) | MIT | Hugging Face 모델 카드 `license: mit` |
| FlagEmbedding | MIT | 저장소 README "licensed under the MIT License" |
| Qwen2.5-14B-Instruct | Apache-2.0 | Hugging Face 모델 카드. Ollama의 `qwen2.5:14b`와 같은 모델인지는 확인하지 못함 |
| chromadb | Apache-2.0 | 설치된 패키지 메타데이터 |
| streamlit | Apache-2.0 | 설치된 패키지 메타데이터 |
| langgraph, langchain-text-splitters, ollama(파이썬 클라이언트), ddgs, python-docx, openpyxl, python-pptx | MIT | 설치된 패키지 메타데이터 |
| openai(파이썬 클라이언트) | Apache-2.0 | 설치된 패키지 메타데이터 |
| numpy, scikit-learn, pandas, pypdf, python-dotenv, olefile | BSD 계열 | 설치된 패키지 메타데이터 |
| sentence-transformers, langchain, langchain-ollama, ragas, datasets, plotly, torch, Ollama 서버 | **확인 필요** | 이 문서를 쓴 환경에 설치되지 않았음 |

확인 필요 항목은 아래로 확인해 표를 고친다.

    .venv-real/bin/pip show sentence-transformers langchain langchain-ollama ragas datasets plotly torch | grep -E "^(Name|License)"

## 4. 이 프로젝트 자체의 라이선스

저장소에 `LICENSE` 파일이 없다. 공개 저장소에 라이선스가 없으면 다른 사람은 기본적으로 사용 권한이 없다.
공개하기 전에 소유자(팀·교육기관 규정 포함)가 직접 정한다.
