# IntellectAgent

**문서 기반 AI 지식 검색 도우미** — 내 컴퓨터에 있는 문서(한글 `.hwp`·`.hwpx`, 워드, 엑셀, 파워포인트, PDF, 마크다운, 텍스트)를 찾아 읽고, **근거가 확인된 답만** 출처와 함께 돌려주는 로컬 RAG 프로그램입니다.

KDT 생성AI 10회차 파이널 프로젝트 (6팀).

- 모든 처리는 내 컴퓨터에서 합니다. 문서가 외부 서버로 나가지 않습니다 (웹 검색 보조 기능을 켠 경우의 검색어만 예외).
- 답변의 모든 문장에 인용이 붙어 있는지 코드로 검사하고, 통과하지 못하면 답하지 않고 거절합니다.
- 근거 문서를 찾지 못하면 지어내지 않고 "찾지 못했습니다"라고 답합니다.

---

## 목차

1. [동작 방식](#동작-방식)
2. [지원하는 문서 형식](#지원하는-문서-형식)
3. [측정 결과와 한계](#측정-결과와-한계)
4. [설치와 실행 (macOS · Linux · Windows)](#설치와-실행-macos--linux--windows)
5. [내 문서로 점검·평가하기](#내-문서로-점검평가하기)
6. [폴더 구조](#폴더-구조)
7. [테스트](#테스트)
8. [설정](#설정)
9. [알려진 한계](#알려진-한계)
10. [출처와 라이선스](#출처와-라이선스)

---

## 동작 방식

```text
질문
 │
 ├─ 색인된 문서가 있나?
 │    ├─ 있음 ─▶ 검색(BGE-M3 · ChromaDB · 재순위) ─▶ 점수 관문 ─▶ Ollama 답변 생성 ─▶ 근거 검증 ─▶ 출처와 함께 답변
 │    │                                              │                                     │
 │    │                                   점수가 낮으면 거절                  인용 누락·오류면 거절
 │    └─ 없음 ─▶ 웹 검색 보조(선택) ─▶ 그래도 없으면 Ollama 내부 지식 (※ 외부 검증 안 됨 표시)
```

핵심 장치는 두 개입니다.

| 장치 | 하는 일 | 코드 |
|---|---|---|
| **점수 관문** | 검색 결과의 점수가 기준(`fallback_min_score`, 기본 0.12)보다 낮으면 생성기를 부르지 않고 거절 | `gating.py`, `pipeline.py` |
| **근거 검증** | 답변의 모든 줄에 인용이 있고, 인용이 실제 검색된 근거를 가리키는지 확인. 하나라도 어기면 거절 | `grounding.py` |

사용 기술: Python 3.11 · Ollama(`qwen2.5:14b`) · BGE-M3 · ChromaDB · LangGraph · RAGAS · Streamlit

## 지원하는 문서 형식

`.md` `.txt` `.pdf` `.docx` `.xlsx` `.pptx` `.hwpx` `.hwp`(5.x)

- `docx`·`xlsx`·`pptx`·`hwpx`는 표준 라이브러리만으로 읽습니다.
- `hwp`는 `olefile`이 필요합니다 (`requirements.txt`에 포함).
- 읽지 못하는 파일(암호 걸린 문서, 배포용 HWP, 손상 파일)은 조용히 건너뛰지 않고 **이유를 남깁니다**.
- 보안: 압축폭탄 상한, XML 엔티티 폭주 차단, 경로 이탈·심볼릭 링크 거부.
- 한글: 자모가 풀린 텍스트(NFD)는 읽을 때 완성형(NFC)으로 맞춥니다.

## 측정 결과와 한계

실제 업무 문서 12개(`.docx`)로 측정했습니다. 질문은 AI가 만든 뒤 사람이 승인했습니다 (답이 있는 질문 37개, 답이 없어야 하는 질문 5개).

| 항목 | 결과 |
|---|---|
| 검색 적중률 (정답 문서가 상위 5개 안) | 1.000 |
| 검색 MRR | 0.955 |
| 답변 성공 (답이 있는 질문 중 답변한 비율) | 32 / 37 = 0.86 |
| 답이 없는 질문을 거절한 비율 | 5 / 5 |
| 응답 시간 (중앙값 / 상위 5%) | 약 14초 / 약 24초 |

이 숫자를 읽을 때 주의할 점입니다.

- 문서가 12개뿐이고 모두 한 사람의 기술 문서입니다. 다른 종류의 문서에서는 결과가 달라질 수 있습니다.
- 질문은 AI가 만들었습니다. 사람이 쓴 질문과 분포가 다를 수 있습니다.
- 답이 없는 질문이 5개뿐이라 "거절 5/5"의 신뢰 구간이 매우 넓습니다.
- 답변이 정확한지는 자동으로 알 수 없습니다. 위 "답변 성공"은 근거 검증을 통과했다는 뜻이며, 내용이 맞는지는 사람이 읽어 확인해야 합니다. 출처 문서가 어긋난 것으로 보이는 경우(`wrong_source`)가 5건 있었습니다 (글자 겹침 기준의 추정).
- 점수 관문만으로는 답이 없는 질문을 걸러내기 어렵습니다 (관문 AUC 0.70). 실제로 거절된 5건은 모두 근거 검증 단계에서 걸러졌습니다.
- 인용 형식은 `[1]`처럼 번호를 쓰는 방식(`citation_style: "index"`)이 20자리 청크 ID 방식보다 크게 좋았습니다 (같은 질문 42개로 비교: 답변 성공 0/37 → 32/37, 0.86). 생성 모델이 긴 ID를 정확히 옮겨 적지 못했기 때문입니다.

평가 방법은 [`docs/AUTORAG_GRAFT.md`](docs/AUTORAG_GRAFT.md)와 `scripts/autorag/real_env_eval.py`를 보세요. 평가 보고서는 `bash run.sh eval`을 실행하면 `evidence/real_eval/report.md`로 만들어집니다.

## 설치와 실행 (macOS · Linux · Windows)

압축을 풀고 설치기를 한 번 실행하면 가상환경, torch, 패키지, Ollama, 모델, 시험까지 알아서 진행합니다. OS 는 자동으로 감지합니다.

| | macOS / Linux | Windows |
|---|---|---|
| 설치 (처음 1회) | `bash install.sh` | `install.bat` (더블클릭 가능) |
| 실행 | `bash run.sh app` | `run.bat app` |

```bash
unzip IntellectAgent_full.zip && cd IntellectAgent
bash install.sh          # Windows: install.bat
bash run.sh doctor       # 환경 진단
bash run.sh app          # 브라우저에서 http://localhost:8501
```

**필요한 것**: Python **3.11** (3.12 이상은 지원하지 않습니다). 없으면 설치기가 설치 방법을 알려 줍니다.

| OS | 파이썬 3.11 설치 | Ollama |
|---|---|---|
| macOS | `brew install python@3.11` | 설치기가 `brew install ollama` 를 제안 |
| Linux | `sudo apt install python3.11 python3.11-venv` | 설치기가 공식 설치 스크립트를 제안 |
| Windows | `winget install -e --id Python.Python.3.11` | 설치기가 `winget install Ollama.Ollama` 를 제안 |

**설치기가 하는 일** (`scripts/setup_env.py`)

1. 컴퓨터 점검 — OS, CPU, 메모리, 디스크, GPU. 메모리 16GB 미만이면 14B 모델에 대해 경고합니다.
2. 가상환경 `.venv-real` 만들기
3. torch — Apple 실리콘은 MPS, NVIDIA 는 CUDA, 그 외는 CPU 전용(작음)
4. 패키지 — `requirements-dev.txt` + `FlagEmbedding`
5. Ollama — 설치 확인(동의하면 설치) → 서버 시작 → `qwen2.5:14b` 내려받기 (약 9GB)
6. BGE-M3 임베딩 모델 내려받기 (약 2.3GB)
7. 테스트와 레드팀 시뮬레이션

결과는 화면 요약과 `install.log`, `install_report.json` 에 남습니다. 중간에 실패해도 같은 명령을 다시 실행하면 이미 끝난 단계는 건너뜁니다.

**설치 옵션**

| 옵션 | 뜻 |
|---|---|
| `--yes` | 묻지 않고 진행 (Ollama 같은 시스템 프로그램 설치는 `--install-ollama` 도 함께 줘야 함) |
| `--dry-run` | 아무것도 바꾸지 않고 할 일만 출력 |
| `--model qwen2.5:7b` | 가벼운 답변 모델 선택 (메모리가 작을 때) |
| `--skip-model` / `--skip-embedding` / `--skip-tests` | 해당 단계 생략 |
| `--recreate` | 가상환경을 지우고 다시 만들기 |

> 예전 명령 `bash run_real_env.sh setup|docs|qa|eval|app` 도 그대로 동작합니다(내부적으로 `install.sh`, `run.sh` 를 호출).
> Windows 설치 스크립트는 Linux 환경에서만 시험했고, Windows 실기에서는 아직 시험하지 못했습니다.

## 내 문서로 점검·평가하기

문서는 `data/real_docs/`에 넣습니다 (Git에는 올라가지 않습니다). 아래 순서로 한 단계씩 실행합니다.

| 단계 | 명령 | 하는 일 | 필요한 것 |
|---|---|---|---|
| 문서 점검 | `bash run.sh docs` | 파일이 제대로 읽히는지, 글자가 깨지지 않았는지 점검 | 모델 불필요 |
| 질문셋 만들기 | `bash run.sh qa` | 문서에서 질문 초안 생성 | Ollama |
| 평가 | `bash run.sh eval` | 검색·거절·종단 답변 평가, 합격/불합격 판정 | 승인된 질문셋 |

- 질문 초안은 사람이 승인해야 평가에 쓰입니다 (`scripts/review_qa.py`).
- 문서 옆에 `파일명.expect.txt`를 두면 "이 문서에서 읽혀야 할 문장"을 지정해 읽기 결과를 검증할 수 있습니다. `.expect.txt`는 색인 대상에서 제외됩니다.
- 평가 옵션(환경변수): `E2E=42`(종단 질문 수), `CITATION_STYLE=index`(기본), `EVAL_OUT=<폴더>`. 답변 생성은 평가 때 `temperature=0`, `seed=42`로 고정됩니다.
- 판정 종료 코드: 0 합격, 1 불합격, 2 입력 오류, 3 무효.

## 폴더 구조

```text
├── app_streamlit.py            # Streamlit 화면
├── install.sh · install.bat    # 설치기 (OS 자동 감지)
├── run.sh · run.bat · manage.py # 실행기: doctor/app/test/redteam/docs/qa/eval
├── run_real_env.sh             # 예전 명령 호환용 연결 스크립트
├── src/intellectagent/         # 핵심 패키지
│   ├── pipeline.py             #   질문 → 검색 → 관문 → 생성 → 검증
│   ├── doc_parsers.py          #   문서 형식별 텍스트 추출 (HWP/HWPX 포함, 순수 파이썬)
│   ├── grounding.py            #   인용 검증
│   ├── gating.py               #   점수 관문
│   ├── bge_m3.py · hybrid.py   #   임베딩·하이브리드 검색
│   └── generation.py           #   Ollama 답변 생성
├── scripts/
│   ├── autorag/                #   실문서 점검·질문 생성·평가 도구
│   ├── redteam/                #   반대 입장(레드팀) 시뮬레이션 하네스
│   ├── setup_env.py · envtools.py  # 설치기 본체·공통 도구
│   ├── mac/                    #   macOS 준비·실행 스크립트
│   └── strict/ · week1..9/     #   단계별 검증 게이트
├── config/                     # 실행·승격·릴리스 설정
├── tests/                      # 회귀 테스트 (실제 HWP 시험 파일 포함)
├── data/sample_docs/           # 공개 예제 문서
├── data/real_docs/             # 내 문서 (Git 제외)
├── docs/                       # 설계·실행·제출 문서
├── THIRD_PARTY_NOTICES.md      # 참고한 오픈소스와 라이선스
└── requirements.txt
```

## 테스트

```bash
bash run.sh test         # Windows: run.bat test
bash run.sh redteam      # 반대 입장 시뮬레이션 (결과: evidence/redteam/results.json)
```

모델이나 Ollama 없이 돌아갑니다 (가짜 생성기·해시 임베더 사용). 의미 검색 품질은 이 테스트로 보증하지 않으며, 위 [측정 결과](#측정-결과와-한계)의 실환경 평가로 확인합니다.
레드팀은 일부러 나쁜 모델 출력·악성 입력으로 앱을 공격해 보는 시험이며, 막은 것과 일부러 남긴 한계는 [`docs/REDTEAM_REPORT.md`](docs/REDTEAM_REPORT.md) 에 정리했습니다.

## 설정

설정 파일은 `config/core.production.json` 입니다. 주요 값:

| 키 | 기본값 | 뜻 |
|---|---|---|
| `fallback_min_score` | 0.12 | 이 점수보다 낮으면 거절 |
| `feature_flags.citation_style` | `"index"` | 인용 형식 (`index` = `[1]`, `id` = 청크 ID) |
| `feature_flags.ollama_temperature` / `ollama_seed` | 없음 | 지정하면 답변 생성을 고정 (재현용) |
| `feature_flags.grounding_check_numbers` | 꺼짐 | 켜면 답변의 숫자가 인용한 근거에 없을 때 거절 (지어낸 숫자 방지, 한글 수사·계산은 못 읽음) |
| `feature_flags.use_empty_rag_web_fallback` | true | 문서가 없을 때 웹 검색 보조 |
| `feature_flags.use_ollama_knowledge_fallback` | true | 웹 근거도 없을 때 Ollama 내부 지식 (외부 검증 안 됨 표시) |

환경 변수: `INTELLECTAGENT_CONFIG`, `OLLAMA_MODEL`, `RUNTIME_PROFILE`.

## 알려진 한계

- 실제 `.hwp`는 시험 파일 1개로만 확인했고, 실제 `.hwpx`와 PDF는 실문서로 검증하지 못했습니다.
- 스캔한 PDF(이미지)는 글자를 읽지 못합니다 (OCR 미지원).
- 근거 검증은 인용 *형식*을 확인합니다. 인용한 근거가 그 내용을 실제로 뒷받침하는지는 확인하지 못하므로(레드팀 보고서 G6·G13), 중요한 내용은 출처 문서로 확인해야 합니다.
- 답변 생성 모델(`qwen2.5:14b`)이 근거를 잘못 요약하는 경우가 있습니다.
- 응답 중앙값은 약 14초입니다 (맥북 M3 기준).
- 평가 규모가 작습니다 ([측정 결과와 한계](#측정-결과와-한계) 참조).
- 한 사람이 AI와 함께 만든 프로젝트입니다.

## 출처와 라이선스

주요 참고 대상은 아래와 같습니다. 모두 정의와 아이디어를 참고해 직접 다시 구현했고 코드는 복사하지 않았습니다.

| 참고 대상 | 라이선스 | 참고한 내용 |
|---|---|---|
| [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) | MIT (`legacy/`는 Apache-2.0) | 검색 지표, BM25, RRF·CC 융합, 격자 탐색, 질문 생성 흐름 |
| [Agentic-AutoRAG](https://github.com/Agentic-Systems-Lab/Agentic-AutoRAG) | MIT | 질문 유형별 시험 구성, 실패 유형 분류(아이디어) |
| [Haystack](https://github.com/deepset-ai/haystack) | Apache-2.0 | 번호 인용 `[1]` 방식 |
| [Self-RAG](https://github.com/AkariAsai/self-rag) | MIT | 관련성 판정 개념 |
| [kordoc](https://github.com/chrisryugj/kordoc) | MIT | 한글 문서 지원 범위 착안(코드 미사용) |

CRAG·HyDE·MAIN-RAG는 논문의 개념만 참고했습니다. 전체 목록과 정확한 저작권자는 `THIRD_PARTY_NOTICES.md`를 보세요.

사용한 모델·라이브러리의 라이선스와 위 항목의 전체 정보는 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에 정리했습니다. 앱 화면 왼쪽의 "오픈소스 고지"에서도 볼 수 있습니다.

이 프로젝트 자체의 라이선스는 아직 정하지 않았습니다. 저장소에 `LICENSE` 파일이 없는 동안에는 다른 사람이 코드를 쓸 권한이 없습니다. 공개 전에 소유자가 정해 파일을 추가하세요.
