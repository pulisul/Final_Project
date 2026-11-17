# 실환경 검증 체크리스트 (팀 컴퓨터)

합성 테스트(pytest)는 **가짜 모델·가짜 HWP** 로 코드의 자기 일관성만 보증한다. 아래 두 도구가 만드는 수치만이 실제 성능·실문서 증거다.
이 저장소를 만든 환경에서는 Ollama, BGE-M3, 한컴 원본 파일을 쓸 수 없어 **두 도구를 실모델·실문서로 돌려 보지 못했다.** 판정 기준(0.80/0.60/0.10/0.80 등)은 제안값(추정)이며 팀이 바꿔도 된다.

## 0. 준비
```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt FlagEmbedding olefile
ollama serve &                 # 이미 떠 있으면 생략
ollama pull qwen2.5:14b
export PYTHONPATH=src          # Windows PowerShell: $env:PYTHONPATH="src"
python -m pytest -q            # 먼저 합성 테스트가 통과하는지(환경 이상 확인)
```

## 1. 실문서 파서 점검 — `check_real_docs.py` (모델 불필요, 5분)
1. 실제로 쓸 문서(특히 .hwp/.hwpx/.doc/.ppt/.pdf)를 한 폴더에 모은다.
2. 문서마다 원본을 한컴/워드/뷰어로 열어 **본문에서 실제 문장 3~5개**를 눈으로 골라 `<파일명>.expect.txt` 에 한 줄씩 적는다(예: `규정.hwp.expect.txt`). 표 안의 문장 1개, 문서 뒷부분 문장 1개를 꼭 포함한다. 줄바꿈·공백 차이는 무시한다.
3. 실행:
```bash
python scripts/autorag/check_real_docs.py <문서폴더> --out evidence/real_docs_check --dump-dir evidence/real_docs_dump
```
4. 읽는 법: **OK** 만 증거(기대 문장이 모두 추출됨). `OK_UNVERIFIED` = 깨지진 않은 듯하나 증거 아님 → expect.txt 를 만든다. `SUSPECT` = 모지바케/글자 깨짐/기대 문장 누락. `FAIL` = 읽기 실패.
5. 실패 시 단서: 보고서의 HWP 구조(버전·압축·암호·**배포용**·DRM). 암호/배포용/DRM 문서는 지원하지 않는다 → 한컴에서 일반 문서로 다시 저장하거나 PDF/HWPX 로 변환.
6. `--dump-dir` 의 `*.extracted.txt` 를 원본과 나란히 열어 순서·표·누락을 눈으로 한 번 더 본다.

## 2. 질문셋 만들기 (Ollama 필요)
```bash
python scripts/autorag/gen_qa.py --root <문서폴더> --out evidence/qa --model qwen2.5:14b --n 60 --unanswerable 10 --oracle --parse-cache <문서폴더 밖 폴더>
```
`evidence/qa/qa_draft.jsonl` 을 사람이 읽고, 질문이 자연스럽고 정답 문서가 맞는 줄만 `"status": "approved"` 로 바꾼다. **답 있는 질문 30개 이상, 답 없는 질문(`should_fallback`) 5개 이상**이어야 수치를 믿을 수 있다. (모델이 만든 질문은 본문 단어를 그대로 쓰는 경향이 있어 검색이 쉬워진다 — 가능하면 일부는 직접 쓴다.)

## 3. 검색·fallback 평가 — `real_env_eval.py`
```bash
python scripts/autorag/real_env_eval.py --root <문서폴더> --qa evidence/qa/qa_draft.jsonl --out evidence/real_eval \
    --chunks 500:50,700:80 --retrievers vector,hybrid_rrf,bge_m3 --top-k 5 --e2e 20
```
- 먼저 사전 점검(임베더 의미 점검, Ollama 서버·모델)이 실패하면 사유를 출력하고 종료 코드 2 로 멈춘다.
- 결과: `report.md`(읽기용), `real_eval.json`, `per_question.csv`(질문 id·순위·gate 값만, 본문 없음), `e2e_samples.jsonl`(--e2e 사용 시, **답변 본문 포함**).
- 읽는 법
  - **hit@k / MRR 의 95% 구간**: 구간이 기준선에 걸치면 질문 수가 부족한 것이다.
  - **추천 임계값**: 오차단율 ≤ 10% 안에서 정거절률이 가장 높은 `fallback_min_score`. `없음` 이면 점수만으로 근거 유무를 못 가르는 상태 → 검색 방식·청킹·질문을 바꿔야 한다. 검색 방식마다 척도가 다르므로(코사인 vs BM25) **쓸 방식의 값**을 `config/core.json` 에 옮긴다.
  - **AUC**: 0.5=무작위, 1.0=완전 분리. 0.8 미만이면 fallback 이 불안정하다.
  - 종료 코드: 0=PASS, 1=FAIL, 2=입력/환경 오류, 3=INVALID(해시 임베더 — 성능 증거 아님).
- 종단 실행의 "거절 비율" 은 fallback 동작만 본다. **답변이 맞는지는 `e2e_samples.jsonl` 을 사람이 읽어 판단**한다.
- 임계값을 바꾸면 같은 질문셋으로 다시 돌려 오차단율/정거절률을 재확인한다(질문셋에 맞춰 튜닝한 값은 낙관적이다 → 새 질문 몇 개로 한 번 더 검증).

## 4. 팀원/멘토에게 공유할 것
보고서(`report.md`, `real_eval.json`, `per_question.csv`, `real_docs_check/report.*`)만 공유한다. 문서 원본, `*_dump`, `e2e_samples.jsonl`, `qa_draft.jsonl` 에는 문서 내용이 들어 있으므로 공유하지 않는다.

## 5. 이 도구로도 알 수 없는 것
- 질문 수가 적으면(<30) 수치가 우연일 수 있다. 판정에 경고가 붙는다.
- 한 번의 측정이며 문서가 바뀌면 다시 재야 한다. 응답 정확도·환각률은 사람 검토가 필요하다.
- 색인 후 삭제된 파일의 제거, `.xls`/RTF/OCR 은 아직 지원하지 않는다.
