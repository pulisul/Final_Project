# STEP 14~15 — 최종 통합 Strict / Release Promotion Runbook

## 목적
STEP 1~13에서 개별 검증한 구성요소를 **하나의 동일한 소스·설정·의존성 기준선**에서 다시 실행하고, 모든 Strict 증거가 통과한 경우에만 최종 릴리스를 승격한다.

## 승격 전 필수조건
- Python 3.11.x
- `requirements-lock.txt` 존재 + `pip check` PASS
- GPU/CUDA Runtime PASS
- BGE-M3 PASS
- Chroma persistence PASS
- BGE reranker PASS
- 실제 문서 기반 candidate benchmark PASS
- Ollama grounded generation PASS
- LangGraph 정상/폴백 양방향 PASS
- RAGAS 20문항 PASS
- Streamlit runtime PASS
- Core Strict Gate PASS
- 전체 pytest 0 fail
- 실제 데이터 사용 (`data/sample_docs`는 승격 금지)

## STEP 14 실행
```bash
source .venv-strict/bin/activate
export PYTHONPATH="$PWD/src"

python scripts/strict/27_step14_integrated_release_candidate.py \
  --model qwen2.5:14b \
  --root data/real_docs \
  --eval data/eval/week6/eval20.jsonl
```

### STEP 14이 재검증하는 순서
1. Preflight
2. STEP 6~9 전체 (GPU/BGE/Chroma/reranker/candidate benchmark)
3. STEP 10~13 전체 (`--full-ragas` 포함)
4. Core Strict Gate
5. 전체 pytest
6. 실행 전/후 핵심 파일 SHA-256 비교
7. 활성 정책 유효성 확인

### 기준선 오염 방지
실행 전/후 다음을 포함한 핵심 파일 SHA를 비교한다.
- `config/*.json`
- requirements 파일
- `requirements-lock.txt`
- `artifacts/week7/active_policy.json`
- `src/intellectagent/*.py`

Strict 실행 중 이 파일이 하나라도 바뀌면 `blocked_by=baseline_changed_during_strict`로 승격을 차단한다.

## 실제 데이터 규칙
최종 승격에서는 `data/sample_docs` 사용을 금지한다. 실험용으로만 확인하려면 `--allow-sample`을 사용할 수 있지만 STEP 15는 sample-data 증거를 절대로 PROMOTED로 만들지 않는다.

## STEP 15 실행
STEP 14가 `overall_pass=true`일 때만:
```bash
python scripts/strict/28_step15_promote_final_release.py
```

생성물:
- `artifacts/strict/FINAL_RELEASE_PROMOTION.json`
- `evidence/strict/STEP15_PROMOTION_VERIFY.json`

`FINAL_RELEASE_PROMOTION.json`에는 다음을 포함한다.
- 고유 `release_id`
- `requirements-lock.txt` SHA-256
- STEP 14 evidence SHA-256
- Strict evidence 파일별 SHA-256
- 현재 active policy
- candidate benchmark / RAGAS / GPU 관련 핵심 지표
- rollback target

## 한 번에 실행
```bash
OLLAMA_MODEL=qwen2.5:14b \
DATA_ROOT=data/real_docs \
EVAL_FILE=data/eval/week6/eval20.jsonl \
bash scripts/strict/29_step14_15_all.sh
```

## 실패 시 원칙
- `blocked_by` 첫 원인만 수정한다.
- Python/torch/CUDA/lock/source/config가 바뀌었으면 STEP 14를 처음부터 다시 실행한다.
- 모델 또는 서비스만 재시작했더라도 최종 승격 직전에는 STEP 14 전체 재검증을 권장한다.
- `FINAL_RELEASE_PROMOTION.json`을 사람이 손으로 만들어서는 안 된다.
