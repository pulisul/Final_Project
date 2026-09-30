# Strict STEP 6~9 Runbook — GPU / BGE-M3 / ChromaDB / BGE Reranker

이 문서는 STEP 1~5 Preflight PASS 이후에만 사용한다.

## STEP 6 — GPU / Driver / PyTorch / CUDA

```bash
source .venv-strict/bin/activate
export PYTHONPATH="$PWD/src"
python scripts/strict/14_step6_gpu_runtime.py
```

GPU 배포가 목표이면 `nvidia-smi`가 보이고 `torch.cuda.is_available()==True`여야 한다. CPU 테스트만 허용하려면 `--allow-cpu`를 명시한다. GPU 서버에서 `--allow-cpu`로 우회해 최종 배포 PASS로 간주하지 않는다.

추가 확인:
```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## STEP 7 — BGE-M3

```bash
python scripts/strict/15_step7_bge_m3_strict.py --model BAAI/bge-m3
```

5개 문장, 동일 차원, finite, non-zero를 검사하고 모델 load/encode 시간과 VRAM 전후를 저장한다.

증거: `evidence/strict/15_bge_m3_strict.json`

## STEP 8 — ChromaDB 영속성

```bash
python scripts/strict/16_step8_chroma_persistence.py
```

임시 PersistentClient에 upsert → query → client 재생성 → 동일 query → delete_doc 순으로 검사한다. 실제 운영 인덱스를 건드리지 않는다.

증거: `evidence/strict/16_chroma_persistence.json`

## STEP 9A — BGE Reranker

```bash
python scripts/strict/17_step9_bge_reranker_strict.py --model BAAI/bge-reranker-v2-m3
```

관련 문장이 무관 문장보다 높은 점수를 받아야 한다. 모델 load/inference 시간과 VRAM을 함께 기록한다.

## STEP 9B — candidate_k 5/10/20

Smoke:
```bash
python scripts/strict/18_step9_candidate_benchmark.py \
  --root data/sample_docs \
  --eval data/eval/week2/eval15.jsonl \
  --candidate-ks 5,10,20
```

실제 제출 성능은 반드시 `data/real_docs`와 실제 검증된 평가셋으로 다시 실행한다. `data/sample_docs` 결과는 파이프라인 검증일 뿐 성능 증거가 아니다.

선택 기준은 미리 고정한다.
1. Hit Rate@k 최대
2. MRR 최대
3. 평균 latency 최소
4. 동일하면 더 작은 candidate_k

## STEP 6~9 일괄 Gate

```bash
python scripts/strict/19_step6_9_gate.py \
  --root data/real_docs \
  --eval data/eval/week6/eval20.jsonl
```

첫 실패 단계에서 중단되며 `evidence/strict/STEP6_9_GATE.json`의 `blocked_by`를 먼저 해결한다.
