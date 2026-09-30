# PART 6 — 개발 6주차: 정량·정성 평가

## 일정 기준
- 09/21: 평가질문 20개 확정 + RAGAS 형식 변환
- 09/22: 모호형·폴백형 점검 + 휴먼 평가표 + RAGAS 2문항 smoke
- 09/23: 로컬 평가환경, 실행 ID·seed, 외부 API 미호출 확인
- 09/24: 기본 검색 평가
- 09/25: 리랭킹 평가 + 원본 로그
- 09/26: 비교표, 폴백 정확도, 평균/P95 지연시간

## 이번 주의 핵심 원칙
1. 기본검색과 리랭킹은 **동일한 20문항·동일 seed**를 사용합니다.
2. 지표를 검색 / 생성 / 폴백 / 지연시간으로 분리합니다.
3. RAGAS는 P0 평가를 대체하지 않습니다. API 변경이나 패키지 문제가 있어도 deterministic 평가와 Release Gate는 살아 있어야 합니다.
4. `LocalOnlyNetworkGuard`는 현재 Python 프로세스의 외부 연결을 차단·기록합니다. 이는 OS 방화벽/패킷캡처와 동등한 증거는 아니므로 최종 Strict 환경에서는 별도 네트워크 관찰도 권장합니다.
5. `eval20.jsonl`은 현재 샘플 문서용 smoke 데이터입니다. 실제 제출 전 실문서 기반 20문항으로 교체해야 합니다.

## 준비 라이브러리
P0 deterministic 평가는 Python 표준 라이브러리 + 기존 Core만으로 동작합니다. RAGAS Strict 평가에는 `ragas>=0.4,<0.5`와 로컬 평가 LLM/embedding adapter가 필요합니다. RAGAS 0.4 계열은 컬렉션 기반 API로 이동 중이므로 2문항 smoke에서 실제 설치 버전의 import/API를 먼저 고정합니다.

## 실행
```bash
cd IntellectAgent_Weekly_Build_v3
export PYTHONPATH="$PWD/src"
bash scripts/week6/run_week6_all.sh
```

## 산출물
- `data/eval/week6/eval20.jsonl`
- `evidence/week6/baseline_summary.json`
- `evidence/week6/baseline_raw.jsonl`
- `evidence/week6/reranked_summary.json`
- `evidence/week6/reranked_raw.jsonl`
- `evidence/week6/evaluation_comparison.json`
- `evidence/week6/human_eval.csv`
- `evidence/week6/network_guard.json`
- `evidence/week6/ragas_import_smoke.json`
- `evidence/week6/WEEK6_GATE.json`

## Week 6 Gate
- 전체 pytest PASS
- 20문항 존재
- baseline/reranked 동일 dataset_version·seed
- 검색/생성/폴백/속도 지표 분리
- 두 조건의 raw log 보관
- Human eval template 존재
- 외부 연결 차단 smoke 증거 존재
- RAGAS는 Smoke Gate에서는 optional, 실제 제출 Strict에서는 별도 PASS 필요

## 실제 RAGAS 0.4 로컬 Ollama smoke
현재 RAGAS 공식 stable 문서의 Collections API를 기준으로 `Faithfulness`와 `ContextRecall`을 사용하며, Ollama는 localhost의 OpenAI-compatible endpoint로 연결합니다.
```bash
pip install openai
python scripts/week6/run_ragas_local_ollama.py --model qwen2.5:14b --limit 2
```
이 단계는 RAGAS/Ollama가 실제 설치된 Python 3.11 환경에서 수행합니다. `LocalOnlyNetworkGuard`가 비-local 연결을 차단하지만, 최종 보안 증거는 OS 방화벽/패킷 관찰과 함께 보강하는 것이 좋습니다.
