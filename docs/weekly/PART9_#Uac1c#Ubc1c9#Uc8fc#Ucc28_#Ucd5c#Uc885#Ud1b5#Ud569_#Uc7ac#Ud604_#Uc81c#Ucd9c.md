# PART 9 — 개발 9주차: 최종 통합·재현·제출

## 10/12 월요일 — 깨끗한 Python 3.11 재현
1. 기존 `.venv`를 재사용하지 않는다.
2. `PYTHON_BIN=python3.11 bash scripts/week9/lock_and_reproduce_py311.sh`를 실행한다.
3. PyTorch는 CPU/CUDA 환경에 맞는 공식 설치를 먼저 수행할 수 있다.
4. 설치가 성공한 환경의 `pip freeze | sort`를 `requirements-lock.txt`로 보관한다.
5. `pip check`, 전체 pytest, Strict Gate가 모두 통과해야 한다.

### 힘든 점
본인 PC에 우연히 설치된 패키지를 코드가 암묵적으로 사용하고 있을 수 있다.

### 극복
새 가상환경에서 실패하면 README를 억지로 수정하기 전에 누락된 **직접 의존성**을 먼저 찾고, smoke+회귀테스트가 통과한 뒤 lock을 갱신한다.

## 10/13 — 구조도·ERD·RL 흐름도
`docs/final/ARCHITECTURE_FINAL.md`, `ERD_FINAL.md`, `RL_FLOW_FINAL.md`를 실제 코드와 대조한다. 문서가 코드를 앞서가거나 존재하지 않는 기능을 완료로 표시하면 안 된다.

## 10/14 — 평가 보고서 + 전체 테스트
`python scripts/week9/run_final_regression.py` 실행. 실제 데이터/BGE/Ollama/RAGAS Strict 결과를 `EVALUATION_REPORT.md`에 반영한다.

## 10/15 — 데이터 분석 보고서
실제 데이터의 문서 수, 용량, 형식, 청크 분포, 실패율을 채운다. 샘플 데이터를 실제 프로젝트 통계로 오인하지 않는다.

## 10/16 — 시연영상·발표자료
`DEMO_SCRIPT.md` 순서로 입력→검색→생성→출처→Fallback→정책→피드백을 촬영한다.

## 10/17 — 최종 제출
`python scripts/week9/build_release_manifest.py`로 SHA-256 manifest를 만들고 `python scripts/week9/run_week9_gate.py --strict`가 PASS인지 확인한다. 제출 완료 화면/메일을 증빙으로 저장한다.

## 10/18 — 휴식
비상 상황이 아니면 코드/문서를 변경하지 않는다.

## 최종 Gate
- 깨끗한 Python 3.11 설치 성공
- requirements-lock.txt 존재
- `pip check` PASS
- 전체 pytest 0 fail
- 각 주차 Gate evidence 존재
- Core Strict + Week4/6/8 Strict PASS
- README 재현 성공
- 시연 전체 흐름 확인
- 제출 문서/영상/소스/Manifest/증빙 보관
