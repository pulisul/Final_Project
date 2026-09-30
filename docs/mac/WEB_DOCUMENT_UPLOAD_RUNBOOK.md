# IntellectAgent Mac 웹 문서 업로드 실행 가이드

## 목적
Streamlit 웹 화면에서 PDF/TXT/MD 문서를 직접 선택하고 `data/real_docs`에 안전하게 저장한 뒤 BGE 임베딩과 Chroma 인덱싱까지 한 번에 수행합니다.

## 처음 실행
```bash
cd /path/to/IntellectAgent_Weekly_Build_v3
source .venv-mac/bin/activate
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export INTELLECTAGENT_CONFIG="$PWD/config/core.production.json"
export RUNTIME_PROFILE=mac-native
export OLLAMA_MODEL=qwen2.5:14b
python -m streamlit run app_streamlit.py
```

브라우저가 자동으로 열리지 않으면 `http://localhost:8501`을 엽니다.

## 웹 사용 순서
1. 왼쪽 사이드바의 **문서 관리**를 엽니다.
2. **RAG에 사용할 문서를 선택하세요**에서 PDF/TXT/MD를 선택합니다.
3. **문서 업로드 + 인덱싱**을 누릅니다.
4. 저장 → BGE 임베딩 → Chroma 인덱싱이 완료될 때까지 진행 표시를 확인합니다.
5. `업로드 N개 · 인덱싱 완료 N개`가 표시되면 질문을 입력합니다.

## 안전장치
- 경로가 포함된 업로드 파일명 거부
- 허용 확장자 외 파일 거부
- 파일 크기 제한 적용
- 빈 파일 거부
- 임시 파일에 먼저 기록 후 `os.replace()`로 원자적 저장
- 같은 파일은 중복 저장하지 않음
- 같은 이름/다른 내용은 SHA-256 일부를 파일명에 붙여 기존 원본 보존
- 저장 후 SafeDocumentLoader로 재검증
- 텍스트를 추출할 수 없는 PDF는 저장 취소
- 인덱싱 후 Streamlit resource cache를 비우고 새 checkpoint fingerprint로 재로딩

## 현재 지원 형식
`config/core.production.json` 기준:
- `.pdf`
- `.txt`
- `.md`

스캔 이미지형 PDF는 OCR이 P0 범위 밖이므로 텍스트 추출이 안 되면 거부됩니다.

## Strict 검증과 웹 실행의 차이
`bash scripts/mac/02_run_mac_strict.sh`는 검증용이며 웹을 계속 열어두지 않습니다.
웹을 직접 사용할 때는 반드시:

```bash
python -m streamlit run app_streamlit.py
```

을 별도로 실행합니다.
