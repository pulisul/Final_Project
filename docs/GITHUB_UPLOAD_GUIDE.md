# GitHub 업로드 가이드

이 폴더 자체가 GitHub 저장소의 루트가 되도록 업로드합니다. 과거 PART ZIP, Strict ZIP, delta ZIP을 저장소 루트에 다시 넣지 않습니다. Git은 commit/history가 delta 이력의 역할을 합니다.

## 1. 저장소 초기화

```bash
cd IntellectAgent
git init
git branch -M main
```

## 2. 업로드 전에 반드시 확인

```bash
git status --short
find data/real_docs -type f
find . -type f -name '*.sqlite3'
find . -type d -name '__pycache__'
```

`data/real_docs`에는 `.gitkeep` 외에 개인 문서가 없어야 합니다.

## 3. 첫 커밋

```bash
git add .
git status
git commit -m "Initial IntellectAgent final project"
```

## 4. GitHub 원격 저장소 연결

```bash
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## 5. 과거 ZIP을 보관하고 싶다면

main branch에 넣기보다 GitHub **Releases**의 asset으로 보관하는 것을 권장합니다.

- Weekly PART ZIPs
- Strict stage ZIPs
- MAC_FULL / FIXED / THREE_TIER ZIPs
- delta ZIPs

## 주의

- `.env`, API key, 토큰을 commit하지 않습니다.
- `data/real_docs`의 개인/교육 문서를 commit하지 않습니다.
- 로컬 Chroma/SQLite DB를 commit하지 않습니다.
- `__pycache__`, `.venv-mac`, 모델 캐시는 commit하지 않습니다.
