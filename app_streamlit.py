from __future__ import annotations

import os
import uuid
from pathlib import Path

try:
    import streamlit as st
except ImportError as e:  # clearer error than a stack trace during strict smoke
    raise SystemExit("streamlit is not installed. Install requirements.txt first.") from e

from intellectagent.config import CoreConfig
from intellectagent.corpus_scan import RootRejected
from intellectagent.document_ingest import (
    UploadValidationError,
    index_runtime_documents,
    list_ingestible_documents,
    save_upload_bytes,
)
from intellectagent.factory import build_components
from intellectagent.folder_ui import (
    add_root,
    apply_roots,
    check_root,
    load_roots,
    registered_documents,
    scan_folder,
    suggest_folders,
)
from intellectagent.generation import OllamaGenerator
from intellectagent.answer_fallback_chain import EmptyRAGFallbackChain
from intellectagent.ollama_knowledge import OllamaKnowledgeEngine
from intellectagent.web_fallback import DDGSWebSearchProvider, WebFallbackEngine, settings_from_feature_flags
from intellectagent.ui_support import (
    FeedbackStore,
    answer_to_ui_payload,
    build_cache_fingerprint,
    inspect_runtime_health,
    load_policy_info,
    stable_json_hash,
)

ROOT = Path(__file__).resolve().parent
CONFIG = (ROOT / os.environ.get("INTELLECTAGENT_CONFIG", "config/core.json")).resolve()
POLICY = ROOT / "artifacts" / "week7" / "active_policy.json"
FEEDBACK_DB = ROOT / "index" / "feedback.sqlite3"

st.set_page_config(page_title="IntellectAgent AI", page_icon="📚", layout="wide")
st.title("IntellectAgent AI")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    if st.session_state.last_payload:
        st.session_state.chat_history.append(st.session_state.last_payload)

try:
    cfg_preview = CoreConfig.load(CONFIG)
except Exception as exc:
    st.error(f"설정 파일을 읽을 수 없습니다: {CONFIG}")
    st.exception(exc)
    st.stop()

# 화면에서 등록한 폴더(index/ui_roots.json)를 설정에 합친다. 저장된 폴더는 실행 때마다 다시 검증된다.
ui_roots, ui_root_notes = load_roots(cfg_preview)
cfg_preview = apply_roots(cfg_preview, ui_roots)

INDEX_MARKER = cfg_preview.evidence_dir / "index_checkpoint.json"
policy = load_policy_info(POLICY)
cache_key = stable_json_hash([
    build_cache_fingerprint(config_path=CONFIG, policy_path=POLICY, index_marker=INDEX_MARKER),
    [str(r) for r in ui_roots],
])[:20]


@st.cache_resource
def load_runtime(_cache_key: str):
    cfg = CoreConfig.load(CONFIG)
    cfg = apply_roots(cfg, load_roots(cfg)[0])
    return cfg, build_components(cfg, runtime_profile=os.environ.get("RUNTIME_PROFILE"))


@st.cache_data(ttl=300, show_spinner=False)
def cached_documents(key: str) -> list[str]:
    """색인 대상 파일 목록(문자열). 폴더가 크면 매 클릭마다 훑지 않도록 캐시하고, 색인·등록 때 비운다."""
    return [str(p) for p in (registered_documents(cfg_preview) if ui_roots else list_ingestible_documents(cfg_preview.allowed_root, cfg_preview.allowed_extensions))]


def _pick_folder(path: str) -> None:
    st.session_state["folder_input"] = path
    st.session_state["folder_scan"] = None


def _shown(path: str) -> str:
    p = Path(path)
    for base in (cfg_preview.allowed_root, *ui_roots):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return path


with st.sidebar:
    st.subheader("📂 폴더에서 문서 찾기")
    st.caption("내 컴퓨터의 폴더를 지정하면 하위 폴더까지 훑어 문서를 찾습니다. 먼저 '찾아보기'로 개수만 확인하고, 맞으면 등록·색인합니다.")
    for note in ui_root_notes:
        st.warning(note)
    if st.button("문서가 많은 폴더 추천", use_container_width=True):
        st.session_state["folder_candidates"] = suggest_folders(cfg_preview)
    for cand in st.session_state.get("folder_candidates") or []:
        st.button(f"{cand['path']} · {cand['supported_files']}개{'+' if cand['truncated'] else ''}", key=f"cand_{cand['path']}",
                  on_click=_pick_folder, args=(cand["path"],), use_container_width=True)
    if st.session_state.get("folder_candidates") == []:
        st.caption("추천할 폴더가 없습니다. 경로를 직접 입력하세요.")
    folder_text = st.text_input("폴더 경로", key="folder_input", placeholder="예: ~/Documents/규정집")
    if st.button("찾아보기 (색인 안 함)", disabled=not folder_text.strip(), use_container_width=True):
        try:
            with st.spinner("폴더를 훑는 중..."):
                st.session_state["folder_scan"] = scan_folder(cfg_preview, check_root(cfg_preview, folder_text))
        except RootRejected as exc:
            st.session_state["folder_scan"] = None
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - 권한 오류 등은 화면에 그대로 보여 준다
            st.session_state["folder_scan"] = None
            st.error(f"폴더를 읽지 못했습니다: {type(exc).__name__}: {exc}")
    scan = st.session_state.get("folder_scan")
    if scan is not None:
        st.markdown(f"**{scan.root}**")
        c1, c2 = st.columns(2)
        c1.metric("색인 대상", f"{scan.unique_files}개")
        c2.metric("용량", f"{scan.total_mb} MB")
        if scan.by_ext:
            st.caption("형식별: " + ", ".join(f"{k} {v}" for k, v in scan.by_ext.items()))
        if scan.duplicates:
            st.caption(f"내용이 같은 파일 {scan.duplicates}개는 한 번만 색인합니다.")
        if scan.truncated:
            st.warning("파일이 너무 많아 일부만 훑었습니다. 더 좁은 폴더를 지정하세요.")
        if scan.skipped:
            with st.expander("제외된 파일 사유"):
                for reason, n in scan.skipped.items():
                    st.write(f"• {reason}: {n}개")
                    for ex in scan.examples.get(reason, [])[:2]:
                        st.caption(ex)
        if scan.samples:
            with st.expander("색인 대상 예시"):
                for s in scan.samples:
                    st.write("•", s)
        if scan.unique_files == 0:
            st.info("색인할 문서가 없습니다. 지원 형식: " + ", ".join(cfg_preview.allowed_extensions))
        elif st.button("이 폴더 등록 + 색인", type="primary", use_container_width=True):
            try:
                add_root(cfg_preview, str(scan.root))
                new_cfg = apply_roots(CoreConfig.load(CONFIG), load_roots(cfg_preview)[0])
                _cfg0, comps = load_runtime(cache_key)
                with st.spinner(f"{scan.unique_files}개 문서를 색인하는 중... (처음에는 오래 걸릴 수 있습니다)"):
                    state = index_runtime_documents(cfg=new_cfg, embedder=comps[0], store=comps[1])
                failed_index = state.get("failed", {})
                st.session_state["ingest_notice"] = {
                    "kind": "folder", "saved": scan.unique_files, "duplicates": scan.duplicates,
                    "indexed": len(state.get("completed", {})), "index_failed": len(failed_index),
                    "failures": [f"{Path(k).name}: {v}" for k, v in failed_index.items()],
                }
                st.session_state["folder_scan"] = None
                load_runtime.clear()
                cached_documents.clear()
                st.session_state.last_payload = None
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error("등록 또는 색인에 실패했습니다.")
                st.exception(exc)
    if ui_roots:
        with st.expander(f"등록된 폴더 {len(ui_roots)}개"):
            for r in ui_roots:
                st.write("•", str(r))
            st.caption("등록 해제는 아직 지원하지 않습니다. 이미 색인된 내용을 지우는 기능이 없어, 해제해도 옛 조각이 검색될 수 있기 때문입니다.")

    st.divider()
    st.subheader("📁 문서 관리")
    st.caption(f"저장 위치: {cfg_preview.allowed_root.relative_to(ROOT) if ROOT in cfg_preview.allowed_root.parents else cfg_preview.allowed_root}")
    uploaded_files = st.file_uploader(
        "RAG에 사용할 문서를 선택하세요",
        type=[ext.lstrip(".") for ext in cfg_preview.allowed_extensions],
        accept_multiple_files=True,
        help=f"허용 형식: {', '.join(cfg_preview.allowed_extensions)} · 파일당 최대 {cfg_preview.max_file_bytes // (1024 * 1024)} MB",
    )

    if st.button("문서 업로드 + 인덱싱", type="primary", disabled=not uploaded_files, use_container_width=True):
        saved = []
        failures = []
        progress = st.progress(0, text="문서를 확인하고 있습니다...")
        for i, item in enumerate(uploaded_files, start=1):
            try:
                result = save_upload_bytes(
                    allowed_root=cfg_preview.allowed_root,
                    filename=item.name,
                    data=item.getvalue(),
                    allowed_extensions=cfg_preview.allowed_extensions,
                    max_file_bytes=cfg_preview.max_file_bytes,
                )
                saved.append(result)
            except Exception as exc:
                failures.append(f"{item.name}: {type(exc).__name__}: {exc}")
            progress.progress(int(i / max(len(uploaded_files), 1) * 45), text=f"파일 저장 {i}/{len(uploaded_files)}")

        if saved:
            try:
                cfg, components = load_runtime(cache_key)
                embedder, store, _retriever, _generator, _rag = components
                progress.progress(55, text="BGE 임베딩 및 Chroma 인덱싱 중...")
                state = index_runtime_documents(cfg=cfg, embedder=embedder, store=store)
                failed_index = state.get("failed", {})
                progress.progress(100, text="인덱싱 완료")
                st.session_state["ingest_notice"] = {
                    "saved": len(saved),
                    "duplicates": sum(1 for x in saved if x.duplicate),
                    "indexed": len(state.get("completed", {})),
                    "index_failed": len(failed_index),
                    "failures": failures + [f"{Path(k).name}: {v}" for k, v in failed_index.items()],
                }
                # Rebuild BGE/Chroma/RAG resources using the new checkpoint fingerprint.
                load_runtime.clear()
                st.session_state.last_payload = None
                st.rerun()
            except Exception as exc:
                progress.empty()
                st.error("파일은 저장했지만 인덱싱에 실패했습니다.")
                st.exception(exc)
        elif failures:
            progress.empty()
            st.error("업로드 가능한 문서가 없습니다.")
            for msg in failures:
                st.caption(msg)

    notice = st.session_state.pop("ingest_notice", None)
    if notice:
        what = "폴더 문서" if notice.get("kind") == "folder" else "업로드"
        if notice["index_failed"] == 0 and not notice["failures"]:
            st.success(f"{what} {notice['saved']}개 · 전체 인덱싱 완료 {notice['indexed']}개")
        else:
            st.warning(f"{what} {notice['saved']}개 · 전체 인덱싱 완료 {notice['indexed']}개 · 확인 필요 {len(notice['failures'])}개")
            for msg in notice["failures"][:10]:
                st.caption(msg)

    docs = cached_documents(cache_key)
    st.caption(f"현재 등록 문서: {len(docs)}개")
    if docs:
        with st.expander("등록 문서 보기"):
            for p in docs[:30]:
                st.write("•", _shown(p))
            if len(docs) > 30:
                st.caption(f"외 {len(docs) - 30}개")

    st.divider()
    st.subheader("런타임 상태")
    for health in inspect_runtime_health(ROOT):
        st.write(("✅" if health.ok else "❌"), health.component, health.detail)
    st.caption(f"Config: {CONFIG.name}")
    st.caption(f"Cache fingerprint: {cache_key}")
    st.caption(f"Policy: {policy.name} / {policy.version}")
    if policy.rollback_applied:
        st.warning(f"RL 정책 롤백 적용: {policy.reason}")

with st.sidebar:
    with st.expander("오픈소스 고지"):
        _notice = Path(__file__).resolve().parent / "THIRD_PARTY_NOTICES.md"
        try:
            st.markdown(_notice.read_text(encoding="utf-8"))
        except OSError:
            st.caption("THIRD_PARTY_NOTICES.md 를 찾지 못했습니다. 저장소 루트의 같은 이름 파일을 참고하세요.")

docs_now = cached_documents(cache_key)
web_settings = settings_from_feature_flags(cfg_preview.feature_flags)
rag = None
empty_rag_chain = None

if docs_now:
    try:
        cfg, components = load_runtime(cache_key)
        _embedder, _store, _retriever, _generator, rag = components
    except Exception as exc:
        st.error("모델 또는 DB 로딩에 실패했습니다. Core 설정과 인덱스 상태를 확인하세요.")
        st.exception(exc)
        st.stop()
    if hasattr(_store, "rows") and not _store.rows:
        # 메모리 저장소(use_chroma=false)는 앱을 켤 때마다·런타임을 다시 만들 때마다 비어 있다. 문서가 있으면 한 번 자동으로 다시 색인한다.
        if st.session_state.get("auto_reindexed") != cache_key:
            st.session_state["auto_reindexed"] = cache_key
            with st.spinner("메모리 저장소를 다시 채우는 중..."):
                try:
                    index_runtime_documents(cfg=cfg, embedder=_embedder, store=_store)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"자동 재색인에 실패했습니다: {type(exc).__name__}: {exc}")
        # 저장소를 제자리에서 채웠으므로 다시 그릴 필요가 없다(rerun 하면 방금 띄운 완료 알림이 사라진다).
    if hasattr(_store, "rows") and not _store.rows:
        st.warning("색인이 비어 있습니다. 현재 설정은 메모리 저장소라 앱을 다시 켜면 사라집니다. "
                   "디스크에 저장하려면 운영 설정(config/core.production.json, Chroma)을 쓰세요.")
        if st.button("지금 다시 색인", type="primary"):
            with st.spinner("문서를 색인하는 중..."):
                index_runtime_documents(cfg=cfg, embedder=_embedder, store=_store)
            st.rerun()
else:
    st.info(
        "RAG 문서가 없습니다. 답변 순서는 ① 인터넷 검색 근거 → "
        "② 인터넷을 사용할 수 없으면 Ollama 내부지식입니다."
    )
    web_engine = None
    if web_settings.enabled:
        try:
            provider = DDGSWebSearchProvider(
                region=web_settings.region,
                safesearch=web_settings.safesearch,
                timeout_seconds=web_settings.timeout_seconds,
            )
            web_engine = WebFallbackEngine(
                provider,
                OllamaGenerator(cfg_preview.ollama_model),
                max_results=web_settings.max_results,
            )
        except Exception as exc:
            st.warning(f"웹 검색 준비 실패: {type(exc).__name__}. Ollama 내부지식으로 대체합니다.")
    knowledge_engine = (
        OllamaKnowledgeEngine(cfg_preview.ollama_model)
        if cfg_preview.feature_flags.get("use_ollama_knowledge_fallback", True)
        else None
    )
    empty_rag_chain = EmptyRAGFallbackChain(web_engine, knowledge_engine)

def render_payload(payload: dict, message_index: int) -> None:
    """Render one completed turn without coupling source schemas to the UI."""
    with st.chat_message("user"):
        st.markdown(payload.get("question") or "")

    with st.chat_message("assistant"):
        if payload.get("fallback"):
            st.warning(payload.get("answer") or "")
        else:
            st.markdown(payload.get("answer") or "")

        st.markdown("#### 출처")
        sources = payload.get("sources") or []
        if not sources:
            st.info("표시할 검색 근거가 없습니다.")

        for i, src in enumerate(sources, start=1):
            source_type = str(src.get("source_type") or "rag")
            score = float(src.get("score") or 0.0)
            chunk_id = str(src.get("chunk_id") or "")
            source_text = str(src.get("text") or "")

            if source_type == "model_internal":
                model_name = str(src.get("model") or "").strip()
                title = str(src.get("title") or "").strip()
                source_label = title or (f"Ollama 내부지식 · {model_name}" if model_name else "Ollama 내부지식")
                with st.expander(f"Ollama 내부지식 {i} · {source_label}"):
                    st.warning(
                        "외부 검증 가능한 문서/URL 출처가 아닙니다. "
                        "Ollama 모델의 사전학습 내부지식 기반 답변입니다."
                    )
                    if model_name:
                        st.write(f"모델: {model_name}")
                    st.write("외부 검증: 안 됨")
                    if source_text:
                        st.write(source_text)
                continue

            source_label = (
                str(src.get("title") or "").strip()
                or str(src.get("source_name") or "").strip()
                or str(src.get("doc_id") or "").strip()
                or "출처 정보"
            )
            prefix = "웹" if source_type == "web" else "문서"
            with st.expander(f"{prefix} 출처 {i} · {source_label} · score={score:.4f}"):
                if chunk_id:
                    st.code(chunk_id)
                if src.get("url"):
                    st.link_button("원문 열기", src["url"])
                if source_text:
                    st.write(source_text)

        st.caption(
            f"활성 정책: {payload['policy']['name']} / {payload['policy']['version']}"
            if payload.get("policy")
            else ""
        )

        store = FeedbackStore(FEEDBACK_DB)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 도움이 됨", key=f"thumb_up_{message_index}"):
                inserted = store.submit(
                    session_id=st.session_state.session_id,
                    question=payload.get("question") or "",
                    response=payload.get("answer") or "",
                    thumb=1,
                    policy=policy,
                )
                st.success("피드백을 저장했습니다." if inserted else "이미 이 답변에 피드백을 제출했습니다.")
        with c2:
            if st.button("👎 개선 필요", key=f"thumb_down_{message_index}"):
                inserted = store.submit(
                    session_id=st.session_state.session_id,
                    question=payload.get("question") or "",
                    response=payload.get("answer") or "",
                    thumb=-1,
                    policy=policy,
                )
                st.info("피드백을 저장했습니다." if inserted else "이미 이 답변에 피드백을 제출했습니다.")


# ChatGPT-style flow: completed turns stay above; the next question box is always last.
for message_index, item in enumerate(st.session_state.chat_history):
    render_payload(item, message_index)

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "질문하세요",
        placeholder="예: RAG는 무엇인가요?",
        key="question_input",
    )
    submitted = st.form_submit_button(
        "질문",
        type="primary",
        use_container_width=True,
    )

if submitted and question.strip():
    if rag is not None:
        spinner_text = "문서 근거를 검색하고 있습니다..."
    elif empty_rag_chain is not None:
        spinner_text = "RAG 문서가 없어 인터넷을 먼저 검색하고, 필요하면 Ollama 내부지식을 사용합니다..."
    else:
        spinner_text = "질문을 처리하고 있습니다..."

    with st.spinner(spinner_text):
        try:
            if rag is not None:
                answer = rag.ask(question.strip())
            elif empty_rag_chain is not None:
                answer = empty_rag_chain.ask(question.strip())
            else:
                raise RuntimeError("사용 가능한 답변 경로가 없습니다.")

            payload = answer_to_ui_payload(answer, policy)
            st.session_state.last_payload = payload
            st.session_state.chat_history.append(payload)
            st.rerun()
        except Exception as exc:
            st.error("질문 처리 중 오류가 발생했습니다. 이전 기준선은 유지됩니다.")
            st.exception(exc)
