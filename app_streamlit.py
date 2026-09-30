from __future__ import annotations

import os
import uuid
from pathlib import Path

try:
    import streamlit as st
except ImportError as e:  # clearer error than a stack trace during strict smoke
    raise SystemExit("streamlit is not installed. Install requirements.txt first.") from e

from intellectagent.config import CoreConfig
from intellectagent.document_ingest import (
    UploadValidationError,
    index_runtime_documents,
    list_ingestible_documents,
    save_upload_bytes,
)
from intellectagent.factory import build_components
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

INDEX_MARKER = cfg_preview.evidence_dir / "index_checkpoint.json"
policy = load_policy_info(POLICY)
cache_key = build_cache_fingerprint(config_path=CONFIG, policy_path=POLICY, index_marker=INDEX_MARKER)


@st.cache_resource
def load_runtime(_cache_key: str):
    cfg = CoreConfig.load(CONFIG)
    return cfg, build_components(cfg, runtime_profile=os.environ.get("RUNTIME_PROFILE"))


with st.sidebar:
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
        if notice["index_failed"] == 0 and not notice["failures"]:
            st.success(f"업로드 {notice['saved']}개 · 인덱싱 완료 {notice['indexed']}개")
        else:
            st.warning(f"업로드 {notice['saved']}개 · 인덱싱 완료 {notice['indexed']}개 · 확인 필요 {len(notice['failures'])}개")
            for msg in notice["failures"][:10]:
                st.caption(msg)

    docs = list_ingestible_documents(cfg_preview.allowed_root, cfg_preview.allowed_extensions)
    st.caption(f"현재 등록 문서: {len(docs)}개")
    if docs:
        with st.expander("등록 문서 보기"):
            for p in docs[:30]:
                st.write("•", p.relative_to(cfg_preview.allowed_root))
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

docs_now = list_ingestible_documents(cfg_preview.allowed_root, cfg_preview.allowed_extensions)
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
