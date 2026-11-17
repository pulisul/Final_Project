"""Streamlit 앱(app_streamlit.py)의 '폴더에서 문서 찾기 → 색인 → 질문' 흐름을 AppTest 로 끝까지 돌린다.
모델·Ollama 없이: 기본 설정(해시 임베더·메모리 저장소) + 가짜 생성기. 의미 검색 품질은 여기서 보증하지 않는다."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    p = tmp_path / "proj"; p.mkdir()
    shutil.copy(REPO / "app_streamlit.py", p / "app_streamlit.py")
    shutil.copytree(REPO / "config", p / "config")
    (p / "data" / "sample_docs").mkdir(parents=True)
    (p / "data" / "sample_docs" / "base.md").write_text("# 기본\n기본 문서 입니다. 사과는 과일입니다.", encoding="utf-8")
    if (REPO / "artifacts").exists():
        shutil.copytree(REPO / "artifacts", p / "artifacts")
    docs = tmp_path / "mydocs"; docs.mkdir()
    (docs / "backup.md").write_text("백업은 매일 새벽 2시에 전체 백업을 수행합니다.\n보관 기간은 30일입니다.", encoding="utf-8")
    (docs / "leave.txt").write_text("연차는 입사 1년 후부터 15일이 부여됩니다.", encoding="utf-8")
    monkeypatch.chdir(p)
    monkeypatch.syspath_prepend(str(REPO / "src"))
    import intellectagent.generation as g
    monkeypatch.setattr(g.OllamaGenerator, "generate", lambda self, q, hits: f"{hits[0].text[:30]} [{hits[0].chunk_id}]")
    return p, docs


def start(p):
    return AppTest.from_file(str(p / "app_streamlit.py"), default_timeout=120).run()


def click(at, label, *, sidebar=False):
    scope = at.sidebar if sidebar else at
    [b for b in scope.button if b.label.startswith(label)][0].click().run()


def ask(at, q):
    at.text_input(key="question_input").set_value(q)      # 폼: 값 입력과 제출을 같은 run 에서 해야 한다
    click(at, "질문")


def test_scan_register_index_restart_reindex_and_ask(proj):
    p, docs = proj
    at = start(p); assert not at.exception
    at.sidebar.text_input(key="folder_input").set_value(str(docs)).run()
    click(at, "찾아보기", sidebar=True)
    assert not at.exception and not at.sidebar.error
    assert [(m.label, m.value) for m in at.sidebar.metric][0] == ("색인 대상", "2개")
    assert not (p / "index" / "ui_roots.json").exists()                  # 훑어보기만으로는 등록·색인되지 않는다
    click(at, "이 폴더 등록", sidebar=True)
    assert not at.exception and not at.sidebar.error
    assert any("폴더 문서 2개" in s.value for s in at.sidebar.success)
    assert (p / "index" / "ui_roots.json").exists()

    ask(at, "백업은 매일 새벽 2시에 전체 백업을 수행합니다")           # 같은 세션: 바로 질문 가능
    shown = [m.value for c in at.chat_message for m in c.markdown]
    assert any("백업은 매일" in s for s in shown) and not at.exception
    assert any("backup.md" in e.label for e in at.expander)               # 출처에 파일 이름이 보인다

    at2 = start(p)                                                        # 앱 재시작: 등록 폴더는 파일에 남고, 메모리 색인은 자동으로 다시 채워진다
    assert any("등록된 폴더 1개" in e.label for e in at2.sidebar.expander)
    assert not any("색인이 비어 있습니다" in w.value for w in at2.warning)
    ask(at2, "백업은 매일 새벽 2시에 전체 백업을 수행합니다")
    assert any("backup.md" in e.label for e in at2.expander) and not at2.exception


def test_dangerous_folder_is_rejected_in_ui_and_nothing_is_saved(proj):
    p, _ = proj
    at = start(p)
    at.sidebar.text_input(key="folder_input").set_value("/etc").run()
    click(at, "찾아보기", sidebar=True)
    assert any("시스템 폴더" in e.value or "루트" in e.value for e in at.sidebar.error)
    assert not (p / "index" / "ui_roots.json").exists() and not at.exception


def test_sidebar_shows_open_source_notice(proj):
    p, _ = proj
    shutil.copy(REPO / "THIRD_PARTY_NOTICES.md", p / "THIRD_PARTY_NOTICES.md")
    at = start(p)
    assert not at.exception
    labels = [e.label for e in at.sidebar.expander]
    assert "오픈소스 고지" in labels
    exp = [e for e in at.sidebar.expander if e.label == "오픈소스 고지"][0]
    assert any("AutoRAG" in m.value for m in exp.markdown)
