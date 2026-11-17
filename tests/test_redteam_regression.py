"""레드팀 하네스(scripts/redteam/redteam.py)를 그대로 돌려, 막기로 한 공격이 다시 뚫리지 않는지 확인한다.
고치지 않고 남긴 한계(G6·G13·G14·P2)는 '여전히 뚫림'으로 고정한다 — 개선하면 이 목록과 docs/REDTEAM_REPORT.md 를 함께 고칠 것."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MUST_DEFEND = ["G1", "G2", "G3", "G4", "G5", "G7", "G8", "G9", "G10", "G11", "G12", "G15", "G16", "G6b",
               "R1", "R2", "R3", "R4", "R5", "R6", "R7", "I1", "I2", "I3", "I4", "I5", "I6", "I7",   # S 계열은 OS별 경로라 일부만 고정
               "I8-docx", "I8-xlsx", "I8-pptx", "I8-hwpx", "I8-hwp", "S1", "S3", "S9"]
KNOWN_OPEN = ["G6", "G13", "G14", "P2"]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("redteam")
    r = subprocess.run([sys.executable, "-I", str(REPO / "scripts/redteam/redteam.py"), str(REPO), str(out)],
                       capture_output=True, text=True, timeout=240, cwd=REPO)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    return {x["sid"]: x for x in json.loads((out / "results.json").read_text(encoding="utf-8"))}


def test_harness_ran_every_scenario_without_internal_errors(results):
    assert len(results) >= 50
    assert not [s for s, r in results.items() if r["category"] == "하네스"]


@pytest.mark.parametrize("sid", MUST_DEFEND)
def test_attack_stays_defended(results, sid):
    if sid not in results:
        pytest.skip(f"{sid} 시나리오 없음(선택 의존성 부재)")
    assert results[sid]["verdict"] == "DEFENDED", results[sid]["observed"]


@pytest.mark.parametrize("sid", KNOWN_OPEN)
def test_known_open_limits_are_documented_as_open(results, sid):
    assert results[sid]["verdict"] == "BREACHED"
