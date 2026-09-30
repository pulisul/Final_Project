from pathlib import Path
import json
from intellectagent.repro import build_manifest

def test_manifest_contains_config_and_file_hashes(tmp_path):
    (tmp_path/"requirements.txt").write_text("x==1\n")
    (tmp_path/"pyproject.toml").write_text("[x]\n")
    c=tmp_path/"config.json"; c.write_text(json.dumps({"a":1}))
    m=build_manifest(tmp_path,c)
    assert m["config_hash"] and "requirements.txt" in m["files"]
