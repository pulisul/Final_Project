from pathlib import Path
import pytest
from intellectagent.loaders import SafeDocumentLoader,UnsafePathError,UnsupportedFileError

def test_outside_root_rejected(tmp_path):
    root=tmp_path/"root"; root.mkdir(); outside=tmp_path/"x.txt"; outside.write_text("x")
    loader=SafeDocumentLoader(root,(".txt",),100)
    with pytest.raises(UnsafePathError): loader.load(outside)

def test_extension_rejected(tmp_path):
    root=tmp_path/"root"; root.mkdir(); p=root/"x.exe"; p.write_text("x")
    loader=SafeDocumentLoader(root,(".txt",),100)
    with pytest.raises(UnsupportedFileError): loader.load(p)
