from pathlib import Path
import hashlib,sys
p=Path("requirements-lock.txt")
if not p.exists(): raise SystemExit("requirements-lock.txt missing")
print("requirements-lock.txt sha256:",hashlib.sha256(p.read_bytes()).hexdigest())
print("python:",sys.version)
