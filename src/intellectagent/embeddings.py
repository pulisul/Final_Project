from __future__ import annotations
import hashlib, math, re
from typing import Protocol, Sequence

class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

class HashEmbedder:
    """Deterministic dependency-free embedder for tests/smoke gates, NOT production quality."""
    def __init__(self, dims: int = 256): self.dims = dims
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out=[]
        for text in texts:
            v=[0.0]*self.dims
            for tok in re.findall(r"[\w가-힣]+", text.lower()):
                h=int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
                v[h % self.dims] += 1.0 if ((h >> 8) & 1) else -1.0
            norm=math.sqrt(sum(x*x for x in v)) or 1.0
            out.append([x/norm for x in v])
        return out

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        self.device = device
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        arr = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, row)) for row in arr]
