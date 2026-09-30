from __future__ import annotations
import re
from dataclasses import dataclass
from .models import SearchHit

@dataclass(frozen=True)
class GroundingResult:
    ok: bool
    cited_ids: tuple[str,...]
    unsupported_citations: tuple[str,...]
    citation_coverage: float

def validate_grounding(answer_text:str,hits:list[SearchHit]) -> GroundingResult:
    valid={h.chunk_id for h in hits}
    cited=tuple(dict.fromkeys(re.findall(r"\[([0-9a-f]{20})\]",answer_text)))
    bad=tuple(x for x in cited if x not in valid)
    claim_lines=[ln.strip() for ln in answer_text.splitlines() if ln.strip()]
    covered=sum(1 for ln in claim_lines if any(f"[{cid}]" in ln for cid in valid))
    coverage=covered/len(claim_lines) if claim_lines else 1.0
    return GroundingResult(bool(cited) and not bad and coverage>=1.0,cited,bad,coverage)
