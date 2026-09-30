from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class PromotionResult:
    feature:str; passed:bool; reasons:tuple[str,...]

class PromotionGate:
    def __init__(self, baseline_file="evidence/baseline_metrics.json", policy_file="config/promotion_policy.json"):
        self.baseline_file=Path(baseline_file); self.policy_file=Path(policy_file)
    def evaluate(self, feature:str, candidate:dict)->PromotionResult:
        reasons=[]
        if not self.baseline_file.exists(): return PromotionResult(feature,False,("baseline metrics missing",))
        baseline=json.loads(self.baseline_file.read_text(encoding="utf-8"))
        policy=json.loads(self.policy_file.read_text(encoding="utf-8"))
        p=policy.get(feature,policy.get("default",{}))
        for metric,min_delta in p.get("min_delta",{}).items():
            b=float(baseline.get(metric,0)); c=float(candidate.get(metric,0))
            if c-b < float(min_delta): reasons.append(f"{metric}: delta {c-b:.4f} < {min_delta}")
        for metric,max_reg in p.get("max_regression",{}).items():
            b=float(baseline.get(metric,0)); c=float(candidate.get(metric,0))
            if b-c > float(max_reg): reasons.append(f"{metric}: regression {b-c:.4f} > {max_reg}")
        if candidate.get("tests_failed",0) != 0: reasons.append("tests_failed != 0")
        return PromotionResult(feature,not reasons,tuple(reasons))
