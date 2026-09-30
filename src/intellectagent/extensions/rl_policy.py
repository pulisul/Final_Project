from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from .base import ExtensionHealth

class LinUCBPolicy:
    name="rl_policy"
    def __init__(self,n_features=4,n_actions=3,alpha=0.8,enabled=False,state_path="artifacts/linucb.json"):
        self.enabled=enabled; self.n_features=n_features; self.n_actions=n_actions; self.alpha=alpha; self.state_path=Path(state_path)
        self.A=[np.eye(n_features) for _ in range(n_actions)]; self.b=[np.zeros(n_features) for _ in range(n_actions)]
    def health(self): return ExtensionHealth(self.name,self.enabled,True,"LinUCB search-policy selector")
    def choose(self,x):
        x=np.asarray(x,dtype=float); scores=[]
        for a in range(self.n_actions):
            inv=np.linalg.inv(self.A[a]); theta=inv@self.b[a]
            scores.append(float(theta@x + self.alpha*math.sqrt(max(0.0,x@inv@x))))
        return int(np.argmax(scores)),scores
    def update(self,action,x,reward):
        x=np.asarray(x,dtype=float); self.A[action]+=np.outer(x,x); self.b[action]+=reward*x
    def save(self):
        self.state_path.parent.mkdir(parents=True,exist_ok=True)
        self.state_path.write_text(json.dumps({"A":[a.tolist() for a in self.A],"b":[b.tolist() for b in self.b],"alpha":self.alpha}),encoding="utf-8")
    def load(self):
        d=json.loads(self.state_path.read_text(encoding="utf-8")); self.A=[np.array(a,float) for a in d["A"]]; self.b=[np.array(b,float) for b in d["b"]]; self.alpha=float(d["alpha"])
    def apply(self,payload): return payload
