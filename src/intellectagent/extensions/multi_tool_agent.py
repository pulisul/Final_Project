from __future__ import annotations
from .base import ExtensionHealth
class MultiToolAgentExtension:
    name="multi_tool_agent"
    def __init__(self, tools:dict, enabled=False,max_steps=4): self.tools=tools; self.enabled=enabled; self.max_steps=max_steps
    def health(self): return ExtensionHealth(self.name,self.enabled,0 < self.max_steps <= 8,"bounded allowlist tool runner")
    def call(self,name,**kwargs):
        if name not in self.tools: raise PermissionError(f"tool not allowed: {name}")
        return self.tools[name](**kwargs)
    def apply(self,payload): return payload
