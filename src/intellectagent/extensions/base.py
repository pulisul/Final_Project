from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any

@dataclass(frozen=True)
class ExtensionHealth:
    name: str
    enabled: bool
    ready: bool
    detail: str = ""

class Extension(Protocol):
    name: str
    def health(self) -> ExtensionHealth: ...
    def apply(self, payload: Any) -> Any: ...
