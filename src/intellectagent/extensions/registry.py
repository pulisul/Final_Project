from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .base import ExtensionHealth

@dataclass
class ExtensionRegistry:
    _items: dict[str, Any] = field(default_factory=dict)

    def register(self, ext: Any) -> None:
        if ext.name in self._items:
            raise ValueError(f"duplicate extension: {ext.name}")
        self._items[ext.name] = ext

    def get(self, name: str) -> Any | None:
        return self._items.get(name)

    def health(self) -> list[ExtensionHealth]:
        return [self._items[k].health() for k in sorted(self._items)]

    def apply_if_ready(self, name: str, payload: Any) -> Any:
        ext = self.get(name)
        if ext is None:
            return payload
        h = ext.health()
        if not (h.enabled and h.ready):
            return payload
        return ext.apply(payload)
