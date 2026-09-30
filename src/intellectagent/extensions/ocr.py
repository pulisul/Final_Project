from __future__ import annotations
from .base import ExtensionHealth
class OCRExtension:
    name="ocr"
    def __init__(self, enabled=False): self.enabled=enabled
    def health(self):
        try:
            import pytesseract  # noqa
            return ExtensionHealth(self.name,self.enabled,True,"pytesseract available")
        except Exception as e:
            return ExtensionHealth(self.name,self.enabled,False,f"optional dependency missing: {e}")
    def apply(self,payload): return payload
