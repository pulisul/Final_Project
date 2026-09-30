from __future__ import annotations
from urllib.parse import urlparse
import ipaddress, socket
from .base import ExtensionHealth

class WebSearchExtension:
    name="web_search"
    def __init__(self, enabled=False, allowed_hosts:tuple[str,...]=()):
        self.enabled=enabled; self.allowed_hosts=set(allowed_hosts)
    def health(self): return ExtensionHealth(self.name,self.enabled,bool(self.allowed_hosts),"allowlist-only web access")
    def validate_url(self,url:str):
        p=urlparse(url)
        if p.scheme not in {"https"}: raise ValueError("https only")
        if not p.hostname or p.hostname not in self.allowed_hosts: raise ValueError("host not allowlisted")
        for info in socket.getaddrinfo(p.hostname,443):
            ip=ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: raise ValueError("private/reserved address blocked")
        return True
    def apply(self,payload): return payload
