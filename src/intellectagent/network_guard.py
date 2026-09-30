from __future__ import annotations

import ipaddress
import json
import socket
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NetworkAttempt:
    host: str
    port: int | None
    allowed: bool
    reason: str


class LocalOnlyNetworkGuard(AbstractContextManager):
    """Application-level guard used during local evaluation.

    It blocks non-loopback outbound socket connections in the current Python process
    and records attempts. This is useful evidence, but it is not a packet-capture or
    OS-firewall proof and cannot cover child processes that bypass this interpreter.
    """

    def __init__(self) -> None:
        self.attempts: list[NetworkAttempt] = []
        self._orig_create_connection = None
        self._orig_socket_connect = None

    @staticmethod
    def _host_allowed(host: str) -> bool:
        if host.lower() in {"localhost", "ip6-localhost"}:
            return True
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        ips = []
        for info in infos:
            sockaddr = info[4]
            if sockaddr:
                ips.append(sockaddr[0])
        if not ips:
            return False
        return all(ipaddress.ip_address(ip).is_loopback for ip in ips)

    def _record_or_raise(self, address: Any) -> None:
        if isinstance(address, tuple):
            host = str(address[0])
            port = int(address[1]) if len(address) > 1 and address[1] is not None else None
        else:
            host, port = str(address), None
        allowed = self._host_allowed(host)
        self.attempts.append(NetworkAttempt(host, port, allowed, "loopback" if allowed else "external_blocked"))
        if not allowed:
            raise RuntimeError(f"external network blocked by LocalOnlyNetworkGuard: {host}:{port}")

    def __enter__(self):
        self._orig_create_connection = socket.create_connection
        self._orig_socket_connect = socket.socket.connect
        guard = self

        def guarded_create_connection(address, *args, **kwargs):
            guard._record_or_raise(address)
            return guard._orig_create_connection(address, *args, **kwargs)

        def guarded_socket_connect(sock, address):
            guard._record_or_raise(address)
            return guard._orig_socket_connect(sock, address)

        socket.create_connection = guarded_create_connection
        socket.socket.connect = guarded_socket_connect
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_create_connection is not None:
            socket.create_connection = self._orig_create_connection
        if self._orig_socket_connect is not None:
            socket.socket.connect = self._orig_socket_connect
        return False

    def evidence(self) -> dict[str, Any]:
        return {
            "guard": "LocalOnlyNetworkGuard",
            "scope": "current Python process only",
            "external_attempt_count": sum(1 for x in self.attempts if not x.allowed),
            "allowed_attempt_count": sum(1 for x in self.attempts if x.allowed),
            "attempts": [x.__dict__ for x in self.attempts],
            "limitation": "Not equivalent to OS firewall or packet capture; child processes require separate controls.",
        }

    def write_evidence(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.evidence(), ensure_ascii=False, indent=2), encoding="utf-8")
