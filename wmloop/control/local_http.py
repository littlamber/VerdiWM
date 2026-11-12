"""Shared transport boundary for trusted-user, loopback-only control services."""

from __future__ import annotations

import ipaddress
import json
import hmac
from pathlib import Path
import secrets
import socket
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from wmloop.storage import atomic_write


class LocalHTTPServer(ThreadingHTTPServer):
    """Never expose privileged local operations on a non-loopback interface."""

    daemon_threads = True
    public_get_routes: frozenset[str] = frozenset()

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler]):
        host, port = address
        if host == "localhost":
            host = "127.0.0.1"
        try:
            ip = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("HTTP_LOOPBACK_ADDRESS_REQUIRED") from exc
        if not ip.is_loopback:
            raise ValueError("HTTP_LOOPBACK_ADDRESS_REQUIRED")
        self.address_family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        super().__init__((host, port), handler)
        try:
            self._auth_directory = tempfile.TemporaryDirectory(prefix="verdiwm-http-")
            self.auth_token = secrets.token_urlsafe(32)
            self.auth_token_path = Path(self._auth_directory.name) / "access-token"
            atomic_write(self.auth_token_path, (self.auth_token + "\n").encode())
        except BaseException:
            self.server_close()
            raise

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            if hasattr(self, "_auth_directory"):
                self._auth_directory.cleanup()

    def get_request(self):
        connection, address = super().get_request()
        try:
            connection.settimeout(10)
        except BaseException:
            connection.close()
            raise
        return connection, address


class LocalRequestHandler(BaseHTTPRequestHandler):
    """Reject DNS rebinding, cross-origin browser access and ambiguous framing."""

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        try:
            if len(hosts) != 1 or len(origins) > 1:
                raise ValueError
            host = urlsplit("http://" + hosts[0])
            allowed = {"localhost", str(self.server.server_address[0])}
            if (
                host.hostname not in allowed
                or (host.port or 80) != self.server.server_port
                or host.username is not None
                or host.password is not None
                or host.path or host.query or host.fragment
            ):
                raise ValueError
            if origins:
                origin = urlsplit(origins[0])
                if (
                    origin.scheme != "http"
                    or origin.netloc.lower() != host.netloc.lower()
                    or origin.path or origin.query or origin.fragment
                ):
                    raise ValueError
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise ValueError
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "LOCAL_ORIGIN_REQUIRED")
            return False
        route = urlsplit(self.path).path
        # Only fixed, data-free browser shells and assets are public. Both
        # read and write APIs require a fresh per-process bearer credential.
        if self.command == "GET" and route in self.server.public_get_routes:
            return True
        authorization = self.headers.get_all("Authorization", [])
        expected = "Bearer " + self.server.auth_token
        if len(authorization) != 1 or not hmac.compare_digest(authorization[0].encode(), expected.encode()):
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json")
            payload = b'{"error":"LOCAL_AUTH_REQUIRED"}\n'
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return False
        if route == "/auth/session" and self.command == "GET":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return False
        return True

    def read_json_object(self) -> dict[str, object]:
        if self.headers.get_all("Transfer-Encoding"):
            raise ValueError("TRANSFER_ENCODING_UNSUPPORTED")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            raise ValueError("CONTENT_LENGTH_INVALID")
        length = int(lengths[0])
        if not 2 <= length <= 1_048_576:
            raise ValueError("REQUEST_BODY_SIZE_INVALID")
        if len(self.headers.get_all("Content-Type", [])) != 1 or self.headers.get_content_type() != "application/json":
            raise ValueError("CONTENT_TYPE_JSON_REQUIRED")
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("REQUEST_BODY_TRUNCATED")
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TimeoutError) as exc:
            raise ValueError("REQUEST_JSON_INVALID") from exc
        if not isinstance(value, dict):
            raise ValueError("REQUEST_OBJECT_REQUIRED")
        return value
