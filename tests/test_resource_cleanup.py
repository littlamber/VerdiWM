"""Local failure injection: owned resources close and operations can be retried."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import socket

import pytest

from wmloop import storage
from wmloop.control import campaign_api, local_http, workbench


def _assert_closed_and_rebind(server):
    address = server.server_address
    assert server.socket.fileno() == -1
    with socket.socket() as replacement:
        replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        replacement.bind(address)


@pytest.mark.parametrize("failure", ["directory", "token", "write"])
def test_local_server_initialization_failure_releases_listener(tmp_path, monkeypatch, failure):
    original_directory = local_http.tempfile.TemporaryDirectory
    captured = []
    original_init = ThreadingHTTPServer.__init__
    def capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured.append(self)
    def directory(*args, **kwargs):
        if failure == "directory":
            raise OSError("fixture directory failure")
        return original_directory(prefix="credential-", dir=tmp_path)
    def fail(*args, **kwargs):
        raise OSError("fixture credential failure")
    with monkeypatch.context() as fault:
        fault.setattr(ThreadingHTTPServer, "__init__", capture)
        fault.setattr(local_http.tempfile, "TemporaryDirectory", directory)
        if failure == "token":
            fault.setattr(local_http.secrets, "token_urlsafe", fail)
        if failure == "write":
            fault.setattr(local_http, "atomic_write", fail)
        with pytest.raises(OSError, match="fixture"):
            local_http.LocalHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    _assert_closed_and_rebind(captured[0])
    assert list(tmp_path.iterdir()) == []
    with local_http.LocalHTTPServer(captured[0].server_address, BaseHTTPRequestHandler) as retried:
        assert retried.auth_token_path.is_file()
    assert not retried.auth_token_path.exists()


@pytest.mark.parametrize("kind", ["workbench", "campaign"])
def test_state_initialization_failure_releases_server_credentials(tmp_path, monkeypatch, kind):
    servers = []
    original_init = local_http.LocalHTTPServer.__init__
    def capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        servers.append(self)
    def fail_store(*args, **kwargs):
        raise OSError("fixture state initialization failure")
    module = workbench if kind == "workbench" else campaign_api
    with monkeypatch.context() as fault:
        fault.setattr(local_http.LocalHTTPServer, "__init__", capture)
        fault.setattr(module, "CampaignStore", fail_store)
        with pytest.raises(OSError, match="fixture state"):
            if kind == "workbench":
                workbench.WorkbenchServer(("127.0.0.1", 0), state_root=tmp_path)
            else:
                campaign_api.serve(tmp_path, port=0)
    _assert_closed_and_rebind(servers[0])
    assert not servers[0].auth_token_path.parent.exists()
    servers[0].server_close()  # Repeated close is harmless.


def test_atomic_write_fdopen_failure_closes_descriptor_and_keeps_previous_value(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b"previous")
    descriptors = []
    def fail_open(descriptor, *args, **kwargs):
        descriptors.append(descriptor)
        raise OSError("fixture fdopen failure")
    with monkeypatch.context() as fault:
        fault.setattr(storage.os, "fdopen", fail_open)
        with pytest.raises(OSError, match="fixture fdopen"):
            storage.atomic_write(target, b"replacement")
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]
    storage.atomic_write(target, b"replacement")
    assert target.read_bytes() == b"replacement"


def test_atomic_replace_failure_preserves_previous_value_and_allows_retry(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b"previous")
    def fail_replace(*args, **kwargs):
        raise OSError("fixture replace failure")
    with monkeypatch.context() as fault:
        fault.setattr(storage.os, "replace", fail_replace)
        with pytest.raises(OSError, match="fixture replace"):
            storage.atomic_write(target, b"replacement")
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]
    storage.atomic_write(target, b"replacement")
    assert target.read_bytes() == b"replacement"


def test_failed_connection_initialization_closes_accepted_socket(monkeypatch):
    class Connection:
        closed = False
        def settimeout(self, timeout):
            raise OSError("fixture socket failure")
        def close(self):
            self.closed = True
    connection = Connection()
    monkeypatch.setattr(ThreadingHTTPServer, "get_request", lambda self: (connection, ("127.0.0.1", 1)))
    with local_http.LocalHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler) as server:
        with pytest.raises(OSError, match="fixture socket"):
            server.get_request()
    assert connection.closed


@pytest.mark.parametrize("failure", ["acquire", "release"])
def test_file_lock_closes_descriptor_on_lock_operation_failure(tmp_path, monkeypatch, failure):
    descriptors = []
    def fail_lock(descriptor, operation):
        descriptors.append(descriptor)
        if (operation == storage.fcntl.LOCK_UN) == (failure == "release"):
            raise OSError("fixture flock failure")
    with monkeypatch.context() as fault:
        fault.setattr(storage.fcntl, "flock", fail_lock)
        with pytest.raises(OSError, match="fixture flock"):
            with storage.exclusive_file_lock(tmp_path / "lock"):
                pass
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    with storage.exclusive_file_lock(tmp_path / "lock", blocking=False):
        pass
