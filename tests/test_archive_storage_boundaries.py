"""CPU checks for portable archive paths and concurrent content writes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import pytest

from wmloop.archive.store import ArchiveInvariantError, ArchiveStore, ContentAddressedStore


def _write_payload(root: str, payload: bytes) -> str:
    return ContentAddressedStore(Path(root)).put_bytes(payload, media_type="application/octet-stream").uri


@pytest.mark.parametrize("kind", ["cas", "archive"])
def test_storage_rejects_linked_parent_before_creating_files(tmp_path, kind):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ArchiveInvariantError, match="ROOT_INVALID|DATABASE_INVALID"):
        if kind == "cas":
            ContentAddressedStore(alias / "objects")
        else:
            ArchiveStore(alias / "archive.sqlite3")
    assert not list(actual.iterdir())


def test_concurrent_cas_writers_preserve_content_and_leave_no_temporary_files(tmp_path):
    payloads = [b"shared fixture", b"other fixture"] * 4
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        refs = list(pool.map(_write_payload, [str(tmp_path)] * len(payloads), payloads))
    cas = ContentAddressedStore(tmp_path)
    assert len(set(refs)) == 2
    for ref, payload in zip(refs, payloads):
        assert cas.read_bytes(ref) == payload
    assert not list(tmp_path.rglob("*.tmp"))


def test_repeated_cas_write_does_not_replace_existing_object(tmp_path):
    cas = ContentAddressedStore(tmp_path)
    first = cas.put_bytes(b"fixture", media_type="text/plain")
    before = first.path.stat()
    second = cas.put_bytes(b"fixture", media_type="text/plain")
    after = second.path.stat()
    assert first.uri == second.uri
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
