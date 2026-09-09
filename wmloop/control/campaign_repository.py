"""Transactional campaign state, with recoverable JSON compatibility projections.

SQLite is authoritative once a legacy campaign has been imported. JSON files
remain inspectable exports; editing them does not mutate a migrated campaign.
Every mutation and dispatch outbox update shares one BEGIN IMMEDIATE transaction.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading

from wmloop.storage import atomic_write, canonical_bytes, checked_path


class CampaignAPIError(ValueError):
    """Stable client-facing campaign validation or storage failure."""


class CampaignRepository:
    def __init__(self, root: Path, *, read_only: bool = False):
        self.root = checked_path(root, code="CAMPAIGN_ROOT_INVALID", error=CampaignAPIError)
        self.database = self.root / "campaigns.sqlite3"
        self.marker = self.root / ".sqlite-campaign-store"
        self.read_only = read_only
        self._mutex = threading.RLock()
        self._local = threading.local()
        if not read_only:
            if self.marker.exists() and not self.database.is_file():
                raise CampaignAPIError("CAMPAIGN_STORE_UNAVAILABLE")
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            with self._connect(write=True) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS campaigns (campaign_id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)")
                connection.execute("CREATE TABLE IF NOT EXISTS outbox (path TEXT PRIMARY KEY, payload BLOB)")
            self.database.chmod(0o600)
            if not self.marker.exists():
                atomic_write(self.marker, b"SQLite campaign state v1; JSON files are projections.\n")
            with self.transaction():
                pass

    @contextmanager
    def _connect(self, *, write: bool = False):
        checked_path(self.database, code="CAMPAIGN_DATABASE_INVALID", error=CampaignAPIError)
        uri = self.database.as_uri() + ("?mode=rwc" if write else "?mode=ro")
        connection = None
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None)
            yield connection
        except sqlite3.Error as exc:
            raise CampaignAPIError("CAMPAIGN_STORE_UNAVAILABLE") from exc
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def transaction(self):
        if self.read_only:
            raise CampaignAPIError("CAMPAIGN_STORE_READ_ONLY")
        with self._mutex:
            if getattr(self._local, "connection", None) is not None:
                yield
                return
            with self._connect(write=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._local.connection = connection
                try:
                    # Replay committed filesystem projections before reading dispatch files.
                    self._flush(connection)
                    yield
                    connection.commit()
                    connection.execute("BEGIN IMMEDIATE")
                    self._flush(connection)
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
                finally:
                    self._local.connection = None

    def _flush(self, connection):
        for path, payload in connection.execute("SELECT path, payload FROM outbox ORDER BY rowid").fetchall():
            target = checked_path(Path(path), code="CAMPAIGN_PROJECTION_INVALID", error=CampaignAPIError)
            if payload is None:
                target.unlink(missing_ok=True)
            else:
                atomic_write(target, payload)
            connection.execute("DELETE FROM outbox WHERE path = ?", (path,))

    def _read_legacy(self, campaign_id):
        path = checked_path(self.root / f"{campaign_id}.json", code="CAMPAIGN_RECORD_INVALID", error=CampaignAPIError)
        if not path.is_file():
            raise CampaignAPIError("CAMPAIGN_NOT_FOUND")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CampaignAPIError("CAMPAIGN_RECORD_INVALID") from exc
        self._validate(value, campaign_id)
        value["state_version"] = 0
        return value

    @staticmethod
    def _validate(value, campaign_id):
        if not isinstance(value, dict) or value.get("campaign_id") != campaign_id or value.get("status") not in {"created", "confirmed", "queued", "running", "completed", "failed", "blocked", "cancelled"}:
            raise CampaignAPIError("CAMPAIGN_RECORD_INVALID")

    def get(self, campaign_id):
        if self.marker.exists() and not self.database.is_file():
            raise CampaignAPIError("CAMPAIGN_STORE_UNAVAILABLE")
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            row = connection.execute("SELECT payload FROM campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()
        elif self.database.exists():
            with self._connect() as reader:
                row = reader.execute("SELECT payload FROM campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()
        else:
            row = None
        if row is None:
            value = self._read_legacy(campaign_id)
            if connection is not None:
                connection.execute("INSERT INTO campaigns VALUES (?, 0, ?)", (campaign_id, canonical_bytes(value).decode()))
            return value
        try:
            value = json.loads(row[0])
        except ValueError as exc:
            raise CampaignAPIError("CAMPAIGN_RECORD_INVALID") from exc
        self._validate(value, campaign_id)
        return value

    def ids(self):
        if self.marker.exists() and not self.database.is_file():
            raise CampaignAPIError("CAMPAIGN_STORE_UNAVAILABLE")
        ids = {p.stem for p in self.root.glob("*.json") if p.is_file() and not p.is_symlink()}
        if self.database.exists():
            with self._connect() as reader:
                ids.update(row[0] for row in reader.execute("SELECT campaign_id FROM campaigns"))
        return sorted(ids)

    def write(self, path: Path, value: dict):
        with self.transaction():
            connection = self._local.connection
            if path.parent == self.root:
                campaign_id = path.stem
                self._validate(value, campaign_id)
                row = connection.execute("SELECT version FROM campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()
                version = row[0] if row else 0
                if row and value.get("state_version") != version:
                    raise CampaignAPIError("CAMPAIGN_VERSION_CONFLICT")
                value["state_version"] = version + 1
                connection.execute("INSERT INTO campaigns VALUES (?, ?, ?) ON CONFLICT(campaign_id) DO UPDATE SET version=excluded.version, payload=excluded.payload", (campaign_id, version + 1, canonical_bytes(value).decode()))
            connection.execute("INSERT INTO outbox VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET payload=excluded.payload", (str(path), canonical_bytes(value) + b"\n"))

    def delete_projection(self, path: Path):
        with self.transaction():
            self._local.connection.execute("INSERT INTO outbox VALUES (?, NULL) ON CONFLICT(path) DO UPDATE SET payload=NULL", (str(path),))
