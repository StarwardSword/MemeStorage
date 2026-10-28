import os
import sqlite3
import time
import uuid
from contextlib import closing

from app import FileDescriptor, FileRecord, OwnerUsage

# Added after the first release, databases made before it get them on start
LATER_COLUMNS = {"owner_id": "TEXT", "size": "INTEGER", "created_at": "REAL"}


class SqliteDatabase:
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string)
        conn.row_factory = sqlite3.Row
        # With WAL a commit then does not wait for fsync; a power loss may drop the last commits
        # but never corrupts the database, and the file is not worth more than that
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def __init__(self, connection_string: str) -> None:
        # Connection string is a path to the database file, only its directory has to be created
        directory = os.path.dirname(connection_string)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.connection_string = connection_string

        with closing(self._conn()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")

            # There is no uuid type in sqlite
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_data(
                    id TEXT PRIMARY KEY NOT NULL
                        CHECK (length(id) = 36),
                    content_type TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    owner_id TEXT,
                    size INTEGER,
                    created_at REAL
                )
                """
            )

            have = {r["name"] for r in conn.execute("PRAGMA table_info(file_data)")}
            for name, kind in LATER_COLUMNS.items():
                if name not in have:
                    conn.execute(f"ALTER TABLE file_data ADD COLUMN {name} {kind}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS file_data_by_owner ON file_data(owner_id)"
            )
            conn.commit()

    def save(
        self, data: FileDescriptor, owner_id: uuid.UUID | None, size: int
    ) -> uuid.UUID:
        with closing(self._conn()) as conn:
            conn.execute(
                """INSERT INTO file_data(id, content_type, original_name, owner_id, size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(data.id),
                    data.content_type,
                    data.original_name,
                    str(owner_id) if owner_id else None,
                    size,
                    time.time(),
                ),
            )
            conn.commit()
            return data.id

    def remove(self, id: uuid.UUID) -> None:
        with closing(self._conn()) as conn:
            conn.execute("DELETE FROM file_data WHERE id = ?", (str(id),))
            conn.commit()

    def get(self, id: uuid.UUID) -> FileDescriptor:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT id, content_type, original_name FROM file_data WHERE id = ?",
                (str(id),),
            ).fetchone()

        if row is None:
            raise FileNotFoundError(str(id))

        return FileDescriptor(
            id=id, original_name=row["original_name"], content_type=row["content_type"]
        )

    # Drops entire query if can't find any of specified items
    def get_multiple(self, ids: list[uuid.UUID]) -> list[FileDescriptor]:
        strs = [str(x) for x in ids]
        idstr = ",".join("?" * len(ids))

        with closing(self._conn()) as conn:
            rows = conn.execute(
                f"SELECT id, content_type, original_name FROM file_data WHERE id IN ({idstr})",
                (strs),
            ).fetchall()

        items = {uuid.UUID(row["id"]): row for row in rows}

        lost = list(dict.fromkeys(str(x) for x in ids if x not in items))
        if lost:
            raise FileNotFoundError(", ".join(lost))

        return [
            FileDescriptor(
                id=id,
                original_name=items[id]["original_name"],
                content_type=items[id]["content_type"],
            )
            for id in ids
        ]

    def usage(self) -> list[OwnerUsage]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT owner_id,
                       substr(content_type, 1, instr(content_type || '/', '/') - 1) AS kind,
                       COUNT(*) AS files,
                       COALESCE(SUM(size), 0) AS bytes,
                       MAX(created_at) AS last_upload
                FROM file_data GROUP BY owner_id, kind
                """
            ).fetchall()

        return [
            OwnerUsage(
                owner_id=uuid.UUID(r["owner_id"]) if r["owner_id"] else None,
                kind=r["kind"],
                files=r["files"],
                bytes=r["bytes"],
                last_upload=r["last_upload"],
            )
            for r in rows
        ]

    # File ids are uuid7, so sorting by id is sorting by time of arrival
    def list_files(
        self, owner_id: uuid.UUID | None, limit: int, offset: int
    ) -> list[FileRecord]:
        where, args = ("WHERE owner_id = ?", [str(owner_id)]) if owner_id else ("", [])
        with closing(self._conn()) as conn:
            rows = conn.execute(
                f"""
                SELECT id, original_name, content_type, owner_id, size, created_at
                FROM file_data {where} ORDER BY id DESC LIMIT ? OFFSET ?
                """,
                (*args, limit, offset),
            ).fetchall()

        return [
            FileRecord(
                id=uuid.UUID(r["id"]),
                original_name=r["original_name"],
                content_type=r["content_type"],
                owner_id=uuid.UUID(r["owner_id"]) if r["owner_id"] else None,
                size=r["size"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
