import os
import sqlite3
import uuid
from contextlib import closing
from typing import override

from app import FailureKind, Job, JobCounts, JobStatus, JobStore

COLUMNS = "id, original_name, description, owner_id, embeddings, text_embeddings, status, error_kind, error_message"


def to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=uuid.UUID(row["id"]),
        original_name=row["original_name"],
        description=row["description"],
        owner_id=uuid.UUID(row["owner_id"]) if row["owner_id"] else None,
        embeddings=row["embeddings"],
        text_embeddings=row["text_embeddings"],
        status=JobStatus(row["status"]),
        error_kind=FailureKind[row["error_kind"]] if row["error_kind"] else None,
        error_message=row["error_message"],
    )


class SqliteJobStore(JobStore):
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def __init__(self, connection_string: str) -> None:
        directory = os.path.dirname(connection_string)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.connection_string = connection_string

        with closing(self._conn()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            # Job id is the file id, a uuid7, so sorting by it is sorting by arrival
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY NOT NULL CHECK (length(id) = 36),
                    original_name TEXT NOT NULL,
                    description TEXT,
                    owner_id TEXT,
                    embeddings BLOB,
                    text_embeddings BLOB,
                    status TEXT NOT NULL,
                    error_kind TEXT,
                    error_message TEXT
                )
                """
            )
            # Columns added after the table first appeared
            have = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
            for name, kind in (
                ("owner_id", "TEXT"),
                ("embeddings", "BLOB"),
                ("text_embeddings", "BLOB"),
            ):
                if name not in have:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs(status, id)"
            )
            conn.commit()

    @override
    def add(self, jobs: list[Job]) -> None:
        with closing(self._conn()) as conn:
            conn.executemany(
                "INSERT INTO jobs(id, original_name, description, owner_id, embeddings, text_embeddings, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(j.id),
                        j.original_name,
                        j.description,
                        str(j.owner_id) if j.owner_id else None,
                        j.embeddings,
                        j.text_embeddings,
                        j.status.value,
                    )
                    for j in jobs
                ],
            )
            conn.commit()

    @override
    def get(self, id: uuid.UUID) -> Job | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                f"SELECT {COLUMNS} FROM jobs WHERE id = ?", (str(id),)
            ).fetchone()
        return None if row is None else to_job(row)

    @override
    def claim(self) -> Job | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                f"""
                UPDATE jobs SET status = ?
                WHERE id = (SELECT id FROM jobs WHERE status = ? ORDER BY id LIMIT 1)
                RETURNING {COLUMNS}
                """,
                (JobStatus.RUNNING.value, JobStatus.PENDING.value),
            ).fetchone()
            conn.commit()
        return None if row is None else to_job(row)

    @override
    def finish(self, id: uuid.UUID, error: tuple[FailureKind, str] | None) -> None:
        status = JobStatus.DONE if error is None else JobStatus.FAILED
        with closing(self._conn()) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_kind = ?, error_message = ? WHERE id = ?",
                (
                    status.value,
                    error[0].name if error else None,
                    error[1] if error else None,
                    str(id),
                ),
            )
            conn.commit()

    @override
    def active_count(self) -> int:
        with closing(self._conn()) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN (?, ?)",
                (JobStatus.PENDING.value, JobStatus.RUNNING.value),
            ).fetchone()[0]

    @override
    def counts(self) -> JobCounts:
        with closing(self._conn()) as conn:
            by_status = conn.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            ).fetchall()
            failures = conn.execute(
                "SELECT error_kind, COUNT(*) FROM jobs WHERE status = ? GROUP BY error_kind",
                (JobStatus.FAILED.value,),
            ).fetchall()
        return JobCounts(
            by_status={r[0]: r[1] for r in by_status},
            failures={(r[0] or "unknown").lower(): r[1] for r in failures},
        )

    @override
    def requeue_running(self) -> int:
        with closing(self._conn()) as conn:
            count = conn.execute(
                "UPDATE jobs SET status = ? WHERE status = ?",
                (JobStatus.PENDING.value, JobStatus.RUNNING.value),
            ).rowcount
            conn.commit()
        return count
