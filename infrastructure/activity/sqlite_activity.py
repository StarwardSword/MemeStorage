import os
import sqlite3
import uuid
from contextlib import closing
from typing import override

from app import (
    ActionStats,
    ActivityStore,
    ActivitySummary,
    ActorStats,
    DayStats,
    Event,
    IpStats,
    QueryCount,
)

# Actions whose detail is the text of a search query
SEARCH_ACTIONS = ("search_files", "search_by_query", "GetByQuery")
TOP = 20

COLUMNS = (
    "ts, transport, action, code, outcome, user_id, actor, duration_ms,"
    " bytes_in, bytes_out, ip, path, detail"
)

WINDOW = "ts >= ? AND ts < ?"


def to_event(row: sqlite3.Row) -> Event:
    return Event(
        ts=row["ts"],
        transport=row["transport"],
        action=row["action"],
        code=row["code"],
        outcome=row["outcome"],
        user_id=uuid.UUID(row["user_id"]) if row["user_id"] else None,
        actor=row["actor"],
        duration_ms=row["duration_ms"],
        bytes_in=row["bytes_in"],
        bytes_out=row["bytes_out"],
        ip=row["ip"],
        path=row["path"],
        detail=row["detail"],
    )


class SqliteActivityStore(ActivityStore):
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    transport TEXT NOT NULL,
                    action TEXT NOT NULL,
                    code TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    user_id TEXT,
                    actor TEXT,
                    duration_ms REAL NOT NULL,
                    bytes_in INTEGER NOT NULL,
                    bytes_out INTEGER NOT NULL,
                    ip TEXT,
                    path TEXT,
                    detail TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS events_by_ts ON events(ts)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS events_by_user ON events(user_id, ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS events_by_action ON events(action, ts)"
            )
            conn.commit()

    @override
    def record(self, events: list[Event]) -> None:
        with closing(self._conn()) as conn:
            conn.executemany(
                f"INSERT INTO events({COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        e.ts,
                        e.transport,
                        e.action,
                        e.code,
                        e.outcome,
                        str(e.user_id) if e.user_id else None,
                        e.actor,
                        e.duration_ms,
                        e.bytes_in,
                        e.bytes_out,
                        e.ip,
                        e.path,
                        e.detail,
                    )
                    for e in events
                ],
            )
            conn.commit()

    @override
    def events(
        self,
        since: float,
        until: float,
        user_id: uuid.UUID | None,
        action: str | None,
        outcome: str | None,
        limit: int,
        offset: int,
    ) -> list[Event]:
        where, args = [WINDOW], [since, until]
        for column, value in (
            ("user_id", user_id),
            ("action", action),
            ("outcome", outcome),
        ):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(str(value))

        with closing(self._conn()) as conn:
            rows = conn.execute(
                f"SELECT {COLUMNS} FROM events WHERE {' AND '.join(where)}"
                " ORDER BY id DESC LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
        return [to_event(r) for r in rows]

    @override
    def summary(self, since: float, until: float) -> ActivitySummary:
        window = (since, until)
        with closing(self._conn()) as conn:
            total = conn.execute(
                f"""
                SELECT COUNT(*) AS n,
                       COALESCE(SUM(outcome = 'ok'), 0) AS ok,
                       COALESCE(SUM(outcome = 'denied'), 0) AS denied,
                       COALESCE(SUM(outcome = 'client_error'), 0) AS client_error,
                       COALESCE(SUM(outcome = 'server_error'), 0) AS server_error,
                       COALESCE(AVG(duration_ms), 0) AS avg_ms,
                       COALESCE(SUM(bytes_in), 0) AS bytes_in,
                       COALESCE(SUM(bytes_out), 0) AS bytes_out
                FROM events WHERE {WINDOW}
                """,
                window,
            ).fetchone()

            transports = conn.execute(
                f"SELECT transport, COUNT(*) FROM events WHERE {WINDOW} GROUP BY transport",
                window,
            ).fetchall()

            actions = conn.execute(
                f"""
                SELECT action, COUNT(*) AS n,
                       SUM(outcome = 'ok') AS ok,
                       SUM(outcome = 'denied') AS denied,
                       SUM(outcome = 'client_error') AS client_error,
                       SUM(outcome = 'server_error') AS server_error,
                       AVG(duration_ms) AS avg_ms, MAX(duration_ms) AS max_ms,
                       SUM(bytes_in) AS bytes_in, SUM(bytes_out) AS bytes_out
                FROM events WHERE {WINDOW} GROUP BY action ORDER BY n DESC
                """,
                window,
            ).fetchall()

            by_action = []
            for a in actions:
                # The value below which 95% of the durations are
                p95 = conn.execute(
                    f"SELECT duration_ms FROM events WHERE {WINDOW} AND action = ?"
                    " ORDER BY duration_ms LIMIT 1 OFFSET ?",
                    (*window, a["action"], (a["n"] - 1) * 95 // 100),
                ).fetchone()
                by_action.append(
                    ActionStats(
                        action=a["action"],
                        requests=a["n"],
                        ok=a["ok"],
                        denied=a["denied"],
                        client_error=a["client_error"],
                        server_error=a["server_error"],
                        avg_ms=a["avg_ms"],
                        p95_ms=p95[0] if p95 else 0,
                        max_ms=a["max_ms"],
                        bytes_in=a["bytes_in"],
                        bytes_out=a["bytes_out"],
                    )
                )

            actors = conn.execute(
                f"""
                SELECT user_id, actor, COUNT(*) AS n,
                       SUM(outcome != 'ok') AS errors,
                       SUM(bytes_in) AS bytes_in, SUM(bytes_out) AS bytes_out,
                       MIN(ts) AS first_seen, MAX(ts) AS last_seen
                FROM events WHERE {WINDOW} GROUP BY user_id, actor ORDER BY n DESC
                """,
                window,
            ).fetchall()

            days = conn.execute(
                f"""
                SELECT strftime('%Y-%m-%d', ts, 'unixepoch') AS day, COUNT(*) AS n,
                       SUM(outcome != 'ok') AS errors,
                       SUM(bytes_in) AS bytes_in, SUM(bytes_out) AS bytes_out
                FROM events WHERE {WINDOW} GROUP BY day ORDER BY day
                """,
                window,
            ).fetchall()

            marks = ",".join("?" * len(SEARCH_ACTIONS))
            queries = conn.execute(
                f"""
                SELECT detail, COUNT(*) AS n FROM events
                WHERE {WINDOW} AND action IN ({marks}) AND detail IS NOT NULL
                GROUP BY detail ORDER BY n DESC LIMIT {TOP}
                """,
                (*window, *SEARCH_ACTIONS),
            ).fetchall()

            ips = conn.execute(
                f"""
                SELECT ip, COUNT(*) AS n, SUM(outcome = 'denied') AS denied
                FROM events WHERE {WINDOW} AND ip IS NOT NULL
                GROUP BY ip ORDER BY n DESC LIMIT {TOP}
                """,
                window,
            ).fetchall()

        return ActivitySummary(
            since=since,
            until=until,
            requests=total["n"],
            ok=total["ok"],
            denied=total["denied"],
            client_error=total["client_error"],
            server_error=total["server_error"],
            avg_ms=total["avg_ms"],
            bytes_in=total["bytes_in"],
            bytes_out=total["bytes_out"],
            by_transport={r[0]: r[1] for r in transports},
            by_action=by_action,
            by_actor=[
                ActorStats(
                    user_id=uuid.UUID(r["user_id"]) if r["user_id"] else None,
                    actor=r["actor"],
                    requests=r["n"],
                    errors=r["errors"],
                    bytes_in=r["bytes_in"],
                    bytes_out=r["bytes_out"],
                    first_seen=r["first_seen"],
                    last_seen=r["last_seen"],
                )
                for r in actors
            ],
            by_day=[
                DayStats(
                    day=r["day"],
                    requests=r["n"],
                    errors=r["errors"],
                    bytes_in=r["bytes_in"],
                    bytes_out=r["bytes_out"],
                )
                for r in days
            ],
            top_queries=[QueryCount(query=r["detail"], count=r["n"]) for r in queries],
            by_ip=[
                IpStats(ip=r["ip"], requests=r["n"], denied=r["denied"]) for r in ips
            ],
        )

    @override
    def prune(self, before: float) -> int:
        with closing(self._conn()) as conn:
            deleted = conn.execute(
                "DELETE FROM events WHERE ts < ?", (before,)
            ).rowcount
            conn.commit()
        return deleted
