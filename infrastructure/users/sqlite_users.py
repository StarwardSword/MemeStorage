import os
import sqlite3
import time
import uuid
from contextlib import closing
from typing import override

from app import User, UserExistsError, UserNotFoundError, UserStore

COLUMNS = "id, name, allow_embeddings, created_at, deleted_at"


def to_user(row: sqlite3.Row) -> User:
    return User(
        id=uuid.UUID(row["id"]),
        name=row["name"],
        allow_embeddings=bool(row["allow_embeddings"]),
        created_at=row["created_at"],
        deleted_at=row["deleted_at"],
    )


class SqliteUserStore(UserStore):
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
            # token_hash is NULL for a deleted user, UNIQUE lets any number of NULLs in
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users(
                    id TEXT PRIMARY KEY NOT NULL CHECK (length(id) = 36),
                    name TEXT NOT NULL UNIQUE,
                    token_hash TEXT UNIQUE,
                    allow_embeddings INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    deleted_at REAL
                )
                """
            )
            conn.commit()

    @override
    def add(self, user: User, token_hash: str) -> None:
        with closing(self._conn()) as conn:
            try:
                conn.execute(
                    "INSERT INTO users(id, name, token_hash, allow_embeddings, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        str(user.id),
                        user.name,
                        token_hash,
                        int(user.allow_embeddings),
                        user.created_at,
                    ),
                )
            except sqlite3.IntegrityError:
                raise UserExistsError(user.name) from None
            conn.commit()

    @override
    def find_by_token(self, token_hash: str) -> User | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                f"SELECT {COLUMNS} FROM users WHERE token_hash = ? AND deleted_at IS NULL",
                (token_hash,),
            ).fetchone()
        return None if row is None else to_user(row)

    @override
    def get(self, id: uuid.UUID) -> User | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                f"SELECT {COLUMNS} FROM users WHERE id = ?", (str(id),)
            ).fetchone()
        return None if row is None else to_user(row)

    @override
    def list(self) -> list[User]:
        with closing(self._conn()) as conn:
            rows = conn.execute(f"SELECT {COLUMNS} FROM users ORDER BY id").fetchall()
        return [to_user(r) for r in rows]

    def _change(self, sql: str, args: tuple[object, ...], id: uuid.UUID) -> None:
        with closing(self._conn()) as conn:
            changed = conn.execute(
                f"{sql} WHERE id = ? AND deleted_at IS NULL", (*args, str(id))
            ).rowcount
            conn.commit()
        if changed == 0:
            raise UserNotFoundError(str(id))

    @override
    def update(self, id: uuid.UUID, allow_embeddings: bool) -> User:
        self._change(
            "UPDATE users SET allow_embeddings = ?", (int(allow_embeddings),), id
        )
        user = self.get(id)
        if user is None:
            raise UserNotFoundError(str(id))
        return user

    @override
    def set_token(self, id: uuid.UUID, token_hash: str) -> bool:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT deleted_at FROM users WHERE id = ?", (str(id),)
            ).fetchone()
            if row is None:
                raise UserNotFoundError(str(id))
            conn.execute(
                "UPDATE users SET token_hash = ?, deleted_at = NULL WHERE id = ?",
                (token_hash, str(id)),
            )
            conn.commit()
        return row["deleted_at"] is not None

    @override
    def delete(self, id: uuid.UUID) -> None:
        self._change(
            "UPDATE users SET token_hash = NULL, deleted_at = ?", (time.time(),), id
        )
