import uuid
import sqlite3
import os
from contextlib import closing
from app import FileDescriptor

class SqliteDatabase():
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string)
        conn.row_factory = sqlite3.Row
        return conn

    def __init__(self, connection_string: str) -> None:
        # Connection string is a path to the database file, only its directory has to be created
        directory = os.path.dirname(connection_string)
        if directory: os.makedirs(directory, exist_ok=True)

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
                    original_name TEXT NOT NULL
                )
                """
            )

    def save(self, data: FileDescriptor) -> uuid.UUID:
        with closing(self._conn()) as conn:
            conn.execute(
                """INSERT INTO file_data(id, content_type, original_name) VALUES (?, ?, ?)""",
                (str(data.id), data.content_type, data.original_name)
            )
            conn.commit()
            return data.id

    def remove(self, id: uuid.UUID) -> None:
        with closing(self._conn()) as conn:
            conn.execute(
                "DELETE FROM file_data WHERE id = ?",
                (str(id),)
            )
            conn.commit()
    
    def get(self, id: uuid.UUID) -> FileDescriptor:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT id, content_type, original_name FROM file_data WHERE id = ?",
                (str(id),)
            ).fetchone()

        if row is None: raise FileNotFoundError(str(id))

        return FileDescriptor(
            id=id,
            original_name=row["original_name"],
            content_type=row["content_type"]
        )

    # Drops entire query if can't find any of specified items
    def get_multiple(self, ids: list[uuid.UUID]) -> list[FileDescriptor]:
        strs = [str(x) for x in ids]
        idstr = ",".join("?" * len(ids))

        with closing(self._conn()) as conn:
            rows = conn.execute(
                f"SELECT id, content_type, original_name FROM file_data WHERE id IN ({idstr})",
                (strs)
            ).fetchall()

        items = {
            uuid.UUID(row["id"]): row
            for row in rows
        }

        lost = list(dict.fromkeys(str(x) for x in ids if x not in items))
        if lost: raise FileNotFoundError(", ".join(lost))

        return [
            FileDescriptor(
                id=id,
                original_name=items[id]["original_name"],
                content_type=items[id]["content_type"]
            ) for id in ids
        ]
        
