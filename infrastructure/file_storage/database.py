import uuid
import sqlite3
from contextlib import closing
from app import FileDescriptor

class SqliteDatabase():
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string)
        conn.row_factory = sqlite3.Row
        return conn

    def __init__(self, connection_string: str) -> None:
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

        return FileDescriptor(
            id=id,
            original_name=row["original_name"],
            content_type=row["content_type"]
        )
    
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

        return [
            FileDescriptor(
                id=id,
                original_name=items[id]["original_name"],
                content_type=items[id]["content_type"]
            ) for id in ids
        ]
        
