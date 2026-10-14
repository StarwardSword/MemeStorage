import os.path
from os import stat
import uuid
from io import BufferedReader
from typing import Generator, override
from pathlib import Path
from app import FileStorage, StoredFile, FileDescriptor
from .database import SqliteDatabase

class LocalFileStorage(FileStorage):
    def __init__(self, root: Path, sqlite_connection_string: str, chunk_size: int = 64 * 1024) -> None:
        if os.path.exists(root) and os.path.isfile(root): raise AttributeError(root, "must be a directory")
        #if not os.path.isabs(root): root = Path(os.path.abspath(root, os.curdir))
        if not os.path.exists(root): os.mkdir(root)
        
        self.root: Path = root
        self.chunk_size: int = chunk_size
        self.database = SqliteDatabase(sqlite_connection_string)

    @override
    def save(self, file: StoredFile) -> Path:
        self.database.save(FileDescriptor(
            id=file.id,
            original_name=file.original_name,
            content_type=file.content_type
        ))

        try:
            path = Path(self.root, str(file.id))
            with open(path, "wb") as f:
                for d in file.data: f.write(d)
        except Exception:
            self.database.remove(file.id)
            raise

        return path

    @override
    def get(self, id: uuid.UUID) -> StoredFile: 
        meta = self.database.get(id)

        filename = Path(self.root, str(id))
        f = open(filename, "rb")
        st = stat(f.fileno())

        return StoredFile(
            id=id,
            data=self._iter_chunks(f),
            content_type=meta.content_type,
            original_name=meta.original_name,
            size=st.st_size,
        )

    def _iter_chunks(self, f: BufferedReader) -> Generator[bytes, None, None]:
        with f:
            while chunk := f.read(self.chunk_size):
                yield chunk

    @override
    def descriptor(self, id: uuid.UUID) -> FileDescriptor:
        meta = self.database.get(id)
        return FileDescriptor(
            id=id,
            original_name=meta.original_name,
            content_type=meta.content_type
        )

    @override
    def descriptors(self, ids: list[uuid.UUID]) -> list[FileDescriptor]:
        return self.database.get_multiple(ids)

    @override
    def remove(self, id: uuid.UUID): 
        filename = Path(self.root, str(id))
        os.remove(filename)
        self.database.remove(id)

        
    @override
    def get_full_path(self, id: uuid.UUID) -> Path:
        return Path(
            os.path.abspath(Path(self.root, str(id)))
        )