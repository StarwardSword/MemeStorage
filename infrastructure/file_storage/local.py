import os.path
import uuid
from io import BufferedReader
from os import stat
from pathlib import Path
from typing import Generator, override

from app import FileDescriptor, FileRecord, FileStorage, OwnerUsage, StoredFile

from .config import Configuration
from .database import SqliteDatabase


class LocalFileStorage(FileStorage):
    def __init__(self, config: Configuration) -> None:
        for directory in (config.root, config.thumbnails_root):
            if os.path.exists(directory) and os.path.isfile(directory):
                raise AttributeError(directory, "must be a directory")
            # if not os.path.isabs(root): root = Path(os.path.abspath(root, os.curdir))
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

        self.root: Path = Path(config.root)
        self.thumbnails_root: Path = Path(config.thumbnails_root)
        self.chunk_size: int = config.chunk_size
        self.database = SqliteDatabase(config.sqlite_connection_string)

    @override
    def save(self, file: StoredFile) -> Path:
        path = Path(self.root, str(file.id))
        try:
            size = 0
            with open(path, "wb") as f:
                for d in file.data:
                    f.write(d)
                    size += len(d)
            # Seen by others only when the file is complete
            self.database.save(
                FileDescriptor(
                    id=file.id,
                    original_name=file.original_name,
                    content_type=file.content_type,
                ),
                file.owner_id,
                size,
            )
        except Exception:
            self.database.remove(file.id)

            try:
                os.remove(path)
            except FileNotFoundError:
                pass

            raise

        return path

    @override
    def get(self, id: uuid.UUID) -> StoredFile:
        meta = self.database.get(id)

        filename = Path(self.root, str(id))
        # Opened here so a missing file fails before the response starts,
        # the generator below owns and closes it once iteration begins
        f = open(filename, "rb")  # noqa: SIM115
        try:
            st = stat(f.fileno())
        except BaseException:
            f.close()
            raise

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
            id=id, original_name=meta.original_name, content_type=meta.content_type
        )

    @override
    def descriptors(self, ids: list[uuid.UUID]) -> list[FileDescriptor]:
        return self.database.get_multiple(ids)

    @override
    def all_descriptors(self) -> list[FileDescriptor]:
        return self.database.get_all()

    @override
    def usage(self) -> list[OwnerUsage]:
        return self.database.usage()

    @override
    def list_files(
        self, owner_id: uuid.UUID | None, limit: int, offset: int
    ) -> list[FileRecord]:
        return self.database.list_files(owner_id, limit, offset)

    @override
    def remove(self, id: uuid.UUID):
        for filename in (Path(self.root, str(id)), self._thumbnail_path(id)):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
        self.database.remove(id)

    # Thumbnail exists when its file does, there is no record of it in the database
    def _thumbnail_path(self, id: uuid.UUID) -> Path:
        return Path(self.thumbnails_root, f"{id}.jpg")

    # Written aside and moved in place, so a thumbnail requested while the backfill script
    # writes it is either missing or whole
    @override
    def save_thumbnail(self, id: uuid.UUID, data: bytes) -> None:
        path = self._thumbnail_path(id)
        # Server and the script are different processes and may write the same thumbnail
        temporary = Path(self.thumbnails_root, f"{id}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass
            raise

    @override
    def has_thumbnail(self, id: uuid.UUID) -> bool:
        return self._thumbnail_path(id).is_file()

    @override
    def get_thumbnail(self, id: uuid.UUID) -> bytes:
        return self._thumbnail_path(id).read_bytes()

    @override
    def get_full_path(self, id: uuid.UUID) -> Path:
        return Path(os.path.abspath(Path(self.root, str(id))))
