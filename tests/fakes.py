import time
from pathlib import Path

from app import (
    ActivityConfiguration,
    ActivityRecorder,
    Application,
    ApplicationConfiguration,
    AuthConfiguration,
    Limits,
)
from app.document_processor import SaveMessage
from infrastructure.activity import SqliteActivityStore
from infrastructure.jobs import SqliteJobStore
from infrastructure.users import SqliteUserStore

ADMIN_TOKEN = "admin-token-for-the-tests-0123456789"


class FakeFiles:
    def __init__(self, delay=0.0):
        self.stored = {}
        self.delay = delay

    def save(self, file):
        # Reads the stream like the real storage, a limit error then leaves nothing behind
        data = b"".join(file.data)
        time.sleep(self.delay)
        self.stored[file.id] = data
        return Path(str(file.id))

    def remove(self, id):
        self.stored.pop(id, None)

    def get_full_path(self, id):
        return Path(str(id))

    def usage(self):
        return []

    def list_files(self, owner_id, limit, offset):
        return []


class NoStorage:
    def remove(self, ids):
        pass


def message(name, size, content_type="image/png", **fields):
    return SaveMessage(
        original_name=name,
        data=iter([b"x" * size]),
        content_type=content_type,
        **fields,
    )


def make_app(directory, files=None, activity=False, **limits) -> Application:
    """Application on real SQLite stores in directory, without a model"""
    db = str(Path(directory, "app.db"))
    return Application(
        embedding_provider=None,
        storage=NoStorage(),
        file_storage=files or FakeFiles(delay=0.05),
        configuration=ApplicationConfiguration(save_strategy="safe"),
        jobs=SqliteJobStore(db),
        users=SqliteUserStore(db),
        auth=AuthConfiguration(admin_token=ADMIN_TOKEN),
        limits=Limits(**limits),
        activity=ActivityRecorder(SqliteActivityStore(db), ActivityConfiguration())
        if activity
        else None,
    )
