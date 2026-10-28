import tempfile
import time
import unittest
import uuid
from pathlib import Path

from app import (
    ActivityConfiguration,
    ActivityRecorder,
    FailureKind,
    FailureReport,
    Job,
    JobStatus,
)
from app.job_worker import JobWorker
from infrastructure.jobs import SqliteJobStore


def job(name="a.png", description=None):
    return Job(id=uuid.uuid7(), original_name=name, description=description)


def wait_until_idle(store):
    deadline = time.time() + 5
    while store.active_count() and time.time() < deadline:
        time.sleep(0.05)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.dir.name, "jobs.db"))
        self.store = SqliteJobStore(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_claims_in_arrival_order_and_finishes(self):
        first, second = job("1"), job("2", "text")
        self.store.add([first, second])
        self.assertEqual(self.store.active_count(), 2)

        claimed = self.store.claim()
        self.assertEqual(claimed.id, first.id)
        self.assertEqual(self.store.get(first.id).status, JobStatus.RUNNING)
        self.assertEqual(self.store.claim().description, "text")
        self.assertIsNone(self.store.claim())

        self.store.finish(first.id, None)
        self.store.finish(second.id, (FailureKind.EMBEDDING_MODEL, "broken"))
        self.assertEqual(self.store.get(first.id).status, JobStatus.DONE)
        failed = self.store.get(second.id)
        self.assertEqual(
            (failed.status, failed.error_kind, failed.error_message),
            (JobStatus.FAILED, FailureKind.EMBEDDING_MODEL, "broken"),
        )
        self.assertEqual(self.store.active_count(), 0)

    def test_running_jobs_are_put_back_after_restart(self):
        j = job()
        self.store.add([j])
        self.store.claim()
        reopened = SqliteJobStore(self.path)
        self.assertEqual(reopened.requeue_running(), 1)
        self.assertEqual(reopened.get(j.id).status, JobStatus.PENDING)
        self.assertIsNone(reopened.get(uuid.uuid7()))


class Descriptor:
    def __init__(self, j):
        self.id = j.id
        self.original_name = j.original_name
        self.content_type = "image/png"


class FakeFiles:
    def __init__(self, jobs):
        self.jobs = {j.id: j for j in jobs}

    def descriptor(self, id):
        return Descriptor(self.jobs[id])


class FakeProcessor:
    def __init__(self):
        self.indexed = []

    def index(self, descriptors, messages):
        self.indexed.append(messages[0].text_description)
        if descriptors[0].original_name == "bad.png":
            return [
                FailureReport(
                    item_id=descriptors[0].id,
                    kind=FailureKind.VECTOR_STORAGE,
                    value="bad.png",
                )
            ]
        return []

    def rollback(self, fails):
        pass


class FakeStorage:
    def __init__(self):
        self.removed = []

    def remove(self, ids):
        self.removed += ids


class WorkerTest(unittest.TestCase):
    def test_worker_indexes_queued_jobs_and_records_failures(self):
        with tempfile.TemporaryDirectory() as d:
            store = SqliteJobStore(str(Path(d, "jobs.db")))
            jobs = [job("ok.png", "first"), job("bad.png", "second")]
            files, processor, storage = FakeFiles(jobs), FakeProcessor(), FakeStorage()
            store.add(jobs)

            worker = JobWorker(
                store,
                processor,
                files,
                storage,
                ActivityRecorder(None, ActivityConfiguration()),
            )
            worker.start()
            try:
                wait_until_idle(store)

                self.assertEqual(processor.indexed, ["first", "second"])
                self.assertEqual(store.get(jobs[0].id).status, JobStatus.DONE)
                bad = store.get(jobs[1].id)
                self.assertEqual(
                    (bad.status, bad.error_kind),
                    (JobStatus.FAILED, FailureKind.VECTOR_STORAGE),
                )
                # A job that was interrupted may be in the vector storage already
                self.assertEqual(storage.removed, [j.id for j in jobs])

                # A job added while the worker sleeps is picked up at once
                late = job("late.png", "third")
                files.jobs[late.id] = late
                store.add([late])
                worker.wake()
                wait_until_idle(store)
                self.assertEqual(store.get(late.id).status, JobStatus.DONE)
            finally:
                worker.stop()


if __name__ == "__main__":
    unittest.main()
