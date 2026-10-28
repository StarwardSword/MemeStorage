import tempfile
import threading
import unittest
import uuid

import numpy as np

from app import EmbeddingsForbiddenError, Job, QueueFullError
from app.job_worker import floats

from .fakes import make_app, message


class StrictQueueTest(unittest.TestCase):
    def test_uploads_at_the_same_time_can_not_go_over_the_limit(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d, max_queue=3)
            start = threading.Barrier(12)
            accepted, refused = [], []

            def upload(i):
                start.wait()
                try:
                    accepted.append(app.submit(message(f"{i}.png", 10)))
                except QueueFullError:
                    refused.append(i)

            threads = [threading.Thread(target=upload, args=(i,)) for i in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual((len(accepted), len(refused)), (3, 9))
            self.assertEqual(app.jobs.active_count(), 3)
            self.assertEqual(app._reserved, 0)

    def test_batches_at_the_same_time_share_the_limit(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d, max_queue=4)
            start = threading.Barrier(3)
            results = []

            def upload():
                start.wait()
                jobs, fails = app.submit_multiple(
                    iter(message(f"{i}.png", 10) for i in range(3))
                )
                results.append((len(jobs), len(fails)))

            threads = [threading.Thread(target=upload) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(sum(j for j, _ in results), 4)
            self.assertEqual(sum(f for _, f in results), 5)
            self.assertEqual(app.jobs.active_count(), 4)

    def test_a_refused_file_gives_its_place_back(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d, max_queue=1, max_image_mb=1)
            jobs, fails = app.submit_multiple(
                iter([message("big.png", 2 * 1024 * 1024)])
            )
            self.assertEqual((len(jobs), len(fails)), (0, 1))
            self.assertEqual(app._reserved, 0)
            self.assertIsInstance(app.submit(message("ok.png", 10)), Job)


class EmbeddingsPermissionTest(unittest.TestCase):
    def test_only_trusted_users_and_the_admin_may_send_embeddings(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            _, plain = app.create_user("plain", allow_embeddings=False)
            _, trusted = app.create_user("trusted", allow_embeddings=True)

            app.check_embeddings(app.authenticate(trusted))
            with self.assertRaises(EmbeddingsForbiddenError):
                app.check_embeddings(app.authenticate(plain))

            from .fakes import ADMIN_TOKEN

            app.check_embeddings(app.authenticate(ADMIN_TOKEN))

    def test_embeddings_wait_in_the_queue_as_they_were_sent(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            vector = np.arange(4, dtype=np.float32)
            job = app.submit(
                message("a.png", 10, embeddings=vector, text_description="cat")
            )
            stored = app.get_job(job.id)
            np.testing.assert_array_equal(floats(stored.embeddings), vector)
            self.assertIsNone(floats(stored.text_embeddings))
            self.assertEqual(stored.description, "cat")

    def test_job_ids_are_file_ids(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            job = app.submit(message("a.png", 10))
            self.assertIn(job.id, app.file_storage.stored)
            self.assertIsNone(app.get_job(uuid.uuid7()))


if __name__ == "__main__":
    unittest.main()
