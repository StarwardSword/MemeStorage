import unittest

from app import FailureKind, Limits, MediaProbe
from app.document_processor import DocumentProcessor
from app.limits import LimitExceededError, limited

from .fakes import FakeFiles, NoStorage, message


class FakeProbe(MediaProbe):
    def __init__(self, seconds):
        self.seconds = seconds

    def duration(self, path, content_type):
        return self.seconds


def processor(limits, probe=None):
    files = FakeFiles()
    return DocumentProcessor(files, None, NoStorage(), "safe", limits, probe), files


class LimitsTest(unittest.TestCase):
    def test_zero_and_missing_mean_no_limit(self):
        limits = Limits(max_video_mb=0)
        self.assertIsNone(limits.max_bytes("video/mp4"))
        self.assertIsNone(limits.max_bytes("image/png"))

    def test_limits_follow_the_type(self):
        limits = Limits(max_image_mb=1, max_video_seconds=5)
        self.assertEqual(limits.max_bytes("image/png"), 1024 * 1024)
        self.assertIsNone(limits.max_bytes("audio/wav"))
        self.assertEqual(limits.max_seconds("video/mp4"), 5)
        self.assertIsNone(limits.max_seconds("image/png"))

    def test_limited_stops_in_the_middle_of_a_stream(self):
        chunks = iter([b"a" * 600_000, b"b" * 600_000])
        with self.assertRaises(LimitExceededError):
            list(limited(chunks, 1024 * 1024))
        self.assertEqual(list(limited(iter([b"abc"]), None)), [b"abc"])


class AcceptTest(unittest.TestCase):
    def test_too_big_file_is_refused_and_removed_others_stay(self):
        p, files = processor(Limits(max_image_mb=1))
        descriptors, _, fails = p.accept(
            iter([message("big.png", 2 * 1024 * 1024), message("ok.png", 10)])
        )
        self.assertEqual([d.original_name for d in descriptors], ["ok.png"])
        self.assertEqual([f.kind for f in fails], [FailureKind.LIMIT_EXCEEDED])
        self.assertIn("big.png", fails[0].value)
        self.assertEqual(len(files.stored), 1)

    def test_batch_limit_refuses_the_rest(self):
        p, files = processor(Limits(max_batch_files=2))
        descriptors, _, fails = p.accept(iter(message(f"{i}.png", 1) for i in range(4)))
        self.assertEqual(len(descriptors), 2)
        self.assertEqual(len(fails), 2)
        self.assertEqual(len(files.stored), 2)

    def test_room_in_the_queue_refuses_the_rest(self):
        p, _ = processor(Limits())
        places = iter([True, False, False])
        descriptors, _, fails = p.accept(
            iter(message(f"{i}.png", 1) for i in range(3)), lambda: next(places)
        )
        self.assertEqual(len(descriptors), 1)
        self.assertTrue(all("queue is full" in f.value for f in fails))

    def test_long_video_is_rejected_before_the_model(self):
        p, files = processor(Limits(max_video_seconds=10), FakeProbe(60))
        descriptors, messages, _ = p.accept(iter([message("v.mp4", 5, "video/mp4")]))
        fails = p.index(descriptors, messages)
        self.assertEqual([f.kind for f in fails], [FailureKind.LIMIT_EXCEEDED])
        self.assertIn("60 s", fails[0].value)
        self.assertEqual(files.stored, {})

    def test_unreadable_media_is_reported(self):
        class Broken(MediaProbe):
            def duration(self, path, content_type):
                raise OSError("bad header")

        p, _ = processor(Limits(max_audio_seconds=10), Broken())
        descriptors, messages, _ = p.accept(iter([message("a.wav", 5, "audio/wav")]))
        fails = p.index(descriptors, messages)
        self.assertEqual([f.kind for f in fails], [FailureKind.EMBEDDING_MODEL])
        self.assertIn("bad header", fails[0].value)


if __name__ == "__main__":
    unittest.main()
