import io
import tempfile
import unittest
import uuid
from pathlib import Path

from PIL import Image

from app import (
    ANIMATED_THUMBNAIL_CONTENT_TYPE,
    THUMBNAIL_CONTENT_TYPE,
    FailureKind,
    FileDescriptor,
    Limits,
    MediaProbe,
    StoredFile,
    Thumbnail,
)
from app.document_processor import DocumentProcessor
from infrastructure.animation import sample_animation
from infrastructure.embedding_providers import GemmaEmbeddings
from infrastructure.embedding_providers.configuration import Configuration as Gemma
from infrastructure.file_storage import Configuration as FilesConfiguration
from infrastructure.file_storage import LocalFileStorage
from infrastructure.media_probe import LocalMediaProbe, gif_stats
from infrastructure.thumbnails import Configuration as ThumbnailsConfiguration
from infrastructure.thumbnails import PillowThumbnailGenerator

from .fakes import FakeFiles, NoStorage, NoThumbnails, message

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def make_gif(
    path: Path, frames: int, delay_ms: int = 100, size=(80, 60), distinct=False, **extra
):
    # Encoders merge equal frames that follow each other, a test that counts frames needs different ones
    color = (
        (lambda i: (i * 11 % 256, i * 23 % 256, i * 47 % 256))
        if distinct
        else (lambda i: COLORS[i % len(COLORS)])
    )
    images = [Image.new("RGB", size, color(i)) for i in range(frames)]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=delay_ms,
        loop=0,
        **extra,
    )
    return path


# Pillow does not tell how long a frame of WebP lasts, it is in the ANMF chunks: 24 bits at byte 12
def webp_durations(data: bytes) -> list[int]:
    found, at = [], 0
    while (at := data.find(b"ANMF", at)) != -1:
        found.append(int.from_bytes(data[at + 8 + 12 : at + 8 + 15], "little"))
        at += 4
    return found


class GifStatsTest(unittest.TestCase):
    def test_blocks_are_counted_like_pillow_does(self):
        with tempfile.TemporaryDirectory() as d:
            for frames, delay in ((2, 100), (7, 40), (30, 20), (3, 500)):
                path = make_gif(Path(d, f"{frames}.gif"), frames, delay)
                with Image.open(path) as image:
                    expected = image.n_frames
                count, seconds = gif_stats(path)
                self.assertEqual(count, expected, (frames, delay))
                self.assertAlmostEqual(seconds, expected * delay / 1000, places=2)

    def test_a_tiny_delay_is_a_tenth_of_a_second_as_in_browsers(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_gif(Path(d, "fast.gif"), 5, 0)
            self.assertEqual(gif_stats(path), (5, 0.5))

    def test_local_color_tables_and_transparency_do_not_confuse_it(self):
        with tempfile.TemporaryDirectory() as d:
            # Every frame has its own palette
            frames = [
                Image.effect_noise((64, 64), 40 + i * 30).convert(
                    "P", palette=Image.ADAPTIVE
                )
                for i in range(4)
            ]
            path = Path(d, "palettes.gif")
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                transparency=0,
            )
            with Image.open(path) as image:
                expected = image.n_frames
            self.assertEqual(gif_stats(path)[0], expected)

    def test_damaged_files_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            junk = Path(d, "junk.gif")
            junk.write_bytes(b"this is not a gif" * 10)
            with self.assertRaises(ValueError):
                gif_stats(junk)
            empty = Path(d, "empty.gif")
            empty.write_bytes(b"")
            with self.assertRaises(ValueError):
                gif_stats(empty)
            # Only the header, no picture in it
            header = Path(d, "header.gif")
            header.write_bytes(b"GIF89a" + b"\x01\x00\x01\x00\x00\x00\x00" + b"\x3b")
            with self.assertRaises(ValueError):
                gif_stats(header)

    def test_the_probe_answers_for_gifs_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_gif(Path(d, "a.gif"), 4, 250)
            probe = LocalMediaProbe()
            self.assertEqual(probe.frames(path, "image/gif"), 4)
            self.assertAlmostEqual(probe.duration(path, "image/gif"), 1.0)
            self.assertIsNone(probe.frames(path, "image/png"))


class SampleTest(unittest.TestCase):
    def test_frames_are_spread_over_the_file_with_both_ends(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_gif(Path(d, "a.gif"), 10, 100)
            animation = sample_animation(path, 4)
            self.assertEqual(len(animation.frames), 4)
            self.assertEqual(animation.duration_ms, 1000)
            # Frame i is COLORS[i % 5]: 0, 3, 6, 9 are picked
            colors = [f.getpixel((0, 0)) for f in animation.frames]
            expected = [COLORS[i % 5] for i in (0, 3, 6, 9)]
            for got, want in zip(colors, expected):
                self.assertTrue(
                    all(abs(a - b) < 12 for a, b in zip(got, want)), (got, want)
                )

    def test_fewer_frames_than_asked_are_all_taken(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_gif(Path(d, "a.gif"), 3, 100)
            self.assertEqual(len(sample_animation(path, 8).frames), 3)

    def test_one_frame_is_not_an_animation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "still.gif")
            Image.new("RGB", (20, 20), (1, 2, 3)).save(path)
            self.assertIsNone(sample_animation(path, 8))

    def test_big_frames_are_shrunk(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_gif(Path(d, "big.gif"), 3, 100, size=(1000, 500))
            frames = sample_animation(path, 3, max_side=200).frames
            self.assertTrue(all(max(f.size) == 200 for f in frames))


class ThumbnailTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.generator = PillowThumbnailGenerator(
            ThumbnailsConfiguration(size=64, gif_frames=5)
        )

    def tearDown(self):
        self.dir.cleanup()

    def test_an_animated_gif_gets_a_moving_thumbnail(self):
        path = make_gif(
            Path(self.dir.name, "a.gif"), 20, 100, size=(300, 200), distinct=True
        )
        thumbnail = self.generator.generate(path, "image/gif")
        self.assertEqual(thumbnail.content_type, ANIMATED_THUMBNAIL_CONTENT_TYPE)
        with Image.open(io.BytesIO(thumbnail.data)) as image:
            self.assertEqual(image.format, "WEBP")
            self.assertEqual(image.n_frames, 5)
            self.assertEqual(max(image.size), 64)
            # 20 frames of 100 ms in 5 frames: each is shown for 400 ms
            self.assertEqual(webp_durations(thumbnail.data), [400] * 5)

    def test_a_gif_of_one_frame_is_a_picture(self):
        path = Path(self.dir.name, "still.gif")
        Image.new("RGB", (300, 200), (9, 9, 9)).save(path)
        thumbnail = self.generator.generate(path, "image/gif")
        self.assertEqual(thumbnail.content_type, THUMBNAIL_CONTENT_TYPE)
        self.assertEqual(Image.open(io.BytesIO(thumbnail.data)).format, "JPEG")

    def test_a_broken_gif_raises(self):
        path = Path(self.dir.name, "broken.gif")
        path.write_bytes(b"GIF89a junk")
        with self.assertRaises(Exception):
            self.generator.generate(path, "image/gif")

    def test_other_pictures_stay_jpeg(self):
        path = Path(self.dir.name, "p.png")
        Image.new("RGB", (300, 200), (9, 9, 9)).save(path)
        self.assertEqual(
            self.generator.generate(path, "image/png").content_type,
            THUMBNAIL_CONTENT_TYPE,
        )


class StorageTest(unittest.TestCase):
    def test_a_thumbnail_keeps_its_type_and_only_one_stays(self):
        with tempfile.TemporaryDirectory() as d:
            files = LocalFileStorage(
                FilesConfiguration(
                    root=str(Path(d, "files")),
                    thumbnails_root=str(Path(d, "thumbs")),
                    sqlite_connection_string=str(Path(d, "app.db")),
                )
            )
            id = uuid.uuid7()
            self.assertFalse(files.has_thumbnail(id))
            with self.assertRaises(FileNotFoundError):
                files.get_thumbnail(id)

            files.save_thumbnail(
                id, Thumbnail(data=b"jpeg", content_type=THUMBNAIL_CONTENT_TYPE)
            )
            self.assertEqual(
                files.get_thumbnail(id).content_type, THUMBNAIL_CONTENT_TYPE
            )

            # The first frame is replaced by a moving one
            files.save_thumbnail(
                id,
                Thumbnail(data=b"webp", content_type=ANIMATED_THUMBNAIL_CONTENT_TYPE),
            )
            got = files.get_thumbnail(id)
            self.assertEqual(
                (got.data, got.content_type), (b"webp", ANIMATED_THUMBNAIL_CONTENT_TYPE)
            )
            self.assertEqual(
                sorted(p.suffix for p in Path(d, "thumbs").iterdir()), [".webp"]
            )

            files.save(
                StoredFile(
                    id=id,
                    data=iter([b"x"]),
                    content_type="image/gif",
                    original_name="a.gif",
                )
            )
            files.remove(id)
            self.assertEqual(list(Path(d, "thumbs").iterdir()), [])


class LimitsTest(unittest.TestCase):
    def processor(self, limits, probe):
        files = FakeFiles()
        return DocumentProcessor(
            files, None, NoStorage(), NoThumbnails(), "safe", limits, probe
        ), files

    def test_a_gif_with_too_many_frames_is_refused_before_anything_is_built(self):
        class Probe(MediaProbe):
            def duration(self, path, content_type):
                return 1.0

            def frames(self, path, content_type):
                return 5000

        class Exploding(NoThumbnails):
            def generate(self, source, content_type):
                raise AssertionError("a refused file must not be decoded")

        p, files = self.processor(Limits(max_gif_frames=300), Probe())
        p.thumbnail_generator = Exploding()
        descriptors, _, fails = p.accept(iter([message("a.gif", 5, "image/gif")]))
        self.assertEqual(descriptors, [])
        self.assertEqual([f.kind for f in fails], [FailureKind.LIMIT_EXCEEDED])
        self.assertIn("5000 frames", fails[0].value)
        self.assertEqual(files.stored, {})

    def test_a_long_gif_is_refused_and_a_short_one_passes(self):
        class Probe(MediaProbe):
            def __init__(self, seconds):
                self.seconds = seconds

            def duration(self, path, content_type):
                return self.seconds

        for seconds, kept in ((45, 0), (4, 1)):
            p, files = self.processor(Limits(max_gif_seconds=30), Probe(seconds))
            descriptors, _, fails = p.accept(iter([message("a.gif", 5, "image/gif")]))
            self.assertEqual((len(descriptors), len(fails)), (kept, 1 - kept))

    def test_limits_of_gifs_do_not_touch_other_pictures(self):
        class Probe(MediaProbe):
            def duration(self, path, content_type):
                raise AssertionError("not asked")

        p, _ = self.processor(Limits(max_gif_seconds=1, max_gif_frames=1), Probe())
        descriptors, _, fails = p.accept(iter([message("a.png", 5, "image/png")]))
        self.assertEqual((len(descriptors), fails), (1, []))


class EmbeddingTest(unittest.TestCase):
    def provider(self, directory):
        # The model is not loaded, only the choice between a picture and a video is checked
        g = object.__new__(GemmaEmbeddings)
        g.config = Gemma(max_batch=1, gif_frames=3)
        files = FakeFiles()
        g.file_storage = files
        return g, files

    def test_an_animated_gif_goes_in_as_a_video_of_its_frames(self):
        with tempfile.TemporaryDirectory() as d:
            g, files = self.provider(d)
            animated = make_gif(Path(d, "a.gif"), 10, 100)
            files.get_full_path = lambda id: animated
            query = g._file_query(
                FileDescriptor(
                    id=uuid.uuid4(), original_name="a.gif", content_type="image/gif"
                )
            )
            self.assertEqual(set(query), {"text", "video"})
            self.assertEqual(len(query["video"]), 3)

    def test_a_still_gif_and_other_pictures_go_in_as_pictures(self):
        with tempfile.TemporaryDirectory() as d:
            g, files = self.provider(d)
            still = Path(d, "s.gif")
            Image.new("RGB", (10, 10)).save(still)
            png = Path(d, "p.png")
            Image.new("RGB", (10, 10)).save(png)
            for path, content_type in ((still, "image/gif"), (png, "image/png")):
                files.get_full_path = lambda id, p=path: p
                query = g._file_query(
                    FileDescriptor(
                        id=uuid.uuid4(),
                        original_name=path.name,
                        content_type=content_type,
                    )
                )
                self.assertEqual(set(query), {"text", "image"})
                self.assertEqual(query["image"], str(path))


if __name__ == "__main__":
    unittest.main()
