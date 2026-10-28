from io import BytesIO
from pathlib import Path
from typing import override

from PIL import Image, ImageOps
from torchcodec import Frame
from torchcodec.decoders import (
    ColorConverter,
    Demuxer,
    VideoDecoder,
    get_container_metadata,
)

from app import (
    ANIMATED_THUMBNAIL_CONTENT_TYPE,
    THUMBNAIL_CONTENT_TYPE,
    Thumbnail,
    ThumbnailGenerator,
)
from app.media import is_gif

from ..animation import sample_animation, to_rgb
from .config import Configuration

# Drawn once with Pillow, the size does not follow the config
PLACEHOLDERS = Path(__file__).with_name("placeholders")


# VideoDecoder can not be created without a duration, and some streams can not seek at all,
# so the first frame is read packet by packet
def first_frame(source: Path) -> Frame:
    demuxer = Demuxer(source)
    decoder = demuxer.streams[0].make_decoder(device="cpu")
    converter = ColorConverter(device="cpu")

    for packet in demuxer:
        for raw in decoder.decode(packet):
            return converter.convert(raw)
    for raw in decoder.drain():
        return converter.convert(raw)

    raise ValueError("video has no frames")


# Frame in the middle of the video, or the first one when the header has no duration.
# Approximate seeking reads only the header instead of scanning the whole file
def video_frame(source: Path) -> Image.Image:
    container = get_container_metadata(source)
    if container.best_video_stream_index is None:
        raise ValueError("file has no video stream")

    stream = container.streams[container.best_video_stream_index]
    duration = (
        stream.duration_seconds_from_header or container.duration_seconds_from_header
    )

    if duration is None:
        frame = first_frame(source)
    else:
        decoder = VideoDecoder(source, seek_mode="approximate")
        begin = decoder.metadata.begin_stream_seconds or 0
        frame = decoder.get_frame_played_at(begin + duration / 2)

    # Frames are [3, height, width], Pillow wants channels last
    return Image.fromarray(frame.data.permute(1, 2, 0).numpy())


class PillowThumbnailGenerator(ThumbnailGenerator):
    def __init__(self, config: Configuration) -> None:
        self.size = config.size
        self.quality = config.quality
        self.gif_frames = config.gif_frames
        self.placeholders = {
            kind: Path(PLACEHOLDERS, f"{kind}.jpg").read_bytes()
            for kind in ("image", "video", "audio")
        }

    def _encode(self, image: Image.Image) -> bytes:
        image = to_rgb(image)
        # Keeps aspect ratio and never enlarges small images
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=self.quality)
        return buffer.getvalue()

    def _image(self, source: Path) -> bytes:
        with Image.open(source) as image:
            return self._encode(ImageOps.exif_transpose(image))

    # Some of the frames, spread evenly, shown for as long as the whole GIF lasts.
    # None for a GIF of one frame, it is a picture like any other
    def _animated(self, source: Path) -> bytes | None:
        animation = sample_animation(source, self.gif_frames, self.size)
        if animation is None:
            return None

        buffer = BytesIO()
        first, *rest = animation.frames
        first.save(
            buffer,
            "WEBP",
            save_all=True,
            append_images=rest,
            duration=max(20, animation.duration_ms // len(animation.frames)),
            loop=0,
            quality=self.quality,
        )
        return buffer.getvalue()

    @override
    def generate(self, source: Path, content_type: str) -> Thumbnail | None:
        if is_gif(content_type):
            animated = self._animated(source)
            if animated is not None:
                return Thumbnail(
                    data=animated, content_type=ANIMATED_THUMBNAIL_CONTENT_TYPE
                )
        if content_type.startswith("image/"):
            return Thumbnail(
                data=self._image(source), content_type=THUMBNAIL_CONTENT_TYPE
            )
        if content_type.startswith("video/"):
            return Thumbnail(
                data=self._encode(video_frame(source)),
                content_type=THUMBNAIL_CONTENT_TYPE,
            )
        return None

    # Only media files are stored, anything unexpected gets the image placeholder
    @override
    def placeholder(self, content_type: str) -> Thumbnail:
        kind = content_type.split("/", 1)[0]
        return Thumbnail(
            data=self.placeholders.get(kind, self.placeholders["image"]),
            content_type=THUMBNAIL_CONTENT_TYPE,
        )
