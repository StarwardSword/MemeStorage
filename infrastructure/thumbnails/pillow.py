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

from app import ThumbnailGenerator

from .config import Configuration

BACKGROUND = (255, 255, 255)

# Drawn once with Pillow, the size does not follow the config
PLACEHOLDERS = Path(__file__).with_name("placeholders")


# JPEG has no alpha, so transparent parts are laid on white instead of turning black
def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, BACKGROUND)
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


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

    @override
    def generate(self, source: Path, content_type: str) -> bytes | None:
        if content_type.startswith("image/"):
            return self._image(source)
        if content_type.startswith("video/"):
            return self._encode(video_frame(source))
        return None

    # Only media files are stored, anything unexpected gets the image placeholder
    @override
    def placeholder(self, content_type: str) -> bytes:
        kind = content_type.split("/", 1)[0]
        return self.placeholders.get(kind, self.placeholders["image"])
