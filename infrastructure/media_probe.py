from pathlib import Path
from typing import BinaryIO, override

import librosa

from app import MediaProbe
from app.media import is_gif


def skip_blocks(f: BinaryIO) -> bytes:
    """Reads the data blocks of GIF up to the terminator, returns the first one"""
    first = None
    while size := f.read(1):
        if size[0] == 0:
            break
        if first is None:
            first = f.read(size[0])
        else:
            f.seek(size[0], 1)
    return first or b""


def color_table(flags: int) -> int:
    return 3 * (2 << (flags & 7)) if flags & 0x80 else 0


def gif_stats(path: Path) -> tuple[int, float]:
    """Frames and seconds of a GIF, read from its blocks without decoding a single picture"""
    frames = 0
    centiseconds = 0
    delay = 10
    with open(path, "rb") as f:
        head = f.read(13)
        if head[:3] != b"GIF":
            raise ValueError("not a GIF")
        f.seek(color_table(head[10]), 1)

        while block := f.read(1):
            if block == b"\x3b":
                break
            if block == b"\x21":
                label = f.read(1)
                data = skip_blocks(f)
                if label == b"\xf9" and len(data) >= 3:
                    delay = int.from_bytes(data[1:3], "little")
            elif block == b"\x2c":
                frames += 1
                centiseconds += delay if delay > 1 else 10
                delay = 10
                f.seek(color_table(f.read(9)[8]) + 1, 1)
                skip_blocks(f)
            else:
                raise ValueError("broken GIF")

    if frames == 0:
        raise ValueError("GIF has no frames")
    return frames, centiseconds / 100


class LocalMediaProbe(MediaProbe):
    @override
    def duration(self, path: Path, content_type: str) -> float | None:
        if is_gif(content_type):
            return gif_stats(path)[1]
        if content_type.startswith("video/"):
            # Imported here: main.py adds the FFmpeg directory before the first import of torchcodec
            from torchcodec.decoders import VideoDecoder

            return VideoDecoder(str(path)).metadata.duration_seconds
        if content_type.startswith("audio/"):
            return float(librosa.get_duration(path=str(path)))
        return None

    @override
    def frames(self, path: Path, content_type: str) -> int | None:
        return gif_stats(path)[0] if is_gif(content_type) else None
