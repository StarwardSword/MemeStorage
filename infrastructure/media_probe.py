from pathlib import Path
from typing import override

import librosa

from app import MediaProbe


class LocalMediaProbe(MediaProbe):
    @override
    def duration(self, path: Path, content_type: str) -> float | None:
        if content_type.startswith("video/"):
            # Imported here: main.py adds the FFmpeg directory before the first import of torchcodec
            from torchcodec.decoders import VideoDecoder

            return VideoDecoder(str(path)).metadata.duration_seconds
        if content_type.startswith("audio/"):
            return float(librosa.get_duration(path=str(path)))
        return None
