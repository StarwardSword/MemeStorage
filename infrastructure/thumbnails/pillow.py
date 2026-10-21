from io import BytesIO
from pathlib import Path
from typing import override

from PIL import Image, ImageOps

from app import ThumbnailGenerator

from .config import Configuration

BACKGROUND = (255, 255, 255)


# JPEG has no alpha, so transparent parts are laid on white instead of turning black
def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, BACKGROUND)
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


class PillowThumbnailGenerator(ThumbnailGenerator):
    def __init__(self, config: Configuration) -> None:
        self.size = config.size
        self.quality = config.quality

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
        return None
