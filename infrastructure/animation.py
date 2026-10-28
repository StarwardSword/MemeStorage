from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BACKGROUND = (255, 255, 255)


# JPEG has no alpha, so transparent parts are laid on white instead of turning black
def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, BACKGROUND)
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


# Browsers show a frame that asks for 10 ms or less for 100 ms, so does this
def delay_ms(raw: int | None) -> int:
    return raw if raw and raw > 10 else 100


@dataclass
class Animation:
    # RGB, evenly spread over the file with the first and the last one among them
    frames: list[Image.Image]
    # Of the whole file, not of the frames picked
    duration_ms: int


def sample_animation(
    path: Path, count: int, max_side: int | None = None
) -> Animation | None:
    """At most count frames of an animated image, None when the file has one frame.

    Decodes every frame once, how many of them there are is limited by limits.max_gif_frames"""
    with Image.open(path) as image:
        total = getattr(image, "n_frames", 1)
        if total <= 1:
            return None

        wanted = {round(x) for x in np.linspace(0, total - 1, min(count, total))}
        frames: list[Image.Image] = []
        duration = 0
        for i in range(total):
            image.seek(i)
            duration += delay_ms(image.info.get("duration"))
            if i in wanted:
                frame = to_rgb(image)
                if max_side:
                    frame.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                frames.append(frame)

    return Animation(frames, duration)
