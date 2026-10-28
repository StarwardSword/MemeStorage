"""
Builds thumbnails for files stored before thumbnails appeared

Usage: python build_thumbnails.py [--gifs]

--gifs also rebuilds the thumbnails of GIFs that have one: those made before GIFs got moving
thumbnails show only the first frame

Reads config.yaml from the current directory, the same as the server, and may run
while the server is running. Nothing is removed: a file a thumbnail can not be built
for is printed and stays in the storage
"""

import os
import sys
from pathlib import Path

# Same as in main.py: torchcodec looks for FFmpeg DLLs once, when it is imported
FFMPEG_DIR = Path(__file__).parent / "ffmpeg"
if hasattr(os, "add_dll_directory") and FFMPEG_DIR.is_dir():
    ffmpeg_dlls = os.add_dll_directory(str(FFMPEG_DIR))

from app import ANIMATED_THUMBNAIL_CONTENT_TYPE
from app.media import is_gif
from configure import Configuration
from infrastructure import LocalFileStorage, PillowThumbnailGenerator


def main() -> None:
    config = Configuration()
    files = LocalFileStorage(config.files)
    generator = PillowThumbnailGenerator(config.thumbnails)

    rebuild_gifs = "--gifs" in sys.argv[1:]
    built = skipped = failed = 0

    for d in files.all_descriptors():
        if files.has_thumbnail(d.id):
            # A GIF that already moves is done, one that shows its first frame is rebuilt on request
            still = (
                files.get_thumbnail(d.id).content_type
                != ANIMATED_THUMBNAIL_CONTENT_TYPE
            )
            if not (rebuild_gifs and is_gif(d.content_type) and still):
                skipped += 1
                continue

        try:
            thumbnail = generator.generate(files.get_full_path(d.id), d.content_type)
        except Exception as e:
            print(f"{d.id} {d.original_name}: {type(e).__name__}: {e}")
            failed += 1
            continue

        # Audio has no own thumbnail, the server shows a placeholder for it
        if thumbnail is None:
            skipped += 1
            continue

        files.save_thumbnail(d.id, thumbnail)
        built += 1

    print(f"Built {built}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
