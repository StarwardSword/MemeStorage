"""
Builds thumbnails for files stored before thumbnails appeared

Usage: python build_thumbnails.py

Reads config.yaml from the current directory, the same as the server, and may run
while the server is running. Nothing is removed: a file a thumbnail can not be built
for is printed and stays in the storage
"""

import os
from pathlib import Path

# Same as in main.py: torchcodec looks for FFmpeg DLLs once, when it is imported
FFMPEG_DIR = Path(__file__).parent / "ffmpeg"
if hasattr(os, "add_dll_directory") and FFMPEG_DIR.is_dir():
    ffmpeg_dlls = os.add_dll_directory(str(FFMPEG_DIR))

from configure import Configuration
from infrastructure import LocalFileStorage, PillowThumbnailGenerator


def main() -> None:
    config = Configuration()
    files = LocalFileStorage(config.files)
    generator = PillowThumbnailGenerator(config.thumbnails)

    built = skipped = failed = 0

    for d in files.all_descriptors():
        if files.has_thumbnail(d.id):
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
