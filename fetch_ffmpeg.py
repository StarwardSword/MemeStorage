import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

# LGPL shared build of FFmpeg 9; month-end autobuilds stay on GitHub for about two years
RELEASE = "autobuild-2026-09-30-13-08"
ARCHIVE = "ffmpeg-n9.0.2-17-g2a571b6068-win64-lgpl-shared-9.0.zip"
SHA256 = "7157177b8a6cb2174c1650ba8c71b363f2c78cba5330f88c4c02cf5b2b880646"
URL = f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{RELEASE}/{ARCHIVE}"

TARGET = Path(__file__).parent / "ffmpeg"
CHUNK = 1 << 20


def download(url: str, file) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:
        while chunk := response.read(CHUNK):
            digest.update(chunk)
            file.write(chunk)
    return digest.hexdigest()


def extract(archive: zipfile.ZipFile, target: Path) -> list[str]:
    extracted = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        is_dll = path.parent.name == "bin" and path.suffix == ".dll"
        is_license = path.name == "LICENSE.txt" and len(path.parts) == 2
        if not (is_dll or is_license):
            continue
        with archive.open(member) as src, open(target / path.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted.append(path.name)
    return extracted


def main():
    print(f"Downloading {URL}")
    with tempfile.TemporaryFile() as file:
        actual = download(URL, file)
        if actual != SHA256:
            sys.exit(f"Checksum mismatch: expected {SHA256}, got {actual}")

        # DLLs from different FFmpeg builds must not be mixed
        shutil.rmtree(TARGET, ignore_errors=True)
        TARGET.mkdir()

        file.seek(0)
        with zipfile.ZipFile(file) as archive:
            extracted = extract(archive, TARGET)

    print(f"Extracted to {TARGET}: {', '.join(extracted)}")


if __name__ == "__main__":
    main()
