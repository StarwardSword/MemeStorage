import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from mimetypes import guess_type
from pathlib import Path
from typing import Iterator

import grpc
from api.grpc.proto import service_pb2, service_pb2_grpc

DEFAULT_SERVER = "localhost:50051"
CHUNK_SIZE = 64 * 1024
TIMEOUT = 300
DEFAULT_JOBS = 4
MEDIA_PREFIXES = ("image/", "video/", "audio/")


class ProtocolError(Exception):
    pass


def connect(server: str) -> service_pb2_grpc.MemeStorageStub:
    return service_pb2_grpc.MemeStorageStub(grpc.insecure_channel(server))


# --- upload

def collect(target: Path, recursive: bool) -> list[Path]:
    if target.is_file(): return [target]
    if not target.is_dir(): raise FileNotFoundError(target)

    candidates = target.rglob("*") if recursive else target.iterdir()
    return sorted(p for p in candidates if p.is_file())


# Server stores the file before embedding it, so unsupported types would leave orphans there
def is_media(path: Path) -> bool:
    content_type = guess_type(path.name)[0]
    return content_type is not None and content_type.startswith(MEDIA_PREFIXES)


def save_requests(path: Path, description: str) -> Iterator[service_pb2.SaveRequest]:
    info = service_pb2.SaveFileInfo(filename=path.name)
    if description: info.text_description = description
    yield service_pb2.SaveRequest(info=info)

    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            yield service_pb2.SaveRequest(data=chunk)


def upload(stub: service_pb2_grpc.MemeStorageStub, target: Path, description: str, recursive: bool, jobs: int) -> int:
    files = collect(target, recursive)
    if not files:
        print(f"No files in {target}")
        return 0

    media: list[Path] = []
    skipped = 0
    for path in files:
        if is_media(path):
            media.append(path)
            continue
        print(f"SKIP  {path}: unsupported type ({guess_type(path.name)[0]})")
        skipped += 1

    # Queued uploads check the flag instead of being cancelled: as_completed never sees futures
    # cancelled by executor.shutdown(cancel_futures=True) and would hang
    stop = threading.Event()

    def upload_one(path: Path) -> str | None:
        if stop.is_set(): return None
        return stub.Save(save_requests(path, description), timeout=TIMEOUT).uuid

    # Channel is thread-safe, so every upload shares it; results are printed in completion order
    uploaded, failed, not_sent = 0, 0, 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(upload_one, path): path for path in media}
        try:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    file_id = future.result()
                except grpc.RpcError as e:
                    print(f"FAIL  {path}: {e.code().name}: {e.details()}")
                    failed += 1
                    # No point in sending the queued files to a server that is down
                    if e.code() == grpc.StatusCode.UNAVAILABLE: stop.set()
                    continue

                if file_id is None:
                    not_sent += 1
                    continue
                print(f"OK    {file_id}  {path}")
                uploaded += 1
        except KeyboardInterrupt:
            # Uploads in flight still finish when the executor exits, queued ones are dropped
            stop.set()
            raise

    if len(files) > 1:
        print(f"\nUploaded: {uploaded}, skipped: {skipped}, failed: {failed}" + (f", not sent: {not_sent}" if not_sent else ""))
    return 1 if failed else 0


# --- download

def unique_path(directory: Path, filename: str) -> Path:
    # Filename comes from the server, never let it escape the output directory
    name = Path(filename).name
    if name in ("", ".", ".."): name = "file"

    target = directory / name
    i = 1
    while target.exists():
        target = directory / f"{Path(name).stem} ({i}){Path(name).suffix}"
        i += 1
    return target


# Streams file into the directory, returns its path and server-side info
def download(stub: service_pb2_grpc.MemeStorageStub, file_id: str, output: Path) -> tuple[Path, service_pb2.DownloadFileInfo]:
    responses = stub.GetFile(service_pb2.UUIDRequest(uuid=file_id), timeout=TIMEOUT)

    first = next(responses, None)
    if first is None or first.WhichOneof("payload") != "info":
        raise ProtocolError("first message must be file info")
    info = first.info

    output.mkdir(parents=True, exist_ok=True)
    target = unique_path(output, info.original_name)
    partial = target.with_name(target.name + ".part")

    try:
        written = 0
        with open(partial, "xb") as f:
            for response in responses:
                if response.WhichOneof("payload") != "chunk": raise ProtocolError("expected data chunk")
                f.write(response.chunk)
                written += len(response.chunk)

        if written != info.size: raise ProtocolError(f"expected {info.size} bytes, got {written}")
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return target, info


def download_all(stub: service_pb2_grpc.MemeStorageStub, ids: list[str], output: Path) -> int:
    failed = 0
    for i, file_id in enumerate(ids):
        try:
            target, info = download(stub, file_id, output)
        except grpc.RpcError as e:
            print(f"{i}: FAIL  {file_id}: {e.code().name}: {e.details()}")
            failed += 1
            if e.code() == grpc.StatusCode.UNAVAILABLE: break
            continue
        except ProtocolError as e:
            print(f"{i}: FAIL  {file_id}: {e}")
            failed += 1
            continue

        print(f"{i}: {file_id}  {info.original_name} ({info.content_type}, {info.size} bytes) -> {target}")
    return 1 if failed else 0


def get(stub: service_pb2_grpc.MemeStorageStub, query: str, output: Path, count: int) -> int:
    try:
        response = stub.GetByQuery(service_pb2.QueryRequest(queries=[query], record_count=count), timeout=TIMEOUT)
    except grpc.RpcError as e:
        print(f"FAIL  {e.code().name}: {e.details()}")
        return 1

    ids = list(response.result[0].ids) if response.result else []
    if not ids:
        print("Nothing found")
        return 0
    return download_all(stub, ids, output)


# --- entry point

def positive_int(value: str) -> int:
    n = int(value)
    if n < 1: raise argparse.ArgumentTypeError("must be >= 1")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="MemeStorage gRPC client")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"server address (default: {DEFAULT_SERVER})")
    commands = parser.add_subparsers(dest="command", required=True)

    upload_cmd = commands.add_parser("upload", help="upload a file or every media file in a directory")
    upload_cmd.add_argument("path", type=Path)
    upload_cmd.add_argument("-d", "--description", default="", help="text description, applied to every uploaded file")
    upload_cmd.add_argument("-r", "--recursive", action="store_true", help="include subdirectories")
    upload_cmd.add_argument("-j", "--jobs", type=positive_int, default=DEFAULT_JOBS, help=f"parallel uploads (default: {DEFAULT_JOBS})")

    get_cmd = commands.add_parser("get", help="find files by text query and save them to a directory")
    get_cmd.add_argument("query")
    get_cmd.add_argument("-o", "--output", type=Path, required=True, help="directory to save files to")
    get_cmd.add_argument("-n", "--count", type=positive_int, default=1, help="number of files to fetch (default: 1)")

    download_cmd = commands.add_parser("download", help="save files with given ids to a directory")
    download_cmd.add_argument("ids", nargs="+")
    download_cmd.add_argument("-o", "--output", type=Path, required=True, help="directory to save files to")

    args = parser.parse_args()
    stub = connect(args.server)

    try:
        match args.command:
            case "upload": return upload(stub, args.path, args.description, args.recursive, args.jobs)
            case "get": return get(stub, args.query, args.output, args.count)
            case _: return download_all(stub, args.ids, args.output)
    except FileNotFoundError as e:
        print(f"Not found: {e}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
