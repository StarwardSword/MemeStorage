import argparse
import sys
import threading
import uuid
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
DEFAULT_BATCH = 32
MEDIA_PREFIXES = ("image/", "video/", "audio/")


class ProtocolError(Exception):
    pass


def connect(server: str) -> service_pb2_grpc.MemeStorageStub:
    return service_pb2_grpc.MemeStorageStub(grpc.insecure_channel(server))


# --- upload


def collect(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)

    candidates = target.rglob("*") if recursive else target.iterdir()
    return sorted(p for p in candidates if p.is_file())


# Server stores the file before embedding it, so unsupported types would leave orphans there
def is_media(path: Path) -> bool:
    content_type = guess_type(path.name)[0]
    return content_type is not None and content_type.startswith(MEDIA_PREFIXES)


def save_multiple_requests(
    paths: list[Path], description: str
) -> Iterator[service_pb2.SaveMultipleRequest]:
    for path in paths:
        info = service_pb2.SaveFileInfo(filename=path.name)
        if description:
            info.text_description = description
        yield service_pb2.SaveMultipleRequest(info=info)

        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                yield service_pb2.SaveMultipleRequest(data=chunk)


def batches(paths: list[Path], size: int) -> list[list[Path]]:
    return [paths[i : i + size] for i in range(0, len(paths), size)]


def failure_kind(failure: service_pb2.SaveFailure) -> str:
    return service_pb2.SaveFailure.Kind.Name(failure.kind)


def upload(
    stub: service_pb2_grpc.MemeStorageStub,
    target: Path,
    description: str,
    recursive: bool,
    jobs: int,
    batch_size: int,
) -> int:
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

    # Returns files the server did not save, None when the group was not sent
    def upload_group(group: list[Path]) -> list[service_pb2.SaveFailure] | None:
        if stop.is_set():
            return None

        response = stub.SaveMultiple(
            save_multiple_requests(group, description), timeout=TIMEOUT * len(group)
        )
        return list(response.failed)

    groups = batches(media, batch_size)

    # Channel is thread-safe, so every upload shares it; results are printed in completion order
    uploaded, failed, not_sent = 0, 0, 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(upload_group, group): group for group in groups}
        try:
            for future in as_completed(futures):
                group = futures[future]
                try:
                    failures = future.result()
                except grpc.RpcError as e:
                    for path in group:
                        print(f"FAIL  {path}: {e.code().name}: {e.details()}")
                    failed += len(group)
                    # No point in sending the queued files to a server that is down
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        stop.set()
                    continue

                if failures is None:
                    not_sent += len(group)
                    continue
                if not failures:
                    for path in group:
                        print(f"OK    {path}")
                    uploaded += len(group)
                    continue

                # Server does not always name the failed file, so which files of the group
                # are saved is unknown, only how many
                for failure in failures:
                    print(f"FAIL  {failure_kind(failure)}: {failure.message}")
                mark = "FAIL" if len(failures) >= len(group) else "PART"
                for path in group:
                    print(f"{mark}  {path}")
                failed += len(failures)
                uploaded += len(group) - len(failures)
        except KeyboardInterrupt:
            # Uploads in flight still finish when the executor exits, queued ones are dropped
            stop.set()
            raise

    if len(files) > 1:
        print(
            f"\nUploaded: {uploaded}, skipped: {skipped}, failed: {failed}"
            + (f", not sent: {not_sent}" if not_sent else "")
        )
    return 1 if failed else 0


# --- download


def unique_path(directory: Path, filename: str) -> Path:
    # Filename comes from the server, never let it escape the output directory
    name = Path(filename).name
    if name in ("", ".", ".."):
        name = "file"

    target = directory / name
    i = 1
    while target.exists():
        target = directory / f"{Path(name).stem} ({i}){Path(name).suffix}"
        i += 1
    return target


# Streams file into the directory, returns its path and server-side info
def download(
    stub: service_pb2_grpc.MemeStorageStub, file_id: str, output: Path
) -> tuple[Path, service_pb2.DownloadFileInfo]:
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
                if response.WhichOneof("payload") != "chunk":
                    raise ProtocolError("expected data chunk")
                f.write(response.chunk)
                written += len(response.chunk)

        if written != info.size:
            raise ProtocolError(f"expected {info.size} bytes, got {written}")
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return target, info


def download_all(
    stub: service_pb2_grpc.MemeStorageStub, ids: list[str], output: Path
) -> int:
    failed = 0
    for i, file_id in enumerate(ids):
        try:
            target, info = download(stub, file_id, output)
        except grpc.RpcError as e:
            print(f"{i}: FAIL  {file_id}: {e.code().name}: {e.details()}")
            failed += 1
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                break
            continue
        except ProtocolError as e:
            print(f"{i}: FAIL  {file_id}: {e}")
            failed += 1
            continue

        print(
            f"{i}: {file_id}  {info.original_name} ({info.content_type}, {info.size} bytes) -> {target}"
        )
    return 1 if failed else 0


def get(
    stub: service_pb2_grpc.MemeStorageStub, query: str, output: Path, count: int
) -> int:
    try:
        response = stub.GetByQuery(
            service_pb2.QueryRequest(queries=[query], record_count=count),
            timeout=TIMEOUT,
        )
    except grpc.RpcError as e:
        print(f"FAIL  {e.code().name}: {e.details()}")
        return 1

    ids = list(response.result[0].ids) if response.result else []
    if not ids:
        print("Nothing found")
        return 0
    return download_all(stub, ids, output)


# --- entry point


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="MemeStorage gRPC client")
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"server address (default: {DEFAULT_SERVER})",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    upload_cmd = commands.add_parser(
        "upload", help="upload a file or every media file in a directory"
    )
    upload_cmd.add_argument("path", type=Path)
    upload_cmd.add_argument(
        "-d",
        "--description",
        default="",
        help="text description, applied to every uploaded file",
    )
    upload_cmd.add_argument(
        "-r", "--recursive", action="store_true", help="include subdirectories"
    )
    upload_cmd.add_argument(
        "-j",
        "--jobs",
        type=positive_int,
        default=DEFAULT_JOBS,
        help=f"parallel uploads (default: {DEFAULT_JOBS})",
    )
    upload_cmd.add_argument(
        "-b",
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH,
        help=f"files sent in one call (default: {DEFAULT_BATCH})",
    )

    get_cmd = commands.add_parser(
        "get", help="find files by text query and save them to a directory"
    )
    get_cmd.add_argument("query")
    get_cmd.add_argument(
        "-o", "--output", type=Path, required=True, help="directory to save files to"
    )
    get_cmd.add_argument(
        "-n",
        "--count",
        type=positive_int,
        default=1,
        help="number of files to fetch (default: 1)",
    )

    download_cmd = commands.add_parser(
        "download", help="save files with given ids to a directory"
    )
    download_cmd.add_argument("ids", nargs="+")
    download_cmd.add_argument(
        "-o", "--output", type=Path, required=True, help="directory to save files to"
    )

    args = parser.parse_args()

    # One malformed id cancels the whole download, nothing is fetched
    if args.command == "download":
        invalid = [x for x in args.ids if not is_uuid(x)]
        if invalid:
            parser.error(f"invalid uuid: {', '.join(invalid)}")

    stub = connect(args.server)

    try:
        match args.command:
            case "upload":
                return upload(
                    stub,
                    args.path,
                    args.description,
                    args.recursive,
                    args.jobs,
                    args.batch_size,
                )
            case "get":
                return get(stub, args.query, args.output, args.count)
            case _:
                return download_all(stub, args.ids, args.output)
    except FileNotFoundError as e:
        print(f"Not found: {e}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
