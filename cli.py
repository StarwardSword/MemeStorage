import argparse
import collections
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import grpc

from api.grpc.proto import service_pb2, service_pb2_grpc

DEFAULT_SERVER = "localhost:50051"
TOKEN_ENV = "MEMESTORAGE_TOKEN"
CHUNK_SIZE = 64 * 1024
TIMEOUT = 300
DEFAULT_JOBS = 4
DEFAULT_BATCH = 32
DEFAULT_MODEL_CONFIG = Path("config.yaml")
MEDIA_PREFIXES = ("image/", "video/", "audio/")
# Same as in main.py: torchcodec looks for FFmpeg DLLs once, when it is imported
FFMPEG_DIR = Path(__file__).parent / "ffmpeg"


class ProtocolError(Exception):
    pass


class ModelConfigError(Exception):
    pass


class _Details(
    collections.namedtuple(
        "_Details",
        (
            "method",
            "timeout",
            "metadata",
            "credentials",
            "wait_for_ready",
            "compression",
        ),
    ),
    grpc.ClientCallDetails,
):
    pass


# Puts the token into the metadata of every call, the commands do not have to know about it
class Authorization(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
):
    def __init__(self, token: str) -> None:
        self.token = token

    def _details(self, d: grpc.ClientCallDetails) -> grpc.ClientCallDetails:
        metadata = [*(d.metadata or ()), ("authorization", f"Bearer {self.token}")]
        return _Details(
            d.method,
            d.timeout,
            metadata,
            d.credentials,
            d.wait_for_ready,
            d.compression,
        )

    def intercept_unary_unary(self, continuation, details, request):
        return continuation(self._details(details), request)

    def intercept_unary_stream(self, continuation, details, request):
        return continuation(self._details(details), request)

    def intercept_stream_unary(self, continuation, details, request_iterator):
        return continuation(self._details(details), request_iterator)


def connect(
    server: str, token: str, tls: bool = False
) -> service_pb2_grpc.MemeStorageStub:
    channel = (
        grpc.secure_channel(server, grpc.ssl_channel_credentials())
        if tls
        else grpc.insecure_channel(server)
    )
    channel = grpc.intercept_channel(channel, Authorization(token))
    return service_pb2_grpc.MemeStorageStub(channel)


# --- local embeddings


class LocalModel:
    """The embedding model of the server run on this machine: files go out with ready embeddings,
    so the server does not run its own model on them. Needs a user with allow_embeddings

    Settings come from the gemma section of a server config. truncate_dim must be the same
    as on the server, otherwise it refuses the vectors for their dimension
    """

    def __init__(self, config_path: Path) -> None:
        # The model and its libraries are heavy, other commands never import them
        if hasattr(os, "add_dll_directory") and FFMPEG_DIR.is_dir():
            self._ffmpeg_dlls = os.add_dll_directory(str(FFMPEG_DIR))

        import yaml

        from app import FileDescriptor
        from infrastructure import GemmaConfiguration, GemmaEmbeddings, OnnxEmbeddings

        with open(config_path, encoding="utf-8") as f:
            section = (yaml.safe_load(f) or {}).get("gemma")
        if section is None:
            raise ModelConfigError(f"no gemma section in {config_path}")
        try:
            config = GemmaConfiguration.model_validate(section)
        except ValueError as e:
            raise ModelConfigError(e) from None

        self._descriptor = FileDescriptor
        # The model reads a file by its id in the server storage, here an id stands for a local path
        self._paths: dict[uuid.UUID, Path] = {}
        provider = OnnxEmbeddings if config.backend == "onnx" else GemmaEmbeddings
        self._provider = provider(config=config, file_storage=self)  # type: ignore[arg-type]

    def get_full_path(self, id: uuid.UUID) -> Path:
        return self._paths[id]

    def text(self, description: str) -> list[float]:
        return self._provider.document([description])[0].tolist()

    # Returns embeddings of the files and why the rest could not be embedded.
    # Called from the upload threads at once, the provider takes care of that like on the server
    def files(
        self, paths: list[Path]
    ) -> tuple[dict[Path, list[float]], dict[Path, str]]:
        items: dict[Path, Any] = {}
        for path in paths:
            id = uuid.uuid4()
            self._paths[id] = path
            items[path] = self._descriptor(
                id=id, original_name=path.name, content_type=guess_type(path.name)[0]
            )

        done: dict[Path, list[float]] = {}
        failed: dict[Path, str] = {}
        try:
            try:
                embeddings = self._provider.document(list(items.values()))
                done = {p: e.tolist() for p, e in zip(paths, embeddings)}
            except Exception:
                # One broken file fails the whole batch, then they are embedded one by one
                for path, item in items.items():
                    try:
                        done[path] = self._provider.document([item])[0].tolist()
                    except Exception as e:
                        failed[path] = f"{type(e).__name__}: {e}"
            return done, failed
        finally:
            for item in items.values():
                del self._paths[item.id]


# --- upload


def collect(target: Path, recursive: bool) -> list[Path]:
    """Collect files from one file or directory."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)

    candidates = target.rglob("*") if recursive else target.iterdir()
    return sorted(p for p in candidates if p.is_file())


def collect_many(targets: list[Path], recursive: bool) -> list[Path]:
    """Collect files from several files/directories, without sending duplicates."""
    result: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        for path in collect(target, recursive):
            identity = path.resolve()
            if identity not in seen:
                result.append(path)
                seen.add(identity)
    return result


# Server stores the file before embedding it, so unsupported types would leave orphans there
def is_media(path: Path) -> bool:
    content_type = guess_type(path.name)[0]
    return content_type is not None and content_type.startswith(MEDIA_PREFIXES)


def save_multiple_requests(
    paths: list[Path],
    description: str,
    embeddings: dict[Path, list[float]] | None = None,
    text_embeddings: list[float] | None = None,
) -> Iterator[service_pb2.SaveMultipleRequest]:
    for path in paths:
        info = service_pb2.SaveFileInfo(filename=path.name)
        if description:
            info.text_description = description
        if embeddings is not None:
            info.embeddings.value.extend(embeddings[path])
        if text_embeddings is not None:
            info.text_embeddings.value.extend(text_embeddings)
        yield service_pb2.SaveMultipleRequest(info=info)

        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                yield service_pb2.SaveMultipleRequest(data=chunk)


def batches(paths: list[Path], size: int) -> list[list[Path]]:
    return [paths[i : i + size] for i in range(0, len(paths), size)]


def failure_kind(failure: service_pb2.SaveFailure) -> str:
    return service_pb2.SaveFailure.Kind.Name(failure.kind)


class GroupResult(NamedTuple):
    sent: list[Path]
    # Files the local model could not embed and why, they are not sent
    unembedded: dict[Path, str]
    # Files the server did not save
    failures: list[service_pb2.SaveFailure]


def upload(
    stub: service_pb2_grpc.MemeStorageStub,
    targets: list[Path],
    description: str,
    recursive: bool,
    jobs: int,
    batch_size: int,
    model: LocalModel | None = None,
) -> int:
    files = collect_many(targets, recursive)
    if not files:
        print("No files found in the supplied paths")
        return 0

    media: list[Path] = []
    skipped = 0
    for path in files:
        if is_media(path):
            media.append(path)
            continue
        print(f"SKIP  {path}: unsupported type ({guess_type(path.name)[0]})")
        skipped += 1

    # The description is the same for every file, so it is embedded once
    try:
        # The description is shared by all files, so calculate its embedding once.
        text_embeddings = (
            model.text(description) if model is not None and description else None
        )
    except Exception as e:
        print(
            f"FAIL  local model could not embed the description: "
            f"{type(e).__name__}: {e}"
        )
        return 1

    # Queued uploads check the flag instead of being cancelled: as_completed never sees futures
    # cancelled by executor.shutdown(cancel_futures=True) and would hang
    stop = threading.Event()

    # None when the group was not sent
    def upload_group(group: list[Path]) -> GroupResult | None:
        if stop.is_set():
            return None

        embeddings, unembedded = None, {}
        if model is not None:
            embeddings, unembedded = model.files(group)
            # The model may have taken a while, the server could have said no meanwhile
            if stop.is_set():
                return None
        sent = [p for p in group if p not in unembedded]
        if not sent:
            return GroupResult(sent, unembedded, [])

        response = stub.SaveMultiple(
            save_multiple_requests(sent, description, embeddings, text_embeddings),
            timeout=TIMEOUT * len(sent),
        )
        return GroupResult(sent, unembedded, list(response.failed))

    groups = batches(media, batch_size)

    # Channel is thread-safe, so every upload shares it; results are printed in completion order
    uploaded, failed, not_sent = 0, 0, 0
    forbidden = False
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(upload_group, group): group for group in groups}
        try:
            for future in as_completed(futures):
                group = futures[future]
                try:
                    result = future.result()
                except grpc.RpcError as e:
                    for path in group:
                        print(f"FAIL  {path}: {e.code().name}: {e.details()}")
                    failed += len(group)
                    # No point in sending the queued files to a server that is down
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        stop.set()
                    continue

                if result is None:
                    not_sent += len(group)
                    continue

                for path, reason in result.unembedded.items():
                    print(f"FAIL  {path}: local model: {reason}")
                failed += len(result.unembedded)
                group, failures = result.sent, result.failures
                if not group:
                    continue

                # Every next file would be refused the same way
                if any(f.kind == service_pb2.SaveFailure.FORBIDDEN for f in failures):
                    forbidden = True
                    stop.set()

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

    if forbidden:
        print(
            "\nThe server refused ready embeddings: the admin allows them with "
            "`cli.py user set <name> --embeddings`"
        )
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


# --- users and statistics, they need the admin token


def when(seconds: float | None) -> str:
    if not seconds:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(seconds))


def size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def table(header: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("  (nothing)")
        return
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def find_user(
    stub: service_pb2_grpc.MemeStorageStub, ref: str, deleted: bool = False
) -> service_pb2.UserInfo | None:
    users = stub.ListUsers(service_pb2.Empty(), timeout=TIMEOUT).users
    if is_uuid(ref):
        return next((u for u in users if u.uuid == str(uuid.UUID(ref))), None)
    # A name is never taken twice, so there is one match at most
    same = [u for u in users if u.name == ref]
    return same[0] if same and (deleted or not same[0].HasField("deleted_at")) else None


def show_user(u: service_pb2.UserInfo) -> str:
    flags = [
        "embeddings" if u.allow_embeddings else "",
        "deleted" if u.HasField("deleted_at") else "",
    ]
    return f"{u.uuid}  {u.name}  {' '.join(f for f in flags if f)}".rstrip()


def user_command(
    stub: service_pb2_grpc.MemeStorageStub, args: argparse.Namespace
) -> int:
    if args.action == "create":
        r = stub.CreateUser(
            service_pb2.CreateUserRequest(
                name=args.name, allow_embeddings=args.embeddings
            ),
            timeout=TIMEOUT,
        )
        print(show_user(r.user))
        print(f"token: {r.token}")
        print("The token is shown once, store it now.")
        return 0

    if args.action == "list":
        users = stub.ListUsers(service_pb2.Empty(), timeout=TIMEOUT).users
        table(
            ["id", "name", "embeddings", "created", "deleted"],
            [
                [
                    u.uuid,
                    u.name,
                    "yes" if u.allow_embeddings else "no",
                    when(u.created_at),
                    when(u.deleted_at) if u.HasField("deleted_at") else "",
                ]
                for u in users
            ],
        )
        return 0

    user = find_user(stub, args.user, deleted=args.action == "token")
    if user is None:
        print(f"No such user: {args.user}")
        return 1

    if args.action == "delete":
        stub.DeleteUser(service_pb2.UUIDRequest(uuid=user.uuid), timeout=TIMEOUT)
        print(
            f"Deleted {user.name}: the token is revoked, the files and the statistics stay"
        )
    elif args.action == "token":
        r = stub.RotateToken(service_pb2.UUIDRequest(uuid=user.uuid), timeout=TIMEOUT)
        if r.restored:
            print(f"{user.name} was deleted and is active again")
        print(f"token: {r.token}")
        print("The old token does not work anymore. The new one is shown once.")
    else:
        updated = stub.UpdateUser(
            service_pb2.UpdateUserRequest(
                uuid=user.uuid, allow_embeddings=args.embeddings
            ),
            timeout=TIMEOUT,
        )
        print(show_user(updated))
    return 0


def stats_command(
    stub: service_pb2_grpc.MemeStorageStub, args: argparse.Namespace
) -> int:
    stored = stub.GetStats(service_pb2.Empty(), timeout=TIMEOUT)
    print(f"Stored files: {stored.files}, {size(stored.bytes)}")
    rows = []
    for u in sorted(stored.users, key=lambda u: -u.bytes):
        by = u.by_type
        rows.append(
            [
                (u.name or "(no owner)") + (" (deleted)" if u.deleted else ""),
                str(u.files),
                size(u.bytes),
                *[
                    f"{by[k].files} / {size(by[k].bytes)}" if k in by else "-"
                    for k in ("image", "video", "audio")
                ],
                when(u.last_upload) if u.HasField("last_upload") else "-",
            ]
        )
    table(["user", "files", "size", "images", "videos", "audio", "last upload"], rows)

    request = service_pb2.ActivityRequest()
    if args.days:
        request.since = time.time() - args.days * 86400
    try:
        a = stub.GetActivity(request, timeout=TIMEOUT)
    except grpc.RpcError as e:
        if e.code() != grpc.StatusCode.UNAVAILABLE:
            raise
        print("\nActivity log is turned off")
        return 0

    print(f"\nActivity, {when(a.since)} - {when(a.until)}")
    print(
        f"  {a.requests} requests: {a.ok} ok, {a.denied} denied, {a.client_error} client errors, "
        f"{a.server_error} server errors; average {a.avg_ms:.0f} ms; "
        f"in {size(a.bytes_in)}, out {size(a.bytes_out)}"
    )
    print("\nBy action")
    table(
        ["action", "requests", "errors", "avg ms", "p95 ms", "max ms", "in", "out"],
        [
            [
                x.action,
                str(x.requests),
                str(x.requests - x.ok),
                f"{x.avg_ms:.0f}",
                f"{x.p95_ms:.0f}",
                f"{x.max_ms:.0f}",
                size(x.bytes_in),
                size(x.bytes_out),
            ]
            for x in a.by_action
        ],
    )
    print("\nBy user")
    table(
        ["user", "requests", "errors", "in", "out", "last seen"],
        [
            [
                x.actor or "(nobody)",
                str(x.requests),
                str(x.errors),
                size(x.bytes_in),
                size(x.bytes_out),
                when(x.last_seen),
            ]
            for x in a.by_actor
        ],
    )
    if a.top_queries:
        print("\nTop queries")
        table(
            ["queries", "query"], [[str(q.count), q.query] for q in a.top_queries[:10]]
        )
    if a.jobs_by_status:
        jobs = ", ".join(f"{k}: {v}" for k, v in sorted(a.jobs_by_status.items()))
        why = ", ".join(f"{k}: {v}" for k, v in sorted(a.job_failures.items()))
        print(f"\nBackground jobs: {jobs}" + (f" (failed: {why})" if why else ""))
    if a.dropped:
        print(f"\n{a.dropped} events were lost, the log writer could not keep up")
    return 0


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


def add_connection_arguments(
    parser: argparse.ArgumentParser, *, suppress_defaults: bool = False
) -> None:
    """Add server/auth flags. Suppressed defaults preserve values parsed globally."""
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--server",
        default=default if suppress_defaults else DEFAULT_SERVER,
        help=f"server address (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--tls",
        action="store_true",
        default=default if suppress_defaults else False,
        help="connect over TLS, for example when the server is behind a reverse proxy",
    )
    parser.add_argument(
        "--token",
        default=default if suppress_defaults else os.environ.get(TOKEN_ENV),
        help=f"user/admin token (default: ${TOKEN_ENV}); prefer the environment variable",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MemeStorage gRPC client")
    add_connection_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    upload_cmd = commands.add_parser(
        "upload", help="upload media files or files from one or more directories"
    )
    # Convenient for upload commands: these flags can appear before or after `upload`.
    add_connection_arguments(upload_cmd, suppress_defaults=True)
    upload_cmd.add_argument(
        "paths",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="file(s) and/or directory(ies) to upload",
    )
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
    upload_cmd.add_argument(
        "-e",
        "--embed",
        action="store_true",
        help="compute embeddings with the model on this machine and send them with the files; "
        "the user needs the right to send ready embeddings",
    )
    upload_cmd.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="config whose gemma section sets up the model for --embed, truncate_dim "
        f"must match the server (default: {DEFAULT_MODEL_CONFIG})",
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

    user_cmd = commands.add_parser("user", help="manage users (admin token)")
    user_actions = user_cmd.add_subparsers(dest="action", required=True)
    create_cmd = user_actions.add_parser(
        "create", help="make a user and print its token"
    )
    create_cmd.add_argument("name")
    create_cmd.add_argument(
        "--embeddings", action="store_true", help="the user may upload ready embeddings"
    )
    user_actions.add_parser("list", help="list users, deleted ones too")
    for action, text in (
        ("delete", "revoke the token, the files and the statistics stay"),
        (
            "token",
            "make a new token, the old one stops working; also brings a deleted user back",
        ),
    ):
        user_actions.add_parser(action, help=text).add_argument(
            "user", help="id or name"
        )
    set_cmd = user_actions.add_parser("set", help="change what a user may do")
    set_cmd.add_argument("user", help="id or name")
    set_cmd.add_argument(
        "--embeddings",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="may upload ready embeddings",
    )

    stats_cmd = commands.add_parser(
        "stats", help="who uploaded what and what happened on the server (admin token)"
    )
    stats_cmd.add_argument(
        "--days",
        type=positive_int,
        default=7,
        help="period of the activity (default: 7)",
    )

    args = parser.parse_args()

    if not args.token:
        parser.error(f"a token is needed: pass --token or set ${TOKEN_ENV}")

    # One malformed id cancels the whole download, nothing is fetched
    if args.command == "download":
        invalid = [x for x in args.ids if not is_uuid(x)]
        if invalid:
            parser.error(f"invalid uuid: {', '.join(invalid)}")

    stub = connect(args.server, args.token, args.tls)

    try:
        match args.command:
            case "upload":
                model = LocalModel(args.model_config) if args.embed else None
                return upload(
                    stub,
                    args.paths,
                    args.description,
                    args.recursive,
                    args.jobs,
                    args.batch_size,
                    model,
                )
            case "get":
                return get(stub, args.query, args.output, args.count)
            case "user":
                return user_command(stub, args)
            case "stats":
                return stats_command(stub, args)
            case _:
                return download_all(stub, args.ids, args.output)
    except FileNotFoundError as e:
        print(f"Not found: {e}")
        return 1
    except ModelConfigError as e:
        print(f"Model config: {e}")
        return 1
    except grpc.RpcError as e:
        print(f"FAIL  {e.code().name}: {e.details()}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
