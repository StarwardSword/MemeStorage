from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from email.message import Message
from mimetypes import guess_type
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    import httpx
except ImportError:  # Keep --help useful if the optional HTTP dependency is missing.
    httpx = None  # type: ignore[assignment]

DEFAULT_SERVER = "http://localhost:8000"
TOKEN_ENV = "MEMESTORAGE_TOKEN"
DEFAULT_MODEL_CONFIG = Path("config.yaml")
DEFAULT_JOBS = 4
DEFAULT_BATCH = 16
TIMEOUT = 300.0
CONNECT_TIMEOUT = 15.0
HTTP_CLIENT_ERROR = httpx.HTTPError if httpx is not None else RuntimeError
MEDIA_PREFIXES = ("image/", "video/", "audio/")
FFMPEG_DIR = Path(__file__).parent / "ffmpeg"


class ModelConfigError(Exception):
    """Invalid or missing local embedding model configuration."""


class ApiError(Exception):
    """An API response or network request failed."""


def normalize_server(server: str) -> str:
    """Accept either a full URL or a host:port pair."""
    server = server.strip().rstrip("/")
    if not server:
        raise ValueError("server URL must not be empty")
    if "://" not in server:
        server = "http://" + server
    if not server.startswith(("http://", "https://")):
        raise ValueError("server must start with http:// or https://")
    return server


def make_client(server: str, token: str):
    if httpx is None:
        raise ApiError("HTTP client dependency missing; install it with: pip install httpx")
    try:
        base_url = normalize_server(server)
    except ValueError as e:
        raise ApiError(str(e)) from e
    return httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(TIMEOUT, connect=CONNECT_TIMEOUT),
        follow_redirects=True,
    )


def response_detail(response: Any) -> str:
    """Return a useful detail for API and transport errors, including streamed responses."""
    try:
        if hasattr(response, "is_stream_consumed") and not response.is_stream_consumed:
            response.read()
        body = response.json()
    except Exception:
        try:
            body = getattr(response, "text", "")
        except Exception:
            body = ""
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, (dict, list)):
            return json.dumps(detail, ensure_ascii=False)
        return str(detail)
    return str(body or getattr(response, "reason_phrase", "request failed"))


def check_response(response: Any) -> Any:
    if response.is_success:
        return response
    raise ApiError(f"HTTP {response.status_code}: {response_detail(response)}")


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be an integer >= 1") from e
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def search_count(value: str) -> int:
    number = positive_int(value)
    if number > 100:
        raise argparse.ArgumentTypeError("must be <= 100")
    return number


def collect(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)
    candidates = target.rglob("*") if recursive else target.iterdir()
    return sorted(path for path in candidates if path.is_file())


def collect_many(targets: list[Path], recursive: bool) -> list[Path]:
    """Collect multiple files/folders while avoiding duplicate paths."""
    result: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        for path in collect(target, recursive):
            identity = path.resolve()
            if identity not in seen:
                seen.add(identity)
                result.append(path)
    return result


def is_media(path: Path) -> bool:
    content_type = guess_type(path.name)[0]
    return content_type is not None and content_type.startswith(MEDIA_PREFIXES)


def batches(paths: list[Path], size: int) -> list[list[Path]]:
    return [paths[index : index + size] for index in range(0, len(paths), size)]


class LocalModel:
    """Load the same local embedding provider used by the MemeStorage server."""

    def __init__(self, config_path: Path) -> None:
        if hasattr(os, "add_dll_directory") and FFMPEG_DIR.is_dir():
            self._ffmpeg_dlls = os.add_dll_directory(str(FFMPEG_DIR))

        try:
            import yaml
            from app import FileDescriptor
            from infrastructure import GemmaConfiguration, GemmaEmbeddings, OnnxEmbeddings
        except ImportError as e:
            raise ModelConfigError(
                f"could not import the project's embedding dependencies: {e}. "
                "Run this client from the MemeStorage project environment."
            ) from e

        try:
            with open(config_path, encoding="utf-8") as handle:
                section = (yaml.safe_load(handle) or {}).get("gemma")
        except OSError as e:
            raise ModelConfigError(f"could not read {config_path}: {e}") from e
        if section is None:
            raise ModelConfigError(f"no gemma section in {config_path}")
        try:
            config = GemmaConfiguration.model_validate(section)
        except ValueError as e:
            raise ModelConfigError(str(e)) from None

        self._descriptor = FileDescriptor
        self._paths: dict[uuid.UUID, Path] = {}
        provider = OnnxEmbeddings if config.backend == "onnx" else GemmaEmbeddings
        self._provider = provider(config=config, file_storage=self)  # type: ignore[arg-type]

    def get_full_path(self, id: uuid.UUID) -> Path:
        return self._paths[id]

    def text(self, description: str) -> list[float]:
        return self._provider.document([description])[0].tolist()

    def files(self, paths: list[Path]) -> tuple[dict[Path, list[float]], dict[Path, str]]:
        items: dict[Path, Any] = {}
        for path in paths:
            item_id = uuid.uuid4()
            self._paths[item_id] = path
            items[path] = self._descriptor(
                id=item_id,
                original_name=path.name,
                content_type=guess_type(path.name)[0],
            )

        done: dict[Path, list[float]] = {}
        failed: dict[Path, str] = {}
        try:
            try:
                encoded = self._provider.document(list(items.values()))
                done = {path: embedding.tolist() for path, embedding in zip(paths, encoded)}
            except Exception:
                # A malformed file can fail a whole embedding batch; retry files individually.
                for path, item in items.items():
                    try:
                        done[path] = self._provider.document([item])[0].tolist()
                    except Exception as e:
                        failed[path] = f"{type(e).__name__}: {e}"
            return done, failed
        finally:
            for item in items.values():
                self._paths.pop(item.id, None)


def print_upload_summary(uploaded: int, failed: int, skipped: int, total: int) -> None:
    print(f"\nUploaded: {uploaded}, skipped: {skipped}, failed: {failed}")
    if total == 0:
        print("No media files found.")


def upload_single(
    client: Any,
    path: Path,
    description: str,
    embedding: list[float] | None = None,
    text_embedding: list[float] | None = None,
) -> None:
    content_type = guess_type(path.name)[0] or "application/octet-stream"
    fields: dict[str, str] = {}
    if description:
        fields["description"] = description
    if embedding is not None:
        fields["embeddings"] = json.dumps(embedding, separators=(",", ":"))
    if text_embedding is not None:
        fields["text_embeddings"] = json.dumps(text_embedding, separators=(",", ":"))

    # httpx streams the opened file into the multipart body instead of reading the
    # entire media file into memory.
    with path.open("rb") as file_handle:
        response = client.post(
            "/files",
            data=fields,
            files={"file": (path.name, file_handle, content_type)},
        )
    check_response(response)


def upload_batch(client: Any, paths: list[Path]) -> list[dict[str, Any]]:
    """Use /files/batch, which accepts multiple files but no description/vectors."""
    with ExitStack() as stack:
        multipart = []
        for path in paths:
            handle = stack.enter_context(path.open("rb"))
            content_type = guess_type(path.name)[0] or "application/octet-stream"
            multipart.append(("files", (path.name, handle, content_type)))
        response = client.post("/files/batch", files=multipart)
    check_response(response)
    body = response.json()
    failed = body.get("failed", [])
    return failed if isinstance(failed, list) else []


def is_fatal_api_error(error: BaseException) -> bool:
    if httpx is not None and isinstance(error, httpx.HTTPError):
        # Transport errors (connection refused, timeout, broken connection) are
        # unlikely to improve by submitting every queued batch again.
        if not isinstance(error, httpx.HTTPStatusError):
            return True
    message = str(error)
    return (
        "HTTP 401:" in message
        or "HTTP 403:" in message
        or any(f"HTTP {code}:" in message for code in range(500, 600))
    )


def upload(
    client: Any,
    targets: list[Path],
    description: str,
    recursive: bool,
    jobs: int,
    batch_size: int,
    model: LocalModel | None,
) -> int:
    try:
        files = collect_many(targets, recursive)
    except FileNotFoundError as e:
        print(f"Not found: {e}", file=sys.stderr)
        return 1

    media: list[Path] = []
    skipped = 0
    for path in files:
        if is_media(path):
            media.append(path)
        else:
            print(f"SKIP  {path}: unsupported type ({guess_type(path.name)[0]})")
            skipped += 1

    if not media:
        print_upload_summary(0, 0, skipped, len(files))
        return 0

    # /files/batch cannot accept descriptions or precomputed embeddings. In those
    # cases, /files is used for each media item so the form fields are preserved.
    use_batch_endpoint = model is None and not description
    text_embedding: list[float] | None = None
    if model is not None and description:
        try:
            text_embedding = model.text(description)
        except Exception as e:
            print(f"FAIL  description embedding: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    groups = batches(media, batch_size)
    stop = threading.Event()

    def process_group(group: list[Path]) -> dict[str, Any]:
        if stop.is_set():
            return {"uploaded": 0, "failed": 0, "skipped": len(group), "lines": [
                ("SKIP", f"{path}: stopped after a previous fatal error") for path in group
            ]}

        lines: list[tuple[str, str]] = []
        group_uploaded = 0
        group_failed = 0
        group_skipped = 0

        if use_batch_endpoint:
            try:
                group_failures = upload_batch(client, group)
            except (ApiError, HTTP_CLIENT_ERROR, OSError) as e:
                lines.append(("FAIL", f"batch ({len(group)} file(s)): {e}"))
                group_failed += len(group)
                if is_fatal_api_error(e):
                    stop.set()
                return {
                    "uploaded": group_uploaded, "failed": group_failed,
                    "skipped": group_skipped, "lines": lines,
                }

            if not group_failures:
                lines.extend(("OK", str(path)) for path in group)
                group_uploaded += len(group)
            else:
                for failure in group_failures:
                    kind = failure.get("kind", "unknown")
                    message = failure.get("message", "no details")
                    lines.append(("FAIL", f"{kind}: {message}"))
                failure_count = min(len(group), len(group_failures))
                group_failed += failure_count
                group_uploaded += max(0, len(group) - failure_count)
                lines.append((
                    "PART",
                    f"batch of {len(group)} file(s): API reported {len(group_failures)} "
                    "failure(s); individual filenames may be unavailable",
                ))
            return {
                "uploaded": group_uploaded, "failed": group_failed,
                "skipped": group_skipped, "lines": lines,
            }

        embeddings: dict[Path, list[float]] = {}
        unembedded: dict[Path, str] = {}
        if model is not None:
            try:
                embeddings, unembedded = model.files(group)
            except Exception as e:
                # A provider-level error outside per-file handling should not crash
                # the executor; retry one file at a time to isolate the issue.
                for path in group:
                    try:
                        single, single_failed = model.files([path])
                        embeddings.update(single)
                        unembedded.update(single_failed)
                    except Exception as single_error:
                        unembedded[path] = f"{type(single_error).__name__}: {single_error}"
                if not embeddings and not unembedded:
                    unembedded.update({path: f"{type(e).__name__}: {e}" for path in group})

        if model is not None:
            for path in group:
                if path not in embeddings and path not in unembedded:
                    unembedded[path] = "embedding provider returned no vector for this file"

        for path, reason in unembedded.items():
            lines.append(("FAIL", f"{path}: local model: {reason}"))
            group_failed += 1

        sent = [path for path in group if path not in unembedded]
        for index, path in enumerate(sent):
            if stop.is_set():
                remaining = sent[index:]
                group_skipped += len(remaining)
                lines.extend(("SKIP", f"{pending}: stopped after a previous fatal error") for pending in remaining)
                break
            try:
                upload_single(
                    client,
                    path,
                    description,
                    embedding=embeddings.get(path) if model is not None else None,
                    text_embedding=text_embedding,
                )
            except (ApiError, HTTP_CLIENT_ERROR, OSError) as e:
                lines.append(("FAIL", f"{path}: {e}"))
                group_failed += 1
                if is_fatal_api_error(e):
                    stop.set()
                    remaining = sent[index + 1 :]
                    group_skipped += len(remaining)
                    lines.extend(("SKIP", f"{pending}: stopped after a previous fatal error") for pending in remaining)
                    break
            else:
                lines.append(("OK", str(path)))
                group_uploaded += 1

        return {
            "uploaded": group_uploaded, "failed": group_failed,
            "skipped": group_skipped, "lines": lines,
        }

    uploaded = failed = not_sent = 0
    # The HTTP client is thread-safe. LocalModel's provider also manages concurrent
    # inference; `jobs` controls the number of groups being processed at once.
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(process_group, group): group for group in groups}
        for future in as_completed(futures):
            group = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"FAIL  batch ({len(group)} file(s)): {type(e).__name__}: {e}", file=sys.stderr)
                failed += len(group)
                continue
            for level, line in result["lines"]:
                stream = sys.stderr if level == "FAIL" else sys.stdout
                print(f"{level:<5} {line}", file=stream)
            uploaded += result["uploaded"]
            failed += result["failed"]
            not_sent += result["skipped"]

    print_upload_summary(uploaded, failed, skipped, len(media))
    if not_sent:
        print(f"Not sent after an earlier error: {not_sent}")
    return 1 if failed else 0


def unique_path(directory: Path, filename: str) -> Path:
    name = Path(filename).name
    if name in ("", ".", ".."):
        name = "file"
    target = directory / name
    index = 1
    while target.exists():
        target = directory / f"{Path(name).stem} ({index}){Path(name).suffix}"
        index += 1
    return target


def filename_from_response(response: Any, fallback: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    if disposition:
        message = Message()
        message["content-disposition"] = disposition
        candidate = message.get_filename()
        if candidate:
            return Path(candidate).name
    return fallback


def download_one(client: Any, file_id: str, output: Path) -> tuple[Path, str, int]:
    with client.stream("GET", f"/files/{quote(file_id, safe='')}", params={"download": "true"}) as response:
        check_response(response)
        target = unique_path(output, filename_from_response(response, f"{file_id}.bin"))
        partial = target.with_name(target.name + ".part")
        written = 0
        output.mkdir(parents=True, exist_ok=True)
        try:
            with partial.open("xb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
            partial.rename(target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return target, response.headers.get("content-type", "unknown"), written


def download_all(client: Any, ids: list[str], output: Path) -> int:
    failed = 0
    for index, file_id in enumerate(ids, start=1):
        try:
            target, content_type, byte_count = download_one(client, file_id, output)
        except (ApiError, HTTP_CLIENT_ERROR, OSError) as e:
            print(f"{index}: FAIL  {file_id}: {e}", file=sys.stderr)
            failed += 1
            if "HTTP 401:" in str(e) or "HTTP 403:" in str(e) or "HTTP 5" in str(e):
                break
            continue
        print(f"{index}: {file_id} ({content_type}, {byte_count} bytes) -> {target}")
    return 1 if failed else 0


def search(client: Any, query: str, count: int) -> list[dict[str, Any]]:
    response = check_response(client.get("/search", params={"q": query, "count": count}))
    result = response.json()
    items = result.get("items", []) if isinstance(result, dict) else []
    return items if isinstance(items, list) else []


def search_command(client: Any, query: str, count: int) -> int:
    try:
        items = search(client, query, count)
    except (ApiError, HTTP_CLIENT_ERROR) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1
    if not items:
        print("Nothing found")
        return 0
    for index, item in enumerate(items, start=1):
        print(
            f"{index}: {item.get('id', '?')}  {item.get('original_name', '(unnamed)')} "
            f"({item.get('content_type', 'unknown')})"
        )
    return 0


def get_command(client: Any, query: str, output: Path, count: int) -> int:
    try:
        items = search(client, query, count)
    except (ApiError, HTTP_CLIENT_ERROR) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1
    ids = [item["id"] for item in items if item.get("id")]
    if not ids:
        print("Nothing found")
        return 0
    return download_all(client, ids, output)


def info_command(client: Any, file_id: str) -> int:
    try:
        response = check_response(client.get(f"/files/{quote(file_id, safe='')}/info"))
    except (ApiError, HTTP_CLIENT_ERROR) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    return 0


def health_command(client: Any) -> int:
    try:
        response = check_response(client.get("/health"))
    except (ApiError, HTTP_CLIENT_ERROR) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    return 0


def format_time(value: float | int | None) -> str:
    if not value:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
    except (TypeError, ValueError, OverflowError):
        return "-"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("  (nothing)")
        return
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    print("  " + "  ".join(value.ljust(width) for value, width in zip(headers, widths)).rstrip())
    print("  " + "  ".join("-" * width for width in widths).rstrip())
    for row in rows:
        print("  " + "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())


def user_command(client: Any, args: argparse.Namespace) -> int:
    try:
        if args.action == "create":
            response = check_response(
                client.post(
                    "/admin/users",
                    json={"name": args.name, "allow_embeddings": args.embeddings},
                )
            )
            result = response.json()
            user = result.get("user", {})
            print(f"{user.get('id', '?')}  {user.get('name', args.name)}")
            print(f"token: {result.get('token', '')}")
            print("The token is shown once; store it now.")
            return 0

        if args.action == "list":
            response = check_response(client.get("/admin/users"))
            users = response.json()
            print_table(
                ["id", "name", "embeddings", "created", "deleted"],
                [
                    [
                        str(user.get("id", "")),
                        str(user.get("name", "")),
                        "yes" if user.get("allow_embeddings") else "no",
                        format_time(user.get("created_at")),
                        format_time(user.get("deleted_at")),
                    ]
                    for user in users
                ],
            )
            return 0

        user_id = args.user
        if args.action == "delete":
            check_response(client.delete(f"/admin/users/{quote(user_id, safe='')}"))
            print(f"Deleted user {user_id}; its token has been revoked.")
        elif args.action == "token":
            response = check_response(client.post(f"/admin/users/{quote(user_id, safe='')}/token"))
            result = response.json()
            if result.get("restored"):
                print("The deleted user was restored.")
            print(f"token: {result.get('token', '')}")
            print("The old token no longer works; the new token is shown once.")
        elif args.action == "set":
            response = check_response(
                client.patch(
                    f"/admin/users/{quote(user_id, safe='')}",
                    json={"allow_embeddings": args.embeddings},
                )
            )
            user = response.json()
            print(
                f"{user.get('id', user_id)}  {user.get('name', '')}  "
                f"embeddings={'yes' if user.get('allow_embeddings') else 'no'}"
            )
        return 0
    except (ApiError, HTTP_CLIENT_ERROR, KeyError, ValueError) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1


def stats_command(client: Any, days: int) -> int:
    try:
        response = check_response(client.get("/admin/stats"))
        stats = response.json()
    except (ApiError, HTTP_CLIENT_ERROR) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1

    print(f"Stored files: {stats.get('files', 0)}, {stats.get('bytes', 0)} bytes")
    rows = []
    for user in sorted(stats.get("users", []), key=lambda item: -int(item.get("bytes", 0) or 0)):
        by_type = user.get("by_type", {}) or {}
        def typed(name: str) -> str:
            entry = by_type.get(name, {})
            if not entry:
                return "-"
            return f"{entry.get('files', 0)} / {entry.get('bytes', 0)} B"
        rows.append(
            [
                str(user.get("name") or "(no owner)") + (" (deleted)" if user.get("deleted") else ""),
                str(user.get("files", 0)),
                f"{user.get('bytes', 0)} B",
                typed("image"),
                typed("video"),
                typed("audio"),
                format_time(user.get("last_upload")),
            ]
        )
    print_table(["user", "files", "size", "images", "videos", "audio", "last upload"], rows)

    since = time.time() - days * 86400
    try:
        activity_response = check_response(client.get("/admin/activity/summary", params={"since": since}))
    except (ApiError, HTTP_CLIENT_ERROR) as e:
        print(f"\nActivity summary unavailable: {e}")
        return 0
    activity = activity_response.json()
    print(
        f"\nActivity since {format_time(activity.get('since'))}: "
        f"{activity.get('requests', 0)} requests; {activity.get('ok', 0)} ok, "
        f"{activity.get('denied', 0)} denied, "
        f"{activity.get('client_error', 0)} client errors, "
        f"{activity.get('server_error', 0)} server errors; "
        f"{activity.get('bytes_in', 0)} bytes in, {activity.get('bytes_out', 0)} bytes out"
    )
    return 0


def add_connection_options(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    default: Any = argparse.SUPPRESS if suppress else DEFAULT_SERVER
    token_default: Any = argparse.SUPPRESS if suppress else os.environ.get(TOKEN_ENV)
    parser.add_argument(
        "--server", default=default,
        help=f"HTTP API base URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--token", default=token_default,
        help=f"Bearer token (default: ${TOKEN_ENV}; avoids exposing it in process arguments)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemeStorage HTTP client")
    add_connection_options(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    upload_parser = commands.add_parser("upload", help="upload media files/folders")
    add_connection_options(upload_parser, suppress=True)
    upload_parser.add_argument("paths", nargs="+", type=Path, help="files and/or folders")
    upload_parser.add_argument("-d", "--description", default="", help="description for every file")
    upload_parser.add_argument("-r", "--recursive", action="store_true", help="include subfolders")
    upload_parser.add_argument("-j", "--jobs", type=positive_int, default=DEFAULT_JOBS,
                               help=f"parallel upload groups (default: {DEFAULT_JOBS})")
    upload_parser.add_argument("-b", "--batch-size", type=positive_int, default=DEFAULT_BATCH,
                               help=f"files per batch (default: {DEFAULT_BATCH})")
    upload_parser.add_argument("-e", "--embed", action="store_true",
                               help="calculate file embeddings locally and send ready vectors")
    upload_parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG,
                               help=f"config.yaml with the gemma section (default: {DEFAULT_MODEL_CONFIG})")

    search_parser = commands.add_parser("search", help="search by text and print matching files")
    add_connection_options(search_parser, suppress=True)
    search_parser.add_argument("query")
    search_parser.add_argument("-n", "--count", type=search_count, default=10)

    get_parser = commands.add_parser("get", help="search by text and download matching files")
    add_connection_options(get_parser, suppress=True)
    get_parser.add_argument("query")
    get_parser.add_argument("-o", "--output", type=Path, required=True)
    get_parser.add_argument("-n", "--count", type=search_count, default=1)

    download_parser = commands.add_parser("download", help="download files by UUID")
    add_connection_options(download_parser, suppress=True)
    download_parser.add_argument("ids", nargs="+")
    download_parser.add_argument("-o", "--output", type=Path, required=True)

    info_parser = commands.add_parser("info", help="show metadata for a file UUID")
    add_connection_options(info_parser, suppress=True)
    info_parser.add_argument("file_id")

    health_parser = commands.add_parser("health", help="check whether the API is up")
    add_connection_options(health_parser, suppress=True)

    user_parser = commands.add_parser("user", help="manage users (admin token required)")
    add_connection_options(user_parser, suppress=True)
    user_actions = user_parser.add_subparsers(dest="action", required=True)

    create_parser = user_actions.add_parser("create", help="create user and print its token")
    add_connection_options(create_parser, suppress=True)
    create_parser.add_argument("name")
    create_parser.add_argument("--embeddings", action="store_true",
                               help="allow this user to upload precomputed embeddings")

    list_parser = user_actions.add_parser("list", help="list users, including deleted users")
    add_connection_options(list_parser, suppress=True)
    for action, description in (("delete", "revoke a user's token"), ("token", "rotate a user's token")):
        subparser = user_actions.add_parser(action, help=description)
        add_connection_options(subparser, suppress=True)
        subparser.add_argument("user", help="user UUID")
    set_parser = user_actions.add_parser("set", help="change user permissions")
    add_connection_options(set_parser, suppress=True)
    set_parser.add_argument("user", help="user UUID")
    set_parser.add_argument("--embeddings", action=argparse.BooleanOptionalAction, required=True,
                            help="whether this user may upload precomputed embeddings")

    stats_parser = commands.add_parser("stats", help="show stored file and activity statistics")
    add_connection_options(stats_parser, suppress=True)
    stats_parser.add_argument("--days", type=positive_int, default=7,
                              help="activity period in days (default: 7)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.token:
        parser.error(f"a token is needed: pass --token or set ${TOKEN_ENV}")

    if args.command == "download":
        invalid = []
        for value in args.ids:
            try:
                uuid.UUID(value)
            except ValueError:
                invalid.append(value)
        if invalid:
            parser.error(f"invalid UUID(s): {', '.join(invalid)}")

    model: LocalModel | None = None
    if args.command == "upload" and args.embed:
        try:
            model = LocalModel(args.model_config)
        except ModelConfigError as e:
            print(f"Model config: {e}", file=sys.stderr)
            return 1

    try:
        client = make_client(args.server, args.token)
    except (ApiError, HTTP_CLIENT_ERROR) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1

    try:
        try:
            if args.command == "upload":
                return upload(
                    client, args.paths, args.description, args.recursive,
                    args.jobs, args.batch_size, model,
                )
            if args.command == "search":
                return search_command(client, args.query, args.count)
            if args.command == "get":
                return get_command(client, args.query, args.output, args.count)
            if args.command == "download":
                return download_all(client, args.ids, args.output)
            if args.command == "info":
                return info_command(client, args.file_id)
            if args.command == "health":
                return health_command(client)
            if args.command == "user":
                return user_command(client, args)
            if args.command == "stats":
                return stats_command(client, args.days)
            parser.error(f"unsupported command: {args.command}")
        except KeyboardInterrupt:
            print("Interrupted", file=sys.stderr)
            return 130
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
