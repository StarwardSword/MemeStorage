import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import ActivityRecorder, Event, Principal, outcome_of

# Health checks come every few seconds and would drown everything else
SKIPPED = ("/health",)


class ActivityMiddleware:
    """Writes one event for every HTTP request: who, what, how it ended, how long, how many bytes.

    A handler adds the text of a search query with `request.state.detail = ...`.
    Plain ASGI and not BaseHTTPMiddleware, so streaming responses stay streaming
    """

    def __init__(self, app: ASGIApp, recorder: ActivityRecorder) -> None:
        self.app = app
        self.recorder = recorder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in SKIPPED:
            await self.app(scope, receive, send)
            return

        ts = time.time()
        started = time.perf_counter()
        code = 500
        sent = 0

        async def counting(message: Message) -> None:
            nonlocal code, sent
            if message["type"] == "http.response.start":
                code = message["status"]
            elif message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, counting)
        finally:
            self._record(scope, ts, time.perf_counter() - started, str(code), sent)

    def _record(
        self, scope: Scope, ts: float, seconds: float, code: str, sent: int
    ) -> None:
        state = scope.get("state") or {}
        principal: Principal | None = state.get("principal")
        route = scope.get("route")
        received = dict(scope["headers"]).get(b"content-length", b"0")
        client = scope.get("client")

        self.recorder.record(
            Event(
                ts=ts,
                transport="http",
                action=getattr(route, "name", None) or "unmatched",
                code=code,
                outcome=outcome_of(code),
                user_id=principal.user_id if principal else None,
                actor=principal.name if principal else None,
                duration_ms=seconds * 1000,
                bytes_in=int(received) if received.isdigit() else 0,
                bytes_out=sent,
                ip=client[0] if client else None,
                path=scope["path"],
                detail=(str(state["detail"])[:500] if state.get("detail") else None),
            )
        )
