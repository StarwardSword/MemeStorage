import ipaddress
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import grpc

from app import ActivityRecorder, Event, Principal, outcome_of

# Handlers leave here who called and what is worth knowing, the interceptor reads it when the call ends.
# A call runs in one thread of the server, so a thread-local value belongs to it alone
CURRENT = threading.local()


def remember(principal: Principal | None = None, detail: str | None = None) -> None:
    if principal is not None:
        CURRENT.principal = principal
    if detail is not None:
        CURRENT.detail = detail


def peer_ip(peer: str) -> str | None:
    """'ipv4:1.2.3.4:5555' or 'ipv6:[::1]:5555' to the address, None for a unix socket"""
    kind, _, rest = peer.partition(":")
    if kind == "ipv4":
        return rest.rsplit(":", 1)[0]
    if kind == "ipv6":
        return rest.rsplit(":", 1)[0].strip("[]")
    return None


def is_loopback(ip: str | None) -> bool:
    try:
        address = ipaddress.ip_address(ip or "")
    except ValueError:
        return False
    # A client of the IPv4 socket on a dual-stack listener shows as ::ffff:127.0.0.1
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback


def client_ip(context: grpc.ServicerContext) -> str | None:
    """Address of the client. The server sits behind nginx on the same machine, which sends the address
    of its client in X-Forwarded-For. It is believed only when the call comes from loopback: from anyone
    else the header is whatever they wrote. A value that is not an address is not written to the log"""
    peer = peer_ip(context.peer())
    if not is_loopback(peer):
        return peer

    forwarded = dict(context.invocation_metadata() or ()).get("x-forwarded-for")
    if not isinstance(forwarded, str):
        return peer
    first = forwarded.split(",")[0].strip()
    try:
        return str(ipaddress.ip_address(first))
    except ValueError:
        return peer


class ActivityInterceptor(grpc.ServerInterceptor):
    """Writes one event for every call, whatever it does and however it ends"""

    def __init__(self, recorder: ActivityRecorder) -> None:
        self.recorder = recorder

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler | None],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        handler = continuation(handler_call_details)
        if handler is None or handler.request_streaming and handler.response_streaming:
            return handler

        action = handler_call_details.method.rsplit("/", 1)[-1]
        make = {
            (False, False): grpc.unary_unary_rpc_method_handler,
            (True, False): grpc.stream_unary_rpc_method_handler,
            (False, True): grpc.unary_stream_rpc_method_handler,
        }[(bool(handler.request_streaming), bool(handler.response_streaming))]
        inner = handler.unary_unary or handler.stream_unary or handler.unary_stream
        return make(
            self._wrap(action, inner, handler),
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    def _wrap(self, action: str, inner: Callable[..., Any], handler: Any) -> Any:
        def counting(messages: Iterator[Any], total: list[int]) -> Iterator[Any]:
            for m in messages:
                total[0] += m.ByteSize()
                yield m

        def finish(
            context: grpc.ServicerContext,
            ts: float,
            started: float,
            received: int,
            sent: int,
            failure: BaseException | None,
        ) -> None:
            code = context.code()
            name = code.name if code else ("UNKNOWN" if failure else "OK")
            principal: Principal | None = getattr(CURRENT, "principal", None)
            self.recorder.record(
                Event(
                    ts=ts,
                    transport="grpc",
                    action=action,
                    code=name,
                    outcome=outcome_of(name),
                    user_id=principal.user_id if principal else None,
                    actor=principal.name if principal else None,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    bytes_in=received,
                    bytes_out=sent,
                    ip=client_ip(context),
                    detail=getattr(CURRENT, "detail", None),
                )
            )

        def begin() -> tuple[float, float]:
            CURRENT.principal = None
            CURRENT.detail = None
            return time.time(), time.perf_counter()

        if handler.response_streaming:

            def streaming(request: Any, context: grpc.ServicerContext) -> Iterator[Any]:
                ts, started = begin()
                sent, failure = 0, None
                try:
                    for message in inner(request, context):
                        sent += message.ByteSize()
                        yield message
                except BaseException as e:
                    failure = e
                    raise
                finally:
                    finish(context, ts, started, request.ByteSize(), sent, failure)

            return streaming

        def unary(request: Any, context: grpc.ServicerContext) -> Any:
            ts, started = begin()
            failure = None
            received = [0]
            try:
                if handler.request_streaming:
                    response = inner(counting(request, received), context)
                else:
                    received[0] = request.ByteSize()
                    response = inner(request, context)
                return response
            except BaseException as e:
                failure = e
                raise
            finally:
                finish(
                    context,
                    ts,
                    started,
                    received[0],
                    response.ByteSize() if failure is None else 0,
                    failure,
                )

        return unary
