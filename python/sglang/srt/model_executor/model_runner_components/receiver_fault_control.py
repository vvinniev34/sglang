from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import msgspec

from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)

RECEIVER_FAULT_MODE_SIGKILL = "sigkill"
RECEIVER_FAULT_MODE_SIGSTOP = "sigstop"

RECEIVER_FAULT_MODE_TO_SIGNAL: dict[str, int] = {
    RECEIVER_FAULT_MODE_SIGKILL: signal.SIGKILL,
    RECEIVER_FAULT_MODE_SIGSTOP: signal.SIGSTOP,
}

INJECT_FAULT_PATH = "/inject_fault"

MAX_REQUEST_BODY_BYTES = 8192
CLOSE_DEADLINE_S = 5.0

_REQUEST_TIMEOUT_S = 10.0
_SERVE_POLL_INTERVAL_S = 0.5


class ReceiverFaultControlError(RuntimeError):
    pass


class ReceiverIdentity(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    receiver_boot_uuid: str
    session_id: str
    rank: int
    control_url: str

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


class InjectFaultRequest(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    request_id: str
    expected_receiver_boot_uuid: str
    expected_session_id: str
    expected_rank: int
    mode: str


class InjectFaultDecision(msgspec.Struct, frozen=True, kw_only=True):
    scheduled: bool
    status_code: int
    body: dict[str, Any]


def _send_signal_to_self(sig: int) -> None:
    os.kill(os.getpid(), sig)


# ======================= receiver-side fault controller =======================


class ReceiverFaultController:
    def __init__(self, *, session_id: str, rank: int) -> None:
        self.receiver_boot_uuid = str(uuid.uuid4())
        self.session_id = session_id
        self.rank = rank

        self._lock = threading.Lock()
        self._active = True
        self._pending_request: Optional[InjectFaultRequest] = None
        self._action_attempted_request_id: Optional[str] = None
        self._fired_request_id: Optional[str] = None
        self._action_result: Optional[str] = None
        self._identity: Optional[ReceiverIdentity] = None
        self._httpd: Optional[_ReceiverFaultControlServer] = None
        self._stop_serving = threading.Event()
        self._serve_thread: Optional[threading.Thread] = None
        self._action_thread: Optional[threading.Thread] = None
        self._action_gate: Optional[threading.Event] = None

    @property
    def identity(self) -> Optional[ReceiverIdentity]:
        return self._identity

    @property
    def fired_request_id(self) -> Optional[str]:
        return self._fired_request_id

    @property
    def action_result(self) -> Optional[str]:
        return self._action_result

    def start(self) -> ReceiverIdentity:
        local_ip = get_local_ip_auto()
        httpd = _ReceiverFaultControlServer(host=local_ip, controller=self)
        httpd.timeout = _SERVE_POLL_INTERVAL_S
        bound_port = httpd.socket.getsockname()[1]
        identity = ReceiverIdentity(
            receiver_boot_uuid=self.receiver_boot_uuid,
            session_id=self.session_id,
            rank=self.rank,
            control_url=NetworkAddress(local_ip, bound_port).to_url(),
        )

        serve_thread = threading.Thread(
            target=self._serve,
            kwargs={"httpd": httpd},
            name=f"receiver-fault-control-rank{self.rank}",
            daemon=True,
        )
        serve_thread.start()

        self._httpd = httpd
        self._serve_thread = serve_thread
        self._identity = identity
        logger.info(
            "Receiver fault control listening: receiver_boot_uuid=%s session_id=%s "
            "rank=%s control_url=%s",
            identity.receiver_boot_uuid,
            identity.session_id,
            identity.rank,
            identity.control_url,
        )
        return identity

    def handle_inject_fault(
        self, *, request: InjectFaultRequest
    ) -> InjectFaultDecision:
        with self._lock:
            if (reason := self._identity_refusal(request=request)) is not None:
                status_code = 400 if reason == "unsupported_mode" else 409
                return self._reject(
                    request=request, reason=reason, status_code=status_code
                )

            pending = self._pending_request
            if pending is None:
                self._pending_request = request
                action_gate = threading.Event()
                action_thread = threading.Thread(
                    target=self._await_response_and_execute_action,
                    kwargs={"request": request, "action_gate": action_gate},
                    name=f"receiver-fault-action-rank{self.rank}",
                    daemon=True,
                )
                self._action_gate = action_gate
                self._action_thread = action_thread
                try:
                    action_thread.start()
                except Exception:
                    self._pending_request = None
                    self._action_gate = None
                    self._action_thread = None
                    self._action_result = "launch_failed"
                    logger.exception(
                        "Receiver fault control could not launch the action thread: "
                        "request_id=%s receiver_boot_uuid=%s rank=%s",
                        request.request_id,
                        self.receiver_boot_uuid,
                        self.rank,
                    )
                    return self._reject(
                        request=request,
                        reason="action_launch_failed",
                        status_code=503,
                    )
                scheduled = True
            elif pending == request:
                scheduled = False
            elif pending.request_id == request.request_id:
                return self._reject(
                    request=request,
                    reason="request_id_payload_conflict",
                    status_code=409,
                )
            else:
                return self._reject(
                    request=request, reason="pending_action_exists", status_code=409
                )

        logger.warning(
            "Receiver fault control accepted: request_id=%s receiver_boot_uuid=%s "
            "session_id=%s rank=%s mode=%s newly_scheduled=%s",
            request.request_id,
            self.receiver_boot_uuid,
            self.session_id,
            self.rank,
            request.mode,
            scheduled,
        )
        return InjectFaultDecision(
            scheduled=scheduled,
            status_code=200,
            body=self._response_body(request=request, status="accepted"),
        )

    def release_scheduled_action(self, *, request: InjectFaultRequest) -> None:
        with self._lock:
            if self._pending_request == request and self._action_gate is not None:
                self._action_gate.set()

    def close(self, *, deadline_s: float = CLOSE_DEADLINE_S) -> None:
        deadline = time.monotonic() + deadline_s

        if not self._lock.acquire(timeout=_remaining(deadline)):
            raise ReceiverFaultControlError(
                f"Receiver fault control for receiver_boot_uuid="
                f"{self.receiver_boot_uuid} session_id={self.session_id} "
                f"rank={self.rank} did not reach a quiescent state within "
                f"{deadline_s}s: an accepted action may still signal this process"
            )
        try:
            self._active = False
            if self._action_gate is not None:
                self._action_gate.set()
        finally:
            self._lock.release()

        self._stop_serving.set()
        if self._httpd is not None:
            self._httpd.server_close()
        for thread in (self._serve_thread, self._action_thread):
            if thread is None:
                continue
            thread.join(timeout=_remaining(deadline))
            if thread.is_alive():
                raise ReceiverFaultControlError(
                    f"Receiver fault control thread {thread.name} for "
                    f"receiver_boot_uuid={self.receiver_boot_uuid} did not stop "
                    f"within {deadline_s}s"
                )

        logger.info(
            "Receiver fault control closed: receiver_boot_uuid=%s session_id=%s rank=%s",
            self.receiver_boot_uuid,
            self.session_id,
            self.rank,
        )

    def _await_response_and_execute_action(
        self, *, request: InjectFaultRequest, action_gate: threading.Event
    ) -> None:
        action_gate.wait()
        self._execute_action(request=request)

    def _execute_action(self, *, request: InjectFaultRequest) -> None:
        with self._lock:
            if (refusal := self._action_refusal(request=request)) is not None:
                self._action_result = f"refused:{refusal}"
                logger.warning(
                    "Receiver fault control skipped action: request_id=%s reason=%s "
                    "receiver_boot_uuid=%s session_id=%s rank=%s mode=%s",
                    request.request_id,
                    refusal,
                    self.receiver_boot_uuid,
                    self.session_id,
                    self.rank,
                    request.mode,
                )
                return

            self._action_attempted_request_id = request.request_id
            logger.warning(
                "Receiver fault control firing: request_id=%s receiver_boot_uuid=%s "
                "session_id=%s rank=%s mode=%s pid=%s",
                request.request_id,
                self.receiver_boot_uuid,
                self.session_id,
                self.rank,
                request.mode,
                os.getpid(),
            )
            try:
                _send_signal_to_self(RECEIVER_FAULT_MODE_TO_SIGNAL[request.mode])
            except Exception:
                self._action_result = "signal_failed"
                logger.exception(
                    "Receiver fault control failed to signal itself: request_id=%s "
                    "receiver_boot_uuid=%s rank=%s mode=%s",
                    request.request_id,
                    self.receiver_boot_uuid,
                    self.rank,
                    request.mode,
                )
                return

            self._fired_request_id = request.request_id
            self._action_result = "signal_sent"

    def _action_refusal(self, *, request: InjectFaultRequest) -> Optional[str]:
        if self._pending_request != request:
            return "request_not_scheduled"
        if self._action_attempted_request_id is not None:
            return "already_attempted"
        return None

    def _identity_refusal(self, *, request: InjectFaultRequest) -> Optional[str]:
        if not self._active:
            return "receiver_inactive"
        if request.expected_receiver_boot_uuid != self.receiver_boot_uuid:
            return "receiver_boot_uuid_mismatch"
        if request.expected_session_id != self.session_id:
            return "session_id_mismatch"
        if request.expected_rank != self.rank:
            return "rank_mismatch"
        if request.mode not in RECEIVER_FAULT_MODE_TO_SIGNAL:
            return "unsupported_mode"
        return None

    def _reject(
        self, *, request: InjectFaultRequest, reason: str, status_code: int
    ) -> InjectFaultDecision:
        logger.warning(
            "Receiver fault control rejected: request_id=%s reason=%s "
            "receiver_boot_uuid=%s session_id=%s rank=%s mode=%s",
            request.request_id,
            reason,
            self.receiver_boot_uuid,
            self.session_id,
            self.rank,
            request.mode,
        )
        body = self._response_body(request=request, status="rejected")
        body["reason"] = reason
        return InjectFaultDecision(scheduled=False, status_code=status_code, body=body)

    def _response_body(
        self, *, request: InjectFaultRequest, status: str
    ) -> dict[str, Any]:
        return {
            "status": status,
            "request_id": request.request_id,
            "receiver_boot_uuid": self.receiver_boot_uuid,
            "session_id": self.session_id,
            "rank": self.rank,
        }

    def _serve(self, *, httpd: _ReceiverFaultControlServer) -> None:
        while not self._stop_serving.is_set():
            try:
                httpd.handle_request()
            except OSError:
                if not self._stop_serving.is_set():
                    logger.exception(
                        "Receiver fault control listener stopped: "
                        "receiver_boot_uuid=%s rank=%s",
                        self.receiver_boot_uuid,
                        self.rank,
                    )
                return


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


# =========================== control http transport ===========================


class _ReceiverFaultControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, host: str, controller: ReceiverFaultController) -> None:
        self.address_family = NetworkAddress(host, 0).family
        self.controller = controller
        super().__init__((host, 0), _ReceiverFaultControlHandler)


class _ReceiverFaultControlHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = _REQUEST_TIMEOUT_S

    def do_POST(self) -> None:
        if self.path != INJECT_FAULT_PATH:
            self._respond(status_code=404, body={"status": "unknown_path"})
            return

        content_length = _parse_content_length(self.headers.get("Content-Length"))
        if content_length is None:
            self._respond(
                status_code=400,
                body={"status": "invalid_request", "reason": "invalid_content_length"},
            )
            return
        if content_length > MAX_REQUEST_BODY_BYTES:
            self._respond(status_code=413, body={"status": "request_too_large"})
            return

        try:
            body = self.rfile.read(content_length)
        except OSError:
            self._respond(
                status_code=408,
                body={"status": "invalid_request", "reason": "body_read_timeout"},
            )
            return
        if len(body) != content_length:
            self._respond(
                status_code=400,
                body={"status": "invalid_request", "reason": "incomplete_body"},
            )
            return

        try:
            request = msgspec.json.decode(body, type=InjectFaultRequest)
        except msgspec.DecodeError as e:
            self._respond(
                status_code=400, body={"status": "invalid_request", "reason": str(e)}
            )
            return

        controller: ReceiverFaultController = self.server.controller
        self._handle_inject_fault(controller=controller, request=request)

    def _handle_inject_fault(
        self,
        *,
        controller: ReceiverFaultController,
        request: InjectFaultRequest,
    ) -> None:
        decision = controller.handle_inject_fault(request=request)
        try:
            self._respond(status_code=decision.status_code, body=decision.body)
        finally:
            if decision.status_code == 200:
                controller.release_scheduled_action(request=request)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("Receiver fault control http: " + format, *args)

    def _respond(self, *, status_code: int, body: dict[str, Any]) -> None:
        payload = msgspec.json.encode(body)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()


def _parse_content_length(header_value: Optional[str]) -> Optional[int]:
    if header_value is None:
        return None
    try:
        content_length = int(header_value)
    except ValueError:
        return None
    if content_length < 0:
        return None
    return content_length
