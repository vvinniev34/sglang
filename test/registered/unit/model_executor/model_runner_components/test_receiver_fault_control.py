import contextlib
import json
import logging
import signal
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace
from typing import Callable, Optional

import msgspec
import pytest

from sglang.srt.entrypoints.engine_info_bootstrap_server import TransferEngineRecord
from sglang.srt.model_executor.model_runner_components import receiver_fault_control
from sglang.srt.model_executor.model_runner_components.receiver_fault_control import (
    INJECT_FAULT_PATH,
    MAX_REQUEST_BODY_BYTES,
    RECEIVER_FAULT_MODE_SIGKILL,
    RECEIVER_FAULT_MODE_SIGSTOP,
    InjectFaultRequest,
    ReceiverFaultControlError,
    ReceiverFaultController,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

SESSION_ID = "10.0.0.7:17123"
RANK = 3


@pytest.fixture(autouse=True)
def sent_signals(monkeypatch):
    """Replace the self-signal boundary so no test can ever signal the pytest process."""
    recorded: list[int] = []

    def record_signal(sig: int) -> None:
        recorded.append(sig)

    monkeypatch.setattr(receiver_fault_control, "_send_signal_to_self", record_signal)
    assert receiver_fault_control._send_signal_to_self is record_signal
    return recorded


@pytest.fixture
def controller():
    controller = ReceiverFaultController(session_id=SESSION_ID, rank=RANK)
    yield controller
    controller.close()


def _request(
    controller: ReceiverFaultController,
    *,
    request_id: str = "req-1",
    receiver_boot_uuid: Optional[str] = None,
    session_id: Optional[str] = None,
    rank: Optional[int] = None,
    mode: str = RECEIVER_FAULT_MODE_SIGKILL,
) -> InjectFaultRequest:
    return InjectFaultRequest(
        request_id=request_id,
        expected_receiver_boot_uuid=(
            controller.receiver_boot_uuid
            if receiver_boot_uuid is None
            else receiver_boot_uuid
        ),
        expected_session_id=SESSION_ID if session_id is None else session_id,
        expected_rank=RANK if rank is None else rank,
        mode=mode,
    )


def _transporter(*, enable_p2p_fault_injection: bool, tp_rank: int = 0):
    return RemoteInstanceWeightTransporter(
        server_args=SimpleNamespace(
            enable_p2p_fault_injection=enable_p2p_fault_injection
        ),
        get_model=lambda: None,
        tp_rank=tp_rank,
        gpu_id=0,
        engine=object(),
        session_id=SESSION_ID,
    )


# ======================= bootstrap metadata wire shape ========================


def test_registration_record_keeps_two_element_wire_info_next_to_identity():
    """One atomic record carries weights and identity, and the legacy field stays a pair."""
    identity = {
        "receiver_boot_uuid": "uuid-a",
        "session_id": SESSION_ID,
        "rank": RANK,
        "control_url": "http://10.0.0.7:41111",
    }
    record = TransferEngineRecord.from_registration(
        info={
            "session_id": SESSION_ID,
            "weights_info_dict": {"w": [1, 2, 3]},
            "receiver_identity": identity,
        }
    )

    response = record.to_response(rank=RANK)

    session_id, weights_info_dict = response["remote_instance_transfer_engine_info"]
    assert session_id == SESSION_ID
    assert weights_info_dict == {"w": [1, 2, 3]}
    assert response["receiver_identity"] == identity
    assert response["rank"] == RANK


def test_registration_without_identity_reports_none_instead_of_faking_one():
    """A rank registering without fault control publishes receiver_identity=None."""
    record = TransferEngineRecord.from_registration(
        info={"session_id": SESSION_ID, "weights_info_dict": {}}
    )

    assert record.receiver_identity is None
    assert record.to_response(rank=0)["receiver_identity"] is None


def test_records_of_different_ranks_keep_their_own_identity():
    """Per-rank records never cross-serve another rank's control endpoint."""
    records = {
        rank: TransferEngineRecord.from_registration(
            info={
                "session_id": f"10.0.0.7:1712{rank}",
                "weights_info_dict": {},
                "receiver_identity": {
                    "receiver_boot_uuid": f"uuid-{rank}",
                    "session_id": f"10.0.0.7:1712{rank}",
                    "rank": rank,
                    "control_url": f"http://10.0.0.7:4111{rank}",
                },
            }
        )
        for rank in (0, 1)
    }

    for rank, record in records.items():
        response = record.to_response(rank=rank)
        assert response["receiver_identity"]["rank"] == rank
        assert response["receiver_identity"]["receiver_boot_uuid"] == f"uuid-{rank}"
        assert response["remote_instance_transfer_engine_info"][0] == (
            f"10.0.0.7:1712{rank}"
        )


# ============================= request validation =============================


def test_inject_fault_request_rejects_unknown_fields():
    """An unknown field such as a caller-chosen pid must not decode."""
    payload = json.dumps(
        {
            "request_id": "req-1",
            "expected_receiver_boot_uuid": "uuid-a",
            "expected_session_id": SESSION_ID,
            "expected_rank": RANK,
            "mode": RECEIVER_FAULT_MODE_SIGKILL,
            "pid": 1234,
        }
    ).encode()

    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(payload, type=InjectFaultRequest)


def test_inject_fault_request_rejects_missing_identity_fields():
    """Every expected-identity field is mandatory, so an unbound request cannot decode."""
    payload = json.dumps(
        {"request_id": "req-1", "mode": RECEIVER_FAULT_MODE_SIGKILL}
    ).encode()

    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(payload, type=InjectFaultRequest)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (
            {"receiver_boot_uuid": "uuid-of-a-dead-incarnation"},
            "receiver_boot_uuid_mismatch",
        ),
        ({"session_id": "10.0.0.7:9999"}, "session_id_mismatch"),
        ({"rank": RANK + 1}, "rank_mismatch"),
    ],
)
def test_identity_mismatch_is_rejected_without_signalling(
    controller, sent_signals, kwargs, reason
):
    """A request bound to another incarnation or rank never reaches the signal boundary."""
    request = _request(controller, **kwargs)

    decision = controller.handle_inject_fault(request=request)

    assert decision.status_code == 409
    assert decision.scheduled is False
    assert decision.body["status"] == "rejected"
    assert decision.body["reason"] == reason

    controller._execute_action(request=request)
    assert sent_signals == []
    assert controller.fired_request_id is None


@pytest.mark.parametrize("mode", ["sigterm", "kill", "", "9"])
def test_unsupported_mode_is_rejected_without_signalling(
    controller, sent_signals, mode
):
    """Only sigkill and sigstop are supported; anything else is refused explicitly."""
    request = _request(controller, mode=mode)

    decision = controller.handle_inject_fault(request=request)

    assert decision.status_code == 400
    assert decision.body["reason"] == "unsupported_mode"

    controller._execute_action(request=request)
    assert sent_signals == []


@pytest.mark.parametrize(
    "mode, expected_signal",
    [
        (RECEIVER_FAULT_MODE_SIGKILL, signal.SIGKILL),
        (RECEIVER_FAULT_MODE_SIGSTOP, signal.SIGSTOP),
    ],
)
def test_matching_request_signals_this_process_only(
    controller, sent_signals, mode, expected_signal
):
    """A fully matching request signals the receiver's own pid with the mapped signal."""
    request = _request(controller, mode=mode)

    decision = controller.handle_inject_fault(request=request)
    assert decision.scheduled is True

    controller.release_scheduled_action(request=request)
    _wait_until(lambda: len(sent_signals) == 1)
    assert sent_signals == [expected_signal]
    assert controller.fired_request_id == request.request_id


def test_accepted_response_does_not_imply_fired(controller, sent_signals):
    """Accepting the request must not be reported as a fired fault."""
    decision = controller.handle_inject_fault(request=_request(controller))

    assert decision.status_code == 200
    assert decision.body["status"] == "accepted"
    assert controller.fired_request_id is None
    assert sent_signals == []


# ========================= idempotency and lifecycle ==========================


def test_repeated_request_id_is_accepted_but_scheduled_once(controller, sent_signals):
    """A retried request_id must not schedule or fire a second action."""
    request = _request(controller, request_id="req-retry")

    first = controller.handle_inject_fault(request=request)
    second = controller.handle_inject_fault(request=request)

    assert first.scheduled is True
    assert second.scheduled is False
    assert second.status_code == 200
    assert second.body["status"] == "accepted"

    controller.release_scheduled_action(request=request)
    controller.release_scheduled_action(request=request)
    _wait_until(lambda: len(sent_signals) == 1)
    assert sent_signals == [signal.SIGKILL]


def test_second_distinct_request_is_refused_while_one_is_pending(
    controller, sent_signals
):
    """A different request_id cannot queue behind an already accepted action."""
    controller.handle_inject_fault(request=_request(controller, request_id="req-1"))

    other = _request(controller, request_id="req-2")
    decision = controller.handle_inject_fault(request=other)

    assert decision.status_code == 409
    assert decision.body["reason"] == "pending_action_exists"

    controller._execute_action(request=other)
    assert sent_signals == []


def test_new_receiver_at_the_same_address_rejects_the_old_request(sent_signals):
    """A restarted receiver reusing session id and rank refuses the previous incarnation's request."""
    old = ReceiverFaultController(session_id=SESSION_ID, rank=RANK)
    old_request = _request(old)
    new = ReceiverFaultController(session_id=SESSION_ID, rank=RANK)

    assert new.receiver_boot_uuid != old.receiver_boot_uuid

    decision = new.handle_inject_fault(request=old_request)
    assert decision.body["reason"] == "receiver_boot_uuid_mismatch"

    new._execute_action(request=old_request)
    assert sent_signals == []

    old.close()
    new.close()


def test_close_does_not_cancel_an_already_accepted_action(sent_signals):
    """Closing the controller releases an accepted action for exactly one attempt."""
    controller = ReceiverFaultController(session_id=SESSION_ID, rank=RANK)
    request = _request(controller)
    assert controller.handle_inject_fault(request=request).scheduled is True

    controller.close()

    assert sent_signals == [signal.SIGKILL]
    assert controller.fired_request_id == request.request_id

    reopened = controller.handle_inject_fault(request=request)
    assert reopened.body["reason"] == "receiver_inactive"


def test_action_thread_start_failure_rejects_and_clears_the_reservation(
    controller, sent_signals, monkeypatch
):
    """A launch failure returns a rejection and leaves the request free to retry."""
    original_start = threading.Thread.start
    action_start_attempts = 0

    def fail_first_action_start(thread) -> None:
        nonlocal action_start_attempts
        if thread.name.startswith("receiver-fault-action-"):
            action_start_attempts += 1
            if action_start_attempts == 1:
                raise RuntimeError("thread launch failed")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_action_start)
    request = _request(controller, request_id="req-launch-retry")

    failed = controller.handle_inject_fault(request=request)

    assert failed.status_code == 503
    assert failed.body["status"] == "rejected"
    assert failed.body["reason"] == "action_launch_failed"
    assert controller._pending_request is None
    assert controller._action_thread is None
    assert controller.action_result == "launch_failed"

    retried = controller.handle_inject_fault(request=request)
    assert retried.status_code == 200
    assert retried.scheduled is True
    controller.release_scheduled_action(request=request)
    _wait_until(lambda: len(sent_signals) == 1)
    assert sent_signals == [signal.SIGKILL]


# ======================== control endpoint and wiring =========================


def test_control_endpoint_serves_requests_on_its_own_daemon_thread(sent_signals):
    """The endpoint answers over HTTP without any scheduler, event loop or collective."""
    controller = ReceiverFaultController(session_id=SESSION_ID, rank=RANK)
    identity = controller.start()
    try:
        assert identity.receiver_boot_uuid == controller.receiver_boot_uuid
        assert identity.session_id == SESSION_ID
        assert identity.rank == RANK
        assert controller._serve_thread.daemon is True

        stale = _request(controller, receiver_boot_uuid="uuid-of-a-dead-incarnation")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(url=identity.control_url + INJECT_FAULT_PATH, request=stale)
        assert excinfo.value.code == 409
        assert sent_signals == []

        body = _post(
            url=identity.control_url + INJECT_FAULT_PATH, request=_request(controller)
        )
        assert body["status"] == "accepted"
        assert body["receiver_boot_uuid"] == controller.receiver_boot_uuid
        _wait_until(lambda: sent_signals == [signal.SIGKILL])
    finally:
        controller.close()


def test_response_write_failure_still_attempts_the_action_once(
    controller, sent_signals
):
    """A disconnected client cannot strand an accepted action before its attempt."""
    handler = object.__new__(receiver_fault_control._ReceiverFaultControlHandler)
    request = _request(controller, request_id="req-broken-response")

    def fail_response(*, status_code, body) -> None:
        assert status_code == 200
        assert body["status"] == "accepted"
        raise BrokenPipeError("client disconnected")

    handler._respond = fail_response
    with pytest.raises(BrokenPipeError):
        handler._handle_inject_fault(controller=controller, request=request)

    _wait_until(lambda: len(sent_signals) == 1)
    assert sent_signals == [signal.SIGKILL]

    responses = []
    handler._respond = lambda *, status_code, body: responses.append(
        (status_code, body)
    )
    handler._handle_inject_fault(controller=controller, request=request)
    assert responses[0][0] == 200
    assert responses[0][1]["status"] == "accepted"
    assert sent_signals == [signal.SIGKILL]


def test_transporter_does_not_start_fault_control_by_default():
    """Without the explicit server arg the receiver publishes no control endpoint."""
    transporter = _transporter(enable_p2p_fault_injection=False)

    transporter.start_fault_controller()

    assert transporter.fault_controller is None
    assert transporter._receiver_identity_payload() is None


def test_transporter_publishes_a_distinct_identity_per_rank():
    """Each receiver rank publishes its own boot uuid and control url."""
    transporters = [
        _transporter(enable_p2p_fault_injection=True, tp_rank=rank) for rank in (0, 1)
    ]
    try:
        for transporter in transporters:
            transporter.start_fault_controller()

        payloads = [t._receiver_identity_payload() for t in transporters]
        assert [p["rank"] for p in payloads] == [0, 1]
        assert all(p["session_id"] == SESSION_ID for p in payloads)
        assert payloads[0]["receiver_boot_uuid"] != payloads[1]["receiver_boot_uuid"]
        assert payloads[0]["control_url"] != payloads[1]["control_url"]
    finally:
        for transporter in transporters:
            transporter.close_fault_controller()


def test_engine_reinit_retires_the_old_identity_before_publishing_a_new_session():
    """Re-initializing the engine invalidates the old controller and republishes weights under the new session."""
    transporter = _transporter(enable_p2p_fault_injection=True)
    try:
        with _fake_mooncake(rpc_ports=[17123, 17124]):
            transporter.init_engine()
            first_controller = transporter.fault_controller
            first_payload = transporter._receiver_identity_payload()
            transporter.weight_info = {"w": [1, 2, 3]}

            transporter.init_engine()
            second_payload = transporter._receiver_identity_payload()

        assert first_controller is not transporter.fault_controller
        assert first_controller._active is False
        assert first_payload["session_id"] != second_payload["session_id"]
        assert (
            first_payload["receiver_boot_uuid"] != second_payload["receiver_boot_uuid"]
        )
        assert second_payload["session_id"] == transporter.session_id
        assert transporter.weight_info is None
    finally:
        transporter.close_fault_controller()


# ===================== signal boundary and engine rebuild =====================


def test_close_refuses_to_return_while_an_action_holds_the_signal_boundary(monkeypatch):
    """Invalidation cannot complete between the final identity check and the signal."""
    controller = ReceiverFaultController(session_id=SESSION_ID, rank=RANK)
    inside_signal = threading.Event()
    release_signal = threading.Event()
    events = []

    def blocking_signal(sig: int) -> None:
        events.append("signal_entered")
        inside_signal.set()
        assert release_signal.wait(timeout=30)
        events.append("signal_returned")

    monkeypatch.setattr(receiver_fault_control, "_send_signal_to_self", blocking_signal)

    request = _request_for(controller)
    assert controller.handle_inject_fault(request=request).scheduled is True
    controller.release_scheduled_action(request=request)
    assert inside_signal.wait(timeout=30)

    with pytest.raises(ReceiverFaultControlError):
        controller.close(deadline_s=0.1)
    events.append("close_refused")

    release_signal.set()
    controller.close(deadline_s=30)
    events.append("closed")

    assert events == [
        "signal_entered",
        "close_refused",
        "signal_returned",
        "closed",
    ]
    assert controller.fired_request_id == request.request_id


def test_engine_reinit_stops_when_the_old_controller_cannot_be_invalidated(monkeypatch):
    """A receiver that may still signal blocks the rebuild instead of getting a new session."""
    transporter = _transporter(enable_p2p_fault_injection=True)
    inside_signal = threading.Event()
    release_signal = threading.Event()

    def blocking_signal(sig: int) -> None:
        inside_signal.set()
        assert release_signal.wait(timeout=30)

    monkeypatch.setattr(receiver_fault_control, "_send_signal_to_self", blocking_signal)

    try:
        with _fake_mooncake(rpc_ports=[17123, 17124]):
            transporter.init_engine()
            stuck_controller = transporter.fault_controller
            stuck_session_id = transporter.session_id
            stuck_engine = transporter.engine

            request = _request_for(stuck_controller)
            assert stuck_controller.handle_inject_fault(request=request).scheduled
            stuck_controller.release_scheduled_action(request=request)
            assert inside_signal.wait(timeout=30)

            with pytest.raises(ReceiverFaultControlError):
                transporter.init_engine()

            assert transporter.session_id == stuck_session_id
            assert transporter.engine is stuck_engine
            assert transporter.fault_controller is stuck_controller
    finally:
        release_signal.set()
        if transporter.fault_controller is not None:
            transporter.close_fault_controller()


def test_signal_failure_is_recorded_instead_of_reported_as_fired(
    controller, monkeypatch, caplog
):
    """A failed self-signal is logged with a stacktrace and never counted as fired."""

    def failing_signal(sig: int) -> None:
        raise OSError("operation not permitted")

    monkeypatch.setattr(receiver_fault_control, "_send_signal_to_self", failing_signal)
    request = _request(controller)
    assert controller.handle_inject_fault(request=request).scheduled is True

    with caplog.at_level(logging.ERROR):
        controller.release_scheduled_action(request=request)
        _wait_until(lambda: controller.action_result == "signal_failed")

    assert controller.fired_request_id is None
    assert controller.action_result == "signal_failed"
    assert "Traceback" in caplog.text


# ============================== request boundary ==============================


def test_repeated_request_id_with_a_different_payload_is_refused(
    controller, sent_signals
):
    """Only a byte-identical retry is idempotent; a conflicting payload under the same id is refused."""
    first = _request(controller, request_id="req-dup", mode=RECEIVER_FAULT_MODE_SIGKILL)
    conflicting = _request(
        controller, request_id="req-dup", mode=RECEIVER_FAULT_MODE_SIGSTOP
    )
    assert controller.handle_inject_fault(request=first).scheduled is True

    decision = controller.handle_inject_fault(request=conflicting)

    assert decision.status_code == 409
    assert decision.body["reason"] == "request_id_payload_conflict"

    controller._execute_action(request=conflicting)
    assert sent_signals == []


def test_parse_content_length_rejects_missing_and_negative_values():
    """A missing or negative Content-Length must not reach rfile.read()."""
    assert receiver_fault_control._parse_content_length(None) is None
    assert receiver_fault_control._parse_content_length("") is None
    assert receiver_fault_control._parse_content_length("-1") is None
    assert receiver_fault_control._parse_content_length("12") == 12


@pytest.mark.parametrize(
    "content_length, expected_status",
    [
        ("-5", "400"),
        ("not-a-number", "400"),
        (str(MAX_REQUEST_BODY_BYTES + 1), "413"),
    ],
)
def test_malformed_content_length_is_answered_without_hanging(
    sent_signals, content_length, expected_status
):
    """A malformed or oversized length is rejected instead of blocking the handler thread."""
    controller = ReceiverFaultController(session_id=SESSION_ID, rank=RANK)
    identity = controller.start()
    try:
        raw = (
            f"POST {INJECT_FAULT_PATH} HTTP/1.1\r\n"
            f"Host: receiver\r\n"
            f"Content-Length: {content_length}\r\n"
            f"\r\n"
        ).encode()

        status_line = _raw_request_status_line(
            control_url=identity.control_url, raw=raw
        )

        assert expected_status in status_line
        assert sent_signals == []
    finally:
        controller.close()


def _request_for(
    controller: ReceiverFaultController,
    *,
    request_id: str = "req-1",
    mode: str = RECEIVER_FAULT_MODE_SIGKILL,
) -> InjectFaultRequest:
    return InjectFaultRequest(
        request_id=request_id,
        expected_receiver_boot_uuid=controller.receiver_boot_uuid,
        expected_session_id=controller.session_id,
        expected_rank=controller.rank,
        mode=mode,
    )


@contextlib.contextmanager
def _fake_mooncake(*, rpc_ports: list):
    ports = iter(rpc_ports)

    class _FakeTransferEngine:
        def initialize(self, *args) -> None:
            self.rpc_port = next(ports)

        def get_rpc_port(self) -> int:
            return self.rpc_port

    engine_module = types.ModuleType("mooncake.engine")
    engine_module.TransferEngine = _FakeTransferEngine
    package = types.ModuleType("mooncake")
    package.engine = engine_module

    saved = {name: sys.modules.get(name) for name in ("mooncake", "mooncake.engine")}
    sys.modules["mooncake"] = package
    sys.modules["mooncake.engine"] = engine_module
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


def _raw_request_status_line(*, control_url: str, raw: bytes) -> str:
    parts = urllib.parse.urlsplit(control_url)
    with socket.create_connection((parts.hostname, parts.port), timeout=10) as conn:
        conn.settimeout(10)
        conn.sendall(raw)
        return conn.recv(4096).decode(errors="replace").splitlines()[0]


def _post(*, url: str, request: InjectFaultRequest) -> dict:
    http_request = urllib.request.Request(
        url,
        data=msgspec.json.encode(request),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=10) as response:
        return json.loads(response.read())


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before the timeout")
