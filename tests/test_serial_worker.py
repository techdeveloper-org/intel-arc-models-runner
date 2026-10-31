"""Failing tests for WP-03: serial inference worker, stream bridge, bounded stop.

Red-first: these tests MUST fail before the implementation exists.
Run: pytest tests/test_serial_worker.py -v
"""

import os
import threading
import time
from unittest.mock import patch

import pytest
import pytest_asyncio


class TestEngineUnavailableError:
    """EngineUnavailableError must exist in core.exceptions."""

    def test_engine_unavailable_error_exists(self) -> None:
        """EngineUnavailableError is importable from arc_runner.core.exceptions."""
        from arc_runner.core.exceptions import EngineUnavailableError  # noqa: F401

    def test_engine_unavailable_error_is_503(self) -> None:
        """EngineUnavailableError carries HTTP 503."""
        from arc_runner.core.exceptions import EngineUnavailableError

        exc = EngineUnavailableError("wedged")
        assert exc.status_code == 503

    def test_engine_wedged_code(self) -> None:
        """Default error_code is ENGINE_WEDGED."""
        from arc_runner.core.exceptions import EngineUnavailableError

        exc = EngineUnavailableError("stuck")
        assert exc.error_code == "ENGINE_WEDGED"

    def test_server_stopping_code(self) -> None:
        """SERVER_STOPPING variant has correct code."""
        from arc_runner.core.exceptions import EngineUnavailableError

        exc = EngineUnavailableError("stopping", error_code="SERVER_STOPPING")
        assert exc.error_code == "SERVER_STOPPING"
        assert exc.status_code == 503


class TestGracefulShutdownConstant:
    """TIMEOUT_GRACEFUL_SHUTDOWN_S must exist in core.constants."""

    def test_constant_exists(self) -> None:
        """TIMEOUT_GRACEFUL_SHUTDOWN_S is importable and is a positive float."""
        from arc_runner.core.constants import TIMEOUT_GRACEFUL_SHUTDOWN_S

        assert isinstance(TIMEOUT_GRACEFUL_SHUTDOWN_S, (int, float))
        assert TIMEOUT_GRACEFUL_SHUTDOWN_S > 0


class TestSerialWorkerImport:
    """SerialWorker must be importable from arc_runner.server.worker."""

    def test_serial_worker_importable(self) -> None:
        """SerialWorker class exists in server.worker."""
        from arc_runner.server.worker import SerialWorker  # noqa: F401

    def test_serial_worker_has_start(self) -> None:
        """SerialWorker exposes a start() method."""
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        assert callable(getattr(w, "start", None))

    def test_serial_worker_has_stop(self) -> None:
        """SerialWorker exposes an async stop() coroutine."""
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        assert callable(getattr(w, "stop", None))

    def test_serial_worker_has_submit(self) -> None:
        """SerialWorker exposes an async submit() coroutine."""
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        assert callable(getattr(w, "submit", None))


@pytest.mark.asyncio
class TestSerialWorkerDaemonThread:
    """The worker thread must be a daemon thread."""

    async def test_worker_thread_is_daemon(self) -> None:
        """After start(), the background thread is a daemon."""
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        w.start()
        try:
            assert w._thread.daemon is True
        finally:
            await w.stop(timeout=2.0)

    async def test_worker_thread_is_alive_after_start(self) -> None:
        """After start() the thread is running."""
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        w.start()
        try:
            assert w._thread.is_alive()
        finally:
            await w.stop(timeout=2.0)


@pytest.mark.asyncio
class TestSerialWorkerJobExecution:
    """Jobs submitted to the worker must run on the worker thread."""

    async def test_job_runs_and_returns_result(self) -> None:
        """submit() resolves with the callable's return value."""
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        w.start()
        try:
            result = await w.submit(lambda: 42)
            assert result == 42
        finally:
            await w.stop(timeout=2.0)

    async def test_job_runs_on_worker_thread(self) -> None:
        """The callable executes on the worker's daemon thread, not the caller thread."""
        from arc_runner.server.worker import SerialWorker

        caller_thread_id = threading.current_thread().ident
        worker_thread_id: list[int] = []

        def capture_thread() -> None:
            worker_thread_id.append(threading.current_thread().ident)

        w = SerialWorker()
        w.start()
        try:
            await w.submit(capture_thread)
            assert worker_thread_id[0] != caller_thread_id
        finally:
            await w.stop(timeout=2.0)

    async def test_jobs_run_serially_not_concurrently(self) -> None:
        """Two jobs submitted concurrently still execute sequentially."""
        import asyncio
        from arc_runner.server.worker import SerialWorker

        order: list[int] = []

        def job_a() -> None:
            order.append(1)
            time.sleep(0.05)

        def job_b() -> None:
            order.append(2)

        w = SerialWorker()
        w.start()
        try:
            t1 = asyncio.create_task(w.submit(job_a))
            t2 = asyncio.create_task(w.submit(job_b))
            await asyncio.gather(t1, t2)
            assert order == [1, 2]
        finally:
            await w.stop(timeout=2.0)

    async def test_exception_in_job_propagates_to_caller(self) -> None:
        """If the callable raises, submit() raises that exception."""
        from arc_runner.server.worker import SerialWorker

        def bad_job() -> None:
            raise ValueError("intentional failure")

        w = SerialWorker()
        w.start()
        try:
            with pytest.raises(ValueError, match="intentional failure"):
                await w.submit(bad_job)
        finally:
            await w.stop(timeout=2.0)


@pytest.mark.asyncio
class TestSerialWorkerBoundedStop:
    """stop() must honour the graceful shutdown timeout."""

    async def test_stop_joins_within_timeout(self) -> None:
        """stop() completes and the thread stops within the allotted timeout."""
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        w.start()
        await w.stop(timeout=2.0)
        assert not w._thread.is_alive()

    async def test_submit_after_stop_raises_engine_unavailable(self) -> None:
        """submit() after stop() raises EngineUnavailableError with SERVER_STOPPING."""
        from arc_runner.core.exceptions import EngineUnavailableError
        from arc_runner.server.worker import SerialWorker

        w = SerialWorker()
        w.start()
        await w.stop(timeout=2.0)

        with pytest.raises(EngineUnavailableError) as exc_info:
            await w.submit(lambda: None)
        assert exc_info.value.error_code == "SERVER_STOPPING"


@pytest.mark.asyncio
class TestSerialWorkerStreamBridge:
    """Stream bridge: all tokens must reach the caller, no loss."""

    async def test_stream_all_tokens_delivered(self) -> None:
        """A streaming job that yields N tokens delivers all N tokens to the caller."""
        from arc_runner.server.worker import SerialWorker

        received: list[str] = []
        tokens = [f"tok{i}" for i in range(50)]

        w = SerialWorker()
        w.start()
        try:
            async for token in await w.submit_stream(iter(tokens)):
                received.append(token)
            assert received == tokens
        finally:
            await w.stop(timeout=2.0)

    async def test_stream_sentinel_closes_iterator(self) -> None:
        """The stream terminates cleanly with no hanging."""
        from arc_runner.server.worker import SerialWorker

        count = 0

        w = SerialWorker()
        w.start()
        try:
            async for _ in await w.submit_stream(iter(["a", "b", "c"])):
                count += 1
            assert count == 3
        finally:
            await w.stop(timeout=2.0)


class TestHealthEndpointWorkerStatus:
    """GET /health must reflect the worker's state (CC-4)."""

    def test_health_open_without_api_key(self) -> None:
        """GET /health succeeds without an API key (CC-4 open-path requirement)."""
        from fastapi.testclient import TestClient
        from arc_runner.server.app import create_app

        with patch.dict(os.environ, {"ARC_API_KEY": "secret-key-for-test"}):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_health_returns_ok_by_default(self) -> None:
        """Without a worker, /health returns status=ok."""
        from fastapi.testclient import TestClient
        from arc_runner.server.app import create_app

        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestEngineUnavailableErrorMapsTo503:
    """EngineUnavailableError raised in a route must map to HTTP 503."""

    def test_engine_unavailable_maps_to_503(self) -> None:
        """The app exception handler maps EngineUnavailableError to 503."""
        from fastapi.testclient import TestClient
        from arc_runner.core.exceptions import EngineUnavailableError
        from arc_runner.server.app import create_app

        app = create_app()

        @app.get("/test-wedge")
        async def wedge_route():
            raise EngineUnavailableError("test wedge")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-wedge")
        assert resp.status_code == 503
