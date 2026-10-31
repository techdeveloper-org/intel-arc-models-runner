"""WP-03 tests: serial inference worker, stream bridge, load-phase wrapping, health, shutdown.

Contracts under test: CC-1 (serial FIFO execution), CC-2 (AsyncTokenStream ordered channel),
CC-3 (/health schema and bounded sync stop), CC-4 (LoadPhaseError wrapping).
Synchronisation uses threading.Event with timeouts; no sleeps.
"""

import asyncio
import os
import queue
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from arc_runner.core.exceptions import ArcRunnerError, EngineUnavailableError

WAIT_S = 2.0


class FakeEngine:
    """Real (non-mock) engine stand-in with the three methods routes call."""

    def __init__(self, tokens=None):
        """Store the tokens that generate_stream will yield."""
        self.tokens = tokens if tokens is not None else ["a", "b", "c"]

    def generate(self, prompt, config):
        """Return a fixed completion."""
        return "result"

    def generate_stream(self, prompt, config):
        """Yield the configured tokens in order."""
        for token in self.tokens:
            yield token

    def get_telemetry(self):
        """Return empty telemetry."""
        return {}


async def _stop(worker, join_timeout_s=WAIT_S):
    """Run the synchronous stop() off the event loop and return its result."""
    return await asyncio.to_thread(worker.stop, join_timeout_s)


class TestEngineUnavailableError:
    """EngineUnavailableError carries 503 and the two worker error codes."""

    def test_engine_unavailable_error_default_code_is_engine_wedged(self):
        """Default error_code is ENGINE_WEDGED with status 503."""
        exc = EngineUnavailableError("stuck")
        assert exc.status_code == 503
        assert exc.error_code == "ENGINE_WEDGED"

    def test_engine_unavailable_error_server_stopping_code_kept(self):
        """SERVER_STOPPING is preserved and still maps to 503."""
        exc = EngineUnavailableError("stopping", error_code="SERVER_STOPPING")
        assert exc.error_code == "SERVER_STOPPING"
        assert exc.status_code == 503


class TestShutdownConstants:
    """Constants and the env-driven graceful shutdown reader in core.constants."""

    def test_constants_join_values_are_positive_floats(self):
        """CANCEL_JOIN_TIMEOUT_S and SHUTDOWN_JOIN_S are positive and equal."""
        from arc_runner.core import constants

        assert constants.CANCEL_JOIN_TIMEOUT_S == 30.0
        assert constants.SHUTDOWN_JOIN_S == constants.CANCEL_JOIN_TIMEOUT_S
        assert constants.GRACEFUL_SHUTDOWN_S == 8.0

    def test_graceful_shutdown_s_unset_returns_default(self, monkeypatch):
        """Without the env var the default of 8 seconds is returned."""
        from arc_runner.core.constants import graceful_shutdown_s

        monkeypatch.delenv("ARC_GRACEFUL_SHUTDOWN_S", raising=False)
        assert graceful_shutdown_s() == 8.0

    def test_graceful_shutdown_s_valid_env_returns_value(self, monkeypatch):
        """A valid ASCII integer is parsed to float."""
        from arc_runner.core.constants import graceful_shutdown_s

        monkeypatch.setenv("ARC_GRACEFUL_SHUTDOWN_S", "5")
        assert graceful_shutdown_s() == 5.0

    @pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5", "1234567890"])
    def test_graceful_shutdown_s_invalid_env_raises(self, monkeypatch, bad):
        """Zero, negatives, non-digits, decimals and over 9 digits are rejected."""
        from arc_runner.core.constants import graceful_shutdown_s

        monkeypatch.setenv("ARC_GRACEFUL_SHUTDOWN_S", bad)
        with pytest.raises(ArcRunnerError):
            graceful_shutdown_s()


class TestLoadPhaseError:
    """CC-4: ctx.load wraps any failure so routes can tell load from post-load errors."""

    def test_load_returns_value_when_fn_succeeds(self):
        """ctx.load passes the return value through."""
        from arc_runner.server.worker import JobContext

        assert JobContext().load(lambda: 7) == 7

    def test_load_wraps_exception_as_load_phase_error(self):
        """The original exception is exposed as __cause__."""
        from arc_runner.server.worker import JobContext, LoadPhaseError

        original = ValueError("boom")

        def fail():
            raise original

        with pytest.raises(LoadPhaseError) as info:
            JobContext().load(fail)
        assert info.value.__cause__ is original

    def test_load_phase_error_str_is_never_empty(self):
        """An exception with an empty message still yields a non-empty str."""
        from arc_runner.server.worker import LoadPhaseError

        assert str(LoadPhaseError(RuntimeError(""))) != ""

    @given(
        exc_type=st.sampled_from([ValueError, KeyError, RuntimeError, OSError]),
        message=st.text(max_size=20),
    )
    @settings(max_examples=50, deadline=None)
    def test_load_property_cause_preserved_for_any_exception(self, exc_type, message):
        """P2: for any exception E raised by fn, LoadPhaseError.__cause__ is E."""
        from arc_runner.server.worker import JobContext, LoadPhaseError

        original = exc_type(message)

        def fail():
            raise original

        with pytest.raises(LoadPhaseError) as info:
            JobContext().load(fail)
        assert info.value.__cause__ is original
        assert str(info.value) != ""


class TestSerialWorkerStructure:
    """CC-1: thread name, daemon flag, queue type and admission lock."""

    async def test_worker_thread_is_named_arc_inference_daemon(self):
        """The single worker thread is a daemon named arc-inference."""
        from arc_runner.server.worker import SerialWorker

        worker = SerialWorker()
        worker.start()
        try:
            assert worker._thread.name == "arc-inference"
            assert worker._thread.daemon is True
            assert worker._thread.is_alive()
        finally:
            await _stop(worker)

    def test_worker_uses_simple_queue_and_admission_lock(self):
        """The job queue is a SimpleQueue and _admit is a lock."""
        from arc_runner.server.worker import SerialWorker

        worker = SerialWorker()
        assert isinstance(worker._job_queue, queue.SimpleQueue)
        assert hasattr(worker._admit, "acquire") and hasattr(worker._admit, "release")
        assert worker._wedged is False

    def test_stop_is_synchronous_not_a_coroutine(self):
        """stop() is a plain function, not async."""
        from arc_runner.server.worker import SerialWorker

        assert not asyncio.iscoroutinefunction(SerialWorker.stop)


class TestSerialWorkerExecution:
    """CC-1: jobs run on the worker thread, one at a time, in FIFO order."""

    async def test_submit_returns_result_of_run(self):
        """submit resolves with the value returned by run(ctx)."""
        from arc_runner.server.worker import SerialWorker

        worker = SerialWorker()
        worker.start()
        try:
            assert await worker.submit(lambda ctx: 42) == 42
        finally:
            await _stop(worker)

    async def test_submit_runs_on_worker_thread_with_job_context(self):
        """run executes off the caller thread and receives a JobContext."""
        from arc_runner.server.worker import JobContext, SerialWorker

        seen = {}

        def run(ctx):
            seen["thread"] = threading.current_thread().name
            seen["ctx"] = ctx
            return None

        worker = SerialWorker()
        worker.start()
        try:
            await worker.submit(run)
        finally:
            await _stop(worker)
        assert seen["thread"] == "arc-inference"
        assert isinstance(seen["ctx"], JobContext)

    async def test_submit_many_jobs_runs_fifo_one_at_a_time(self):
        """Concurrent submissions execute in submission order, never overlapping."""
        from arc_runner.server.worker import SerialWorker

        order = []
        active = []
        peak = []
        guard = threading.Lock()

        def make(index):
            def run(ctx):
                with guard:
                    active.append(index)
                    peak.append(len(active))
                order.append(index)
                with guard:
                    active.remove(index)
                return index

            return run

        worker = SerialWorker()
        worker.start()
        try:
            tasks = [asyncio.create_task(worker.submit(make(i))) for i in range(10)]
            results = await asyncio.gather(*tasks)
        finally:
            await _stop(worker)
        assert order == list(range(10))
        assert results == list(range(10))
        assert max(peak) == 1

    async def test_submit_exception_in_run_propagates_to_caller(self):
        """An exception raised by run is re-raised by submit and the worker survives."""
        from arc_runner.server.worker import SerialWorker

        def bad(ctx):
            raise ValueError("intentional failure")

        worker = SerialWorker()
        worker.start()
        try:
            with pytest.raises(ValueError, match="intentional failure"):
                await worker.submit(bad)
            assert await worker.submit(lambda ctx: "still alive") == "still alive"
        finally:
            await _stop(worker)

    async def test_submit_load_failure_surfaces_as_load_phase_error(self):
        """A failure inside ctx.load reaches the caller as LoadPhaseError."""
        from arc_runner.server.worker import LoadPhaseError, SerialWorker

        original = FileNotFoundError("no model")

        def run(ctx):
            def load():
                raise original

            ctx.load(load)

        worker = SerialWorker()
        worker.start()
        try:
            with pytest.raises(LoadPhaseError) as info:
                await worker.submit(run)
        finally:
            await _stop(worker)
        assert info.value.__cause__ is original

    async def test_submit_engine_call_runs_through_fake_engine(self):
        """A route-shaped job loads, generates and reads telemetry on the worker."""
        from arc_runner.server.worker import SerialWorker

        def run(ctx):
            engine = ctx.load(lambda: FakeEngine())
            return engine.generate("p", None), engine.get_telemetry()

        worker = SerialWorker()
        worker.start()
        try:
            assert await worker.submit(run) == ("result", {})
        finally:
            await _stop(worker)


class TestSerialWorkerBusyAndHealthState:
    """CC-3: busy_for_s reports the running job's age, None when idle."""

    async def test_busy_for_s_is_none_when_idle(self):
        """An idle worker reports None."""
        from arc_runner.server.worker import SerialWorker

        worker = SerialWorker()
        worker.start()
        try:
            assert worker.busy_for_s is None
        finally:
            await _stop(worker)

    async def test_busy_for_s_is_float_while_job_runs_then_none(self):
        """During a job busy_for_s is a non-negative float; afterwards None again."""
        from arc_runner.server.worker import SerialWorker

        started = threading.Event()
        release = threading.Event()

        def run(ctx):
            started.set()
            release.wait(WAIT_S)

        worker = SerialWorker()
        worker.start()
        try:
            task = asyncio.create_task(worker.submit(run))
            assert await asyncio.to_thread(started.wait, WAIT_S)
            value = worker.busy_for_s
            assert isinstance(value, float) and value >= 0.0
            release.set()
            await task
            assert worker.busy_for_s is None
        finally:
            release.set()
            await _stop(worker)


class TestSerialWorkerStop:
    """CC-1 and CC-3: bounded sync stop, wedge detection, rejection after stop."""

    async def test_stop_clean_returns_true_and_joins_thread(self):
        """An idle worker stops cleanly within the timeout."""
        from arc_runner.server.worker import SerialWorker

        worker = SerialWorker()
        worker.start()
        assert await _stop(worker) is True
        assert not worker._thread.is_alive()
        assert worker._wedged is False

    async def test_submit_after_stop_raises_server_stopping(self):
        """submit after a clean stop raises EngineUnavailableError(SERVER_STOPPING)."""
        from arc_runner.server.worker import SerialWorker

        worker = SerialWorker()
        worker.start()
        await _stop(worker)
        with pytest.raises(EngineUnavailableError) as info:
            await worker.submit(lambda ctx: None)
        assert info.value.error_code == "SERVER_STOPPING"

    async def test_stop_with_stuck_job_returns_false_and_marks_wedged(self):
        """A job that outlives the join timeout wedges the worker; later submits are refused."""
        from arc_runner.server.worker import SerialWorker

        started = threading.Event()
        release = threading.Event()

        def stuck(ctx):
            started.set()
            release.wait(10.0)

        worker = SerialWorker()
        worker.start()
        task = asyncio.create_task(worker.submit(stuck))
        try:
            assert await asyncio.to_thread(started.wait, WAIT_S)
            assert await _stop(worker, 0.1) is False
            assert worker._wedged is True
            with pytest.raises(EngineUnavailableError):
                await worker.submit(lambda ctx: None)
        finally:
            release.set()
            await task
            worker._thread.join(WAIT_S)

    async def test_job_queued_before_stop_is_rejected_not_run(self):
        """A job still queued when stop() begins fails with SERVER_STOPPING and never runs."""
        from arc_runner.server.worker import SerialWorker

        started = threading.Event()
        release = threading.Event()
        ran = []

        def blocker(ctx):
            started.set()
            release.wait(WAIT_S)

        def queued(ctx):
            ran.append(True)

        worker = SerialWorker()
        worker.start()
        first = asyncio.create_task(worker.submit(blocker))
        assert await asyncio.to_thread(started.wait, WAIT_S)
        second = asyncio.create_task(worker.submit(queued))
        await asyncio.sleep(0)
        stopper = asyncio.create_task(_stop(worker, WAIT_S))
        while not worker._stopping:
            await asyncio.sleep(0)
        release.set()
        await first
        assert await stopper is True
        with pytest.raises(EngineUnavailableError) as info:
            await second
        assert info.value.error_code == "SERVER_STOPPING"
        assert ran == []

    @given(payload=st.integers())
    @settings(max_examples=20, deadline=None)
    def test_submit_property_rejected_after_stop_for_any_callable(self, payload):
        """P3: for any callable, submit after stop raises EngineUnavailableError."""
        from arc_runner.server.worker import SerialWorker

        async def scenario():
            worker = SerialWorker()
            worker.start()
            assert await _stop(worker) is True
            with pytest.raises(EngineUnavailableError):
                await worker.submit(lambda ctx: payload)

        asyncio.run(scenario())


class TestAsyncTokenStream:
    """CC-2: ordered channel of (chunk|done|error) events from worker thread to async consumer."""

    async def test_stream_yields_chunks_then_done_in_order(self):
        """Pushed chunks arrive in order, followed by done, then iteration ends."""
        from arc_runner.server.worker import AsyncTokenStream

        loop = asyncio.get_running_loop()
        stream = AsyncTokenStream()
        stream.push_chunk("a", loop)
        stream.push_chunk("b", loop)
        stream.push_done({"n": 2}, loop)
        events = [event async for event in stream]
        assert events == [("chunk", "a"), ("chunk", "b"), ("done", {"n": 2})]

    async def test_stream_error_event_ends_iteration(self):
        """An error event is delivered and iteration then stops."""
        from arc_runner.server.worker import AsyncTokenStream

        loop = asyncio.get_running_loop()
        stream = AsyncTokenStream()
        exc = RuntimeError("load failed")
        stream.push_error(exc, loop)
        events = [event async for event in stream]
        assert events == [("error", exc)]

    def test_stream_cancel_sets_is_cancelled(self):
        """cancel() flips is_cancelled which the producer checks between chunks."""
        from arc_runner.server.worker import AsyncTokenStream

        stream = AsyncTokenStream()
        assert stream.is_cancelled is False
        stream.cancel()
        assert stream.is_cancelled is True

    async def test_stream_through_worker_delivers_every_token(self):
        """A route-shaped streaming job loses no tokens across the thread boundary."""
        from arc_runner.server.worker import AsyncTokenStream, SerialWorker

        tokens = [f"tok{i}" for i in range(300)]
        loop = asyncio.get_running_loop()
        stream = AsyncTokenStream()

        def run(ctx):
            engine = ctx.load(lambda: FakeEngine(tokens))
            for text in engine.generate_stream("p", None):
                if stream.is_cancelled:
                    break
                stream.push_chunk(text, loop)
            stream.push_done(engine.get_telemetry(), loop)

        worker = SerialWorker()
        worker.start()
        try:
            await worker.submit(run)
            events = [event async for event in stream]
        finally:
            await _stop(worker)
        assert [p for kind, p in events if kind == "chunk"] == tokens
        assert events[-1] == ("done", {})

    @given(texts=st.lists(st.text(max_size=8), max_size=40))
    @settings(max_examples=50, deadline=None)
    def test_stream_property_preserves_push_order(self, texts):
        """P1: for any pushed sequence the consumer receives the same order."""
        from arc_runner.server.worker import AsyncTokenStream

        async def scenario():
            loop = asyncio.get_running_loop()
            stream = AsyncTokenStream()
            for text in texts:
                stream.push_chunk(text, loop)
            stream.push_done(None, loop)
            return [event async for event in stream]

        events = asyncio.run(scenario())
        assert [p for kind, p in events if kind == "chunk"] == texts
        assert events[-1] == ("done", None)


class TestHealthEndpoint:
    """CC-3: /health schema, open path, and 503 only for wedged."""

    def test_health_open_without_api_key(self):
        """GET /health succeeds without credentials when an API key is configured."""
        from arc_runner.server.app import create_app

        with patch.dict(os.environ, {"ARC_API_KEY": "secret-key-for-test"}):
            client = TestClient(create_app())
            assert client.get("/health").status_code == 200

    def test_health_without_worker_reports_ok_idle(self):
        """No worker attached reports the full schema as ok and idle."""
        from arc_runner.server.app import create_app

        resp = TestClient(create_app()).get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "busy": False, "busy_for_s": None}

    def test_health_busy_worker_reports_ok_with_busy_for_s(self):
        """A busy worker is still HTTP 200 with busy true and its age."""
        from arc_runner.server.app import create_app

        app = create_app()
        app.state.worker = SimpleNamespace(_wedged=False, busy_for_s=1.5)
        resp = TestClient(app).get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "busy": True, "busy_for_s": 1.5}

    def test_health_wedged_worker_returns_503(self):
        """Only a wedged worker yields 503 with status wedged."""
        from arc_runner.server.app import create_app

        app = create_app()
        app.state.worker = SimpleNamespace(_wedged=True, busy_for_s=None)
        resp = TestClient(app).get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"status": "wedged", "busy": False, "busy_for_s": None}

    def test_health_head_method_is_allowed(self):
        """HEAD /health is accepted for probes."""
        from arc_runner.server.app import create_app

        assert TestClient(create_app()).head("/health").status_code == 200


class TestLifespan:
    """T3: the app lifespan owns the SerialWorker lifecycle."""

    def test_lifespan_starts_worker_and_stops_it_on_shutdown(self):
        """app.state.worker is a live SerialWorker inside the context and joined after."""
        from arc_runner.server.app import create_app
        from arc_runner.server.worker import SerialWorker

        app = create_app()
        with TestClient(app):
            worker = app.state.worker
            assert isinstance(worker, SerialWorker)
            assert worker._thread.is_alive()
        assert not worker._thread.is_alive()
        assert worker._wedged is False


class TestEngineUnavailableMapsTo503:
    """EngineUnavailableError raised in a route maps to HTTP 503."""

    def test_engine_unavailable_in_route_returns_503_with_code(self):
        """The app exception handler returns 503 and the error code."""
        from arc_runner.server.app import create_app

        app = create_app()

        @app.get("/test-wedge")
        async def wedge_route():
            raise EngineUnavailableError("test wedge")

        resp = TestClient(app, raise_server_exceptions=False).get("/test-wedge")
        assert resp.status_code == 503
        assert resp.json()["code"] == "ENGINE_WEDGED"


class TestCliGracefulShutdown:
    """T4: uvicorn receives the graceful shutdown timeout."""

    def test_run_server_passes_timeout_graceful_shutdown(self, monkeypatch):
        """uvicorn.Config is built with timeout_graceful_shutdown from graceful_shutdown_s()."""
        import importlib

        import uvicorn

        cli_main = importlib.import_module("arc_runner.cli.main")

        captured = {}

        class StopHere(Exception):
            """Raised to stop run_server after Config is built."""

        def fake_config(*args, **kwargs):
            captured.update(kwargs)
            raise StopHere

        monkeypatch.setenv("ARC_GRACEFUL_SHUTDOWN_S", "6")
        monkeypatch.setattr(uvicorn, "Config", fake_config)
        with pytest.raises(StopHere):
            cli_main.run_server(host="127.0.0.1", port=0)
        assert captured["timeout_graceful_shutdown"] == 6
