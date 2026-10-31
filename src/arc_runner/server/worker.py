"""Serial inference worker: one daemon thread runs every engine call, one job at a time.

Contracts (WP-03):
    CC-1: a single daemon thread "arc-inference" drains a queue.SimpleQueue in FIFO order.
        An admission lock makes the flag check plus enqueue (submit) and the flag set plus
        sentinel enqueue (stop) atomic. The lock is never held across a job, a join, I/O or
        a callback.
    CC-2: AsyncTokenStream carries ordered ("chunk", text) | ("done", telemetry) |
        ("error", exc) events from the worker thread to an async consumer.
    CC-3: stop() is synchronous, bounded and returns a bool; busy_for_s feeds /health.
    CC-4: JobContext.load wraps any failure of the engine-load step as LoadPhaseError so
        routes can tell load failures from failures after the load.

Engine, hub, cli and core packages must not import this module.
"""

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from arc_runner.core.exceptions import EngineUnavailableError
from arc_runner.core.logging import get_logger

logger = get_logger("server.worker")

T = TypeVar("T")

_SENTINEL = object()
_TERMINAL_EVENTS = frozenset({"done", "error"})


class LoadPhaseError(Exception):
    """Marks a failure raised while loading or compiling an engine.

    The original exception is available as ``__cause__``; ``str()`` is never empty.
    """

    def __init__(self, cause: Exception) -> None:
        """Wrap the load failure.

        Args:
            cause: The exception raised by the load step.
        """
        super().__init__(str(cause) or type(cause).__name__)
        self.__cause__ = cause


class JobContext:
    """Handed to every job's run(ctx) callable on the worker thread."""

    def load(self, fn: Callable[[], T]) -> T:
        """Run an engine-load step and tag any failure as a load-phase failure.

        Args:
            fn: Zero-argument callable that loads or fetches the engine.

        Returns:
            T: The value returned by ``fn``.

        Raises:
            LoadPhaseError: Wrapping any Exception raised by ``fn``.
        """
        try:
            return fn()
        except Exception as exc:
            raise LoadPhaseError(exc) from exc


class AsyncTokenStream:
    """Ordered channel from the worker thread to an async consumer (CC-2).

    The producer calls push_chunk, then exactly one of push_done or push_error, from the
    worker thread. The consumer iterates with ``async for event_type, payload in stream``;
    iteration ends after the done or error event. The consumer calls cancel() when it stops
    early so the producer can stop between chunks.
    """

    def __init__(self) -> None:
        """Create an empty stream."""
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._cancel_event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        """True once the consumer has called cancel()."""
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Ask the producer to stop; safe to call from any thread, any number of times."""
        self._cancel_event.set()

    def push_chunk(self, text: str, loop: asyncio.AbstractEventLoop) -> None:
        """Queue a text chunk for the consumer.

        Args:
            text: Decoded text of the chunk.
            loop: Event loop of the consumer.
        """
        self._push(("chunk", text), loop)

    def push_done(self, telemetry: Any, loop: asyncio.AbstractEventLoop) -> None:
        """Queue the terminal success event.

        Args:
            telemetry: Engine telemetry of the finished request.
            loop: Event loop of the consumer.
        """
        self._push(("done", telemetry), loop)

    def push_error(self, exc: BaseException, loop: asyncio.AbstractEventLoop) -> None:
        """Queue the terminal failure event.

        Args:
            exc: The exception that ended the job.
            loop: Event loop of the consumer.
        """
        self._push(("error", exc), loop)

    def _push(self, item: tuple[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """Hand an event to the consumer's loop; drop it with a warning if that loop is closed."""
        try:
            loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            logger.warning("stream event dropped, event loop closed", extra={"event_type": item[0]})

    async def __aiter__(self) -> AsyncIterator[tuple[str, Any]]:
        """Yield events in push order and stop after the done or error event."""
        while True:
            item = await self._queue.get()
            yield item
            if item[0] in _TERMINAL_EVENTS:
                return


def _make_fail(future: "asyncio.Future[Any]", loop: asyncio.AbstractEventLoop) -> Callable[[BaseException], None]:
    """Build the callback that fails a job's future on its loop if it is still pending.

    Args:
        future: Future the caller of submit() awaits.
        loop: Event loop that owns the future.

    Returns:
        Callable[[BaseException], None]: Callback to run through call_soon_threadsafe.
    """

    def fail(exc: BaseException) -> None:
        """Set the exception unless the future already finished or was cancelled."""
        if not future.done():
            future.set_exception(exc)

    return fail


class SerialWorker:
    """Runs every engine call on one daemon thread so the event loop never blocks (CC-1, CC-3).

    Attributes:
        _job_queue: FIFO of (run, fail, future) tuples and the stop sentinel.
        _admit: Admission lock for submit() and stop().
        _stopping: Set by stop(); new and still-queued jobs are refused.
        _wedged: Set when stop() could not join the thread in time.
        _busy: True while a job runs.
        _current: (run, start time) of the running job, else None.
        _loop: Event loop captured by start().
        _thread: The daemon worker thread.
    """

    def __init__(self) -> None:
        """Create a stopped worker; call start() from async code to activate it."""
        self._job_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._admit = threading.Lock()
        self._stopping = False
        self._wedged = False
        self._busy = False
        self._current: tuple[Callable[..., Any], float] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="arc-inference")

    @property
    def busy_for_s(self) -> float | None:
        """Seconds since the running job started, or None when idle."""
        current = self._current
        if current is None:
            return None
        return time.monotonic() - current[1]

    def start(self) -> None:
        """Capture the running event loop and start the worker thread.

        Must be called from async code (the app lifespan) so results can be delivered
        to the right loop.
        """
        self._loop = asyncio.get_running_loop()
        self._thread.start()
        logger.info("serial worker started", extra={"thread": self._thread.name})

    def stop(self, join_timeout_s: float | None = None) -> bool:
        """Refuse new work, let the queue drain, and join the thread within a deadline.

        Blocks, so async callers must run it off the event loop. Jobs still queued when
        the sentinel is reached fail with SERVER_STOPPING instead of running.

        Args:
            join_timeout_s: Seconds to wait for the thread; None uses SHUTDOWN_JOIN_S.

        Returns:
            bool: True when the thread exited, False when it is wedged.
        """
        from arc_runner.core.constants import SHUTDOWN_JOIN_S

        deadline = join_timeout_s if join_timeout_s is not None else SHUTDOWN_JOIN_S
        with self._admit:
            self._stopping = True
            self._job_queue.put(_SENTINEL)
        self._thread.join(deadline)
        alive = self._thread.is_alive()
        if alive:
            self._wedged = True
            logger.error("serial worker wedged", extra={"timeout_s": deadline})
        else:
            logger.info("serial worker stopped", extra={"thread": self._thread.name})
        return not alive

    async def submit(self, run: Callable[[JobContext], T]) -> T:
        """Queue a job and await its result without blocking the event loop.

        Args:
            run: Callable taking a JobContext; runs on the worker thread.

        Returns:
            T: The value returned by ``run``.

        Raises:
            EngineUnavailableError: SERVER_STOPPING when the worker is stopping or wedged.
            LoadPhaseError: When ``run`` used ctx.load and the load step failed.
            Exception: Any other exception raised by ``run``.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        with self._admit:
            if self._stopping or self._wedged:
                raise EngineUnavailableError("Inference worker is shutting down", error_code="SERVER_STOPPING")
            self._job_queue.put((run, _make_fail(future, loop), future))
        return await future

    def _post(self, fn: Callable[..., None], *args: Any) -> None:
        """Run a callback on the event loop; log and drop it if the loop is already closed."""
        try:
            self._loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            logger.warning("job result dropped, event loop closed")

    def _run(self) -> None:
        """Worker thread body: take jobs in FIFO order and run them one at a time."""
        while True:
            job = self._job_queue.get()
            if job is _SENTINEL:
                break
            run, fail, future = job
            self._current = (run, time.monotonic())
            refuse = self._stopping or self._wedged
            self._busy = True
            error: Exception | None = None
            result: Any = None
            try:
                if refuse:
                    error = EngineUnavailableError("Inference worker is shutting down", error_code="SERVER_STOPPING")
                else:
                    result = run(JobContext())
            except Exception as exc:
                error = exc
            finally:
                self._busy = False
                self._current = None
            if error is not None:
                self._post(fail, error)
            else:
                self._post(self._resolve, future, result)

    @staticmethod
    def _resolve(future: "asyncio.Future[Any]", value: Any) -> None:
        """Resolve the future with the job result unless it already finished."""
        if not future.done():
            future.set_result(value)
