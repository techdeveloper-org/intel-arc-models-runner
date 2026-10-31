"""Serial inference worker: one daemon thread, one FIFO queue, one job at a time.

Implements the WP-03 contract (CC-1..CC-4):
- CC-1: one daemon thread drains a threading.Queue; jobs run serially
- CC-2: routes submit callables and await the result; no route holds an engine object
- CC-3: callers use submit() which bridges the async/sync boundary via asyncio.Future
- CC-4: worker exposes is_busy / is_stopping flags consumed by the /health route
"""

import asyncio
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from arc_runner.core.exceptions import EngineUnavailableError
from arc_runner.core.logging import get_logger

logger = get_logger("server.worker")

T = TypeVar("T")

_SENTINEL = object()


class SerialWorker:
    """Single-threaded inference worker that serialises all engine calls.

    A daemon thread drains a ``threading.Queue``. Each job is a zero-argument callable;
    the worker runs it synchronously and resolves an ``asyncio.Future`` with the return
    value (or exception). Callers ``await submit(fn)`` from async code and never touch
    the engine directly.

    Attributes:
        _thread: The daemon background thread.
        _job_queue: Thread-safe FIFO queue shared between callers and the worker thread.
        _stopping: Set when ``stop()`` has been called; blocks new submissions.
        _busy: Set while a job is executing on the worker thread.
        _loop: The event loop captured at ``start()`` time for Future callbacks.
    """

    def __init__(self) -> None:
        """Initialise the worker in a stopped state; call start() to activate it."""
        self._job_queue: queue.Queue[Any] = queue.Queue()
        self._stopping = False
        self._busy = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread = threading.Thread(
            target=self._run, daemon=True, name="arc-serial-worker"
        )

    @property
    def is_busy(self) -> bool:
        """True while a job is executing on the worker thread."""
        return self._busy

    @property
    def is_stopping(self) -> bool:
        """True once stop() has been called."""
        return self._stopping

    def start(self) -> None:
        """Start the daemon worker thread.

        Captures the running event loop so the thread can resolve Futures on it via
        call_soon_threadsafe. Must be called from an async context (e.g., the lifespan).
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._thread.start()
        logger.info("serial worker started", extra={"thread": self._thread.name})

    async def stop(self, timeout: float | None = None) -> None:
        """Signal the worker to stop and wait for the thread to exit.

        Args:
            timeout: Seconds to wait for a clean join. None uses TIMEOUT_GRACEFUL_SHUTDOWN_S.

        Raises:
            EngineUnavailableError: With ENGINE_WEDGED if the thread does not exit in time.
        """
        from arc_runner.core.constants import TIMEOUT_GRACEFUL_SHUTDOWN_S

        deadline = timeout if timeout is not None else TIMEOUT_GRACEFUL_SHUTDOWN_S
        self._stopping = True
        self._job_queue.put(_SENTINEL)
        await asyncio.to_thread(self._thread.join, deadline)
        if self._thread.is_alive():
            logger.error("worker thread did not exit within timeout", extra={"timeout_s": deadline})
            raise EngineUnavailableError(
                f"Inference worker did not stop within {deadline}s", error_code="ENGINE_WEDGED"
            )
        logger.info("serial worker stopped", extra={"thread": self._thread.name})

    async def submit(self, fn: Callable[[], T]) -> T:
        """Submit a blocking callable and await its result on the worker thread.

        The callable runs synchronously on the daemon thread. The result (or exception)
        is forwarded to the caller's event loop via call_soon_threadsafe on a Future.

        Args:
            fn: Zero-argument callable to run on the worker thread.

        Returns:
            The return value of ``fn``.

        Raises:
            EngineUnavailableError: With SERVER_STOPPING if the worker has been stopped.
            Exception: Any exception raised by ``fn`` is re-raised here.
        """
        if self._stopping:
            raise EngineUnavailableError(
                "Inference worker is shutting down", error_code="SERVER_STOPPING"
            )
        loop = self._loop or asyncio.get_event_loop()
        future: asyncio.Future[T] = loop.create_future()
        self._job_queue.put((fn, future))
        return await future

    async def submit_stream(self, token_iter: Iterator[str]) -> "AsyncIterator[str]":
        """Submit a synchronous token iterator and return an async generator.

        Tokens are forwarded one at a time from the worker thread to the caller's event
        loop via call_soon_threadsafe. A sentinel ends the stream cleanly with no loss.

        Args:
            token_iter: Synchronous iterator that yields string tokens.

        Returns:
            Async generator that yields each token in order.

        Raises:
            EngineUnavailableError: With SERVER_STOPPING if the worker has been stopped.
        """
        if self._stopping:
            raise EngineUnavailableError(
                "Inference worker is shutting down", error_code="SERVER_STOPPING"
            )
        loop = self._loop or asyncio.get_event_loop()
        token_queue: asyncio.Queue[Any] = asyncio.Queue()

        def produce() -> None:
            try:
                for token in token_iter:
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, _SENTINEL)

        future: asyncio.Future[None] = loop.create_future()
        self._job_queue.put((produce, future))

        async def consume() -> "AsyncIterator[str]":
            while True:
                item = await token_queue.get()
                if item is _SENTINEL:
                    break
                yield item

        return consume()

    def _run(self) -> None:
        """Worker thread main loop: drain the job queue and execute callables serially."""
        while True:
            job = self._job_queue.get()
            if job is _SENTINEL:
                break
            fn, future = job
            self._busy = True
            try:
                result = fn()
                self._loop.call_soon_threadsafe(future.set_result, result)
            except Exception as exc:
                self._loop.call_soon_threadsafe(future.set_exception, exc)
            finally:
                self._busy = False
