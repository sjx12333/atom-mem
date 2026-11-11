"""Run coroutines on a single persistent event loop.

graphiti_core's ``Neo4jDriver`` is an *async* driver that must be driven from
one event loop. The service's previous helper (``_run_async``) spawned a fresh
``asyncio.run`` loop per call, in different threads, so the one shared driver
was used from many different loops. That corrupts the driver's connection pool
and surfaces as ``'NoneType' object has no attribute 'send'`` on every query
after the first, which is exactly the failure the WebUI saw (search silently
degraded and every closed-session extraction landed in ``retry_locked``).

This runner pins every coroutine to a single daemon loop regardless of the
calling thread, so the async graph driver is always used from one loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class AsyncLoopRunner:
    """Submit coroutines to a single, always-running background event loop.

    ``run()`` is synchronous: it blocks the calling thread until the coroutine
    finishes and re-raises the coroutine's exception (if any). It is safe to
    call from any thread *except* the loop's own thread; calling from the loop
    thread would deadlock, so that case raises an explicit error.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._spin, name="graphiti-async-loop", daemon=True
        )
        self._thread.start()

    def _spin(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro: Any) -> Any:
        """Run ``coro`` on the dedicated loop and return its result.

        Blocks until the coroutine completes. Exceptions raised inside the
        coroutine are re-raised in the calling thread.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("AsyncLoopRunner.run must not be called from the loop thread")
        self._ready.wait()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        """Stop the loop thread (used at process exit; the thread is daemon)."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        try:
            self._thread.join(timeout=2)
        except RuntimeError:
            pass
