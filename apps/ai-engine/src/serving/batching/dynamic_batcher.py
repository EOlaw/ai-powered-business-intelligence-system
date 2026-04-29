"""
InsightSerenity AI Engine — Dynamic Request Batcher
====================================================
Dynamic batching groups concurrent inference requests into a single
forward pass, amortising the fixed cost (memory transfers, CUDA kernel
launches) across multiple requests for better GPU utilisation.

Without batching:
    Request 1 arrives → forward pass (10ms) → response
    Request 2 arrives → forward pass (10ms) → response
    Request 3 arrives → forward pass (10ms) → response
    Total: 30ms, GPU utilisation ~30%

With dynamic batching (batch_size=3):
    Requests 1,2,3 arrive within 5ms window
    → One forward pass (12ms) → 3 responses
    Total: 12ms per request, GPU utilisation ~90%

The batcher waits for a configurable window (max_wait_ms) before
dispatching accumulated requests as a batch. Requests that arrive
after the window starts wait for the next batch.

For streaming generation (token-by-token), dynamic batching is applied
at the decode step level — multiple sequences are generated in parallel
with the same transformer forward pass, using padding masks.

This module provides the infrastructure; the actual batching policy
is controlled by configuration at the API layer.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")   # Request type
R = TypeVar("R")   # Result type


@dataclass
class BatchItem(Generic[T]):
    """One item in the batch queue with its associated future."""
    request:   T
    future:    asyncio.Future
    queued_at: float = field(default_factory=time.perf_counter)


class DynamicBatcher(Generic[T, R]):
    """
    Collects individual requests and dispatches them as a batch.

    The batch is dispatched when either:
        a) max_batch_size requests are queued, OR
        b) max_wait_ms milliseconds have passed since the first request

    This is the "continuous batching" pattern used by vLLM and TGI.

    Args:
        process_batch:  Async callable that takes List[T] → List[R].
                        This is the actual model forward pass.
        max_batch_size: Maximum requests per batch. Default 16.
        max_wait_ms:    Maximum milliseconds to wait for a full batch. Default 20.
    """

    def __init__(
        self,
        process_batch: Callable[[List[T]], List[R]],
        max_batch_size: int   = 16,
        max_wait_ms:    float = 20.0,
    ) -> None:
        self.process_batch  = process_batch
        self.max_batch_size = max_batch_size
        self.max_wait_ms    = max_wait_ms / 1000.0   # Convert to seconds

        self._queue:    List[BatchItem] = []
        self._lock:     asyncio.Lock    = asyncio.Lock()
        self._dispatch_task: Optional[asyncio.Task] = None
        self._stats = {"total_batches": 0, "total_requests": 0}

    async def submit(self, request: T) -> R:
        """
        Submit a single request and wait for its result.

        Internally, the request is queued and processed as part of a batch.
        The caller awaits a Future that is resolved when the batch completes.

        Args:
            request: The inference request.

        Returns:
            The result corresponding to this request.
        """
        loop   = asyncio.get_event_loop()
        future = loop.create_future()
        item   = BatchItem(request=request, future=future)

        async with self._lock:
            self._queue.append(item)
            # If batch is full, dispatch immediately
            if len(self._queue) >= self.max_batch_size:
                await self._dispatch_batch()
            elif self._dispatch_task is None or self._dispatch_task.done():
                # Start a delayed dispatch task
                self._dispatch_task = asyncio.create_task(self._delayed_dispatch())

        return await future

    async def _delayed_dispatch(self) -> None:
        """Wait max_wait_ms, then dispatch whatever is in the queue."""
        await asyncio.sleep(self.max_wait_ms)
        async with self._lock:
            if self._queue:
                await self._dispatch_batch()

    async def _dispatch_batch(self) -> None:
        """
        Process all queued requests as one batch and resolve their futures.
        Must be called while holding self._lock.
        """
        if not self._queue:
            return

        batch = self._queue[:self.max_batch_size]
        self._queue = self._queue[self.max_batch_size:]

        requests = [item.request for item in batch]

        start = time.perf_counter()
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.process_batch, requests
            )
        except Exception as e:
            # Fail all futures in this batch
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
            logger.error("Batch dispatch failed", error=str(e), batch_size=len(batch))
            return

        elapsed = (time.perf_counter() - start) * 1000
        self._stats["total_batches"]  += 1
        self._stats["total_requests"] += len(batch)

        logger.debug(
            "Batch dispatched",
            size=len(batch),
            elapsed_ms=round(elapsed, 1),
        )

        # Resolve each future with its corresponding result
        for item, result in zip(batch, results):
            if not item.future.done():
                item.future.set_result(result)

    def stats(self) -> Dict[str, Any]:
        """Return batching statistics."""
        total_req   = self._stats["total_requests"]
        total_batch = self._stats["total_batches"]
        avg_size    = round(total_req / max(total_batch, 1), 2)
        return {
            **self._stats,
            "avg_batch_size":   avg_size,
            "queue_depth":      len(self._queue),
        }
