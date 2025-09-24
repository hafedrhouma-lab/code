import asyncio
import concurrent.futures
import inspect
import threading
from collections.abc import Coroutine
from functools import wraps
from typing import TypeVar, Callable, Union, ParamSpec

import ace.newrelic

T = TypeVar("T")
CoroutineOrFunc = Union[Coroutine[None, None, T], Callable[[], Coroutine[None, None, T]], Callable[[], T]]

D_Return = TypeVar("D_Return")
D_Spec = ParamSpec("D_Spec")


def to_thread(func: Callable[D_Spec, D_Return]) -> Callable[D_Spec, Coroutine[None, None, D_Return]]:
    @wraps(func)
    async def wrapper(*args: D_Spec.args, **kwargs: D_Spec.kwargs) -> D_Return:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def to_coro(coro_or_func: CoroutineOrFunc) -> Coroutine[None, None, T]:
    if inspect.iscoroutine(coro_or_func):
        return coro_or_func
    if inspect.iscoroutinefunction(coro_or_func):
        return coro_or_func()
    if inspect.isfunction(coro_or_func):

        async def to_async(func):
            func()

        return to_async(coro_or_func)


class AsyncThread:
    def __init__(self, coro: Coroutine):
        self.loop: asyncio.AbstractEventLoop | None = None

        async def run():
            ace.newrelic.init_thread()
            self.loop = asyncio.get_running_loop()
            await coro

        self.thread = threading.Thread(target=asyncio.run, args=(run(),))
        self.thread.start()

    @property
    def completed(self) -> bool:
        return not self.loop or self.loop.is_closed()

    def just_run(self, coro_or_func: CoroutineOrFunc) -> concurrent.futures.Future:
        """
        Schedules a coroutine on the executor's thread, that's all
        """
        return asyncio.run_coroutine_threadsafe(to_coro(coro_or_func), self.loop)

    # Inspired from https://gist.github.com/lars-tiede/956206c8d97cbc3454cb
    async def run(self, coro_or_func: CoroutineOrFunc) -> T:
        """
        Schedules a coroutine on the executor's thread, awaits until it has run and returns the result
        """
        client_loop = asyncio.get_event_loop()

        # Schedule the coroutine safely on the executor's loop, get a concurrent.future back
        fut = asyncio.run_coroutine_threadsafe(to_coro(coro_or_func), self.loop)

        # Set up a threading.Event that fires when the future is finished
        finished = threading.Event()
        fut.add_done_callback(lambda _: finished.set())

        # Wait on that event on the default executor (ThreadPoolExecutor), yielding control to our loop
        await client_loop.run_in_executor(None, finished.wait)

        # The coroutine result is now available
        return fut.result()

    def wait(self, timeout: float = None):
        self.thread.join(timeout)
