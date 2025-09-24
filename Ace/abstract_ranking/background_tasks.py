import asyncio
from random import randint


async def random_wait(max_wait_sec: int):
    wait_sec = randint(0, max_wait_sec)  # [0, max_wait_sec]
    await asyncio.sleep(wait_sec)
