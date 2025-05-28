import asyncio
from tqdm import tqdm
from typing import Callable, Awaitable, Any, Iterable

def ensure_async(func: Callable) -> Callable:
    if asyncio.iscoroutinefunction(func):
        return func
    async def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

async def async_run_concurrent(
    items: Iterable,
    worker: Callable[[Any], Awaitable[Any]],
    max_concurrency: int = 10,
    on_result: Callable[[Any], None] = None,
    desc: str = "Processing"
):
    sem = asyncio.Semaphore(max_concurrency)
    worker = ensure_async(worker)
    async def sem_task(item):
        async with sem:
            return await worker(item)
    tasks = [asyncio.create_task(sem_task(item)) for item in items]
    results = []
    with tqdm(total=len(tasks), desc=desc) as pbar:
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if on_result:
                    on_result(result)
                results.append(result)
                pbar.update(1)
        except KeyboardInterrupt:
            print("⏹️ Interrupted—progress up to here has been saved.")
            return results
    return results
