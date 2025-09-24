from functools import cache

from ultron.config.config import get_ultron_serving_config

SEMANTIC_CACHE_WORKER = None


@cache
def initialize_semantic_cache_worker(version="v01"):
    global SEMANTIC_CACHE_WORKER
    if SEMANTIC_CACHE_WORKER is None:
        from ultron.logic import SemanticCacheWorker

        config = get_ultron_serving_config()
        SEMANTIC_CACHE_WORKER = SemanticCacheWorker(
            version=version, semantic_cache_model_name=config.openai_cache.semantic_cache_model
        )


def get_semantic_cache_worker():
    return SEMANTIC_CACHE_WORKER
