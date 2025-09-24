#!/usr/bin/env python3
import logging

import openai
import sse_starlette

import ace
from ace.configs.config import AppPostgresConfig
from ace.storage import db
from ace.storage.db import connections
from ultron import SERVICE_NAME
from ultron.api.router import build_main_router
from ultron.api.v1.semantic_search.dependencies import initialize_semantic_cache_worker
from ultron.config.config import get_ultron_serving_config, UltronServingConfig
from ultron.runners import cross_encoder, text_embeddings


def build_app() -> ace.AceService:
    app = ace.AceService(SERVICE_NAME, runners=[text_embeddings.get_runner(), cross_encoder.get_runner()])

    config: "UltronServingConfig" = get_ultron_serving_config()
    postgres_config: "AppPostgresConfig" = config.storage.postgres

    # Skip debug logs from sse_starlette
    logging.getLogger(sse_starlette.sse.__name__).setLevel(logging.INFO)

    @app.on_api_startup()
    async def api_init():  # DB connection pool (API thread)
        openai.api_key = config.external_apis.openai_api_key
        await db.init_connection_pool(
            service_name=SERVICE_NAME, query_timeout=postgres_config.main_query_timeout, config=postgres_config
        )
        initialize_semantic_cache_worker(version="v01")

    @app.on_api_shutdown()
    async def app_shutdown():
        await db.clear_connection_pool()

    @app.ready.check()
    def is_ready(_) -> bool:
        return bool(connections())

    app.api.include_router(build_main_router())

    return app


app: "ace.AceService" = build_app()

if __name__ == "__main__":
    app.run()
