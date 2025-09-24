import asyncio
import logging
import os
import socket
import typing as t
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import attr
import bentoml
import fastapi
import fastapi_rfc7807.middleware
import newrelic.agent
import pytz
import structlog
import uvicorn
from bentoml import Runner, Model
from decouple import config
from fastapi import FastAPI
from fastapi.types import DecoratedCallable
from rocketry import Rocketry

import ace
from ace.api.middlewares import add_middlewares
from ace.api.routes.route import ErrorHandlingRoute
from ace.configs.config import StageType
from ace.perf import perf_manager

if t.TYPE_CHECKING:
    from starlette.applications import Starlette
    from structlog.stdlib import BoundLogger

HealthCheckFunc = t.Union[t.Callable[[FastAPI], t.Coroutine[None, None, bool]], t.Callable[[FastAPI], bool]]
LOG: "BoundLogger" = structlog.get_logger()


# See https://github.com/bentoml/BentoML/issues/2630#issuecomment-1171761294
class HealthChecks:
    checks: dict[str, HealthCheckFunc] = {}

    def check(self, name: str = None) -> t.Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            async def async_check(app: FastAPI) -> bool:
                return func(app)

            check_name = name if name is not None else func.__name__
            self.checks[check_name] = func if asyncio.iscoroutinefunction(func) else async_check

            return func

        return decorator

    async def is_healthy(self, app: FastAPI) -> bool:
        results = await asyncio.gather(*[c(app) for c in self.checks.values()])

        return all(results)


class AceService:
    name: str
    debug: bool

    api: FastAPI
    background: Rocketry
    bentoml_service: bentoml.Service

    ready: HealthChecks = HealthChecks()

    _api_startup_hooks: list[t.Callable] = []
    _api_shutdown_hooks: list[t.Callable] = []
    _background_startup_hooks: list[t.Callable] = []

    def on_api_startup(self) -> t.Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            async def async_hook():
                return func()

            self._api_startup_hooks.append(func if asyncio.iscoroutinefunction(func) else async_hook)

            return func

        return decorator

    def on_api_shutdown(self) -> t.Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            async def async_hook():
                return func()

            self._api_shutdown_hooks.append(func if asyncio.iscoroutinefunction(func) else async_hook)
            return func

        return decorator

    def on_background_startup(
        self,
    ) -> t.Callable[[DecoratedCallable], DecoratedCallable]:
        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            async def async_hook():
                return func()

            self._background_startup_hooks.append(func if asyncio.iscoroutinefunction(func) else async_hook)

            return func

        return decorator

    class BentoMLService(bentoml.Service):
        app: "AceService" = attr.field(init=False, default=None)

        def __init__(
            self,
            app: "AceService",
            *,
            runners: t.Optional[list[Runner]] = None,
            models: t.Optional[list[Model]] = None,
        ):
            super().__init__(app.name, runners=runners, models=models)
            object.__setattr__(self, "app", app)

    async def run_api_startup_hooks(self):
        try:
            await asyncio.gather(*(async_hook() for async_hook in self._api_startup_hooks))
        except Exception as ex:
            LOG.exception(f"API startup tasks failed: {ex}")
            newrelic.agent.notice_error()
            os._exit(1)

    async def run_api_shutdown_hooks(self):
        await asyncio.gather(*(async_hook() for async_hook in self._api_shutdown_hooks))

    def __init__(self, name: str, runners: t.Optional[list[Runner]] = None):
        super().__init__()

        self.name = name
        self.debug = ace.DEBUG

        # time specified in any background tasks should be in common time zone, let's use UTC
        self.background = Rocketry(execution="async", timezone=pytz.UTC)
        self.bentoml_service = self.BentoMLService(self, runners=runners)

        @asynccontextmanager
        async def custom_lifespan_hooks(app: FastAPI):
            await self.run_api_startup_hooks()
            yield
            await self.run_api_shutdown_hooks()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            bentoml_http_app: "Starlette" = self.bentoml_service.asgi_app  # type: ignore
            async with bentoml_http_app.router.lifespan_context(app):  # local runners initialization
                async with custom_lifespan_hooks(app):
                    yield

        self._root_api = FastAPI(
            debug=self.debug,
            docs_url="/swagger",
            title="Ace",
            lifespan=lifespan,  # type: ignore
        )
        fastapi_rfc7807.middleware.register(self._root_api)

        @self._root_api.get("/readyz", summary="Kubernetes readiness probe")
        async def is_api_ready(request: fastapi.Request):
            newrelic.agent.set_transaction_name(f"/{name}/readyz")
            newrelic.agent.ignore_transaction()

            if await self.ready.is_healthy(request.app):
                return {"status": "OK"}

            raise fastapi.HTTPException(status_code=503, detail="Service is not ready")

        self.api = FastAPI(
            debug=self.debug,
            docs_url="/swagger",
            title=f"Ace ({name})",
        )
        add_middlewares(api=self.api, service_name=name)

        # log request body on any exception
        self.api.router.route_class = ErrorHandlingRoute

        # It is important to register this middleware after any other,
        # because it swallows all exceptions and just returns a "normal" response
        fastapi_rfc7807.middleware.register(self.api)

        self._root_api.mount(f"/{name}", self.api)

    def _init_server(self) -> uvicorn.Server:
        # TODO: add more detailed configs instead of having one flag "debug" which rules everywhere
        # Activate access log for debug purposes in the QA environment,
        # it helps to recognize that HTTP request reached the specific server instance.
        bento_host = config("BENTOML_HOST", default="0.0.0.0")
        bento_port = config("BENTOML_PORT", cast=int, default=3000)

        LOG.info(f"Instantiating uvicorn server: host={bento_host}, port={bento_port}")

        access_log: bool = ace.STAGE not in (StageType.PROD.value, StageType.PROD.value.upper())

        server_config = uvicorn.Config(
            self._root_api,
            host=bento_host,
            port=bento_port,
            log_config=None,  # Just use global structlog
            access_log=access_log,
        )
        server_config.setup_event_loop()
        server = uvicorn.Server(config=server_config)

        return server

    def run(self, sockets: list[socket.socket] = None):
        """
        See site-packages/bentoml_cli/worker/http_api_server.py for the BentoML part
        See ace/bentoml/http_api_server.py for our custom runner
        Also inspired by https://rocketry.readthedocs.io/en/stable/cookbook/fastapi.html
        """
        ace.log.configure(debug=self.debug)

        api_server = self._init_server()

        async def run_api_server():
            loop = asyncio.get_event_loop()
            loop.set_default_executor(ThreadPoolExecutor(initializer=ace.newrelic.init_thread))

            await api_server.serve(sockets)

        async def run_background_tasks():
            """1) Run background startup tasks
               If at least one of startup tasks failed, then
               - Report error to New Relic.
               - Shutdown service.
            2) Run background periodic tasks.
            """
            try:
                with perf_manager(
                    "Finished background startup hooks",
                    description_before="Executing background startup hooks",
                    level=logging.INFO,
                ):
                    await asyncio.gather(*[c() for c in self._background_startup_hooks])
            except Exception as ex:
                error_msg = f"Background tasks startup failed: {ex}"
                LOG.exception(error_msg)
                newrelic.agent.notice_error()
                os._exit(1)
            else:
                await self.background.serve(debug=self.debug)

        background_thread = ace.asyncio.AsyncThread(run_background_tasks())

        asyncio.run(run_api_server())

        if not background_thread.completed:
            background_thread.just_run(lambda: self.background.session.shut_down()).result()

        # Kubernetes default timeout is 30s, so wait less to make sure we don't get killed
        background_thread.wait(10)
