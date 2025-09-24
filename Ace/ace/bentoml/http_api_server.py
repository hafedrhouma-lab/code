#!/usr/bin/env python3

# See site-packages/bentoml_cli/worker/http_api_server.py
# This is the main API entry point, which is (or will be) used to run the API in a container

import json
import os
import socket

import click


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,  # TODO Does not really work...
    )
)
@click.argument("bento", type=click.STRING)
@click.option(
    "--fd",
    type=click.INT,
    required=False,
    help="File descriptor of the socket to listen on",
)
@click.option(
    "--runner-map",
    type=click.STRING,
    envvar="BENTOML_RUNNER_MAP",
    help="JSON string of runners map, default sets to envars `BENTOML_RUNNER_MAP`",
)
@click.option(
    "--working-dir",
    type=click.Path(exists=True),
    help="Working directory for the API server",
)
@click.option(
    "--worker-id",
    required=False,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. "
    "Otherwise start a standalone server with a supervisor process.",
)
@click.option("--backlog", type=click.INT, default=2048, help="Backlog size for the socket")
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
@click.option(
    "--ssl-version",
    type=int,
    default=None,
    help="SSL version to use (see stdlib 'ssl' module)",
)
@click.option(
    "--ssl-ciphers",
    type=str,
    default=None,
    help="Ciphers to use (see stdlib 'ssl' module)",
)
def main(
    bento: str,
    fd: int | None,
    runner_map: str | None,
    working_dir: str | None,
    worker_id: int,
    backlog: int,
    prometheus_dir: str | None,
    ssl_version: int | None,
    ssl_ciphers: str | None,
):
    if working_dir is None:
        if os.path.isdir(os.path.expanduser(bento)):
            working_dir = os.path.expanduser(bento)
        else:
            working_dir = "."

    import newrelic.agent
    import ace

    newrelic.agent.initialize()
    ace.log.configure()

    import bentoml
    from bentoml._internal.context import component_context
    from bentoml.container import BentoMLContainer

    component_context.component_type = "api_server"
    component_context.component_index = worker_id

    if worker_id is None:
        # This server is running in standalone mode and should not be concerned with the status of its runners
        BentoMLContainer.config.runner_probe.enabled.set(False)

    # By default, all the runners will be automatically initialized via .init_local()
    if runner_map is not None:
        BentoMLContainer.remote_runner_mapping.set(json.loads(runner_map))

    svc = bentoml.load(bento, working_dir=working_dir, standalone_load=True)
    if not isinstance(svc, ace.AceService.BentoMLService):
        raise TypeError(f"Expected BentoMLService, got {type(svc).__name__} instead")
    app = svc.app

    BentoMLContainer.development_mode.set(app.debug)

    component_context.component_name = svc.name
    if svc.tag is None:
        component_context.bento_name = svc.name
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version or "not available"

    # Skip the uvicorn internal supervisor
    app.run([socket.socket(fileno=fd)] if fd is not None else None)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
