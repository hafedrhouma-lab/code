#!/usr/bin/env python3

# See site-packages/bentoml_cli/serve.py
# A temporary solution to run API and model runners in a container, will be removed when we move model runners to a
# separate container

import os

import click
from bentoml.container import BentoMLContainer


@click.command()
@click.argument("bento", type=click.STRING)
@click.option(
    "-p",
    "--port",
    type=click.INT,
    default=BentoMLContainer.http.port.get(),
    help="The port to listen on for the REST api server",
    envvar="BENTOML_PORT",
    show_default=True,
)
@click.option(
    "--host",
    type=click.STRING,
    default=BentoMLContainer.http.host.get(),
    help="The host to bind for the REST api server",
    envvar="BENTOML_HOST",
    show_default=True,
)
@click.option(
    "--working-dir",
    type=click.Path(),
    help="When loading from source code, specify the directory to find the Service instance",
    default=None,
    show_default=True,
)
def main(bento: str, port: int, host: str, working_dir: str):
    if working_dir is None:
        if os.path.isdir(os.path.expanduser(bento)):
            working_dir = os.path.expanduser(bento)
        else:
            working_dir = "."

    import ace

    ace.log.configure()

    import bentoml.serve

    # THIS IS the main override
    bentoml.serve.SCRIPT_API_SERVER = "ace.bentoml.http_api_server"
    # BentoML exports "serve" function in the __init__.py file, which overlaps with the so-called module "serve" in the
    # same directory... That's the reason autocompletion does not work.

    from bentoml.serve import serve_http_production

    serve_http_production(
        bento,
        working_dir=working_dir,
        port=port,
        host=host,
        backlog=BentoMLContainer.api_server_config.backlog.get(),
        api_workers=1,
    )


if __name__ == "__main__":
    main()
