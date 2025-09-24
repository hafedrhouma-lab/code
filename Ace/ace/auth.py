from decouple import config
from fastapi import FastAPI

from starlette_oauth2_api import AuthenticateMiddleware

STAGE = config("STAGE", default="qa")
STS_AUTH_DISABLED = config("STS_AUTH_DISABLED", cast=bool, default=False)

auth_provider_host = "https://id.talabat.com/" if STAGE == "prod" else "https://id-qa.dhhmena.com/"


def register(app: FastAPI) -> None:
    if not STS_AUTH_DISABLED:
        app.add_middleware(
            AuthenticateMiddleware,
            providers={
                "talabat": {
                    "keys": auth_provider_host + ".well-known/openid-configuration/jwks",
                    "issuer": auth_provider_host,
                    "audience": auth_provider_host + "resources",  # TODO More specific audience
                }
            },
            public_paths={"/openapi.json", "/swagger", "/livez", "/readyz"},
        )
