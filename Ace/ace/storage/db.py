import asyncio
import logging
import typing as t
from asyncio import AbstractEventLoop, gather

import asyncpg.types
import newrelic.agent
import numpy as np
import polars as pl
import pydantic
from asyncpg import Record, UndefinedTableError
from asyncpg import create_pool, Pool, Connection
from decouple import config
from pgvector.asyncpg import register_vector
from pydantic import BaseModel

import ace
from ace.configs.manager import ConfigManager

if t.TYPE_CHECKING:
    from ace.configs.config import AppPostgresConfig

logger = logging.getLogger()

STAGE = config("STAGE", default="qa")
DB_TYPES = {
    "integer": int,
    "int": int,
    "smallint": int,
    "bigint": int,
    "int2": int,
    "int4": int,
    "int8": int,
    "int2[]": pl.datatypes.List(int),
    "int4[]": pl.datatypes.List(int),
    "int8[]": pl.datatypes.List(int),
    "numeric": float,
    "text": str,
    "varchar": str,
    "double precision[]": pl.datatypes.List(float),
    "double precision": float,
    "float": float,
    "float2": float,
    "float4": float,
    "float8": float,
    # TODO Check why it is failing...
    # 'date': str,
    # 'timestamp': str,
}

# "Both asyncpg.Pool and asyncpg.Connection bind to the loop on which they were created"
# (from https://github.com/MagicStack/asyncpg/issues/293#issuecomment-391157754)
_pools: dict[AbstractEventLoop, Pool] = {}


def _convert_db_type(column_type: asyncpg.types.Type) -> t.Type:
    return DB_TYPES.get(column_type.name, pl.datatypes.Unknown)


def connections() -> t.Optional[Pool]:
    return _pools.get(asyncio.get_running_loop())


async def _init_connection(conn: Connection) -> None:
    """
    To speed up things (we do NOT need exact Decimals for the model)

    See also https://github.com/MagicStack/asyncpg/issues/282
    """
    await conn.set_type_codec("numeric", encoder=str, decoder=float, schema="pg_catalog", format="text")
    await register_vector(conn)


async def init_connection_pool(
    service_name: str = "api",
    pool_name: str = None,
    min_size=1,
    max_size=100,
    query_timeout=500,  # Milliseconds
    config: "AppPostgresConfig" = None,
) -> None:
    global _pools

    if not config:
        config = ConfigManager.load_configuration().storage.postgres

    if ace.DEBUG:
        query_timeout = None

    server_settings = {
        "application_name": f"ace-{service_name}" + (f"_{pool_name}" if pool_name else ""),
    }
    if query_timeout:
        server_settings["statement_timeout"] = str(query_timeout)

    logger.debug(f"Initializing a connection pool {pool_name} ...")
    command_timeout_seconds: float = (query_timeout and (query_timeout * 1.5) / 1000) or None
    try:
        _pools[asyncio.get_running_loop()] = await create_pool(
            # Connection settings
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database,
            server_settings=server_settings,
            # Pool settings
            init=_init_connection,
            timeout=config.connection_timeout,
            command_timeout=command_timeout_seconds,
            min_size=min_size,
            max_size=max_size,
        )
        async with connections().acquire() as conn:
            db_version = await conn.fetch("SELECT version();")
    except asyncio.exceptions.TimeoutError as ex:
        raise TimeoutError(
            f"Can not connect to database, query timeout={query_timeout}, config={config.connection_info_dict()}"
        ) from ex

    logger.info(
        f"Created SQL database connection pool {pool_name}: "
        f"host={config.host}, port={config.port}, db={config.database}. DB version: {db_version}"
    )


async def clear_connection_pool():
    global _pools
    connection_cnt = len(_pools)
    await gather(*[pool.close() for pool in _pools.values()])
    _pools.clear()
    logger.info(f"Closed #{connection_cnt} DB connections")


@newrelic.agent.function_trace()
async def fetch_as_df(conn: Connection, query: str, *args) -> pl.DataFrame:
    """
    See https://github.com/MagicStack/asyncpg/issues/17 for the discussion
    """
    stmt = await conn._prepare(query, use_cache=True)  # TODO Why use_cache is not exposed in the public interface?..
    schema = {a.name: _convert_db_type(a.type) for a in stmt.get_attributes()}
    records = await stmt.fetch(*args)

    if not records:
        # An empty data frame requires the full schema (column names + types), because there is no data from which types
        # can be inferred
        df = pl.DataFrame(schema=schema)
    else:
        from polars.utils._construction import _unpack_schema, _post_apply_columns
        from polars.polars import PyDataFrame

        # Use the full schema (column names + types), to not infer it from the data...
        column_names, schema_overrides = _unpack_schema(schema)
        schema_override = {col: schema_overrides.get(col, pl.datatypes.Unknown) for col in column_names}
        internal_df = PyDataFrame.read_rows(records, 0, schema_override or None)  # infer_schema_length
        if schema_override:
            structs = {col: tp for col, tp in schema_override.items() if isinstance(tp, pl.datatypes.Struct)}
            internal_df = _post_apply_columns(internal_df, column_names, structs, schema_overrides=schema_overrides)
        df = pl.wrap_df(internal_df)

    newrelic.agent.add_custom_span_attribute("ace.db.records", df.height)

    return df


@newrelic.agent.function_trace()
async def fetch_embedding(conn: Connection, query: str, *args) -> np.ndarray:
    """
    Fetches a single array cell (first row, first column) and returns the result as a NumPy array.
    """
    stmt = await conn._prepare(query, use_cache=True)

    # Extract the array column from the result and convert it to a NumPy array
    array_cell = await stmt.fetchval(*args)
    if not array_cell:
        return np.array([])
    embedding = np.array(array_cell)
    newrelic.agent.add_custom_span_attribute("ace.db.embedding_size", embedding.size)

    return embedding


@newrelic.agent.function_trace()
async def fetch_row_as_dict(conn: asyncpg.Connection, query: str, *args) -> dict:
    """
    Fetches a single row and returns the result as a dictionary.
    """
    try:
        stmt = await conn._prepare(
            query,
            use_cache=True,
            timeout=2,  # seconds
        )
    except UndefinedTableError as ex:
        logger.error(f"Failed to prepare statement for query {query}. Error: {ex}")
        raise

    if not (record := await stmt.fetchrow(*args)):
        return {}

    # Convert the record to a dictionary
    row_dict = {desc.name: record[idx] for idx, desc in enumerate(stmt.get_attributes())}

    return row_dict


@newrelic.agent.function_trace()
async def fetch_rows_as_dicts(conn: asyncpg.Connection, query: str, *args) -> list:
    """
    Fetches multiple rows and returns the result as a list of dictionaries.
    """
    stmt = await conn._prepare(query, use_cache=True)
    records: list["Record"] = await stmt.fetch(*args)

    if not records:
        return []

    # Convert the records to a list of dictionaries
    rows_dicts = [{desc.name: record[idx] for idx, desc in enumerate(stmt.get_attributes())} for record in records]

    return rows_dicts


PT = t.TypeVar("PT", bound=BaseModel)  # Pydantic Type


@newrelic.agent.function_trace()
async def fetch_rows_as_pydantic_type(conn: asyncpg.Connection, query: str, target_type: t.Type[PT], *args) -> list[PT]:
    """
    Fetches multiple rows and return list of rows transformed to a specific type.
    """
    stmt = await conn._prepare(query, use_cache=True)
    records: list[Record] = await stmt.fetch(*args)

    if not records:
        return []

    return pydantic.parse_obj_as(list[target_type], records)


async def assert_table_exists(table_name: str, msg: str):
    """Check that table exists and isn't empty.
    - if table is absent, then raise UndefinedTableError
    """

    query = f"""
        SELECT EXISTS (
            SELECT FROM
                pg_tables
            WHERE
                schemaname = 'public' AND
                tablename  = '{table_name}'
        );
    """

    async with connections().acquire() as conn:
        result: dict = await fetch_row_as_dict(conn=conn, query=query)
    if result["exists"] is False:
        raise UndefinedTableError(f"[{msg}] Table {table_name} is not present")

    logger.info(f"[{msg}] Table {table_name} present.")
