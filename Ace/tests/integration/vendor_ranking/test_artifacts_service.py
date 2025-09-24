import polars as pl
import pytest
import pytest_asyncio
from polars.testing import assert_frame_equal

from ace.storage import db
from vendor_ranking.context import Context
from abstract_ranking.two_tower import TTVersion


async def load_db_table(table_name: str = None, query: str = None) -> "pl.DataFrame":
    assert table_name or query
    async with db.connections().acquire() as conn:
        return await db.fetch_as_df(conn, query or f"SELECT * FROM {table_name};")


@pytest_asyncio.fixture(scope="function")
async def df_vl_feature_vendors_v3() -> "pl.DataFrame":
    query = """
        SELECT DISTINCT ON (chain_id) chain_id, chain_name
        FROM vl_feature_vendors_v3
    """
    return await load_db_table(query=query)


# noinspection PyMethodMayBeStatic
class ArtefactsServiceTest:
    @pytest.mark.asyncio
    async def test_load_chain_embeddings_from_db(
        self, app_context: "Context", df_vl_feature_vendors_v3: "pl.DataFrame"
    ):
        country = "AE"
        artifacts_service = app_context.artifacts_service_registry.get(version=TTVersion.V22)
        df_chain_embeddings: "pl.DataFrame" = await artifacts_service.load_embeddings_from_db(country)
        df_chain_embeddings_for_two_tower_v22: "pl.DataFrame" = await load_db_table(
            query=f"SELECT * FROM {artifacts_service.embeddings_table_name} WHERE country = '{country}';"
        )

        # 1: All chain names must be present
        assert all(df_chain_embeddings["chain_name"].to_list())

        # 2: Built chain embeddings table must be exactly the same as initial table with chain embeddings,
        #    that table can't be larger or smaller that the initial one.
        chain_embeddings_dropped_name = df_chain_embeddings.drop("chain_name").sort("chain_id")
        df_chain_embeddings_for_two_tower_v22 = df_chain_embeddings_for_two_tower_v22.sort("chain_id")
        assert_frame_equal(
            chain_embeddings_dropped_name,
            df_chain_embeddings_for_two_tower_v22,
            check_column_order=False,
        )
