import numpy as np
import polars as pl
import pytest

from ace.storage import db


@pytest.mark.asyncio
async def test_fetch_as_df():
    STATIC_QUERY = "select * from f_get_vendor_ranking_features($1, $2)"

    await db.init_connection_pool()
    async with db.connections().acquire() as conn:
        response = await db.fetch_as_df(conn, STATIC_QUERY, 4, [15011, 34657, 38054])
        no_of_rows = response.shape[0]
        assert isinstance(response, pl.DataFrame)
        assert no_of_rows == 3


@pytest.mark.asyncio
async def test_fetch_embedding():
    CUSTOMER_EMBEDDINGS_QUERY = "SELECT embeddings FROM ese_account_embeddings WHERE account_id = $1"
    customer_id = 19418951

    expected_output = np.array(
        [
            -0.00415295,
            1.49018597,
            0.95861303,
            -0.03416792,
            0.7633566,
            -0.92173871,
            1.83671926,
            0.33872641,
            1.4597392,
            -0.20598308,
            0.52319734,
            0.25793209,
            -1.65270206,
            -0.46745176,
            0.74581267,
            1.24393753,
            1.70152333,
            -0.06389324,
            0.36804105,
            0.61147207,
            2.44693331,
            1.03659014,
            0.0377212,
            0.38320204,
            -1.7660643,
            0.08267613,
            -1.05609986,
            -0.73362384,
            -0.94307286,
            -0.33281332,
            -0.07153799,
            -0.22507049,
            0.85242084,
            0.91231287,
            0.10975831,
            0.73859076,
            1.8160383,
            1.10756127,
            0.67258245,
            1.29837311,
            3.04796737,
            -1.49831432,
            -0.49573151,
            0.05392122,
            -0.84136553,
            -1.83869323,
            -0.23493886,
            -1.69774388,
            0.829476,
            2.06795185,
        ]
    )

    await db.init_connection_pool()
    async with db.connections().acquire() as conn:
        response = await db.fetch_embedding(conn, CUSTOMER_EMBEDDINGS_QUERY, customer_id)
        assert isinstance(response, np.ndarray)
        assert len(response) == len(expected_output)
        assert response.all() == expected_output.all()


@pytest.mark.asyncio
async def test_fetch_embedding_empty_response():
    CUSTOMER_EMBEDDINGS_QUERY = "SELECT embeddings FROM ese_account_embeddings WHERE account_id = $1"
    customer_id = -1

    expected_output = np.array([])

    await db.init_connection_pool()
    async with db.connections().acquire() as conn:
        response = await db.fetch_embedding(conn, CUSTOMER_EMBEDDINGS_QUERY, customer_id)
        assert isinstance(response, np.ndarray)
        assert len(response) == len(expected_output)
        assert response.all() == expected_output.all()


@pytest.mark.asyncio
async def test_fetch_row_as_dict():
    await db.init_connection_pool()
    async with db.connections().acquire() as conn:
        row_dict = await db.fetch_row_as_dict(conn, "SELECT 1 AS a, 2 AS b")
        assert row_dict == {"a": 1, "b": 2}
