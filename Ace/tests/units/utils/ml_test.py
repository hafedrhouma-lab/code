import numpy as np
import polars as pl

from ace import ml


def test_with_cosine_similarity():
    df = pl.DataFrame(
        {
            "vendor_id": [1, 2, 3],
            "embeddings": [
                np.array([1, 2, 3]),
                np.array([4, 5, 6]),
                np.array([7, 8, 9]),
            ],
        }
    )
    customer = np.array([1, 2, 3])

    result = ml.with_cosine_similarity(df, customer)
    assert result["cosine_similarity"].to_list() == [
        1.0,
        0.9746318461970762,
        0.9594119455666703,
    ]
