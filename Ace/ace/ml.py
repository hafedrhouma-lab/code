import newrelic.agent
import numpy as np
import polars as pl


# TODO Compile with Numba
@newrelic.agent.function_trace()
def with_cosine_similarity(
    df: pl.DataFrame, customer: np.ndarray, embeddings_column_name: str = "embeddings"
) -> pl.DataFrame:
    # TODO Try Polars once again, also upgrade to Pandas 2
    embeddings = np.array(df[embeddings_column_name].to_pandas().tolist())
    cosine_similarities = _cosine_similarity(embeddings, customer)

    return df.with_columns(pl.Series("cosine_similarity", cosine_similarities)).fill_nan(None)


@newrelic.agent.function_trace()
def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_norm = np.linalg.norm(vector)
    # TODO Compare shapes, raise an exception if they don't match
    # See https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul
    dot_products = matrix @ vector
    magnitudes_matrix = np.linalg.norm(matrix, axis=1)

    # Check for zero magnitudes
    zero_magnitudes_mask = magnitudes_matrix == 0

    # Replace zero magnitudes with 1 to avoid division by zero
    magnitudes_matrix[zero_magnitudes_mask] = 1

    cosine_similarities = dot_products / (magnitudes_matrix * vector_norm)

    # Set cosine similarities with zero magnitudes to NaN
    cosine_similarities[zero_magnitudes_mask] = np.nan

    return cosine_similarities
