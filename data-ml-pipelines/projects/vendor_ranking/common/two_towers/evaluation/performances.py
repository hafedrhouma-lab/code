import pandas as pd
import numpy as np

from projects.vendor_ranking.common.two_towers.src.utils.eval_utils import (
    get_embeddings_similarity
)
from projects.vendor_ranking.common.two_towers.evaluation.embedding_helpers import (
    create_embedding_df
)


def calculate_performance(user_features, sorted_chain_ids, top_k=31):
    """
    Calculate performance metrics for user and chain predictions.

    This function computes the performance metrics for each user by comparing
    the predicted chain rankings to the actual chain IDs. It outputs a DataFrame
    with columns `user_id`, `item_id`, `prediction`, and `target`.

    Args:
        user_features (pd.DataFrame): A DataFrame containing user features, including:
            - 'account_id': Unique identifier for the user.
            - 'chain_id': The ground truth chain ID for the user.
        sorted_chain_ids (np.ndarray): A 2D NumPy array where each row contains
                                       predicted chain IDs ranked by similarity scores.
        top_k (int): The number of top-ranked chains to consider for performance evaluation.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - 'user_id': The user ID (account_id).
            - 'item_id': The chain ID being evaluated.
            - 'prediction': The rank of the chain ID in the predictions.
            - 'target': A binary value indicating whether the chain ID matches the ground truth (1 if match, else 0).

    Example:
        Input:
            user_features:
                account_id  chain_id
                101         5
                102         7

            sorted_chain_ids:
                [[5, 8, 2],
                 [7, 3, 6]]

            top_k: 2

        Output:
            user_id  item_id  prediction  target
            101      5        1           1
            101      8        2           0
            102      7        1           1
            102      3        2           0
    """
    performance = [
        {
            'user_id': account_id,
            'item_id': item_id,
            'prediction': rank,
            'target': int(str(item_id) == str(chain_id))
        }
        for account_id, chain_id, predicted_chains in zip(
            user_features['account_id'], user_features['chain_id'], sorted_chain_ids
        )
        for rank, item_id in enumerate(predicted_chains[:top_k], start=1)
    ]
    return pd.DataFrame(performance)


def rank_and_evaluate(user_embeddings, chain_embeddings, user_features, chain_features):
    """Process embeddings and performance for a given evaluation date."""
    sim_scores = get_embeddings_similarity(
        user_embeddings,
        chain_embeddings
    )
    sorted_indices = np.argsort(sim_scores, axis=1)[:, ::-1]
    sorted_chain_ids = chain_features.chain_id.values[sorted_indices]

    embedding_df = create_embedding_df(chain_features, chain_embeddings)
    performance = calculate_performance(user_features, sorted_chain_ids)

    return embedding_df, performance
