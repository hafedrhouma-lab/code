import pandas as pd


def get_embeddings(model, user_features, chain_features):
    """Compute user and chain embeddings."""
    user_embeddings = model.two_tower_model.user_model(user_features).numpy()
    chain_embeddings = model.two_tower_model.chain_model(chain_features).numpy()
    return user_embeddings, chain_embeddings


def create_embedding_df(chain_features, chain_embeddings):
    """
    Create a DataFrame with chain embeddings and metadata.

    This function combines chain embeddings with metadata (chain_id and chain_name)
    from the chain_features DataFrame to produce a structured output.

    Args:
        chain_features (pd.DataFrame): A DataFrame containing metadata with at least
                                       'chain_id' and 'chain_name' columns.
        chain_embeddings (np.ndarray): A 2D NumPy array containing embedding vectors
                                       (one row per chain).

    Returns:
        pd.DataFrame: A DataFrame containing 'chain_id', 'chain_name', and embedding values.

    Example:
        Input:
            chain_features:
                chain_id  chain_name
                1         Chain A
                2         Chain B

            chain_embeddings:
                [[0.1, 0.2, 0.3],
                 [0.4, 0.5, 0.6]]

        Output:
            chain_id  chain_name  embedding_val1  embedding_val2  embedding_val3
            1         Chain A     0.1            0.2             0.3
            2         Chain B     0.4            0.5             0.6
    """
    embedding_columns = [f'embedding_val{i + 1}' for i in range(chain_embeddings.shape[1])]
    embedding_df = pd.DataFrame(chain_embeddings, columns=embedding_columns)
    embedding_df['chain_id'] = chain_features['chain_id'].values
    embedding_df['chain_name'] = chain_features['chain_name'].values
    return embedding_df[['chain_id', 'chain_name'] + embedding_columns]
