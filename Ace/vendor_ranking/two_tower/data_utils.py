import numpy as np


def df_to_np_embeddings(embeddings_df):
    return np.array([np.array([x for x in y]) for y in embeddings_df.values]).squeeze()


def get_embeddings_similarity(query_embedding, candidate_embeddings):
    sim_score = query_embedding.dot(candidate_embeddings.T)
    sim_score = sim_score / np.linalg.norm(query_embedding) / np.linalg.norm(candidate_embeddings, axis=1)
    return sim_score
