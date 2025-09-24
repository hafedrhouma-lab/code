import pandas as pd
from tqdm import tqdm
import numpy as np


def get_simulated_candidate_probabilities(train_ds):
    batch_count = 0
    chain_dict = {}
    total = 2000
    for b in tqdm(train_ds, total=total, desc='simulating mixed dataset'):
        n, mns = b
        batch_count += 1
        unique_chains = np.unique(
            np.concatenate(
                [
                    n['chain_id'].numpy().astype(int),
                    mns['chain_id'].numpy().astype(int)
                ]
            )
        )
        for ch in unique_chains:
            chain_dict[ch] = chain_dict.get(ch, 0) + 1

        if batch_count == total:
            break

    chain_probs_df = pd.DataFrame(chain_dict.items(), columns=['chain_id', 'candidate_probability'])
    chain_probs_df['candidate_probability'] = chain_probs_df['candidate_probability'] / batch_count
    # tensorflow needs 32 not the default 64
    chain_probs_df['candidate_probability'] = chain_probs_df['candidate_probability'].astype('float32')
    chain_probs_df['chain_id'] = chain_probs_df['chain_id'].astype(str)

    return chain_probs_df
