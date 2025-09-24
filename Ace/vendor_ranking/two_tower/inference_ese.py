import os

import numpy as np
import pandas as pd


def get_ese_chain_embeddings(embeddings_date, country_code):
    filename = f'ese_chains_embeddings_{embeddings_date.replace("-","_")}_{country_code}.parquet'
    if os.path.exists(filename):
        return pd.read_parquet(filename)
    else:
        query = f"""
            SELECT
                ch_emb.chain_id,
                json_extract_array(ch_emb.embeddings) as chain_embeddings
                FROM `tlb-data-prod.data_platform.ese_chain_embeddings_history` ch_emb
                WHERE DATE(ch_emb.dwh_entry_timestamp) = "{embeddings_date}"
                AND chain_id in
                (
                    SELECT distinct chain_id
                    FROM `tlb-data-prod.data_platform.fct_order` f
                    join `tlb-data-prod.data_platform.dim_location` l on f.location_id = l.location_id
                    where f.country_iso = '{country_code}' and is_successful and vertical = 'food' and not is_guest
                )
        """
        chain_embeddings = read_query(query)
        chain_embeddings = chain_embeddings.set_index("chain_id")
        chain_embeddings["chain_embeddings"] = chain_embeddings.chain_embeddings.apply(
            lambda emb: np.array([float(v) for v in emb])
        )
        chain_embeddings.to_parquet(filename)
        return chain_embeddings
