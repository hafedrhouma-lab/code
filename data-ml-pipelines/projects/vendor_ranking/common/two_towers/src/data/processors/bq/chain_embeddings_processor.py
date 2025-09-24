import numpy as np
from ..base_processor import BaseProcessor


class ChainEmbeddingsProcessor(BaseProcessor):
    def process(self):
        chain_embeddings = self.df
        chain_embeddings = chain_embeddings.set_index("chain_id")
        chain_embeddings["chain_embeddings"] = chain_embeddings.chain_embeddings.apply(
            lambda emb: np.array([float(v) for v in emb])
        )
        return chain_embeddings
