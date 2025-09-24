from dataclasses import dataclass

from projects.vendor_ranking.common.two_towers.src.data import (
    query_loader, get_data_fetcher
)
from projects.vendor_ranking.common.two_towers.src.data.processors.bq import (
    GeohashChainsProcessor,
    ChainEmbeddingsProcessor
)
from projects.vendor_ranking.common.two_towers.src.utils.eval_utils import get_similar_geohashes


@dataclass
class PrepareUtils:
    def __init__(
            self,
            train_date_start,
            train_date_end,
            country_code,
            embedding_date
    ):
        # GEOHASH CHAINS
        geohash_chains_query = query_loader.load_query(
            'geohash_chains.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_geohash_chains = get_data_fetcher().fetch_data(
            description='geohash chains',
            source="sql",
            query=geohash_chains_query
        )
        geohash_chains_processor = GeohashChainsProcessor(df_geohash_chains)
        geohash_chains_df = geohash_chains_processor.process()
        self.geo_to_chains, self.geo_to_parent_geo = get_similar_geohashes(geohash_chains_df)

        # ESE CHAIN EMBEDDINGS
        embeddings_query = query_loader.load_query(
            'ese_chain_embeddings.sql.j2',
            embedding_date=embedding_date,
            country_code=country_code
        )
        embeddings_df = get_data_fetcher().fetch_data(
            description='ese chain embeddings',
            source="sql",
            query=embeddings_query,
        )
        chain_embeddings_processor = ChainEmbeddingsProcessor(embeddings_df)
        self.ese_chain_embeddings = chain_embeddings_processor.process()
