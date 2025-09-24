import numpy as np
from dataclasses import dataclass

from projects.vendor_ranking.common.two_towers.src.data.processors.bq import (
    SearchVocabProcessor,
    CuisinesNamesProcessor,
    ChainsIdsProcessor,
    AccountIdsProcessor,
    GeohashIdsProcessor,
    UserAreaIdsProcessor,
    VendorAreaIdsProcessor,
    ChainFeaturesProcessor,
    OrderFeaturesProcessor,
    ChainNamesProcessor
)

from projects.vendor_ranking.common.two_towers.src.data import (
    get_data_fetcher,
    query_loader
)


@dataclass
class PrepareData:
    def __init__(
            self,
            train_date_start,
            train_date_end,
            test_date_start,
            test_date_end,
            country_code,
            query_features: list,
            candidate_features: list
    ):
        # CUISINE NAMES
        cuisines_names_query = query_loader.load_query(
            'cuisines_names.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_cuisines_names_query = get_data_fetcher().fetch_data(
            description='unique cuisine name',
            source="sql",
            query=cuisines_names_query
        )
        cuisines_names_processor = CuisinesNamesProcessor(df_cuisines_names_query)
        self.unique_cuisines_names = cuisines_names_processor.process()

        # SEARCH VOCAB
        search_vocab_query = query_loader.load_query(
            'search_vocab_list.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_search_vocab = get_data_fetcher().fetch_data(
            description='search vocabulary list',
            source="sql",
            query=search_vocab_query
        )
        search_vocab_processor = SearchVocabProcessor(df_search_vocab)
        self.search_vocab_list = search_vocab_processor.process()

        # CHAINS IDS
        chains_ids_query = query_loader.load_query(
            'chains_ids.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_chains_ids = get_data_fetcher().fetch_data(
            description='unique chains ids',
            source="sql",
            query=chains_ids_query
        )
        chains_ids_processor = ChainsIdsProcessor(df_chains_ids)
        self.unique_chain_ids = chains_ids_processor.process()

        # ACCOUNT IDS
        account_ids_query = query_loader.load_query(
            'account_ids.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_account_ids = get_data_fetcher().fetch_data(
            description='unique account ids',
            source="sql",
            query=account_ids_query
        )
        account_ids_processor = AccountIdsProcessor(df_account_ids)
        self.unique_accounts_ids = account_ids_processor.process()

        # GEOHASH IDS
        geohash_ids_query = query_loader.load_query(
            'geohash6_ids.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_geohash_ids = get_data_fetcher().fetch_data(
            description='unique geohash Ids',
            source="sql",
            query=geohash_ids_query
        )
        geohash_ids_processor = GeohashIdsProcessor(df_geohash_ids)
        self.unique_geohash6_ids = geohash_ids_processor.process()

        # AREA IDS
        user_area_ids_query = query_loader.load_query(
            'user_area_ids.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_user_area_ids = get_data_fetcher().fetch_data(
            description='unique user area Ids',
            source="sql",
            query=user_area_ids_query
        )
        user_area_ids_processor = UserAreaIdsProcessor(df_user_area_ids)
        unique_user_area_ids = user_area_ids_processor.process()

        vendor_area_ids_query = query_loader.load_query(
            'vendor_area_ids.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_vendor_area_ids = get_data_fetcher().fetch_data(
            description='unique vendor area Ids',
            source="sql",
            query=vendor_area_ids_query
        )
        vendor_area_ids_processor = VendorAreaIdsProcessor(df_vendor_area_ids)
        unique_vendor_area_ids = vendor_area_ids_processor.process()

        self.unique_area_ids = np.union1d(
            unique_user_area_ids,
            unique_vendor_area_ids
        ).tolist()

        # CHAIN NAMES
        chain_names_query = query_loader.load_query(
            'chain_names.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code
        )
        df_chain_names = get_data_fetcher().fetch_data(
            description='chain names',
            source="sql",
            query=chain_names_query
        )
        chain_names_processor = ChainNamesProcessor(df_chain_names)
        self.chain_names = chain_names_processor.process()

        # CHAINS FEATURES
        chain_features_query = query_loader.load_query(
            'chain_features.sql.j2',
            start_date=train_date_start,
            end_date=train_date_end,
            country_code=country_code,
            columns = candidate_features
        )
        df_chain_features = get_data_fetcher().fetch_data(
            description='chain features',
            source="sql",
            query=chain_features_query
        )
        chain_features_processor = ChainFeaturesProcessor(
            df_chain_features,
            candidate_features
        )
        chain_features = chain_features_processor.process()
        self.chain_features_df = chain_features

        # TEST DATASET
        order_features_query = query_loader.load_query(
            'order_features.sql.j2',
            start_date=test_date_start,
            end_date=test_date_end,
            country_code=country_code,
            columns=query_features
        )
        df_order_features = get_data_fetcher().fetch_data(
            description='order features for Test Data',
            source="sql",
            query=order_features_query
        )
        order_features_processor = OrderFeaturesProcessor(
            df_order_features,
            query_features,
            test_data=True
        )
        test_df = order_features_processor.process()
        test_df = test_df[
            (test_df['account_id'].isin(self.unique_accounts_ids)) &
            (test_df['chain_id'].isin(self.unique_chain_ids))
        ]
        self.test_df = test_df.merge(chain_features, on='chain_id')

