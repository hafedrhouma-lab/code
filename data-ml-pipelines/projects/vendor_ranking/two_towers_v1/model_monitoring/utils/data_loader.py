from projects.vendor_ranking.common.two_towers.src.data import (
    query_loader,
    get_data_fetcher,
    get_feature_store
)

from projects.vendor_ranking.common.two_towers.src.cli.prepare import (
    ORDER_FEATURE_COLUMNS,
    CHAIN_FEATURE_COLUMNS
)

from projects.vendor_ranking.common.two_towers.src.data.processors.feast import (
    OrderFeaturesProcessor,
    ChainFeaturesProcessor
)

from projects.vendor_ranking.two_towers_v1.model_monitoring.utils.dates_handler import get_date_range


def get_order_features(train_date_start, train_date_end, country, query_features):
    order_features_query = query_loader.load_query(
        'order_features.sql.j2',
        start_date=train_date_start,
        end_date=train_date_end,
        country_code=country,
        orders_profile_table=get_feature_store().get_data_source("orders_profile").get_table_query_string(),
        chain_profile_table=get_feature_store().get_data_source("chain_profile").get_table_query_string()
    )

    df_order_features = get_data_fetcher().fetch_data(
        description='Order Features For train Data',
        source="feast",
        entity_sql=order_features_query,
        features=ORDER_FEATURE_COLUMNS
    )

    order_features_processor = OrderFeaturesProcessor(
        df_order_features,
        query_features
    )
    order_features = order_features_processor.process()
    return order_features


def get_chain_features(train_date_start, train_date_end, country, candidate_features):
    chain_features_query = query_loader.load_query(
        'chain_features.sql.j2',
        start_date=train_date_start,
        end_date=train_date_end,
        country_code=country,
        orders_profile_table=get_feature_store().get_data_source("orders_profile").get_table_query_string(),
        chain_profile_table=get_feature_store().get_data_source("chain_profile").get_table_query_string()
    )

    df_chain_features = get_data_fetcher().fetch_data(
        description='chain features',
        source="feast",
        entity_sql=chain_features_query,
        features=CHAIN_FEATURE_COLUMNS
    )

    chain_features_processor = ChainFeaturesProcessor(
        df_chain_features,
        candidate_features
    )
    chain_features = chain_features_processor.process()
    return chain_features


def get_towers_features(
        start_date,
        end_date,
        country,
        query_features,
        candidate_features,
        training_data = False,
        days_back=31
):
    """Process embeddings and performance for a given evaluation date."""

    date_minus_thirty, date_minus_one = get_date_range(
        end_date if training_data else start_date,
        days_back
    )
    user_features = get_order_features(
        start_date,
        end_date,
        country,
        query_features
    )
    chain_features = get_chain_features(
        date_minus_thirty,
        date_minus_one,
        country,
        candidate_features
    )

    return user_features, chain_features
