import os
import click
from datetime import datetime

from dotenv import load_dotenv

from projects.vendor_ranking.two_towers_v1.model_monitoring.data.datasets import Dataset
from projects.vendor_ranking.two_towers_v1.model_monitoring.utils import (
    group_data_by_table
)
from projects.vendor_ranking.common.two_towers.evaluation import (
    load_mlflow_model
)
from projects.vendor_ranking.common.two_towers.src.utils.misc_utils import (
    get_date_minus_days
)
from projects.vendor_ranking.two_towers_v1.model_monitoring.monitoring import (
    DataDriftDetector,
    target_drift_test,
    MetricProcessor,
    USER_FEATURES_TEXT,
    USER_FEATURES_NON_TEXT,
    CHAIN_FEATURES_TEXT,
    CHAIN_FEATURES_NON_TEXT
)

from projects.vendor_ranking.common.two_towers.src.utils.bq_client import (
    write_to_bq,
    delete_data_for_date
)
from projects.vendor_ranking.two_towers_v1 import PROD_MODEL_ALIAS


CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

DATA_DRIFT_TABLE_NAME = 'data_playground.two_tower_data_drifts'
TARGET_DRIFT_TABLE_NAME = 'data_playground.two_tower_target_drifts'
MODEL_METRICS_TABLE_NAME = 'data_playground.two_tower_model_metrics'


class ModelMonitoring:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self):
        # DATA DRIFT
        user_drift_detector = DataDriftDetector(
            self.dataset.user_features_reference,
            self.dataset.user_features_current
        )
        user_data_drift = user_drift_detector.detect_drift(
            text_features=USER_FEATURES_TEXT,
            non_text_features=USER_FEATURES_NON_TEXT
        )
        user_data_drift = self.update_columns(user_data_drift)

        chain_drift_detector = DataDriftDetector(
            self.dataset.chain_features_reference,
            self.dataset.chain_features_current
        )
        chain_data_drift = chain_drift_detector.detect_drift(
            text_features=CHAIN_FEATURES_TEXT,
            non_text_features=CHAIN_FEATURES_NON_TEXT
        )
        chain_data_drift = self.update_columns(chain_data_drift)

        # MODEL DRIFT
        target_drift = target_drift_test(
            self.dataset.performance_reference,
            self.dataset.performance_current
        )
        target_drift = self.update_columns(target_drift)

        # MODEL METRICS
        processor = MetricProcessor()
        model_metrics = processor.compute_metrics(
            self.dataset.performance_reference,
            self.dataset.performance_current,
            self.dataset.df_training_orders
        )
        model_metrics = self.update_columns(model_metrics)

        # WRITE TO BIG QUERY
        tables_and_data = [
            (DATA_DRIFT_TABLE_NAME, user_data_drift),
            (DATA_DRIFT_TABLE_NAME, chain_data_drift),
            (TARGET_DRIFT_TABLE_NAME, target_drift),
            (MODEL_METRICS_TABLE_NAME, model_metrics),
        ]

        grouped_tables_and_data = group_data_by_table(tables_and_data)

        # Process each table once
        for table_name, data_frames in grouped_tables_and_data.items():
            # Delete existing data for the specified date
            delete_data_for_date(
                table_name,
                datetime.strptime(self.dataset.end_current_date, '%Y-%m-%d'),
                self.dataset.country,
                date_column='test_date'
            )
            # Write all data frames to the table
            for data_frame in data_frames:
                write_to_bq(
                    df=data_frame,
                    table_name=table_name,
                    if_exists="append"
                )

    def update_columns(self, dataframe):
        dataframe['test_date'] = self.dataset.end_current_date
        dataframe['country_code'] = self.dataset.country
        dataframe['reference_start_date'] = self.dataset.start_reference_date
        dataframe['reference_end_date'] = self.dataset.end_reference_date
        dataframe['current_start_date'] = self.dataset.start_current_date
        dataframe['current_end_date'] = self.dataset.end_current_date
        return dataframe


@click.command()
@click.option(
    "--country",
    type=click.Choice(
        ['AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO', 'EG'],
        case_sensitive=False
    ),
    required=True,
    help="Country code for model comparison"
)
# @click.option(
#     "--current_date",
#     required=True,
#     help="End date for the current period (format: YYYY-MM-DD)."
# )
def main(country):
    click.echo(f"Selected country: {country}")

    load_dotenv(override=True)
    exp_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    model_name = f"{exp_name}_{country.lower()}"
    model = load_mlflow_model(
        model_name,
        PROD_MODEL_ALIAS
    )

    dataset = Dataset(
        country=country,
        mlflow_model=model,
        start_current_date=get_date_minus_days(CURRENT_DATE, 7),
        end_current_date=get_date_minus_days(CURRENT_DATE, 1)
    )

    model_monitoring = ModelMonitoring(dataset)
    model_monitoring()


if __name__ == '__main__':
    main()
