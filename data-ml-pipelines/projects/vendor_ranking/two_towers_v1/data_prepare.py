import argparse
import structlog
import cloudpickle
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from projects.vendor_ranking.common.two_towers.src.cli.prepare.prepare_tf_records import PrepareTFRecords
from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_param import PrepareData
from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_utils import PrepareUtils
from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_train_df import TrainingDatasetGenerator

from projects.vendor_ranking.common.two_towers.src.utils.load_utils import (
    load_model_config,
    save_model_config
)
from projects.vendor_ranking.common.two_towers.src.utils.misc_utils import (
    get_date_range,
    get_date_minus_days
)
from projects.vendor_ranking.common.two_towers.src.utils.bq_client import upload_to_gcs

from projects.vendor_ranking.two_towers_v1 import (
    PARAMS_INPUT_DATA_PATH,
    DATA_TRAINING_UTILS,
    BUCKET_NAME,
    TFRECORDS_FILE_NAME,
    FEATURE_DESCRIPTION_FILE,
    DESTINATION_BLOB_NAME
)

load_dotenv(override=True)
LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")


class DataPrepare:
    def __init__(self, country):
        self.country = country
        self.config_file_name = f"model_config_{self.country}.yaml"
        self.PARAMS_PATH = Path(__file__).parent.resolve() / "model_configs" / self.config_file_name

    def prepare(self):
        params = load_model_config(self.PARAMS_PATH)

        days_back = params.get('training_data_days')
        train_date_start, train_date_end = get_date_range(
            get_date_minus_days(CURRENT_DATE, 9),
            days_back
        )
        test_date_start, test_date_end = \
            get_date_minus_days(CURRENT_DATE, 8), get_date_minus_days(CURRENT_DATE, 1)

        params.update({
            'train_date_start': train_date_start,
            'train_date_end': train_date_end,
            'test_date_start': test_date_start,
            'test_date_end': test_date_end
        })

        LOG.info(
            f"Training will occur from "
            f"{params.get('train_date_start')} to {params.get('train_date_end')}, "
        )
        LOG.info(
            f"Testing will occur from "
            f"{params.get('test_date_start')} to {params.get('test_date_end')}"
        )

        params_input_data = PrepareData(
            train_date_start=params.get('train_date_start'),
            train_date_end=params.get('train_date_end'),
            test_date_start=params.get('test_date_start'),
            test_date_end=params.get('test_date_end'),
            country_code=params.get('country'),
            query_features=params.get('query_features'),
            candidate_features=params.get('candidate_features')
        )
        data_training_utils = PrepareUtils(
            params.get('train_date_start'),
            params.get('train_date_end'),
            params.get('country'),
            params.get('EMBEDDING_DATE')
        )

        chains_count = params_input_data.chain_features_df.shape[0]
        params.update(
            {'ns_ratio': chains_count / params['train_batch']}
        )
        save_model_config(self.config_file_name, params)

        train_data_generator = TrainingDatasetGenerator(
            train_date_start=params.get('train_date_start'),
            train_date_end=params.get('train_date_end'),
            params=params
        )
        train_ds = train_data_generator.generate()

        LOG.info(
            "Training data row counts",
            positive_data_lines=train_data_generator.positive_data.shape[0],
            negative_data_lines=train_data_generator.negative_data.shape[0]
        )

        tfrecords_preparer = PrepareTFRecords(
            train_ds,
            train_data_generator.positive_data.shape[0],
            train_data_generator.negative_data.shape[0],
            params,
            FEATURE_DESCRIPTION_FILE,
            TFRECORDS_FILE_NAME
        )
        tfrecords_preparer.generate()

        with open(PARAMS_INPUT_DATA_PATH, "wb") as f:
            cloudpickle.dump(params_input_data, f)
        with open(DATA_TRAINING_UTILS, "wb") as f:
            cloudpickle.dump(data_training_utils, f)

        destination_blob = f"{DESTINATION_BLOB_NAME}/{params['country']}/"

        upload_to_gcs(BUCKET_NAME, PARAMS_INPUT_DATA_PATH, destination_blob)
        upload_to_gcs(BUCKET_NAME, DATA_TRAINING_UTILS, destination_blob)
        upload_to_gcs(BUCKET_NAME, FEATURE_DESCRIPTION_FILE, destination_blob)
        upload_to_gcs(BUCKET_NAME, TFRECORDS_FILE_NAME, destination_blob)
        upload_to_gcs(BUCKET_NAME, self.config_file_name, destination_blob)

    def __call__(self):
        self.prepare()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two tower model.")
    parser.add_argument(
        "--country",
        required=True,
        choices=['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    args = parser.parse_args()

    preparer = DataPrepare(args.country)
    preparer()
