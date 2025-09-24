from pathlib import Path

USER_WEIGHTS_PATH = "user_weights"
CHAIN_WEIGHTS_PATH = "chain_weights"
PARAMS_INPUT_DATA_PATH = "params_input_data.pkl"
DATA_TRAINING_UTILS = "data_training_utils.pkl"
ESE_CHAIN_EMBEDDINGS_PATH = "ese_vector_embedding_layer.parquet"

BASE_PATH = Path(__file__).parent
MODEL_CONFIG_PATH: str = str((BASE_PATH / "model_config.yaml").absolute())

CONDA_ENV_PATH = BASE_PATH / "conda.yaml"

FEATURE_DESCRIPTION_FILE = 'feature_description.pkl'
TFRECORDS_FILE_NAME = "flattened_train_ds_bq.tfrecord"

BUCKET_NAME = "tlb-data-dev-data-algorithms-content-optimization"
DESTINATION_BLOB_NAME = "two_tower_dev_scaling/model_training"

PROD_MODEL_ALIAS = 'ace_champion_model_staging'
CHALLENGER_MODEL_ALIAS = 'ace_challenger_model'
