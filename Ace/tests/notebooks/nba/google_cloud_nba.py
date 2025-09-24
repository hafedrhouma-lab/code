import pickle
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd

from ipm.data.datasets.fetch import GetData
from ipm.utils import cmab_utils as cmab
from tutils import fs_utils as fs
from tutils.db_utils import BigQuery

path = Path("/UAE_NeuralRegressor_reward.pkl")
nn_agent = pickle.loads(path.read_bytes())


def load_config():
    gcs_path = (
        f"{BigQuery().project}-data-algorithms-content-optimization/ace/ace_artifacts/ipm/cmab/config/config.yaml"
    )
    config = fs.load_from_gcs(gcs_path=gcs_path)

    return config


def get_inference_banner_context():
    data_fetcher = GetData()
    INFERENCE_BANNER_DICT = data_fetcher.get_banner_context()
    return INFERENCE_BANNER_DICT


def load_last_trained_model(day, model_names):
    # Load YAML config, once a day from S3
    config = load_config()

    experiment_agents = {}

    for i, model in enumerate(model_names):
        for reward_col in ["reward", "iReward"]:
            experiment_agents[f"{model}_{reward_col}"] = fs.load_from_gcs(
                f"{BigQuery().project}-data-algorithms-content-optimization/ace/ace_artifacts/ipm/cmab/agents/UAE_{model}_{reward_col}/{day}/UAE_{model}_{reward_col}.pkl"
            )
            print(experiment_agents[f"{model}_{reward_col}"].last_updated)
            experiment_agents[f"{model}_{reward_col}"].set_config(config, market="UAE")

    return experiment_agents


TODAY = datetime.now().strftime("%Y%m%d")

# Yesterday date
DAY = (date.today() - timedelta(days=0)).strftime("%Y%m%d")

# Load trained model, once a day from S3
experiment_agents = load_last_trained_model(day=DAY, model_names=["CatboostRegressor", "NeuralRegressor"])

# Load the active banners contexts, once a day
BANNERS_CONTEXT = get_inference_banner_context()

#############################################################################################################################

# print(f"model lasted updated at: {experiment_agents['variant5'].last_updated}")

data_fetcher = GetData()
df_user = data_fetcher.get_user_context(account_id=26536999)
df_banner_context = data_fetcher.get_banner_context()
df_user_processed = cmab.etl_test(df_user)

for variant, agent in experiment_agents.items():
    print(f"{variant.ljust(30)}")
    banners, scores = agent.expected_rewards(user_context=df_user_processed, banners_context=df_banner_context)
    print("===========================================================")

user_context_x = pd.read_parquet("/user_context.parquet")
banner_context_x = pd.read_parquet("/Users/andreiromanov/DEV/talabat/Ace_tmp/banner_context.parquet")


for variant, agent in experiment_agents.items():
    print(f"{variant.ljust(30)}")
    banners, scores = agent.expected_rewards(user_context=user_context_x, banners_context=banner_context_x)
    print("===========================================================")

banners, scores = nn_agent.expected_rewards(user_context=df_user_processed, banners_context=df_banner_context)
print(banners, scores)
