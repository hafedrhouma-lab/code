import os
import structlog
import click
from mlflow import MlflowClient
from dotenv import load_dotenv
from evidently.report import Report
from evidently.metrics import RecallTopKMetric
from evidently.pipeline.column_mapping import ColumnMapping
from projects.vendor_ranking.common.two_towers.evaluation import (
    load_mlflow_model,
    get_embeddings,
    rank_and_evaluate
)

from projects.vendor_ranking.two_towers_v1 import (
    PROD_MODEL_ALIAS,
    CHALLENGER_MODEL_ALIAS
)
load_dotenv(override=True)
LOG = structlog.getLogger(__name__)


class ModelComparer:
    def __init__(self, country: str):
        self.country = country
        self.experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.model_name = f"{self.experiment_name}_{country.lower()}"
        self.prod_model_alias = PROD_MODEL_ALIAS
        self.challenger_model_alias = CHALLENGER_MODEL_ALIAS
        self.client = MlflowClient()

    def _get_model(self, alias: str):
        return load_mlflow_model(self.model_name, alias)

    @staticmethod
    def evaluate_models(prod_model, challenger_model, user_features, chain_features):
        prod_user_embeddings, prod_chain_embeddings = get_embeddings(
            prod_model,
            user_features,
            chain_features
        )
        challenger_user_embeddings, challenger_chain_embeddings = get_embeddings(
            challenger_model,
            user_features,
            chain_features
        )
        prod_performance = rank_and_evaluate(
            prod_user_embeddings,
            prod_chain_embeddings,
            user_features,
            chain_features
        )[1]
        challenger_performance = rank_and_evaluate(
            challenger_user_embeddings,
            challenger_chain_embeddings,
            user_features,
            chain_features
        )[1]

        return prod_performance, challenger_performance

    @staticmethod
    def _compute_recall(prod_performance, challenger_performance):
        report = Report(
            metrics=[RecallTopKMetric(k=10, no_feedback_users=True)]
        )
        column_mapping = ColumnMapping(
            recommendations_type='rank',
            target='target',
            prediction='prediction',
            item_id='item_id',
            user_id='user_id',
        )
        report.run(
            reference_data=prod_performance,
            current_data=challenger_performance,
            column_mapping=column_mapping
        )
        return report.as_dict()['metrics'][0]

    def compare_and_update(self):
        prod_model = self._get_model(self.prod_model_alias)
        challenger_model = self._get_model(self.challenger_model_alias)

        user_features = challenger_model.two_tower_model.params_input_data.test_df
        chain_features = challenger_model.two_tower_model.params_input_data.chain_features_df

        prod_performance, challenger_performance = self.evaluate_models(
            prod_model,
            challenger_model,
            user_features,
            chain_features
        )
        metric_reorder = self._compute_recall(prod_performance, challenger_performance)
        current_value_reorder = metric_reorder['result']['current_value']
        reference_value_reorder = metric_reorder['result']['reference_value']

        user_features['is_new_order'] = user_features.apply(
            lambda row: row['chain_id'] not in (
                    row['freq_chains'] + " " + row['user_prev_chains']
            ).split(), axis=1
        )
        new_ordering_account = user_features[user_features.is_new_order][['account_id']]

        metric_discovery = self._compute_recall(
            prod_performance[prod_performance.user_id.isin(new_ordering_account.account_id)],
            challenger_performance[challenger_performance.user_id.isin(new_ordering_account.account_id)]
        )
        current_value_discovery = metric_discovery['result']['current_value']
        reference_value_discovery = metric_discovery['result']['reference_value']

        LOG.info(f"Challenger Model Reorder Recall@10: {current_value_reorder}")
        LOG.info(f"Production Model Reorder Recall@10: {reference_value_reorder}")
        LOG.info(f"Challenger Model Discovery Recall@10: {current_value_discovery}")
        LOG.info(f"Production Model Discovery Recall@10: {reference_value_discovery}")

        if (current_value_reorder > reference_value_reorder) and (current_value_discovery > reference_value_discovery):
            challenger_model_version = self.client.get_model_version_by_alias(
                self.model_name,
                self.challenger_model_alias
            ).version
            self.client.set_registered_model_alias(
                self.model_name,
                self.prod_model_alias,
                challenger_model_version
            )
            self.client.delete_registered_model_alias(
                self.model_name,
                self.challenger_model_alias
            )

            LOG.info(
                f"Updated alias of challenger model "
                f"(version {challenger_model_version}) to '{self.prod_model_alias}'."
            )
        else:
            LOG.info(
                "Challenger model does not perform better than the prod model. "
            )


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
def main(country):
    click.echo(f"Selected country: {country}")

    model_comparer = ModelComparer(country)
    model_comparer.compare_and_update()


if __name__ == "__main__":
    main()
