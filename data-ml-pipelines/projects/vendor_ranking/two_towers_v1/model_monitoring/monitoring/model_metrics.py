# This will compute all relevant metrics to detect concept drift for example
import pandas as pd
from evidently.pipeline.column_mapping import ColumnMapping

from evidently.report import Report
from evidently.metrics import PrecisionTopKMetric
from evidently.metrics import RecallTopKMetric
from evidently.metrics import FBetaTopKMetric
from evidently.metrics import MAPKMetric
from evidently.metrics import NDCGKMetric
from evidently.metrics import NoveltyMetric
from evidently.metrics import PopularityBias


class MetricProcessor:
    """Class to compute and process metrics for model evaluation."""

    def __init__(self):
        self.metrics = [
            PrecisionTopKMetric(k=10, no_feedback_users=True),
            RecallTopKMetric(k=10, no_feedback_users=True),
            FBetaTopKMetric(k=10, no_feedback_users=True),
            MAPKMetric(k=10, no_feedback_users=True),
            NDCGKMetric(k=10, no_feedback_users=True),
            NoveltyMetric(k=10),
            PopularityBias(k=10),
        ]

        self.custom_processors = {
            "PopularityBias": self._process_popularity_bias
        }

    def compute_metrics(self, reference_data, current_data, training_data):
        """Run the metric computation and return a consolidated DataFrame."""
        report = Report(metrics=self.metrics)
        column_mapping = ColumnMapping(
            recommendations_type='rank',
            target='target',
            prediction='prediction',
            item_id='chain_name',
            user_id='user_id',
        )
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
            additional_data={'current_train_data': training_data},
        )
        monitoring_metrics = report.as_dataframe()

        processed_metrics = [
            self.custom_processors.get(metric_name, self._default_process)(
                metric_name, df
            )
            for metric_name, df in monitoring_metrics.items()
        ]

        return pd.concat(processed_metrics, axis=0)

    @staticmethod
    def _default_process(metric_name, df):
        """Default processing for metrics."""
        return df[['metric_id', 'k', 'current_value']]

    @staticmethod
    def _process_popularity_bias(metric_name, df):
        """Custom processing for PopularityBias metric."""
        apr = df[['metric_id', 'k', 'current_apr']].rename(
            columns={'current_apr': 'current_value'}
        )
        apr['metric_id'] = 'PopularityBias_Apr'

        coverage = df[['metric_id', 'k', 'current_coverage']].rename(
            columns={'current_coverage': 'current_value'}
        )
        coverage['metric_id'] = 'PopularityBias_Coverage'

        gini = df[['metric_id', 'k', 'current_gini']].rename(
            columns={'current_gini': 'current_value'}
        )
        gini['metric_id'] = 'PopularityBias_Gini'

        return pd.concat([apr, coverage, gini], axis=0)
