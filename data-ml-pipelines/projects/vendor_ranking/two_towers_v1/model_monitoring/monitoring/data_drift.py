import pandas as pd
from typing import List
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping


DATA_DRIFT_COLUMNS = [
    'column_name',
    'column_type',
    'stattest_name',
    'stattest_threshold',
    'drift_score',
    'drift_detected'
]


class DataDriftDetector:
    """Class for detecting data drift in textual and non-textual features."""

    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """
        Initialize the DataDriftDetector with reference and current datasets.

        :param reference_data: Reference dataset.
        :param current_data: Current dataset.
        """
        self.reference_data = reference_data
        self.current_data = current_data

    def detect_drift(self, text_features: List[str], non_text_features: List[str]) -> pd.DataFrame:
        """
        Detect drift for both text and non-text features.

        :param text_features: List of textual feature column names.
        :param non_text_features: List of non-text feature column names.
        :return: Concatenated DataFrame of drift results for text and non-text features.
        """
        drift_non_text = self._detect_non_text_drift(non_text_features)
        drift_text = self._detect_text_drift(text_features)
        data_drift = pd.concat([drift_non_text, drift_text], axis=0)

        return data_drift.drop(columns=['type', 'metric_fingerprint', 'metric_id'])

    def _detect_non_text_drift(self, non_text_features: List[str]) -> pd.DataFrame:
        """
        Detect drift for non-textual features.

        :param non_text_features: List of non-text feature column names.
        :return: DataFrame containing drift results for non-text features.
        """
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data[non_text_features],
            current_data=self.current_data[non_text_features]
        )
        drift_data = report.as_dataframe()
        drift_non_text_df = drift_data['DataDriftTable'].reset_index(drop=True)

        return drift_non_text_df

    def _detect_text_drift(self, text_features: List[str]) -> pd.DataFrame:
        """
        Detect drift for textual features.

        :param text_features: List of textual feature column names.
        :return: DataFrame containing drift results for text features.
        """
        column_mapping = ColumnMapping()
        column_mapping.text_features = text_features

        report = Report(metrics=[
            DataDriftPreset(
                num_stattest='ks',
                cat_stattest='psi',
                num_stattest_threshold=0.2,
                cat_stattest_threshold=0.2
            )
        ])
        report.run(
            reference_data=self.reference_data[text_features],
            current_data=self.current_data[text_features],
            column_mapping=column_mapping
        )

        drift_data = report.as_dataframe()
        drift_text_df = drift_data['DataDriftTable'].reset_index(drop=True)

        return drift_text_df
